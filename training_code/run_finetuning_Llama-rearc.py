import os
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments
from datasets import Dataset

# Add these imports for Accelerate + DeepSpeed
import json
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import load_unsloth_4bit, keep_single_char_tokens, save_model_and_tokenizer
from model_tools import load_peft_state, merge_peft_into_base
from arc_downloader import download_arc_data

# input paths
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'  # auto-downloaded from huggingface.co
re_arc_path = os.path.join('input', 're_arc')  # https://github.com/michaelhodel/re-arc
download_arc_data(re_arc_path)  # Fixed variable name

# output paths
save_model_path = os.path.join('pretrained_models', "Llama-3.2-3B-ReArc")

# Create DeepSpeed configuration file
ds_config = {
    "train_batch_size": 4,  # This will be per GPU
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "weight_decay": 0.0
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": "auto"
        }
    },
    "fp16": {
        "enabled": not is_bfloat16_supported()
    },
    "bf16": {
        "enabled": is_bfloat16_supported()
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "contiguous_gradients": True,
        "overlap_comm": True
    }
}

# Save the config to a file
os.makedirs('configs', exist_ok=True)
with open('configs/ds_config.json', 'w') as f:
    json.dump(ds_config, f, indent=4)

for action in ['train', 'merge']:
    # continue if task already accomplished
    if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
        continue
    if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
        continue

    # load base model & reduce embedding size
    model = tokenizer = None  # free memory
    model, tokenizer = load_unsloth_4bit(base_model)
    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

    # set formatting options
    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I',
        reply_beg='\n+/-=O',
        reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n',
        max_tokens=128000,
    )

    # create lora model
    lora_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']
    model = FastLanguageModel.get_peft_model(
        model=model,
        target_modules=lora_layers,
        r=256,
        lora_alpha=24,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    if action == 'train':
        # load training data
        train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=368, sizes=[6], seed=42)

        # augment data set and transform to list (eventually removing examples to stay below the max. token count)
        train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset_augment = train_dataset.augment(**train_aug_opts)
        train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

        # run training
        FastLanguageModel.for_training(model)
        tokenizer.padding_side = 'right'
        
        # Initialize the accelerator with DeepSpeed
        accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(config="configs/ds_config.json"))
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=Dataset.from_list(train_dataset_as_list),
            dataset_text_field="text",
            max_seq_length=fmt_opts['max_tokens'],
            packing=False,
            data_collator=InputMaskingDataCollator(
                instruction_template=fmt_opts['query_beg'],
                response_template=fmt_opts['reply_beg'],
                mlm=False,
                tokenizer=tokenizer,
                mask_first_n_examples=1,
            ),
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                warmup_ratio=0.25,
                num_train_epochs=1,
                learning_rate=1e-4,
                embedding_learning_rate=1e-5,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.00,
                lr_scheduler_type='cosine',
                seed=42,
                output_dir='tmp_output',
                save_strategy='no',
                report_to='none',
                # Add DeepSpeed configuration
                deepspeed="configs/ds_config.json",
                # Add distributed training settings
                local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            ),
        )
        
        # Use the accelerator to prepare your trainer components
        trainer.model, trainer.optimizer = accelerator.prepare(
            trainer.model, trainer.create_optimizer()
        )
        
        trainer_stats = unsloth_train(trainer)
        
        # Save model (ensure this works with distributed training)
        if accelerator.is_main_process:
            save_model_and_tokenizer(f'{save_model_path}-lora', accelerator.unwrap_model(model), tokenizer)

    if action == 'merge':
        # Initialize accelerator for the merge process too
        accelerator = Accelerator()
        
        # Wrap the model
        model = accelerator.prepare(model)
        
        # load peft weights and merge
        load_peft_state(accelerator.unwrap_model(model), f'{save_model_path}-lora')
        model = merge_peft_into_base(accelerator.unwrap_model(model))
        
        # Save only from the main process
        if accelerator.is_main_process:
            save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)