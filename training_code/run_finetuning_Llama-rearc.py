import os
import torch
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments
from datasets import Dataset

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import load_unsloth_4bit, keep_single_char_tokens, save_model_and_tokenizer
from model_tools import load_peft_state, merge_peft_into_base
from arc_downloader import download_arc_data

# Detect number of GPUs available
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs detected: {num_gpus}")

# Get current device based on local_rank
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

# input paths
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'  # auto-downloaded from huggingface.co
re_arc_path = os.path.join('input', 're_arc')  # https://github.com/michaelhodel/re-arc
download_arc_data(re_arc_path)

# output paths
save_model_path = os.path.join('pretrained_models', "Llama-3.2-3B-ReArc")

# Modify the model loading function in model_tools.py
# Modified version that takes a specific device as input
def device_aware_load_unsloth_4bit(model_name, current_device=None):
    """
    Load a model in 4-bit precision with device awareness for distributed training
    """
    from transformers import AutoTokenizer
    
    # If we're in distributed mode, make sure we're loading the model on the right device
    if current_device is not None:
        # Use FastLanguageModel with explicit device mapping
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
            device_map={"": current_device}  # Assign to specific device
        )
    else:
        # Original behavior for single GPU
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
        )
        
    return model, tokenizer

for action in ['train', 'merge']:
    # continue if task already accomplished
    if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
        continue
    if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
        continue

    # load base model & reduce embedding size with correct device
    model = tokenizer = None  # free memory
    
    # Get current device index
    current_device = local_rank
    
    # Load model with device-aware function
    model, tokenizer = device_aware_load_unsloth_4bit(base_model, current_device)
    
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
        train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=12, sizes=[6], seed=42)

        # augment data set and transform to list
        train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset_augment = train_dataset.augment(**train_aug_opts)
        train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

        # run training
        FastLanguageModel.for_training(model)
        tokenizer.padding_side = 'right'
        
        # Configure single GPU mode instead of distributed mode
        # If multi-GPU is detected, we're deliberately ignoring it
        # and letting each rank train its own model
        training_args = TrainingArguments(
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
            output_dir=f'tmp_output_rank{local_rank}',
            save_strategy='no',
            report_to='none',
            # Process-specific local rank
            local_rank=local_rank,
        )
        
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
            args=training_args,
        )
        
        # Train the model
        trainer.train()
        
        # Each process saves its own model
        save_model_and_tokenizer(f'{save_model_path}-lora-rank{local_rank}', model, tokenizer)

    if action == 'merge':
        # Load peft weights and merge
        # Use rank-specific model
        load_peft_state(model, f'{save_model_path}-lora-rank{local_rank}')
        model = merge_peft_into_base(model)
        
        # Each process saves its own merged model
        save_model_and_tokenizer(f'{save_model_path}-merged-rank{local_rank}', model, tokenizer)

print(f"Training and merging complete for rank {local_rank}!")