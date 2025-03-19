import os
import json
import torch
import transformers
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset

from arc_loader import ArcDataset
from model_tools import keep_single_char_tokens
from model_tools import save_model_and_tokenizer
from model_tools import load_peft_state, merge_peft_into_base
from arc_downloader import download_arc_data

# Detect number of GPUs available
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs detected: {num_gpus}")

# Get current device based on local_rank
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
print(f"Process {local_rank} using device: {device}")

# input paths
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'  # auto-downloaded from huggingface.co
re_arc_path = os.path.join('input', 're_arc')  # https://github.com/michaelhodel/re-arc
download_arc_data(re_arc_path)

# output paths
save_model_path = os.path.join('pretrained_models', "Llama-3.2-3B-ReArc")

# Create DeepSpeed configuration file
ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
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
        "enabled": True
    },
    "bf16": {
        "enabled": False
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

    # Load model and tokenizer using standard HF approach
    print(f"Loading model from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        trust_remote_code=True,
        # Don't use device_map='auto' for distributed training
        device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    
    # Make sure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Keep only certain tokens to reduce embedding size
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

    # Create LoRA configuration
    lora_config = LoraConfig(
        r=256,
        lora_alpha=24,
        lora_dropout=0,
        bias="none",
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 
            'gate_proj', 'up_proj', 'down_proj', 
            'embed_tokens', 'lm_head'
        ],
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    if action == 'train':
        # load training data
        train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=12, sizes=[6], seed=42)

        # augment data set and transform to list
        train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset_augment = train_dataset.augment(**train_aug_opts)
        train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)
        
        # Function to process dataset to match HF format
        def process_dataset_for_hf(dataset_list):
            processed_data = []
            
            for item in dataset_list:
                # Tokenize the text
                tokenized = tokenizer(item["text"], truncation=False, padding=False)
                
                # Format for HF trainer
                processed_data.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": tokenized["input_ids"].copy()  # For causal LM, labels = input_ids
                })
            
            return processed_data
            
        # Process dataset to match HF format
        processed_data = process_dataset_for_hf(train_dataset_as_list)
        hf_dataset = Dataset.from_list(processed_data)
        
        # Print the first example to verify format
        print("Example from processed dataset:")
        example = hf_dataset[0]
        print(f"Keys: {list(example.keys())}")
        print(f"Input IDs length: {len(example['input_ids'])}")
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        # Create training arguments with DeepSpeed enabled
        training_args = TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_ratio=0.25,
            num_train_epochs=1,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            weight_decay=0.00,
            lr_scheduler_type='cosine',
            seed=42,
            output_dir='tmp_output',
            save_strategy='no',
            report_to='none',
            # DeepSpeed config
            deepspeed="configs/ds_config.json",
            # Required for distributed training
            local_rank=local_rank,
        )
        
        # Create data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create the trainer with standard HF components
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save only from the main process (rank 0)
        if local_rank == 0:
            model.save_pretrained(f'{save_model_path}-lora')
            tokenizer.save_pretrained(f'{save_model_path}-lora')

    if action == 'merge':
        # Load peft weights and merge
        model = model.merge_and_unload()
        
        # Save only from the main process
        if local_rank == 0:
            model.save_pretrained(f'{save_model_path}-merged')
            tokenizer.save_pretrained(f'{save_model_path}-merged')

print("Training and merging complete!")