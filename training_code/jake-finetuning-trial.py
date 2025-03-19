#!/usr/bin/env python

# Copyright 2024 Daniel Franzen and Jan Disselhoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# (License text omitted for brevity)
#
# NOTE: This script is adapted to use standard PyTorch DDP within the
#       Hugging Face Trainer, instead of DeepSpeed.

import os
import torch

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
)
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model, 
    PeftModel
)
from datasets import Dataset

# Custom ARC loader (assuming arc_loader.py is in the same directory)
from arc_loader import ArcDataset

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9} GB")

# Paths
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'
re_arc_path = os.path.join('input', 're_arc')  # dataset folder
save_model_path = os.path.join('pretrained_models', "Llama-3.2-3B-ReArc")

def load_model_4bit(model_name_or_path):
    """
    Load a model in 4-bit precision using bitsandbytes.
    """
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    # Ensure a pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    return model, tokenizer

def keep_single_char_tokens(model, tokenizer, keep=None, remove_unk=False):
    """
    Keep only the specified tokens in the embedding matrix 
    (useful if you want to reduce vocabulary).
    """
    if keep is None:
        return
    
    vocab = tokenizer.get_vocab()
    orig_embeds = model.get_input_embeddings().weight.data
    
    # Identify indices of tokens to keep
    keep_indices = []
    for token in keep:
        # Handle single chars
        if len(token) == 1:
            for i, t in enumerate(vocab.keys()):
                if token in t and len(t) == 1 and i < orig_embeds.shape[0]:
                    keep_indices.append(i)
        else:
            # Multi-char token
            idx = tokenizer.convert_tokens_to_ids(token)
            if (idx != tokenizer.unk_token_id or not remove_unk) and idx < orig_embeds.shape[0]:
                keep_indices.append(idx)
    
    keep_indices = list(set(keep_indices))
    keep_indices = [idx for idx in keep_indices if idx < orig_embeds.shape[0]]
    
    if len(keep_indices) == orig_embeds.shape[0]:
        print("Keeping all embedding tokens, no reduction needed.")
        return keep_indices
    
    print(f"Reducing embedding matrix from {orig_embeds.shape[0]} to {len(keep_indices)} tokens.")
    
    # Map old->new indices
    new_embeds_dict = {old_idx: i for i, old_idx in enumerate(keep_indices)}
    
    # Resize token embeddings
    model.resize_token_embeddings(len(keep_indices))
    
    # Copy original embeddings
    for old_idx, new_idx in new_embeds_dict.items():
        model.get_input_embeddings().weight.data[new_idx] = orig_embeds[old_idx]
        
    print(f"Successfully reduced embedding matrix to {len(keep_indices)} tokens.")
    return keep_indices

def load_tokenized_dataset(dataset_list, tokenizer, max_length=2048):
    """
    Converts a list of {'text': ...} items to a Hugging Face Dataset and tokenizes it.
    """
    simple_dataset = []
    for item in dataset_list:
        if isinstance(item, dict) and 'text' in item:
            simple_dataset.append({'raw_text': item['text']})
    
    dataset = Dataset.from_list(simple_dataset)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['raw_text'],
            padding=False, 
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['raw_text'],
        desc="Tokenizing texts"
    )
    return tokenized_dataset

class InputMaskingDataCollator:
    """
    Data collator that optionally masks instruction tokens so the model
    primarily learns from the "response" portion in a causal LM setup.
    """
    def __init__(self, tokenizer, instruction_template="I", response_template="\n+/-=O", mlm=False, mask_first_n_examples=1):
        self.tokenizer = tokenizer
        self.instruction_template = instruction_template
        self.response_template = response_template
        self.mlm = mlm
        self.mask_first_n_examples = mask_first_n_examples
    
    def __call__(self, features):
        if not isinstance(features, list) or len(features) == 0 or 'input_ids' not in features[0]:
            raise ValueError("Features must be a list of dicts with 'input_ids'")
            
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        
        if not self.mlm:
            # Causal LM
            batch["labels"] = batch["input_ids"].clone()
            
            # Optionally mask out instruction portion in the first few examples
            for i in range(len(features)):
                if i >= len(batch["input_ids"]):
                    continue
                
                text = self.tokenizer.decode(batch["input_ids"][i])
                instr_pos = text.find(self.instruction_template)
                resp_pos = text.find(self.response_template)
                
                # If found both templates, mask the instruction part
                if instr_pos != -1 and resp_pos != -1 and i < self.mask_first_n_examples:
                    instr_token_pos = len(self.tokenizer.encode(text[:instr_pos])) - 1
                    resp_token_pos  = len(self.tokenizer.encode(text[:resp_pos])) - 1
                    if 0 <= instr_token_pos < resp_token_pos < batch["labels"][i].shape[0]:
                        batch["labels"][i, instr_token_pos:resp_token_pos] = -100
        
        return batch

def setup_peft_model(model, r=256, lora_alpha=24, target_modules=None):
    """
    Prepare a model for LoRA (PEFT) training in k-bit precision.
    """
    if target_modules is None:
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 
            'gate_proj', 'up_proj', 'down_proj',
            'embed_tokens', 'lm_head'
        ]
    
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    return model

def save_model_and_tokenizer(save_path, model, tokenizer):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")

def load_peft_state(model, peft_model_path):
    """
    Load LoRA adapter weights from a directory into a base model.
    """
    if not isinstance(model, PeftModel):
        model = PeftModel.from_pretrained(model, peft_model_path)
    else:
        model.load_adapter(peft_model_path)
    return model

def merge_peft_into_base(model):
    """
    Merge LoRA adapter weights into the base model (turning it into a normal model).
    """
    model = model.merge_and_unload()
    return model

####################
# Main Execution
####################

if __name__ == "__main__":
    for action in ['train', 'merge']:
        # If the LoRA model directory or merged model directory already exists, skip
        if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
            print("LoRA directory already exists, skipping training.")
            continue
        if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
            print("Merged model directory already exists, skipping merge.")
            continue
        
        # Free memory if needed
        model = None
        tokenizer = None
        
        # Load base model & reduce embedding size
        model, tokenizer = load_model_4bit(base_model)
        keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=') + tokenizer.tokenize('\n')
        keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)
        
        # Some custom formatting
        fmt_opts = dict(
            preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
            query_beg='I',
            reply_beg='\n+/-=O',
            reply_end='\n' + tokenizer.eos_token,
            lines_sep='\n',
            max_tokens=128000,
        )
        
        # Create LoRA model
        lora_layers = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 
            'gate_proj', 'up_proj', 'down_proj',
            'embed_tokens', 'lm_head'
        ]
        model = setup_peft_model(
            model, r=256, lora_alpha=24, target_modules=lora_layers
        )
        
        if action == 'train':
            # --- Load and prepare the dataset ---
            train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=4, sizes=[6], seed=42)
            train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
            train_dataset_augment = train_dataset.augment(**train_aug_opts)
            train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)
            
            train_dataset_tokenized = load_tokenized_dataset(
                train_dataset_as_list, 
                tokenizer,
                max_length=fmt_opts['max_tokens']
            )
            
            # Data collator
            tokenizer.padding_side = 'right'
            data_collator = InputMaskingDataCollator(
                instruction_template=fmt_opts['query_beg'],
                response_template=fmt_opts['reply_beg'],
                mlm=False,
                tokenizer=tokenizer,
                mask_first_n_examples=1,
            )
            
            # --- Training arguments (no DeepSpeed) ---
            training_args = TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                warmup_ratio=0.25,
                num_train_epochs=1,
                learning_rate=1e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.00,
                lr_scheduler_type='cosine',
                seed=42,
                output_dir='tmp_output',
                save_strategy='no',
                report_to='none',
                remove_unused_columns=False,  # Keep all columns in dataset
                # DDP is automatically enabled by Hugging Face if more than one GPU is available.
            )
            
            # --- Trainer ---
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset_tokenized,
                data_collator=data_collator,
                args=training_args,
            )
            
            # Train the model
            trainer.train()
            
            # Save LoRA adapter + tokenizer
            save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)
        
        if action == 'merge':
            # Load PEFT (LoRA) weights and merge into base
            model = load_peft_state(model, f'{save_model_path}-lora')
            model = merge_peft_into_base(model)
            save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)
    
    print("All done!")
