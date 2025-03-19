# Copyright 2024 Daniel Franzen and Jan Disselhoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import keep_single_char_tokens, save_model_and_tokenizer

# input paths
base_model = 'nvidia/Mistral-NeMo-Minitron-8B-Base'  # auto-downloaded from huggingface.co
arc_data_path = os.path.join('input/arc-data/ARC-Data/input', 'arc-prize-2024')  # as on kaggle arc prize 2024
re_arc_path = os.path.join('input/arc-data/ARC-Data/input', 're_arc')  # https://github.com/michaelhodel/re-arc
neoneye_path = os.path.join('input/arc-data/ARC-Data/input', 'arc-dataset-collection')  # https://github.com/neoneye/arc-dataset-collection

# output paths
save_model_path = os.path.join('pretrained_models', "Mistral-NeMo-Minitron-Full")

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer for 4-bit quantization"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Make sure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization with BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load the model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    
    # Prepare model for training with 4-bit quantization
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def merge_lora_weights(base_model_path, adapter_path, output_path):
    """Merge LoRA weights into the base model and save"""
    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    
    # Load and merge LoRA weights
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    # Save the merged model
    save_model_and_tokenizer(output_path, model, tokenizer)
    
    return model, tokenizer

for action in ['train', 'merge']:
    # continue if task already accomplished
    if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
        continue
    if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
        continue

    # load base model & reduce embedding size
    model = tokenizer = None  # free memory
    model, tokenizer = load_model_and_tokenizer(base_model)
    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=')+tokenizer.tokenize('\n')
    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

    # set formatting options
    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I',
        reply_beg='\n+/-=O',
        reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n',
        max_tokens=8192,
    )

    if action == 'train':
        # Configure LoRA
        lora_config = LoraConfig(
            r=256,
            lora_alpha=24,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        
        # load training data
        arc_eval_set = ArcDataset.load_from_json(os.path.join(arc_data_path, 'arc-agi_evaluation_challenges.json'))
        arc_eval_set = arc_eval_set.load_solutions(os.path.join(arc_data_path, 'arc-agi_evaluation_solutions.json'))
        concept_arc = ArcDataset.load_from_neoneye(os.path.join(neoneye_path, 'dataset', 'ConceptARC'))
        mix_datasets = {
            'arceval': arc_eval_set.move_test_to_train().repeat(128),
            'concept': concept_arc.move_test_to_train().repeat(128),
        }
        train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=644, sizes=[6], seed=42, mix_datasets=mix_datasets)

        # augment data set and transform to list
        train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset_augment = train_dataset.augment(**train_aug_opts)
        train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

        # Create DeepSpeed config
        deepspeed_config = {
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
            },
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
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-4,
                    "warmup_num_steps": "auto",
                    "total_num_steps": "auto"
                }
            },
            "gradient_accumulation_steps": 2,
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": not torch.cuda.is_bf16_supported(),
            },
            "bf16": {
                "enabled": torch.cuda.is_bf16_supported(),
            },
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": 4,
        }

        # run training with DeepSpeed
        tokenizer.padding_side = 'right'
        training_args = TrainingArguments(
            output_dir='tmp_output',
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_ratio=0.25,
            learning_rate=1e-4,
            weight_decay=0.00,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy='no',
            report_to='none',
            seed=42,
            deepspeed=deepspeed_config,  # Enable DeepSpeed
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_list(train_dataset_as_list),
            data_collator=InputMaskingDataCollator(
                instruction_template=fmt_opts['query_beg'],
                response_template=fmt_opts['reply_beg'],
                mlm=False,
                tokenizer=tokenizer,
                mask_first_n_examples=1,
            ),
        )
        
        trainer.train()
        model.save_pretrained(f'{save_model_path}-lora')
        tokenizer.save_pretrained(f'{save_model_path}-lora')

    if action == 'merge':
        # load peft weights and merge
        merge_lora_weights(base_model, f'{save_model_path}-lora', f'{save_model_path}-merged')