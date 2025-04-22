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
import time
import shutil
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
import logging

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import keep_single_char_tokens, save_model_and_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# input paths
base_model = 'nvidia/Mistral-NeMo-Minitron-8B-Base'  # auto-downloaded from huggingface.co
arc_data_path_1 = os.path.join('input', 'arc-prize-2024')  # as on kaggle arc prize 2024
arc_data_path_2 = os.path.join('input', 'arc-prize-2025')
re_arc_path = os.path.join('input', 're_arc')  # https://github.com/michaelhodel/re-arc
neoneye_path = os.path.join('input', 'arc-dataset-collection')  # https://github.com/neoneye/arc-dataset-collection

# output paths
save_model_path = os.path.join('pretrained_models', "Mistral-NeMo-Minitron-Full")

def download_model(model_name):
    """Pre-download the model to ensure it's available before training starts"""
    logger.info(f"Pre-downloading model: {model_name}")
    start_time = time.time()
    
    # Download tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Downloaded tokenizer in {time.time() - start_time:.2f} seconds")
    
    # Then download model without loading it into memory
    try:
        # Just download the model files without loading the model
        AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            local_files_only=False
        )
        logger.info(f"Successfully pre-downloaded model in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error pre-downloading model: {e}")
    
    return tokenizer

def check_dataset_availability():
    """Check if datasets are available and accessible"""
    logger.info("Checking dataset availability...")
    
    data_paths = [
        (arc_data_path_1, "ARC Prize Dataset 2024"),
        (arc_data_path_2, "ARC Prize Dataset 2025"),
        (re_arc_path, "RE-ARC Dataset"),
        (neoneye_path, "NeonEye Dataset Collection")
    ]
    
    all_available = True
    for path, name in data_paths:
        if os.path.exists(path):
            logger.info(f"✓ {name} found at {path}")
            # Check for specific required files based on the dataset
            if name == "ARC Prize Dataset 2024":
                eval_file = os.path.join(path, 'arc-agi_evaluation_challenges.json')
                solutions_file = os.path.join(path, 'arc-agi_evaluation_solutions.json')
                if not (os.path.exists(eval_file) and os.path.exists(solutions_file)):
                    logger.warning(f"⚠ {name} is missing required files")
                    all_available = False
            elif name == "ARC Prize Dataset 2025":
                eval_file = os.path.join(path, 'arc-agi_evaluation_challenges.json')
                solutions_file = os.path.join(path, 'arc-agi_evaluation_solutions.json')
                if not (os.path.exists(eval_file) and os.path.exists(solutions_file)):
                    logger.warning(f"⚠ {name} is missing required files")
                    all_available = False
            elif name == "NeonEye Dataset Collection":
                concept_path = os.path.join(path, 'dataset', 'ConceptARC')
                if not os.path.exists(concept_path):
                    logger.warning(f"⚠ ConceptARC not found in {name}")
                    all_available = False
        else:
            logger.error(f"✗ {name} not found at {path}")
            all_available = False
    
    return all_available

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer for 4-bit quantization"""
    logger.info(f"Loading model and tokenizer from {model_name}")
    start_time = time.time()
    
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
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # device_map="auto"
    )
    
    # Prepare model for training with 4-bit quantization
    model = prepare_model_for_kbit_training(model)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    return model, tokenizer

def merge_lora_weights(base_model_path, adapter_path, output_path):
    """Merge LoRA weights into the base model and save"""
    logger.info(f"Merging LoRA weights from {adapter_path} into base model")
    start_time = time.time()
    
    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # device_map="auto"
    )
    
    # Load and merge LoRA weights
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    # Save the merged model
    save_model_and_tokenizer(output_path, model, tokenizer)
    logger.info(f"LoRA weights merged in {time.time() - start_time:.2f} seconds")
    
    return model, tokenizer

def create_output_dirs():
    """Create necessary output directories"""
    dirs = [
        'tmp_output',
        'offload',
        os.path.dirname(save_model_path)
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

def main():
    """Main execution function with pre-downloads and checks"""
    logger.info("Starting training preparation")
    
    # Create necessary directories
    create_output_dirs()
    
    # First download the model
    logger.info("=== Step 1: Pre-downloading model ===")
    tokenizer = download_model(base_model)
    
    # Check if datasets are available
    logger.info("=== Step 2: Checking datasets ===")
    datasets_available = check_dataset_availability()
    if not datasets_available:
        logger.warning("Some datasets are not available. This may affect training.")
    
    # Continue with regular training process
    for action in ['train', 'merge']:
        # continue if task already accomplished
        if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
            logger.info(f"Skipping {action} as {save_model_path}-lora already exists")
            continue
        if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
            logger.info(f"Skipping {action} as {save_model_path}-merged already exists")
            continue

        logger.info(f"=== Starting {action} phase ===")
        
        # load base model & reduce embedding size
        model = tokenizer = None  # free memory
        model, tokenizer = load_model_and_tokenizer(base_model)
        
        logger.info("Reducing embedding size to only necessary tokens")
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
            logger.info("Configuring LoRA adapter")
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
            logger.info("Loading and preparing training data")
            # arc_train_set_1 = ArcDataset.load_from_json(os.path.join(arc_data_path_1, 'arc-agi_training_challenges.json'))
            # arc_train_set_1 = arc_train_set_1.load_solutions(os.path.join(arc_data_path_1, 'arc-agi_training_solutions.json'))
            # arc_train_set_2 = ArcDataset.load_from_json(os.path.join(arc_data_path_2, 'arc-agi_training_challenges.json'))
            # arc_train_set_2 = arc_train_set_2.load_solutions(os.path.join(arc_data_path_2, 'arc-agi_training_solutions.json'))
            arc_eval_set_1 = ArcDataset.load_from_json(os.path.join(arc_data_path_1, 'arc-agi_evaluation_challenges.json'))
            arc_eval_set_1 = arc_eval_set_1.load_solutions(os.path.join(arc_data_path_1, 'arc-agi_evaluation_solutions.json'))
            # arc_eval_set_2 = ArcDataset.load_from_json(os.path.join(arc_data_path_2, 'arc-agi_evaluation_challenges.json'))
            # arc_eval_set_2 = arc_eval_set_2.load_solutions(os.path.join(arc_data_path_2, 'arc-agi_evaluation_solutions.json'))
            # concept_arc = ArcDataset.load_from_neoneye(os.path.join(neoneye_path, 'dataset', 'ConceptARC'))
            mix_datasets = {
                'arceval_1': arc_eval_set_1.move_test_to_train().repeat(10),
                # 'arceval_1': arc_eval_set_1.move_test_to_train().repeat(128),
                # 'arceval_2': arc_eval_set_2.move_test_to_train().repeat(128),
                # 'arctrain_1': arc_train_set_1.move_test_to_train().repeat(128),
                # 'arctrain_2': arc_train_set_2.move_test_to_train().repeat(128),
                # 'concept': concept_arc.move_test_to_train().repeat(128),
                # 'concept': concept_arc.move_test_to_train().repeat(10),
            }
            #train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=644, sizes=[6], seed=42, mix_datasets=mix_datasets)
            train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=1, sizes=[6], seed=42, mix_datasets=mix_datasets)

            # augment data set and transform to list
            logger.info("Augmenting training data")
            train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
            train_dataset_augment = train_dataset.augment(**train_aug_opts)
            train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)
            logger.info(f"Prepared {len(train_dataset_as_list)} training examples")
            print(f"First example: {train_dataset_as_list[0]}")            

            # Create DeepSpeed config
            logger.info("Setting up DeepSpeed configuration")
            deepspeed_config = {
                "zero_optimization": {
                    "stage": 1,  # Less aggressive memory optimization for single GPU
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
                "gradient_accumulation_steps": 1,  # No need for gradient accumulation with A100
                "gradient_clipping": 1.0,
                "fp16": {
                    "enabled": not torch.cuda.is_bf16_supported(),
                },
                "bf16": {
                    "enabled": torch.cuda.is_bf16_supported(),
                },
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": 8,  # Increased batch size for A100
            }

            # run training with DeepSpeed
            tokenizer.padding_side = 'right'
            training_args = TrainingArguments(
                output_dir='tmp_output',
                num_train_epochs=1,
                per_device_train_batch_size=8,  # Increased to match DeepSpeed config
                gradient_accumulation_steps=1,  # No need for gradient accumulation
                warmup_ratio=0.25,
                learning_rate=1e-4,
                weight_decay=0.00,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                save_strategy='no',
                report_to='none',
                seed=42,
                deepspeed=deepspeed_config,
                remove_unused_columns=False,
            )
            
            logger.info("Starting training with DeepSpeed")
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
            logger.info(f"Training complete, saving model to {save_model_path}-lora")
            model.save_pretrained(f'{save_model_path}-lora')
            tokenizer.save_pretrained(f'{save_model_path}-lora')

        if action == 'merge':
            # load peft weights and merge
            logger.info("Merging LoRA weights into base model")
            merge_lora_weights(base_model, f'{save_model_path}-lora', f'{save_model_path}-merged')

    logger.info("Training process complete")

if __name__ == "__main__":
    main()