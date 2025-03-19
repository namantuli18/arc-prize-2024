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
from accelerate import Accelerator
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments
from datasets import Dataset

from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import load_unsloth_4bit, keep_single_char_tokens, save_model_and_tokenizer
from model_tools import load_peft_state, merge_peft_into_base
from arc_downloader import download_arc_data

# -------------------------
# Initialize Accelerator
# -------------------------
accelerator = Accelerator(mixed_precision="bf16" if is_bfloat16_supported() else "fp16")
torch.manual_seed(42)  # Ensure deterministic results across GPUs

# -------------------------
# Input/Output Paths
# -------------------------
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'  # Auto-downloaded from huggingface.co
re_arc_path = os.path.join('input', 're_arc')  # https://github.com/michaelhodel/re-arc
download_arc_data(re_arc_path)

save_model_path = os.path.join('pretrained_models', "Llama-3.2-3B-ReArc")

# DeepSpeed configuration for multi-GPU training
ds_config = {
    "fp16": {
        "enabled": not torch.cuda.is_bf16_supported(),
    },
    "bf16": {
        "enabled": torch.cuda.is_bf16_supported(),
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": 2,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}

def load_model_4bit(model_name_or_path):
    """
    Load a model in 4-bit precision using bitsandbytes, ensuring DeepSpeed compatibility.
    """
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # First load tokenizer to get correct padding token
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load model with quantization config but WITHOUT device_map
    # Let DeepSpeed handle the device mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    return model, tokenizer

def keep_single_char_tokens(model, tokenizer, keep=None, remove_unk=False):
    """
    Keep only the specified tokens in the embedding matrix.
    """
    if keep is None:
        return
    
    # Get the vocabulary and original embedding matrix
    vocab = tokenizer.get_vocab()
    orig_embeds = model.get_input_embeddings().weight.data
    
    # Identify indices of tokens to keep
    keep_indices = []
    for token in keep:
        # Handle single characters
        if len(token) == 1:
            for i, t in enumerate(vocab.keys()):
                if token in t and len(t) == 1 and i < orig_embeds.shape[0]:
                    keep_indices.append(i)
        else:
            # Handle multi-character tokens
            idx = tokenizer.convert_tokens_to_ids(token)
            if (idx != tokenizer.unk_token_id or not remove_unk) and idx < orig_embeds.shape[0]:
                keep_indices.append(idx)
    
    # Get unique indices and ensure they're all valid
    keep_indices = list(set(keep_indices))
    keep_indices = [idx for idx in keep_indices if idx < orig_embeds.shape[0]]
    
    # If we're keeping all tokens, no need to resize
    if len(keep_indices) == orig_embeds.shape[0]:
        print("Keeping all embedding tokens, no reduction needed.")
        return keep_indices
    
    print(f"Reducing embedding matrix from {orig_embeds.shape[0]} to {len(keep_indices)} tokens")
    
    # Create new embeddings dictionary
    new_embeds_dict = {}
    for i, idx in enumerate(keep_indices):
        new_embeds_dict[idx] = i
    
    # Create mapping for tokenizer
    # This is important for correctly mapping token IDs during training
    tokenizer.add_special_tokens({'additional_special_tokens': [f'[UNUSED{i}]' for i in range(len(keep_indices))]})
    
    # Create a completely new embedding matrix
    model.resize_token_embeddings(len(keep_indices))
    
    # Copy original embeddings to new ones
    for old_idx, new_idx in new_embeds_dict.items():
        model.get_input_embeddings().weight.data[new_idx] = orig_embeds[old_idx]
        
    print(f"Successfully reduced embedding matrix to {len(keep_indices)} tokens.")
    return keep_indices

def load_tokenized_dataset(dataset_list, tokenizer, max_length=2048):
    """
    Properly tokenize a dataset from ArcDataset.as_list() format.
    
    The ArcDataset.as_list() returns examples with:
    - 'text': full prompt with train examples and test query
    - 'key': identifier for the problem
    - 'train': formatted training examples
    - 'query': formatted test query
    - 'input': combined train+query
    - 'reply': expected output
    """
    # First convert to a simple format with just text
    simple_dataset = []
    for item in dataset_list:
        if isinstance(item, dict) and 'text' in item:
            simple_dataset.append({'raw_text': item['text']})
    
    # Create dataset from simplified list
    dataset = Dataset.from_list(simple_dataset)
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['raw_text'],
            padding=False,  # We'll handle padding in the data collator
            truncation=True,
            max_length=max_length,
            return_tensors=None  # Return as list, not tensor
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['raw_text'],
        desc="Tokenizing texts"
    )
    
    return tokenized_dataset

class InputMaskingDataCollator:
    """
    Data collator that masks input tokens for training.
    Handles the ARC format where we want to mask the instruction tokens
    so the model only learns to predict the response.
    """
    def __init__(self, tokenizer, instruction_template="I", response_template="\n+/-=O", mlm=False, mask_first_n_examples=1):
        self.tokenizer = tokenizer
        self.instruction_template = instruction_template
        self.response_template = response_template
        self.mlm = mlm
        self.mask_first_n_examples = mask_first_n_examples
    
    def __call__(self, features):
        # Features should already have input_ids
        if not isinstance(features, list) or len(features) == 0 or 'input_ids' not in features[0]:
            raise ValueError("Features must be a list of dictionaries with 'input_ids'")
            
        # Pad the batch
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        if not self.mlm:
            # For standard causal language modeling
            batch["labels"] = batch["input_ids"].clone()
            
            # Find position of response_template in each sequence
            for i in range(len(features)):
                if i >= len(batch["input_ids"]):
                    continue
                    
                # Convert IDs to text to find response marker
                text = self.tokenizer.decode(batch["input_ids"][i])
                
                # Find position of instruction and response templates
                instr_pos = text.find(self.instruction_template)
                resp_pos = text.find(self.response_template)
                
                if instr_pos != -1 and resp_pos != -1 and i < self.mask_first_n_examples:
                    # Calculate token positions
                    instr_token_pos = len(self.tokenizer.encode(text[:instr_pos])) - 1  # -1 because of BOS
                    resp_token_pos = len(self.tokenizer.encode(text[:resp_pos])) - 1    # -1 because of BOS
                    
                    # Mask out labels for the instruction part (set to -100)
                    if 0 <= instr_token_pos < resp_token_pos < batch["labels"][i].shape[0]:
                        batch["labels"][i, instr_token_pos:resp_token_pos] = -100
        
        return batch

def setup_peft_model(model, r=256, lora_alpha=24, target_modules=None):
    """
    Setup a model for PEFT training using LoRA with DeepSpeed compatibility.
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                        'gate_proj', 'up_proj', 'down_proj',
                        'embed_tokens', 'lm_head']
    
    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}  # Better compatibility with DeepSpeed
    )
    
    # Define LoRA config optimized for DeepSpeed
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    return model

def save_model_and_tokenizer(save_path, model, tokenizer):
    """
    Save model and tokenizer to disk.
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the model
    model.save_pretrained(save_path)
    
    # Save the tokenizer
    tokenizer.save_pretrained(save_path)
    
    print(f"Model and tokenizer saved to: {save_path}")

def load_peft_state(model, peft_model_path):
    """
    Load PEFT adapter weights.
    """
    # Load the PEFT model
    if not isinstance(model, PeftModel):
        model = PeftModel.from_pretrained(model, peft_model_path)
    else:
        # If it's already a PeftModel, load the adapter
        model.load_adapter(peft_model_path)
    
    return model

def merge_peft_into_base(model):
    """
    Merge PEFT adapters into the base model.
    """
    # Merge weights
    model = model.merge_and_unload()
    return model

# Main execution logic
for action in ['train', 'merge']:
    # Skip if task already accomplished
    if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
        continue
    if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
        continue

    # Load base model & reduce embedding size
    model = tokenizer = None  # Free memory
    model, tokenizer = load_unsloth_4bit(base_model)

    device = torch.device(f"cuda:{accelerator.local_process_index}")

    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=') + tokenizer.tokenize('\n')
    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

    # Set formatting options
    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I',
        reply_beg='\n+/-=O',
        reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n',
        max_tokens=128000,
    )

    # Create LoRA model
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

    # Prepare model with accelerate
    model, tokenizer = accelerator.prepare(model, tokenizer)

    if action == 'train':
        # Load training data
        train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=368, sizes=[6], seed=42)

        # Augment dataset and transform to list (remove examples if exceeding max tokens)
        train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset_augment = train_dataset.augment(**train_aug_opts)
        train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

        # Prepare model for training
        FastLanguageModel.for_training(model)
        tokenizer.padding_side = 'right'

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
            ),
        )

        trainer_stats = unsloth_train(trainer)

        # Save only on the main process
        if accelerator.is_main_process:
            save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)

    if action == 'merge':
        # Load PEFT weights and merge
        load_peft_state(model, f'{save_model_path}-lora')
        model = merge_peft_into_base(model)

        if accelerator.is_main_process:
            save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)

