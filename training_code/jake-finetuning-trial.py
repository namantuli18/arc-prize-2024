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
import psutil
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import time
import sys

# Set environment variables to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Helps with CUDA initialization issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # Limit memory splits
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # Limit CUDA connections
os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debugging
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand for better stability

# Add parent directory to path to import credentials
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_loader import ArcDataset

# Initialize CUDA and check availability
def init_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    # Set CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # Print CUDA information
    print(f"\n=== CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print("========================\n")
    
    return device

# Initialize CUDA at startup
try:
    device = init_cuda()
except RuntimeError as e:
    print(f"Error initializing CUDA: {e}")
    sys.exit(1)

# Try importing wandb and credentials, but don't fail if not available
try:
    import wandb
    from credentials import WANDB_API_KEY
    WANDB_AVAILABLE = True
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases (wandb) not installed or credentials not found. Install with: pip install wandb")
    print("Make sure to create credentials.py with your WANDB_API_KEY")

def get_gpu_memory_info():
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        return "No GPU available"
    
    info = []
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1e9
        memory_reserved = torch.cuda.memory_reserved(i) / 1e9
        memory_total = gpu.total_memory / 1e9
        info.append(f"GPU {i} ({gpu.name}): {memory_allocated:.2f}/{memory_total:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    return "\n".join(info)

def get_system_memory_info():
    """Get system memory information."""
    memory = psutil.virtual_memory()
    return f"System Memory: {memory.used / 1e9:.2f}/{memory.total / 1e9:.2f} GB used ({memory.percent}%)"

def print_system_info():
    """Print comprehensive system information."""
    print("\n=== System Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print("\n=== GPU Memory Information ===")
    print(get_gpu_memory_info())
    print("\n=== System Memory Information ===")
    print(get_system_memory_info())
    print("========================\n")

# Print initial system info
print_system_info()

# input paths
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'  # auto-downloaded from huggingface.co
re_arc_path = os.path.join('input', 're_arc')  # https://github.com/michaelhodel/re-arc

# output paths
save_model_path = os.path.join('pretrained_models', "Llama-3.2-3B-ReArc")

# Check if wandb is configured
USE_WANDB = WANDB_AVAILABLE and os.getenv('WANDB_API_KEY') is not None
if USE_WANDB:
    print("Weights & Biases logging enabled. Make sure you're logged in with: wandb login")
else:
    print("Weights & Biases logging disabled. To enable:")
    print("1. Install wandb: pip install wandb")
    print("2. Login: wandb login")
    print("3. Set WANDB_API_KEY environment variable")

# DeepSpeed configuration for multi-GPU training
ds_config = {
    "fp16": {
        "enabled": not torch.cuda.is_bf16_supported(),
    },
    "bf16": {
        "enabled": torch.cuda.is_bf16_supported(),
    },
    "zero_optimization": {
        "stage": 3,  # Stage 3 for maximum memory efficiency
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
            "fast_init": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
            "fast_init": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e7,  # Reduced from auto
        "stage3_prefetch_bucket_size": 5e7,  # Reduced from auto
        "stage3_param_persistence_threshold": 1e6,  # Reduced from auto
        "stage3_gather_16bit_weights_on_model_save": True,
        "round_robin_gradients": True
    },
    "gradient_accumulation_steps": 4,  # Increased from 2
    "gradient_clipping": 1.0,
    "steps_per_print": 1,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": True,
    "flops_profiler": {
        "enabled": True,
        "profile_step": 1,
        "module_depth": -1,
        "top_modules": 3,
        "detailed": True,
        "output_file": "flops_profiler.log"
    },
    "monitor": {
        "enabled": True,
        "tensorboard": {
            "enabled": True,
            "output_path": "tensorboard_logs",
            "job_name": "arc_training"
        },
        "wandb": {
            "enabled": True,
            "project": "arc-prize-2024",
            "team": None,
            "group": "deepseed_training"
        },
        "csv": {
            "enabled": True,
            "output_path": "deepspeed_logs",
            "job_name": "arc_training"
        }
    },
    "memory_breakdown": True,  # Enable memory breakdown
    "zero_allow_untested_optimizer": True,  # Allow untested optimizers
    "zero_force_ds_checkpoint_optimizer_state_shard": True  # Force optimizer state sharding
}

def load_model_4bit(model_name):
    """Load model in 4-bit quantization with proper device handling."""
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
            padding_side='right'  # Ensure consistent padding
        )
        
        # Load model with proper device handling
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically handle device placement
            load_in_4bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=False,  # Disable KV cache for training
            max_memory={0: "12GB"},  # Limit GPU memory usage
            offload_folder="offload",  # Enable offloading
            offload_state_dict=True,  # Enable state dict offloading
        )
        
        # Ensure model is on the correct device
        model = model.to(device)
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

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
    
    try:
        # Prepare the model for k-bit training
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
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
        
        # Ensure model is on the correct device
        model = model.to(device)
        
        return model
    except Exception as e:
        print(f"Error setting up PEFT model: {e}")
        raise

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

class GPUMemoryCallback(TrainerCallback):
    """Callback to track GPU memory usage during training."""
    def __init__(self):
        self.start_time = None
        self.last_log_time = None
        self.log_interval = 60  # Log every 60 seconds

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        print("\n=== Starting Training ===")
        print_system_info()

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            elapsed_time = current_time - self.start_time
            print(f"\n=== Training Progress (Step {state.global_step}) ===")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print_system_info()
            
            # Log to wandb if available
            if USE_WANDB and wandb.run is not None:
                wandb.log({
                    "step": state.global_step,
                    "time_elapsed": elapsed_time,
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,
                    "system_memory_used": psutil.virtual_memory().used / 1e9,
                    "system_memory_total": psutil.virtual_memory().total / 1e9,
                })

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print("\n=== Training Complete ===")
        print(f"Total training time: {total_time:.2f} seconds")
        print_system_info()

class DeepSpeedMonitorCallback(TrainerCallback):
    """Callback to monitor DeepSpeed training progress and metrics."""
    def __init__(self):
        self.start_time = None
        self.last_log_time = None
        self.log_interval = 30  # Log every 30 seconds
        self.initialized = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        print("\n=== DeepSpeed Training Started ===")
        print(f"DeepSpeed config: {args.deepspeed}")
        print_system_info()
        
        # Create log directories
        os.makedirs("deepspeed_logs", exist_ok=True)
        os.makedirs("tensorboard_logs", exist_ok=True)
        
        self.initialized = True

    def on_step_end(self, args, state, control, **kwargs):
        if not self.initialized:
            return
            
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            elapsed_time = current_time - self.start_time
            
            # Get DeepSpeed engine if available
            if hasattr(kwargs.get('model', None), 'engine'):
                engine = kwargs['model'].engine
                
                # Log DeepSpeed metrics
                metrics = {
                    "step": state.global_step,
                    "time_elapsed": elapsed_time,
                    "loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
                    "learning_rate": state.log_history[-1].get("learning_rate", 0) if state.log_history else 0,
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,
                    "system_memory_used": psutil.virtual_memory().used / 1e9,
                    "system_memory_total": psutil.virtual_memory().total / 1e9,
                }
                
                # Add DeepSpeed-specific metrics if available
                if hasattr(engine, 'optimizer'):
                    optimizer = engine.optimizer
                    if hasattr(optimizer, 'cur_scale'):
                        metrics['loss_scale'] = optimizer.cur_scale
                    if hasattr(optimizer, 'overflow'):
                        metrics['gradient_overflow'] = optimizer.overflow
                
                # Print metrics
                print("\n=== DeepSpeed Training Progress ===")
                print(f"Step: {metrics['step']}")
                print(f"Loss: {metrics['loss']:.4f}")
                print(f"Learning Rate: {metrics['learning_rate']:.2e}")
                print(f"Time Elapsed: {metrics['time_elapsed']:.2f}s")
                print("\n=== Memory Usage ===")
                print(f"GPU Memory: {metrics['gpu_memory_allocated']:.2f}/{metrics['gpu_memory_reserved']:.2f} GB")
                print(f"System Memory: {metrics['system_memory_used']:.2f}/{metrics['system_memory_total']:.2f} GB")
                print("==============================\n")
                
                # Log to wandb if available
                if USE_WANDB and wandb.run is not None:
                    wandb.log(metrics)
                
                # Save metrics to CSV
                csv_path = os.path.join("deepspeed_logs", "training_metrics.csv")
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w') as f:
                        f.write(",".join(metrics.keys()) + "\n")
                with open(csv_path, 'a') as f:
                    f.write(",".join(str(v) for v in metrics.values()) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print("\n=== DeepSpeed Training Complete ===")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Final loss: {state.log_history[-1].get('loss', 0) if state.log_history else 0}")
        print_system_info()
        
        # Save final metrics
        if USE_WANDB and wandb.run is not None:
            wandb.log({
                "final_loss": state.log_history[-1].get('loss', 0) if state.log_history else 0,
                "total_training_time": total_time,
                "total_steps": state.global_step
            })

# Main execution logic
for action in ['train', 'merge']:
    try:
        # continue if task already accomplished
        if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
            continue
        if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
            continue

        # load base model & reduce embedding size
        model = tokenizer = None  # free memory
        torch.cuda.empty_cache()  # Clear GPU memory
        
        print(f"\nLoading model for {action}...")
        model, tokenizer = load_model_4bit(base_model)
        
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
        model = setup_peft_model(model, r=256, lora_alpha=24, target_modules=lora_layers)

        if action == 'train':
            print("\nLoading training data...")
            # load training data
            train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=4, sizes=[6], seed=42)

            # augment data set and transform to list
            train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
            train_dataset_augment = train_dataset.augment(**train_aug_opts)
            train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

            # Process and tokenize the dataset
            train_dataset_tokenized = load_tokenized_dataset(
                train_dataset_as_list, 
                tokenizer, 
                max_length=fmt_opts['max_tokens']
            )
            
            # Configure data collator
            tokenizer.padding_side = 'right'
            data_collator = InputMaskingDataCollator(
                instruction_template=fmt_opts['query_beg'],
                response_template=fmt_opts['reply_beg'],
                mlm=False,
                tokenizer=tokenizer,
                mask_first_n_examples=1,
            )

            # Setup training arguments with DeepSpeed
            training_args = TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,  # Increased from 2
                warmup_ratio=0.25,
                num_train_epochs=1,
                learning_rate=1e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                logging_first_step=True,
                logging_nan_inf_filter=False,
                optim="adamw_8bit",
                weight_decay=0.00,
                lr_scheduler_type='cosine',
                seed=42,
                output_dir='tmp_output',
                save_strategy='no',
                report_to=['wandb'] if USE_WANDB else 'none',
                deepspeed=ds_config,
                remove_unused_columns=False,
                logging_dir='logs',
                logging_level='info',
                dataloader_pin_memory=True,
                dataloader_num_workers=0,  # Reduced from 4 to avoid tokenizer parallelism issues
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                no_cuda=False,
                local_rank=-1,
                device_map="auto",
                max_grad_norm=1.0,  # Added explicit gradient clipping
                label_names=["labels"],  # Added explicit label names
            )

            # Initialize wandb if available
            if USE_WANDB:
                wandb.init(
                    project="arc-prize-2024",
                    name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "model_name": base_model,
                        "learning_rate": 1e-4,
                        "batch_size": 1,
                        "gradient_accumulation_steps": 4,
                        "warmup_ratio": 0.25,
                        "num_train_epochs": 1,
                        "lora_r": 256,
                        "lora_alpha": 24,
                        "target_modules": lora_layers,
                    }
                )

            # Setup trainer
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset_tokenized,
                data_collator=data_collator,
                args=training_args,
                callbacks=[GPUMemoryCallback(), DeepSpeedMonitorCallback()],  # Add both callbacks
            )
            
            print("\nStarting training...")
            # Train the model
            trainer.train()
            
            # Close wandb run if it was initialized
            if USE_WANDB:
                wandb.finish()
            
            # Save the trained model
            save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)

        if action == 'merge':
            print("\nMerging PEFT weights...")
            # load peft weights and merge
            model = load_peft_state(model, f'{save_model_path}-lora')
            model = merge_peft_into_base(model)
            save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)
            
    except Exception as e:
        print(f"\nError during {action}: {e}")
        raise
    finally:
        # Clean up
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()
