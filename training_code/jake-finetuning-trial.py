import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

from arc_loader import ArcDataset

# Set tokenizers parallelism to False to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get local rank for distributed training
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# Set the device
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

# Initialize process group only when doing distributed training
if world_size > 1:
    # Use the SLURM environment variables to determine addresses if available
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ.get('SLURM_PROCID'))
        local_rank = int(os.environ.get('SLURM_LOCALID'))
        world_size = int(os.environ.get('SLURM_NTASKS'))
        
    # Initialize process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

# Print system info only from rank 0
if local_rank == 0:
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9} GB")
    print(f"Using {world_size} GPUs with local_rank: {local_rank}")

# input paths
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'  # auto-downloaded from huggingface.co
re_arc_path = os.path.join('input', 're_arc')  # https://github.com/michaelhodel/re-arc

# output paths
save_model_path = os.path.join('pretrained_models', "Llama-3.2-3B-ReArc")


def load_model_4bit(model_name_or_path, device_map=None):
    """
    Load a model in 4-bit precision using bitsandbytes.
    Modified to support better DDP compatibility.
    """
    from transformers import BitsAndBytesConfig
    
    # First load tokenizer
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
    
    # Special handling for DDP with 4-bit models:
    # 1. We use a simpler quantization config
    # 2. We avoid double quantization
    # 3. We ensure proper compute dtype
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,  # Must be False for DDP with 4-bit models
    )

    # For 4-bit models with DDP, each process must handle its own GPU
    if device_map is None:
        device_map = {"": local_rank}
    
    # Load model with specific configuration for DDP compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=compute_dtype,
    )
        
    return model, tokenizer


def keep_single_char_tokens(model, tokenizer, keep=None, remove_unk=False):
    """
    Keep only the specified tokens in the embedding matrix.
    """
    if keep is None:
        return
    
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
    
    keep_indices = list(set(keep_indices))
    keep_indices = [idx for idx in keep_indices if idx < orig_embeds.shape[0]]
    
    # If we're keeping all tokens, no need to resize
    if len(keep_indices) == orig_embeds.shape[0]:
        if local_rank == 0:
            print("Keeping all embedding tokens, no reduction needed.")
        return keep_indices
    
    if local_rank == 0:
        print(f"Reducing embedding matrix from {orig_embeds.shape[0]} to {len(keep_indices)} tokens")
    
    new_embeds_dict = {}
    for i, idx in enumerate(keep_indices):
        new_embeds_dict[idx] = i
    
    # Add some placeholder special tokens so HF is okay with the new vocab size
    tokenizer.add_special_tokens({'additional_special_tokens': [f'[UNUSED{i}]' for i in range(len(keep_indices))]})
    
    # Resize
    model.resize_token_embeddings(len(keep_indices))
    
    # Copy original embeddings to new ones
    for old_idx, new_idx in new_embeds_dict.items():
        model.get_input_embeddings().weight.data[new_idx] = orig_embeds[old_idx]
        
    if local_rank == 0:
        print(f"Successfully reduced embedding matrix to {len(keep_indices)} tokens.")
    return keep_indices


def load_tokenized_dataset(dataset_list, tokenizer, max_length=2048):
    """
    Properly tokenize a dataset from ArcDataset.as_list() format.
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
    Data collator that masks certain input tokens for training.
    """
    def __init__(self, tokenizer, instruction_template="I", response_template="\n+/-=O",
                 mlm=False, mask_first_n_examples=1):
        self.tokenizer = tokenizer
        self.instruction_template = instruction_template
        self.response_template = response_template
        self.mlm = mlm
        self.mask_first_n_examples = mask_first_n_examples
    
    def __call__(self, features):
        if not isinstance(features, list) or len(features) == 0 or 'input_ids' not in features[0]:
            raise ValueError("Features must be a list of dictionaries with 'input_ids'")
            
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        if not self.mlm:
            batch["labels"] = batch["input_ids"].clone()
            
            # Find position of response_template in each sequence
            for i in range(len(features)):
                if i >= len(batch["input_ids"]):
                    continue
                
                text = self.tokenizer.decode(batch["input_ids"][i])
                
                instr_pos = text.find(self.instruction_template)
                resp_pos = text.find(self.response_template)
                
                # Only mask the instruction part for the first N examples
                if instr_pos != -1 and resp_pos != -1 and i < self.mask_first_n_examples:
                    instr_token_pos = len(self.tokenizer.encode(text[:instr_pos])) - 1
                    resp_token_pos = len(self.tokenizer.encode(text[:resp_pos])) - 1
                    
                    if 0 <= instr_token_pos < resp_token_pos < batch["labels"][i].shape[0]:
                        batch["labels"][i, instr_token_pos:resp_token_pos] = -100
        
        return batch


def setup_peft_model(model, r=256, lora_alpha=24, target_modules=None):
    """
    Setup a model for PEFT training (LoRA).
    Modified for better DDP compatibility.
    """
    if target_modules is None:
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
            'embed_tokens', 'lm_head'
        ]
    
    # Prepare model for k-bit training with specific settings for DDP
    # The critical setting here is use_reentrant=False which prevents 
    # issues with backward pass synchronization in DDP
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    # Configure LoRA with DDP-friendly settings
    # Specific things that help with DDP:
    # - No dropout (more deterministic across ranks)
    # - No bias parameters (fewer parameters to sync)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    
    # Apply PEFT to the model
    model = get_peft_model(model, lora_config)
    
    # Critical for PEFT models with DDP - ensures that input gradients
    # are properly calculated and synchronized across GPUs
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    return model


def save_model_and_tokenizer(save_path, model, tokenizer):
    # Only save from rank 0
    if local_rank != 0:
        return
        
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")


def load_peft_state(model, peft_model_path):
    if not isinstance(model, PeftModel):
        model = PeftModel.from_pretrained(model, peft_model_path)
    else:
        model.load_adapter(peft_model_path)
    return model


def merge_peft_into_base(model):
    model = model.merge_and_unload()
    return model


if __name__ == "__main__":
    try:
        # Initial barrier to ensure all processes start together
        if world_size > 1:
            torch.distributed.barrier()
            
        if local_rank == 0:
            print(f"Starting distributed training with {world_size} processes")
            
        # We cycle through 'train' and then 'merge'
        for action in ['train', 'merge']:
            # Skip completed steps (but only check on rank 0)
            skip_action = False
            if local_rank == 0:
                if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
                    print(f"LoRA model already exists at {save_model_path}-lora, skipping training")
                    skip_action = True
                if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
                    print(f"Merged model already exists at {save_model_path}-merged, skipping merge")
                    skip_action = True
            
            # Broadcast skip_action decision from rank 0 to all ranks
            if world_size > 1:
                skip_action_tensor = torch.tensor([1 if skip_action else 0], device=device)
                torch.distributed.broadcast(skip_action_tensor, src=0)
                skip_action = bool(skip_action_tensor.item())
                torch.distributed.barrier()
                
            if skip_action:
                continue

            if local_rank == 0:
                print(f"Starting {action} phase")
                
            # Load base model
            model, tokenizer = load_model_4bit(base_model)
            
            # Define tokens to keep
            keep_tok = list(
                'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-='
            ) + tokenizer.tokenize('\n')
            
            # Synchronize after model loading
            if world_size > 1:
                torch.distributed.barrier()
                
            # Reduce embedding size
            keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

            # Formatting options
            fmt_opts = dict(
                preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
                query_beg='I',
                reply_beg='\n+/-=O',
                reply_end='\n' + tokenizer.eos_token,
                lines_sep='\n',
                max_tokens=128000,
            )

            # Set up LoRA
            lora_layers = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
                'embed_tokens', 'lm_head'
            ]
            
            model = setup_peft_model(model, r=256, lora_alpha=24, target_modules=lora_layers)
            
            # Synchronize after model preparation
            if world_size > 1:
                torch.distributed.barrier()

            if action == 'train':
                # Load and prepare training data
                if local_rank == 0:
                    print("Loading training data")
                    
                # Load training data
                train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=4, sizes=[6], seed=42)
                train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
                train_dataset_augment = train_dataset.augment(**train_aug_opts)
                train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

                # Tokenize dataset
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

                # Training arguments with proper DDP settings
                training_args = TrainingArguments(
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    warmup_ratio=0.25,
                    num_train_epochs=1,
                    learning_rate=1e-4,
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    logging_steps=10,
                    # Use PyTorch's native AdamW for better DDP compatibility
                    optim="adamw_torch",  
                    weight_decay=0.00,
                    lr_scheduler_type='cosine',
                    seed=42,
                    output_dir='tmp_output',
                    save_strategy='no',
                    report_to='none',
                    # Critical for PEFT models with DDP
                    ddp_find_unused_parameters=True, 
                    remove_unused_columns=False,
                    # DDP-specific optimizations
                    ddp_bucket_cap_mb=25,
                    dataloader_pin_memory=False,
                    # Set proper local_rank for distributed
                    local_rank=local_rank,
                    # Disable default DDP plugin for better compatibility with quantized models
                    ddp_backend="no_c10d",
                )
                
                if local_rank == 0:
                    print("Setting up trainer")

                # Set up Trainer
                trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset_tokenized,
                    data_collator=data_collator,
                    args=training_args,
                )
                
                if local_rank == 0:
                    print("Starting training")

                # Train the model
                trainer.train()

                # Synchronize after training
                if world_size > 1:
                    torch.distributed.barrier()
                    
                # Save model (only rank 0 will actually save)
                save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)

            if action == 'merge':
                # Only merge on rank 0 to avoid conflicts
                if local_rank == 0:
                    print("Starting merge process")
                    # Merge LoRA weights into base
                    model = load_peft_state(model, f'{save_model_path}-lora')
                    model = merge_peft_into_base(model)
                    save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)
                    print("Merge completed successfully")
                
                # Synchronize after merge
                if world_size > 1:
                    torch.distributed.barrier()
                    
        if local_rank == 0:
            print("All operations completed successfully")
                    
    except Exception as e:
        # Print any exceptions for debugging
        print(f"Process {local_rank} encountered an exception: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always clean up the process group
        if world_size > 1:
            dist.destroy_process_group()