import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

from arc_loader import ArcDataset

# Set the local rank for DDP from the environment variable (default to 0 if not set)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device (local rank): {torch.cuda.current_device()}")
print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9} GB")

# Input paths
#base_model = 'nvidia/Mistral-NeMo-Minitron-8B-Base'  # auto-downloaded from Hugging Face
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'
re_arc_path = os.path.join('input/arc-data/ARC-Data/input', 're_arc')
# Output path
#save_model_path = os.path.join('pretrained_models', "DDP-Mistral-Nemo-8B-ReArc")
save_model_path = os.path.join('pretrained_models', "DDP-LLama-ReArc-8GPU")

def load_model_4bit(model_name_or_path):
    """
    Load a model in 4-bit precision using bitsandbytes.
    Now loads the model onto the correct device using a device map.
    """
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
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

    # Load model with quantization config on the correct device
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": local_rank}  # Load the model on the GPU corresponding to local_rank
    )
    
    return model, tokenizer

def keep_single_char_tokens(model, tokenizer, keep=None, remove_unk=False):
    if keep is None:
        return
    vocab = tokenizer.get_vocab()
    orig_embeds = model.get_input_embeddings().weight.data

    keep_indices = []
    for token in keep:
        if len(token) == 1:
            for i, t in enumerate(vocab.keys()):
                if token in t and len(t) == 1 and i < orig_embeds.shape[0]:
                    keep_indices.append(i)
        else:
            idx = tokenizer.convert_tokens_to_ids(token)
            if (idx != tokenizer.unk_token_id or not remove_unk) and idx < orig_embeds.shape[0]:
                keep_indices.append(idx)
    
    keep_indices = list(set(keep_indices))
    keep_indices = [idx for idx in keep_indices if idx < orig_embeds.shape[0]]
    
    if len(keep_indices) == orig_embeds.shape[0]:
        print("Keeping all embedding tokens, no reduction needed.")
        return keep_indices

    print(f"Reducing embedding matrix from {orig_embeds.shape[0]} to {len(keep_indices)} tokens")
    
    new_embeds_dict = {}
    for i, idx in enumerate(keep_indices):
        new_embeds_dict[idx] = i
    
    tokenizer.add_special_tokens({'additional_special_tokens': [f'[UNUSED{i}]' for i in range(len(keep_indices))]})
    model.resize_token_embeddings(len(tokenizer))
    
    for old_idx, new_idx in new_embeds_dict.items():
        model.get_input_embeddings().weight.data[new_idx] = orig_embeds[old_idx]
        
    print(f"Successfully reduced embedding matrix to {len(keep_indices)} tokens.")
    return keep_indices

def load_tokenized_dataset(dataset_list, tokenizer, max_length=2048):
    from datasets import Dataset

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

class SimpleDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

def setup_peft_model(model, r=256, lora_alpha=24, target_modules=None):
    if target_modules is None:
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 
            'gate_proj', 'up_proj', 'down_proj',
            'embed_tokens', 'lm_head'
        ]
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

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
    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    return model

def save_model_and_tokenizer(save_path, model, tokenizer):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")

def load_peft_state(model, peft_model_path):
    from peft import PeftModel
    model = PeftModel.from_pretrained(
        model,
        peft_model_path
    )
    return model

def merge_peft_into_base(model):
    model = model.merge_and_unload()
    return model

######################################################################
#                         MAIN EXECUTION LOGIC                       #
######################################################################
for action in ['train', 'merge']:
    if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
        continue
    if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
        continue

    print(f"\n=== [ACTION: {action}] Loading base model ===")
    model, tokenizer = load_model_4bit(base_model)
    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=') + tokenizer.tokenize('\n')
    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I',
        reply_beg='\n+/-=O',
        reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n',
        max_tokens=8192,
    )

    print("Before re-mapping:")
    print(" Special tokens map:", tokenizer.special_tokens_map)
    print(" bos_token:", tokenizer.bos_token)
    print(" bos_token_id:", tokenizer.bos_token_id)

    if tokenizer.bos_token == "<|begin_of_text|>":
        tokenizer.bos_token = None

    tokenizer.bos_token = "<bos>"
    tokenizer.add_special_tokens({"bos_token": "<bos>"})
    model.config.bos_token_id = tokenizer.bos_token_id
    model.resize_token_embeddings(len(tokenizer))

    print("\nAfter re-mapping:")
    print(" Special tokens map:", tokenizer.special_tokens_map)
    print(" bos_token:", tokenizer.bos_token)
    print(" bos_token_id:", tokenizer.bos_token_id)

    lora_layers = [
        'q_proj','k_proj','v_proj','o_proj',
        'gate_proj','up_proj','down_proj','embed_tokens','lm_head'
    ]
    model = setup_peft_model(model, r=256, lora_alpha=24, target_modules=lora_layers)

    # No need for explicit model.cuda() here because the model was loaded on the correct device via device_map

    if action == 'train':
        print("=== Starting TRAINING phase ===")
        train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=4, sizes=[6], seed=42)
        train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset_augment = train_dataset.augment(**train_aug_opts)
        train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)
        ten_percent_size = int(0.1 * len(train_dataset_as_list))
        train_dataset_as_list = train_dataset_as_list[:ten_percent_size]
        train_dataset_tokenized = load_tokenized_dataset(train_dataset_as_list, tokenizer, max_length=fmt_opts['max_tokens'])

        print("Final tokenized dataset size:", len(train_dataset_tokenized))

        data_collator = SimpleDataCollator(tokenizer)

        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_ratio=0.25,
            num_train_epochs=1,
            learning_rate=1e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.0,
            lr_scheduler_type='cosine',
            seed=42,
            output_dir='tmp_output',
            save_strategy='no',
            report_to='none',
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,  
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset_tokenized,
            data_collator=data_collator,
            args=training_args,
        )

        print("[Trainer] About to call trainer.train() under DDP.")
        trainer.train()
        print("[Trainer] Training finished.")

        save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)

    if action == 'merge':
        print("=== Starting MERGE phase ===")
        model = load_peft_state(model, f'{save_model_path}-lora')
        model = merge_peft_into_base(model)
        save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)
