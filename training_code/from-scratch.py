import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

from arc_loader import ArcDataset

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9} GB")

# Input paths
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'  # auto-downloaded from huggingface.co
re_arc_path = os.path.join('input/arc-data/ARC-Data/input', 're_arc')  # https://github.com/michaelhodel/re-arc

# Output path
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

    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    return model, tokenizer

def keep_single_char_tokens(model, tokenizer, keep=None, remove_unk=False):
    """
    Keep only the specified tokens in the embedding matrix (optional).
    """
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
    model.resize_token_embeddings(len(keep_indices))
    
    for old_idx, new_idx in new_embeds_dict.items():
        model.get_input_embeddings().weight.data[new_idx] = orig_embeds[old_idx]
        
    print(f"Successfully reduced embedding matrix to {len(keep_indices)} tokens.")
    return keep_indices

def load_tokenized_dataset(dataset_list, tokenizer, max_length=2048):
    """
    Tokenize a list of strings for causal language modeling.
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
    Data collator that masks certain parts of the input if needed.
    """
    def __init__(self, tokenizer, instruction_template="I", response_template="\n+/-=O", mlm=False, mask_first_n_examples=1):
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
            
            for i in range(len(features)):
                if i >= len(batch["input_ids"]):
                    continue
                text = self.tokenizer.decode(batch["input_ids"][i])
                
                instr_pos = text.find(self.instruction_template)
                resp_pos = text.find(self.response_template)
                
                # Mask out labels for the instruction part
                if instr_pos != -1 and resp_pos != -1 and i < self.mask_first_n_examples:
                    instr_token_pos = len(self.tokenizer.encode(text[:instr_pos])) - 1
                    resp_token_pos = len(self.tokenizer.encode(text[:resp_pos])) - 1
                    if 0 <= instr_token_pos < resp_token_pos < batch["labels"][i].shape[0]:
                        batch["labels"][i, instr_token_pos:resp_token_pos] = -100
        
        return batch
    

class DebuggingDataCollator(InputMaskingDataCollator):
    """
    A debugging collator that intercepts out-of-range token IDs and prints them
    before they cause the "device-side assert" in the embedding lookup.
    """
    def __init__(self, tokenizer, model, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.model = model

    def __call__(self, features):
        # Use the parent collator to get a properly padded batch
        batch = super().__call__(features)

        # How many tokens the embedding actually has
        embedding_size = self.model.get_input_embeddings().weight.size(0)

        # Check each token in input_ids
        for i, input_ids_example in enumerate(batch["input_ids"]):
            for j, token_id in enumerate(input_ids_example):
                if token_id >= embedding_size or token_id < 0:
                    # Decode the single out-of-range token
                    token_str = self.tokenizer.decode([token_id])
                    print(
                        f"Out-of-range token encountered: '{token_str}' "
                        f"(ID: {token_id}) at position {j} in batch item {i}. "
                        f"Embedding size is {embedding_size}."
                    )
                    # Optionally raise an error so training stops immediately
                    raise ValueError("Encountered out-of-range token ID!")

        return batch


def setup_peft_model(model, r=256, lora_alpha=24, target_modules=None):
    """
    Setup a model for PEFT training using LoRA.
    """
    if target_modules is None:
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 
            'gate_proj', 'up_proj', 'down_proj',
            'embed_tokens', 'lm_head'
        ]
    
    # Prepare model for k-bit training (PEFT)
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
    """
    Save model and tokenizer to disk.
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")

def load_peft_state(model, peft_model_path):
    """
    Load PEFT adapter weights.
    """
    
    # Directly wraps (or updates) your model with the LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        peft_model_path
    )
    return model
    # if not isinstance(model, PeftModel):
    #     model = PeftModel.from_pretrained(model, peft_model_path)
    # else:
    #     model.load_adapter(peft_model_path)
    # return model

def merge_peft_into_base(model):
    """
    Merge LoRA/PEFT adapters into the base model weights.
    """
    model = model.merge_and_unload()
    return model

# Main execution logic
for action in ['train', 'merge']:
    if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
        continue
    if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
        continue

    # Load base model & reduce embedding size
    model, tokenizer = load_model_4bit(base_model)
    keep_tok = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.:,;*+/-=') + tokenizer.tokenize('\n')
    keep_single_char_tokens(model, tokenizer, keep=keep_tok, remove_unk=True)

    # Formatting options
    fmt_opts = dict(
        preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
        query_beg='I',
        reply_beg='\n+/-=O',
        reply_end='\n' + tokenizer.eos_token,
        lines_sep='\n',
        max_tokens=8192,
    )

    # 1) Check your current special tokens
    print("Before re-mapping:")
    print(" Special tokens map:", tokenizer.special_tokens_map)
    print(" bos_token:", tokenizer.bos_token)
    print(" bos_token_id:", tokenizer.bos_token_id)

    # 2) Remove the old token if needed
    # If your tokenizer has bos_token == '<|begin_of_text|>', reset it
    if tokenizer.bos_token == "<|begin_of_text|>":
        tokenizer.bos_token = None

    # 3) Define a new <bos> token and add it to the tokenizer
    tokenizer.bos_token = "<bos>"
    tokenizer.add_special_tokens({"bos_token": "<bos>"})  # ensures <bos> is recognized


    # 4) Update your model config if needed
    model.config.bos_token_id = tokenizer.bos_token_id

    # 5) (Optional) If you want the model’s embedding matrix to include the new token
    #    and if <bos> did not exist before, call resize_token_embeddings:
    model.resize_token_embeddings(len(tokenizer))

    # 6) Verify the changes
    print("\nAfter re-mapping:")
    print(" Special tokens map:", tokenizer.special_tokens_map)
    print(" bos_token:", tokenizer.bos_token)
    print(" bos_token_id:", tokenizer.bos_token_id)

    # Create LoRA model
    lora_layers = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens','lm_head']
    model = setup_peft_model(model, r=256, lora_alpha=24, target_modules=lora_layers)

    if action == 'train':
        # Load training data
        train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=4, sizes=[6], seed=42)
        train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset_augment = train_dataset.augment(**train_aug_opts)
        train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)


        # >>> Here is the key step <<<
        # Take only 10% of the entire list
        ten_percent_size = int(0.1 * len(train_dataset_as_list))
        train_dataset_as_list = train_dataset_as_list[:ten_percent_size]


        # Tokenize
        train_dataset_tokenized = load_tokenized_dataset(
            train_dataset_as_list, 
            tokenizer, 
            max_length=fmt_opts['max_tokens']
        )
        
        # Data collator
        tokenizer.padding_side = 'right'
        # data_collator = InputMaskingDataCollator(
        #     instruction_template=fmt_opts['query_beg'],
        #     response_template=fmt_opts['reply_beg'],
        #     mlm=False,
        #     tokenizer=tokenizer,
        #     mask_first_n_examples=1,
        # )
        debug_data_collator = DebuggingDataCollator(
            tokenizer=tokenizer,
            model=model,
            instruction_template="I",
            response_template="\n+/-=O",
            mlm=False,
            mask_first_n_examples=1,
        )

        # **TrainingArguments** without DeepSpeed
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
            weight_decay=0.0,
            lr_scheduler_type='cosine',
            seed=42,
            output_dir='tmp_output',
            save_strategy='no',
            report_to='none',
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset_tokenized,
            data_collator=debug_data_collator,  #ADDING DEBUGGING COLLATOR
            args=training_args,
        )

        
        # Train the model
        trainer.train()
        
        # Save the trained LoRA model
        save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)

    if action == 'merge':
        # Load LoRA weights and merge
        model = load_peft_state(model, f'{save_model_path}-lora')
        model = merge_peft_into_base(model)
        save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)
