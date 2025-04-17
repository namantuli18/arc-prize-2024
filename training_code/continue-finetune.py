import os
import sys
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import load_unsloth_4bit, keep_single_char_tokens, save_model_and_tokenizer
from model_tools import merge_peft_into_base

# This script avoids using Unsloth-specific training functions to bypass the shape assertion error

merged_model_path = 'namannn/arc-nemo_full'
new_data_path = '/kaggle/input/arc-prize-2025'

# Output paths for the new training run
save_model_path = os.path.join('pretrained_models', "Mistral-NeMo-Minitron-Full-continued")

# Add prints for debugging
print("Loading model from", merged_model_path)
# Load your merged model
model, tokenizer = load_unsloth_4bit(merged_model_path)

# Set formatting options
fmt_opts = dict(
    preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + tokenizer.eos_token,
    lines_sep='\n',
    max_tokens=8192,
)

# Ensure tokenizer settings are correct
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

# Print tokenizer info for debugging
print(f"Tokenizer: {tokenizer.__class__.__name__}")
print(f"Vocabulary size: {len(tokenizer)}")
print(f"Model class: {model.__class__.__name__}")

# Important: Disable Unsloth's fast forward and use standard PEFT
print("Creating standard PEFT model (not using Unsloth's FastLanguageModel)")
# Define standard PEFT config instead of using Unsloth's wrapper
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=64,  # Using a smaller rank for stability
    lora_alpha=16, 
    lora_dropout=0.0,
    bias="none",
)

# Get standard PEFT model
model = get_peft_model(model, peft_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

print("Loading and preparing dataset")
# Load your new training data
new_dataset = ArcDataset.load_from_json(os.path.join(new_data_path, 'arc-agi_training_challenges.json'))
new_dataset = new_dataset.load_solutions(os.path.join(new_data_path, 'arc-agi_training_solutions.json'))

# Augment the data (use simpler augmentation if needed)
train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=1)
train_dataset_augment = new_dataset.augment(**train_aug_opts)
train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

# Print dataset info
print(f"Number of training examples: {len(train_dataset_as_list)}")
if train_dataset_as_list:
    print(f"First example length: {len(train_dataset_as_list[0]['text'])}")

# Use the normal data collator for compatibility
data_collator = InputMaskingDataCollator(
    instruction_template=fmt_opts['query_beg'],
    response_template=fmt_opts['reply_beg'],
    mlm=False,
    tokenizer=tokenizer,
    mask_first_n_examples=1,
)

print("Configuring trainer")
# Use standard Transformers Trainer (not Unsloth's)
training_args = TrainingArguments(
    output_dir='tmp_output',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.25,
    num_train_epochs=1,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.00,
    lr_scheduler_type='cosine',
    seed=43,
    save_strategy='no',
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=Dataset.from_list(train_dataset_as_list),
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting training with standard Trainer")
try:
    trainer_stats = trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}", file=sys.stderr)
    print("Printing inputs/outputs that may help debug:")
    # Try to examine a sample batch for debugging
    try:
        sample_inputs = {k: v[:1] for k, v in data_collator([train_dataset_as_list[0]]).items()}
        print(f"Sample input keys: {sample_inputs.keys()}")
        print(f"Sample input shapes: {[(k, v.shape) for k, v in sample_inputs.items() if hasattr(v, 'shape')]}")
    except Exception as debug_error:
        print(f"Error during debugging: {debug_error}")
    raise

print("Saving model...")
save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)

# Merge the weights
try:
    model = merge_peft_into_base(model)
    save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)
    print("Model merged and saved successfully!")
except Exception as e:
    print(f"Error during model merging: {e}. Skipping merge step.")
    print("The LoRA model has been saved successfully and can be used as is.")
