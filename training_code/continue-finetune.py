import os
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments
from datasets import Dataset
from arc_loader import ArcDataset
from model_tools import InputMaskingDataCollator
from model_tools import load_unsloth_4bit, keep_single_char_tokens, save_model_and_tokenizer
from model_tools import merge_peft_into_base
from arc_downloader import download_arc_data

merged_model_path = 'namannn/arc-nemo_full'
new_data_path = '/kaggle/input/arc-prize-2025'

# Output paths for the new training run
save_model_path = os.path.join('pretrained_models', "Mistral-NeMo-Minitron-Full-continued")

# Load your merged model directly
model, tokenizer = load_unsloth_4bit(merged_model_path)

# set formatting options (keep the same as before for consistency)
fmt_opts = dict(
    preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + tokenizer.eos_token,
    lines_sep='\n',
    max_tokens=8192,
)

# Create a new PEFT model based on your merged model
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

# Load your new training data
new_dataset = ArcDataset.load_from_json(os.path.join(new_data_path, 'arc-agi_training_challenges.json'))
new_dataset = new_dataset.load_solutions(os.path.join(new_data_path, 'arc-agi_training_solutions.json'))

# Augment the data
train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=1)
train_dataset_augment = new_dataset.augment(**train_aug_opts)
train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

# Print some debug information about the dataset
print(f"Number of training examples: {len(train_dataset_as_list)}")
if train_dataset_as_list:
    print(f"Example text length: {len(train_dataset_as_list[0]['text'])}")
    print(f"First few characters of example: {train_dataset_as_list[0]['text'][:100]}...")

# Prepare model for training
FastLanguageModel.for_training(model)
tokenizer.padding_side = 'right'

# Create a custom data collator that handles the input format correctly
data_collator = InputMaskingDataCollator(
    instruction_template=fmt_opts['query_beg'],
    response_template=fmt_opts['reply_beg'],
    mlm=False,
    tokenizer=tokenizer,
    mask_first_n_examples=1,
)

# Check a sample batch to ensure correct formatting
sample_dataset = Dataset.from_list(train_dataset_as_list[:2])
sample_batch = data_collator([sample_dataset[0], sample_dataset[1]])
print(f"Sample batch keys: {sample_batch.keys()}")
print(f"Sample batch input_ids shape: {sample_batch['input_ids'].shape}")
print(f"Sample batch labels shape: {sample_batch['labels'].shape}")

# Configure the trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=Dataset.from_list(train_dataset_as_list),
    dataset_text_field="text",
    max_seq_length=fmt_opts['max_tokens'],
    packing=False,  # Try setting this to False if it was True
    data_collator=data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_ratio=0.25,
        num_train_epochs=1,
        learning_rate=5e-5,
        embedding_learning_rate=5e-6,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type='cosine',
        seed=43,
        output_dir='tmp_output',
        save_strategy='no',
        report_to='none',
    ),
)

# Try training with the default compute_loss function first
try:
    print("Starting training...")
    trainer_stats = trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    print("Trying with unsloth_train instead...")
    try:
        trainer_stats = unsloth_train(trainer)
        print("Training with unsloth_train completed successfully!")
    except Exception as e:
        print(f"Error with unsloth_train: {e}")
        print("Please check the error message and dataset format")
        raise

# If training succeeds, save the model
print("Saving model...")
save_model_and_tokenizer(f'{save_model_path}-lora', model, tokenizer)

# Merge the weights
model = merge_peft_into_base(model)
save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)
print("Model saved successfully!")
