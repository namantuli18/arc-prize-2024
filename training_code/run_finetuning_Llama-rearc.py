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
import torch.distributed as dist
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
# Distributed Training Setup
# -------------------------
dist.init_process_group(backend='nccl')  # Use 'nccl' for GPU, 'gloo' for CPU
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
torch.manual_seed(42)  # Ensure deterministic results across GPUs

# -------------------------
# Input/Output Paths
# -------------------------
base_model = 'chuanli11/Llama-3.2-3B-Instruct-uncensored'  # Auto-downloaded from huggingface.co
re_arc_path = os.path.join('input', 're_arc')  # https://github.com/michaelhodel/re-arc
download_arc_data(re_arc_path)

save_model_path = os.path.join('pretrained_models', "Llama-3.2-3B-ReArc")

# -------------------------
# Training/Merging Pipeline
# -------------------------
for action in ['train', 'merge']:
    # Skip if task already accomplished
    if action == 'train' and os.path.exists(f'{save_model_path}-lora'):
        continue
    if action == 'merge' and os.path.exists(f'{save_model_path}-merged'):
        continue

    # Load base model & reduce embedding size
    model = tokenizer = None  # Free memory
    model, tokenizer = load_unsloth_4bit(base_model)
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

    if action == 'train':
        # Load training data
        train_dataset = ArcDataset.load_from_rearc(re_arc_path, n=12, sizes=[6], seed=42)

        # Augment dataset and transform to list (remove examples if exceeding max tokens)
        train_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        train_dataset_augment = train_dataset.augment(**train_aug_opts)
        train_dataset_as_list = train_dataset_augment.as_list(len_name='text', **fmt_opts)

        # Prepare model for training
        FastLanguageModel.for_training(model)
        tokenizer.padding_side = 'right'

        # -------------------------
        # Distributed Training: DDP
        # -------------------------
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # Set to True if there are unused params
        )

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

        if dist.get_rank() == 0:  # Save only on rank 0
            save_model_and_tokenizer(f'{save_model_path}-lora', model.module, tokenizer)

    if action == 'merge':
        # Load PEFT weights and merge
        load_peft_state(model.module, f'{save_model_path}-lora')
        model = merge_peft_into_base(model.module)

        if dist.get_rank() == 0:
            save_model_and_tokenizer(f'{save_model_path}-merged', model, tokenizer)

# -------------------------
# Clean up distributed resources
# -------------------------
dist.destroy_process_group()
