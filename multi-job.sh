#!/bin/bash
#SBATCH --job-name=arc_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100_80GB:4  # Request 4 GPUs
#SBATCH --partition=general

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ft-llm

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables for better stability
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# Run the training script using torchrun for DDP
torchrun --nproc_per_node=4 --master_port=29501 training_code/multi-gpu-mistral.py
