#!/bin/bash
#SBATCH --job-name=arc_infer
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=short

# Activate conda environment
source ~/.bashrc
conda activate ft-llm

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables for stability
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print some useful information
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python path: $(which python)"
echo "Current directory: $(pwd)"

# Run inference
python training_code/evaluate.py

echo "Job finished at $(date)" 