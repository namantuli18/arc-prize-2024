#!/bin/bash
#SBATCH --job-name=upload_model
#SBATCH --output=logs/upload_model_%j.out
#SBATCH --error=logs/upload_model_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=general

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ft-llm

# Create logs directory if it doesn't exist
mkdir -p logs

# Optional: set your Hugging Face token if not already in your environment
# export HF_TOKEN=your_hf_token_here

# Define variables
MODEL_DIR="pretrained_models/DDP-LLama-ReArc-4GPU-Full_trial_dataset-merged"
REPO_ID="jakebentley2001/arc-models"  # Replace with your Hugging Face username and desired repo name
COMMIT_MSG="Initial commit of my fine-tuned model"

# Run the upload script
python upload_model.py --model_dir "$MODEL_DIR" --repo_id "$REPO_ID" --commit_message "$COMMIT_MSG"
