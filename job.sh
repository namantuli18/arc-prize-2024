#!/bin/bash
#SBATCH --job-name=trial_ft
#SBATCH --output=trial_ft.out
#SBATCH --error=trial_ft.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:A6000:1

# Load your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ft-llm

# Set Hugging Face cache directory
export TRANSFORMERS_CACHE="/scratch/$USER/hf_cache/models"

# Create offload directory
mkdir -p offload

# Change directory to your project folder
cd ~/arc-prize-2024

# Run the training script
python3 training_code/jake-finetuning-trial.py 