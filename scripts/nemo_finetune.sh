#!/bin/bash

#SBATCH --job-name=finetune_nemo         # Job name
#SBATCH --output=finetune_nemo_arc_2025_%j.out   # Standard output (%j = Job ID)
#SBATCH --error=finetune_nemo_arc_2025_%j.err    # Standard error (%j = Job ID)
#SBATCH --partition=general            # Partition with A100 GPUs
#SBATCH --gres=gpu:A100_40GB:2                     # Request 1 GPU
#SBATCH --chdir=/data/user_data/akhild/Dev/arc-prize-2024        #Working directory
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --mem=200Gb                       # Memory allocation
#SBATCH --time=70:00:10                   # Time limit (hh:mm:ss)
#SBATCH --mail-type=ALL                  # Email notifications for all events
#SBATCH --mail-user=akhild@andrew.cmu.edu  # Email address for notifications

# Load Conda into the shell environment
eval "$(conda shell.bash hook)"          # Initialize Conda for this shell session

# Activate the Conda environment
conda activate capstone                  # Replace 'capstone' with the name of your Conda environment

# Run your Python script
python training_code/jake-fine-tune.py --wandb_project "arc-capstone" --wandb_entity "akhildua-carnegie-mellon-university" --wandb_name "nemo_8b_arc_2025"
