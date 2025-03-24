#!/usr/bin/env python
import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder

def main():
    parser = argparse.ArgumentParser(
        description="Upload a model directory to the Hugging Face Hub."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the directory containing the model files."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The repository id on the Hugging Face Hub (e.g., 'jakebentley2001/arc-models')."
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Initial commit of my fine-tuned model",
        help="Commit message for the upload."
    )
    args = parser.parse_args()

    # Optionally, if you have an HF token in your environment variable HF_TOKEN, it will be used automatically.
    hf_token = os.environ.get("HF_TOKEN", None)

    api = HfApi()

    # Create repository if it doesn't exist
    try:
        repo_info = api.repo_info(args.repo_id, token=hf_token)
        print(f"Repository '{args.repo_id}' already exists.")
    except Exception as e:
        print(f"Repository '{args.repo_id}' not found, creating it...")
        create_repo(args.repo_id, token=hf_token, repo_type="model", exist_ok=True)
    
    # Upload the folder
    print(f"Uploading folder '{args.model_dir}' to repository '{args.repo_id}'...")
    upload_folder(
        repo_id=args.repo_id,
        folder_path=args.model_dir,
        commit_message=args.commit_message,
        ignore_patterns=["*.pyc", ".git/*"],
        token=hf_token
    )
    print("Model uploaded successfully.")

if __name__ == "__main__":
    main()
