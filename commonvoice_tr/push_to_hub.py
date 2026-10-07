#!/usr/bin/env python3
"""
Upload trained SLAM-ASR Turkish Checkpoint & LoRA weights to Hugging Face Hub
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo, login

def main():
    parser = argparse.ArgumentParser(description="Upload SLAM-ASR Model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g. username/slam-asr-tr-commonvoice)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/slam_qwen_tr", help="Local directory containing model checkpoints")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    api = HfApi()

    print(f"Creating/verifying repository: {args.repo_id} ...")
    create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"Uploading files from {args.checkpoint_dir} to Hugging Face Hub ({args.repo_id}) ...")
    
    api.upload_folder(
        folder_path=args.checkpoint_dir,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload trained SLAM-ASR Turkish checkpoints"
    )

    print(f"✅ Upload Complete! Model available at: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
