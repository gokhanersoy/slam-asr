#!/usr/bin/env python3
"""
SLAM-ASR Evaluation Script for Common Voice Turkish
Loads fine-tuned model checkpoint, runs batch evaluation on test set, computes WER and CER.
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from src.models.slam_model import model_factory
from src.datasets.speech_dataset import get_speech_dataset
from src.utils.compute_wer import compute_wer_cer

def main():
    parser = argparse.ArgumentParser(description="Evaluate SLAM-ASR Turkish Model")
    parser.add_argument("--config_path", type=str, default="conf/prompt.yaml", help="Path to prompt.yaml config file")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/slam_qwen_tr/best_checkpoint.pt", help="Path to checkpoint file")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file for evaluation predictions")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on number of test samples to evaluate")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    train_cfg = cfg.train_config
    model_cfg = cfg.model_config
    dataset_cfg = cfg.dataset_config
    dataset_cfg.inference_mode = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating SLAM-ASR Turkish on device: {device}")

    # Load Model
    model, tokenizer = model_factory(
        train_config=train_cfg,
        model_config=model_cfg,
        peft_config=cfg.get("peft_config", None),
        ckpt_path=args.checkpoint_path if os.path.exists(args.checkpoint_path) else None
    )
    model.to(device)
    model.eval()

    # Load Test Dataset
    test_dataset = get_speech_dataset(dataset_cfg, tokenizer, split="test")
    if args.max_samples is not None and args.max_samples < len(test_dataset.data_list):
        test_dataset.data_list = test_dataset.data_list[:args.max_samples]
        print(f"Subsetting test set to first {args.max_samples} samples.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=getattr(train_cfg, "val_batch_size", 4),
        shuffle=False,
        collate_fn=test_dataset.collator,
        num_workers=2
    )

    predictions = []
    references = []
    keys = []
    results = []

    print("Running batch decoding...")
    prompt_str = getattr(dataset_cfg, "prompt", "Aşağıdaki konuşma sesini Türkçe metne dönüştürün. Sadece konuşmanın tam transcriptini yazın.")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            audio_mels = batch["audio_mel"].to(device)
            batch_targets = batch["targets"]
            batch_keys = batch["keys"]

            for idx in range(audio_mels.size(0)):
                mel = audio_mels[idx : idx + 1]
                pred_text = model.generate_transcription(
                    audio_mel=mel,
                    prompt=prompt_str,
                    max_new_tokens=128
                )
                target_text = batch_targets[idx]
                key = batch_keys[idx]

                predictions.append(pred_text)
                references.append(target_text)
                keys.append(key)

                results.append({
                    "key": key,
                    "target": target_text,
                    "prediction": pred_text
                })

    metrics = compute_wer_cer(predictions, references)
    
    print("\n" + "=" * 50)
    print("TURKISH COMMON VOICE ASR EVALUATION RESULTS")
    print("=" * 50)
    print(f"Word Error Rate (WER): {metrics['wer']:.2f}%")
    print(f"Character Error Rate (CER): {metrics['cer']:.2f}%")
    print("=" * 50)

    print("\nSample Transcriptions:")
    for item in results[:5]:
        print(f"Key: {item['key']}")
        print(f"  Target:     {item['target']}")
        print(f"  Prediction: {item['prediction']}")
        print("-" * 50)

    eval_output = {
        "metrics": {
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "total_samples": len(predictions)
        },
        "results": results
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved detailed evaluation results to {args.output_file}")

if __name__ == "__main__":
    main()
