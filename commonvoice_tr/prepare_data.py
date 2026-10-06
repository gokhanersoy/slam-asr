#!/usr/bin/env python3
"""
Dataset Preparation Script for Common Voice Turkish (TR)
Downloads Common Voice TR from HuggingFace, saves audio to disk (16kHz WAV),
and generates SLAM-LLM format JSONL files: train.jsonl, val.jsonl, test.jsonl
"""

import os
import json
import argparse
import soundfile as sf
import torchaudio
from tqdm import tqdm
from datasets import load_dataset, Audio

def process_and_export_split(dataset_split, split_name, output_dir, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
    
    total_duration_sec = 0.0
    exported_records = 0
    
    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for idx, example in enumerate(tqdm(dataset_split, desc=f"Exporting {split_name}")):
            audio_info = example.get("audio")
            text = example.get("sentence", "").strip()
            
            if not text or not audio_info:
                continue
            
            array = audio_info["array"]
            sr = audio_info["sampling_rate"]
            
            # Resample to 16000Hz if needed
            if sr != 16000:
                audio_tensor = torch.from_numpy(array).unsqueeze(0).float()
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                audio_tensor = resampler(audio_tensor)
                array = audio_tensor.squeeze(0).numpy()
                sr = 16000

            wav_filename = f"{split_name}_{idx:06d}.wav"
            wav_path = os.path.join(audio_dir, wav_filename)
            
            sf.write(wav_path, array, sr)
            duration = len(array) / sr
            total_duration_sec += duration
            
            record = {
                "key": f"CV_TR_{split_name}_{idx:06d}",
                "source": os.path.abspath(wav_path),
                "target": text,
                "duration": round(duration, 2)
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported_records += 1

    print(f"[{split_name.upper()}] Exported {exported_records} items ({total_duration_sec / 3600.0:.2f} hours) -> {jsonl_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Common Voice Turkish Dataset for SLAM-ASR")
    parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_17_0", help="HuggingFace dataset ID")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for jsonl manifests and wavs")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional max train samples for quick experiments")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Optional max val samples")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Optional max test samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    audio_base_dir = os.path.join(args.output_dir, "audio")

    print(f"Loading Turkish dataset: {args.dataset_name} ...")
    try:
        cv_data = load_dataset(args.dataset_name, "tr", trust_remote_code=True)
    except Exception as e:
        print(f"Fallback to common_voice_11_0: {e}")
        cv_data = load_dataset("mozilla-foundation/common_voice_11_0", "tr", trust_remote_code=True)

    cv_data = cv_data.cast_column("audio", Audio(sampling_rate=16000))

    for split in ["train", "validation", "test"]:
        if split in cv_data:
            data_split = cv_data[split]
            split_name = "val" if split == "validation" else split
            
            if split == "train" and args.max_train_samples:
                data_split = data_split.select(range(min(len(data_split), args.max_train_samples)))
            elif split == "validation" and args.max_val_samples:
                data_split = data_split.select(range(min(len(data_split), args.max_val_samples)))
            elif split == "test" and args.max_test_samples:
                data_split = data_split.select(range(min(len(data_split), args.max_test_samples)))

            process_and_export_split(
                dataset_split=data_split,
                split_name=split_name,
                output_dir=args.output_dir,
                audio_dir=os.path.join(audio_base_dir, split_name)
            )

if __name__ == "__main__":
    import torch
    main()
