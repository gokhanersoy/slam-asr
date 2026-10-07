#!/usr/bin/env python3
"""
Dataset Preparation Script for Common Voice Turkish (TR)
Supports loading directly from Google Drive / Local TSV directory (cv-corpus-25.0 / 17.0 / etc.)
or HuggingFace datasets as fallback.
Generates SLAM-LLM format JSONL files: train.jsonl, val.jsonl, test.jsonl
"""

import os
import sys
import csv
import json
import argparse
import torch
import soundfile as sf
import torchaudio
import whisper
from tqdm import tqdm

def process_local_tsv_split(tsv_path, clips_dir, split_name, output_dir, audio_dir, max_samples=None):
    os.makedirs(audio_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
    
    if not os.path.exists(tsv_path):
        print(f"⚠️ Warning: TSV file not found: {tsv_path}")
        return

    records = []
    with open(tsv_path, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in, delimiter="\t")
        for row in reader:
            clip_name = row.get("path", "").strip()
            text = row.get("sentence", "").strip()
            if clip_name and text:
                records.append((clip_name, text))
                
    if max_samples:
        records = records[:max_samples]

    total_duration_sec = 0.0
    exported_records = 0

    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for idx, (clip_name, text) in enumerate(tqdm(records, desc=f"Exporting local {split_name}")):
            clip_full_path = os.path.join(clips_dir, clip_name)
            if not os.path.exists(clip_full_path):
                base = os.path.splitext(clip_name)[0]
                possible_paths = [
                    os.path.join(clips_dir, f"{base}.mp3"),
                    os.path.join(clips_dir, f"{base}.wav"),
                    os.path.join(clips_dir, f"{base}.flac"),
                ]
                found = False
                for p in possible_paths:
                    if os.path.exists(p):
                        clip_full_path = p
                        found = True
                        break
                if not found:
                    continue

            try:
                audio_array = whisper.load_audio(clip_full_path)
            except Exception:
                continue

            sr = 16000
            wav_filename = f"{split_name}_{idx:06d}.wav"
            wav_path = os.path.join(audio_dir, wav_filename)

            sf.write(wav_path, audio_array, sr)
            duration = len(audio_array) / sr
            total_duration_sec += duration

            record = {
                "key": f"CV25_TR_{split_name}_{idx:06d}",
                "source": os.path.abspath(wav_path),
                "target": text,
                "duration": round(duration, 2)
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported_records += 1

    print(f"[{split_name.upper()}] Exported {exported_records} items ({total_duration_sec / 3600.0:.2f} hours) -> {jsonl_path}")

def process_hf_split(dataset_split, split_name, output_dir, audio_dir):
    os.makedirs(audio_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
    
    total_duration_sec = 0.0
    exported_records = 0
    
    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for idx, example in enumerate(tqdm(dataset_split, desc=f"Exporting HF {split_name}")):
            audio_info = example.get("audio")
            text = example.get("sentence", "").strip()
            
            if not text or not audio_info:
                continue
            
            array = audio_info["array"]
            sr = audio_info["sampling_rate"]
            
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
    parser.add_argument("--cv_dir", type=str, default="/content/drive/MyDrive/datasets/speech/CommonVoice/cv-corpus-25.0-2026-03-09/tr", help="Path to local Common Voice TR corpus directory")
    parser.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_17_0", help="HuggingFace dataset ID (fallback)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for jsonl manifests and wavs")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace Access Token")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional max train samples for quick experiments")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Optional max val samples")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Optional max test samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    audio_base_dir = os.path.join(args.output_dir, "audio")

    # 1. Try local Google Drive / TSV corpus directory first
    if args.cv_dir and os.path.exists(args.cv_dir):
        print(f"📁 Local Common Voice directory found: {args.cv_dir}")
        clips_dir = os.path.join(args.cv_dir, "clips")

        # Map splits to TSVs
        split_tsv_map = {
            "train": os.path.join(args.cv_dir, "train.tsv"),
            "val": os.path.join(args.cv_dir, "dev.tsv") if os.path.exists(os.path.join(args.cv_dir, "dev.tsv")) else os.path.join(args.cv_dir, "validated.tsv"),
            "test": os.path.join(args.cv_dir, "test.tsv")
        }

        for split_name, tsv_path in split_tsv_map.items():
            max_s = args.max_train_samples if split_name == "train" else (args.max_val_samples if split_name == "val" else args.max_test_samples)
            process_local_tsv_split(
                tsv_path=tsv_path,
                clips_dir=clips_dir,
                split_name=split_name,
                output_dir=args.output_dir,
                audio_dir=os.path.join(audio_base_dir, split_name),
                max_samples=max_s
            )
        return

    # 2. Fallback to Hugging Face datasets if local directory not found
    from datasets import load_dataset, Audio
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
        except Exception:
            token = True

    print(f"Loading Turkish Common Voice dataset from HF: {args.dataset_name} ...")

    try:
        cv_data = load_dataset(args.dataset_name, "tr", token=token)
        cv_data = cv_data.cast_column("audio", Audio(sampling_rate=16000))

        for split in ["train", "validation", "test"]:
            if split in cv_data:
                data_split = cv_data[split]
                split_name = "val" if split == "validation" else split
                
                max_s = args.max_train_samples if split == "train" else (args.max_val_samples if split == "validation" else args.max_test_samples)
                if max_s:
                    data_split = data_split.select(range(min(len(data_split), max_s)))

                process_hf_split(
                    dataset_split=data_split,
                    split_name=split_name,
                    output_dir=args.output_dir,
                    audio_dir=os.path.join(audio_base_dir, split_name)
                )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise e

if __name__ == "__main__":
    main()
