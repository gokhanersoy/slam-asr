#!/usr/bin/env python3
"""
Dataset Preparation Script for Local Mozilla Common Voice Corpus (TSV + clips)
Supports reading from Google Drive path e.g.:
/content/drive/MyDrive/datasets/speech/ComonVoice/cv-corpus-25.0-2026-03-09
Processes TSVs (train.tsv, dev.tsv, test.tsv), converts clips (.mp3 / .wav) to 16kHz WAV on Colab SSD,
and generates SLAM-LLM format JSONL files: train.jsonl, val.jsonl, test.jsonl
"""

import os
import sys
import csv
import json
import argparse
import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm

def find_cv_structure(base_dir):
    """
    Finds the directory containing TSV files and clips folder.
    Checks base_dir, base_dir/tr, and subdirectories.
    """
    candidates = [
        base_dir,
        os.path.join(base_dir, "tr"),
    ]
    
    # Also search one level deep
    if os.path.exists(base_dir):
        for sub in os.listdir(base_dir):
            full_sub = os.path.join(base_dir, sub)
            if os.path.isdir(full_sub):
                candidates.append(full_sub)
                candidates.append(os.path.join(full_sub, "tr"))

    for cand in candidates:
        if os.path.exists(cand):
            clips_dir = os.path.join(cand, "clips")
            has_train = os.path.exists(os.path.join(cand, "train.tsv"))
            if os.path.exists(clips_dir) and has_train:
                return cand, clips_dir
            
    # Fallback if clips is directly in base_dir
    if os.path.exists(os.path.join(base_dir, "clips")):
        return base_dir, os.path.join(base_dir, "clips")

    return base_dir, os.path.join(base_dir, "clips")

def process_tsv_split(tsv_path, clips_dir, split_name, output_dir, audio_dir, max_samples=None):
    os.makedirs(audio_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
    
    if not os.path.exists(tsv_path):
        print(f"⚠️ TSV file not found: {tsv_path}, skipping split {split_name}.")
        return

    records = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_name = row.get("path", "").strip()
            sentence = row.get("sentence", "").strip()
            if audio_name and sentence:
                records.append((audio_name, sentence))

    if max_samples:
        records = records[:max_samples]

    print(f"[{split_name.upper()}] Processing {len(records)} samples from {tsv_path} ...")

    exported_count = 0
    total_duration_sec = 0.0

    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for idx, (audio_name, text) in enumerate(tqdm(records, desc=f"Exporting {split_name}")):
            # Audio path in clips
            src_audio_path = os.path.join(clips_dir, audio_name)
            if not os.path.exists(src_audio_path):
                # Try adding extension if missing
                if os.path.exists(src_audio_path + ".mp3"):
                    src_audio_path = src_audio_path + ".mp3"
                elif os.path.exists(src_audio_path + ".wav"):
                    src_audio_path = src_audio_path + ".wav"
                else:
                    continue

            # Load audio using torchaudio or soundfile
            try:
                waveform, sr = torchaudio.load(src_audio_path)
            except Exception as e:
                # Fallback load
                try:
                    import whisper
                    audio_np = whisper.load_audio(src_audio_path)
                    waveform = torch.from_numpy(audio_np).unsqueeze(0)
                    sr = 16000
                except Exception:
                    continue

            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16000 Hz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
                sr = 16000

            out_wav_filename = f"{split_name}_{idx:06d}.wav"
            out_wav_path = os.path.join(audio_dir, out_wav_filename)
            
            array = waveform.squeeze(0).numpy()
            sf.write(out_wav_path, array, sr)

            duration = len(array) / sr
            total_duration_sec += duration

            record = {
                "key": f"CV_{split_name}_{idx:06d}",
                "source": os.path.abspath(out_wav_path),
                "target": text,
                "duration": round(duration, 2)
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported_count += 1

    print(f"[{split_name.upper()}] Successfully exported {exported_count} items ({total_duration_sec / 3600.0:.2f} hours) -> {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Common Voice Local Corpus (TSVs + clips) for SLAM-ASR")
    parser.add_argument("--cv_dir", type=str, required=True, help="Path to local Common Voice directory (e.g. /content/drive/MyDrive/.../cv-corpus-25.0-2026-03-09)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for jsonl manifests and 16kHz wavs")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional max train samples for fast experiments")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Optional max val samples")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Optional max test samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    audio_base_dir = os.path.join(args.output_dir, "audio")

    root_dir, clips_dir = find_cv_structure(args.cv_dir)
    print(f"📂 Found Common Voice Root: {root_dir}")
    print(f"🔊 Found Audio Clips Dir:  {clips_dir}")

    # Map standard split TSVs
    splits_map = {
        "train": os.path.join(root_dir, "train.tsv"),
        "val": os.path.join(root_dir, "dev.tsv") if os.path.exists(os.path.join(root_dir, "dev.tsv")) else os.path.join(root_dir, "validation.tsv"),
        "test": os.path.join(root_dir, "test.tsv")
    }

    for split_name, tsv_file in splits_map.items():
        max_s = None
        if split_name == "train":
            max_s = args.max_train_samples
        elif split_name == "val":
            max_s = args.max_val_samples
        elif split_name == "test":
            max_s = args.max_test_samples

        process_tsv_split(
            tsv_path=tsv_file,
            clips_dir=clips_dir,
            split_name=split_name,
            output_dir=args.output_dir,
            audio_dir=os.path.join(audio_base_dir, split_name),
            max_samples=max_s
        )

if __name__ == "__main__":
    main()
