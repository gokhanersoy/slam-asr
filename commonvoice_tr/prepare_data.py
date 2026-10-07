#!/usr/bin/env python3
"""
Dataset Preparation Script for Common Voice Turkish 27.0 via Mozilla Data Collective (MDC)
Downloads dataset directly from mozilladatacollective.com using CV_API key,
extracts tar.gz archive, converts audio to 16kHz WAV, and generates SLAM-LLM JSONL manifests.
"""

import os
import sys
import tarfile
import csv
import json
import argparse
import requests
import torch
import soundfile as sf
import torchaudio
from tqdm import tqdm

MDC_DATASET_ID = "cmu5wkah500c2o10719lem8m8"
MDC_API_URL = f"https://mozilladatacollective.com/api/datasets/{MDC_DATASET_ID}/download"

def get_cv_api_key(args_api_key=None):
    if args_api_key:
        return args_api_key
    if "CV_API" in os.environ:
        return os.environ["CV_API"]
    try:
        from google.colab import userdata
        key = userdata.get('CV_API')
        if key:
            return key
    except Exception:
        pass
    return None

def download_dataset_from_mdc(api_key, save_path):
    print("Fetching presigned download URL from Mozilla Data Collective...")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(MDC_API_URL, headers=headers)
    if response.status_code != 200:
        print(f"❌ MDC API Error ({response.status_code}): {response.text}")
        sys.exit(1)
    
    data = response.json()
    download_url = data.get("downloadUrl")
    if not download_url:
        print(f"❌ No downloadUrl found in response: {data}")
        sys.exit(1)

    print(f"Downloading Common Voice 27.0 Turkish dataset to {save_path} ...")
    dl_res = requests.get(download_url, stream=True)
    total_size = int(dl_res.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in dl_res.iter_content(chunk_size=1024 * 1024):
            size = f.write(chunk)
            bar.update(size)

    print("✅ Download completed!")

def extract_tar_archive(tar_path, extract_dir):
    print(f"Extracting archive {tar_path} -> {extract_dir} ...")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("✅ Extraction completed!")

def find_dataset_root(base_dir):
    # Find directory containing train.tsv / dev.tsv
    for root, dirs, files in os.walk(base_dir):
        if "train.tsv" in files:
            return root
    return base_dir

def process_tsv_split(tsv_path, clips_dir, split_name, output_dir, audio_dir, max_samples=None):
    if not os.path.exists(tsv_path):
        print(f"⚠️ Warning: TSV file {tsv_path} not found. Skipping split '{split_name}'.")
        return

    os.makedirs(audio_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")

    total_duration_sec = 0.0
    exported_records = 0

    with open(tsv_path, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in, delimiter="\t")
        rows = list(reader)

    if max_samples and max_samples < len(rows):
        rows = rows[:max_samples]

    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for idx, row in enumerate(tqdm(rows, desc=f"Processing {split_name}")):
            audio_filename = row.get("path", "")
            text = row.get("sentence", "").strip()

            if not audio_filename or not text:
                continue

            # Full path to original clip (mp3/wav)
            src_audio_path = os.path.join(clips_dir, audio_filename)
            if not os.path.exists(src_audio_path):
                # Try adding extension if missing
                if os.path.exists(src_audio_path + ".mp3"):
                    src_audio_path += ".mp3"
                elif os.path.exists(src_audio_path + ".wav"):
                    src_audio_path += ".wav"
                else:
                    continue

            wav_filename = f"{split_name}_{idx:06d}.wav"
            wav_path = os.path.join(audio_dir, wav_filename)

            try:
                waveform, sr = torchaudio.load(src_audio_path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    waveform = resampler(waveform)
                    sr = 16000

                array = waveform.squeeze(0).numpy()
                sf.write(wav_path, array, sr)
                duration = len(array) / sr
                total_duration_sec += duration

                record = {
                    "key": f"CV27_TR_{split_name}_{idx:06d}",
                    "source": os.path.abspath(wav_path),
                    "target": text,
                    "duration": round(duration, 2)
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported_records += 1
            except Exception as e:
                continue

    print(f"[{split_name.upper()}] Exported {exported_records} items ({total_duration_sec / 3600.0:.2f} hours) -> {jsonl_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Common Voice 27.0 Turkish Dataset via Mozilla Data Collective")
    parser.add_argument("--api_key", type=str, default=None, help="Mozilla Data Collective API key (CV_API)")
    parser.add_argument("--tar_path", type=str, default="data/cv27_tr.tar.gz", help="Path to save or existing dataset tar.gz")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for manifests and converted wavs")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional max train samples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Optional max val samples")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Optional max test samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    extract_dir = os.path.join(args.output_dir, "raw_extracted")

    api_key = get_cv_api_key(args.api_key)

    # Step 1: Download if tar.gz does not exist
    if not os.path.exists(args.tar_path) and not os.path.exists(extract_dir):
        if not api_key:
            print("\n" + "=" * 70)
            print("⚠️ HATA: Mozilla Data Collective CV_API Anahtarı Bulunamadı!")
            print("=" * 70)
            print("Lütfen Colab Secrets (Gizli Anahtarlar) alanına `CV_API` adıyla")
            print("Mozilla Data Collective API anahtarınızı ekleyin.")
            print("=" * 70 + "\n")
            sys.exit(1)
        
        download_dataset_from_mdc(api_key, args.tar_path)

    # Step 2: Extract tar.gz if extract_dir does not exist
    if os.path.exists(args.tar_path) and not os.path.exists(extract_dir):
        extract_tar_archive(args.tar_path, extract_dir)

    root_dir = find_dataset_root(extract_dir)
    clips_dir = os.path.join(root_dir, "clips")
    audio_output_base = os.path.join(args.output_dir, "audio")

    # Step 3: Process TSVs and convert audio
    print(f"Processing dataset from {root_dir} ...")
    
    splits = [
        ("train.tsv", "train", args.max_train_samples),
        ("dev.tsv", "val", args.max_val_samples),
        ("test.tsv", "test", args.max_test_samples),
    ]

    for tsv_name, split_name, max_s in splits:
        tsv_path = os.path.join(root_dir, tsv_name)
        process_tsv_split(
            tsv_path=tsv_path,
            clips_dir=clips_dir,
            split_name=split_name,
            output_dir=args.output_dir,
            audio_dir=os.path.join(audio_output_base, split_name),
            max_samples=max_s
        )

    print("\n🎉 Common Voice 27.0 Turkish Dataset Preparation Complete!")

if __name__ == "__main__":
    main()
