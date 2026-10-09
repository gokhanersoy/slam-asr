#!/usr/bin/env python3
"""
SLAM-ASR Fine-Tuning Script for Common Voice Turkish
Trains the Linear Projector and LoRA weights on Qwen2.5 / Llama-3 with Whisper-large-v3 encoder.
"""

import os
import sys
import time
import gc
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.models.slam_model import model_factory
from src.datasets.speech_dataset import get_speech_dataset
from src.utils.compute_wer import compute_wer_cer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@hydra.main(config_path="conf", config_name="prompt", version_base=None)
def main(cfg: DictConfig):
    print("=" * 60)
    print("SLAM-ASR Training Start (Turkish Common Voice)")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_cfg = cfg.train_config
    model_cfg = cfg.model_config
    dataset_cfg = cfg.dataset_config
    peft_cfg = cfg.get("peft_config", None)

    os.makedirs(train_cfg.output_dir, exist_ok=True)

    # 1. Instantiate Model & Tokenizer
    model, tokenizer = model_factory(
        train_config=train_cfg,
        model_config=model_cfg,
        peft_config=peft_cfg
    )

    # 2. Build Datasets & Loaders
    train_dataset = get_speech_dataset(dataset_cfg, tokenizer, split="train")
    val_dataset = get_speech_dataset(dataset_cfg, tokenizer, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size_training,
        shuffle=True,
        collate_fn=train_dataset.collator,
        num_workers=getattr(train_cfg, "num_workers_dataloader", 2),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=getattr(train_cfg, "val_batch_size", 4),
        shuffle=False,
        collate_fn=val_dataset.collator,
        num_workers=getattr(train_cfg, "num_workers_dataloader", 2),
    )

    # 3. Setup Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.lr),
        weight_decay=0.01
    )

    total_steps = len(train_loader) * train_cfg.num_epochs // train_cfg.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg.use_fp16 and torch.cuda.is_available())

    # 4. Training Loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(train_cfg.num_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_mel = batch["audio_mel"].to(device)
            modality_mask = batch["modality_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda', enabled=train_cfg.use_fp16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_mel=audio_mel,
                    modality_mask=modality_mask,
                    labels=labels
                )
                loss = outputs.loss / train_cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * train_cfg.gradient_accumulation_steps

            if (step + 1) % train_cfg.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_step % 50 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch [{epoch+1}/{train_cfg.num_epochs}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item()*train_cfg.gradient_accumulation_steps:.4f} LR: {current_lr:.6f}")

                # Validation interval
                if global_step % train_cfg.validation_interval == 0:
                    model.eval()
                    val_loss = 0.0
                    gc.collect()
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        for val_batch in val_loader:
                            v_ids = val_batch["input_ids"].to(device)
                            v_mask = val_batch["attention_mask"].to(device)
                            v_mel = val_batch["audio_mel"].to(device)
                            v_mmask = val_batch["modality_mask"].to(device)
                            v_labels = val_batch["labels"].to(device)

                            with torch.amp.autocast('cuda', enabled=train_cfg.use_fp16):
                                v_out = model(
                                    input_ids=v_ids,
                                    attention_mask=v_mask,
                                    audio_mel=v_mel,
                                    modality_mask=v_mmask,
                                    labels=v_labels
                                )
                                val_loss += v_out.loss.item()
                                del v_out, v_ids, v_mask, v_mel, v_mmask, v_labels

                    val_loss /= len(val_loader)
                    print(f" validation Loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ckpt_save_path = os.path.join(train_cfg.output_dir, "best_checkpoint.pt")
                        torch.save(model.state_dict(), ckpt_save_path)
                        print(f"--> Saved best model checkpoint to {ckpt_save_path}")

                    gc.collect()
                    torch.cuda.empty_cache()
                    model.train()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} Completed in {elapsed/60.0:.2f} mins. Avg Train Loss: {total_loss/len(train_loader):.4f}")

    # Save final model
    final_ckpt_path = os.path.join(train_cfg.output_dir, "final_checkpoint.pt")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Training Complete! Saved final checkpoint to {final_ckpt_path}")

if __name__ == "__main__":
    main()
