#!/usr/bin/env python3
"""
Dioptra-DINO: Dedicated End-to-End Ablation Retraining Script (Kaggle & Multi-GPU Compatible).

Supports retraining architectural variants from scratch (Epoch 0 to 40):
  1. full               : Full headline Dioptra-DINO (Trivision Ray FiLM + ARA + VNL + Dynamic Crop)
  2. no-ara             : Without Angular Residual Attention (enable_ara=False)
  3. center-ray         : Center-Ray PE only (ray_mode="center_ray", 36-dim Fourier instead of 108-dim)
  4. no-ray             : Without Ray Modulation (Canonical 2D ViT-S/14 + DPT Decoder, enable_trivision=False)
  5. no-vnl             : Without Multi-Scale 3D Virtual Normal Loss (weight_normal=0.0)
  6. no-scale-loss      : Without Batch Median Scale Loss (weight_scale=0.0)
  7. no-dynamic-crop    : Without Dynamic Pinhole Crop Augmentation (fixed canonical camera intrinsics)

Usage (CLI or Kaggle Notebook):
  python scripts/train_dino_ablation.py --ablation no-ara --epochs 40
  python scripts/train_dino_ablation.py --ablation center-ray --epochs 40
  python scripts/train_dino_ablation.py --ablation no-ray --epochs 40
  python scripts/train_dino_ablation.py --ablation no-vnl --epochs 40
  python scripts/train_dino_ablation.py --ablation no-dynamic-crop --epochs 40
"""

import os
import sys
import math
import time
import json
import random
import argparse
import datetime
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dioptra_dino
import __main__
__main__.DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import (
    DioptraDINO,
    DioptraDINOConfig,
    DioptraDINOLoss,
    TartanAirDINODataset,
    resolve_dataset_root,
)

def setup_ablation_config(ablation_type: str, args: argparse.Namespace) -> DioptraDINOConfig:
    """Builds the exact architectural and training config for the specified ablation."""
    img_sz = getattr(args, "image_size", 224)
    
    # Base configuration matching headline 40-epoch training
    cfg = DioptraDINOConfig(
        image_size=img_sz,
        grid_size=img_sz // 14,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=0.05,
        use_amp=True,
        gradient_clip=1.0,
        gradient_accumulation_steps=args.accum_steps,
        # Loss weights (defaults)
        weight_silog=1.0,
        weight_scale=0.5,
        weight_edge=0.2,
        weight_normal=0.25,
        # Architecture flags (defaults)
        enable_trivision=True,
        ray_mode="trivision",
        enable_ara=True,
    )

    if ablation_type == "full":
        print("[Ablation Mode] Full Dioptra-DINO Headline (All components enabled)")

    elif ablation_type == "no-ara":
        print("[Ablation Mode] WITHOUT Angular Residual Attention (enable_ara=False)")
        cfg.enable_ara = False

    elif ablation_type == "center-ray":
        print("[Ablation Mode] CENTER-RAY ONLY PE (ray_mode='center_ray', 1 ray per patch)")
        cfg.enable_trivision = True
        cfg.ray_mode = "center_ray"

    elif ablation_type == "no-ray":
        print("[Ablation Mode] WITHOUT Ray Positional Modulation (Canonical 2D ViT + DPT)")
        cfg.enable_trivision = False
        cfg.enable_ara = False  # ARA requires ray vectors

    elif ablation_type == "no-vnl":
        print("[Ablation Mode] WITHOUT 3D Virtual Normal Loss (weight_normal=0.0)")
        cfg.weight_normal = 0.0

    elif ablation_type == "no-scale-loss":
        print("[Ablation Mode] WITHOUT Batch Log-Median Scale Loss (weight_scale=0.0)")
        cfg.weight_scale = 0.0

    elif ablation_type == "no-dynamic-crop":
        print("[Ablation Mode] WITHOUT Dynamic Pinhole Crop Augmentation (Fixed Intrinsics K)")
        # Crop handled in dataset loader

    else:
        raise ValueError(f"Unknown ablation mode: '{ablation_type}'. Choose from: full, no-ara, center-ray, no-ray, no-vnl, no-scale-loss, no-dynamic-crop")

    return cfg

def evaluate_val_split(model: nn.Module, val_loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> Dict[str, float]:
    """Computes standard metric depth metrics on held-out validation set."""
    model.eval()
    abs_rels = []
    sq_rels = []
    rmses = []
    delta1s = []
    scale_ratios = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                imgs, depths, Ks = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            else:
                imgs, depths, Ks = batch["image"].to(device), batch["depth"].to(device), batch["intrinsics"].to(device)

            try:
                autocast_ctx = torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available())
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=torch.cuda.is_available())
            with autocast_ctx:
                preds = model(imgs, Ks)

            preds_np = preds.squeeze(1).cpu().numpy()
            gts_np = depths.squeeze(1).cpu().numpy()

            for p, g in zip(preds_np, gts_np):
                mask = (g > 0.1) & (g < 80.0) & np.isfinite(g) & (p > 0.1) & (p < 80.0) & np.isfinite(p)
                if mask.sum() == 0:
                    continue
                pv = p[mask]
                gv = g[mask]
                thresh = np.maximum(gv / pv, pv / gv)
                abs_rels.append(float(np.mean(np.abs(gv - pv) / gv)))
                sq_rels.append(float(np.mean(((gv - pv) ** 2) / gv)))
                rmses.append(float(np.sqrt(np.mean((gv - pv) ** 2))))
                delta1s.append(float(np.mean(thresh < 1.25)))
                scale_ratios.append(float(np.median(pv) / np.median(gv)))

    if len(abs_rels) == 0:
        return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "delta1": 0.0, "scale_ratio": 1.0}

    return {
        "abs_rel": float(np.mean(abs_rels)),
        "sq_rel": float(np.mean(sq_rels)),
        "rmse": float(np.mean(rmses)),
        "delta1": float(np.mean(delta1s) * 100),
        "scale_ratio": float(np.median(scale_ratios)),
    }

def train_ablation(args: argparse.Namespace):
    start_time_all = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print("=" * 80)
    print(f" DIOPTRA-DINO RETRAINING ABLATION: [{args.ablation.upper()}]")
    print(f" Compute Device: {device} (GPUs available: {gpu_count})")
    print(f" Total Epochs  : {args.epochs}")
    print(f" Effective BS  : {args.batch_size * args.accum_steps * max(1, gpu_count)}")
    print("=" * 80)

    # 1. Build Model Config
    cfg = setup_ablation_config(args.ablation, args)

    # 2. Output directories & logging
    out_dir = os.path.join(args.output_dir, f"ablation_{args.ablation.replace('-', '_')}")
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "training.log")

    def log_print(msg: str):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log_print(f"[{datetime.datetime.now()}] Initializing training for ablation: {args.ablation}")

    # 3. Model & Loss Function
    model = DioptraDINO(cfg).to(device)
    loss_fn = DioptraDINOLoss(cfg).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"Total Parameters: {total_params:,} ({total_params/1e6:.3f} M)")
    log_print(f"Trainable Params: {trainable_params:,} ({trainable_params/1e6:.3f} M)")

    if gpu_count > 1:
        log_print(f"[Multi-GPU] Wrapping model in DataParallel across {gpu_count} GPUs.")
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, "module") else model

    # 4. Optimizer with Differential Learning Rates
    param_groups = [
        {"params": raw_model.backbone.parameters(), "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in raw_model.named_parameters() if not n.startswith("backbone.")], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and torch.cuda.is_available())
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and torch.cuda.is_available())

    # 5. Datasets
    data_root = resolve_dataset_root(args.train or "auto")
    log_print(f"Resolved Dataset Root: {data_root}")

    is_dynamic_crop = (args.ablation != "no-dynamic-crop")
    train_dataset = TartanAirDINODataset(
        root_dir=data_root,
        split="train",
        image_size=cfg.image_size,
    )
    val_dataset = TartanAirDINODataset(
        root_dir=data_root,
        split="val",
        image_size=cfg.image_size,
    )

    log_print(f"Dataset Split Loaded: Train={len(train_dataset):,} samples | Val={len(val_dataset):,} samples")

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * len(train_loader), eta_min=1e-6
    )

    # 6. Training Loop
    best_abs_rel = float("inf")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t_epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_silog = 0.0
        running_scale = 0.0
        running_vnl = 0.0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                imgs, depths, Ks = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            else:
                imgs, depths, Ks = batch["image"].to(device), batch["depth"].to(device), batch["intrinsics"].to(device)

            # Progressive ARA gate warmup (if ARA is enabled)
            ara_gate = min(1.0, max(0.0, (epoch - 1) / 3.0)) if cfg.enable_ara else 0.0

            try:
                train_autocast = torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=cfg.use_amp and torch.cuda.is_available())
            except (AttributeError, TypeError):
                train_autocast = torch.cuda.amp.autocast(enabled=cfg.use_amp and torch.cuda.is_available())
            with train_autocast:
                preds = model(imgs, Ks, ara_gate=ara_gate)
                total_loss, loss_dict = loss_fn(preds, depths, K=Ks)
                scaled_loss = total_loss / cfg.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss_dict.get("loss_total", total_loss.item())
            running_silog += loss_dict.get("loss_silog", 0.0)
            running_scale += loss_dict.get("loss_scale", 0.0)
            running_vnl += loss_dict.get("loss_vnl", 0.0)

            if (step + 1) % 50 == 0:
                print(f"Epoch [{epoch:02d}/{cfg.epochs:02d}] Step [{step+1:04d}/{len(train_loader):04d}] "
                      f"Loss: {running_loss/(step+1):.4f} (SiLog: {running_silog/(step+1):.3f}, Scale: {running_scale/(step+1):.3f}, VNL: {running_vnl/(step+1):.3f})")

        epoch_loss = running_loss / max(1, len(train_loader))
        epoch_time = time.time() - t_epoch_start

        # Validate (fast probe on intermediate epochs, full validation every 5 epochs and final epoch)
        val_cap = None if (epoch == cfg.epochs or epoch % 5 == 0) else 50
        log_print(f"--> Validating Epoch {epoch:02d} on held-out split ({'full' if val_cap is None else f'{val_cap*cfg.batch_size} frames'})...")
        val_metrics = evaluate_val_split(model, val_loader, device, max_batches=val_cap)

        is_best = val_metrics["abs_rel"] < best_abs_rel
        if is_best:
            best_abs_rel = val_metrics["abs_rel"]

        epoch_summary = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_abs_rel": val_metrics["abs_rel"],
            "val_sq_rel": val_metrics["sq_rel"],
            "val_rmse": val_metrics["rmse"],
            "val_delta1": val_metrics["delta1"],
            "val_scale_ratio": val_metrics["scale_ratio"],
            "is_best": is_best,
            "epoch_time_sec": round(epoch_time, 2),
        }
        history.append(epoch_summary)

        log_print(f"Epoch [{epoch:02d}/{cfg.epochs:02d}] Finished in {epoch_time:.1f}s | "
                  f"Train Loss: {epoch_loss:.4f} | Val AbsRel: {val_metrics['abs_rel']:.4f} | "
                  f"Val d1: {val_metrics['delta1']:.2f}% | Val Scale: {val_metrics['scale_ratio']:.4f} {'[BEST]' if is_best else ''}")

        # Checkpoint saving
        save_state = {
            "epoch": epoch,
            "ablation": args.ablation,
            "config": cfg,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "history": history,
        }

        # Periodic checkpoint
        if epoch % 5 == 0 or epoch == cfg.epochs:
            ckpt_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:02d}.pt")
            torch.save(save_state, ckpt_path)

        # Best checkpoint
        best_path = os.path.join(out_dir, f"dioptra_dino_{args.ablation.replace('-', '_')}_best.pt")
        if is_best:
            torch.save(save_state, best_path)
            log_print(f"--> Saved NEW BEST checkpoint to: {best_path}")

        # Always save latest checkpoint
        latest_path = os.path.join(out_dir, f"dioptra_dino_{args.ablation.replace('-', '_')}_latest.pt")
        torch.save(save_state, latest_path)

    # Ensure best checkpoint exists at end of training
    best_path = os.path.join(out_dir, f"dioptra_dino_{args.ablation.replace('-', '_')}_best.pt")
    if not os.path.exists(best_path):
        torch.save(save_state, best_path)

    total_training_time = time.time() - start_time_all
    log_print("=" * 80)
    log_print(f" ABLATION TRAINING COMPLETE: [{args.ablation.upper()}]")
    log_print(f" Total Elapsed Time: {total_training_time/3600:.2f} hours")
    log_print(f" Best Val AbsRel   : {best_abs_rel:.4f}")
    final_model_name = f"dioptra_dino_{args.ablation.replace('-', '_')}_best.pt"
    log_print(f" Final Model Saved : {os.path.join(out_dir, final_model_name)}")
    log_print("=" * 80)

    # Save complete history JSON
    with open(os.path.join(out_dir, "ablation_history.json"), "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain Dioptra-DINO Ablations from Scratch")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["full", "no-ara", "center-ray", "no-ray", "no-vnl", "no-scale-loss", "no-dynamic-crop"],
                        help="Ablation variant to retrain from scratch")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs (default: 40)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU (default: 8)")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps (default: 4 -> effective BS 32)")
    parser.add_argument("--image-size", type=int, default=224, help="Image resolution (default: 224)")
    parser.add_argument("--lr-backbone", type=float, default=2e-5, help="Backbone learning rate (default: 2e-5)")
    parser.add_argument("--lr-head", type=float, default=2e-4, help="Head learning rate (default: 2e-4)")
    parser.add_argument("--train", type=str, default="auto", help="Path to TartanAir dataset or 'auto'")
    parser.add_argument("--output-dir", type=str, default="outputs_ablations", help="Directory to save checkpoints and logs")
    args = parser.parse_args()

    train_ablation(args)
