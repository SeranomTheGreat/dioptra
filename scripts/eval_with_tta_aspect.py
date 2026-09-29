#!/usr/bin/env python3
"""
Advanced Evaluation Harness with Aspect-Preserving Crop & Test-Time Augmentation (TTA).
Evaluates Dioptra-DINO checkpoints with:
  1. Standard Aspect Stretch (Baseline)
  2. Aspect-Preserving Center-Crop (Removes optical distortion)
  3. Aspect-Preserving Crop + Horizontal Flip Ensembling (TTA)
"""

import os
import sys
import glob
import json
import time
import csv
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import dioptra_dino
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
import __main__
__main__.DioptraDINOConfig = dioptra_dino.DioptraDINOConfig

K_SCANNET = np.array([[577.87, 0.0, 319.5], [0.0, 577.87, 239.5], [0.0, 0.0, 1.0]], dtype=np.float32)
K_NYU = np.array([[518.8579, 0.0, 325.5824], [0.0, 518.8579, 253.7362], [0.0, 0.0, 1.0]], dtype=np.float32)

IMG_SZ = 336


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_d: float = 0.1, max_d: float = 10.0) -> dict:
    mask = (gt > min_d) & (gt <= max_d) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if mask.sum() == 0:
        return {"abs_rel": 0.0, "rmse": 0.0, "a1": 0.0, "scale_ratio": 1.0}
    p, g = pred[mask], gt[mask]
    thresh = np.maximum(g / np.maximum(p, 1e-6), p / np.maximum(g, 1e-6))
    return {
        "abs_rel": float(np.mean(np.abs(g - p) / g)),
        "rmse": float(np.sqrt(np.mean((g - p) ** 2))),
        "a1": float((thresh < 1.25).mean()),
        "scale_ratio": float(np.median(p) / np.median(g)),
    }


def evaluate_suite(model: DioptraDINO, pairs: List[Tuple[str, str, np.ndarray]], protocol: str, device: str) -> dict:
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    rows = []
    with torch.no_grad():
        for cp, dp, K_base in pairs:
            img_pil = Image.open(cp).convert("RGB")
            gt = np.array(Image.open(dp), dtype=np.float32)
            if gt.max() > 250:
                gt = gt / 1000.0  # mm to meters
            W, H = img_pil.size

            if protocol == "stretch":
                img_res = img_pil.resize((IMG_SZ, IMG_SZ), Image.BILINEAR)
                K = K_base.copy()
                K[0, 0] *= IMG_SZ / float(W)
                K[1, 1] *= IMG_SZ / float(H)
                K[0, 2] *= IMG_SZ / float(W)
                K[1, 2] *= IMG_SZ / float(H)
                t_img = torch.from_numpy(np.array(img_res, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                t_K = torch.from_numpy(K).unsqueeze(0).to(device)
                pred = model((t_img - mean_t) / std_t, t_K, ara_gate=1.0).squeeze().cpu().numpy()
                p_full = np.array(Image.fromarray(pred).resize((W, H), Image.BILINEAR), dtype=np.float32)
                eval_gt = gt
            elif protocol in ("center_crop", "center_crop_tta"):
                # Preserve 1:1 aspect ratio by center cropping square min(W, H)
                dim = min(W, H)
                left = (W - dim) // 2
                top = (H - dim) // 2
                img_crop = img_pil.crop((left, top, left + dim, top + dim))
                img_res = img_crop.resize((IMG_SZ, IMG_SZ), Image.BILINEAR)
                K = K_base.copy()
                K[0, 2] -= left
                K[1, 2] -= top
                scale = float(IMG_SZ) / float(dim)
                K[0, 0] *= scale
                K[1, 1] *= scale
                K[0, 2] *= scale
                K[1, 2] *= scale

                t_img = torch.from_numpy(np.array(img_res, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                t_K = torch.from_numpy(K).unsqueeze(0).to(device)

                if protocol == "center_crop":
                    pred = model((t_img - mean_t) / std_t, t_K, ara_gate=1.0).squeeze().cpu().numpy()
                else:
                    # Test-Time Augmentation (Horizontal Flip)
                    img_flip = img_res.transpose(Image.FLIP_LEFT_RIGHT)
                    t_flip = torch.from_numpy(np.array(img_flip, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                    p1 = model((t_img - mean_t) / std_t, t_K, ara_gate=1.0)
                    p2 = model((t_flip - mean_t) / std_t, t_K, ara_gate=1.0)
                    p2_unflip = torch.flip(p2, dims=[-1])
                    pred = ((p1 + p2_unflip) * 0.5).squeeze().cpu().numpy()

                p_full = np.array(Image.fromarray(pred).resize((dim, dim), Image.BILINEAR), dtype=np.float32)
                eval_gt = gt[top:top + dim, left:left + dim]

            m = compute_metrics(p_full, eval_gt)
            rows.append(m)

    avg = {k: float(np.mean([r[k] for r in rows])) for k in ["abs_rel", "rmse", "a1", "scale_ratio"]}
    return avg


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    ckpt_path = os.path.expanduser("~/Downloads/checkpoint_step_latest (4).zip")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", DioptraDINOConfig(image_size=IMG_SZ, grid_size=IMG_SZ // 14, freeze_backbone=False))
    model = DioptraDINO(cfg)
    sd = ckpt["model_state_dict"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    # ScanNet pairs
    scan_colors = sorted(glob.glob("data/scannet_tiny/scene00/color/*.jpg"))
    scan_pairs = [(cp, cp.replace("/color/", "/depth/").replace(".jpg", ".png"), K_SCANNET) for cp in scan_colors]

    # NYU 50-pair sample
    nyu_colors = sorted(glob.glob("test_samples/indoor_suite/nyu_depth_v2/image_left/*.png"))
    nyu_pairs = [(cp, cp.replace("image_left", "depth_left").replace(".png", "_depth.npy"), K_NYU) for cp in nyu_colors]

    print("\n" + "=" * 80)
    print("INFERENCE PROTOCOL COMPARISON ON UNSEEN DATASETS (TPU CHECKPOINT)")
    print("=" * 80)

    header = f"{'Dataset':<15} | {'Protocol':<20} | {'AbsRel ↓':<10} | {'RMSE ↓':<10} | {'δ1 ↑':<10} | {'Scale':<10}"
    print(header)
    print("-" * len(header))

    for name, pairs in [("ScanNet-tiny", scan_pairs), ("NYU-Depth-v2", nyu_pairs)]:
        for proto in ["stretch", "center_crop", "center_crop_tta"]:
            res = evaluate_suite(model, pairs, proto, device)
            print(f"{name:<15} | {proto:<20} | {res['abs_rel']:<10.4f} | {res['rmse']:<10.3f} | {res['a1']*100:<9.1f}% | {res['scale_ratio']:<10.4f}")
        print("-" * len(header))


if __name__ == "__main__":
    main()
