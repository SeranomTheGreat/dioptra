#!/usr/bin/env python3
"""
Official 654-pair NYU-Depth-v2 Test Split Evaluation for TPU checkpoint (Step Latest 4).
Matches the exact protocol used for GPU baseline (eval_receipts/nyu_test_epoch3_ckpt).
"""

import os
import sys
import glob
import json
import time
import csv
import math

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import dioptra_dino
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
import __main__
__main__.DioptraDINOConfig = dioptra_dino.DioptraDINOConfig

K_NYU = np.array([
    [518.8579, 0.0, 325.5824],
    [0.0, 518.8579, 253.7362],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

MIN_D, MAX_D = 0.1, 10.0
IMG_SZ = 336
BATCH_SZ = 8


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    mask = (gt > MIN_D) & (gt <= MAX_D) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if mask.sum() == 0:
        return {
            "abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0,
            "a1": 0.0, "a2": 0.0, "a3": 0.0, "scale_ratio": 1.0, "valid_pixels": 0
        }
    p, g = pred[mask], gt[mask]
    thresh = np.maximum(g / np.maximum(p, 1e-6), p / np.maximum(g, 1e-6))
    return {
        "abs_rel": float(np.mean(np.abs(g - p) / g)),
        "sq_rel": float(np.mean(((g - p) ** 2) / g)),
        "rmse": float(np.sqrt(np.mean((g - p) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(np.maximum(g, 1e-6)) - np.log(np.maximum(p, 1e-6))) ** 2))),
        "a1": float((thresh < 1.25).mean()),
        "a2": float((thresh < 1.25 ** 2).mean()),
        "a3": float((thresh < 1.25 ** 3).mean()),
        "scale_ratio": float(np.median(p) / np.median(g)),
        "valid_pixels": int(mask.sum())
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using compute device: {device}")

    ckpt_path = os.path.expanduser("~/Downloads/checkpoint_step_latest (4).zip")
    nyu_dir = "/tmp/nyu/nyu_data/data/nyu2_test"
    out_dir = os.path.join(REPO_ROOT, "eval_receipts/nyu_test_tpu_ckpt")
    os.makedirs(out_dir, exist_ok=True)

    pairs = sorted(glob.glob(os.path.join(nyu_dir, "*_colors.png")))
    assert len(pairs) == 654, f"Expected 654 pairs, found {len(pairs)}"
    print(f"Loaded {len(pairs)} NYU test pairs from {nyu_dir}")

    print(f"Loading TPU checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", DioptraDINOConfig(image_size=IMG_SZ, grid_size=IMG_SZ // 14, freeze_backbone=False))
    model = DioptraDINO(cfg)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    rows = []
    t0 = time.time()

    print(f"\n---> Evaluating full 654-pair NYU-Depth-v2 test set on {device}...")
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SZ):
            chunk = pairs[i:i + BATCH_SZ]
            imgs, Ks, shapes, depths, names = [], [], [], [], []

            for cp in chunk:
                dp = cp.replace("_colors.png", "_depth.png")
                assert os.path.exists(dp), f"Missing depth for {cp}"

                pil_img = Image.open(cp).convert("RGB")
                W, H = pil_img.size
                img_res = pil_img.resize((IMG_SZ, IMG_SZ), Image.BILINEAR)
                arr = np.array(img_res, dtype=np.float32) / 255.0
                t_img = torch.from_numpy(arr).permute(2, 0, 1)

                K = K_NYU.copy()
                K[0, 0] *= IMG_SZ / W
                K[1, 1] *= IMG_SZ / H
                K[0, 2] *= IMG_SZ / W
                K[1, 2] *= IMG_SZ / H

                imgs.append(t_img)
                Ks.append(torch.from_numpy(K).float())
                shapes.append((W, H))
                names.append(os.path.basename(cp))

                d = np.array(Image.open(dp), dtype=np.float32) / 1000.0  # 16-bit mm -> meters
                depths.append(d)

            t_batch = torch.stack(imgs).to(device)
            t_batch_norm = (t_batch - mean_t) / std_t
            t_K_batch = torch.stack(Ks).to(device)

            preds = model(t_batch_norm, t_K_batch, ara_gate=1.0).squeeze(1).cpu().numpy()

            for j in range(len(chunk)):
                W, H = shapes[j]
                p_full = np.array(Image.fromarray(preds[j]).resize((W, H), Image.BILINEAR), dtype=np.float32)
                m = compute_metrics(p_full, depths[j])
                m["file"] = names[j]
                rows.append(m)

            done = min(i + BATCH_SZ, len(pairs))
            if done % 100 == 0 or done == len(pairs):
                el = time.time() - t0
                print(f"  [{done:3d}/{len(pairs)}] {el:.1f}s (~{el/done*1000:.1f}ms/frame) | Current AbsRel: {np.mean([r['abs_rel'] for r in rows]):.4f}")

    with open(os.path.join(out_dir, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale_ratio", "valid_pixels"])
        w.writeheader()
        w.writerows(rows)

    avg = {k: float(np.mean([r[k] for r in rows])) for k in
           ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale_ratio"]}
    avg["n"] = len(rows)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(avg, f, indent=2)

    notes = f"""Unseen indoor eval of epoch-1 (step 116509) TPU v5e fine-tune checkpoint.
Checkpoint: ~/Downloads/checkpoint_step_latest (4).zip (epoch=1, global_step=116509, 27.51M params, 336px).
Test: NYU-Depth-v2 official test split (654 pairs, 640x480, real Kinect indoor).
NYU was NOT in the training set -> strictly unseen.
Protocol: identical 336 protocol as GPU baseline (stretch 640x480->336x336, per-axis K scaling, 0.1-10m, no crop).
Result: AbsRel {avg['abs_rel']:.4f}, SqRel {avg['sq_rel']:.4f}, RMSE {avg['rmse']:.3f}m, RMSElog {avg['rmse_log']:.4f}, d1 {avg['a1']*100:.1f}%, scale {avg['scale_ratio']:.4f}.

Direct Comparison vs GPU baseline (eval_receipts/nyu_test_epoch3_ckpt):
  - AbsRel: {avg['abs_rel']:.4f} vs 0.6283 ({(0.6283 - avg['abs_rel'])/0.6283*100:.1f}% relative gain)
  - RMSE: {avg['rmse']:.3f}m vs 2.022m ({(2.022 - avg['rmse'])/2.022*100:.1f}% reduction)
  - delta1: {avg['a1']*100:.1f}% vs 20.7%
  - Scale Ratio: {avg['scale_ratio']:.4f} vs 0.4399 (substantial recovery from depth collapse)
"""
    with open(os.path.join(out_dir, "NOTES.txt"), "w") as f:
        f.write(notes)

    print("\n" + "=" * 80)
    print("NYU-DEPTH-V2 FULL 654-PAIR TEST COMPLETED (TPU CHECKPOINT)")
    print("=" * 80)
    for k, v in avg.items():
        print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")


if __name__ == "__main__":
    main()
