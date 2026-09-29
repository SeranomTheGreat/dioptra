#!/usr/bin/env python3
"""
evaluate_epoch_10.py
--------------------
Rigorous evaluation of the newly fine-tuned Dioptra-DINO Epoch 10 model (336x336 high-resolution)
across 16 indoor benchmark environments (715 test frames) on Apple Silicon MPS hardware.

Compares head-to-head against:
1. Dioptra-DINO Epoch 5 Baseline (prior checkpoint)
2. Depth Anything V2 Metric Indoor Small (CVPR 2024)
3. Metric3D ViT-Small (TPAMI 2024)

Outputs:
- mac_outputs/epoch_10_evaluation_results.json
- mac_outputs/epoch_10_evaluation_summary.md
"""

import os
import sys
import glob
import json
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image as PILImage
import cv2
import torch
import torch.nn.functional as F

import dioptra_dino
sys.modules["__main__"].DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINO, DioptraDINOConfig
from benchmark_all_indoor_competitors import (
    scale_intrinsics,
    compute_metrics,
    depth_to_surface_normals,
    K_TARTANAIR_V1,
    K_TARTANAIR_V2,
    K_NYU_V2,
    K_SCANNET,
)
from benchmark_highres_indoor_suite import define_indoor_dataset_splits


class Epoch10Evaluator:
    def __init__(self, checkpoint_path: str, device: str = "mps"):
        self.device = device
        cfg = DioptraDINOConfig(image_size=336, grid_size=24)
        self.model = DioptraDINO(cfg)
        print(f"[Epoch 10 Eval] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print(f"[Epoch 10 Eval] Model loaded ({self.params:,} params) on {device} ✓")

    def predict(self, pil_img: PILImage.Image, K_native: np.ndarray, orig_w: int, orig_h: int, res: int = 336) -> Tuple[np.ndarray, float]:
        img_res = pil_img.resize((res, res), PILImage.BILINEAR)
        img_np = np.array(img_res, dtype=np.float32) / 255.0
        img_norm = (img_np - self.mean) / self.std
        img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        K_res = scale_intrinsics(K_native, orig_w, orig_h, res, res)
        K_t = torch.from_numpy(K_res).unsqueeze(0).to(self.device)

        if self.device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.model(img_t, K_t, ara_gate=1.0)
            pred_full = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return pred_full.squeeze().cpu().numpy().astype(np.float32), dt_ms


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_path = "staging_epoch_10/dioptra_dino_epoch_10.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    evaluator = Epoch10Evaluator(ckpt_path, device=device)
    splits = define_indoor_dataset_splits()

    os.makedirs("mac_outputs", exist_ok=True)
    all_results = {}

    print("\n" + "=" * 80)
    print(f"BENCHMARKING DIOPTRA-DINO EPOCH 10 (NATIVE 336x336) ON {device.upper()}")
    print(f"Test Environments: {len(splits)}")
    print("=" * 80 + "\n")

    eval_res = 336

    for split in splits:
        split_id = split["id"]
        split_name = split["name"]
        cat = split["category"]
        img_pat = split["img_pattern"]
        depth_dir = split["depth_dir"]
        K = split["K"]
        max_eval_d = split["max_eval_depth"]
        max_f = split.get("max_frames", 50)

        img_files = sorted(glob.glob(img_pat))
        if not img_files:
            continue
        if max_f and len(img_files) > max_f:
            img_files = img_files[:max_f]

        metrics_list = []
        latencies = []

        for img_path in img_files:
            base = os.path.basename(img_path)
            stem = os.path.splitext(base)[0]

            possible_names = [
                f"{stem}_depth.npy",
                f"{stem}.npy",
                f"{stem}.png",
                f"{stem}_lcam_front_depth.png",
                f"{stem}_left_depth.npy",
            ]
            d_path = None
            for candidate in possible_names:
                p = os.path.join(depth_dir, candidate)
                if os.path.exists(p):
                    d_path = p
                    break

            if not d_path:
                continue

            try:
                img = PILImage.open(img_path).convert("RGB")
                orig_w, orig_h = img.size

                if d_path.endswith(".npy"):
                    gt = np.load(d_path).astype(np.float32)
                elif "scannet" in split_id:
                    gt = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                elif "tartanair2" in split_id.lower() or "depth.png" in d_path.lower():
                    raw = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)
                    if raw is not None and raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 4:
                        gt = np.ascontiguousarray(raw).view(np.float32).squeeze(-1)
                    else:
                        gt = raw.astype(np.float32) if raw is not None else np.zeros((orig_h, orig_w), dtype=np.float32)
                else:
                    gt = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

                if gt.ndim == 3:
                    gt = gt.squeeze()

                pred, dt = evaluator.predict(img, K, orig_w, orig_h, res=eval_res)
                m = compute_metrics(pred, gt, K=K, min_depth=0.1, max_depth=max_eval_d)
                if m and "abs_rel" in m:
                    metrics_list.append(m)
                    latencies.append(dt)
            except Exception as e:
                continue

        if not metrics_list:
            continue

        avg_m = {
            "abs_rel": float(np.mean([x["abs_rel"] for x in metrics_list])),
            "sq_rel": float(np.mean([x["sq_rel"] for x in metrics_list])),
            "rmse": float(np.mean([x["rmse"] for x in metrics_list])),
            "mae": float(np.mean([x["mae"] for x in metrics_list])),
            "silog": float(np.mean([x["silog"] for x in metrics_list])),
            "delta1": float(np.mean([x["delta1"] for x in metrics_list])),
            "delta2": float(np.mean([x["delta2"] for x in metrics_list])),
            "delta3": float(np.mean([x["delta3"] for x in metrics_list])),
            "scale_ratio": float(np.mean([x["scale_ratio"] for x in metrics_list])),
            "normal_mae": float(np.mean([x["normal_mae"] for x in metrics_list])),
            "latency_ms": float(np.mean(latencies)),
            "fps": float(1000.0 / np.mean(latencies)),
            "num_evaluated": len(metrics_list),
        }

        all_results[split_id] = {
            "name": split_name,
            "category": cat,
            "metrics": avg_m,
        }

        print(
            f"✓ {split_name:<30s} | AbsRel: {avg_m['abs_rel']:.4f} | RMSE: {avg_m['rmse']:.3f}m | "
            f"δ1: {avg_m['delta1']*100:.1f}% | Scale: {avg_m['scale_ratio']:.3f} | Latency: {avg_m['latency_ms']:.1f}ms"
        )

    # Compute Macro Averages
    macro_avg = {
        "abs_rel": float(np.mean([v["metrics"]["abs_rel"] for v in all_results.values()])),
        "sq_rel": float(np.mean([v["metrics"]["sq_rel"] for v in all_results.values()])),
        "rmse": float(np.mean([v["metrics"]["rmse"] for v in all_results.values()])),
        "mae": float(np.mean([v["metrics"]["mae"] for v in all_results.values()])),
        "silog": float(np.mean([v["metrics"]["silog"] for v in all_results.values()])),
        "delta1": float(np.mean([v["metrics"]["delta1"] for v in all_results.values()])),
        "delta2": float(np.mean([v["metrics"]["delta2"] for v in all_results.values()])),
        "delta3": float(np.mean([v["metrics"]["delta3"] for v in all_results.values()])),
        "scale_ratio": float(np.mean([v["metrics"]["scale_ratio"] for v in all_results.values()])),
        "normal_mae": float(np.mean([v["metrics"]["normal_mae"] for v in all_results.values()])),
        "latency_ms": float(np.mean([v["metrics"]["latency_ms"] for v in all_results.values()])),
        "fps": float(1000.0 / np.mean([v["metrics"]["latency_ms"] for v in all_results.values()])),
        "total_environments": len(all_results),
        "total_frames": sum(v["metrics"]["num_evaluated"] for v in all_results.values()),
    }

    print("\n" + "=" * 80)
    print("MACRO-AVERAGE INDOOR RESULTS (EPOCH 10 @ 336x336)")
    print("=" * 80)
    print(f"Total Environments Evaluated: {macro_avg['total_environments']}")
    print(f"Total Test Frames           : {macro_avg['total_frames']}")
    print(f"Mean AbsRel Error           : {macro_avg['abs_rel']:.4f}")
    print(f"Mean RMSE                   : {macro_avg['rmse']:.3f} m")
    print(f"Mean MAE                    : {macro_avg['mae']:.3f} m")
    print(f"Delta 1 (< 1.25) Accuracy   : {macro_avg['delta1']*100:.2f}%")
    print(f"Delta 2 (< 1.25^2) Accuracy : {macro_avg['delta2']*100:.2f}%")
    print(f"Metric Scale Ratio          : {macro_avg['scale_ratio']:.3f}")
    print(f"Surface Normal MAE          : {macro_avg['normal_mae']:.1f}°")
    print(f"Inference Latency (MPS)     : {macro_avg['latency_ms']:.1f} ms ({macro_avg['fps']:.1f} FPS)")
    print("=" * 80 + "\n")

    # Save JSON
    out_data = {
        "model": "Dioptra-DINO (Epoch 10 High-Res)",
        "resolution": 336,
        "device": device,
        "macro_average": macro_avg,
        "per_environment": all_results,
    }
    with open("mac_outputs/epoch_10_evaluation_results.json", "w") as f:
        json.dump(out_data, f, indent=2)

    # Save Markdown Summary
    md = "# Dioptra-DINO: Epoch 10 High-Resolution (336x336) Indoor Benchmark Report\n\n"
    md += f"**Evaluated on Apple Silicon GPU (`{device}`) across {macro_avg['total_environments']} indoor environments ({macro_avg['total_frames']} frames)**.\n\n"
    md += "## 1. Head-to-Head Indoor Comparison\n\n"
    md += "| Model | Resolution | Tokens | AbsRel ↓ | RMSE (m) ↓ | MAE (m) ↓ | $\\delta_1 < 1.25$ ↑ | Scale Ratio | Normal MAE (°) ↓ | Latency (ms) ↓ | FPS ↑ |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    md += f"| **Dioptra-DINO Epoch 10 (Ours)** | **336x336** | **576** | **{macro_avg['abs_rel']:.4f}** | **{macro_avg['rmse']:.3f}m** | **{macro_avg['mae']:.3f}m** | **{macro_avg['delta1']*100:.1f}%** | **{macro_avg['scale_ratio']:.3f}** | **{macro_avg['normal_mae']:.1f}°** | **{macro_avg['latency_ms']:.1f} ms** | **{macro_avg['fps']:.1f} FPS** |\n"
    md += "| *Dioptra-DINO Epoch 5 Baseline* | 224x224 | 256 | 0.4700 | 3.180m | 2.091m | 34.9% | 0.825 | 55.4° | 35.8 ms | 27.9 FPS |\n"
    md += "| *Depth Anything V2 Metric* | 518x518 | 1,369 | 0.7068 | 4.301m | 2.566m | 18.3% | 1.588 | 41.7° | 110.1 ms | 9.4 FPS |\n"
    md += "| *Metric3D ViT-Small* | 616x1064 | 3,344 | 0.2976 | 3.870m | 1.875m | 55.6% | 1.014 | 43.7° | 549.3 ms | 1.8 FPS |\n\n"

    md += "## 2. Per-Environment Detailed Breakdown\n\n"
    md += "| Environment | Category | AbsRel ↓ | RMSE (m) ↓ | MAE (m) ↓ | $\\delta_1 < 1.25$ ↑ | Scale Ratio | Normal MAE (°) ↓ | Latency (ms) |\n"
    md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for split_id, v in all_results.items():
        m = v["metrics"]
        md += f"| {v['name']} | {v['category']} | {m['abs_rel']:.4f} | {m['rmse']:.3f}m | {m['mae']:.3f}m | {m['delta1']*100:.1f}% | {m['scale_ratio']:.3f} | {m['normal_mae']:.1f}° | {m['latency_ms']:.1f}ms |\n"

    with open("mac_outputs/epoch_10_evaluation_summary.md", "w") as f:
        f.write(md)

    print("✓ Successfully saved results to mac_outputs/epoch_10_evaluation_summary.md")


if __name__ == "__main__":
    main()
