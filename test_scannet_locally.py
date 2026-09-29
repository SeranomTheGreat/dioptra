#!/usr/bin/env python3
"""
Dioptra-DINO: Local Evaluation on Unseen ScanNet Dataset
======================================================
Tests trained Dioptra-DINO checkpoints on unseen ScanNet RGB-D frames.

ScanNet Specifications:
  - Sensor: Occipital Structure Sensor (iPad 640x480)
  - Depth format: 16-bit PNG in millimeters (depth_raw / 1000.0 = metric meters)
  - Valid range: 0.4m to 10.0m
  - Canonical Intrinsics:
      fx = 577.87, fy = 577.87, cx = 319.5, cy = 239.5
"""

import os
import sys
import glob
import math
import time
import json
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Dioptra-DINO architecture
from dioptra_dino import DioptraDINOConfig, DioptraDINO
from evaluate_unseen import compute_depth_metrics, depth_to_surface_normals, save_visual_panel

# Canonical ScanNet Camera Intrinsics
K_SCANNET = np.array([
    [577.87, 0.0, 319.5],
    [0.0, 577.87, 239.5],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


def load_model(checkpoint_path: str, device: str = "cpu") -> DioptraDINO:
    """Load Dioptra-DINO model with trained weights."""
    print(f"[ScanNet Local Eval] Loading checkpoint: {checkpoint_path}")
    cfg = DioptraDINOConfig()
    model = DioptraDINO(cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    load_res = model.load_state_dict(cleaned, strict=False)
    print(f"[ScanNet Local Eval] Weights loaded! Missing keys: {len(load_res.missing_keys)}, Unexpected keys: {len(load_res.unexpected_keys)}")
    model = model.to(device)
    model.eval()
    return model


def evaluate_scannet(
    model: DioptraDINO,
    scannet_dir: str,
    output_dir: str = "outputs_scannet_eval",
    resolutions: Tuple[int, ...] = (224, 392),
    device: str = "cpu",
    checkpoint_name: str = "Epoch 5 Step 37,500",
) -> Dict[str, Any]:
    """Run evaluation across ScanNet unseen frames."""
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    color_files = sorted(glob.glob(os.path.join(scannet_dir, "**", "color", "*.jpg"), recursive=True))
    if not color_files:
        raise FileNotFoundError(f"No ScanNet color images found in {scannet_dir}!")

    print(f"\nDiscovered {len(color_files)} ScanNet evaluation frames in {scannet_dir}")

    all_results = {}

    for res in resolutions:
        print("\n" + "=" * 80)
        print(f"EVALUATING SCANNET AT {res}x{res} RESOLUTION (Grid: {res//14}x{res//14} = {(res//14)**2} ViT Tokens)")
        print("=" * 80)

        # Scale intrinsics to evaluation resolution
        scale_x = res / 640.0
        scale_y = res / 480.0
        K_res = K_SCANNET.copy()
        K_res[0, :] *= scale_x
        K_res[1, :] *= scale_y

        K_tensor = torch.from_numpy(K_res).unsqueeze(0).to(device)

        frame_metrics = []

        for idx, color_path in enumerate(color_files):
            # Locate corresponding depth file
            depth_path = color_path.replace("/color/", "/depth/").replace(".jpg", ".png")
            if not os.path.isfile(depth_path):
                print(f"Warning: Depth file not found for {color_path}, skipping.")
                continue

            frame_id = os.path.splitext(os.path.basename(color_path))[0]

            # 1. Load and preprocess RGB image
            pil_img = PILImage.open(color_path).convert("RGB")
            pil_img_res = pil_img.resize((res, res), PILImage.BILINEAR)
            img_np = np.array(pil_img_res, dtype=np.float32) / 255.0

            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_np - mean) / std
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

            # 2. Load and preprocess Ground Truth Depth map
            pil_depth = PILImage.open(depth_path)
            pil_depth_res = pil_depth.resize((res, res), PILImage.NEAREST)
            depth_raw = np.array(pil_depth_res, dtype=np.float32)
            gt_depth = depth_raw / 1000.0  # 16-bit mm -> meters

            # 3. Model Inference
            t0 = time.time()
            with torch.no_grad():
                pred_depth_tensor = model(img_tensor, K_tensor)
            infer_time = (time.time() - t0) * 1000.0

            pred_depth = pred_depth_tensor.squeeze().cpu().numpy()

            # 4. Compute Metrics
            m = compute_depth_metrics(
                pred=pred_depth,
                gt=gt_depth,
                min_depth=0.4,
                max_depth=10.0,
                crop_box=None,
                K=K_res,
            )

            if m:
                m["infer_time_ms"] = infer_time
                m["frame_id"] = frame_id
                frame_metrics.append(m)

                # 5. Save 5-Panel Qualitative Comparison Panel
                vis_path = os.path.join(vis_dir, f"scannet_{res}p_frame_{frame_id}.png")
                save_visual_panel(
                    rgb_np=np.array(pil_img_res),
                    gt_depth=gt_depth,
                    pred_depth=pred_depth,
                    out_path=vis_path,
                    domain_name=f"ScanNet frame {frame_id}",
                    K=K_res,
                    max_display_depth=6.0,
                )

                print(f"  [Frame {frame_id:4s}] AbsRel: {m['abs_rel']:.4f} | RMSE: {m['rmse']:.3f}m | delta<1.25: {m['delta1']*100:.1f}% | NormMAE: {m['normal_mae']:.1f}° | Time: {infer_time:.1f}ms")

        # Aggregate metrics across all frames
        if frame_metrics:
            agg = {}
            for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "silog", "delta1", "delta2", "delta3", "edge_rmse", "normal_mae", "normal_acc11", "normal_acc22", "infer_time_ms"]:
                vals = [fm[k] for fm in frame_metrics if k in fm and not math.isnan(fm[k])]
                agg[k] = float(np.mean(vals)) if vals else float("nan")

            key_name = f"scannet_unseen_{res}x{res}"
            all_results[key_name] = {
                "checkpoint": checkpoint_name,
                "resolution": f"{res}x{res}",
                "num_frames": len(frame_metrics),
                "metrics": agg,
                "per_frame": frame_metrics,
            }

            print(f"\n--- Aggregate Results for ScanNet ({res}x{res}, N={len(frame_metrics)}) ---")
            print(f"  AbsRel           : {agg['abs_rel']:.4f}")
            print(f"  SqRel            : {agg['sq_rel']:.4f}")
            print(f"  RMSE (meters)    : {agg['rmse']:.4f} m")
            print(f"  RMSE log         : {agg['rmse_log']:.4f}")
            print(f"  SiLog (alpha=0.85): {agg['silog']:.4f}")
            print(f"  delta < 1.25     : {agg['delta1']*100:.2f}%")
            print(f"  delta < 1.25^2   : {agg['delta2']*100:.2f}%")
            print(f"  delta < 1.25^3   : {agg['delta3']*100:.2f}%")
            print(f"  Edge RMSE        : {agg['edge_rmse']:.4f}")
            print(f"  Normal MAE       : {agg['normal_mae']:.2f}° (Planarity error)")
            print(f"  Normal < 11.25°  : {agg['normal_acc11']*100:.2f}%")
            print(f"  Mean Latency     : {agg['infer_time_ms']:.1f} ms / frame")

    # Save summary JSON
    summary_json = os.path.join(output_dir, "scannet_metrics_summary.json")
    with open(summary_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved ScanNet evaluation summary JSON: {summary_json}")

    return all_results


def main():
    scannet_dir = "data/scannet_tiny/scene00"
    ckpt_path = "staging_harryson/checkpoint_step_latest.pt"

    if not os.path.exists(ckpt_path):
        cands = sorted(glob.glob("staging_*/*.pt")) + sorted(glob.glob("outputs*/*.pt"))
        if cands:
            ckpt_path = cands[0]
        else:
            raise FileNotFoundError("No checkpoint found!")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device} for local evaluation.")

    model = load_model(ckpt_path, device=device)
    evaluate_scannet(
        model=model,
        scannet_dir=scannet_dir,
        output_dir="outputs_scannet_eval",
        resolutions=(224, 392),
        device=device,
        checkpoint_name="Dioptra-DINO Epoch 5 Step 37,500",
    )


if __name__ == "__main__":
    main()
