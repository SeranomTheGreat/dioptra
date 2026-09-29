#!/usr/bin/env python3
"""
Benchmark True Metric Depth Competitor Foundation Models
=========================================================
Direct head-to-head comparison on identical unseen testing frames:
1. Dioptra-DINO Step Latest (Ours, 25.4M params, Ray-Conditioned Metric Depth)
2. Metric3D ViT-Small (TPAMI 2024 / CVPR 2023, 37.5M params, Camera-Conditioned Metric Foundation Model)
3. Depth Anything V2 Metric Small (CVPR 2024, 24.8M params, True Metric Depth Foundation Model)

All models evaluated strictly on DIRECT METRIC DEPTH IN METRES (No oracle median scaling, no affine fitting).
Runs locally on Apple Silicon Mac using Metal Performance Shaders (MPS).
"""

import os
import sys
import glob
import math
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Register Dioptra-DINO
import dioptra_dino
sys.modules["__main__"].DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINOConfig, DioptraDINO


# ---------------------------------------------------------------------------
# Camera Intrinsics
# ---------------------------------------------------------------------------

K_TARTANAIR_NATIVE = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_SCANNET_NATIVE = np.array([
    [577.87, 0.0, 319.5],
    [0.0, 577.87, 239.5],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


def scale_intrinsics(K: np.ndarray, orig_w: float, orig_h: float, new_w: float, new_h: float) -> np.ndarray:
    K_out = K.copy()
    K_out[0, 0] *= (new_w / orig_w)
    K_out[0, 2] *= (new_w / orig_w)
    K_out[1, 1] *= (new_h / orig_h)
    K_out[1, 2] *= (new_h / orig_h)
    return K_out


# ---------------------------------------------------------------------------
# Metric Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 80.0) -> Dict[str, float]:
    """Compute standard metric depth evaluation metrics directly in metres."""
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & np.isfinite(pred) & (gt > 0)
    if np.sum(mask) < 20:
        return {}

    p = np.clip(pred[mask], min_depth, max_depth)
    g = gt[mask]

    abs_rel = float(np.mean(np.abs(p - g) / g))
    sq_rel = float(np.mean(((p - g) ** 2) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))

    log_diff = np.log(p) - np.log(g)
    rmse_log = float(np.sqrt(np.mean(log_diff ** 2)))
    silog = float(np.sqrt(np.mean(log_diff ** 2) - 0.85 * (np.mean(log_diff) ** 2)))

    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25 ** 2))
    delta3 = float(np.mean(ratio < 1.25 ** 3))

    scale_ratio = float(np.median(p) / (np.median(g) + 1e-6))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "silog": silog,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "scale_ratio": scale_ratio,
        "valid_pixels": int(np.sum(mask)),
    }


# ---------------------------------------------------------------------------
# Model Loaders
# ---------------------------------------------------------------------------

class DioptraDINOEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "mps"):
        self.device = device
        cfg = DioptraDINOConfig()
        self.model = DioptraDINO(cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict(self, pil_img: PILImage.Image, K_native: np.ndarray, orig_w: int, orig_h: int) -> Tuple[np.ndarray, float]:
        img_224 = pil_img.resize((224, 224), PILImage.BILINEAR)
        img_np = np.array(img_224, dtype=np.float32) / 255.0
        img_norm = (img_np - self.mean) / self.std
        img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        K_224 = scale_intrinsics(K_native, orig_w, orig_h, 224, 224)
        K_t = torch.from_numpy(K_224).unsqueeze(0).to(self.device)

        if self.device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_224 = self.model(img_t, K_t, ara_gate=1.0)
            # Interpolate to native resolution for direct evaluation against GT
            pred_full = F.interpolate(pred_224, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return pred_full.squeeze().cpu().numpy(), dt_ms


class Metric3DEvaluator:
    def __init__(self, device: str = "mps"):
        self.device = device
        print("Loading Metric3D ViT-Small via torch.hub...")
        self.model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
        self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.input_size = (616, 1064)
        self.padding = [123.675, 116.28, 103.53]
        self.mean = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1).to(device)

    def predict(self, rgb_origin: np.ndarray, K_native: np.ndarray, orig_w: int, orig_h: int) -> Tuple[np.ndarray, float]:
        fx, fy, cx, cy = K_native[0, 0], K_native[1, 1], K_native[0, 2], K_native[1, 2]
        intrinsic = [fx, fy, cx, cy]

        scale = min(self.input_size[0] / orig_h, self.input_size[1] / orig_w)
        rgb = cv2.resize(rgb_origin, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_LINEAR)
        intrinsic_scaled = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

        pad_h = self.input_size[0] - rgb.shape[0]
        pad_w = self.input_size[1] - rgb.shape[1]
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb_padded = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)

        tensor_rgb = torch.from_numpy(rgb_padded.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        norm_rgb = (tensor_rgb - self.mean) / self.std

        if self.device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_depth, _, _ = self.model.inference({"input": norm_rgb})

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_h_half : pred_depth.shape[0] - (pad_h - pad_h_half), pad_w_half : pred_depth.shape[1] - (pad_w - pad_w_half)]
        pred_depth = F.interpolate(pred_depth[None, None, :, :], (orig_h, orig_w), mode="bilinear", align_corners=False).squeeze()

        # De-canonical focal transform to real metres
        canonical_to_real_scale = intrinsic_scaled[0] / 1000.0
        pred_metric = pred_depth * canonical_to_real_scale
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return pred_metric.cpu().numpy(), dt_ms


class DepthAnythingV2MetricEvaluator:
    def __init__(self, mode: str = "indoor", device: str = "mps"):
        self.device = device
        self.mode = mode
        if mode == "indoor":
            self.model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        else:
            self.model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

        print(f"Loading Depth Anything V2 Metric ({mode}) from {self.model_id}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(device)
        self.model.eval()
        self.params = sum(p.numel() for p in self.model.parameters())

    def predict(self, pil_img: PILImage.Image, orig_w: int, orig_h: int) -> Tuple[np.ndarray, float]:
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        if self.device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
            pred_t = F.interpolate(outputs.predicted_depth.unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return pred_t.squeeze().cpu().numpy(), dt_ms


# ---------------------------------------------------------------------------
# Visual Panel Generation
# ---------------------------------------------------------------------------

def save_comparison_panel(
    rgb_np: np.ndarray,
    gt_depth: np.ndarray,
    pred_dioptra: np.ndarray,
    pred_metric3d: np.ndarray,
    pred_dav2: np.ndarray,
    out_path: str,
    domain_title: str,
    max_d: float = 12.0,
):
    """Save 6-panel visual comparison showing RGB, GT, and predictions from all true metric models."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), dpi=150)

    # 1. RGB
    axes[0].imshow(rgb_np)
    axes[0].set_title(f"RGB ({domain_title})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    valid_gt = (gt_depth > 0.1) & (gt_depth < 80.0) & np.isfinite(gt_depth)
    vmax = min(max_d, float(np.percentile(gt_depth[valid_gt], 98))) if np.any(valid_gt) else max_d

    # 2. GT Depth
    im_gt = axes[1].imshow(gt_depth, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[1].set_title(f"Ground Truth ({vmax:.1f}m)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Dioptra-DINO Step Latest
    im_dio = axes[2].imshow(pred_dioptra, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[2].set_title("Dioptra-DINO (Ours, Metric)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im_dio, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Metric3D ViT-Small
    im_m3d = axes[3].imshow(pred_metric3d, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[3].set_title("Metric3D ViT-Small (TPAMI)", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im_m3d, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. Depth Anything V2 Metric
    im_da = axes[4].imshow(pred_dav2, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[4].set_title("Depth Anything V2 (Metric)", fontsize=11, fontweight="bold")
    axes[4].axis("off")
    plt.colorbar(im_da, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    checkpoint_path: str,
    device: str = "mps",
    max_samples: int = 50,
    output_dir: str = "mac_outputs",
):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations_true_metric")
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 85)
    print("TRUE METRIC DEPTH FOUNDATION MODEL BENCHMARK ON UNSEEN DATA (APPLE SILICON MPS)")
    print("=" * 85)

    # Initialize all models
    print("\n[1/4] Initializing Dioptra-DINO Step Latest...")
    eval_dioptra = DioptraDINOEvaluator(checkpoint_path, device=device)

    print("\n[2/4] Initializing Metric3D ViT-Small...")
    eval_metric3d = Metric3DEvaluator(device=device)

    print("\n[3/4] Initializing Depth Anything V2 Metric Small (Indoor)...")
    eval_dav2_in = DepthAnythingV2MetricEvaluator(mode="indoor", device=device)

    print("\n[4/4] Initializing Depth Anything V2 Metric Small (Outdoor)...")
    eval_dav2_out = DepthAnythingV2MetricEvaluator(mode="outdoor", device=device)

    # Define benchmark domains
    domains = [
        {
            "id": "carwelding",
            "name": "TartanAir CarWelding (Industrial Workcell)",
            "pattern": "test_samples/unseen_carwelding_*/image_left/*.png",
            "depth_replace": ("image_left", "depth_left", ".png", "_depth.npy"),
            "K": K_TARTANAIR_NATIVE,
            "orig_w": 640, "orig_h": 480,
            "is_16bit": False,
            "min_d": 0.1, "max_d": 60.0,
            "da_mode": "outdoor",
        },
        {
            "id": "office",
            "name": "TartanAir Office (Indoor Navigation)",
            "pattern": "test_samples/unseen_office_*/image_left/*.png",
            "depth_replace": ("image_left", "depth_left", ".png", "_depth.npy"),
            "K": K_TARTANAIR_NATIVE,
            "orig_w": 640, "orig_h": 480,
            "is_16bit": False,
            "min_d": 0.1, "max_d": 20.0,
            "da_mode": "indoor",
        },
        {
            "id": "japanesealley",
            "name": "TartanAir JapaneseAlley (Outdoor Street)",
            "pattern": "test_samples/unseen_japanesealley_*/image_left/*.png",
            "depth_replace": ("image_left", "depth_left", ".png", "_depth.npy"),
            "K": K_TARTANAIR_NATIVE,
            "orig_w": 640, "orig_h": 480,
            "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
            "da_mode": "outdoor",
        },
        {
            "id": "gascola",
            "name": "TartanAir Gascola (Robotics Quarry)",
            "pattern": "test_samples/unseen_gascola_*/image_left/*.png",
            "depth_replace": ("image_left", "depth_left", ".png", "_depth.npy"),
            "K": K_TARTANAIR_NATIVE,
            "orig_w": 640, "orig_h": 480,
            "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
            "da_mode": "outdoor",
        },
        {
            "id": "scannet",
            "name": "ScanNet (Real-World iPad Structure Sensor)",
            "pattern": "data/scannet_tiny/**/color/*.jpg",
            "depth_replace": ("/color/", "/depth/", ".jpg", ".png"),
            "K": K_SCANNET_NATIVE,
            "orig_w": 640, "orig_h": 480,
            "is_16bit": True,
            "min_d": 0.4, "max_d": 10.0,
            "da_mode": "indoor",
        },
    ]

    all_results = {}

    for dom in domains:
        img_files = sorted(glob.glob(dom["pattern"], recursive=True))
        pairs = []
        for p in img_files:
            rep = dom["depth_replace"]
            dp = p.replace(rep[0], rep[1]).replace(rep[2], rep[3])
            if os.path.exists(dp):
                pairs.append((p, dp))

        pairs = pairs[:max_samples]
        if not pairs:
            continue

        print(f"\n{'='*80}")
        print(f"EVALUATING DOMAIN: {dom['name'].upper()} ({len(pairs)} pairs)")
        print(f"{'='*80}")

        eval_dav2 = eval_dav2_in if dom["da_mode"] == "indoor" else eval_dav2_out

        m_dio_list, m_m3d_list, m_da_list = [], [], []
        lat_dio_list, lat_m3d_list, lat_da_list = [], [], []

        vis_saved = 0

        for idx, (img_path, depth_path) in enumerate(pairs):
            # Load raw RGB and GT depth
            pil_img = PILImage.open(img_path).convert("RGB")
            cv_img = cv2.imread(img_path)[:, :, ::-1]

            if dom["is_16bit"]:
                gt_depth = np.array(PILImage.open(depth_path), dtype=np.float32) / 1000.0
            else:
                gt_depth = np.load(depth_path).astype(np.float32)

            # 1. Dioptra-DINO Prediction
            pred_dio, t_dio = eval_dioptra.predict(pil_img, dom["K"], dom["orig_w"], dom["orig_h"])
            lat_dio_list.append(t_dio)

            # 2. Metric3D Prediction
            pred_m3d, t_m3d = eval_metric3d.predict(cv_img, dom["K"], dom["orig_w"], dom["orig_h"])
            lat_m3d_list.append(t_m3d)

            # 3. Depth Anything V2 Metric Prediction
            pred_da, t_da = eval_dav2.predict(pil_img, dom["orig_w"], dom["orig_h"])
            lat_da_list.append(t_da)

            # Compute individual metrics
            m_dio = compute_metrics(pred_dio, gt_depth, min_depth=dom["min_d"], max_depth=dom["max_d"])
            m_m3d = compute_metrics(pred_m3d, gt_depth, min_depth=dom["min_d"], max_depth=dom["max_d"])
            m_da = compute_metrics(pred_da, gt_depth, min_depth=dom["min_d"], max_depth=dom["max_d"])

            if m_dio and m_m3d and m_da:
                m_dio_list.append(m_dio)
                m_m3d_list.append(m_m3d)
                m_da_list.append(m_da)

            # Periodic visual comparison
            if vis_saved < 3 and (idx % max(1, len(pairs) // 3) == 0):
                vis_file = os.path.join(vis_dir, f"cmp_{dom['id']}_frame_{idx:03d}.png")
                save_comparison_panel(
                    rgb_np=np.array(pil_img),
                    gt_depth=gt_depth,
                    pred_dioptra=pred_dio,
                    pred_metric3d=pred_m3d,
                    pred_dav2=pred_da,
                    out_path=vis_file,
                    domain_title=f"{dom['id'].upper()} Frame {idx}",
                    max_d=min(12.0, dom["max_d"]),
                )
                vis_saved += 1

            if (idx + 1) % 10 == 0 or (idx + 1) == len(pairs):
                print(f"  [{idx+1:02d}/{len(pairs):02d}] AbsRel -> Dioptra: {m_dio.get('abs_rel', 0):.4f} | Metric3D: {m_m3d.get('abs_rel', 0):.4f} | DA-V2: {m_da.get('abs_rel', 0):.4f}")

        # Aggregate domain metrics
        def _aggregate(m_list, lat_list):
            if not m_list:
                return {}
            keys = ["abs_rel", "sq_rel", "rmse", "rmse_log", "silog", "delta1", "delta2", "delta3", "scale_ratio"]
            agg = {k: float(np.mean([m[k] for m in m_list])) for k in keys}
            agg["latency_ms"] = float(np.mean(lat_list[1:] if len(lat_list) > 1 else lat_list))
            agg["fps"] = float(1000.0 / agg["latency_ms"]) if agg["latency_ms"] > 0 else 0.0
            return agg

        agg_dio = _aggregate(m_dio_list, lat_dio_list)
        agg_m3d = _aggregate(m_m3d_list, lat_m3d_list)
        agg_da = _aggregate(m_da_list, lat_da_list)

        all_results[dom["id"]] = {
            "name": dom["name"],
            "samples": len(m_dio_list),
            "dioptra_dino": agg_dio,
            "metric3d": agg_m3d,
            "depth_anything_v2_metric": agg_da,
        }

        print(f"\n--- Summary for {dom['name']} ---")
        print(f"  Dioptra-DINO    : AbsRel={agg_dio['abs_rel']:.4f}, RMSE={agg_dio['rmse']:.3f}m, d1={agg_dio['delta1']*100:.1f}%, Scale={agg_dio['scale_ratio']:.3f}, FPS={agg_dio['fps']:.1f}")
        print(f"  Metric3D        : AbsRel={agg_m3d['abs_rel']:.4f}, RMSE={agg_m3d['rmse']:.3f}m, d1={agg_m3d['delta1']*100:.1f}%, Scale={agg_m3d['scale_ratio']:.3f}, FPS={agg_m3d['fps']:.1f}")
        print(f"  DA-V2 Metric    : AbsRel={agg_da['abs_rel']:.4f}, RMSE={agg_da['rmse']:.3f}m, d1={agg_da['delta1']*100:.1f}%, Scale={agg_da['scale_ratio']:.3f}, FPS={agg_da['fps']:.1f}")

    # Save summary JSON
    json_path = os.path.join(output_dir, "true_metric_competitors_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved JSON benchmark summary: {json_path}")

    # Generate Markdown Table
    md_path = os.path.join(output_dir, "true_metric_competitors_summary.md")
    with open(md_path, "w") as f:
        f.write("# True Metric Depth Foundation Model Comparison\n\n")
        f.write("Direct zero-shot metric evaluation without test-time oracle scaling or affine fitting on Apple Silicon MPS.\n\n")

        for dom_id, res in all_results.items():
            f.write(f"### {res['name']} (N={res['samples']})\n\n")
            f.write("| Model | Parameters | Direct AbsRel (↓) | RMSE (m ↓) | SiLog (↓) | δ < 1.25 (↑) | δ < 1.25² (↑) | Scale Ratio | MPS Latency | FPS |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

            d = res["dioptra_dino"]
            m = res["metric3d"]
            a = res["depth_anything_v2_metric"]

            f.write(f"| **Dioptra-DINO Step Latest (Ours)** | **25.4M** | **{d['abs_rel']:.4f}** | **{d['rmse']:.3f}m** | **{d['silog']:.4f}** | **{d['delta1']*100:.1f}%** | **{d['delta2']*100:.1f}%** | **{d['scale_ratio']:.3f}** | **{d['latency_ms']:.1f} ms** | **{d['fps']:.1f}** |\n")
            f.write(f"| Metric3D ViT-Small (TPAMI 2024) | 37.5M | {m['abs_rel']:.4f} | {m['rmse']:.3f}m | {m['silog']:.4f} | {m['delta1']*100:.1f}% | {m['delta2']*100:.1f}% | {m['scale_ratio']:.3f} | {m['latency_ms']:.1f} ms | {m['fps']:.1f} |\n")
            f.write(f"| Depth Anything V2 Metric (CVPR 2024) | 24.8M | {a['abs_rel']:.4f} | {a['rmse']:.3f}m | {a['silog']:.4f} | {a['delta1']*100:.1f}% | {a['delta2']*100:.1f}% | {a['scale_ratio']:.3f} | {a['latency_ms']:.1f} ms | {a['fps']:.1f} |\n\n")

    print(f"✓ Saved Markdown benchmark report: {md_path}")
    print("\n" + "=" * 85)
    print("ALL TRUE METRIC COMPETITOR BENCHMARKS COMPLETED!")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark True Metric Depth Competitors")
    parser.add_argument("--checkpoint", type=str, default="staging_volsiai/checkpoint_step_latest.pt")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="mac_outputs")
    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )
