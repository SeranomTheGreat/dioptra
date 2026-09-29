#!/usr/bin/env python3
"""
Comprehensive Indoor Metric Depth Benchmark & Foundation Model Evaluation
==========================================================================
Direct head-to-head comparison across 18 purely indoor environments:
1. Dioptra-DINO Step Latest (Ours, 25.4M params, Ray-Conditioned Metric Depth)
2. Metric3D ViT-Small (TPAMI 2024 / CVPR 2023, 37.5M params, Canonical Camera Foundation Model)
3. Depth Anything V2 Metric Indoor Small (CVPR 2024, 24.8M params, SOTA Monocular Metric Foundation Model)

Evaluates on:
- TartanAir v1 Indoors: carwelding (P001, P002), office (P001, P002), office2 (P000), hospital (P001), abandonedfactory (P005, 200-frames), abandonedfactory_night (P001)
- TartanAir v2 Indoors: AmericanDiner, ArchVizTinyHouseDay, ArchVizTinyHouseNight, Prison, RetroOffice, House, Supermarket
- Real-World Indoors: NYU-Depth V2 (Official test set), ScanNet (scene00 handheld RGB-D)

Direct metric depth in metres. Zero affine fitting or test-time alignment.
Runs on Apple Silicon MPS backend.
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

K_TARTANAIR_V1 = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_TARTANAIR_V2 = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 320.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_NYU_V2 = np.array([
    [518.8579, 0.0, 325.5824],
    [0.0, 518.8579, 253.7362],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_SCANNET = np.array([
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


def depth_to_surface_normals(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth > 0.05) & (depth < 80.0) & np.isfinite(depth)
    d = np.where(valid, depth, 0.0)

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

    dX_du = cv2.Sobel(X, cv2.CV_64F, 1, 0, ksize=3)
    dY_du = cv2.Sobel(Y, cv2.CV_64F, 1, 0, ksize=3)
    dZ_du = cv2.Sobel(Z, cv2.CV_64F, 1, 0, ksize=3)

    dX_dv = cv2.Sobel(X, cv2.CV_64F, 0, 1, ksize=3)
    dY_dv = cv2.Sobel(Y, cv2.CV_64F, 0, 1, ksize=3)
    dZ_dv = cv2.Sobel(Z, cv2.CV_64F, 0, 1, ksize=3)

    Tu = np.stack([dX_du, dY_du, dZ_du], axis=-1)
    Tv = np.stack([dX_dv, dY_dv, dZ_dv], axis=-1)

    normals = np.cross(Tu, Tv)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-6, 1.0, norm)
    normals = normals / norm

    flip_mask = normals[:, :, 2] > 0
    normals[flip_mask] = -normals[flip_mask]
    invalid_mask = np.isnan(normals).any(axis=-1) | (~valid)
    normals[invalid_mask] = [0.0, 0.0, -1.0]
    return normals.astype(np.float32)


# ---------------------------------------------------------------------------
# Metric Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, gt: np.ndarray, K: Optional[np.ndarray] = None, min_depth: float = 0.1, max_depth: float = 80.0) -> Dict[str, float]:
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & np.isfinite(pred) & (gt > 0)
    if np.sum(mask) < 20:
        return {}

    p = np.clip(pred[mask], min_depth, max_depth)
    g = gt[mask]

    abs_rel = float(np.mean(np.abs(p - g) / g))
    sq_rel = float(np.mean(((p - g) ** 2) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mae = float(np.mean(np.abs(p - g)))

    log_diff = np.log(p) - np.log(g)
    rmse_log = float(np.sqrt(np.mean(log_diff ** 2)))
    silog = float(np.sqrt(np.mean(log_diff ** 2) - 0.85 * (np.mean(log_diff) ** 2)))

    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25 ** 2))
    delta3 = float(np.mean(ratio < 1.25 ** 3))

    scale_ratio = float(np.median(p) / (np.median(g) + 1e-6))

    # Surface Normal Angular Error
    normal_mae = float("nan")
    normal_acc11 = float("nan")
    if K is not None:
        try:
            norm_gt = depth_to_surface_normals(gt, K)
            norm_pred = depth_to_surface_normals(pred, K)
            dot = np.sum(norm_gt * norm_pred, axis=-1)
            dot = np.clip(dot, -1.0, 1.0)
            angles_deg = np.arccos(dot) * (180.0 / math.pi)
            valid_angles = angles_deg[mask]
            valid_angles = valid_angles[np.isfinite(valid_angles)]
            if len(valid_angles) > 10:
                normal_mae = float(np.mean(valid_angles))
                normal_acc11 = float(np.mean(valid_angles < 11.25))
        except Exception:
            pass

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "mae": mae,
        "rmse_log": rmse_log,
        "silog": silog,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "scale_ratio": scale_ratio,
        "normal_mae": normal_mae,
        "normal_acc11": normal_acc11,
        "valid_pixels": int(np.sum(mask)),
    }


# ---------------------------------------------------------------------------
# Evaluator Wrappers
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
            pred_full = F.interpolate(pred_224, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return pred_full.squeeze().cpu().numpy(), dt_ms


class Metric3DEvaluator:
    def __init__(self, device: str = "mps"):
        self.device = device
        print("[Metric3D] Loading Metric3D ViT-Small via torch.hub...")
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
        self.model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        print(f"[Depth Anything V2] Loading Metric ({mode}) from {self.model_id}...")
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

def save_indoor_comparison_panel(
    rgb_np: np.ndarray,
    gt_depth: np.ndarray,
    pred_dioptra: np.ndarray,
    pred_metric3d: np.ndarray,
    pred_dav2: np.ndarray,
    out_path: str,
    domain_title: str,
    max_d: float = 10.0,
):
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

    # 3. Dioptra-DINO
    im_dio = axes[2].imshow(pred_dioptra, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[2].set_title("Dioptra-DINO (Ours, Metric)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im_dio, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Metric3D ViT-Small
    im_m3d = axes[3].imshow(pred_metric3d, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[3].set_title("Metric3D ViT-Small (TPAMI)", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im_m3d, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. Depth Anything V2 Metric Indoor
    im_dav2 = axes[4].imshow(pred_dav2, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[4].set_title("Depth Anything V2 Metric", fontsize=11, fontweight="bold")
    axes[4].axis("off")
    plt.colorbar(im_dav2, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="staging_harryson/checkpoint_step_latest.pt")
    parser.add_argument("--out-json", type=str, default="mac_outputs/indoor_benchmark_results.json")
    parser.add_argument("--out-md", type=str, default="mac_outputs/indoor_benchmark_summary.md")
    parser.add_argument("--vis-dir", type=str, default="mac_outputs/visualizations_indoor")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("COMPREHENSIVE ALL-INDOOR METRIC DEPTH BENCHMARK")
    print(f"Device: {args.device} | Checkpoint: {args.checkpoint}")
    print(f"{'='*80}\n")

    # Initialize Models
    dioptra = DioptraDINOEvaluator(args.checkpoint, device=args.device)
    metric3d = Metric3DEvaluator(device=args.device)
    dav2 = DepthAnythingV2MetricEvaluator(mode="indoor", device=args.device)

    print("\n✓ Models loaded successfully:")
    print(f"  Dioptra-DINO: {dioptra.params/1e6:.2f}M params")
    print(f"  Metric3D ViT-Small: {metric3d.params/1e6:.2f}M params")
    print(f"  Depth Anything V2 Metric Small: {dav2.params/1e6:.2f}M params\n")

    # Define all indoor benchmark environments
    domains = [
        # TartanAir v1 Indoor
        {
            "id": "carwelding_p001",
            "name": "CarWelding / P001",
            "category": "Industrial Robotics",
            "img_pattern": "test_samples/unseen_carwelding_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_carwelding_p001/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "max_eval_frames": 25,
        },
        {
            "id": "carwelding_p002",
            "name": "CarWelding / P002",
            "category": "Industrial Robotics",
            "img_pattern": "test_samples/unseen_carwelding_p002/image_left/*.png",
            "depth_dir": "test_samples/unseen_carwelding_p002/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "max_eval_frames": 30,
        },
        {
            "id": "office_p001",
            "name": "Office / P001",
            "category": "Commercial Office",
            "img_pattern": "test_samples/unseen_office_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_office_p001/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "max_eval_frames": 30,
        },
        {
            "id": "office_p002",
            "name": "Office / P002",
            "category": "Commercial Office",
            "img_pattern": "test_samples/unseen_office_p002/image_left/*.png",
            "depth_dir": "test_samples/unseen_office_p002/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "max_eval_frames": 30,
        },
        {
            "id": "office2_p000",
            "name": "Office2 / P000",
            "category": "Executive Workspace",
            "img_pattern": "test_samples/unseen_office2_p000/image_left/*.png",
            "depth_dir": "test_samples/unseen_office2_p000/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "max_eval_frames": 50,
        },
        {
            "id": "hospital_p001",
            "name": "Hospital / P001",
            "category": "Medical Facility",
            "img_pattern": "test_samples/unseen_hospital_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_hospital_p001/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "max_eval_frames": 50,
        },
        {
            "id": "abandonedfactory_p005",
            "name": "AbandonedFactory / P005",
            "category": "Industrial Warehouse",
            "img_pattern": "test_samples/unseen_abandonedfactory_p005/image_left/*.png",
            "depth_dir": "test_samples/unseen_abandonedfactory_p005/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "max_eval_frames": 50,
        },
        {
            "id": "abandonedfactory_night_p001",
            "name": "Factory Night / P001",
            "category": "Extreme Low-Light",
            "img_pattern": "test_samples/unseen_abandonedfactory_night_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_abandonedfactory_night_p001/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "max_eval_frames": 50,
        },
        {
            "id": "abandonedfactory_200",
            "name": "AbandonedFactory / 200-Frames",
            "category": "Warehouse Long-Run",
            "img_pattern": "test_samples/unseen_200_abandonedfactory/image_left/*.png",
            "depth_dir": "test_samples/unseen_200_abandonedfactory/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "max_eval_frames": 200,
        },
        # TartanAir v2 Indoor
        {
            "id": "tartanair2_americandiner",
            "name": "American Diner / P000",
            "category": "Restaurant Interior",
            "img_pattern": "test_samples/indoor_suite/tartanair2_americandiner/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_americandiner/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V2,
            "max_eval_frames": 50,
        },
        {
            "id": "tartanair2_archviztinyhouseday",
            "name": "Tiny House Day / P000",
            "category": "Residential Architectural",
            "img_pattern": "test_samples/indoor_suite/tartanair2_archviztinyhouseday/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_archviztinyhouseday/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V2,
            "max_eval_frames": 50,
        },
        {
            "id": "tartanair2_archviztinyhousenight",
            "name": "Tiny House Night / P000",
            "category": "Residential Night",
            "img_pattern": "test_samples/indoor_suite/tartanair2_archviztinyhousenight/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_archviztinyhousenight/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V2,
            "max_eval_frames": 50,
        },
        {
            "id": "tartanair2_prison",
            "name": "Prison / P000",
            "category": "Institutional Cells",
            "img_pattern": "test_samples/indoor_suite/tartanair2_prison/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_prison/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V2,
            "max_eval_frames": 50,
        },
        {
            "id": "tartanair2_retrooffice",
            "name": "Retro Office / P000",
            "category": "Retro Workspace",
            "img_pattern": "test_samples/indoor_suite/tartanair2_retrooffice/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_retrooffice/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V2,
            "max_eval_frames": 50,
        },
        {
            "id": "tartanair2_house",
            "name": "Suburban House / P000",
            "category": "Residential Multi-Room",
            "img_pattern": "test_samples/indoor_suite/tartanair2_house/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_house/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V2,
            "max_eval_frames": 50,
        },
        {
            "id": "tartanair2_supermarket",
            "name": "Supermarket / P000",
            "category": "Retail Grocery",
            "img_pattern": "test_samples/indoor_suite/tartanair2_supermarket/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_supermarket/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V2,
            "max_eval_frames": 50,
        },
        # Real-World Indoor Datasets
        {
            "id": "nyu_depth_v2",
            "name": "NYU-Depth V2 (Official Test)",
            "category": "Real Kinect RGB-D",
            "img_pattern": "test_samples/indoor_suite/nyu_depth_v2/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/nyu_depth_v2/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_NYU_V2,
            "max_eval_frames": 50,
        },
        {
            "id": "scannet_scene00",
            "name": "ScanNet (scene00)",
            "category": "Real Handheld RGB-D",
            "img_pattern": "data/scannet_tiny/scene00/color/*.jpg",
            "depth_dir": "data/scannet_tiny/scene00/depth",
            "depth_suffix": ".png",
            "is_scannet_png": True,
            "K": K_SCANNET,
            "max_eval_frames": 10,
        },
    ]

    all_results = {}
    models = ["dioptra", "metric3d", "depth_anything_v2"]
    model_labels = {
        "dioptra": "Dioptra-DINO (Ours)",
        "metric3d": "Metric3D ViT-S",
        "depth_anything_v2": "Depth Anything V2 Metric",
    }

    total_tested_frames = 0
    total_eval_trials = 0

    for dom in domains:
        dom_id = dom["id"]
        dom_name = dom["name"]
        K_domain = dom["K"]
        is_scannet = dom.get("is_scannet_png", False)

        img_files = sorted(glob.glob(dom["img_pattern"]))
        if not img_files:
            print(f"Skipping {dom_name}: no images found at {dom['img_pattern']}")
            continue

        if len(img_files) > dom["max_eval_frames"]:
            step = max(1, len(img_files) // dom["max_eval_frames"])
            img_files = img_files[::step][:dom["max_eval_frames"]]

        print(f"\n{'-'*75}")
        print(f"Evaluating {dom_name} ({dom['category']}) -> {len(img_files)} frames")
        print(f"{'-'*75}")

        dom_metrics = {m: [] for m in models}
        dom_latencies = {m: [] for m in models}

        sample_saved = False

        for idx, img_path in enumerate(img_files):
            stem = os.path.basename(img_path).split(".")[0]

            # Resolve depth path
            if is_scannet:
                depth_path = os.path.join(dom["depth_dir"], f"{stem}.png")
            else:
                depth_path = os.path.join(dom["depth_dir"], f"{stem}_depth.npy")

            if not os.path.exists(depth_path):
                continue

            # Load RGB
            pil_img = PILImage.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size
            rgb_np = np.array(pil_img)

            # Load GT
            if is_scannet:
                gt_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                gt = gt_raw.astype(np.float32) / 1000.0
            else:
                gt = np.load(depth_path).astype(np.float32)

            # 1. Dioptra-DINO
            pred_dio, dt_dio = dioptra.predict(pil_img, K_domain, orig_w, orig_h)
            m_dio = compute_metrics(pred_dio, gt, K=K_domain)
            if m_dio:
                dom_metrics["dioptra"].append(m_dio)
                dom_latencies["dioptra"].append(dt_dio)

            # 2. Metric3D
            pred_m3d, dt_m3d = metric3d.predict(rgb_np, K_domain, orig_w, orig_h)
            m_m3d = compute_metrics(pred_m3d, gt, K=K_domain)
            if m_m3d:
                dom_metrics["metric3d"].append(m_m3d)
                dom_latencies["metric3d"].append(dt_m3d)

            # 3. Depth Anything V2
            pred_dav2, dt_dav2 = dav2.predict(pil_img, orig_w, orig_h)
            m_dav2 = compute_metrics(pred_dav2, gt, K=K_domain)
            if m_dav2:
                dom_metrics["depth_anything_v2"].append(m_dav2)
                dom_latencies["depth_anything_v2"].append(dt_dav2)

            total_tested_frames += 1
            total_eval_trials += 3

            # Save visual comparison for the first frame of each split
            if not sample_saved and m_dio and m_m3d and m_dav2:
                vis_out = os.path.join(args.vis_dir, f"indoor_{dom_id}_comparison.png")
                save_indoor_comparison_panel(
                    rgb_np, gt, pred_dio, pred_m3d, pred_dav2,
                    vis_out, f"{dom_name} ({dom['category']})"
                )
                paper_vis_out = os.path.join("paper/figures", f"indoor_{dom_id}_comparison.png")
                save_indoor_comparison_panel(
                    rgb_np, gt, pred_dio, pred_m3d, pred_dav2,
                    paper_vis_out, f"{dom_name} ({dom['category']})"
                )
                sample_saved = True

        # Aggregate metrics for this domain
        dom_summary = {}
        for m in models:
            if not dom_metrics[m]:
                continue
            agg = {}
            for k in dom_metrics[m][0].keys():
                vals = [x[k] for x in dom_metrics[m] if not math.isnan(x[k])]
                agg[k] = float(np.mean(vals)) if vals else float("nan")
            agg["latency_ms"] = float(np.mean(dom_latencies[m]))
            agg["fps"] = float(1000.0 / agg["latency_ms"]) if agg["latency_ms"] > 0 else 0.0
            dom_summary[m] = agg

            print(f"  [{model_labels[m]:25s}] AbsRel: {agg['abs_rel']:.4f} | RMSE: {agg['rmse']:.3f}m | d1: {agg['delta1']*100:.1f}% | ScaleRatio: {agg['scale_ratio']:.3f} | Latency: {agg['latency_ms']:.1f}ms")

        all_results[dom_id] = {
            "name": dom_name,
            "category": dom["category"],
            "frames": len(dom_metrics["dioptra"]),
            "models": dom_summary,
        }

    # Macro-Averages Across All Indoor Environments
    macro_averages = {}
    for m in models:
        macro_agg = {}
        metric_keys = ["abs_rel", "sq_rel", "rmse", "mae", "rmse_log", "silog", "delta1", "delta2", "delta3", "scale_ratio", "normal_mae", "normal_acc11", "latency_ms", "fps"]
        for k in metric_keys:
            vals = [all_results[d]["models"][m][k] for d in all_results if m in all_results[d]["models"] and not math.isnan(all_results[d]["models"][m][k])]
            macro_agg[k] = float(np.mean(vals)) if vals else float("nan")
        macro_averages[m] = macro_agg

    output_payload = {
        "metadata": {
            "title": "Comprehensive All-Indoor Metric Depth Benchmark",
            "total_tested_frames": total_tested_frames,
            "total_eval_trials": total_eval_trials,
            "total_environments": len(all_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": args.device,
        },
        "macro_averages": macro_averages,
        "domains": all_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n✓ Saved JSON summary: {args.out_json}")

    # Generate Markdown Table Summary
    with open(args.out_md, "w") as f:
        f.write("# Comprehensive All-Indoor Metric Depth Benchmark\n\n")
        f.write(f"**Total Frames Tested:** {total_tested_frames} | **Total Neural Inferences:** {total_eval_trials} | **Total Indoor Environments:** {len(all_results)}\n\n")
        f.write("### Macro-Average Indoor Performance Across All Environments\n\n")
        f.write("| Model | Architecture | Params | AbsRel ↓ | RMSE (m) ↓ | MAE (m) ↓ | $\\delta_1 < 1.25$ ↑ | Scale Ratio | Normal MAE (°) ↓ | Latency (ms) ↓ | Throughput (FPS) ↑ |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m in models:
            p_str = "25.4M" if m == "dioptra" else ("37.5M" if m == "metric3d" else "24.8M")
            arch_str = "ViT-S + Ray ARA Head" if m == "dioptra" else ("ViT-S Canonical" if m == "metric3d" else "ViT-S Metric Head")
            avg = macro_averages[m]
            norm_str = f"{avg['normal_mae']:.1f}°" if not math.isnan(avg['normal_mae']) else "N/A"
            f.write(f"| **{model_labels[m]}** | {arch_str} | {p_str} | **{avg['abs_rel']:.4f}** | **{avg['rmse']:.3f}m** | {avg['mae']:.3f}m | **{avg['delta1']*100:.1f}%** | **{avg['scale_ratio']:.3f}** | {norm_str} | **{avg['latency_ms']:.1f} ms** | **{avg['fps']:.1f} FPS** |\n")

        f.write("\n\n### Per-Environment Detailed Breakdown\n\n")
        f.write("| Environment | Category | Frames | Model | AbsRel ↓ | RMSE (m) ↓ | $\\delta_1 < 1.25$ ↑ | Scale Ratio | Normal MAE (°) ↓ |\n")
        f.write("| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for dom_id, d in all_results.items():
            for m in models:
                if m not in d["models"]:
                    continue
                res = d["models"][m]
                norm_str = f"{res['normal_mae']:.1f}°" if not math.isnan(res['normal_mae']) else "N/A"
                f.write(f"| {d['name']} | {d['category']} | {d['frames']} | {model_labels[m]} | **{res['abs_rel']:.4f}** | {res['rmse']:.3f}m | {res['delta1']*100:.1f}% | {res['scale_ratio']:.3f} | {norm_str} |\n")

    print(f"✓ Saved Markdown summary: {args.out_md}")
    print(f"\n{'='*80}")
    print(f"ALL-INDOOR EVALUATION COMPLETE: {total_eval_trials} INFERENCES ACROSS {len(all_results)} DOMAINS")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
