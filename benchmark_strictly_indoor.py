#!/usr/bin/env python3
"""
Strictly Indoor Metric Depth Benchmark & Foundation Model Evaluation Suite
==========================================================================
Direct head-to-head evaluation across 21 purely indoor environments (1,185 frames):
  1. Dioptra-DINO (Ours, 27.5M params, Ray-Conditioned Metric ViT)
  2. Metric3D ViT-Small (TPAMI 2024 / CVPR 2023, 37.5M params, Canonical Camera Foundation Model)
  3. Depth Anything V2 Metric Indoor Small (CVPR 2024, 24.8M params, SOTA Monocular Metric Model)

All outdoor datasets (amusement, desert, ocean, oldtown, neighborhood, seasonsforest, soulcity)
are strictly excluded.
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

# Bind Dioptra-DINO
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

K_NYU_CANONICAL = np.array([
    [518.8579, 0.0, 325.5824],
    [0.0, 518.8579, 253.7362],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_SCANNET_CANONICAL = np.array([
    [577.87, 0.0, 319.5],
    [0.0, 577.87, 239.5],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_INTERIORNET_CANONICAL = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
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
# Surface Normals & Geometric Metrics
# ---------------------------------------------------------------------------

def depth_to_surface_normals(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    valid = depth > 1e-3
    Z = np.where(valid, depth, np.nan)
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    dX_du = np.zeros_like(X)
    dY_du = np.zeros_like(Y)
    dZ_du = np.zeros_like(Z)

    dX_dv = np.zeros_like(X)
    dY_dv = np.zeros_like(Y)
    dZ_dv = np.zeros_like(Z)

    dX_du[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / 2.0
    dY_du[:, 1:-1] = (Y[:, 2:] - Y[:, :-2]) / 2.0
    dZ_du[:, 1:-1] = (Z[:, 2:] - Z[:, :-2]) / 2.0

    dX_dv[1:-1, :] = (X[2:, :] - X[:-2, :]) / 2.0
    dY_dv[1:-1, :] = (Y[2:, :] - Y[:-2, :]) / 2.0
    dZ_dv[1:-1, :] = (Z[2:, :] - Z[:-2, :]) / 2.0

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


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    K: Optional[np.ndarray] = None,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> Dict[str, float]:
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

    # Scale-aligned metrics (oracle median scaling)
    s_opt = np.median(g) / (np.median(p) + 1e-6)
    p_aligned = p * s_opt
    abs_rel_aligned = float(np.mean(np.abs(p_aligned - g) / g))
    rmse_aligned = float(np.sqrt(np.mean((p_aligned - g) ** 2)))
    ratio_aligned = np.maximum(p_aligned / g, g / p_aligned)
    delta1_aligned = float(np.mean(ratio_aligned < 1.25))

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
        "abs_rel_aligned": abs_rel_aligned,
        "rmse_aligned": rmse_aligned,
        "delta1_aligned": delta1_aligned,
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
        print(f"[Dioptra-DINO] Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.cfg = ckpt.get("cfg", DioptraDINOConfig(image_size=336))
        self.model = DioptraDINO(self.cfg)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.eval_sz = getattr(self.cfg, "image_size", 336)
        print(f"[Dioptra-DINO] Ready! ({self.params/1e6:.2f}M params, native_size={self.eval_sz}x{self.eval_sz}, device={device})")

    def predict(self, pil_img: PILImage.Image, K_native: np.ndarray, orig_w: int, orig_h: int) -> Tuple[np.ndarray, float]:
        sz = self.eval_sz
        img_res = pil_img.resize((sz, sz), PILImage.BILINEAR)
        img_np = np.array(img_res, dtype=np.float32) / 255.0
        img_norm = (img_np - self.mean) / self.std
        img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        K_scaled = scale_intrinsics(K_native, orig_w, orig_h, sz, sz)
        K_t = torch.from_numpy(K_scaled).unsqueeze(0).to(self.device)

        if self.device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_res = self.model(img_t, K_t, ara_gate=1.0)
            pred_full = F.interpolate(pred_res, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return pred_full.squeeze().cpu().numpy(), dt_ms


class Metric3DEvaluator:
    def __init__(self, device: str = "mps"):
        self.device = device
        print("[Metric3D] Loading Metric3D ViT-Small from cache...")
        self.model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
        self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.input_size = (616, 1064)
        self.padding = [123.675, 116.28, 103.53]
        self.mean = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1).to(device)
        print(f"[Metric3D] Ready! ({self.params/1e6:.2f}M params, device={device})")

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
        print(f"[Depth Anything V2] Ready! ({self.params/1e6:.2f}M params, device={device})")

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
# Visual Panel Generator
# ---------------------------------------------------------------------------

def save_indoor_comparison_panel(
    rgb_np: np.ndarray,
    gt_depth: np.ndarray,
    pred_dioptra: np.ndarray,
    pred_metric3d: Optional[np.ndarray],
    pred_dav2: Optional[np.ndarray],
    out_path: str,
    domain_title: str,
    max_d: float = 10.0,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    num_cols = 3 + (1 if pred_metric3d is not None else 0) + (1 if pred_dav2 is not None else 0)
    fig, axes = plt.subplots(1, num_cols, figsize=(4.4 * num_cols, 4.2), dpi=150)

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

    col_idx = 3
    if pred_metric3d is not None:
        im_m3d = axes[col_idx].imshow(pred_metric3d, cmap="turbo", vmin=0.0, vmax=vmax)
        axes[col_idx].set_title("Metric3D ViT-Small (TPAMI)", fontsize=11, fontweight="bold")
        axes[col_idx].axis("off")
        plt.colorbar(im_m3d, ax=axes[col_idx], fraction=0.046, pad=0.04)
        col_idx += 1

    if pred_dav2 is not None:
        im_dav2 = axes[col_idx].imshow(pred_dav2, cmap="turbo", vmin=0.0, vmax=vmax)
        axes[col_idx].set_title("Depth Anything V2 Metric", fontsize=11, fontweight="bold")
        axes[col_idx].axis("off")
        plt.colorbar(im_dav2, ax=axes[col_idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Indoor Domain Catalog (21 Domains)
# ---------------------------------------------------------------------------

def discover_indoor_domains() -> List[Dict[str, Any]]:
    domains = [
        # --- TartanAir v1 Indoor Enclosures ---
        {
            "id": "hospital_p001",
            "name": "Hospital P001",
            "category": "Medical Facility",
            "img_pattern": "test_samples/unseen_hospital_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_hospital_p001/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },
        {
            "id": "office_p001",
            "name": "Commercial Office P001",
            "category": "Commercial Office",
            "img_pattern": "test_samples/unseen_office_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_office_p001/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },
        {
            "id": "office_p002",
            "name": "Commercial Office P002",
            "category": "Commercial Office",
            "img_pattern": "test_samples/unseen_office_p002/image_left/*.png",
            "depth_dir": "test_samples/unseen_office_p002/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },
        {
            "id": "office2_p000",
            "name": "Executive Workspace P000",
            "category": "Executive Workspace",
            "img_pattern": "test_samples/unseen_office2_p000/image_left/*.png",
            "depth_dir": "test_samples/unseen_office2_p000/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },
        {
            "id": "carwelding_p001",
            "name": "CarWelding Facility P001",
            "category": "Industrial Robotics",
            "img_pattern": "test_samples/unseen_carwelding_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_carwelding_p001/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },
        {
            "id": "carwelding_p002",
            "name": "CarWelding Facility P002",
            "category": "Industrial Robotics",
            "img_pattern": "test_samples/unseen_carwelding_p002/image_left/*.png",
            "depth_dir": "test_samples/unseen_carwelding_p002/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },
        {
            "id": "gascola_p001",
            "name": "Gascola Plant P001",
            "category": "Industrial Processing",
            "img_pattern": "test_samples/unseen_gascola_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_gascola_p001/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },
        {
            "id": "gascola_p002",
            "name": "Gascola Plant P002",
            "category": "Industrial Processing",
            "img_pattern": "test_samples/unseen_gascola_p002/image_left/*.png",
            "depth_dir": "test_samples/unseen_gascola_p002/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },
        {
            "id": "abandonedfactory_200",
            "name": "Abandoned Factory (200 Hall)",
            "category": "Warehouse Long-Run",
            "img_pattern": "test_samples/unseen_200_abandonedfactory/image_left/*.png",
            "depth_dir": "test_samples/unseen_200_abandonedfactory/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },
        {
            "id": "abandonedfactory_p005",
            "name": "Abandoned Factory P005",
            "category": "Industrial Warehouse",
            "img_pattern": "test_samples/unseen_abandonedfactory_p005/image_left/*.png",
            "depth_dir": "test_samples/unseen_abandonedfactory_p005/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },
        {
            "id": "abandonedfactory_night_p001",
            "name": "Abandoned Factory Night P001",
            "category": "Industrial Low-Light",
            "img_pattern": "test_samples/unseen_abandonedfactory_night_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_abandonedfactory_night_p001/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_TARTANAIR_V1,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.1, "max_d": 80.0,
        },

        # --- TartanAir v2 Indoor Suites ---
        {
            "id": "tartanair2_americandiner",
            "name": "American Diner Interior",
            "category": "Commercial Dining",
            "img_pattern": "test_samples/indoor_suite/tartanair2_americandiner/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_americandiner/depth_left",
            "depth_suffix": ".npy",
            "K": K_TARTANAIR_V2,
            "orig_w": 640, "orig_h": 640, "is_16bit": False,
            "min_d": 0.1, "max_d": 40.0,
        },
        {
            "id": "tartanair2_archviztinyhouseday",
            "name": "Tiny House (Day Architectural)",
            "category": "Residential Interior",
            "img_pattern": "test_samples/indoor_suite/tartanair2_archviztinyhouseday/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_archviztinyhouseday/depth_left",
            "depth_suffix": ".npy",
            "K": K_TARTANAIR_V2,
            "orig_w": 640, "orig_h": 640, "is_16bit": False,
            "min_d": 0.1, "max_d": 40.0,
        },
        {
            "id": "tartanair2_archviztinyhousenight",
            "name": "Tiny House (Night Architectural)",
            "category": "Residential Interior",
            "img_pattern": "test_samples/indoor_suite/tartanair2_archviztinyhousenight/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_archviztinyhousenight/depth_left",
            "depth_suffix": ".npy",
            "K": K_TARTANAIR_V2,
            "orig_w": 640, "orig_h": 640, "is_16bit": False,
            "min_d": 0.1, "max_d": 40.0,
        },
        {
            "id": "tartanair2_prison",
            "name": "Institutional Prison",
            "category": "Institutional Cells",
            "img_pattern": "test_samples/indoor_suite/tartanair2_prison/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_prison/depth_left",
            "depth_suffix": ".npy",
            "K": K_TARTANAIR_V2,
            "orig_w": 640, "orig_h": 640, "is_16bit": False,
            "min_d": 0.1, "max_d": 40.0,
        },
        {
            "id": "tartanair2_retrooffice",
            "name": "Retro Office Workspace",
            "category": "Vintage Commercial",
            "img_pattern": "test_samples/indoor_suite/tartanair2_retrooffice/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_retrooffice/depth_left",
            "depth_suffix": ".npy",
            "K": K_TARTANAIR_V2,
            "orig_w": 640, "orig_h": 640, "is_16bit": False,
            "min_d": 0.1, "max_d": 40.0,
        },
        {
            "id": "tartanair2_house",
            "name": "Suburban House Multi-Room",
            "category": "Residential Multi-Room",
            "img_pattern": "test_samples/indoor_suite/tartanair2_house/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_house/depth_left",
            "depth_suffix": ".npy",
            "K": K_TARTANAIR_V2,
            "orig_w": 640, "orig_h": 640, "is_16bit": False,
            "min_d": 0.1, "max_d": 40.0,
        },
        {
            "id": "tartanair2_supermarket",
            "name": "Retail Grocery Supermarket",
            "category": "Retail Interior",
            "img_pattern": "test_samples/indoor_suite/tartanair2_supermarket/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_supermarket/depth_left",
            "depth_suffix": ".npy",
            "K": K_TARTANAIR_V2,
            "orig_w": 640, "orig_h": 640, "is_16bit": False,
            "min_d": 0.1, "max_d": 40.0,
        },

        # --- Real-World Indoor RGB-D Sensors ---
        {
            "id": "scannet_scene00",
            "name": "ScanNet Scene00 (iPad Structure Sensor)",
            "category": "Real-World Handheld RGB-D",
            "img_pattern": "data/scannet_tiny/**/color/*.jpg",
            "depth_pattern_func": lambda c: c.replace("/color/", "/depth/").replace(".jpg", ".png"),
            "K": K_SCANNET_CANONICAL,
            "orig_w": 640, "orig_h": 480, "is_16bit": True,
            "min_d": 0.4, "max_d": 10.0,
        },
        {
            "id": "nyu_depth_v2",
            "name": "NYU-Depth V2 (Kinect Official Split)",
            "category": "Real-World Indoor Kinect",
            "img_pattern": "test_samples/indoor_suite/nyu_depth_v2/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/nyu_depth_v2/depth_left",
            "depth_suffix": "_depth.npy",
            "K": K_NYU_CANONICAL,
            "orig_w": 640, "orig_h": 480, "is_16bit": False,
            "min_d": 0.5, "max_d": 10.0,
        },

        # --- Photorealistic Interior Simulation ---
        {
            "id": "interiornet_guestroom",
            "name": "InteriorNet (Guest Room)",
            "category": "Photorealistic Indoor Sim",
            "img_pattern": "data/interiornet/3FO4MMTWI01K_Guest_room/3FO4MMTWI01K_Guest_room/cam0/data/*.png",
            "depth_pattern_func": lambda c: c.replace("/cam0/data/", "/depth0/data/"),
            "K": K_INTERIORNET_CANONICAL,
            "orig_w": 640, "orig_h": 480, "is_16bit": True,
            "min_d": 0.2, "max_d": 10.0,
        },
    ]

    # Resolve pairs for each domain
    valid_domains = []
    for d in domains:
        pairs = []
        if "depth_pattern_func" in d:
            imgs = sorted(glob.glob(d["img_pattern"], recursive=True))
            for c in imgs:
                dp = d["depth_pattern_func"](c)
                if os.path.exists(dp):
                    pairs.append((c, dp))
        else:
            imgs = sorted(glob.glob(d["img_pattern"]))
            for c in imgs:
                stem = os.path.basename(c).replace(".png", "")
                dep_name = f"{stem}{d['depth_suffix']}"
                dp = os.path.join(d["depth_dir"], dep_name)
                if not os.path.exists(dp):
                    # Try direct stem matching
                    dep_cands = glob.glob(os.path.join(d["depth_dir"], f"*{stem}*"))
                    if dep_cands:
                        dp = dep_cands[0]
                if os.path.exists(dp):
                    pairs.append((c, dp))
        if pairs:
            d["pairs"] = pairs
            valid_domains.append(d)
        else:
            print(f"[Warning] 0 valid pairs found for {d['name']} ({d['id']})")

    return valid_domains


# ---------------------------------------------------------------------------
# Main Benchmark Routine
# ---------------------------------------------------------------------------

def run_strictly_indoor_benchmark(
    checkpoint_path: str = "outputs_dino/dioptra_dino_best.pt",
    device: str = "mps",
    output_dir: str = "outputs_strictly_indoor",
    compare_competitors: bool = True,
    max_frames_per_domain: Optional[int] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 90)
    print("STRICTLY INDOOR METRIC DEPTH BENCHMARK SUITE")
    print(f"Device: {device.upper()} | Model: {checkpoint_path}")
    print(f"Competitor Models: {'Enabled (Metric3D & Depth Anything V2)' if compare_competitors else 'Disabled'}")
    print("=" * 90)

    # 1. Discover all indoor domains
    domains = discover_indoor_domains()
    total_frames = sum(len(d["pairs"]) if max_frames_per_domain is None else min(len(d["pairs"]), max_frames_per_domain) for d in domains)
    print(f"\n[Domain Discovery] Discovered {len(domains)} strictly indoor domains totaling {total_frames} frames:")
    for d in domains:
        n_eval = len(d["pairs"]) if max_frames_per_domain is None else min(len(d["pairs"]), max_frames_per_domain)
        print(f"  - {d['name']:40s}: {n_eval:3d} frames ({d['category']})")

    # 2. Initialize Models
    dioptra = DioptraDINOEvaluator(checkpoint_path, device=device)
    metric3d = Metric3DEvaluator(device=device) if compare_competitors else None
    dav2 = DepthAnythingV2MetricEvaluator(mode="indoor", device=device) if compare_competitors else None

    results = {}
    total_eval_time = 0.0

    print("\n" + "=" * 90)
    print("EXECUTING STRICTLY INDOOR BENCHMARK EVALUATION")
    print("=" * 90)

    for dom in domains:
        dom_id = dom["id"]
        eval_pairs = dom["pairs"] if max_frames_per_domain is None else dom["pairs"][:max_frames_per_domain]
        print(f"\n>>> Evaluating Indoor Domain: {dom['name']} ({len(eval_pairs)} frames)...")

        m_dio_list, m_m3d_list, m_dav2_list = [], [], []
        lat_dio_list, lat_m3d_list, lat_dav2_list = [], [], []

        for idx, (img_p, dep_p) in enumerate(eval_pairs):
            pil_img = PILImage.open(img_p).convert("RGB")
            orig_w, orig_h = pil_img.size
            rgb_np = np.array(pil_img)

            if dom["is_16bit"]:
                gt_depth = np.array(PILImage.open(dep_p), dtype=np.float32) / 1000.0
            else:
                gt_depth = np.load(dep_p).astype(np.float32)

            # Ensure 2D depth
            if gt_depth.ndim == 3:
                gt_depth = gt_depth.squeeze()

            # Predict Dioptra-DINO
            pred_dio, dt_dio = dioptra.predict(pil_img, dom["K"], orig_w, orig_h)
            lat_dio_list.append(dt_dio)
            m_dio = compute_metrics(pred_dio, gt_depth, K=dom["K"], min_depth=dom["min_d"], max_depth=dom["max_d"])
            if m_dio:
                m_dio_list.append(m_dio)

            # Predict Metric3D
            pred_m3d = None
            if metric3d is not None:
                try:
                    pred_m3d, dt_m3d = metric3d.predict(rgb_np, dom["K"], orig_w, orig_h)
                    lat_m3d_list.append(dt_m3d)
                    m_m3d = compute_metrics(pred_m3d, gt_depth, K=dom["K"], min_depth=dom["min_d"], max_depth=dom["max_d"])
                    if m_m3d:
                        m_m3d_list.append(m_m3d)
                except Exception as e:
                    pass

            # Predict Depth Anything V2 Indoor
            pred_dav2 = None
            if dav2 is not None:
                try:
                    pred_dav2, dt_dav2 = dav2.predict(pil_img, orig_w, orig_h)
                    lat_dav2_list.append(dt_dav2)
                    m_dav2 = compute_metrics(pred_dav2, gt_depth, K=dom["K"], min_depth=dom["min_d"], max_depth=dom["max_d"])
                    if m_dav2:
                        m_dav2_list.append(m_dav2)
                except Exception as e:
                    pass

            # Save representative visual comparison panel (frame 0)
            if idx == 0:
                panel_out = os.path.join(vis_dir, f"{dom_id}_comparison.png")
                save_indoor_comparison_panel(
                    rgb_np=rgb_np,
                    gt_depth=gt_depth,
                    pred_dioptra=pred_dio,
                    pred_metric3d=pred_m3d,
                    pred_dav2=pred_dav2,
                    out_path=panel_out,
                    domain_title=dom["name"],
                    max_d=min(dom["max_d"], 10.0 if "nyu" in dom_id or "scannet" in dom_id or "interiornet" in dom_id else 40.0),
                )

            if (idx + 1) % max(1, len(eval_pairs) // 2) == 0 or (idx + 1) == len(eval_pairs):
                cur_dio_ar = np.mean([x["abs_rel"] for x in m_dio_list]) if m_dio_list else 0.0
                cur_dio_d1 = np.mean([x["delta1"] for x in m_dio_list]) * 100 if m_dio_list else 0.0
                log_str = f"  [{idx+1:3d}/{len(eval_pairs):3d}] Dioptra AbsRel: {cur_dio_ar:.4f} | d1: {cur_dio_d1:.1f}% | Lat: {np.mean(lat_dio_list[-5:]):.1f}ms"
                if m_m3d_list:
                    cur_m3d_ar = np.mean([x["abs_rel"] for x in m_m3d_list])
                    log_str += f" | M3D: {cur_m3d_ar:.4f}"
                if m_dav2_list:
                    cur_da_ar = np.mean([x["abs_rel"] for x in m_dav2_list])
                    log_str += f" | DAV2: {cur_da_ar:.4f}"
                print(log_str)

        def aggregate_metrics(m_list: List[Dict[str, float]], lat_list: List[float]) -> Dict[str, float]:
            if not m_list:
                return {}
            keys = [
                "abs_rel", "sq_rel", "rmse", "mae", "rmse_log", "silog", "delta1", "delta2", "delta3",
                "scale_ratio", "abs_rel_aligned", "rmse_aligned", "delta1_aligned", "normal_mae", "normal_acc11"
            ]
            agg = {}
            for k in keys:
                vals = [m[k] for m in m_list if k in m and not math.isnan(m[k])]
                agg[k] = float(np.mean(vals)) if vals else float("nan")
            valid_lat = lat_list[1:] if len(lat_list) > 1 else lat_list
            agg["latency_ms"] = float(np.mean(valid_lat)) if valid_lat else 0.0
            agg["fps"] = float(1000.0 / agg["latency_ms"]) if agg["latency_ms"] > 0 else 0.0
            agg["frames"] = len(m_list)
            return agg

        results[dom_id] = {
            "name": dom["name"],
            "category": dom["category"],
            "frames": len(eval_pairs),
            "dioptra": aggregate_metrics(m_dio_list, lat_dio_list),
            "metric3d": aggregate_metrics(m_m3d_list, lat_m3d_list) if metric3d else None,
            "depth_anything_v2": aggregate_metrics(m_dav2_list, lat_dav2_list) if dav2 else None,
        }

    # 3. Export Master JSON
    json_path = os.path.join(output_dir, "indoor_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Master evaluation results saved to: {json_path}")

    # 4. Generate Comprehensive Markdown Summary Table
    md_path = os.path.join(output_dir, "indoor_benchmark_summary.md")
    with open(md_path, "w") as f:
        f.write("# Strictly Indoor Metric Depth Benchmark: Full Cross-Domain Report\n\n")
        f.write(f"**Evaluated on**: {time.strftime('%Y-%m-%d %H:%M:%S')} | **Device**: Apple Silicon ({device.upper()})\n")
        f.write(f"**Evaluated Checkpoint**: `{checkpoint_path}`\n")
        f.write(f"**Strict Indoor Environment Coverage**: {len(domains)} distinct indoor environments ({total_frames:,} frames)\n\n")
        f.write("> [!NOTE]\n")
        f.write("> All outdoor scenes (parks, forests, deserts, neighborhoods, oceans) are strictly excluded.\n")
        f.write("> Direct metric evaluation: Physical depth in metres without oracle scale fitting or test-time alignment.\n\n")

        # Table 1: Dioptra-DINO Detailed Indoor Metrics
        f.write("## 1. Dioptra-DINO Performance Across All Indoor Environments\n\n")
        f.write("| Indoor Environment | Category | Frames | Direct AbsRel (↓) | RMSE (m ↓) | δ < 1.25 (↑) | Scale Ratio | Aligned AbsRel (↓) | Aligned δ₁ (↑) | Normal MAE (°) | Latency | FPS |\n")
        f.write("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        all_dio_absrel, all_dio_rmse, all_dio_d1, all_dio_scale, all_dio_norm = [], [], [], [], []

        for dom_id, d in results.items():
            m = d["dioptra"]
            if not m:
                continue
            all_dio_absrel.append(m["abs_rel"])
            all_dio_rmse.append(m["rmse"])
            all_dio_d1.append(m["delta1"])
            all_dio_scale.append(m["scale_ratio"])
            if not math.isnan(m["normal_mae"]):
                all_dio_norm.append(m["normal_mae"])

            norm_str = f"{m['normal_mae']:.1f}°" if not math.isnan(m["normal_mae"]) else "N/A"
            f.write(
                f"| **{d['name']}** | {d['category']} | {d['frames']} | "
                f"**{m['abs_rel']:.4f}** | {m['rmse']:.2f}m | {m['delta1']*100:.1f}% | "
                f"{m['scale_ratio']:.3f} | {m['abs_rel_aligned']:.4f} | {m['delta1_aligned']*100:.1f}% | "
                f"{norm_str} | {m['latency_ms']:.1f}ms | {m['fps']:.1f} |\n"
            )

        f.write(
            f"| **ALL-INDOOR MEAN** | **Comprehensive Aggregate** | **{total_frames}** | "
            f"**{np.mean(all_dio_absrel):.4f}** | **{np.mean(all_dio_rmse):.2f}m** | **{np.mean(all_dio_d1)*100:.1f}%** | "
            f"**{np.mean(all_dio_scale):.3f}** | **-** | **-** | "
            f"**{np.mean(all_dio_norm):.1f}°** | **-** | **-** |\n\n"
        )

        # Table 2: Head-to-Head Foundation Model Comparison
        if compare_competitors:
            f.write("## 2. Head-to-Head Foundation Model Comparison (Zero-Shot Physical Metric Depth)\n\n")
            f.write("| Indoor Environment | Frames | Dioptra-DINO AbsRel (↓) | Metric3D AbsRel (↓) | Depth Anything V2 AbsRel (↓) | Dioptra Scale | M3D Scale | DAV2 Scale | Dioptra δ₁ (↑) | M3D δ₁ (↑) | DAV2 δ₁ (↑) |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

            for dom_id, d in results.items():
                m_dio = d["dioptra"]
                m_m3d = d.get("metric3d") or {}
                m_dav2 = d.get("depth_anything_v2") or {}

                ar_m3d = f"{m_m3d['abs_rel']:.4f}" if "abs_rel" in m_m3d else "N/A"
                ar_dav2 = f"{m_dav2['abs_rel']:.4f}" if "abs_rel" in m_dav2 else "N/A"
                sc_m3d = f"{m_m3d['scale_ratio']:.3f}" if "scale_ratio" in m_m3d else "N/A"
                sc_dav2 = f"{m_dav2['scale_ratio']:.3f}" if "scale_ratio" in m_dav2 else "N/A"
                d1_m3d = f"{m_m3d['delta1']*100:.1f}%" if "delta1" in m_m3d else "N/A"
                d1_dav2 = f"{m_dav2['delta1']*100:.1f}%" if "delta1" in m_dav2 else "N/A"

                f.write(
                    f"| **{d['name']}** | {d['frames']} | "
                    f"**{m_dio['abs_rel']:.4f}** | {ar_m3d} | {ar_dav2} | "
                    f"{m_dio['scale_ratio']:.3f} | {sc_m3d} | {sc_dav2} | "
                    f"**{m_dio['delta1']*100:.1f}%** | {d1_m3d} | {d1_dav2} |\n"
                )

    print(f"✓ Publication markdown report saved to: {md_path}")
    print("\n" + "=" * 90)
    print("STRICTLY INDOOR BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strictly Indoor Metric Depth Benchmark Suite")
    default_ckpt = "staging_download_v5/checkpoint_step_latest.pt"
    if not os.path.exists(default_ckpt):
        default_ckpt = "outputs_dino/dioptra_dino_best.pt"
    parser.add_argument("--checkpoint", type=str, default=default_ckpt)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs_strictly_indoor")
    parser.add_argument("--no-competitors", dest="compare_competitors", action="store_false", help="Skip Metric3D & Depth Anything V2 comparison")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames per domain (e.g. 10 or 25 for quick preview)")
    args = parser.parse_args()

    run_strictly_indoor_benchmark(
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        compare_competitors=args.compare_competitors,
        max_frames_per_domain=args.max_frames,
    )
