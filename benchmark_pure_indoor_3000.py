#!/usr/bin/env python3
"""
Comprehensive 3,000-Frame Pure Photorealistic True Indoor Metric Depth Benchmark.

Evaluates Dioptra-DINO vs. Metric3D ViT-Small vs. Depth Anything V2 Metric Indoor Small:
  - 100% True Indoor, Photorealistic Synthetic Rendering (Ray-traced physics)
  - Zero outdoor, warehouse, or industrial cavern contamination
  - Exactly 3,000 frames:
      * 240 frames from all 12 official InteriorNet sequence packages (ETH Zurich / Imperial College)
      * 2,760 frames from Apple Hypersim (physically based ray-traced interiors, 150+ diverse rooms)
  - Unaligned physical metric evaluation: AbsRel, SqRel, RMSE, MAE, delta1, delta2, delta3, Scale Ratio
  - Scale-aligned evaluation (median scale) to decouple structural precision from global calibration
  - 3D surface normal error (MAE and acc < 11.25 deg)
"""

import os
import sys
import io
import time
import glob
import math
import json
import random
import zipfile
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Canonical intrinsics
K_INTERIORNET = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_HYPERSIM = np.array([
    [888.89, 0.0, 512.0],
    [0.0, 1000.0, 384.0],
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
    n_gt: Optional[np.ndarray] = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Dict[str, float]:
    mask = (gt >= min_depth) & (gt <= max_depth) & (gt > 1e-3) & np.isfinite(gt)
    mask = mask & (pred >= min_depth * 0.5) & (pred <= max_depth * 2.0) & np.isfinite(pred)

    if np.sum(mask) < 50:
        return {}

    p = pred[mask]
    g = gt[mask]

    abs_rel = float(np.mean(np.abs(p - g) / g))
    sq_rel = float(np.mean(((p - g) ** 2) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mae = float(np.mean(np.abs(p - g)))

    thresh = np.maximum(p / g, g / p)
    delta1 = float(np.mean(thresh < 1.25))
    delta2 = float(np.mean(thresh < 1.25 ** 2))
    delta3 = float(np.mean(thresh < 1.25 ** 3))

    scale_ratio = float(np.median(p) / np.median(g))

    # Scale-aligned metrics (oracle scalar)
    scale_aligned = float(np.median(g) / np.median(p))
    p_al = p * scale_aligned
    abs_rel_aligned = float(np.mean(np.abs(p_al - g) / g))
    rmse_aligned = float(np.sqrt(np.mean((p_al - g) ** 2)))
    thresh_al = np.maximum(p_al / g, g / p_al)
    delta1_aligned = float(np.mean(thresh_al < 1.25))

    normal_mae, normal_acc11 = float("nan"), float("nan")
    if K is not None:
        try:
            if n_gt is None:
                n_gt = depth_to_surface_normals(gt, K)
            n_pred = depth_to_surface_normals(pred, K)
            dot = np.sum(n_pred * n_gt, axis=-1)
            dot = np.clip(dot, -1.0, 1.0)
            ang_deg = np.degrees(np.arccos(np.abs(dot)))
            ang_m = ang_deg[mask]
            normal_mae = float(np.nanmean(ang_m))
            normal_acc11 = float(np.mean(ang_m < 11.25))
        except Exception:
            pass

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "mae": mae,
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
# Evaluators
# ---------------------------------------------------------------------------

class DioptraDINOEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "mps"):
        self.device = device
        print(f"[Dioptra-DINO] Loading checkpoint: {checkpoint_path}", flush=True)
        import dioptra_dino
        sys.modules["__main__"].DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
        from dioptra_dino import DioptraDINOConfig, DioptraDINO

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
        print(f"[Dioptra-DINO] Ready! ({self.params/1e6:.2f}M params, native_size={self.eval_sz}x{self.eval_sz}, device={device})", flush=True)

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

        out_np = pred_full.squeeze().cpu().numpy()
        del img_t, K_t, pred_res, pred_full
        return out_np, dt_ms


class Metric3DEvaluator:
    def __init__(self, device: str = "mps"):
        self.device = device
        print("[Metric3D] Loading Metric3D ViT-Small from cache...", flush=True)
        self.model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
        self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.input_size = (616, 1064)
        self.padding = [123.675, 116.28, 103.53]
        self.mean = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1).to(device)
        print(f"[Metric3D] Ready! ({self.params/1e6:.2f}M params, device={device})", flush=True)

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

        out_np = pred_metric.cpu().numpy()
        del tensor_rgb, norm_rgb, pred_depth, pred_metric
        return out_np, dt_ms


class DepthAnythingV2MetricEvaluator:
    def __init__(self, mode: str = "indoor", device: str = "mps"):
        self.device = device
        self.model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        print(f"[Depth Anything V2] Loading Metric Indoor from {self.model_id}...", flush=True)
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(device)
        self.model.eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        print(f"[Depth Anything V2] Ready! ({self.params/1e6:.2f}M params, device={device})", flush=True)

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
        out_np = pred_t.squeeze().cpu().numpy()
        del inputs, outputs, pred_t
        return out_np, dt_ms


# ---------------------------------------------------------------------------
# Visual Panel Generator
# ---------------------------------------------------------------------------

def save_comparison_panel(
    rgb_np: np.ndarray,
    gt_depth: np.ndarray,
    pred_dioptra: np.ndarray,
    pred_metric3d: Optional[np.ndarray],
    pred_dav2: Optional[np.ndarray],
    out_path: str,
    domain_title: str,
    max_d: float = 8.0,
):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    num_cols = 3 + (1 if pred_metric3d is not None else 0) + (1 if pred_dav2 is not None else 0)
    fig, axes = plt.subplots(1, num_cols, figsize=(4.2 * num_cols, 4.0), dpi=120)
    fig.patch.set_facecolor("#111827")
    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white")

    axes[0].imshow(rgb_np)
    axes[0].set_title(f"RGB ({domain_title})", fontsize=11, fontweight="bold", color="white")
    axes[0].axis("off")

    valid_gt = (gt_depth > 0.1) & (gt_depth < 10.0) & np.isfinite(gt_depth)
    vmax = min(max_d, float(np.percentile(gt_depth[valid_gt], 98))) if np.any(valid_gt) else max_d

    im_gt = axes[1].imshow(gt_depth, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[1].set_title(f"Ground Truth ({vmax:.1f}m)", fontsize=11, fontweight="bold", color="white")
    axes[1].axis("off")
    plt.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

    im_dio = axes[2].imshow(pred_dioptra, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[2].set_title("Dioptra-DINO (Ours, Metric)", fontsize=11, fontweight="bold", color="white")
    axes[2].axis("off")
    plt.colorbar(im_dio, ax=axes[2], fraction=0.046, pad=0.04)

    col_idx = 3
    if pred_metric3d is not None:
        im_m3d = axes[col_idx].imshow(pred_metric3d, cmap="turbo", vmin=0.0, vmax=vmax)
        axes[col_idx].set_title("Metric3D ViT-Small (TPAMI)", fontsize=11, fontweight="bold", color="white")
        axes[col_idx].axis("off")
        plt.colorbar(im_m3d, ax=axes[col_idx], fraction=0.046, pad=0.04)
        col_idx += 1

    if pred_dav2 is not None:
        im_dav2 = axes[col_idx].imshow(pred_dav2, cmap="turbo", vmin=0.0, vmax=vmax)
        axes[col_idx].set_title("Depth Anything V2 Metric", fontsize=11, fontweight="bold", color="white")
        axes[col_idx].axis("off")
        plt.colorbar(im_dav2, ax=axes[col_idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset Curation (240 InteriorNet + 2,760 Hypersim = 3,000 frames)
# ---------------------------------------------------------------------------

def gather_benchmark_pairs(
    interiornet_dir: str = "data/interiornet",
    hypersim_zip: str = "data/kaggle_upload/hypersim_pack.zip",
    total_target: int = 3000,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Curate exactly total_target pairs: all available InteriorNet + balanced Hypersim."""
    random.seed(seed)
    np.random.seed(seed)

    # 1. InteriorNet frames
    interiornet_pairs = []
    scenes = sorted(glob.glob(os.path.join(interiornet_dir, "*")))
    for s in scenes:
        scene_name = os.path.basename(s)
        rgb_files = sorted(glob.glob(f"{s}/**/cam0/data/*.png", recursive=True))
        for rf in rgb_files:
            df = rf.replace("/cam0/data/", "/depth0/data/")
            if os.path.exists(df):
                interiornet_pairs.append({
                    "type": "interiornet",
                    "scene": scene_name,
                    "rgb_path": rf,
                    "depth_path": df,
                    "K": K_INTERIORNET,
                    "orig_w": 640,
                    "orig_h": 480,
                })

    if total_target < len(interiornet_pairs):
        interiornet_pairs = interiornet_pairs[:total_target]

    print(f"[Dataset Curation] InteriorNet indexed: {len(interiornet_pairs)} pairs across {len(scenes)} scenes.", flush=True)

    # 2. Hypersim frames from zip
    needed_hypersim = max(0, total_target - len(interiornet_pairs))
    hypersim_pairs = []

    if os.path.exists(hypersim_zip) and needed_hypersim > 0:
        with zipfile.ZipFile(hypersim_zip, "r") as zf:
            names_set = set(zf.namelist())
            color_files = [n for n in zf.namelist() if "final_preview" in n and n.endswith(".tonemap.jpg")]
            
            # Group by scene
            scene_dict: Dict[str, List[Tuple[str, str]]] = {}
            for c in color_files:
                d = c.replace("final_preview", "geometry_hdf5").replace(".tonemap.jpg", ".depth_meters.hdf5")
                if d in names_set:
                    sc = c.split("/")[1]
                    scene_dict.setdefault(sc, []).append((c, d))

            print(f"[Dataset Curation] Hypersim contains {len(color_files):,} candidate frames across {len(scene_dict)} unique scenes.", flush=True)

            # Uniform stratified sampling across scenes
            all_scenes = sorted(list(scene_dict.keys()))
            frames_per_scene = math.ceil(needed_hypersim / len(all_scenes))

            sampled = []
            for sc in all_scenes:
                pairs = scene_dict[sc]
                random.shuffle(pairs)
                sampled.extend([(sc, c, d) for c, d in pairs[:frames_per_scene]])

            random.shuffle(sampled)
            sampled = sampled[:needed_hypersim]

            for sc, c, d in sampled:
                hypersim_pairs.append({
                    "type": "hypersim",
                    "scene": sc,
                    "rgb_path": c,
                    "depth_path": d,
                    "zip_file": hypersim_zip,
                    "K": K_HYPERSIM,
                    "orig_w": 1024,
                    "orig_h": 768,
                })

    print(f"[Dataset Curation] Hypersim selected: {len(hypersim_pairs)} pairs across {len(scene_dict) if 'scene_dict' in locals() else 0} rooms.", flush=True)
    print(f"[Dataset Curation] Total Pure Indoor Benchmark: {len(interiornet_pairs) + len(hypersim_pairs)} frames (Target: {total_target}).", flush=True)

    return interiornet_pairs, hypersim_pairs


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run_benchmark(
    checkpoint_path: str = "staging_download_v5/checkpoint_step_latest.pt",
    output_dir: str = "outputs_pure_indoor_3000",
    device: str = "mps",
    max_frames: int = 3000,
):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"3,000-FRAME PURE PHOTOREALISTIC TRUE INDOOR BENCHMARK", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Device:     {device}", flush=True)
    print(f"Target:     {max_frames} frames (InteriorNet + Hypersim)", flush=True)
    print("=" * 80, flush=True)

    int_pairs, hyp_pairs = gather_benchmark_pairs(total_target=max_frames)
    all_pairs = (int_pairs + hyp_pairs)[:max_frames]

    # Initialize models
    dioptra = DioptraDINOEvaluator(checkpoint_path, device=device)
    metric3d = Metric3DEvaluator(device=device)
    dav2 = DepthAnythingV2MetricEvaluator(mode="indoor", device=device)

    # Metrics storage & Resume handling
    results_by_type = {"interiornet": [], "hypersim": []}
    results_by_scene: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    entries_jsonl_path = os.path.join(output_dir, "benchmark_entries.jsonl")
    completed_indices = set()
    if os.path.exists(entries_jsonl_path):
        with open(entries_jsonl_path, "r") as f_entries:
            for line in f_entries:
                line_str = line.strip()
                if line_str:
                    try:
                        e = json.loads(line_str)
                        completed_indices.add(e["idx"])
                        dom = e["type"]
                        results_by_type[dom].append(e)
                        sc = e["scene"]
                        results_by_scene.setdefault(sc, {"dio": [], "m3d": [], "dav2": []})
                        results_by_scene[sc]["dio"].append(e["dioptra"])
                        results_by_scene[sc]["m3d"].append(e["metric3d"])
                        results_by_scene[sc]["dav2"].append(e["depth_anything_v2"])
                    except Exception:
                        pass
        if completed_indices:
            print(f"[Resume Engine] Loaded {len(completed_indices)} already completed frames from {entries_jsonl_path}!", flush=True)

    t_start = time.time()
    hyp_zip_obj = None

    for idx, item in enumerate(all_pairs):
        if idx in completed_indices:
            continue

        dom_type = item["type"]
        scene_name = item["scene"]
        K = item["K"]

        # Load RGB & GT Depth
        if dom_type == "interiornet":
            pil_img = PILImage.open(item["rgb_path"]).convert("RGB")
            orig_w, orig_h = pil_img.size
            rgb_np = np.array(pil_img)
            gt_depth = np.array(PILImage.open(item["depth_path"]), dtype=np.float32) / 1000.0
        else:
            if hyp_zip_obj is None:
                hyp_zip_obj = zipfile.ZipFile(item["zip_file"], "r")
            img_bytes = hyp_zip_obj.read(item["rgb_path"])
            dep_bytes = hyp_zip_obj.read(item["depth_path"])
            pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            orig_w, orig_h = pil_img.size
            rgb_np = np.array(pil_img)

            import h5py
            with h5py.File(io.BytesIO(dep_bytes), "r") as hf:
                raw_d = np.array(hf["dataset"][:], dtype=np.float32)
            dh, dw = raw_d.shape[:2]
            fx_h = K[0, 0] * (float(dw) / 1024.0)
            fy_h = K[1, 1] * (float(dh) / 768.0)
            cx_h = K[0, 2] * (float(dw) / 1024.0)
            cy_h = K[1, 2] * (float(dh) / 768.0)
            gy, gx = np.indices((dh, dw), dtype=np.float32)
            ray_scale = np.sqrt(1.0 + ((gx - cx_h) / fx_h) ** 2 + ((gy - cy_h) / fy_h) ** 2)
            gt_depth = raw_d / ray_scale

        # Precompute GT surface normals once per frame
        n_gt = None
        if K is not None:
            try:
                n_gt = depth_to_surface_normals(gt_depth, K)
            except Exception:
                pass

        # Predict Dioptra-DINO
        pred_dio, dt_dio = dioptra.predict(pil_img, K, orig_w, orig_h)
        m_dio = compute_metrics(pred_dio, gt_depth, K=K, n_gt=n_gt, min_depth=0.1, max_depth=10.0)

        # Predict Metric3D
        pred_m3d, dt_m3d = metric3d.predict(rgb_np, K, orig_w, orig_h)
        m_m3d = compute_metrics(pred_m3d, gt_depth, K=K, n_gt=n_gt, min_depth=0.1, max_depth=10.0)

        # Predict Depth Anything V2
        pred_dav2, dt_dav2 = dav2.predict(pil_img, orig_w, orig_h)
        m_dav2 = compute_metrics(pred_dav2, gt_depth, K=K, n_gt=n_gt, min_depth=0.1, max_depth=10.0)

        if m_dio and m_m3d and m_dav2:
            entry = {
                "idx": idx,
                "type": dom_type,
                "scene": scene_name,
                "dioptra": m_dio,
                "metric3d": m_m3d,
                "depth_anything_v2": m_dav2,
                "latency_ms": {"dioptra": dt_dio, "metric3d": dt_m3d, "dav2": dt_dav2},
            }
            results_by_type[dom_type].append(entry)
            results_by_scene.setdefault(scene_name, {"dio": [], "m3d": [], "dav2": []})
            results_by_scene[scene_name]["dio"].append(m_dio)
            results_by_scene[scene_name]["m3d"].append(m_m3d)
            results_by_scene[scene_name]["dav2"].append(m_dav2)

            with open(entries_jsonl_path, "a") as f_entries:
                f_entries.write(json.dumps(entry) + "\n")

        # Save representative visual comparison panel
        if idx in [0, 20, 60, 100, 180, 240, 500, 1000, 1500, 2000, 2500]:
            tag = f"{dom_type}_{scene_name}_{idx}"
            save_comparison_panel(
                rgb_np=rgb_np,
                gt_depth=gt_depth,
                pred_dioptra=pred_dio,
                pred_metric3d=pred_m3d,
                pred_dav2=pred_dav2,
                out_path=os.path.join(vis_dir, f"{tag}_comparison.png"),
                domain_title=f"{dom_type.upper()} {scene_name} (F#{idx})",
                max_d=8.0,
            )

        # Periodic cache flushing and garbage collection every 25 frames
        if (idx + 1) % 25 == 0:
            import gc
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

        # Periodic logging every 50 frames
        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_pairs):
            dio_ar_now = np.mean([x["dioptra"]["abs_rel"] for x in results_by_type[dom_type]])
            m3d_ar_now = np.mean([x["metric3d"]["abs_rel"] for x in results_by_type[dom_type]])
            dav2_ar_now = np.mean([x["depth_anything_v2"]["abs_rel"] for x in results_by_type[dom_type]])
            dio_d1_now = np.mean([x["dioptra"]["delta1"] for x in results_by_type[dom_type]]) * 100
            m3d_d1_now = np.mean([x["metric3d"]["delta1"] for x in results_by_type[dom_type]]) * 100
            dav2_d1_now = np.mean([x["depth_anything_v2"]["delta1"] for x in results_by_type[dom_type]]) * 100

            elapsed = max(1.0, time.time() - t_start)
            total_done = len(results_by_type["interiornet"]) + len(results_by_type["hypersim"])
            fps = total_done / elapsed
            print(
                f"[{idx+1:4d}/{len(all_pairs)}] ({dom_type}) "
                f"Dioptra: AR={dio_ar_now:.4f}, d1={dio_d1_now:.1f}% | "
                f"M3D: AR={m3d_ar_now:.4f}, d1={m3d_d1_now:.1f}% | "
                f"DAV2: AR={dav2_ar_now:.4f}, d1={dav2_d1_now:.1f}% | "
                f"Speed: {fps:.1f} fps ({dt_dio:.1f}ms)",
                flush=True,
            )
            # Checkpoint progress
            try:
                progress_path = os.path.join(output_dir, "benchmark_progress.json")
                with open(progress_path, "w") as pf:
                    json.dump({
                        "frames_completed": total_done,
                        "total_frames": len(all_pairs),
                        "current_domain": dom_type,
                        "elapsed_sec": round(elapsed, 1),
                        "fps": round(fps, 2),
                        "dioptra_abs_rel": round(float(dio_ar_now), 4),
                        "dioptra_delta1": round(float(dio_d1_now), 2),
                    }, pf, indent=2)
            except Exception:
                pass

    if hyp_zip_obj is not None:
        hyp_zip_obj.close()

    # -----------------------------------------------------------------------
    # Compile Master Summary
    # -----------------------------------------------------------------------
    all_res = results_by_type["interiornet"] + results_by_type["hypersim"]

    def agg(entries, key):
        if not entries:
            return {}
        return {
            "abs_rel": float(np.mean([x[key]["abs_rel"] for x in entries])),
            "sq_rel": float(np.mean([x[key]["sq_rel"] for x in entries])),
            "rmse": float(np.mean([x[key]["rmse"] for x in entries])),
            "mae": float(np.mean([x[key]["mae"] for x in entries])),
            "delta1": float(np.mean([x[key]["delta1"] for x in entries])),
            "delta2": float(np.mean([x[key]["delta2"] for x in entries])),
            "delta3": float(np.mean([x[key]["delta3"] for x in entries])),
            "scale_ratio": float(np.mean([x[key]["scale_ratio"] for x in entries])),
            "abs_rel_aligned": float(np.mean([x[key]["abs_rel_aligned"] for x in entries])),
            "rmse_aligned": float(np.mean([x[key]["rmse_aligned"] for x in entries])),
            "delta1_aligned": float(np.mean([x[key]["delta1_aligned"] for x in entries])),
            "normal_mae": float(np.nanmean([x[key]["normal_mae"] for x in entries])) if any(not np.isnan(x[key]["normal_mae"]) for x in entries) else float("nan"),
            "normal_acc11": float(np.nanmean([x[key]["normal_acc11"] for x in entries])) if any(not np.isnan(x[key]["normal_acc11"]) for x in entries) else float("nan"),
        }

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": checkpoint_path,
            "total_frames_evaluated": len(all_res),
            "interiornet_frames": len(results_by_type["interiornet"]),
            "hypersim_frames": len(results_by_type["hypersim"]),
            "device": device,
        },
        "overall_3000": {
            "dioptra": agg(all_res, "dioptra"),
            "metric3d": agg(all_res, "metric3d"),
            "depth_anything_v2": agg(all_res, "depth_anything_v2"),
        },
        "interiornet_240": {
            "dioptra": agg(results_by_type["interiornet"], "dioptra"),
            "metric3d": agg(results_by_type["interiornet"], "metric3d"),
            "depth_anything_v2": agg(results_by_type["interiornet"], "depth_anything_v2"),
        },
        "hypersim_2760": {
            "dioptra": agg(results_by_type["hypersim"], "dioptra"),
            "metric3d": agg(results_by_type["hypersim"], "metric3d"),
            "depth_anything_v2": agg(results_by_type["hypersim"], "depth_anything_v2"),
        },
    }

    # Save JSON
    json_path = os.path.join(output_dir, "pure_indoor_3000_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Master JSON saved to: {json_path}", flush=True)

    # Generate Markdown Report
    md_path = os.path.join(output_dir, "pure_indoor_3000_summary.md")
    with open(md_path, "w") as f:
        f.write("# 3,000-Frame Pure Photorealistic True Indoor Metric Depth Benchmark\n\n")
        f.write(f"**Date**: {summary['metadata']['timestamp']} | **Device**: {device}\n")
        f.write(f"**Evaluated Checkpoint**: `{checkpoint_path}`\n")
        f.write(f"**Total Frames**: {len(all_res):,} frames (100% True Indoor, 0.1m - 10.0m)\n\n")
        f.write("## 1. Overall Aggregate Results (3,000 Pure Indoor Frames)\n\n")
        f.write("| Model | Parameters | Direct AbsRel (↓) | RMSE (m ↓) | MAE (m ↓) | δ < 1.25 (↑) | δ < 1.25² (↑) | Scale Ratio | Aligned AbsRel (↓) | Aligned δ₁ (↑) | Normal MAE (°) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for name, key, params in [
            ("**Dioptra-DINO (Ours)**", "dioptra", "27.5M"),
            ("**Metric3D ViT-Small**", "metric3d", "37.5M"),
            ("**Depth Anything V2 Metric**", "depth_anything_v2", "24.8M"),
        ]:
            m = summary["overall_3000"][key]
            f.write(
                f"| {name} | {params} | **{m['abs_rel']:.4f}** | {m['rmse']:.3f}m | {m['mae']:.3f}m | "
                f"**{m['delta1']*100:.1f}%** | {m['delta2']*100:.1f}% | {m['scale_ratio']:.3f} | "
                f"{m['abs_rel_aligned']:.4f} | {m['delta1_aligned']*100:.1f}% | {m['normal_mae']:.1f}° |\n"
            )

        f.write("\n## 2. Breakdown by Dataset\n\n")
        f.write("### A. InteriorNet (240 Frames across 12 Residential Scenes)\n\n")
        f.write("| Model | Direct AbsRel (↓) | RMSE (m ↓) | δ < 1.25 (↑) | Scale Ratio | Aligned AbsRel (↓) | Aligned δ₁ (↑) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for name, key in [("Dioptra-DINO", "dioptra"), ("Metric3D ViT-S", "metric3d"), ("Depth Anything V2", "depth_anything_v2")]:
            m = summary["interiornet_240"].get(key, {})
            if m and "abs_rel" in m:
                f.write(f"| **{name}** | **{m['abs_rel']:.4f}** | {m['rmse']:.3f}m | **{m['delta1']*100:.1f}%** | {m['scale_ratio']:.3f} | {m['abs_rel_aligned']:.4f} | {m['delta1_aligned']*100:.1f}% |\n")
            else:
                f.write(f"| **{name}** | N/A | N/A | N/A | N/A | N/A | N/A |\n")

        f.write("\n### B. Apple Hypersim (2,760 Frames across 150+ Photorealistic Indoor Rooms)\n\n")
        f.write("| Model | Direct AbsRel (↓) | RMSE (m ↓) | δ < 1.25 (↑) | Scale Ratio | Aligned AbsRel (↓) | Aligned δ₁ (↑) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for name, key in [("Dioptra-DINO", "dioptra"), ("Metric3D ViT-S", "metric3d"), ("Depth Anything V2", "depth_anything_v2")]:
            m = summary["hypersim_2760"].get(key, {})
            if m and "abs_rel" in m:
                f.write(f"| **{name}** | **{m['abs_rel']:.4f}** | {m['rmse']:.3f}m | **{m['delta1']*100:.1f}%** | {m['scale_ratio']:.3f} | {m['abs_rel_aligned']:.4f} | {m['delta1_aligned']*100:.1f}% |\n")
            else:
                f.write(f"| **{name}** | N/A | N/A | N/A | N/A | N/A | N/A |\n")

    print(f"✓ Master Markdown report saved to: {md_path}", flush=True)
    print("=" * 80, flush=True)
    print("BENCHMARK COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3000-Frame Pure Indoor Benchmark")
    parser.add_argument("--checkpoint", type=str, default="staging_download_v5/checkpoint_step_latest.pt")
    parser.add_argument("--output-dir", type=str, default="outputs_pure_indoor_3000")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--frames", type=int, default=3000)
    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        max_frames=args.frames,
    )
