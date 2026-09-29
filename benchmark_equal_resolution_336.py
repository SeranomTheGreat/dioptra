#!/usr/bin/env python3
"""
benchmark_equal_resolution_336.py
---------------------------------
Rigorous Head-to-Head-to-Head Foundation Model Benchmark on an EXACT EVEN PLAYING FIELD:
All models evaluated at the EXACT SAME NATIVE RESOLUTION OF DIOPTRA: 336 x 336!

Models Evaluated:
1. Dioptra-DINO (27.51M Params) - Native 336x336
2. Metric3D ViT-Small (37.50M Params) - Constrained to 336x336
3. UniDepth V2 ViT-Small (34.18M Params, CVPR 2024) - Constrained to 336x336

Test Suite (100 Verified Indoor Frames):
- ScanNet Scene00 (10 frames) - Handheld iPad Structure Sensor
- InteriorNet (30 frames) - Synthetic Multi-Room Residential Architecture
- Apple Hypersim (60 frames) - Ray-traced Photorealistic Interiors
"""

import os
import sys
import glob
import time
import json
import zipfile
import io
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dioptra Setup
import dioptra_dino
sys.modules["__main__"].DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINOConfig, DioptraDINO

from benchmark_pure_indoor_3000 import (
    compute_metrics,
    scale_intrinsics,
    K_INTERIORNET,
    K_HYPERSIM,
)

K_SCANNET = np.array([
    [577.87, 0.0, 319.5],
    [0.0, 577.87, 239.5],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


# ==============================================================================
# Model Evaluators at Exactly 336x336
# ==============================================================================

class DioptraDINO336Evaluator:
    def __init__(self, checkpoint_path: str, device: str = "mps"):
        self.device = device
        print(f"[Dioptra-DINO] Loading checkpoint: {checkpoint_path}", flush=True)
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
        print(f"[Dioptra-DINO] Ready! ({self.params/1e6:.2f}M params, input_size=336x336, device={device})", flush=True)

    def predict(self, pil_img: PILImage.Image, K_native: np.ndarray, orig_w: int, orig_h: int) -> Tuple[np.ndarray, float]:
        img_res = pil_img.resize((336, 336), PILImage.BILINEAR)
        img_np = np.array(img_res, dtype=np.float32) / 255.0
        img_norm = (img_np - self.mean) / self.std
        img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        K_scaled = scale_intrinsics(K_native, orig_w, orig_h, 336, 336)
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


class Metric3D336Evaluator:
    def __init__(self, device: str = "mps"):
        self.device = device
        print("[Metric3D] Loading Metric3D ViT-Small from torch hub...", flush=True)
        self.model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
        self.model = self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.input_size = (336, 336)
        self.padding = [123.675, 116.28, 103.53]
        self.mean = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1).to(device)
        print(f"[Metric3D] Ready! ({self.params/1e6:.2f}M params, input_size=336x336, device={device})", flush=True)

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
        rgb_padded = cv2.copyMakeBorder(
            rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
            cv2.BORDER_CONSTANT, value=self.padding
        )

        tensor_rgb = torch.from_numpy(rgb_padded.transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
        norm_rgb = (tensor_rgb - self.mean) / self.std

        if self.device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_depth, _, _ = self.model.inference({"input": norm_rgb})
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_h_half : pred_depth.shape[0] - (pad_h - pad_h_half),
            pad_w_half : pred_depth.shape[1] - (pad_w - pad_w_half)
        ]
        pred_depth = F.interpolate(pred_depth[None, None, :, :], (orig_h, orig_w), mode="bilinear", align_corners=False).squeeze()

        canonical_to_real_scale = intrinsic_scaled[0] / 1000.0
        pred_metric = (pred_depth * canonical_to_real_scale).cpu().numpy()
        del tensor_rgb, norm_rgb, pred_depth
        return pred_metric, dt_ms


class UniDepthV2336Evaluator:
    def __init__(self, device: str = "mps"):
        self.device = device
        print("[UniDepth V2] Loading UniDepth V2 ViT-Small from torch hub...", flush=True)
        self.model = torch.hub.load(
            "lpiccinelli-eth/UniDepth",
            "UniDepth",
            version="v2",
            backbone="vits14",
            pretrained=True,
        )
        self.model = self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        print(f"[UniDepth V2] Ready! ({self.params/1e6:.2f}M params, input_size=336x336, device={device})", flush=True)

    def predict(self, pil_img: PILImage.Image, K_native: np.ndarray, orig_w: int, orig_h: int) -> Tuple[np.ndarray, float]:
        img_336 = pil_img.resize((336, 336), PILImage.BILINEAR)
        rgb_t = torch.from_numpy(np.array(img_336)).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        K_scaled = K_native.copy().astype(np.float32)
        K_scaled[0, 0] *= 336.0 / orig_w
        K_scaled[0, 2] *= 336.0 / orig_w
        K_scaled[1, 1] *= 336.0 / orig_h
        K_scaled[1, 2] *= 336.0 / orig_h
        K_t = torch.from_numpy(K_scaled).unsqueeze(0).to(self.device)

        if self.device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.infer(rgb_t, K_t)
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        depth = out["depth"]
        pred_full = F.interpolate(depth, (orig_h, orig_w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        del rgb_t, K_t, out, depth
        return pred_full, dt_ms


# ==============================================================================
# Visualization Generator (5 Panels)
# ==============================================================================

def save_5way_panel(
    save_path: str,
    rgb_np: np.ndarray,
    gt_d: np.ndarray,
    dio_d: np.ndarray,
    m3d_d: np.ndarray,
    uni_d: np.ndarray,
    valid_mask: np.ndarray,
    title_prefix: str,
    m_dio: Dict[str, float],
    m_m3d: Dict[str, float],
    m_uni: Dict[str, float],
):
    vmax = float(np.percentile(gt_d[valid_mask], 98)) if np.sum(valid_mask) > 0 else 5.0
    vmax = max(vmax, 2.0)

    fig, axes = plt.subplots(1, 5, figsize=(25, 4.8), facecolor="#0e1117")
    for ax in axes:
        ax.axis("off")

    # 1. RGB
    axes[0].imshow(rgb_np)
    axes[0].set_title(f"RGB ({title_prefix})", color="white", fontsize=11, fontweight="bold")

    # 2. GT
    gt_vis = np.where(valid_mask, gt_d, np.nan)
    im1 = axes[1].imshow(gt_vis, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[1].set_title(f"Ground Truth ({vmax:.1f}m)", color="white", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    # 3. Dioptra-DINO
    im2 = axes[2].imshow(dio_d, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[2].set_title(f"Dioptra-DINO 336 (AR: {m_dio['abs_rel']:.3f})", color="#58a6ff", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    # 4. Metric3D
    im3 = axes[3].imshow(m3d_d, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[3].set_title(f"Metric3D 336 (AR: {m_m3d['abs_rel']:.3f})", color="#f0883e", fontsize=11, fontweight="bold")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    # 5. UniDepth V2
    im4 = axes[4].imshow(uni_d, cmap="turbo", vmin=0.0, vmax=vmax)
    axes[4].set_title(f"UniDepth V2 336 (AR: {m_uni['abs_rel']:.3f})", color="#ec70a6", fontsize=11, fontweight="bold")
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ==============================================================================
# Main Benchmark Pipeline
# ==============================================================================

def run_equal_resolution_benchmark(
    dioptra_ckpt: str = "staging_volsiai_fine_tuned/dioptra_dino_best.pt",
    output_dir: str = "outputs_equal_resolution_336",
    device: str = "mps",
):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 80)
    print("EQUAL RESOLUTION FOUNDATION BENCHMARK (ALL MODELS @ 336x336)")
    print(f"Dioptra Checkpoint: {dioptra_ckpt}")
    print(f"Device:             {device}")
    print("=" * 80)

    # 1. Models
    dioptra = DioptraDINO336Evaluator(dioptra_ckpt, device=device)
    metric3d = Metric3D336Evaluator(device=device)
    unidepth = UniDepthV2336Evaluator(device=device)

    # 2. Gather Test Frames
    pairs = []

    # A. ScanNet (10 frames)
    scannet_root = "data/scannet_tiny/scene00"
    if os.path.exists(scannet_root):
        colors = sorted(glob.glob(f"{scannet_root}/color/*.jpg"))[:10]
        for c in colors:
            d = c.replace("/color/", "/depth/").replace(".jpg", ".png")
            if os.path.exists(d):
                pairs.append({"suite": "ScanNet", "scene": "Scene00", "rgb": c, "depth": d, "K": K_SCANNET, "type": "png_mm"})

    # B. InteriorNet (30 frames)
    int_scenes = sorted(glob.glob("data/interiornet/*"))[:6]
    for s in int_scenes:
        sc_name = os.path.basename(s)
        rgbs = sorted(glob.glob(f"{s}/**/cam0/data/*.png", recursive=True))[:5]
        for r in rgbs:
            d = r.replace("/cam0/data/", "/depth0/data/")
            if os.path.exists(d):
                pairs.append({"suite": "InteriorNet", "scene": sc_name, "rgb": r, "depth": d, "K": K_INTERIORNET, "type": "png_mm"})

    # C. Apple Hypersim (60 frames)
    hyp_zip = "data/kaggle_upload/hypersim_pack.zip"
    if os.path.exists(hyp_zip):
        with zipfile.ZipFile(hyp_zip, "r") as zf:
            c_files = [n for n in zf.namelist() if "final_preview" in n and n.endswith(".tonemap.jpg")][:60]
            for c in c_files:
                d = c.replace("final_preview", "geometry_hdf5").replace(".tonemap.jpg", ".depth_meters.hdf5")
                sc = c.split("/")[1]
                pairs.append({"suite": "Hypersim", "scene": sc, "rgb": c, "depth": d, "K": K_HYPERSIM, "type": "hyp_zip", "zip": hyp_zip})

    print(f"[Dataset] Indexed {len(pairs)} test frames across ScanNet, InteriorNet, and Hypersim.", flush=True)

    results_by_suite: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        "ScanNet": {"dio": [], "m3d": [], "uni": []},
        "InteriorNet": {"dio": [], "m3d": [], "uni": []},
        "Hypersim": {"dio": [], "m3d": [], "uni": []},
    }
    all_dio: List[Dict[str, float]] = []
    all_m3d: List[Dict[str, float]] = []
    all_uni: List[Dict[str, float]] = []
    latencies = {"dio": [], "m3d": [], "uni": []}

    hyp_zip_obj = None

    for idx, item in enumerate(pairs):
        suite = item["suite"]
        sc = item["scene"]
        K = item["K"]
        dtype = item["type"]

        if dtype == "png_mm":
            pil_img = PILImage.open(item["rgb"]).convert("RGB")
            w, h = pil_img.size
            rgb_np = np.array(pil_img)
            gt_d = np.array(PILImage.open(item["depth"]), dtype=np.float32) / 1000.0
        else:
            if hyp_zip_obj is None:
                hyp_zip_obj = zipfile.ZipFile(item["zip"], "r")
            img_b = hyp_zip_obj.read(item["rgb"])
            dep_b = hyp_zip_obj.read(item["depth"])
            pil_img = PILImage.open(io.BytesIO(img_b)).convert("RGB")
            w, h = pil_img.size
            rgb_np = np.array(pil_img)

            import h5py
            with h5py.File(io.BytesIO(dep_b), "r") as hf:
                raw_d = np.array(hf["dataset"][:], dtype=np.float32)
            dh, dw = raw_d.shape[:2]
            fx_h = K[0, 0] * (float(dw) / 1024.0)
            fy_h = K[1, 1] * (float(dh) / 768.0)
            cx_h = K[0, 2] * (float(dw) / 1024.0)
            cy_h = K[1, 2] * (float(dh) / 768.0)
            K = np.array([[fx_h, 0.0, cx_h], [0.0, fy_h, cy_h], [0.0, 0.0, 1.0]], dtype=np.float32)
            gt_d = raw_d

        valid = (gt_d > 0.1) & (gt_d < 10.0) & ~np.isnan(gt_d)
        if np.sum(valid) < 50:
            continue

        # Predict Dioptra
        d_dio, lat_dio = dioptra.predict(pil_img, K, w, h)
        m_dio = compute_metrics(d_dio, gt_d, valid)

        # Predict Metric3D
        d_m3d, lat_m3d = metric3d.predict(rgb_np, K, w, h)
        m_m3d = compute_metrics(d_m3d, gt_d, valid)

        # Predict UniDepth V2
        d_uni, lat_uni = unidepth.predict(pil_img, K, w, h)
        m_uni = compute_metrics(d_uni, gt_d, valid)

        results_by_suite[suite]["dio"].append(m_dio)
        results_by_suite[suite]["m3d"].append(m_m3d)
        results_by_suite[suite]["uni"].append(m_uni)
        all_dio.append(m_dio)
        all_m3d.append(m_m3d)
        all_uni.append(m_uni)
        latencies["dio"].append(lat_dio)
        latencies["m3d"].append(lat_m3d)
        latencies["uni"].append(lat_uni)

        # Save Visualizations periodically
        if idx in [0, 5, 12, 25, 45, 75]:
            v_name = f"equal336_comp_{idx}_{suite}.png"
            v_path = os.path.join(vis_dir, v_name)
            save_5way_panel(
                v_path, rgb_np, gt_d, d_dio, d_m3d, d_uni, valid,
                f"{suite} - {sc}", m_dio, m_m3d, m_uni
            )

        if (idx + 1) % 10 == 0 or (idx + 1) == len(pairs):
            print(
                f"[{idx+1:3d}/{len(pairs)}] ({suite}) "
                f"Dioptra AR={np.mean([x['abs_rel'] for x in all_dio]):.4f}, d1={np.mean([x['delta1'] for x in all_dio])*100:.1f}% | "
                f"Metric3D AR={np.mean([x['abs_rel'] for x in all_m3d]):.4f}, d1={np.mean([x['delta1'] for x in all_m3d])*100:.1f}% | "
                f"UniDepth AR={np.mean([x['abs_rel'] for x in all_uni]):.4f}, d1={np.mean([x['delta1'] for x in all_uni])*100:.1f}%",
                flush=True
            )

    if hyp_zip_obj is not None:
        hyp_zip_obj.close()

    # 3. Aggregate
    def agg(m_list):
        if not m_list:
            return {}
        return {
            "abs_rel": float(np.mean([x["abs_rel"] for x in m_list])),
            "rmse": float(np.mean([x["rmse"] for x in m_list])),
            "mae": float(np.mean([x["mae"] for x in m_list])),
            "delta1": float(np.mean([x["delta1"] for x in m_list])),
            "delta2": float(np.mean([x["delta2"] for x in m_list])),
            "scale_ratio": float(np.mean([x["scale_ratio"] for x in m_list])),
            "abs_rel_aligned": float(np.mean([x["abs_rel_aligned"] for x in m_list])),
            "delta1_aligned": float(np.mean([x["delta1_aligned"] for x in m_list])),
        }

    summary = {
        "metadata": {
            "resolution": "336x336",
            "checkpoint": dioptra_ckpt,
            "device": device,
            "frames": len(all_dio),
        },
        "overall": {
            "dioptra": agg(all_dio),
            "metric3d": agg(all_m3d),
            "unidepth": agg(all_uni),
            "mean_latency_ms": {
                "dioptra": float(np.mean(latencies["dio"])),
                "metric3d": float(np.mean(latencies["m3d"])),
                "unidepth": float(np.mean(latencies["uni"])),
            },
        },
        "by_dataset": {
            "scannet": {
                "dioptra": agg(results_by_suite["ScanNet"]["dio"]),
                "metric3d": agg(results_by_suite["ScanNet"]["m3d"]),
                "unidepth": agg(results_by_suite["ScanNet"]["uni"]),
            },
            "interiornet": {
                "dioptra": agg(results_by_suite["InteriorNet"]["dio"]),
                "metric3d": agg(results_by_suite["InteriorNet"]["m3d"]),
                "unidepth": agg(results_by_suite["InteriorNet"]["uni"]),
            },
            "hypersim": {
                "dioptra": agg(results_by_suite["Hypersim"]["dio"]),
                "metric3d": agg(results_by_suite["Hypersim"]["m3d"]),
                "unidepth": agg(results_by_suite["Hypersim"]["uni"]),
            },
        },
    }

    out_json = os.path.join(output_dir, "equal_resolution_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown Report
    out_md = os.path.join(output_dir, "equal_resolution_summary.md")
    with open(out_md, "w") as f:
        f.write("# Equal Resolution Foundation Benchmark: Dioptra vs. Metric3D vs. UniDepth (@ 336x336)\n\n")
        f.write(f"**Resolution Constraint**: All models forced to **identical 336x336 input resolution** for a completely even playing field.\n")
        f.write(f"**Evaluated Checkpoint**: `{dioptra_ckpt}`\n")
        f.write(f"**Test Frames**: {len(all_dio)} indoor frames (ScanNet + InteriorNet + Apple Hypersim)\n\n")
        f.write("## 1. Overall Comparative Results (@ 336x336)\n\n")
        f.write("| Model | Parameters | Resolution | Direct AbsRel (↓) | RMSE (m ↓) | MAE (m ↓) | δ < 1.25 (↑) | Scale Ratio | Aligned AbsRel (↓) | Aligned δ₁ (↑) | Device Latency |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for name, key, params in [
            ("**Dioptra-DINO (Ours)**", "dioptra", "27.5M"),
            ("**Metric3D ViT-Small**", "metric3d", "37.5M"),
            ("**UniDepth V2 (CVPR '24)**", "unidepth", "34.2M"),
        ]:
            m = summary["overall"][key]
            lat = summary["overall"]["mean_latency_ms"][key]
            f.write(
                f"| {name} | {params} | 336x336 | **{m['abs_rel']:.4f}** | {m['rmse']:.3f}m | {m['mae']:.3f}m | "
                f"**{m['delta1']*100:.1f}%** | {m['scale_ratio']:.3f} | {m['abs_rel_aligned']:.4f} | "
                f"{m['delta1_aligned']*100:.1f}% | **{lat:.1f}ms** |\n"
            )

        f.write("\n## 2. Breakdown by Dataset (@ 336x336)\n\n")
        for ds_name, ds_key in [
            ("ScanNet (Real iPad Sensor)", "scannet"),
            ("InteriorNet (Residential Rooms)", "interiornet"),
            ("Apple Hypersim (Ray-Traced Rooms)", "hypersim"),
        ]:
            f.write(f"### {ds_name}\n\n")
            f.write("| Model | Direct AbsRel (↓) | RMSE (m ↓) | δ < 1.25 (↑) | Scale Ratio | Aligned AbsRel (↓) |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
            for name, key in [("Dioptra-DINO", "dioptra"), ("Metric3D ViT-S", "metric3d"), ("UniDepth V2", "unidepth")]:
                m = summary["by_dataset"][ds_key][key]
                if m:
                    f.write(f"| **{name}** | **{m['abs_rel']:.4f}** | {m['rmse']:.3f}m | **{m['delta1']*100:.1f}%** | {m['scale_ratio']:.3f} | {m['abs_rel_aligned']:.4f} |\n")
                else:
                    f.write(f"| **{name}** | N/A | N/A | N/A | N/A | N/A |\n")
            f.write("\n")

    print(f"\n✓ Master JSON saved to: {out_json}")
    print(f"✓ Master Markdown report saved to: {out_md}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Equal Resolution Foundation Benchmark (@ 336x336)")
    parser.add_argument("--dioptra-ckpt", type=str, default="staging_volsiai_fine_tuned/dioptra_dino_best.pt")
    parser.add_argument("--out-dir", type=str, default="outputs_equal_resolution_336")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    run_equal_resolution_benchmark(
        dioptra_ckpt=args.dioptra_ckpt,
        output_dir=args.out_dir,
        device=args.device,
    )
