#!/usr/bin/env python3
"""
Dioptra-DINO vs. UniDepth V2 (CVPR 2024 Foundation Model): Direct Comparative Benchmark.

Evaluates Dioptra-DINO (Fine-Tuned Best, 27.5M) vs. UniDepth V2 ViT-Small (34.2M):
  - Exactly 100 indoor frames:
      * 30 frames from InteriorNet (living rooms, bedrooms, dining rooms)
      * 60 frames from Apple Hypersim (photorealistic ray-traced interiors)
      * 10 frames from ScanNet Scene00 (iPad Structure Sensor handheld RGB-D)
  - Direct physical metric evaluation (0.1m - 10.0m)
  - Scale-aligned evaluation (median scale)
  - Side-by-side visual comparison panels
"""

import os
import sys
import io
import time
import glob
import json
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

# Import UniDepth
sys.path.append(os.path.expanduser("~/.cache/torch/hub/lpiccinelli-eth_UniDepth_main"))

# Import Dioptra-DINO
import dioptra_dino
sys.modules["__main__"].DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINOConfig, DioptraDINO
from benchmark_pure_indoor_3000 import (
    DioptraDINOEvaluator,
    K_INTERIORNET,
    K_HYPERSIM,
    compute_metrics,
    depth_to_surface_normals,
)

K_SCANNET = np.array([
    [577.87, 0.0, 319.5],
    [0.0, 577.87, 239.5],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


class UniDepthV2Evaluator:
    def __init__(self, device: str = "mps"):
        self.device = device
        print("[UniDepth V2] Loading UniDepth V2 ViT-Small (DINOv2 backbone)...", flush=True)
        self.model = torch.hub.load(
            "lpiccinelli-eth/UniDepth",
            "UniDepth",
            version="v2",
            backbone="vits14",
            pretrained=True,
        )
        self.model = self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        print(f"[UniDepth V2] Ready! ({self.params/1e6:.2f}M params, device={device})", flush=True)

    def predict(self, pil_img: PILImage.Image, K_native: np.ndarray, orig_w: int, orig_h: int) -> Tuple[np.ndarray, float]:
        rgb_t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0).float()
        K_t = torch.from_numpy(K_native.astype(np.float32)).unsqueeze(0).float()

        if self.device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.infer(rgb_t, K_t)
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        depth = out["depth"].squeeze().cpu().numpy()
        del rgb_t, K_t, out
        return depth, dt_ms


def run_unidepth_benchmark(
    dioptra_ckpt: str = "staging_volsiai_fine_tuned/dioptra_dino_best.pt",
    output_dir: str = "outputs_unidepth_comparison",
    device: str = "mps",
):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 80)
    print("DIRECT FOUNDATION BENCHMARK: DIOPTRA-DINO VS. UNIDEPTH V2 (CVPR 2024)")
    print(f"Dioptra Checkpoint: {dioptra_ckpt}")
    print(f"Device:             {device}")
    print("=" * 80)

    # 1. Models
    dioptra = DioptraDINOEvaluator(dioptra_ckpt, device=device)
    unidepth = UniDepthV2Evaluator(device=device)

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
        "ScanNet": {"dio": [], "uni": []},
        "InteriorNet": {"dio": [], "uni": []},
        "Hypersim": {"dio": [], "uni": []},
    }
    all_dio: List[Dict[str, float]] = []
    all_uni: List[Dict[str, float]] = []
    latencies = {"dio": [], "uni": []}

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
            gy, gx = np.indices((dh, dw), dtype=np.float32)
            ray_scale = np.sqrt(1.0 + ((gx - cx_h) / fx_h) ** 2 + ((gy - cy_h) / fy_h) ** 2)
            gt_d = raw_d / ray_scale

        # Precompute GT normal
        n_gt = None
        try:
            n_gt = depth_to_surface_normals(gt_d, K)
        except Exception:
            pass

        # Inference Dioptra
        p_dio, dt_dio = dioptra.predict(pil_img, K, w, h)
        m_dio = compute_metrics(p_dio, gt_d, K=K, n_gt=n_gt, min_depth=0.1, max_depth=10.0)

        # Inference UniDepth
        p_uni, dt_uni = unidepth.predict(pil_img, K, w, h)
        m_uni = compute_metrics(p_uni, gt_d, K=K, n_gt=n_gt, min_depth=0.1, max_depth=10.0)

        if m_dio and m_uni:
            results_by_suite[suite]["dio"].append(m_dio)
            results_by_suite[suite]["uni"].append(m_uni)
            all_dio.append(m_dio)
            all_uni.append(m_uni)
            latencies["dio"].append(dt_dio)
            latencies["uni"].append(dt_uni)

        # Visual Comparison Panel
        if idx in [0, 5, 12, 25, 45, 75]:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=120)
            fig.patch.set_facecolor("#111827")
            for ax in axes:
                ax.set_facecolor("#111827")
                ax.tick_params(colors="white")

            axes[0].imshow(rgb_np)
            axes[0].set_title(f"RGB ({suite} - {sc[:12]})", color="white", fontsize=10, fontweight="bold")
            axes[0].axis("off")

            valid_gt = (gt_d > 0.1) & (gt_d < 10.0) & np.isfinite(gt_d)
            vmax = min(8.0, float(np.percentile(gt_d[valid_gt], 98))) if np.any(valid_gt) else 8.0

            im0 = axes[1].imshow(gt_d, cmap="turbo", vmin=0.0, vmax=vmax)
            axes[1].set_title(f"Ground Truth ({vmax:.1f}m)", color="white", fontsize=10, fontweight="bold")
            axes[1].axis("off")
            plt.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04)

            im1 = axes[2].imshow(p_dio, cmap="turbo", vmin=0.0, vmax=vmax)
            axes[2].set_title(f"Dioptra-DINO (AR: {m_dio.get('abs_rel', 0):.3f})", color="#60a5fa", fontsize=10, fontweight="bold")
            axes[2].axis("off")
            plt.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)

            im2 = axes[3].imshow(p_uni, cmap="turbo", vmin=0.0, vmax=vmax)
            axes[3].set_title(f"UniDepth V2 CVPR'24 (AR: {m_uni.get('abs_rel', 0):.3f})", color="#f472b6", fontsize=10, fontweight="bold")
            axes[3].axis("off")
            plt.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)

            plt.tight_layout()
            vis_path = os.path.join(vis_dir, f"unidepth_comp_{idx}_{suite}.png")
            plt.savefig(vis_path, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(pairs):
            dio_ar = np.mean([x["abs_rel"] for x in all_dio])
            uni_ar = np.mean([x["abs_rel"] for x in all_uni])
            dio_d1 = np.mean([x["delta1"] for x in all_dio]) * 100
            uni_d1 = np.mean([x["delta1"] for x in all_uni]) * 100
            print(f"[{idx+1:3d}/{len(pairs)}] ({suite}) Dioptra AR={dio_ar:.4f}, d1={dio_d1:.1f}% | UniDepth AR={uni_ar:.4f}, d1={uni_d1:.1f}%", flush=True)

        if (idx + 1) % 20 == 0:
            import gc
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

    if hyp_zip_obj:
        hyp_zip_obj.close()

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
            "checkpoint": dioptra_ckpt,
            "device": device,
            "frames": len(all_dio),
        },
        "overall": {
            "dioptra": agg(all_dio),
            "unidepth": agg(all_uni),
            "mean_latency_ms": {"dioptra": float(np.mean(latencies["dio"])), "unidepth": float(np.mean(latencies["uni"]))},
        },
        "by_dataset": {
            "scannet": {"dioptra": agg(results_by_suite["ScanNet"]["dio"]), "unidepth": agg(results_by_suite["ScanNet"]["uni"])},
            "interiornet": {"dioptra": agg(results_by_suite["InteriorNet"]["dio"]), "unidepth": agg(results_by_suite["InteriorNet"]["uni"])},
            "hypersim": {"dioptra": agg(results_by_suite["Hypersim"]["dio"]), "unidepth": agg(results_by_suite["Hypersim"]["uni"])},
        },
    }

    out_json = os.path.join(output_dir, "unidepth_vs_dioptra_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown Report
    out_md = os.path.join(output_dir, "unidepth_vs_dioptra_summary.md")
    with open(out_md, "w") as f:
        f.write("# Foundation Model Benchmark: Dioptra-DINO vs. UniDepth V2 (CVPR 2024)\n\n")
        f.write(f"**Evaluated Checkpoint**: `{dioptra_ckpt}`\n")
        f.write(f"**Baseline**: UniDepth V2 ViT-Small (DINOv2-Small backbone, 34.2M params)\n")
        f.write(f"**Test Frames**: {len(all_dio)} indoor frames (ScanNet + InteriorNet + Apple Hypersim)\n\n")
        f.write("## 1. Overall Comparative Results\n\n")
        f.write("| Model | Parameters | Direct AbsRel (↓) | RMSE (m ↓) | MAE (m ↓) | δ < 1.25 (↑) | Scale Ratio | Aligned AbsRel (↓) | Aligned δ₁ (↑) | Device Latency |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for name, key, params in [
            ("**Dioptra-DINO (Ours)**", "dioptra", "27.5M"),
            ("**UniDepth V2 (CVPR 2024)**", "unidepth", "34.2M"),
        ]:
            m = summary["overall"][key]
            lat = summary["overall"]["mean_latency_ms"][key]
            f.write(
                f"| {name} | {params} | **{m['abs_rel']:.4f}** | {m['rmse']:.3f}m | {m['mae']:.3f}m | "
                f"**{m['delta1']*100:.1f}%** | {m['scale_ratio']:.3f} | {m['abs_rel_aligned']:.4f} | "
                f"{m['delta1_aligned']*100:.1f}% | **{lat:.1f}ms** |\n"
            )

        f.write("\n## 2. Breakdown by Dataset\n\n")
        for ds_name, ds_key in [("ScanNet (Real iPad Sensor)", "scannet"), ("InteriorNet (Residential Rooms)", "interiornet"), ("Apple Hypersim (Ray-Traced Rooms)", "hypersim")]:
            f.write(f"### {ds_name}\n\n")
            f.write("| Model | Direct AbsRel (↓) | RMSE (m ↓) | δ < 1.25 (↑) | Scale Ratio | Aligned AbsRel (↓) |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
            for m_name, m_key in [("Dioptra-DINO", "dioptra"), ("UniDepth V2", "unidepth")]:
                m = summary["by_dataset"][ds_key][m_key]
                if m and "abs_rel" in m:
                    f.write(f"| **{m_name}** | **{m['abs_rel']:.4f}** | {m['rmse']:.3f}m | **{m['delta1']*100:.1f}%** | {m['scale_ratio']:.3f} | {m['abs_rel_aligned']:.4f} |\n")
                else:
                    f.write(f"| **{m_name}** | N/A | N/A | N/A | N/A | N/A |\n")
            f.write("\n")

    print(f"\n✓ Master JSON saved to: {out_json}")
    print(f"✓ Master Markdown report saved to: {out_md}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dioptra-ckpt", default="staging_volsiai_fine_tuned/dioptra_dino_best.pt")
    parser.add_argument("--out-dir", default="outputs_unidepth_comparison")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    run_unidepth_benchmark(args.dioptra_ckpt, args.out_dir, args.device)
