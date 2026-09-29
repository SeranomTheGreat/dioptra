#!/usr/bin/env python3
"""
Dioptra-DINO: Direct Progression Benchmark (Step 76,206 vs. Epoch 5 Fine-Tuned).

Compares:
  - Checkpoint A (Baseline): staging_download_v5/checkpoint_step_latest.pt (Step 76,206)
  - Checkpoint B (Fine-Tuned): staging_volsiai_fine_tuned/dioptra_dino_best.pt (Epoch 5 / Best)

Evaluates on:
  1. ScanNet Scene00 (Real-world iPad Structure Sensor RGB-D)
  2. InteriorNet (Photorealistic residential interior rooms)
  3. Apple Hypersim (Photorealistic ray-traced multi-room environments)
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

import dioptra_dino
sys.modules["__main__"].DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINOConfig, DioptraDINO

# Canonical intrinsics
K_SCANNET = np.array([
    [577.87, 0.0, 319.5],
    [0.0, 577.87, 239.5],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

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


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 10.0) -> Dict[str, float]:
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

    # Scale-aligned
    scale_aligned = float(np.median(g) / np.median(p))
    p_al = p * scale_aligned
    abs_rel_aligned = float(np.mean(np.abs(p_al - g) / g))
    rmse_aligned = float(np.sqrt(np.mean((p_al - g) ** 2)))
    thresh_al = np.maximum(p_al / g, g / p_al)
    delta1_aligned = float(np.mean(thresh_al < 1.25))

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
        "valid_pixels": int(np.sum(mask)),
    }


class DioptraDINOModel:
    def __init__(self, checkpoint_path: str, device: str = "mps"):
        self.device = device
        print(f"Loading Dioptra-DINO: {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.cfg = ckpt.get("cfg", DioptraDINOConfig(image_size=336))
        self.model = DioptraDINO(self.cfg)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(device).eval()
        self.epoch = ckpt.get("epoch", "N/A")
        self.step = ckpt.get("global_step", "N/A")
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.eval_sz = getattr(self.cfg, "image_size", 336)
        print(f"Ready: Epoch={self.epoch}, Step={self.step}, Size={self.eval_sz}x{self.eval_sz}", flush=True)

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


def run_progression_benchmark(
    ckpt_a_path: str = "staging_download_v5/checkpoint_step_latest.pt",
    ckpt_b_path: str = "staging_volsiai_fine_tuned/dioptra_dino_best.pt",
    output_dir: str = "outputs_progression_comparison",
    device: str = "mps",
):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("=" * 75)
    print("DIOPTRA-DINO PROGRESSION BENCHMARK: STEP 76,206 VS. FINE-TUNED BEST")
    print(f"Model A (Baseline):   {ckpt_a_path}")
    print(f"Model B (Fine-Tuned): {ckpt_b_path}")
    print(f"Device:               {device}")
    print("=" * 75)

    model_a = DioptraDINOModel(ckpt_a_path, device=device)
    model_b = DioptraDINOModel(ckpt_b_path, device=device)

    # 1. Gather ScanNet Scene00
    scannet_pairs = []
    scannet_root = "data/scannet_frames/scene0000_00"
    if os.path.exists(scannet_root):
        colors = sorted(glob.glob(f"{scannet_root}/color/*.jpg"))[:15]
        for c in colors:
            d = c.replace("/color/", "/depth/").replace(".jpg", ".png")
            if os.path.exists(d):
                scannet_pairs.append({"domain": "ScanNet Scene00", "rgb": c, "depth": d, "K": K_SCANNET, "type": "scannet_png"})

    # 2. Gather InteriorNet
    interiornet_pairs = []
    int_scenes = sorted(glob.glob("data/interiornet/*"))[:6]
    for s in int_scenes:
        sc_name = os.path.basename(s)
        rgbs = sorted(glob.glob(f"{s}/**/cam0/data/*.png", recursive=True))[:5]
        for r in rgbs:
            d = r.replace("/cam0/data/", "/depth0/data/")
            if os.path.exists(d):
                interiornet_pairs.append({"domain": f"InteriorNet ({sc_name[:15]})", "rgb": r, "depth": d, "K": K_INTERIORNET, "type": "interiornet_png"})

    # 3. Gather Hypersim
    hypersim_pairs = []
    hyp_zip = "data/kaggle_upload/hypersim_pack.zip"
    if os.path.exists(hyp_zip):
        with zipfile.ZipFile(hyp_zip, "r") as zf:
            c_files = [n for n in zf.namelist() if "final_preview" in n and n.endswith(".tonemap.jpg")][:60]
            for c in c_files:
                d = c.replace("final_preview", "geometry_hdf5").replace(".tonemap.jpg", ".depth_meters.hdf5")
                sc = c.split("/")[1]
                hypersim_pairs.append({"domain": f"Hypersim ({sc})", "rgb": c, "depth": d, "K": K_HYPERSIM, "type": "hypersim_zip", "zip": hyp_zip})

    all_test = scannet_pairs + interiornet_pairs + hypersim_pairs
    print(f"Total benchmark test frames: {len(all_test)} (ScanNet={len(scannet_pairs)}, InteriorNet={len(interiornet_pairs)}, Hypersim={len(hypersim_pairs)})")

    results_a: List[Dict[str, float]] = []
    results_b: List[Dict[str, float]] = []
    by_suite: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        "ScanNet": {"a": [], "b": []},
        "InteriorNet": {"a": [], "b": []},
        "Hypersim": {"a": [], "b": []},
    }

    hyp_zip_obj = None

    for idx, item in enumerate(all_test):
        dom = item["domain"]
        dtype = item["type"]
        K = item["K"]

        if dtype in ["scannet_png", "interiornet_png"]:
            pil_img = PILImage.open(item["rgb"]).convert("RGB")
            w, h = pil_img.size
            rgb_np = np.array(pil_img)
            gt_d = np.array(PILImage.open(item["depth"]), dtype=np.float32) / 1000.0
            suite_key = "ScanNet" if "ScanNet" in dom else "InteriorNet"
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
            suite_key = "Hypersim"

        # Predictions
        pred_a, dt_a = model_a.predict(pil_img, K, w, h)
        pred_b, dt_b = model_b.predict(pil_img, K, w, h)

        m_a = compute_metrics(pred_a, gt_d, min_depth=0.1, max_depth=10.0)
        m_b = compute_metrics(pred_b, gt_d, min_depth=0.1, max_depth=10.0)

        if m_a and m_b:
            results_a.append(m_a)
            results_b.append(m_b)
            by_suite[suite_key]["a"].append(m_a)
            by_suite[suite_key]["b"].append(m_b)

        if idx in [0, 5, 12, 20, 35, 60]:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=120)
            fig.patch.set_facecolor("#111827")
            for ax in axes:
                ax.set_facecolor("#111827")
                ax.tick_params(colors="white")

            axes[0].imshow(rgb_np)
            axes[0].set_title(f"RGB ({dom})", color="white", fontsize=10, fontweight="bold")
            axes[0].axis("off")

            valid_gt = (gt_d > 0.1) & (gt_d < 10.0) & np.isfinite(gt_d)
            vmax = min(8.0, float(np.percentile(gt_d[valid_gt], 98))) if np.any(valid_gt) else 8.0

            im0 = axes[1].imshow(gt_d, cmap="turbo", vmin=0.0, vmax=vmax)
            axes[1].set_title(f"Ground Truth ({vmax:.1f}m)", color="white", fontsize=10, fontweight="bold")
            axes[1].axis("off")
            plt.colorbar(im0, ax=axes[1], fraction=0.046, pad=0.04)

            im1 = axes[2].imshow(pred_a, cmap="turbo", vmin=0.0, vmax=vmax)
            axes[2].set_title(f"Step 76,206 (AR: {m_a.get('abs_rel', 0):.3f})", color="white", fontsize=10, fontweight="bold")
            axes[2].axis("off")
            plt.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)

            im2 = axes[3].imshow(pred_b, cmap="turbo", vmin=0.0, vmax=vmax)
            axes[3].set_title(f"Fine-Tuned Best (AR: {m_b.get('abs_rel', 0):.3f})", color="#4ade80", fontsize=10, fontweight="bold")
            axes[3].axis("off")
            plt.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)

            plt.tight_layout()
            vis_path = os.path.join(vis_dir, f"progression_{idx}_{suite_key}.png")
            plt.savefig(vis_path, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

        if (idx + 1) % 15 == 0:
            import gc
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            print(f"[{idx+1:3d}/{len(all_test)}] Model A AR={np.mean([x['abs_rel'] for x in results_a]):.4f} | Model B AR={np.mean([x['abs_rel'] for x in results_b]):.4f}", flush=True)

    if hyp_zip_obj:
        hyp_zip_obj.close()

    def agg(res_list):
        if not res_list:
            return {}
        return {
            "abs_rel": float(np.mean([x["abs_rel"] for x in res_list])),
            "rmse": float(np.mean([x["rmse"] for x in res_list])),
            "mae": float(np.mean([x["mae"] for x in res_list])),
            "delta1": float(np.mean([x["delta1"] for x in res_list])),
            "scale_ratio": float(np.mean([x["scale_ratio"] for x in res_list])),
            "abs_rel_aligned": float(np.mean([x["abs_rel_aligned"] for x in res_list])),
            "delta1_aligned": float(np.mean([x["delta1_aligned"] for x in res_list])),
        }

    summary = {
        "overall": {"step_76206": agg(results_a), "finetuned_best": agg(results_b)},
        "scannet": {"step_76206": agg(by_suite["ScanNet"]["a"]), "finetuned_best": agg(by_suite["ScanNet"]["b"])},
        "interiornet": {"step_76206": agg(by_suite["InteriorNet"]["a"]), "finetuned_best": agg(by_suite["InteriorNet"]["b"])},
        "hypersim": {"step_76206": agg(by_suite["Hypersim"]["a"]), "finetuned_best": agg(by_suite["Hypersim"]["b"])},
    }

    out_json = os.path.join(output_dir, "progression_comparison.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("PROGRESSION SUMMARY RESULTS:")
    print("=" * 75)
    print(f"Overall Direct AbsRel: Step 76,206 = {summary['overall']['step_76206']['abs_rel']:.4f}  -->  Fine-Tuned Best = {summary['overall']['finetuned_best']['abs_rel']:.4f}")
    print(f"Overall delta1:        Step 76,206 = {summary['overall']['step_76206']['delta1']*100:.1f}%  -->  Fine-Tuned Best = {summary['overall']['finetuned_best']['delta1']*100:.1f}%")
    print(f"ScanNet AbsRel:        Step 76,206 = {summary['scannet']['step_76206']['abs_rel']:.4f}  -->  Fine-Tuned Best = {summary['scannet']['finetuned_best']['abs_rel']:.4f}")
    print(f"Hypersim AbsRel:       Step 76,206 = {summary['hypersim']['step_76206']['abs_rel']:.4f}  -->  Fine-Tuned Best = {summary['hypersim']['finetuned_best']['abs_rel']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", default="staging_download_v5/checkpoint_step_latest.pt")
    parser.add_argument("--ckpt-b", default="staging_volsiai_fine_tuned/dioptra_dino_best.pt")
    parser.add_argument("--out-dir", default="outputs_progression_comparison")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    run_progression_benchmark(args.ckpt_a, args.ckpt_b, args.out_dir, args.device)
