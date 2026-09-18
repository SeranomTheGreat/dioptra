"""
Dioptra-DINO vs. External Foundation Model Benchmark.

Evaluates on the 200 unseen held-out frames (TartanAir AbandonedFactory P010):
1. Dioptra-DINO (Ours, 27.51M params, Metric Camera-Ray Conditioning)
2. Canonical 2D ViT + DPT (Internal Baseline, 26.10M params, No Rays)
3. Depth Anything V2 Small (External Foundation Baseline, 24.79M params, DINOv2-Small)
   - Evaluated under MiDaS Disparity Affine Alignment (s * d + t)
   - Evaluated under Median-Scaling (Eigen Protocol)

Device: Apple M3 Silicon (MPS), PyTorch 2.8.0.
"""

import os
import sys
import time
import glob
import json
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dioptra_dino
import __main__
__main__.DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 80.0) -> Dict[str, float]:
    """Compute standard metric depth evaluation metrics."""
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & (pred > min_depth) & (pred < max_depth) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0, "a1": 0.0, "a2": 0.0, "a3": 0.0, "scale_ratio": 1.0}

    p = pred[mask]
    g = gt[mask]

    thresh = np.maximum((g / p), (p / g))
    a1 = float((thresh < 1.25).mean())
    a2 = float((thresh < 1.25 ** 2).mean())
    a3 = float((thresh < 1.25 ** 3).mean())

    rmse = float(np.sqrt(((g - p) ** 2).mean()))
    rmse_log = float(np.sqrt(((np.log(g) - np.log(p)) ** 2).mean()))
    abs_rel = float((np.abs(g - p) / g).mean())
    sq_rel = float((((g - p) ** 2) / g).mean())
    scale_ratio = float(np.median(p) / np.median(g))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "scale_ratio": scale_ratio,
        "valid_pixels": int(mask.sum()),
    }


def preprocess_dioptra(img_path: str, gt_path: str, img_size: int = 224, device: torch.device = torch.device("cpu")):
    """Preprocess sample for Dioptra-DINO."""
    raw_img = Image.open(img_path).convert("RGB")
    W_orig, H_orig = raw_img.size
    img_resized = raw_img.resize((img_size, img_size), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    norm_img = (img_np - mean) / std
    tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)

    gt_raw = np.load(gt_path)
    gt_pil = Image.fromarray(gt_raw.astype(np.float32))
    gt_resized = gt_pil.resize((img_size, img_size), Image.NEAREST)
    gt_np = np.array(gt_resized, dtype=np.float32)

    fx = 320.0 * (img_size / W_orig)
    fy = 320.0 * (img_size / H_orig)
    cx = 320.0 * (img_size / W_orig)
    cy = 240.0 * (img_size / H_orig)
    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    return tensor_img, gt_np, K


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Benchmark] Running on device: {device} (Apple M3)")

    data_dir = "test_samples/unseen_200_abandonedfactory"
    img_files = sorted(glob.glob(os.path.join(data_dir, "image_left", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(data_dir, "depth_left", "*.npy")))

    assert len(img_files) == 200, f"Expected 200 images, got {len(img_files)}"
    assert len(depth_files) == 200, f"Expected 200 depth files, got {len(depth_files)}"
    pairs = list(zip(img_files, depth_files))
    print(f"[Benchmark] Loaded {len(pairs)} test pairs from {data_dir}")

    results = {}

    # -------------------------------------------------------------
    # 1. Evaluate Headline Dioptra-DINO
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("Evaluating: Dioptra-DINO (Full Headline Model, Metric Depth)")
    print("="*70)
    cfg_full = DioptraDINOConfig(freeze_backbone=False)
    model_full = DioptraDINO(cfg_full).to(device)
    state = torch.load("outputs/dioptra_dino_best.pt", map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    sd_clean = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model_full.load_state_dict(sd_clean, strict=False)
    model_full.eval()
    p_count_full = sum(p.numel() for p in model_full.parameters())

    # Warmup
    dummy_img, dummy_gt, dummy_K = preprocess_dioptra(pairs[0][0], pairs[0][1], device=device)
    for _ in range(5):
        with torch.no_grad():
            _ = model_full(dummy_img, dummy_K)
    if device.type == "mps":
        torch.mps.synchronize()

    m_list_full = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for img_p, gt_p in pairs:
            inp, gt_np, K = preprocess_dioptra(img_p, gt_p, device=device)
            pred = model_full(inp, K).squeeze().cpu().numpy()
            m_list_full.append(compute_metrics(pred, gt_np))
    if device.type == "mps":
        torch.mps.synchronize()
    total_time_full = time.perf_counter() - t0
    fps_full = len(pairs) / total_time_full
    latency_full = (total_time_full / len(pairs)) * 1000.0

    results["dioptra_dino"] = {
        "name": "Dioptra-DINO (Ours)",
        "backbone": "DINOv2-Small + Ray-FiLM + ARA",
        "params": p_count_full,
        "abs_rel": float(np.mean([m["abs_rel"] for m in m_list_full])),
        "sq_rel": float(np.mean([m["sq_rel"] for m in m_list_full])),
        "rmse": float(np.mean([m["rmse"] for m in m_list_full])),
        "rmse_log": float(np.mean([m["rmse_log"] for m in m_list_full])),
        "delta1": float(np.mean([m["a1"] for m in m_list_full]) * 100),
        "delta2": float(np.mean([m["a2"] for m in m_list_full]) * 100),
        "delta3": float(np.mean([m["a3"] for m in m_list_full]) * 100),
        "scale_ratio": float(np.mean([m["scale_ratio"] for m in m_list_full])),
        "latency_ms": latency_full,
        "fps": fps_full,
    }
    print(f"Result: AbsRel={results['dioptra_dino']['abs_rel']:.4f}, RMSE={results['dioptra_dino']['rmse']:.4f}, d1={results['dioptra_dino']['delta1']:.2f}%, Latency={latency_full:.1f}ms ({fps_full:.1f} FPS)")

    del model_full
    if device.type == "mps":
        torch.mps.empty_cache()

    # -------------------------------------------------------------
    # 2. Evaluate Canonical 2D ViT + DPT (Internal Baseline)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("Evaluating: Canonical 2D ViT + DPT (Ablation Variant b, Metric Depth)")
    print("="*70)
    cfg_noray = DioptraDINOConfig(freeze_backbone=False)
    model_noray = DioptraDINO(cfg_noray).to(device)
    model_noray.ray_modulation = None
    model_noray.ara_refine = None
    state_noray = torch.load("outputs_ablations/ablation_no_ray/dioptra_dino_no_ray_best.pt", map_location=device, weights_only=False)
    sd_n = state_noray.get("model_state_dict", state_noray)
    sd_n_clean = {k[7:] if k.startswith("module.") else k: v for k, v in sd_n.items()}
    model_noray.load_state_dict(sd_n_clean, strict=False)
    model_noray.eval()
    p_count_noray = sum(p.numel() for p in model_noray.parameters())

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model_noray(dummy_img, dummy_K)
    if device.type == "mps":
        torch.mps.synchronize()

    m_list_noray = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for img_p, gt_p in pairs:
            inp, gt_np, K = preprocess_dioptra(img_p, gt_p, device=device)
            pred = model_noray(inp, K).squeeze().cpu().numpy()
            m_list_noray.append(compute_metrics(pred, gt_np))
    if device.type == "mps":
        torch.mps.synchronize()
    total_time_noray = time.perf_counter() - t0
    fps_noray = len(pairs) / total_time_noray
    latency_noray = (total_time_noray / len(pairs)) * 1000.0

    results["canonical_vit_dpt"] = {
        "name": "Canonical 2D ViT + DPT",
        "backbone": "DINOv2-Small + Standard DPT",
        "params": p_count_noray,
        "abs_rel": float(np.mean([m["abs_rel"] for m in m_list_noray])),
        "sq_rel": float(np.mean([m["sq_rel"] for m in m_list_noray])),
        "rmse": float(np.mean([m["rmse"] for m in m_list_noray])),
        "rmse_log": float(np.mean([m["rmse_log"] for m in m_list_noray])),
        "delta1": float(np.mean([m["a1"] for m in m_list_noray]) * 100),
        "delta2": float(np.mean([m["a2"] for m in m_list_noray]) * 100),
        "delta3": float(np.mean([m["a3"] for m in m_list_noray]) * 100),
        "scale_ratio": float(np.mean([m["scale_ratio"] for m in m_list_noray])),
        "latency_ms": latency_noray,
        "fps": fps_noray,
    }
    print(f"Result: AbsRel={results['canonical_vit_dpt']['abs_rel']:.4f}, RMSE={results['canonical_vit_dpt']['rmse']:.4f}, d1={results['canonical_vit_dpt']['delta1']:.2f}%, Latency={latency_noray:.1f}ms ({fps_noray:.1f} FPS)")

    del model_noray
    if device.type == "mps":
        torch.mps.empty_cache()

    # -------------------------------------------------------------
    # 3. Evaluate Depth Anything V2 Small (External Foundation Baseline)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("Evaluating: Depth Anything V2 Small (External Foundation Baseline)")
    print("="*70)
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    proc_da = AutoImageProcessor.from_pretrained(model_id)
    model_da = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model_da.eval()
    p_count_da = sum(p.numel() for p in model_da.parameters())

    # Warmup
    raw_img_0 = Image.open(pairs[0][0]).convert("RGB")
    inputs_0 = proc_da(images=raw_img_0, return_tensors="pt").to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = model_da(**inputs_0)
    if device.type == "mps":
        torch.mps.synchronize()

    m_list_da_midas = []
    m_list_da_median = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for img_p, gt_p in pairs:
            raw_img = Image.open(img_p).convert("RGB")
            gt_raw = np.load(gt_p)
            gt_pil = Image.fromarray(gt_raw.astype(np.float32))
            gt_resized = gt_pil.resize((224, 224), Image.NEAREST)
            gt_np = np.array(gt_resized, dtype=np.float32)

            inputs = proc_da(images=raw_img, return_tensors="pt").to(device)
            outputs = model_da(**inputs)
            pred_disp = F.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            ).squeeze().cpu().numpy()

            valid = (gt_np > 0.1) & (gt_np < 80.0) & np.isfinite(gt_np) & (pred_disp > 0)
            if valid.sum() > 0:
                g_depth = gt_np[valid]
                g_disp = 1.0 / g_depth
                d = pred_disp[valid]

                # Protocol A: MiDaS Scale-and-Shift (Least squares affine in disparity space)
                A = np.vstack([d, np.ones_like(d)]).T
                s, t = np.linalg.lstsq(A, g_disp, rcond=None)[0]
                aligned_disp = np.maximum(s * d + t, 1e-4)
                aligned_depth = 1.0 / aligned_disp
                
                full_aligned_depth = np.zeros_like(gt_np)
                full_aligned_depth[valid] = aligned_depth
                m_list_da_midas.append(compute_metrics(full_aligned_depth, gt_np))

                # Protocol B: Median-Scaling (Standard Monocular Benchmark)
                inv_d = 1.0 / np.maximum(d, 1e-3)
                scale_med = np.median(g_depth) / np.median(inv_d)
                med_depth = inv_d * scale_med
                full_med_depth = np.zeros_like(gt_np)
                full_med_depth[valid] = med_depth
                m_list_da_median.append(compute_metrics(full_med_depth, gt_np))

    if device.type == "mps":
        torch.mps.synchronize()
    total_time_da = time.perf_counter() - t0
    fps_da = len(pairs) / total_time_da
    latency_da = (total_time_da / len(pairs)) * 1000.0

    results["depth_anything_v2_midas"] = {
        "name": "Depth Anything V2-Small (MiDaS Affine)",
        "backbone": "DINOv2-Small (Pretrained Foundation)",
        "params": p_count_da,
        "abs_rel": float(np.mean([m["abs_rel"] for m in m_list_da_midas])),
        "sq_rel": float(np.mean([m["sq_rel"] for m in m_list_da_midas])),
        "rmse": float(np.mean([m["rmse"] for m in m_list_da_midas])),
        "rmse_log": float(np.mean([m["rmse_log"] for m in m_list_da_midas])),
        "delta1": float(np.mean([m["a1"] for m in m_list_da_midas]) * 100),
        "delta2": float(np.mean([m["a2"] for m in m_list_da_midas]) * 100),
        "delta3": float(np.mean([m["a3"] for m in m_list_da_midas]) * 100),
        "scale_ratio": float(np.mean([m["scale_ratio"] for m in m_list_da_midas])),
        "latency_ms": latency_da,
        "fps": fps_da,
    }

    results["depth_anything_v2_median"] = {
        "name": "Depth Anything V2-Small (Median-Scaled)",
        "backbone": "DINOv2-Small (Pretrained Foundation)",
        "params": p_count_da,
        "abs_rel": float(np.mean([m["abs_rel"] for m in m_list_da_median])),
        "sq_rel": float(np.mean([m["sq_rel"] for m in m_list_da_median])),
        "rmse": float(np.mean([m["rmse"] for m in m_list_da_median])),
        "rmse_log": float(np.mean([m["rmse_log"] for m in m_list_da_median])),
        "delta1": float(np.mean([m["a1"] for m in m_list_da_median]) * 100),
        "delta2": float(np.mean([m["a2"] for m in m_list_da_median]) * 100),
        "delta3": float(np.mean([m["a3"] for m in m_list_da_median]) * 100),
        "scale_ratio": float(np.mean([m["scale_ratio"] for m in m_list_da_median])),
        "latency_ms": latency_da,
        "fps": fps_da,
    }

    print(f"DA-V2 (MiDaS Affine):  AbsRel={results['depth_anything_v2_midas']['abs_rel']:.4f}, RMSE={results['depth_anything_v2_midas']['rmse']:.4f}, d1={results['depth_anything_v2_midas']['delta1']:.2f}%, Latency={latency_da:.1f}ms ({fps_da:.1f} FPS)")
    print(f"DA-V2 (Median-Scaled): AbsRel={results['depth_anything_v2_median']['abs_rel']:.4f}, RMSE={results['depth_anything_v2_median']['rmse']:.4f}, d1={results['depth_anything_v2_median']['delta1']:.2f}%, Latency={latency_da:.1f}ms ({fps_da:.1f} FPS)")

    # -------------------------------------------------------------
    # Save Results & Print Summary Table
    # -------------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    out_json = "outputs/external_baseline_benchmark_200.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Benchmark] Results saved to: {out_json}")

    print("\n" + "="*105)
    print(f"{'Model':<38} | {'Params':<8} | {'AbsRel ↓':<8} | {'RMSE ↓':<8} | {'δ1 ↑':<7} | {'Scale':<6} | {'FPS (MPS)':<8}")
    print("="*105)
    for k, v in results.items():
        p_str = f"{v['params']/1e6:.2f}M"
        print(f"{v['name']:<38} | {p_str:<8} | {v['abs_rel']:<8.4f} | {v['rmse']:<8.4f} | {v['delta1']:<6.2f}% | {v['scale_ratio']:<6.4f} | {v['fps']:<8.1f}")
    print("="*105)


if __name__ == "__main__":
    main()
