#!/usr/bin/env python3
"""
Benchmark Dioptra-DINO (Checkpoint Step 159,000 / Latest) exclusively on Unseen Indoor Environments
with Reason A (Camera Intrinsics Disconnect & K-scaling) Fixed.
"""

import os
import sys
import glob
import time
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dioptra_dino
sys.modules["__main__"].DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINO, DioptraDINOConfig
from benchmark_all_indoor_competitors import depth_to_surface_normals

# Canonical Native Camera Intrinsics
K_TARTAN_V1 = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_TARTAN_V2 = np.array([
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


def compute_metrics(pred: np.ndarray, gt: np.ndarray, K: np.ndarray, min_depth: float = 0.1, max_depth: float = 50.0):
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & np.isfinite(pred) & (gt > 0)
    if np.sum(mask) < 20:
        return {}

    p = np.clip(pred[mask], min_depth, max_depth)
    g = gt[mask]

    # 1. Raw Absolute Metric
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

    # 2. Scale-Aligned (Median Scaled) Metric
    s = np.median(g) / (np.median(p) + 1e-6)
    p_aligned = p * s
    abs_rel_aligned = float(np.mean(np.abs(p_aligned - g) / g))
    rmse_aligned = float(np.sqrt(np.mean((p_aligned - g) ** 2)))
    ratio_aligned = np.maximum(p_aligned / g, g / p_aligned)
    delta1_aligned = float(np.mean(ratio_aligned < 1.25))

    # Surface Normal MAE
    normal_mae = float("nan")
    try:
        norm_gt = depth_to_surface_normals(gt, K)
        norm_pred = depth_to_surface_normals(pred, K)
        dot = np.clip(np.sum(norm_gt * norm_pred, axis=-1), -1.0, 1.0)
        angles_deg = np.arccos(dot) * (180.0 / math.pi)
        valid_angles = angles_deg[mask]
        valid_angles = valid_angles[np.isfinite(valid_angles)]
        if len(valid_angles) > 10:
            normal_mae = float(np.mean(valid_angles))
    except Exception:
        pass

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "mae": mae,
        "silog": silog,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
        "scale_ratio": scale_ratio,
        "abs_rel_aligned": abs_rel_aligned,
        "rmse_aligned": rmse_aligned,
        "delta1_aligned": delta1_aligned,
        "normal_mae": normal_mae,
    }


def define_unseen_indoor_suite():
    return [
        # TartanAir v2 Photorealistic Indoor Suites (640x640 Square Sensors)
        {
            "id": "tartanair2_retrooffice",
            "name": "Retro Office / P000",
            "category": "Commercial Retro",
            "img_pattern": "test_samples/indoor_suite/tartanair2_retrooffice/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_retrooffice/depth_left",
            "K_native": K_TARTAN_V2,
            "max_eval_depth": 25.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_house",
            "name": "Suburban House / P000",
            "category": "Residential Multi-Room",
            "img_pattern": "test_samples/indoor_suite/tartanair2_house/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_house/depth_left",
            "K_native": K_TARTAN_V2,
            "max_eval_depth": 25.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_americandiner",
            "name": "American Diner / P000",
            "category": "Restaurant Interior",
            "img_pattern": "test_samples/indoor_suite/tartanair2_americandiner/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_americandiner/depth_left",
            "K_native": K_TARTAN_V2,
            "max_eval_depth": 30.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_supermarket",
            "name": "Supermarket / P000",
            "category": "Retail Grocery",
            "img_pattern": "test_samples/indoor_suite/tartanair2_supermarket/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_supermarket/depth_left",
            "K_native": K_TARTAN_V2,
            "max_eval_depth": 35.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_prison",
            "name": "Prison / P000",
            "category": "Institutional Cells",
            "img_pattern": "test_samples/indoor_suite/tartanair2_prison/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_prison/depth_left",
            "K_native": K_TARTAN_V2,
            "max_eval_depth": 30.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_archviztinyhouseday",
            "name": "Tiny House Day / P000",
            "category": "Residential Architectural",
            "img_pattern": "test_samples/indoor_suite/tartanair2_archviztinyhouseday/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_archviztinyhouseday/depth_left",
            "K_native": K_TARTAN_V2,
            "max_eval_depth": 25.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_archviztinyhousenight",
            "name": "Tiny House Night / P000",
            "category": "Residential Night",
            "img_pattern": "test_samples/indoor_suite/tartanair2_archviztinyhousenight/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_archviztinyhousenight/depth_left",
            "K_native": K_TARTAN_V2,
            "max_eval_depth": 25.0,
            "max_frames": 50,
        },
        # TartanAir v1 Indoor Environments (640x480 Rectangular Sensors)
        {
            "id": "unseen_hospital_p001",
            "name": "Hospital / P001",
            "category": "Medical Facility",
            "img_pattern": "test_samples/unseen_hospital_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_hospital_p001/depth_left",
            "K_native": K_TARTAN_V1,
            "max_eval_depth": 50.0,
            "max_frames": 50,
        },
        {
            "id": "unseen_office_p001",
            "name": "Office / P001",
            "category": "Commercial Office",
            "img_pattern": "test_samples/unseen_office_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_office_p001/depth_left",
            "K_native": K_TARTAN_V1,
            "max_eval_depth": 50.0,
            "max_frames": 30,
        },
        {
            "id": "unseen_office_p002",
            "name": "Office / P002",
            "category": "Commercial Office",
            "img_pattern": "test_samples/unseen_office_p002/image_left/*.png",
            "depth_dir": "test_samples/unseen_office_p002/depth_left",
            "K_native": K_TARTAN_V1,
            "max_eval_depth": 50.0,
            "max_frames": 30,
        },
        {
            "id": "unseen_office2_p000",
            "name": "Office2 / P000",
            "category": "Executive Workspace",
            "img_pattern": "test_samples/unseen_office2_p000/image_left/*.png",
            "depth_dir": "test_samples/unseen_office2_p000/depth_left",
            "K_native": K_TARTAN_V1,
            "max_eval_depth": 50.0,
            "max_frames": 50,
        },
        # Real-World Physical Sensor Suites
        {
            "id": "nyu_depth_v2",
            "name": "NYU-Depth V2 (Official Test)",
            "category": "Real Kinect RGB-D",
            "img_pattern": "test_samples/indoor_suite/nyu_depth_v2/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/nyu_depth_v2/depth_left",
            "K_native": K_NYU_V2,
            "max_eval_depth": 10.0,
            "max_frames": 50,
        },
        {
            "id": "scannet_scene00",
            "name": "ScanNet (scene00)",
            "category": "Real Handheld RGB-D",
            "img_pattern": "data/scannet_tiny/scene00/color/*.jpg",
            "depth_dir": "data/scannet_tiny/scene00/depth",
            "K_native": K_SCANNET,
            "max_eval_depth": 10.0,
            "max_frames": 10,
        },
    ]


def load_gt_depth(depth_dir: str, stem: str, is_scannet: bool = False) -> np.ndarray:
    if is_scannet:
        dp = os.path.join(depth_dir, f"{stem}.png")
        if os.path.exists(dp):
            raw = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
            return raw.astype(np.float32) / 1000.0  # mm to meters

    candidates = [
        f"{stem}_depth.npy",
        f"{stem}.npy",
        f"{stem}_left_depth.npy",
        f"{stem}_lcam_front_depth.png",
        f"{stem}.png",
    ]
    for c in candidates:
        dp = os.path.join(depth_dir, c)
        if os.path.exists(dp):
            if dp.endswith(".npy"):
                return np.load(dp).astype(np.float32)
            else:
                raw = Image.open(dp)
                arr = np.array(raw).astype(np.float32)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    return np.ascontiguousarray(arr).view(np.float32).squeeze(-1)
                elif arr.ndim == 3:
                    return arr[..., 0]
                return arr
    return None


def run_benchmark():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[{time.strftime('%X')}] Initializing Dioptra-DINO on {device}...")

    # Load latest checkpoint
    ckpt_paths = [
        os.path.expanduser("~/Downloads/checkpoint_step_latest (2).zip"),
        "staging_harryson/checkpoint_step_latest.pt",
        "extracted_checkpoint/checkpoint_step_latest.pt",
    ]
    ckpt_path = next((p for p in ckpt_paths if os.path.exists(p)), None)
    if not ckpt_path:
        raise FileNotFoundError("Could not find downloaded checkpoint!")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", DioptraDINOConfig(image_size=336, grid_size=24))
    model = DioptraDINO(cfg)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    cleaned = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device).eval()
    print("Model successfully loaded and initialized ✓")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    suite = define_unseen_indoor_suite()
    results = {}

    print("\n" + "=" * 95)
    print("DIOPTRA-DINO INDOOR-ONLY EVALUATION SUITE (REASON A FIXED: CAMERA INTRINSICS & K-SCALING)")
    print("=" * 95)

    for env in suite:
        env_id = env["id"]
        name = env["name"]
        cat = env["category"]
        K_native = env["K_native"]
        max_d = env["max_eval_depth"]
        max_f = env.get("max_frames", 50)
        is_scannet = "scannet" in env_id

        imgs = sorted(glob.glob(env["img_pattern"]))
        if not imgs:
            print(f"⚠️  No images found for {name} ({env['img_pattern']})")
            continue
        if max_f and len(imgs) > max_f:
            imgs = imgs[:max_f]

        env_metrics = []
        latencies = []

        for ip in imgs:
            stem = os.path.splitext(os.path.basename(ip))[0]
            gt = load_gt_depth(env["depth_dir"], stem, is_scannet=is_scannet)
            if gt is None:
                continue

            pil = Image.open(ip).convert("RGB")
            w, h = pil.size

            # Network input resolution: 336 x 336
            img_res = pil.resize((336, 336), Image.BILINEAR)
            arr = (np.array(img_res, dtype=np.float32) / 255.0 - mean) / std
            t_img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # REASON A FIX: Exact per-domain camera intrinsic scaling to 336x336
            K_scaled = scale_intrinsics(K_native, orig_w=float(w), orig_h=float(h), new_w=336.0, new_h=336.0)
            t_K = torch.from_numpy(K_scaled).unsqueeze(0).float().to(device)

            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                pred_res = model(t_img, t_K, ara_gate=1.0)
                pred_full = F.interpolate(pred_res, size=(h, w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

            if device == "mps":
                torch.mps.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0

            m = compute_metrics(pred_full, gt, K=K_native, min_depth=0.1, max_depth=max_d)
            if m:
                env_metrics.append(m)
                latencies.append(dt_ms)

        if not env_metrics:
            continue

        agg = {}
        for k in env_metrics[0].keys():
            vals = [x[k] for x in env_metrics if not math.isnan(x[k])]
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg["latency_ms"] = float(np.mean(latencies))
        agg["frames"] = len(env_metrics)
        agg["name"] = name
        agg["category"] = cat
        results[env_id] = agg

        print(f"✓ {name:30s} [{agg['frames']:2d} frames] | AbsRel: {agg['abs_rel']:.4f} | RMSE: {agg['rmse']:.3f}m | δ1: {agg['delta1']*100:.1f}% | Scale: {agg['scale_ratio']:.3f} | Rel AbsRel: {agg['abs_rel_aligned']:.4f}")

    # Summary table
    print("\n" + "=" * 95)
    print("FINAL INDOOR RESULTS SUMMARY TABLE (REASON A FIXED)")
    print("=" * 95)
    header = f"{'Environment':<30} | {'Category':<22} | {'AbsRel':<8} | {'RMSE (m)':<9} | {'δ1 < 1.25':<9} | {'ScaleRatio':<10} | {'Rel AbsRel':<10}"
    print(header)
    print("-" * len(header))

    macro_absrel = np.mean([r["abs_rel"] for r in results.values()])
    macro_rmse = np.mean([r["rmse"] for r in results.values()])
    macro_delta1 = np.mean([r["delta1"] for r in results.values()])
    macro_scale = np.mean([r["scale_ratio"] for r in results.values()])
    macro_rel_absrel = np.mean([r["abs_rel_aligned"] for r in results.values()])

    for env_id, r in results.items():
        print(f"{r['name']:<30} | {r['category']:<22} | {r['abs_rel']:<8.4f} | {r['rmse']:<9.3f} | {r['delta1']*100:<8.1f}% | {r['scale_ratio']:<10.3f} | {r['abs_rel_aligned']:<10.4f}")

    print("-" * len(header))
    print(f"{'MACRO-AVERAGE (All Indoors)':<30} | {'ALL 13 INDOOR SUITES':<22} | {macro_absrel:<8.4f} | {macro_rmse:<9.3f} | {macro_delta1*100:<8.1f}% | {macro_scale:<10.3f} | {macro_rel_absrel:<10.4f}")
    print("=" * 95)

    # Save to Markdown
    md_path = "mac_outputs/indoor_benchmark_reason_a_fixed.md"
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("# Dioptra-DINO: Indoor-Only Benchmark Report (Reason A Fixed)\n\n")
        f.write("**Model:** Dioptra-DINO Checkpoint Step 159,000 (Epoch 1 @ 54% schedule)\n")
        f.write(f"**Evaluated on:** Apple Silicon GPU (`{device}`) across {len(results)} Unseen Indoor Environments ({sum(r['frames'] for r in results.values())} frames)\n\n")
        f.write("### 1. Macro Summary\n\n")
        f.write("| Metric | Raw Absolute Metric | Scale-Aligned / Structural Metric |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Macro AbsRel ↓** | **{macro_absrel:.4f}** | **{macro_rel_absrel:.4f}** |\n")
        f.write(f"| **Macro RMSE ↓** | **{macro_rmse:.3f} m** | **{np.mean([r['rmse_aligned'] for r in results.values()]):.3f} m** |\n")
        f.write(f"| **Macro $\\delta_1 < 1.25$ ↑** | **{macro_delta1*100:.1f}%** | **{np.mean([r['delta1_aligned'] for r in results.values()])*100:.1f}%** |\n")
        f.write(f"| **Macro Scale Ratio** | **{macro_scale:.3f}** | **1.000** |\n\n")

        f.write("### 2. Per-Environment Detailed Breakdown\n\n")
        f.write("| Environment | Category | Frames | Raw AbsRel ↓ | Raw RMSE (m) ↓ | Raw $\\delta_1$ ↑ | Scale Ratio | Rel AbsRel ↓ | Normal MAE (°) ↓ |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for env_id, r in results.items():
            norm_str = f"{r['normal_mae']:.1f}°" if not math.isnan(r['normal_mae']) else "N/A"
            f.write(f"| {r['name']} | {r['category']} | {r['frames']} | **{r['abs_rel']:.4f}** | {r['rmse']:.3f} m | {r['delta1']*100:.1f}% | {r['scale_ratio']:.3f} | {r['abs_rel_aligned']:.4f} | {norm_str} |\n")

    print(f"\nSaved Markdown report to: {md_path}")


if __name__ == "__main__":
    run_benchmark()
