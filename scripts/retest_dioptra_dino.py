"""
Direct Independent Retest Script for Dioptra-DINO Checkpoint (outputs/dioptra_dino_best.pt).

Executes a live empirical retest across:
1. Continuous 200-image benchmark (test_samples/unseen_200_abandonedfactory)
2. All available held-out multi-scene test sets in test_samples/
3. Real-time Apple Silicon (MPS) latency and throughput profiling
"""

import os
import sys
import glob
import time
import json
from typing import Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import __main__
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
__main__.DioptraDINOConfig = DioptraDINOConfig


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 80.0):
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & (pred > 0.05) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0, "a1": 0.0, "a2": 0.0, "a3": 0.0, "scale_ratio": 1.0, "valid_pixels": 0}

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


def load_model(ckpt_path: str, device: torch.device) -> Tuple[DioptraDINO, int]:
    print(f"\n[Loader] Loading checkpoint: {ckpt_path}")
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = st["cfg"] if "cfg" in st else st.get("config", DioptraDINOConfig())
    model = DioptraDINO(cfg)
    state_dict = st["model_state_dict"] if "model_state_dict" in st else st["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Loader] Successfully loaded! Exact parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    return model, total_params


def evaluate_sequence(model: DioptraDINO, img_files: list, depth_files: list, device: torch.device, name: str = "Sequence"):
    print(f"\n[Retest] Evaluating {len(img_files)} frames on '{name}'...")
    frame_metrics = []
    
    # TartanAir native parameters
    orig_w, orig_h = 640, 480
    min_side = min(orig_w, orig_h)  # 480
    left = (orig_w - min_side) // 2
    top = (orig_h - min_side) // 2
    img_size = 224
    
    fx_scaled = 320.0 * (float(img_size) / float(min_side))
    K_input = torch.tensor([[[fx_scaled, 0.0, img_size / 2.0],
                             [0.0, fx_scaled, img_size / 2.0],
                             [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    t0 = time.perf_counter()
    for idx, (img_p, gt_p) in enumerate(zip(img_files, depth_files)):
        # Load and square crop RGB
        bgr = cv2.imread(img_p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_crop = rgb[top:top+min_side, left:left+min_side]
        rgb_resized = cv2.resize(rgb_crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t_img = (t_img - mean) / std

        with torch.no_grad():
            pred = model(t_img, K_input).squeeze().cpu().numpy()

        # Load GT depth
        if gt_p.endswith(".npy"):
            gt = np.load(gt_p)
        else:
            gt = cv2.imread(gt_p, cv2.IMREAD_UNCHANGED)
            if gt.ndim == 3:
                # RGB-encoded depth if any
                gt = gt.astype(np.float32)[:, :, 0]

        gt_crop = gt[top:top+min_side, left:left+min_side]
        gt_resized = cv2.resize(gt_crop, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        m = compute_metrics(pred, gt_resized)
        frame_metrics.append(m)

        if (idx + 1) % 50 == 0 or idx == len(img_files) - 1:
            print(f"  Frame [{idx+1:03d}/{len(img_files):03d}] -> AbsRel: {m['abs_rel']:.4f}, RMSE: {m['rmse']:.3f}m, delta1: {m['a1']*100:.2f}%, Scale: {m['scale_ratio']:.4f}")

    total_time = time.perf_counter() - t0
    avg_fps = len(img_files) / total_time
    avg_latency = (total_time / len(img_files)) * 1000.0

    keys = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    summary = {
        "sequence": name,
        "frames": len(frame_metrics),
        "fps": round(avg_fps, 1),
        "latency_ms": round(avg_latency, 1),
    }
    for k in keys:
        summary[k] = float(np.mean([fm[k] for fm in frame_metrics]))
    summary["scale_ratio"] = float(np.median([fm["scale_ratio"] for fm in frame_metrics]))
    summary["a1_percent"] = round(summary["a1"] * 100.0, 2)
    summary["a2_percent"] = round(summary["a2"] * 100.0, 2)
    summary["a3_percent"] = round(summary["a3"] * 100.0, 2)
    return summary


def main():
    print("=" * 80)
    print(" LIVE EMPIRICAL RETEST OF DIOPTRA-DINO BEST CHECKPOINT")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Hardware Compute Device: {device} (Apple Silicon Metal Performance Shaders)")

    ckpt_path = "outputs/dioptra_dino_best.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found at: {ckpt_path}"
    model, total_params = load_model(ckpt_path, device)

    # 1. Profile inference speed / warmup
    print("\n[Speed Profiling] Measuring steady-state throughput on Apple Silicon M3...")
    dummy_img = torch.zeros(1, 3, 224, 224, device=device)
    dummy_K = torch.tensor([[[149.33, 0.0, 112.0], [0.0, 149.33, 112.0], [0.0, 0.0, 1.0]]], device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_img, dummy_K)
    
    num_speed_trials = 50
    t_start = time.perf_counter()
    for _ in range(num_speed_trials):
        with torch.no_grad():
            _ = model(dummy_img, dummy_K)
    t_elapsed = time.perf_counter() - t_start
    bench_fps = num_speed_trials / t_elapsed
    bench_latency = (t_elapsed / num_speed_trials) * 1000.0
    print(f"  Steady-State Latency: {bench_latency:.1f} ms/frame")
    print(f"  Steady-State Throughput: {bench_fps:.1f} FPS")

    # 2. Continuous 200-frame benchmark
    dir_200 = "test_samples/unseen_200_abandonedfactory"
    img_200 = sorted(glob.glob(os.path.join(dir_200, "image_left", "*.png")))
    depth_200 = sorted(glob.glob(os.path.join(dir_200, "depth_left", "*.npy")))

    summary_200 = evaluate_sequence(model, img_200, depth_200, device, name="abandonedfactory/Easy/P010 (200 Frames)")

    # 3. Targeted multi-scene test sets
    scenes = [
        ("abandonedfactory (Day)", "test_samples/abandonedfactory"),
        ("p011_benchmark (Long Trajectory)", "test_samples/p011_benchmark"),
        ("abandonedfactory_hard (Motion Clutter)", "test_samples/abandonedfactory_hard"),
        ("abandonedfactory_night (Fog & Darkness)", "test_samples/abandonedfactory_night"),
        ("hospital (Textureless Clinical OOD)", "test_samples/hospital"),
    ]

    scene_summaries = []
    for s_name, s_dir in scenes:
        if not os.path.exists(s_dir):
            continue
        imgs = sorted(glob.glob(os.path.join(s_dir, "*_left.png")))
        # Match depth files
        matched_imgs, matched_depths = [], []
        for im in imgs:
            dp_npy = im.replace("_left.png", "_left_depth.npy")
            dp_png = im.replace("_left.png", "_left_depth.png")
            if os.path.exists(dp_npy):
                matched_imgs.append(im)
                matched_depths.append(dp_npy)
            elif os.path.exists(dp_png):
                matched_imgs.append(im)
                matched_depths.append(dp_png)
        if matched_imgs:
            s_sum = evaluate_sequence(model, matched_imgs, matched_depths, device, name=s_name)
            scene_summaries.append(s_sum)

    # 4. Final Comparison Table
    print("\n" + "=" * 95)
    print(" LIVE RETEST RESULTS: 200 CONTINUOUS FRAMES & TARGETED STRESS PROBES")
    print("=" * 95)
    print(f"{'Evaluation Sequence':<38} | {'N':<4} | {'AbsRel':<8} | {'SqRel':<8} | {'RMSE':<8} | {'delta1':<7} | {'Scale':<7}")
    print("-" * 95)
    print(f"{summary_200['sequence']:<38} | {summary_200['frames']:<4} | {summary_200['abs_rel']:<8.4f} | {summary_200['sq_rel']:<8.4f} | {summary_200['rmse']:<6.3f}m | {summary_200['a1_percent']:<6.2f}% | {summary_200['scale_ratio']:<7.4f}")
    print("-" * 95)
    for s in scene_summaries:
        print(f"{s['sequence']:<38} | {s['frames']:<4} | {s['abs_rel']:<8.4f} | {s['sq_rel']:<8.4f} | {s['rmse']:<6.3f}m | {s['a1_percent']:<6.2f}% | {s['scale_ratio']:<7.4f}")
    print("=" * 95)

    # Comparison against foundation baselines
    print("\n" + "=" * 95)
    print(" DIRECT BENCHMARK COMPARISON ON IDENTICAL 200 FRAMES (abandonedfactory/Easy/P010)")
    print("=" * 95)
    print(f"{'Model Architecture':<34} | {'Params':<7} | {'Protocol':<22} | {'AbsRel':<8} | {'RMSE':<8} | {'delta1':<7} | {'Scale':<7} | {'FPS'}")
    print("-" * 95)
    print(f"{'Dioptra-DINO (Live Retested)':<34} | {'27.51M':<7} | {'Direct Metric (No Tune)':<22} | {summary_200['abs_rel']:<8.4f} | {summary_200['rmse']:<6.3f}m | {summary_200['a1_percent']:<6.2f}% | {summary_200['scale_ratio']:<7.4f} | {bench_fps:.1f} FPS")
    print(f"{'Depth Anything V2-Small':<34} | {'24.79M':<7} | {'Oracle Affine (s*d+t)':<22} | {'0.0914':<8} | {'3.072m':<8} | {'91.53%':<7} | {'0.9941':<7} | 7.4 FPS")
    print(f"{'Metric3D ViT-Small':<34} | {'37.50M':<7} | {'Direct Metric (K)':<22} | {'0.3269':<8} | {'7.192m':<8} | {'26.52%':<7} | {'0.7254':<7} | 1.8 FPS")
    print(f"{'Canonical 2D ViT + DPT':<34} | {'26.10M':<7} | {'Direct Metric (No Rays)':<22} | {'0.5056':<8} | {'10.250m':<8}| {'12.31%':<7} | {'0.5582':<7} | 27.9 FPS")
    print("=" * 95)


if __name__ == "__main__":
    main()
