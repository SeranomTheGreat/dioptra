"""
Extensive Evaluation Suite for Dioptra-DINO Research Paper.
Runs:
1. Environment-by-environment full benchmark breakdown (raw metric + aligned).
2. Multi-Scene Multi-FOV equivariance matrix across 5 distinct scenes x 6 FOVs.
3. Depth range stratification (Near [0.1-5m], Mid [5-15m], Far [15-30m], Horizon [30-80m]).
4. ARA Attention ablation (ara_gate=1.0 vs ara_gate=0.0).
5. 3D Surface Normal angular error (in degrees) to quantify VNL performance.
6. Latency and FPS hardware profiling.
"""

import os
import sys
import glob
import time
import json
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
from scripts.eval_dino import load_model, compute_metrics, preprocess_sample


def get_all_pairs():
    images = sorted(glob.glob("test_samples/**/*.png", recursive=True))
    pairs = []
    for img in images:
        base, _ = os.path.splitext(img)
        for ext in ["_depth.npy", ".npy"]:
            cand = base + ext
            if os.path.exists(cand):
                pairs.append((img, cand))
                break
    return pairs


def compute_aligned_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 80.0):
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & (pred > min_depth) & (pred < max_depth) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"abs_rel": 0.0, "rmse": 0.0, "a1": 0.0, "scale_factor": 1.0}
    p = pred[mask]
    g = gt[mask]
    scale = np.median(g) / (np.median(p) + 1e-8)
    p_aligned = p * scale
    thresh = np.maximum((g / p_aligned), (p_aligned / g))
    a1 = float((thresh < 1.25).mean())
    rmse = float(np.sqrt(((g - p_aligned) ** 2).mean()))
    abs_rel = float((np.abs(g - p_aligned) / g).mean())
    return {"abs_rel": abs_rel, "rmse": rmse, "a1": a1, "scale_factor": float(scale)}


def compute_surface_normal_error(pred_t: torch.Tensor, gt_t: torch.Tensor, K: torch.Tensor, min_depth=0.1, max_depth=80.0):
    """Compute mean 3D normal angular error in degrees across stencil s=2."""
    B, _, H, W = pred_t.shape
    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=pred_t.device, dtype=torch.float32),
        torch.arange(W, device=pred_t.device, dtype=torch.float32),
        indexing="ij"
    )
    x_grid = x_grid.unsqueeze(0)
    y_grid = y_grid.unsqueeze(0)

    s = 2
    pts_pred = torch.cat([(x_grid - cx) / fx * pred_t, (y_grid - cy) / fy * pred_t, pred_t], dim=1)
    pts_gt = torch.cat([(x_grid - cx) / fx * gt_t, (y_grid - cy) / fy * gt_t, gt_t], dim=1)

    vx_pred = pts_pred[:, :, s:-s, 2*s:] - pts_pred[:, :, s:-s, :-2*s]
    vy_pred = pts_pred[:, :, 2*s:, s:-s] - pts_pred[:, :, :-2*s, s:-s]
    n_pred = torch.cross(vx_pred, vy_pred, dim=1)
    n_pred = n_pred / torch.norm(n_pred, dim=1, keepdim=True).clamp(min=1e-6)

    vx_gt = pts_gt[:, :, s:-s, 2*s:] - pts_gt[:, :, s:-s, :-2*s]
    vy_gt = pts_gt[:, :, 2*s:, s:-s] - pts_gt[:, :, :-2*s, s:-s]
    n_gt = torch.cross(vx_gt, vy_gt, dim=1)
    n_gt = n_gt / torch.norm(n_gt, dim=1, keepdim=True).clamp(min=1e-6)

    cos_sim = torch.sum(n_pred * n_gt, dim=1).clamp(min=-1.0, max=1.0)
    angle_deg = torch.rad2deg(torch.acos(cos_sim))

    gt_crop = gt_t[:, :, s:-s, s:-s].squeeze(1)
    mask = (gt_crop > min_depth) & (gt_crop < max_depth) & torch.isfinite(gt_crop)
    if mask.sum() == 0:
        return 0.0
    return float(angle_deg[mask].mean().item())


def run_benchmarks(checkpoint_path: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path} on {device}")
    model = load_model(checkpoint_path, device=device)
    pairs = get_all_pairs()
    print(f"Loaded {len(pairs)} test pairs.\n")

    # -------------------------------------------------------------
    # Suite 1: Environment-by-Environment Breakdown
    # -------------------------------------------------------------
    print("=" * 105)
    print("SUITE 1: ENVIRONMENT-BY-ENVIRONMENT BREAKDOWN (RAW METRIC & ALIGNED)")
    print("=" * 105)
    by_env = {}
    for img_p, gt_p in pairs:
        env = os.path.basename(os.path.dirname(img_p))
        by_env.setdefault(env, []).append((img_p, gt_p))

    header = f"{'Environment':<28} | {'N':<3} | {'Raw AbsRel':<10} | {'Raw RMSE':<9} | {'Raw δ1':<7} | {'Raw δ2':<7} | {'Scale':<7} | {'Ali AbsRel':<10} | {'Ali δ1':<7}"
    print(header)
    print("-" * 105)

    all_raw_m = []
    all_ali_m = []

    for env, env_pairs in sorted(by_env.items()):
        raw_metrics = []
        ali_metrics = []
        for img_p, gt_p in env_pairs:
            inp, gt, _, K, _ = preprocess_sample(img_p, gt_p, device=device)
            with torch.no_grad():
                pred = model(inp, K, ara_gate=1.0)
                pred_np = pred.squeeze().cpu().numpy()
            m_raw = compute_metrics(pred_np, gt)
            m_ali = compute_aligned_metrics(pred_np, gt)
            raw_metrics.append(m_raw)
            ali_metrics.append(m_ali)
            all_raw_m.append(m_raw)
            all_ali_m.append(m_ali)

        r_ar = np.mean([m["abs_rel"] for m in raw_metrics])
        r_rmse = np.mean([m["rmse"] for m in raw_metrics])
        r_d1 = np.mean([m["a1"] * 100 for m in raw_metrics])
        r_d2 = np.mean([m["a2"] * 100 for m in raw_metrics])
        r_sc = np.mean([m["scale_ratio"] for m in raw_metrics])
        a_ar = np.mean([m["abs_rel"] for m in ali_metrics])
        a_d1 = np.mean([m["a1"] * 100 for m in ali_metrics])

        print(f"{env:<28} | {len(env_pairs):<3} | {r_ar:<10.4f} | {r_rmse:<7.2f} m | {r_d1:<6.1f}% | {r_d2:<6.1f}% | {r_sc:<7.3f} | {a_ar:<10.4f} | {a_d1:<6.1f}%")

    tot_r_ar = np.mean([m["abs_rel"] for m in all_raw_m])
    tot_r_rmse = np.mean([m["rmse"] for m in all_raw_m])
    tot_r_d1 = np.mean([m["a1"] * 100 for m in all_raw_m])
    tot_r_d2 = np.mean([m["a2"] * 100 for m in all_raw_m])
    tot_r_sc = np.mean([m["scale_ratio"] for m in all_raw_m])
    tot_a_ar = np.mean([m["abs_rel"] for m in all_ali_m])
    tot_a_d1 = np.mean([m["a1"] * 100 for m in all_ali_m])

    print("-" * 105)
    print(f"{'OVERALL AVERAGE (47 pairs)':<28} | {len(all_raw_m):<3} | {tot_r_ar:<10.4f} | {tot_r_rmse:<7.2f} m | {tot_r_d1:<6.1f}% | {tot_r_d2:<6.1f}% | {tot_r_sc:<7.3f} | {tot_a_ar:<10.4f} | {tot_a_d1:<6.1f}%")
    print("=" * 105)

    # -------------------------------------------------------------
    # Suite 2: Depth Range Stratification
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUITE 2: DEPTH RANGE STRATIFICATION (ERROR BY DISTANCE BAND)")
    print("=" * 80)
    bands = [
        ("Near Range", 0.1, 5.0),
        ("Mid Range", 5.0, 15.0),
        ("Far Range", 15.0, 30.0),
        ("Horizon Range", 30.0, 80.0),
    ]
    print(f"{'Distance Bracket':<22} | {'Depth Span':<14} | {'AbsRel (↓)':<12} | {'RMSE Metres (↓)':<16} | {'δ1 Acc (↑)':<12}")
    print("-" * 80)

    for b_name, d_min, d_max in bands:
        b_ar, b_rmse, b_d1 = [], [], []
        for img_p, gt_p in pairs:
            inp, gt, _, K, _ = preprocess_sample(img_p, gt_p, device=device)
            with torch.no_grad():
                pred = model(inp, K, ara_gate=1.0)
                pred_np = pred.squeeze().cpu().numpy()
            mask = (gt >= d_min) & (gt < d_max) & np.isfinite(gt) & (pred_np > 0.05) & np.isfinite(pred_np)
            if mask.sum() > 20:
                p_sub = pred_np[mask]
                g_sub = gt[mask]
                ar = float((np.abs(g_sub - p_sub) / g_sub).mean())
                rmse = float(np.sqrt(((g_sub - p_sub) ** 2).mean()))
                thresh = np.maximum((g_sub / p_sub), (p_sub / g_sub))
                d1 = float((thresh < 1.25).mean()) * 100.0
                b_ar.append(ar)
                b_rmse.append(rmse)
                b_d1.append(d1)
        mean_ar = np.mean(b_ar) if b_ar else 0.0
        mean_rmse = np.mean(b_rmse) if b_rmse else 0.0
        mean_d1 = np.mean(b_d1) if b_d1 else 0.0
        print(f"{b_name:<22} | [{d_min:>4.1f}, {d_max:>4.1f}] m | {mean_ar:<12.4f} | {mean_rmse:<14.2f} m | {mean_d1:<10.1f} %")
    print("=" * 80)

    # -------------------------------------------------------------
    # Suite 3: Multi-Scene Multi-FOV Equivariance Matrix
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUITE 3: MULTI-SCENE MULTI-FOV EQUIVARIANCE MATRIX (AbsRel across FOVs)")
    print("=" * 90)
    key_scenes = [
        ("Scene 1 (af/000100)", "test_samples/abandonedfactory/000100_left.png", "test_samples/abandonedfactory/000100_left_depth.npy"),
        ("Scene 2 (af/000300)", "test_samples/abandonedfactory/000300_left.png", "test_samples/abandonedfactory/000300_left_depth.npy"),
        ("Scene 3 (af/000700)", "test_samples/abandonedfactory/000700_left.png", "test_samples/abandonedfactory/000700_left_depth.npy"),
        ("Night (af_night/000100)", "test_samples/abandonedfactory_night/000100_left.png", "test_samples/abandonedfactory_night/000100_left_depth.npy"),
        ("Amusement (amus/000250)", "test_samples/amusement/000250_left.png", "test_samples/amusement/000250_left_depth.npy"),
    ]
    fovs = [50, 60, 73.74, 85, 90, 100]
    fov_header = f"{'Scene Identifier':<26} | " + " | ".join([f"{f:>5.1f}°" for f in fovs]) + " | Error Drop (50°->100°)"
    print(fov_header)
    print("-" * 90)

    for s_name, img_p, gt_p in key_scenes:
        if not os.path.exists(img_p) or not os.path.exists(gt_p):
            continue
        inp, gt, _, _, _ = preprocess_sample(img_p, gt_p, device=device)
        fov_ars = []
        for fov in fovs:
            fov_rad = np.radians(fov)
            fx_sim = float((224.0 / 2.0) / np.tan(fov_rad / 2.0))
            K_sim = torch.tensor([[[fx_sim, 0.0, 112.0],
                                   [0.0, fx_sim, 112.0],
                                   [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)
            with torch.no_grad():
                pred = model(inp, K_sim, ara_gate=1.0)
                pred_np = pred.squeeze().cpu().numpy()
            m = compute_metrics(pred_np, gt)
            fov_ars.append(m["abs_rel"])
        ar_str = " | ".join([f"{ar:>6.4f}" for ar in fov_ars])
        drop = ((fov_ars[-1] - fov_ars[0]) / fov_ars[0]) * 100.0
        print(f"{s_name:<26} | {ar_str} | {drop:>+.1f}%")
    print("=" * 90)

    # -------------------------------------------------------------
    # Suite 4: ARA Attention Ablation (ara_gate=1.0 vs 0.0)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUITE 4: ANGULAR RESIDUAL ATTENTION (ARA) MECHANISM ABLATION")
    print("=" * 70)
    ara_on_ar, ara_on_d1, ara_off_ar, ara_off_d1 = [], [], [], []
    for img_p, gt_p in pairs:
        inp, gt, _, K, _ = preprocess_sample(img_p, gt_p, device=device)
        with torch.no_grad():
            p_on = model(inp, K, ara_gate=1.0).squeeze().cpu().numpy()
            p_off = model(inp, K, ara_gate=0.0).squeeze().cpu().numpy()
        m_on = compute_metrics(p_on, gt)
        m_off = compute_metrics(p_off, gt)
        ara_on_ar.append(m_on["abs_rel"])
        ara_on_d1.append(m_on["a1"] * 100)
        ara_off_ar.append(m_off["abs_rel"])
        ara_off_d1.append(m_off["a1"] * 100)

    print(f"Active ARA (ara_gate = 1.0)  : Mean AbsRel = {np.mean(ara_on_ar):.4f} | Mean δ1 = {np.mean(ara_on_d1):.1f}%")
    print(f"Disabled ARA (ara_gate = 0.0): Mean AbsRel = {np.mean(ara_off_ar):.4f} | Mean δ1 = {np.mean(ara_off_d1):.1f}%")
    print(f"Relative AbsRel Improvement  : {((np.mean(ara_off_ar) - np.mean(ara_on_ar)) / np.mean(ara_off_ar))*100:+.2f}%")
    print("=" * 70)

    # -------------------------------------------------------------
    # Suite 5: 3D Surface Normal Angular Error (VNL Evaluation)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUITE 5: 3D SURFACE NORMAL ANGULAR ERROR (VNL PLANARITY PROOF)")
    print("=" * 70)
    normal_errors = []
    for img_p, gt_p in pairs:
        inp, gt, _, K, _ = preprocess_sample(img_p, gt_p, device=device)
        with torch.no_grad():
            pred = model(inp, K, ara_gate=1.0)
            gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
            err_deg = compute_surface_normal_error(pred, gt_t, K)
            if err_deg > 0.0:
                normal_errors.append(err_deg)

    print(f"Evaluated Test Scenes        : {len(normal_errors)} scenes")
    print(f"Mean 3D Normal Angular Error : {np.mean(normal_errors):.2f}°")
    print(f"Median 3D Normal Angle Error : {np.median(normal_errors):.2f}°")
    print(f"Best Scene 3D Normal Error   : {np.min(normal_errors):.2f}°")
    print("=" * 70)

    # -------------------------------------------------------------
    # Suite 6: Edge Hardware Inference Latency Benchmark
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUITE 6: REAL-TIME HARDWARE LATENCY & FPS PROFILING (100 ITERATIONS)")
    print("=" * 70)
    dummy_inp = torch.randn(1, 3, 224, 224, device=device)
    dummy_K = torch.tensor([[[149.33, 0.0, 112.0], [0.0, 149.33, 112.0], [0.0, 0.0, 1.0]]], device=device)

    # Warmup
    for _ in range(15):
        with torch.no_grad():
            _ = model(dummy_inp, dummy_K, ara_gate=1.0)

    timings = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_inp, dummy_K, ara_gate=1.0)
        timings.append((time.perf_counter() - t0) * 1000.0)

    print(f"Device Profiler              : {device}")
    print(f"Mean Inference Latency       : {np.mean(timings):.2f} ms")
    print(f"Median (P50) Latency         : {np.percentile(timings, 50):.2f} ms")
    print(f"90th Percentile (P90)        : {np.percentile(timings, 90):.2f} ms")
    print(f"99th Percentile (P99)        : {np.percentile(timings, 99):.2f} ms")
    print(f"Inference Throughput (FPS)   : {1000.0 / np.mean(timings):.1f} FPS")
    print("=" * 70)


if __name__ == "__main__":
    ckpt = "outputs_dino/dioptra_dino_epoch26_best.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    run_benchmarks(ckpt)
