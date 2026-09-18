"""
Comprehensive Fair-Ground Metric Depth Benchmark Suite.

Evaluates on the 200 continuous unseen held-out frames (TartanAir abandonedfactory/Easy/P010):
1. Dioptra-DINO (Ours, 27.51M params, Metric Camera-Ray Conditioning + ARA)
2. UniDepth-V2 ViT-Small (34.18M params, Pseudo-Spherical Ray Representation + Exact K)
   - Direct Metric (Camera-Conditioned)
3. Metric3D ViT-Small (37.50M params, Canonical Camera Transform + Exact K)
   - Direct Metric (Camera-Conditioned)
   - Oracle Median Scaling
4. Depth Anything V2-Small Metric-Outdoor (24.79M params, VKITTI2 Fine-Tuned)
   - Direct Metric
5. Depth Anything V2-Small Metric-Indoor (24.79M params, Hypersim Fine-Tuned)
   - Direct Metric
6. ZoeDepth ZoeD_NK (346.10M params, BEiT-L + Metric Bins)
   - Direct Metric
7. Depth Anything V2-Small Relative (24.79M params, DINOv2-Small)
   - MiDaS Disparity Affine Alignment (s * d + t)
   - Disparity Median Scaling

Runs on Apple Silicon M3 (MPS) with PyTorch 2.8.0.
"""

import os
import sys
import glob
import json
import time
from typing import Dict, List, Tuple, Optional
import cv2
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
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & (pred > 0.01) & np.isfinite(pred)
    if mask.sum() == 0:
        return {
            "abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0,
            "a1": 0.0, "a2": 0.0, "a3": 0.0, "scale_ratio": 1.0, "valid_pixels": 0
        }

    p = pred[mask]
    g = gt[mask]

    thresh = np.maximum((g / p), (p / g))
    a1 = float((thresh < 1.25).mean())
    a2 = float((thresh < 1.25 ** 2).mean())
    a3 = float((thresh < 1.25 ** 3).mean())

    rmse = float(np.sqrt(((g - p) ** 2).mean()))
    rmse_log = float(np.sqrt(((np.log(np.clip(g, 1e-3, None)) - np.log(np.clip(p, 1e-3, None))) ** 2).mean()))
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


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 85)
    print(" COMPREHENSIVE FAIR-GROUND METRIC DEPTH BENCHMARK SUITE")
    print(f" Execution Device: {device} (Apple M3 GPU)")
    print("=" * 85)

    # 1. Dataset verification
    data_dir = "test_samples/unseen_200_abandonedfactory"
    img_files = sorted(glob.glob(os.path.join(data_dir, "image_left", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(data_dir, "depth_left", "*.npy")))

    assert len(img_files) == 200, f"Expected 200 images, got {len(img_files)}"
    assert len(depth_files) == 200, f"Expected 200 depth files, got {len(depth_files)}"
    print(f"[Dataset] Verified 200 continuous frames from TartanAir AbandonedFactory P010.")

    # 2. Camera Intrinsics
    orig_W, orig_H = 640, 480
    orig_K = torch.tensor([
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    # Square pinhole crop for Dioptra-DINO: 480x480 -> 224x224
    crop_size = 224
    crop_fx = 320.0 * (crop_size / 480.0)  # 149.3333
    crop_fy = 320.0 * (crop_size / 480.0)
    crop_cx = (320.0 - 80.0) * (crop_size / 480.0)  # 112.0
    crop_cy = 240.0 * (crop_size / 480.0)  # 112.0
    dioptra_K = torch.tensor([
        [crop_fx, 0.0, crop_cx],
        [0.0, crop_fy, crop_cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device).unsqueeze(0)
    mean_dino = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_dino = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # =========================================================================
    # Model Loading
    # =========================================================================
    models = {}

    # (A) Dioptra-DINO
    print("\n[Loading 1/7] Dioptra-DINO (Ours, 27.51M params)...")
    dioptra_ckpt = "outputs/dioptra_dino_best.pt"
    cfg = DioptraDINOConfig(freeze_backbone=False)
    model_dioptra = DioptraDINO(cfg).to(device)
    st = torch.load(dioptra_ckpt, map_location=device, weights_only=False)
    sd = st.get("model_state_dict", st)
    clean_state = {k.replace("module.", ""): v for k, v in sd.items()}
    model_dioptra.load_state_dict(clean_state, strict=False)
    model_dioptra.eval()
    models["dioptra_dino"] = model_dioptra
    print(" -> Dioptra-DINO successfully initialized from outputs/dioptra_dino_best.pt.")

    # (B) UniDepth V2 ViT-Small
    print("\n[Loading 2/7] UniDepth V2 ViT-Small (34.18M params)...")
    model_unidepth = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version="v2", backbone="vits14", pretrained=True, trust_repo=True)
    model_unidepth.to(device).eval()
    models["unidepth_v2"] = model_unidepth
    print(" -> UniDepth V2 successfully initialized.")

    # (C) Metric3D ViT-Small
    print("\n[Loading 3/7] Metric3D ViT-Small (37.50M params)...")
    model_metric3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
    model_metric3d.to(device).eval()
    models["metric3d"] = model_metric3d
    print(" -> Metric3D successfully initialized.")

    # (D) Depth Anything V2 Metric-Outdoor-Small
    print("\n[Loading 4/7] Depth Anything V2 Metric-Outdoor-Small (24.79M params)...")
    repo_da_outdoor = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    proc_da_outdoor = AutoImageProcessor.from_pretrained(repo_da_outdoor)
    model_da_outdoor = AutoModelForDepthEstimation.from_pretrained(repo_da_outdoor).to(device).eval()
    models["da_metric_outdoor"] = (model_da_outdoor, proc_da_outdoor)
    print(" -> Depth Anything V2 Metric-Outdoor successfully initialized.")

    # (E) Depth Anything V2 Metric-Indoor-Small
    print("\n[Loading 5/7] Depth Anything V2 Metric-Indoor-Small (24.79M params)...")
    repo_da_indoor = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    proc_da_indoor = AutoImageProcessor.from_pretrained(repo_da_indoor)
    model_da_indoor = AutoModelForDepthEstimation.from_pretrained(repo_da_indoor).to(device).eval()
    models["da_metric_indoor"] = (model_da_indoor, proc_da_indoor)
    print(" -> Depth Anything V2 Metric-Indoor successfully initialized.")

    # (F) ZoeDepth ZoeD_NK
    print("\n[Loading 6/7] ZoeDepth ZoeD_NK (346.10M params)...")
    model_zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True, trust_repo=True)
    model_zoe.to(device).eval()
    models["zoedepth"] = model_zoe
    print(" -> ZoeDepth successfully initialized.")

    # (G) Depth Anything V2 Small (Relative)
    print("\n[Loading 7/7] Depth Anything V2 Small Relative (24.79M params)...")
    repo_da_rel = "depth-anything/Depth-Anything-V2-Small-hf"
    proc_da_rel = AutoImageProcessor.from_pretrained(repo_da_rel)
    model_da_rel = AutoModelForDepthEstimation.from_pretrained(repo_da_rel).to(device).eval()
    models["da_relative"] = (model_da_rel, proc_da_rel)
    print(" -> Depth Anything V2 Relative successfully initialized.")

    # =========================================================================
    # Latency and Throughput Profiling (Apple M3 GPU)
    # =========================================================================
    print("\n" + "=" * 85)
    print(" PROFILING INFERENCE THROUGHPUT & LATENCY ON APPLE SILICON M3")
    print("=" * 85)

    fps_stats = {}
    num_profile_warmup = 5
    num_profile_iters = 20

    # 1. Profile Dioptra-DINO
    dummy_dioptra_img = torch.zeros(1, 3, 224, 224, device=device)
    for _ in range(num_profile_warmup):
        with torch.no_grad():
            _ = model_dioptra(dummy_dioptra_img, dioptra_K)
    t0 = time.perf_counter()
    for _ in range(num_profile_iters):
        with torch.no_grad():
            _ = model_dioptra(dummy_dioptra_img, dioptra_K)
    dioptra_lat = (time.perf_counter() - t0) / num_profile_iters * 1000.0
    fps_stats["dioptra_dino"] = {"latency_ms": dioptra_lat, "fps": 1000.0 / dioptra_lat}
    print(f"Dioptra-DINO:           {fps_stats['dioptra_dino']['fps']:.1f} FPS ({dioptra_lat:.1f} ms)")

    # 2. Profile UniDepth V2
    dummy_unidepth_rgb = torch.randint(0, 255, (1, 3, 480, 640), dtype=torch.uint8, device=device)
    for _ in range(num_profile_warmup):
        with torch.no_grad():
            _ = model_unidepth.infer(dummy_unidepth_rgb, camera=orig_K)
    t0 = time.perf_counter()
    for _ in range(num_profile_iters):
        with torch.no_grad():
            _ = model_unidepth.infer(dummy_unidepth_rgb, camera=orig_K)
    unidepth_lat = (time.perf_counter() - t0) / num_profile_iters * 1000.0
    fps_stats["unidepth_v2"] = {"latency_ms": unidepth_lat, "fps": 1000.0 / unidepth_lat}
    print(f"UniDepth V2 ViT-Small:  {fps_stats['unidepth_v2']['fps']:.1f} FPS ({unidepth_lat:.1f} ms)")

    # 3. Profile Metric3D
    m3d_input_size = (616, 1064)
    dummy_m3d_input = torch.zeros(1, 3, m3d_input_size[0], m3d_input_size[1], device=device)
    for _ in range(num_profile_warmup):
        with torch.no_grad():
            _ = model_metric3d.inference({"input": dummy_m3d_input})
    t0 = time.perf_counter()
    for _ in range(num_profile_iters):
        with torch.no_grad():
            _ = model_metric3d.inference({"input": dummy_m3d_input})
    m3d_lat = (time.perf_counter() - t0) / num_profile_iters * 1000.0
    fps_stats["metric3d"] = {"latency_ms": m3d_lat, "fps": 1000.0 / m3d_lat}
    print(f"Metric3D ViT-Small:     {fps_stats['metric3d']['fps']:.1f} FPS ({m3d_lat:.1f} ms)")

    # 4. Profile Depth Anything V2 Metric Outdoor
    dummy_pil = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    da_inputs = proc_da_outdoor(images=dummy_pil, return_tensors="pt").to(device)
    for _ in range(num_profile_warmup):
        with torch.no_grad():
            _ = model_da_outdoor(**da_inputs).predicted_depth
    t0 = time.perf_counter()
    for _ in range(num_profile_iters):
        with torch.no_grad():
            _ = model_da_outdoor(**da_inputs).predicted_depth
    da_out_lat = (time.perf_counter() - t0) / num_profile_iters * 1000.0
    fps_stats["da_metric_outdoor"] = {"latency_ms": da_out_lat, "fps": 1000.0 / da_out_lat}
    fps_stats["da_metric_indoor"] = {"latency_ms": da_out_lat, "fps": 1000.0 / da_out_lat}
    print(f"DA-v2 Metric Outdoor:   {fps_stats['da_metric_outdoor']['fps']:.1f} FPS ({da_out_lat:.1f} ms)")

    # 5. Profile ZoeDepth
    for _ in range(num_profile_warmup):
        with torch.no_grad():
            _ = model_zoe.infer_pil(dummy_pil)
    t0 = time.perf_counter()
    for _ in range(num_profile_iters):
        with torch.no_grad():
            _ = model_zoe.infer_pil(dummy_pil)
    zoe_lat = (time.perf_counter() - t0) / num_profile_iters * 1000.0
    fps_stats["zoedepth"] = {"latency_ms": zoe_lat, "fps": 1000.0 / zoe_lat}
    print(f"ZoeDepth ZoeD_NK:       {fps_stats['zoedepth']['fps']:.1f} FPS ({zoe_lat:.1f} ms)")

    # 6. Profile Depth Anything V2 Relative
    da_rel_inputs = proc_da_rel(images=dummy_pil, return_tensors="pt").to(device)
    for _ in range(num_profile_warmup):
        with torch.no_grad():
            _ = model_da_rel(**da_rel_inputs).predicted_depth
    t0 = time.perf_counter()
    for _ in range(num_profile_iters):
        with torch.no_grad():
            _ = model_da_rel(**da_rel_inputs).predicted_depth
    da_rel_lat = (time.perf_counter() - t0) / num_profile_iters * 1000.0
    fps_stats["da_relative"] = {"latency_ms": da_rel_lat, "fps": 1000.0 / da_rel_lat}
    print(f"DA-v2 Relative:         {fps_stats['da_relative']['fps']:.1f} FPS ({da_rel_lat:.1f} ms)")

    # =========================================================================
    # Evaluation Loop Across All 200 Frames
    # =========================================================================
    print("\n" + "=" * 85)
    print(" EXECUTING EVALUATION ACROSS ALL 200 HELD-OUT FRAMES")
    print("=" * 85)

    model_keys = [
        "dioptra_dino",
        "unidepth_v2_direct",
        "metric3d_direct",
        "metric3d_median",
        "da_metric_outdoor_direct",
        "da_metric_indoor_direct",
        "zoedepth_direct",
        "da_rel_midas_affine",
        "da_rel_median"
    ]

    frame_metrics = {k: [] for k in model_keys}
    per_frame_records = []

    mean_rgb = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1).to(device)
    std_rgb = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1).to(device)

    for idx, (img_path, depth_path) in enumerate(zip(img_files, depth_files)):
        fname = os.path.basename(img_path)
        raw_bgr = cv2.imread(img_path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        raw_pil = Image.fromarray(raw_rgb)
        raw_gt = np.load(depth_path).astype(np.float32)

        record = {"frame_idx": idx, "filename": fname}

        # ---------------------------------------------------------------------
        # 1. Dioptra-DINO (Native 480x480 Square Crop -> 224x224)
        # ---------------------------------------------------------------------
        left_crop = (orig_W - orig_H) // 2  # 80 px
        top_crop = 0
        min_side = orig_H  # 480 px

        rgb_crop = raw_rgb[top_crop : top_crop + min_side, left_crop : left_crop + min_side]
        rgb_resized = cv2.resize(rgb_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t_img = (t_img - mean_dino) / std_dino

        with torch.no_grad():
            dioptra_pred_tensor = model_dioptra(t_img, dioptra_K)
            dioptra_pred_224 = dioptra_pred_tensor.squeeze().cpu().numpy()

        gt_crop = raw_gt[top_crop : top_crop + min_side, left_crop : left_crop + min_side]
        crop_gt_224 = cv2.resize(gt_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
        m_dioptra = compute_metrics(dioptra_pred_224, crop_gt_224)
        frame_metrics["dioptra_dino"].append(m_dioptra)
        record["dioptra_abs_rel"] = m_dioptra["abs_rel"]
        record["dioptra_rmse"] = m_dioptra["rmse"]
        record["dioptra_delta1"] = m_dioptra["a1"] * 100.0
        record["dioptra_scale"] = m_dioptra["scale_ratio"]

        # ---------------------------------------------------------------------
        # 2. UniDepth V2 ViT-Small (Pseudo-Spherical Ray, Exact Intrinsics K)
        # ---------------------------------------------------------------------
        unidepth_rgb_tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            unidepth_out = model_unidepth.infer(unidepth_rgb_tensor, camera=orig_K)
            unidepth_pred = unidepth_out["depth"].squeeze().cpu().numpy()

        m_unidepth = compute_metrics(unidepth_pred, raw_gt)
        frame_metrics["unidepth_v2_direct"].append(m_unidepth)
        record["unidepth_abs_rel"] = m_unidepth["abs_rel"]
        record["unidepth_rmse"] = m_unidepth["rmse"]
        record["unidepth_delta1"] = m_unidepth["a1"] * 100.0
        record["unidepth_scale"] = m_unidepth["scale_ratio"]

        # ---------------------------------------------------------------------
        # 3. Metric3D ViT-Small (Native 616x1064, Canonical Transform)
        # ---------------------------------------------------------------------
        intrinsic = [320.0, 320.0, 320.0, 240.0]
        scale_m3d = min(m3d_input_size[0] / orig_H, m3d_input_size[1] / orig_W)
        rgb_m3d = cv2.resize(raw_rgb, (int(orig_W * scale_m3d), int(orig_H * scale_m3d)), interpolation=cv2.INTER_LINEAR)
        intrinsic_scaled = [intrinsic[0] * scale_m3d, intrinsic[1] * scale_m3d, intrinsic[2] * scale_m3d, intrinsic[3] * scale_m3d]

        pad_h = m3d_input_size[0] - rgb_m3d.shape[0]
        pad_w = m3d_input_size[1] - rgb_m3d.shape[1]
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb_padded = cv2.copyMakeBorder(rgb_m3d, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])

        tensor_rgb = torch.from_numpy(rgb_padded.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
        norm_rgb = (tensor_rgb - mean_rgb) / std_rgb

        with torch.no_grad():
            pred_depth, _, _ = model_metric3d.inference({"input": norm_rgb})

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_h_half : pred_depth.shape[0] - (pad_h - pad_h_half), pad_w_half : pred_depth.shape[1] - (pad_w - pad_w_half)]
        pred_depth = F.interpolate(pred_depth[None, None, :, :], (orig_H, orig_W), mode="bilinear").squeeze().cpu().numpy()

        canonical_to_real_scale = intrinsic_scaled[0] / 1000.0
        m3d_pred_direct = np.clip(pred_depth * canonical_to_real_scale, 0.0, 300.0)

        m_m3d_dir = compute_metrics(m3d_pred_direct, raw_gt)
        frame_metrics["metric3d_direct"].append(m_m3d_dir)
        record["m3d_direct_abs_rel"] = m_m3d_dir["abs_rel"]
        record["m3d_direct_rmse"] = m_m3d_dir["rmse"]
        record["m3d_direct_delta1"] = m_m3d_dir["a1"] * 100.0
        record["m3d_direct_scale"] = m_m3d_dir["scale_ratio"]

        # Oracle Median Scaling for Metric3D
        mask_gt = (raw_gt > 0.1) & (raw_gt < 80.0) & np.isfinite(raw_gt) & (m3d_pred_direct > 0.05)
        if mask_gt.sum() > 0:
            scale_oracle = np.median(raw_gt[mask_gt]) / np.median(m3d_pred_direct[mask_gt])
            m3d_pred_median = m3d_pred_direct * scale_oracle
        else:
            m3d_pred_median = m3d_pred_direct
        m_m3d_med = compute_metrics(m3d_pred_median, raw_gt)
        frame_metrics["metric3d_median"].append(m_m3d_med)
        record["m3d_median_abs_rel"] = m_m3d_med["abs_rel"]
        record["m3d_median_rmse"] = m_m3d_med["rmse"]
        record["m3d_median_delta1"] = m_m3d_med["a1"] * 100.0

        # ---------------------------------------------------------------------
        # 4. Depth Anything V2 Metric Outdoor (VKITTI2 Fine-Tuned)
        # ---------------------------------------------------------------------
        da_out_inputs = proc_da_outdoor(images=raw_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            pred_outdoor = model_da_outdoor(**da_out_inputs).predicted_depth
            pred_outdoor = F.interpolate(pred_outdoor.unsqueeze(1), size=(orig_H, orig_W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

        m_da_out = compute_metrics(pred_outdoor, raw_gt)
        frame_metrics["da_metric_outdoor_direct"].append(m_da_out)
        record["da_outdoor_abs_rel"] = m_da_out["abs_rel"]
        record["da_outdoor_rmse"] = m_da_out["rmse"]
        record["da_outdoor_delta1"] = m_da_out["a1"] * 100.0
        record["da_outdoor_scale"] = m_da_out["scale_ratio"]

        # ---------------------------------------------------------------------
        # 5. Depth Anything V2 Metric Indoor (Hypersim Fine-Tuned)
        # ---------------------------------------------------------------------
        da_in_inputs = proc_da_indoor(images=raw_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            pred_indoor = model_da_indoor(**da_in_inputs).predicted_depth
            pred_indoor = F.interpolate(pred_indoor.unsqueeze(1), size=(orig_H, orig_W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

        m_da_in = compute_metrics(pred_indoor, raw_gt)
        frame_metrics["da_metric_indoor_direct"].append(m_da_in)
        record["da_indoor_abs_rel"] = m_da_in["abs_rel"]
        record["da_indoor_rmse"] = m_da_in["rmse"]
        record["da_indoor_delta1"] = m_da_in["a1"] * 100.0
        record["da_indoor_scale"] = m_da_in["scale_ratio"]

        # ---------------------------------------------------------------------
        # 6. ZoeDepth ZoeD_NK (Metric Bins)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            pred_zoe = model_zoe.infer_pil(raw_pil)

        m_zoe = compute_metrics(pred_zoe, raw_gt)
        frame_metrics["zoedepth_direct"].append(m_zoe)
        record["zoe_abs_rel"] = m_zoe["abs_rel"]
        record["zoe_rmse"] = m_zoe["rmse"]
        record["zoe_delta1"] = m_zoe["a1"] * 100.0
        record["zoe_scale"] = m_zoe["scale_ratio"]

        # ---------------------------------------------------------------------
        # 7. Depth Anything V2 Relative (MiDaS Disparity Affine & Median)
        # ---------------------------------------------------------------------
        da_rel_inputs = proc_da_rel(images=raw_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            pred_rel = model_da_rel(**da_rel_inputs).predicted_depth
            pred_rel = F.interpolate(pred_rel.unsqueeze(1), size=(orig_H, orig_W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

        mask_rel = (raw_gt > 0.1) & (raw_gt < 80.0) & np.isfinite(raw_gt) & (pred_rel > 0)
        d_vals = pred_rel[mask_rel]
        inv_gt = 1.0 / raw_gt[mask_rel]

        A = np.vstack([d_vals, np.ones_like(d_vals)]).T
        s_aff, t_aff = np.linalg.lstsq(A, inv_gt, rcond=None)[0]
        aligned_disp = np.maximum(s_aff * d_vals + t_aff, 1e-4)
        aligned_depth_da = 1.0 / aligned_disp
        full_da_affine = np.zeros_like(raw_gt)
        full_da_affine[mask_rel] = aligned_depth_da

        m_da_midas = compute_metrics(full_da_affine, raw_gt)
        frame_metrics["da_rel_midas_affine"].append(m_da_midas)
        record["da_midas_abs_rel"] = m_da_midas["abs_rel"]
        record["da_midas_rmse"] = m_da_midas["rmse"]
        record["da_midas_delta1"] = m_da_midas["a1"] * 100.0

        # Disparity median scaling: s * (1/d) ≈ Z_gt
        inv_d = 1.0 / np.maximum(d_vals, 1e-3)
        scale_disp = np.median(raw_gt[mask_rel]) / np.median(inv_d)
        med_depth_da = inv_d * scale_disp
        full_da_median = np.zeros_like(raw_gt)
        full_da_median[mask_rel] = med_depth_da

        m_da_med = compute_metrics(full_da_median, raw_gt)
        frame_metrics["da_rel_median"].append(m_da_med)
        record["da_median_abs_rel"] = m_da_med["abs_rel"]
        record["da_median_rmse"] = m_da_med["rmse"]
        record["da_median_delta1"] = m_da_med["a1"] * 100.0

        per_frame_records.append(record)
        if (idx + 1) % 10 == 0 or (idx + 1) == 200:
            print(f"[{idx+1:03d}/200] Dioptra: {m_dioptra['abs_rel']:.4f} | UniDepth: {m_unidepth['abs_rel']:.4f} | Metric3D: {m_m3d_dir['abs_rel']:.4f} | DA-Out: {m_da_out['abs_rel']:.4f} | ZoeDepth: {m_zoe['abs_rel']:.4f}")
            sys.stdout.flush()

    # =========================================================================
    # Aggregation & Summary Table
    # =========================================================================
    print("\n" + "=" * 110)
    print(" SUMMARY BENCHMARK RESULTS ACROSS ALL 200 HELD-OUT FRAMES (ABANDONEDFACTORY P010)")
    print("=" * 110)
    header = f"{'Model Architecture':<30} | {'Alignment Mode':<22} | {'AbsRel':<7} | {'SqRel':<7} | {'RMSE (m)':<8} | {'δ1 < 1.25':<9} | {'Scale':<6} | {'FPS':<6}"
    print(header)
    print("-" * 110)

    summary_results = {}
    display_configs = [
        ("Dioptra-DINO (Ours)", "dioptra_dino", "Direct Metric (K)", "dioptra_dino"),
        ("UniDepth-V2 ViT-Small", "unidepth_v2_direct", "Direct Metric (K)", "unidepth_v2"),
        ("Metric3D ViT-Small", "metric3d_direct", "Direct Metric (K)", "metric3d"),
        ("Metric3D ViT-Small", "metric3d_median", "Oracle Median Scale", "metric3d"),
        ("Depth Anything V2-S Metric", "da_metric_outdoor_direct", "Direct Metric (VKITTI2)", "da_metric_outdoor"),
        ("Depth Anything V2-S Metric", "da_metric_indoor_direct", "Direct Metric (Hypersim)", "da_metric_indoor"),
        ("ZoeDepth ZoeD_NK", "zoedepth_direct", "Direct Metric (Bins)", "zoedepth"),
        ("Depth Anything V2-S Rel", "da_rel_midas_affine", "Oracle MiDaS Affine", "da_relative"),
        ("Depth Anything V2-S Rel", "da_rel_median", "Disparity Median Scale", "da_relative"),
    ]

    for model_name, metric_key, align_mode, fps_key in display_configs:
        m_list = frame_metrics[metric_key]
        mean_abs_rel = float(np.mean([m["abs_rel"] for m in m_list]))
        mean_sq_rel = float(np.mean([m["sq_rel"] for m in m_list]))
        mean_rmse = float(np.mean([m["rmse"] for m in m_list]))
        mean_a1 = float(np.mean([m["a1"] for m in m_list])) * 100.0
        mean_scale = float(np.mean([m["scale_ratio"] for m in m_list]))
        fps = fps_stats[fps_key]["fps"]

        summary_results[metric_key] = {
            "model": model_name,
            "alignment": align_mode,
            "abs_rel": mean_abs_rel,
            "sq_rel": mean_sq_rel,
            "rmse": mean_rmse,
            "a1": mean_a1,
            "scale_ratio": mean_scale,
            "fps": fps
        }

        row = f"{model_name:<30} | {align_mode:<22} | {mean_abs_rel:<7.4f} | {mean_sq_rel:<7.4f} | {mean_rmse:<8.3f} | {mean_a1:<8.2f}% | {mean_scale:<6.4f} | {fps:<5.1f}"
        print(row)

    print("=" * 110)

    # Save to JSON and CSV
    os.makedirs("outputs", exist_ok=True)
    json_path = "outputs/comprehensive_metric_suite_results.json"
    with open(json_path, "w") as fp:
        json.dump({"summary": summary_results, "fps_stats": fps_stats}, fp, indent=2)
    print(f"\n[Saved] Summary results written to {json_path}")

    # Write per-frame CSV
    import pandas as pd
    df = pd.DataFrame(per_frame_records)
    csv_path = "outputs/comprehensive_per_frame_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Saved] Per-frame records written to {csv_path}")


if __name__ == "__main__":
    main()
