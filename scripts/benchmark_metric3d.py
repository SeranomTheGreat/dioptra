"""
Benchmark Metric3D ViT-Small (CVPR 2023 / TPAMI 2024) across the 200 continuous
unseen indoor test frames of TartanAir (abandonedfactory/Easy/P010).

Metric3D is a camera-conditioned metric depth foundation model that takes RGB and
camera intrinsics, predicting true physical depth in metres without oracle alignment.

Saves results to: outputs/metric3d_200_results.json
"""

import os
import sys
import glob
import json
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 80.0):
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & (pred > 0.05) & np.isfinite(pred)
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


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Metric3D] Benchmarking Metric3D ViT-Small on device: {device}")

    # Load Metric3D ViT-Small
    print("[Metric3D] Loading model via torch.hub...")
    model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True)
    model.to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Metric3D] Parameter count: {total_params / 1e6:.2f}M ({total_params:,} parameters)")

    data_dir = "test_samples/unseen_200_abandonedfactory"
    img_files = sorted(glob.glob(os.path.join(data_dir, "image_left", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(data_dir, "depth_left", "*.npy")))

    assert len(img_files) == 200, f"Expected 200 images, got {len(img_files)}"
    assert len(depth_files) == 200, f"Expected 200 depth files, got {len(depth_files)}"
    print(f"[Metric3D] Found {len(img_files)} continuous test frames.")

    input_size = (616, 1064)
    padding = [123.675, 116.28, 103.53]
    mean = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1).to(device)
    std = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1).to(device)

    # Measure inference throughput / latency
    print("[Metric3D] Profiling inference latency on Apple Silicon M3...")
    dummy_h, dummy_w = 480, 640
    scale = min(input_size[0] / dummy_h, input_size[1] / dummy_w)
    pad_h = input_size[0] - int(dummy_h * scale)
    pad_w = input_size[1] - int(dummy_w * scale)
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2

    dummy_input = torch.zeros(1, 3, input_size[0], input_size[1], device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.inference({"input": dummy_input})

    num_trials = 20
    t0 = time.perf_counter()
    for _ in range(num_trials):
        with torch.no_grad():
            _ = model.inference({"input": dummy_input})
    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / num_trials) * 1000.0
    fps = num_trials / elapsed
    print(f"[Metric3D] Inference Speed: {fps:.2f} FPS ({latency_ms:.1f} ms/frame)")

    frame_metrics = []
    print("\n[Metric3D] Running continuous evaluation across 200 frames...")

    for idx, (img_path, gt_path) in enumerate(zip(img_files, depth_files)):
        rgb_origin = cv2.imread(img_path)[:, :, ::-1]
        h, w = rgb_origin.shape[:2]

        # Native TartanAir pinhole camera intrinsics
        intrinsic = [320.0, 320.0, 320.0, 240.0]

        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        intrinsic_scaled = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

        pad_h = input_size[0] - rgb.shape[0]
        pad_w = input_size[1] - rgb.shape[1]
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb_padded = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)

        tensor_rgb = torch.from_numpy(rgb_padded.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
        norm_rgb = (tensor_rgb - mean) / std

        with torch.no_grad():
            pred_depth, _, _ = model.inference({"input": norm_rgb})

        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_h_half : pred_depth.shape[0] - (pad_h - pad_h_half), pad_w_half : pred_depth.shape[1] - (pad_w - pad_w_half)]
        pred_depth = F.interpolate(pred_depth[None, None, :, :], (h, w), mode="bilinear").squeeze().cpu().numpy()

        # De-canonical transform: scales canonical focal length (1000px) to real focal length
        canonical_to_real_scale = intrinsic_scaled[0] / 1000.0
        pred_depth_metric = pred_depth * canonical_to_real_scale
        pred_depth_metric = np.clip(pred_depth_metric, 0.0, 300.0)

        gt_depth = np.load(gt_path)

        metrics = compute_metrics(pred_depth_metric, gt_depth)
        frame_metrics.append(metrics)

        if (idx + 1) % 50 == 0 or idx == 199:
            print(f"  Frame [{idx + 1:03d}/200] - AbsRel: {metrics['abs_rel']:.4f}, RMSE: {metrics['rmse']:.3f}m, delta1: {metrics['a1'] * 100:.2f}%, Scale: {metrics['scale_ratio']:.4f}")

    # Compute aggregate metrics
    keys = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale_ratio"]
    summary = {
        "model_name": "Metric3D-ViT-Small (CVPR 2023 / TPAMI 2024)",
        "parameters": total_params,
        "parameters_m": round(total_params / 1e6, 2),
        "alignment": "Direct Metric (Camera-Conditioned, No Test-Time Alignment)",
        "frames_evaluated": len(frame_metrics),
        "latency_ms": round(latency_ms, 1),
        "fps": round(fps, 1),
    }

    for k in keys:
        vals = [fm[k] for fm in frame_metrics]
        if k == "scale_ratio":
            summary[k] = round(float(np.median(vals)), 4)
        else:
            summary[k] = round(float(np.mean(vals)), 4)

    summary["a1_percent"] = round(summary["a1"] * 100.0, 2)
    summary["a2_percent"] = round(summary["a2"] * 100.0, 2)
    summary["a3_percent"] = round(summary["a3"] * 100.0, 2)

    print("\n" + "=" * 80)
    print("FINAL 200-FRAME BENCHMARK RESULTS: METRIC3D VIT-SMALL")
    print("=" * 80)
    print(f"  Parameters:     {summary['parameters_m']} M")
    print(f"  Throughput:     {summary['fps']} FPS ({summary['latency_ms']} ms)")
    print(f"  AbsRel:         {summary['abs_rel']:.4f} ({summary['abs_rel'] * 100:.2f}%)")
    print(f"  SqRel:          {summary['sq_rel']:.4f}")
    print(f"  RMSE:           {summary['rmse']:.3f} m")
    print(f"  RMSE (log):     {summary['rmse_log']:.4f}")
    print(f"  delta < 1.25:   {summary['a1_percent']}%")
    print(f"  delta < 1.25^2: {summary['a2_percent']}%")
    print(f"  delta < 1.25^3: {summary['a3_percent']}%")
    print(f"  Scale Ratio:    {summary['scale_ratio']:.4f}")
    print("=" * 80)

    out_file = "outputs/metric3d_200_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Metric3D] Verified results saved to: {out_file}")


if __name__ == "__main__":
    main()
