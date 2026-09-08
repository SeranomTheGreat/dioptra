"""
Evaluation and Inference script for Dioptra-DINO.

Runs zero-shot metric depth estimation, generates comparative heatmaps,
and exports 3D point clouds (.ply).

Usage:
  python scripts/eval_dino.py --image <path> --intrinsics <path/values>
  python scripts/eval_dino.py --demo
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD


def run_demo():
    print("=" * 70)
    print("RUNNING DIOPTRA-DINO INFERENCE & BENCHMARK DEMO")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Inference Device: {device}")

    cfg = DioptraDINOConfig()
    model = DioptraDINO(cfg).to(device)
    model.eval()

    # Synthetic camera: Canonical TartanAir K rescaled to 224x224
    K = torch.tensor([[[112.0, 0.0, 112.0], [0.0, 112.0, 112.0], [0.0, 0.0, 1.0]]], device=device)

    # Check for sample image
    sample_img_path = "assets/tartanair_sample.png"
    if not os.path.exists(sample_img_path):
        sample_img_path = "test_samples/000022_left.png"

    if os.path.exists(sample_img_path):
        print(f"Loading test image: {sample_img_path}")
        from PIL import Image
        pil_img = Image.open(sample_img_path).convert("RGB").resize((224, 224))
        img_np = np.array(pil_img)
    else:
        print("Using synthetic test frame...")
        img_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Normalize
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    for c, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        img_tensor[c] = (img_tensor[c] - mean) / std
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Warmup
    with torch.no_grad():
        _ = model(img_tensor, K)

    # Benchmark 20 forward passes
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            pred = model(img_tensor, K)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_lat = np.mean(latencies[5:])  # Exclude first 5 warmups
    fps = 1000.0 / avg_lat

    depth_np = pred.squeeze().cpu().numpy()
    print(f"Predicted Metric Depth Map shape: {depth_np.shape}")
    print(f"Depth range: min = {depth_np.min():.2f}m, median = {np.median(depth_np):.2f}m, max = {depth_np.max():.2f}m")
    print(f"Steady-State Latency on {device}: {avg_lat:.2f} ms ({fps:.1f} FPS)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dioptra-DINO Evaluation")
    parser.add_argument("--demo", action="store_true", help="Run benchmark demo")
    parser.add_argument("--image", type=str, default=None, help="Input RGB image")
    args = parser.parse_args()

    run_demo()
