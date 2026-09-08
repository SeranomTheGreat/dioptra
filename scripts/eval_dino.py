"""
Dioptra-DINO Comprehensive Evaluation, Benchmark, and Multi-FOV Sweep Suite.

Supports:
1. Multi-FOV Sweep Visualization (Figure 8 style across 50° - 100° FOVs).
2. Held-out Quantitative Evaluation Benchmark (AbsRel, RMSE, delta_1, delta_2, Scale Error).
3. Multi-Scene Qualitative Grid Comparison (RGB | Ground Truth | Dioptra-DINO | Error Heatmap).
4. Interactive Kaggle / CLI execution.

Usage:
  # Multi-FOV Sweep on a specific sample
  python scripts/eval_dino.py --checkpoint outputs_dino/dioptra_dino_best.pt --sweep --image test_samples/abandonedfactory/000300_left.png

  # Full benchmark across all test scenes
  python scripts/eval_dino.py --checkpoint outputs_dino/dioptra_dino_best.pt --benchmark

  # Quick demo with pre-trained weights
  python scripts/eval_dino.py --demo
"""

import argparse
import glob
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD


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


def load_model(checkpoint_path: Optional[str] = None, device: torch.device = torch.device("cpu")) -> DioptraDINO:
    """Load Dioptra-DINO model, optionally with fine-tuned checkpoint weights."""
    cfg = DioptraDINOConfig(freeze_backbone=False)
    model = DioptraDINO(cfg).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[Dioptra-DINO Eval] Loading weights from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        # Strip DataParallel 'module.' prefix if present
        cleaned_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                cleaned_state[k[7:]] = v
            else:
                cleaned_state[k] = v
        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        print(f"[Dioptra-DINO Eval] Weights loaded (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
    else:
        if checkpoint_path:
            print(f"[Dioptra-DINO Eval] Warning: Checkpoint {checkpoint_path} not found. Running with pre-trained DINOv2 backbone.")
        else:
            print("[Dioptra-DINO Eval] Initialized with pre-trained DINOv2 backbone.")

    model.eval()
    return model


def preprocess_sample(img_path: str, gt_path: Optional[str] = None, img_size: int = 224, device: torch.device = torch.device("cpu")):
    """Load and center-crop RGB image and optional GT depth map to square img_size."""
    raw_img = Image.open(img_path).convert("RGB")
    W_orig, H_orig = raw_img.size
    min_side = min(H_orig, W_orig)
    top = (H_orig - min_side) // 2
    left = (W_orig - min_side) // 2

    raw_cropped = raw_img.crop((left, top, left + min_side, top + min_side))
    img_resized = raw_cropped.resize((img_size, img_size), Image.Resampling.BILINEAR)

    img_t = TF.to_tensor(img_resized)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    input_t = ((img_t - mean) / std).unsqueeze(0).to(device)

    gt_np = None
    if gt_path and os.path.exists(gt_path):
        try:
            gt_arr = np.load(gt_path).astype(np.float32)
            gt_t = torch.from_numpy(gt_arr)
            gt_cropped = gt_t[top:top + min_side, left:left + min_side]
            gt_depth = TF.resize(gt_cropped.unsqueeze(0), [img_size, img_size], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
            gt_np = gt_depth.numpy()
        except Exception as e:
            print(f"[Dioptra-DINO Eval] Error reading GT {gt_path}: {e}")

    # Native camera matrix for TartanAir 224x224 center-crop (FOV ≈ 73.74°)
    # fx_native = 320 * (224 / 480) = 149.33 px
    fx_native = 320.0 * (float(img_size) / float(min_side))
    K_native = torch.tensor([[[fx_native, 0.0, img_size / 2.0],
                             [0.0, fx_native, img_size / 2.0],
                             [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

    return input_t, gt_np, img_t, K_native, fx_native


def run_fov_sweep(
    model: DioptraDINO,
    img_path: str,
    gt_path: Optional[str] = None,
    output_path: str = "outputs_dino/fig_dino_multi_fov_sweep.png",
    device: torch.device = torch.device("cpu"),
    fovs: List[float] = [50.0, 60.0, 73.74, 85.0, 90.0, 100.0],
):
    """Run multi-FOV sweep on single image and render publication-grade comparison figure."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img_size = 224
    input_t, gt_np, img_t, K_native, fx_native = preprocess_sample(img_path, gt_path, img_size=img_size, device=device)

    # Determine colormap bounds from GT (filtering sky pixels >80m)
    if gt_np is not None:
        valid_gt = (gt_np > 0.1) & (gt_np < 80.0) & np.isfinite(gt_np)
        if valid_gt.sum() > 0:
            vmax = float(np.percentile(gt_np[valid_gt], 98))
            vmin = max(0.0, float(np.percentile(gt_np[valid_gt], 2)))
        else:
            vmin, vmax = 0.0, 20.0
    else:
        vmin, vmax = 0.0, 20.0

    sweep_results = []
    with torch.no_grad():
        for fov in fovs:
            fov_rad = np.radians(fov)
            fx = (img_size / 2.0) / np.tan(fov_rad / 2.0)
            K = torch.tensor([[[fx, 0.0, img_size / 2.0],
                               [0.0, fx, img_size / 2.0],
                               [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

            pred_t = model(input_t, K, ara_gate=1.0)
            pred_np = pred_t.squeeze().cpu().numpy()

            metrics = compute_metrics(pred_np, gt_np) if gt_np is not None else {}
            sweep_results.append({
                "fov": fov,
                "fx": fx,
                "depth": pred_np,
                "metrics": metrics,
            })

    # Render figure: 2 rows x (2 + len(fovs)) cols
    num_cols = 2 + len(fovs)
    fig, axes = plt.subplots(2, num_cols, figsize=(3.2 * num_cols, 6.5), constrained_layout=True)

    # (0, 0): Input RGB
    axes[0, 0].imshow(img_t.permute(1, 2, 0).numpy())
    scene_name = os.path.basename(os.path.dirname(img_path))
    file_name = os.path.basename(img_path)
    axes[0, 0].set_title(f"Input RGB\n{scene_name}/{file_name}", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # (0, 1): Ground Truth
    if gt_np is not None:
        gt_disp = np.clip(gt_np, vmin, vmax)
        im_gt = axes[0, 1].imshow(gt_disp, cmap="plasma", vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"Ground Truth\nRange: [{vmin:.1f}, {vmax:.1f}] m", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")
        cbar = plt.colorbar(im_gt, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar.set_label("Metric Depth (m)", fontsize=9)
    else:
        axes[0, 1].text(0.5, 0.5, "No GT Depth\nAvailable", ha="center", va="center", fontsize=11)
        axes[0, 1].axis("off")

    # (1, 0): Info Box
    axes[1, 0].axis("off")
    info_text = (
        "Dioptra-DINO\n"
        "Foundation-Assisted Depth\n\n"
        "• DINOv2-Small (vits14)\n"
        "• Trivision Ray (FiLM)\n"
        "• Angular Residual Attn\n"
        "• Decoupled Scale Supv"
    )
    axes[1, 0].text(
        0.5, 0.5, info_text, ha="center", va="center", fontsize=10, fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4f8", edgecolor="#3b82f6", linewidth=1.5)
    )

    # (1, 1): Quantitative Curve
    if gt_np is not None and "abs_rel" in sweep_results[0]["metrics"]:
        ar_vals = [r["metrics"]["abs_rel"] for r in sweep_results]
        axes[1, 1].plot(fovs, ar_vals, "o-", color="#2563eb", linewidth=2.2, markersize=6, label="AbsRel Error")
        axes[1, 1].axvline(73.74, color="#dc2626", linestyle="--", alpha=0.7, label="Native FOV (73.7°)")
        axes[1, 1].set_title("AbsRel Error vs FOV", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Synthetic Camera FOV (°)", fontsize=9)
        axes[1, 1].set_ylabel("AbsRel Error", fontsize=9)
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)
        axes[1, 1].legend(fontsize=8, loc="upper right")
    else:
        axes[1, 1].text(0.5, 0.5, "Error Curve", ha="center", va="center", fontsize=11)
        axes[1, 1].axis("off")

    # Sweep Columns: Top row = Predicted Depth, Bottom row = Absolute Error map or Relative Difference
    for i, res in enumerate(sweep_results):
        col = i + 2
        fov_val = res["fov"]
        fx_val = res["fx"]
        pred_d = res["depth"]
        metrics = res["metrics"]

        # Top row: Prediction
        pred_disp = np.clip(pred_d, vmin, vmax)
        axes[0, col].imshow(pred_disp, cmap="plasma", vmin=vmin, vmax=vmax)
        ar_str = f" | AR: {metrics.get('abs_rel', 0):.3f}" if "abs_rel" in metrics else ""
        is_native = abs(fov_val - 73.74) < 1.0
        title_color = "#16a34a" if is_native else "black"
        lbl = f"★ FOV {fov_val:.1f}° (Native)" if is_native else f"FOV {fov_val:.0f}°"
        axes[0, col].set_title(f"{lbl}\nfx={fx_val:.0f}px{ar_str}", fontsize=10, fontweight="bold", color=title_color)
        axes[0, col].axis("off")

        # Bottom row: Error Heatmap (or Difference from Native)
        if gt_np is not None:
            err_map = np.abs(pred_d - gt_np) / np.maximum(gt_np, 0.1)
            err_map[~valid_gt] = 0.0
            im_err = axes[1, col].imshow(err_map, cmap="inferno", vmin=0.0, vmax=0.6)
            axes[1, col].set_title(f"Rel Error Map\nMed: {np.median(err_map[valid_gt]):.3f}", fontsize=9)
            axes[1, col].axis("off")
        else:
            diff_from_first = np.abs(pred_d - sweep_results[0]["depth"])
            axes[1, col].imshow(diff_from_first, cmap="inferno")
            axes[1, col].set_title("Diff from 50°", fontsize=9)
            axes[1, col].axis("off")

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Dioptra-DINO Eval] Multi-FOV Sweep saved successfully: {output_path}")
    return output_path


def run_benchmark(
    model: DioptraDINO,
    test_dir: str = "test_samples",
    device: torch.device = torch.device("cpu"),
    output_dir: str = "outputs_dino",
    max_samples: int = 50,
) -> Dict[str, float]:
    """Run full benchmark on held-out test scenes and generate summary metrics & table."""
    os.makedirs(output_dir, exist_ok=True)
    all_pngs = sorted(glob.glob(os.path.join(test_dir, "**", "*_left.png"), recursive=True))
    if not all_pngs:
        all_pngs = sorted(glob.glob(os.path.join(test_dir, "**", "*.png"), recursive=True))

    pairs = []
    for p in all_pngs:
        fname = os.path.basename(p)
        if "depth" in fname.lower():
            continue
        base, _ = os.path.splitext(p)
        cand_gt = [f"{base}_depth.npy", p.replace(".png", "_depth.npy"), p.replace("image_left", "depth_left").replace(".png", ".npy")]
        for c in cand_gt:
            if os.path.exists(c):
                pairs.append((p, c))
                break

    if not pairs:
        print(f"[Dioptra-DINO Eval] No image/depth pairs found in {test_dir}!")
        return {}

    print(f"[Dioptra-DINO Eval] Found {len(pairs)} test pairs. Evaluating up to {min(len(pairs), max_samples)} samples...")
    metrics_list = []

    for idx, (img_p, gt_p) in enumerate(pairs[:max_samples]):
        input_t, gt_np, _, K_native, _ = preprocess_sample(img_p, gt_p, img_size=224, device=device)
        if gt_np is None:
            continue
        with torch.no_grad():
            pred_t = model(input_t, K_native, ara_gate=1.0)
            pred_np = pred_t.squeeze().cpu().numpy()

        m = compute_metrics(pred_np, gt_np)
        metrics_list.append(m)
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Sample [{idx+1}/{min(len(pairs), max_samples)}]: AbsRel = {m['abs_rel']:.4f}, RMSE = {m['rmse']:.3f}m, δ1 = {m['a1']*100:.1f}%")

    if not metrics_list:
        return {}

    avg_metrics = {
        "abs_rel": float(np.mean([m["abs_rel"] for m in metrics_list])),
        "sq_rel": float(np.mean([m["sq_rel"] for m in metrics_list])),
        "rmse": float(np.mean([m["rmse"] for m in metrics_list])),
        "rmse_log": float(np.mean([m["rmse_log"] for m in metrics_list])),
        "a1": float(np.mean([m["a1"] for m in metrics_list])),
        "a2": float(np.mean([m["a2"] for m in metrics_list])),
        "a3": float(np.mean([m["a3"] for m in metrics_list])),
        "scale_ratio": float(np.mean([m["scale_ratio"] for m in metrics_list])),
    }

    print("\n" + "=" * 70)
    print("DIOPTRA-DINO QUANTITATIVE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Evaluated Samples : {len(metrics_list)}")
    print(f"AbsRel (Error ↓)  : {avg_metrics['abs_rel']:.4f}")
    print(f"SqRel  (Error ↓)  : {avg_metrics['sq_rel']:.4f}")
    print(f"RMSE   (Metres ↓) : {avg_metrics['rmse']:.4f} m")
    print(f"RMSE log (Log ↓)  : {avg_metrics['rmse_log']:.4f}")
    print(f"δ < 1.25 (Acc ↑)  : {avg_metrics['a1'] * 100:.2f}%")
    print(f"δ < 1.25²(Acc ↑)  : {avg_metrics['a2'] * 100:.2f}%")
    print(f"δ < 1.25³(Acc ↑)  : {avg_metrics['a3'] * 100:.2f}%")
    print(f"Scale Ratio (1.0) : {avg_metrics['scale_ratio']:.4f}")
    print("=" * 70 + "\n")

    # Generate multi-scene comparison figure across 4 diverse scenes
    render_multiscene_grid(model, pairs[:4], output_path=os.path.join(output_dir, "fig_dino_multiscene_eval.png"), device=device)

    return avg_metrics


def render_multiscene_grid(
    model: DioptraDINO,
    pairs: List[Tuple[str, str]],
    output_path: str = "outputs_dino/fig_dino_multiscene_eval.png",
    device: torch.device = torch.device("cpu"),
):
    """Render 4-row x 4-column grid (RGB | Ground Truth | Dioptra-DINO | Rel Error)."""
    if not pairs:
        return
    n = min(4, len(pairs))
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        img_p, gt_p = pairs[i]
        input_t, gt_np, img_t, K_native, _ = preprocess_sample(img_p, gt_p, img_size=224, device=device)
        with torch.no_grad():
            pred_t = model(input_t, K_native, ara_gate=1.0)
            pred_np = pred_t.squeeze().cpu().numpy()

        val = (gt_np > 0.1) & (gt_np < 80.0) & np.isfinite(gt_np)
        vmax = float(np.percentile(gt_np[val], 98)) if val.sum() > 0 else 20.0
        vmin = max(0.0, float(np.percentile(gt_np[val], 2))) if val.sum() > 0 else 0.0

        m = compute_metrics(pred_np, gt_np)

        # Col 0: RGB
        axes[i, 0].imshow(img_t.permute(1, 2, 0).numpy())
        scene = os.path.basename(os.path.dirname(img_p))
        name = os.path.basename(img_p)
        axes[i, 0].set_title(f"Scene {i+1}: {scene}\n{name}", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # Col 1: Ground Truth
        axes[i, 1].imshow(np.clip(gt_np, vmin, vmax), cmap="plasma", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Ground Truth\nMax: {vmax:.1f}m", fontsize=10, fontweight="bold")
        axes[i, 1].axis("off")

        # Col 2: Dioptra-DINO Prediction
        axes[i, 2].imshow(np.clip(pred_np, vmin, vmax), cmap="plasma", vmin=vmin, vmax=vmax)
        axes[i, 2].set_title(f"Dioptra-DINO\nAbsRel: {m['abs_rel']:.3f} | RMSE: {m['rmse']:.2f}m", fontsize=10, fontweight="bold", color="#16a34a")
        axes[i, 2].axis("off")

        # Col 3: Relative Error
        err = np.abs(pred_np - gt_np) / np.maximum(gt_np, 0.1)
        err[~val] = 0.0
        axes[i, 3].imshow(err, cmap="inferno", vmin=0.0, vmax=0.5)
        axes[i, 3].set_title(f"Rel Error (|d - d*| / d*)\nδ1 Acc: {m['a1']*100:.1f}%", fontsize=10)
        axes[i, 3].axis("off")

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Dioptra-DINO Eval] Multi-scene evaluation grid saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Dioptra-DINO Evaluation Suite")
    parser.add_argument("--checkpoint", type=str, default="outputs_dino/dioptra_dino_best.pt", help="Path to checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to single input RGB image")
    parser.add_argument("--depth", type=str, default=None, help="Path to matching GT depth .npy file")
    parser.add_argument("--sweep", action="store_true", help="Run multi-FOV sweep")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark across test samples")
    parser.add_argument("--test-dir", type=str, default="test_samples", help="Directory containing test samples")
    parser.add_argument("--output-dir", type=str, default="outputs_dino", help="Directory to save figures and metrics")
    parser.add_argument("--demo", action="store_true", help="Run quick benchmark latency demo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[Dioptra-DINO Eval] Execution Device: {device}")

    if args.demo:
        model = load_model(None, device=device)
        print("[Dioptra-DINO Eval] Running demo latency test...")
        dummy_img = torch.randn(1, 3, 224, 224, device=device)
        dummy_K = torch.tensor([[[149.33, 0.0, 112.0], [0.0, 149.33, 112.0], [0.0, 0.0, 1.0]]], device=device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_img, dummy_K)
            t0 = time.perf_counter()
            for _ in range(20):
                _ = model(dummy_img, dummy_K)
            lat = (time.perf_counter() - t0) / 20.0 * 1000.0
        print(f"[Dioptra-DINO Eval] Latency on {device}: {lat:.2f} ms ({1000.0/lat:.1f} FPS)")
        return

    model = load_model(args.checkpoint, device=device)

    # If specific image provided, run sweep or inference
    if args.image:
        gt_path = args.depth
        if gt_path is None:
            # Auto-detect depth file
            base, _ = os.path.splitext(args.image)
            cand = [f"{base}_depth.npy", args.image.replace(".png", "_depth.npy")]
            for c in cand:
                if os.path.exists(c):
                    gt_path = c
                    break
        sweep_out = os.path.join(args.output_dir, "fig_dino_multi_fov_sweep.png")
        run_fov_sweep(model, args.image, gt_path, output_path=sweep_out, device=device)

    # If benchmark requested or no args provided
    if args.benchmark or not args.image:
        test_dir = args.test_dir
        if not os.path.exists(test_dir) and os.path.exists("test_samples"):
            test_dir = "test_samples"
        run_benchmark(model, test_dir=test_dir, device=device, output_dir=args.output_dir)

        # Run sweep on default canonical factory frame if available
        sample_img = "test_samples/abandonedfactory/000300_left.png"
        sample_gt = "test_samples/abandonedfactory/000300_left_depth.npy"
        if os.path.exists(sample_img):
            sweep_out = os.path.join(args.output_dir, "fig_dino_multi_fov_sweep.png")
            run_fov_sweep(model, sample_img, sample_gt, output_path=sweep_out, device=device)


if __name__ == "__main__":
    main()
