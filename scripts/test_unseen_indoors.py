"""
Comprehensive Evaluation of Trained Dioptra-DINO on Unseen Indoor Environments.

Evaluates the 40-epoch trained checkpoint (outputs/dioptra_dino_best.pt) on:
  1. test_samples/abandonedfactory (held-out indoor warehouse scene)
  2. test_samples/abandonedfactory_night (extreme low-light warehouse scene)
  3. test_samples/abandonedfactory_hard (complex trajectory warehouse scene)
  4. test_samples/hospital (completely unseen hospital indoor environment)
  5. test_samples/p011_benchmark (long-range held-out trajectory)

Also generates:
  - Multi-FOV Sweep figure on unseen indoor sample (Figure 8 style across 50°-100° FOVs)
  - Qualitative comparison grid (RGB | Ground Truth | Prediction | Error Heatmap)
"""

import os
import sys
import glob
import math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import __main__
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
setattr(__main__, "DioptraDINOConfig", DioptraDINOConfig)

def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 80.0):
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & (pred > min_depth) & (pred < max_depth) & np.isfinite(pred)
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

def preprocess_sample(img_path: str, gt_path: str, img_size: int = 224, device: torch.device = torch.device("cpu")):
    raw_img = Image.open(img_path).convert("RGB")
    W_orig, H_orig = raw_img.size
    min_side = min(H_orig, W_orig)
    top = (H_orig - min_side) // 2
    left = (W_orig - min_side) // 2

    raw_cropped = raw_img.crop((left, top, left + min_side, top + min_side))
    img_resized = raw_cropped.resize((img_size, img_size), Image.Resampling.BILINEAR)

    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    input_t = ((img_t - mean) / std).unsqueeze(0).to(device)

    gt_np = None
    if gt_path and os.path.exists(gt_path):
        if gt_path.endswith(".png"):
            gt_raw = np.array(Image.open(gt_path), dtype=np.float32)
            gt_arr = gt_raw / 1000.0 if gt_raw.max() > 250.0 else gt_raw
        else:
            gt_arr = np.load(gt_path).astype(np.float32)
        gt_t = torch.from_numpy(gt_arr)
        gt_cropped = gt_t[top:top + min_side, left:left + min_side]
        gt_depth = F.interpolate(gt_cropped.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode="nearest").squeeze(0).squeeze(0)
        gt_np = gt_depth.numpy()

    fx_native = 320.0 * (float(img_size) / float(min_side))
    K_native = torch.tensor([[[fx_native, 0.0, img_size / 2.0],
                             [0.0, fx_native, img_size / 2.0],
                             [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

    return input_t, gt_np, img_np, K_native

def load_checkpoint(ckpt_path: str, device: torch.device):
    print(f"[Dioptra-DINO] Loading checkpoint from: {ckpt_path}")
    cfg = DioptraDINOConfig(freeze_backbone=False)
    model = DioptraDINO(cfg).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    cleaned_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state, strict=False)
    model.eval()
    return model, state.get("epoch", "Unknown")

def run_multi_fov_sweep(model, img_path, gt_path, out_path, device):
    print(f"[Dioptra-DINO] Rendering Multi-FOV sweep on unseen sample: {img_path}")
    fov_degrees = [50, 60, 70, 80, 90, 100]
    preds = []
    ratios = []

    for fov_deg in fov_degrees:
        fov_rad = math.radians(fov_deg)
        f_val = 112.0 / math.tan(fov_rad / 2.0)
        K = torch.tensor([[[f_val, 0.0, 112.0],
                           [0.0, f_val, 112.0],
                           [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)
        inp, gt_np, img_np, _ = preprocess_sample(img_path, gt_path, img_size=224, device=device)
        with torch.no_grad():
            pred = model(inp, K).squeeze().cpu().numpy()
        preds.append(pred)
        if gt_np is not None:
            mask = (gt_np > 0.1) & (gt_np < 80.0) & np.isfinite(gt_np) & (pred > 0.1) & (pred < 80.0) & np.isfinite(pred)
            if mask.any():
                ratios.append(float(np.median(pred[mask]) / np.median(gt_np[mask])))
            else:
                ratios.append(1.0)

    fig, axes = plt.subplots(2, len(fov_degrees), figsize=(3.2 * len(fov_degrees), 6.5))
    fig.suptitle("Dioptra-DINO: Focal Equivariance Sweep on Unseen Indoor Environment", fontsize=14, fontweight="bold", y=0.98)

    for idx, fov_deg in enumerate(fov_degrees):
        pred = preds[idx]
        ratio = ratios[idx] if ratios else 1.0
        axes[0, idx].imshow(img_np)
        axes[0, idx].set_title(f"FOV: {fov_deg}°\nScale Ratio: {ratio:.3f}", fontsize=11, fontweight="bold")
        axes[0, idx].axis("off")

        im = axes[1, idx].imshow(pred, cmap="inferno", vmin=0.2, vmax=15.0)
        axes[1, idx].set_title(f"Pred Range: [{pred.min():.1f}, {pred.max():.1f}]m", fontsize=9)
        axes[1, idx].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Dioptra-DINO] Multi-FOV sweep saved to: {out_path}")

def run_qualitative_grid(model, sample_pairs, out_path, device):
    print(f"[Dioptra-DINO] Rendering qualitative evaluation grid ({len(sample_pairs)} samples)...")
    fig, axes = plt.subplots(len(sample_pairs), 4, figsize=(14, 3.2 * len(sample_pairs)))
    fig.suptitle("Dioptra-DINO: Zero-Shot Indoor Qualitative Evaluation", fontsize=15, fontweight="bold", y=1.00)

    cols = ["Input RGB", "Ground Truth (Metric)", "Dioptra-DINO (Ours)", "Absolute Error Heatmap"]
    for col_idx, col_name in enumerate(cols):
        axes[0, col_idx].set_title(col_name, fontsize=12, fontweight="bold", pad=8)

    for row_idx, (scene_name, img_path, gt_path) in enumerate(sample_pairs):
        inp, gt_np, img_np, K = preprocess_sample(img_path, gt_path, img_size=224, device=device)
        with torch.no_grad():
            pred = model(inp, K).squeeze().cpu().numpy()

        vmax = max(gt_np[gt_np < 80].max() if (gt_np is not None and (gt_np < 80).any()) else 10.0, 10.0)
        vmax = min(vmax, 25.0)

        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_ylabel(scene_name, fontsize=10, fontweight="bold")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        axes[row_idx, 1].imshow(gt_np, cmap="inferno", vmin=0.2, vmax=vmax)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(pred, cmap="inferno", vmin=0.2, vmax=vmax)
        axes[row_idx, 2].axis("off")

        diff = np.abs(pred - gt_np)
        diff[~np.isfinite(gt_np) | (gt_np <= 0.1)] = 0.0
        im_err = axes[row_idx, 3].imshow(diff, cmap="magma", vmin=0.0, vmax=1.5)
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Dioptra-DINO] Qualitative grid saved to: {out_path}")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"======================================================================")
    print(f" DIOPTRA-DINO ZERO-SHOT INDOOR GENERALIZATION BENCHMARK")
    print(f" Device: {device}")
    print(f"======================================================================")

    ckpt_path = "outputs/dioptra_dino_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    model, epoch = load_checkpoint(ckpt_path, device)
    print(f"Model checkpoint successfully loaded! Trained Epoch: {epoch}")

    # Discover test scenes in test_samples/
    scenes = {
        "abandonedfactory (Day - Held Out)": "test_samples/abandonedfactory",
        "abandonedfactory_night (Low-Light)": "test_samples/abandonedfactory_night",
        "abandonedfactory_hard (Fast Motion)": "test_samples/abandonedfactory_hard",
        "hospital (Unseen Environment)": "test_samples/hospital",
        "p011_benchmark (Long Trajectory)": "test_samples/p011_benchmark",
    }

    all_metrics = {}
    grid_samples = []

    for scene_label, scene_dir in scenes.items():
        if not os.path.exists(scene_dir):
            continue
        png_files = sorted(glob.glob(os.path.join(scene_dir, "*_left.png")) + glob.glob(os.path.join(scene_dir, "*.png")))
        # Filter out depth or right camera images
        png_files = [p for p in png_files if "depth" not in p and "right" not in p and "rcam" not in p]

        pairs = []
        for p in png_files:
            base = p.replace(".png", "")
            cand_gts = [f"{base}_depth.npy", f"{base}.npy", p.replace("_left.png", "_left_depth.npy")]
            for g in cand_gts:
                if os.path.exists(g):
                    pairs.append((p, g))
                    break

        if not pairs:
            continue

        print(f"\n--- Evaluating Scene: {scene_label} ({len(pairs)} pairs) ---")
        scene_m = []
        for img_p, gt_p in pairs:
            inp, gt_np, _, K = preprocess_sample(img_p, gt_p, img_size=224, device=device)
            with torch.no_grad():
                pred = model(inp, K).squeeze().cpu().numpy()
            m = compute_metrics(pred, gt_np)
            scene_m.append(m)

        mean_absrel = np.mean([m["abs_rel"] for m in scene_m])
        mean_rmse = np.mean([m["rmse"] for m in scene_m])
        mean_a1 = np.mean([m["a1"] for m in scene_m]) * 100
        mean_a2 = np.mean([m["a2"] for m in scene_m]) * 100
        mean_scale = np.mean([m["scale_ratio"] for m in scene_m])

        all_metrics[scene_label] = {
            "abs_rel": mean_absrel,
            "rmse": mean_rmse,
            "a1": mean_a1,
            "a2": mean_a2,
            "scale_ratio": mean_scale,
            "count": len(scene_m)
        }

        print(f"  AbsRel Error     : {mean_absrel:.4f}")
        print(f"  RMSE (meters)    : {mean_rmse:.3f} m")
        print(f"  δ < 1.25 Acc     : {mean_a1:.2f}%")
        print(f"  δ < 1.25² Acc    : {mean_a2:.2f}%")
        print(f"  Median Scale     : {mean_scale:.4f}")

        # Pick first sample for grid
        if pairs:
            grid_samples.append((scene_label.split(" (")[0], pairs[0][0], pairs[0][1]))

    # Print summary table
    print("\n" + "=" * 78)
    print(" SUMMARY BENCHMARK TABLE ACROSS ALL UNSEEN INDOOR ENVIRONMENTS")
    print("=" * 78)
    print(f"{'Scene':<38} | {'AbsRel':<8} | {'RMSE (m)':<8} | {'δ < 1.25':<9} | {'Scale Ratio':<10}")
    print("-" * 78)
    for s_name, m in all_metrics.items():
        print(f"{s_name:<38} | {m['abs_rel']:<8.4f} | {m['rmse']:<8.3f} | {m['a1']:<8.2f}% | {m['scale_ratio']:<10.4f}")
    print("=" * 78)

    # Multi-FOV sweep on first sample of abandonedfactory
    sweep_img = "test_samples/abandonedfactory/000300_left.png"
    sweep_gt = "test_samples/abandonedfactory/000300_left_depth.npy"
    if os.path.exists(sweep_img) and os.path.exists(sweep_gt):
        run_multi_fov_sweep(model, sweep_img, sweep_gt, "outputs/unseen_multi_fov_sweep.png", device)

    # Qualitative comparison grid
    if grid_samples:
        run_qualitative_grid(model, grid_samples, "outputs/unseen_indoors_eval_grid.png", device)

    print("\n>>> ALL UNSEEN INDOOR EVALUATIONS COMPLETE! <<<")

if __name__ == "__main__":
    main()
