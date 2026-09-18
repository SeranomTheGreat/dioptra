"""
Comprehensive Dioptra-DINO Ablation Suite Across 200 Unseen Benchmark Images.

Evaluates:
Group A: Architectural Component Knockouts on Headline Model (Epoch 40, outputs/dioptra_dino_best.pt):
  1. Full Headline Model (Trivision Ray FiLM + ARA Attention Bias + DPT Head)
  2. w/o Angular Residual Attention Bias (ara_gate = 0.0: standard self-attention)
  3. w/o ARA Refinement Block entirely (bypassing layer)
  4. w/o Trivision Frustum Triplet (Center-Ray Only: [rc, rc, rc])
  5. w/o Ray Positional Modulation (bypassing FiLM ray injection)

Group B: Training Epoch Progression:
  6. Dioptra-DINO Epoch 13 (outputs_dino/dioptra_dino_epoch_13.pt)
  7. Dioptra-DINO Epoch 26 (outputs_dino/dioptra_dino_epoch26_best.pt)
  8. Dioptra-DINO Epoch 40 (Best Headline, outputs/dioptra_dino_best.pt)

Computes: AbsRel, SqRel, RMSE, RMSE Log, delta1, delta2, delta3, Median Scale Ratio.
Generates:
  - JSON results: outputs/comprehensive_dino_ablations_200.json
  - Visualization plot: paper/figures/fig_ablation_component_breakdown.png
"""

import os
import sys
import glob
import time
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dioptra_dino
import __main__
__main__.DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD

def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 80.0):
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
    }

def preprocess_sample(img_path: str, gt_path: str, img_size: int = 224, device: torch.device = torch.device("cpu")):
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

    # Standard TartanAir camera calibration matrix scaled to img_size
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

def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state)
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return state.get("epoch", "N/A")

def evaluate_variant(model: nn.Module, pairs: list, device: torch.device, mode: str = "full"):
    m_list = []
    
    # Store original methods if monkey patching
    orig_unproject = model.ray_modulation._unproject_rays if model.ray_modulation is not None else None
    orig_ara_refine = model.ara_refine
    orig_ray_mod = model.ray_modulation

    try:
        if mode == "center_only":
            def center_unproject(intrinsics, is_flipped=None, num_tokens=256):
                fourier, rc = orig_unproject(intrinsics, is_flipped, num_tokens)
                all_rays = torch.cat([rc, rc, rc], dim=-1)
                freq_bands = 2.0 ** torch.arange(model.ray_modulation.num_freqs, device=intrinsics.device, dtype=intrinsics.dtype) * np.pi
                prod = all_rays.unsqueeze(-1) * freq_bands.view(1, 1, 1, -1)
                fourier_center = torch.cat([torch.sin(prod), torch.cos(prod)], dim=-1).flatten(2)
                return fourier_center, rc
            model.ray_modulation._unproject_rays = center_unproject

        elif mode == "no_ray":
            model.ray_modulation = None

        elif mode == "no_ara_block":
            model.ara_refine = None

        for img_p, gt_p in pairs:
            inp, gt_np, K = preprocess_sample(img_p, gt_p, img_size=224, device=device)
            with torch.no_grad():
                if mode == "no_ara_bias":
                    pred = model(inp, K, ara_gate=0.0).squeeze().cpu().numpy()
                else:
                    pred = model(inp, K, ara_gate=1.0).squeeze().cpu().numpy()
            m_list.append(compute_metrics(pred, gt_np))

    finally:
        # Restore original structure
        if orig_ray_mod is not None:
            model.ray_modulation = orig_ray_mod
            if orig_unproject is not None:
                model.ray_modulation._unproject_rays = orig_unproject
        if orig_ara_refine is not None:
            model.ara_refine = orig_ara_refine

    # Aggregate metrics
    return {
        "abs_rel": float(np.mean([m["abs_rel"] for m in m_list])),
        "sq_rel": float(np.mean([m["sq_rel"] for m in m_list])),
        "rmse": float(np.mean([m["rmse"] for m in m_list])),
        "rmse_log": float(np.mean([m["rmse_log"] for m in m_list])),
        "delta1": float(np.mean([m["a1"] for m in m_list]) * 100),
        "delta2": float(np.mean([m["a2"] for m in m_list]) * 100),
        "delta3": float(np.mean([m["a3"] for m in m_list]) * 100),
        "scale_ratio": float(np.median([m["scale_ratio"] for m in m_list])),
    }

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 80)
    print(" DIOPTRA-DINO: COMPREHENSIVE ABLATION BENCHMARK (200 UNSEEN IMAGES)")
    print(f" Compute Device: {device}")
    print("=" * 80)

    # 1. Collect all 200 benchmark pairs
    data_dir = "test_samples/unseen_200_abandonedfactory"
    png_files = sorted(glob.glob(os.path.join(data_dir, "image_left", "*.png")))
    pairs = []
    for p in png_files:
        base = os.path.basename(p).replace("_left.png", "")
        gt = os.path.join(data_dir, "depth_left", f"{base}_left_depth.npy")
        if os.path.exists(gt):
            pairs.append((p, gt))
    pairs = pairs[:200]
    print(f"[Data] Loaded {len(pairs)} continuous test image/depth pairs.\n")

    # Define ablation variants
    variants = [
        # --- Group A: Component Knockouts on Headline Model (Epoch 40) ---
        {
            "id": "full_headline_ep40",
            "name": "Dioptra-DINO Headline (Epoch 40)",
            "group": "Component Knockouts",
            "ckpt": "outputs/dioptra_dino_best.pt",
            "mode": "full",
            "desc": "Full architecture (Trivision Ray FiLM + ARA Bias + DPT Head)",
        },
        {
            "id": "no_ara_bias",
            "name": "w/o ARA Angular Bias (ara_gate=0.0)",
            "group": "Component Knockouts",
            "ckpt": "outputs/dioptra_dino_best.pt",
            "mode": "no_ara_bias",
            "desc": "Angular Residual Attention penalty disabled (standard self-attention)",
        },
        {
            "id": "no_ara_block",
            "name": "w/o ARA Refinement Block",
            "group": "Component Knockouts",
            "ckpt": "outputs/dioptra_dino_best.pt",
            "mode": "no_ara_block",
            "desc": "Geometric ARA transformer refinement block bypassed entirely",
        },
        {
            "id": "center_ray_only",
            "name": "w/o Trivision Triplet (Center-Ray Only)",
            "group": "Component Knockouts",
            "ckpt": "outputs/dioptra_dino_best.pt",
            "mode": "center_only",
            "desc": "Ray unprojection uses center ray [rc, rc, rc] without corner rays",
        },
        {
            "id": "no_ray_modulation",
            "name": "w/o Ray Positional Modulation",
            "group": "Component Knockouts",
            "ckpt": "outputs/dioptra_dino_best.pt",
            "mode": "no_ray",
            "desc": "Camera ray unprojection and FiLM modulation completely disabled",
        },
        # --- Group B: Training Epoch Progression ---
        {
            "id": "dino_ep13",
            "name": "Dioptra-DINO (Epoch 13)",
            "group": "Training Progression",
            "ckpt": "outputs_dino/dioptra_dino_epoch_13.pt",
            "mode": "full",
            "desc": "Intermediate checkpoint after 13 epochs of training",
        },
        {
            "id": "dino_ep26",
            "name": "Dioptra-DINO (Epoch 26)",
            "group": "Training Progression",
            "ckpt": "outputs_dino/dioptra_dino_epoch26_best.pt",
            "mode": "full",
            "desc": "Intermediate best checkpoint after 26 epochs of training",
        },
    ]

    all_results = []
    base_cfg = DioptraDINOConfig(freeze_backbone=False)

    for v in variants:
        print(f"--> Benchmarking: {v['name']} ({v['desc']})")
        t0 = time.time()

        # Initialize and load model
        model = DioptraDINO(base_cfg).to(device)
        ep = load_checkpoint(model, v["ckpt"], device)

        # Run evaluation
        metrics = evaluate_variant(model, pairs, device, mode=v["mode"])
        dt = time.time() - t0

        metrics["id"] = v["id"]
        metrics["name"] = v["name"]
        metrics["group"] = v["group"]
        metrics["checkpoint"] = v["ckpt"]
        metrics["epoch"] = ep
        metrics["description"] = v["desc"]
        metrics["eval_time_sec"] = round(dt, 2)
        all_results.append(metrics)

        print(f"    AbsRel: {metrics['abs_rel']:.4f} | SqRel: {metrics['sq_rel']:.4f} | RMSE: {metrics['rmse']:.3f}m | delta1: {metrics['delta1']:.2f}% | Scale: {metrics['scale_ratio']:.4f} ({dt:.1f}s)")

    # Save to JSON
    out_json = "outputs/comprehensive_dino_ablations_200.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] All ablation results saved to: {out_json}")

    # Generate Publication-Grade Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Dioptra-DINO: Comprehensive Ablation Study (200 Unseen Benchmark Frames)", fontsize=13, fontweight="bold")

    labels = [
        "Headline (Ep 40)",
        "w/o ARA Bias",
        "w/o ARA Block",
        "Center-Ray Only",
        "w/o Ray Mod",
        "Epoch 13",
        "Epoch 26"
    ]
    colors = ["#1b7837", "#2b5c8f", "#4575b4", "#d73027", "#a50026", "#762a83", "#9970ab"]

    abs_rels = [r["abs_rel"] for r in all_results]
    delta1s = [r["delta1"] for r in all_results]
    scales = [r["scale_ratio"] for r in all_results]

    # Subplot 1: AbsRel
    bars1 = axes[0].bar(labels, abs_rels, color=colors, edgecolor="black", alpha=0.9)
    axes[0].set_title("Absolute Relative Error (AbsRel) ↓", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("AbsRel (lower is better)")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars1:
        y = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2.0, y + 0.005, f"{y:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Subplot 2: Threshold Accuracy delta1
    bars2 = axes[1].bar(labels, delta1s, color=colors, edgecolor="black", alpha=0.9)
    axes[1].set_title("Threshold Accuracy (δ < 1.25) ↑", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("δ < 1.25 % (higher is better)")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars2:
        y = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2.0, y + 0.8, f"{y:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Subplot 3: Metric Scale Ratio
    bars3 = axes[2].bar(labels, scales, color=colors, edgecolor="black", alpha=0.9)
    axes[2].axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Ideal Metric Scale (1.000)")
    axes[2].set_title("Metric Scale Ratio (s / s_gt) → 1.000", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Median Scale Ratio")
    axes[2].tick_params(axis="x", rotation=40)
    axes[2].legend(loc="upper left")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars3:
        y = b.get_height()
        axes[2].text(b.get_x() + b.get_width()/2.0, y + 0.02, f"{y:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out_fig = "paper/figures/fig_ablation_component_breakdown.png"
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Ablation comparison plot saved to: {out_fig}")

    print("\n" + "=" * 80)
    print(" ABLATION BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Variant Name':<42} | {'AbsRel':<7} | {'SqRel':<7} | {'RMSE (m)':<8} | {'delta1':<7} | {'Scale':<7}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['name']:<42} | {r['abs_rel']:<7.4f} | {r['sq_rel']:<7.4f} | {r['rmse']:<8.3f} | {r['delta1']:<6.2f}% | {r['scale_ratio']:<7.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
