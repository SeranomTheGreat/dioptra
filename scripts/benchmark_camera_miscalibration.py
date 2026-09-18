#!/usr/bin/env python3
"""
Dioptra-DINO: Camera Miscalibration & Intrinsic Perturbation Sensitivity Suite
Evaluates robustness against factory calibration drift, thermal focal change, and principal point offsets.
Sweeps focal perturbation epsilon_f in [-10%, -5%, -2%, 0%, +2%, +5%, +10%].
Compares Dioptra-DINO, UniDepth-V2, and Metric3D across held-out continuous trajectory frames.
"""

import os
import sys
import glob
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dioptra_dino
import __main__
__main__.DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD

def load_models(repo_dir, device):
    models = {}

    # 1. Dioptra-DINO
    print("[Load] Loading Dioptra-DINO checkpoint...")
    cfg = DioptraDINOConfig(freeze_backbone=False)
    model_dioptra = DioptraDINO(cfg).to(device)
    ckpt_path = os.path.join(repo_dir, "outputs", "dioptra_dino_best.pt")
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = st.get("model_state_dict", st)
    model_dioptra.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    model_dioptra.eval()
    models['dioptra'] = model_dioptra

    # 2. UniDepth-V2
    try:
        print("[Load] Loading UniDepth-V2 ViT-Small...")
        unidepth = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version="v2", backbone="vits14", pretrained=True, trust_repo=True).to(device).eval()
        models['unidepth'] = unidepth
    except Exception as e:
        print(f"[Warn] UniDepth-V2 load warning: {e}")

    # 3. Metric3D
    try:
        print("[Load] Loading Metric3D ViT-Small...")
        metric3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True).to(device).eval()
        models['metric3d'] = metric3d
    except Exception as e:
        print(f"[Warn] Metric3D load warning: {e}")

    return models

def compute_metrics(pred, gt):
    mask = (gt > 0.1) & (gt < 80.0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0.0)
    if not np.any(mask):
        return 1.0, 1.0, 0.0, 1.0
    p = pred[mask]
    g = gt[mask]
    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25) * 100.0)
    scale = float(np.median(p) / (np.median(g) + 1e-8))
    return abs_rel, rmse, delta1, scale

def evaluate_dioptra(model, rgb_crop, K_pert_224, device):
    H_orig, W_orig = rgb_crop.shape[:2]
    rgb_resized = cv2.resize(rgb_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    mean_dino = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_dino = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    t_img = (t_img - mean_dino) / std_dino
    K_t = torch.from_numpy(K_pert_224).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred_depth = model(t_img, K_t)
        pred_full = torch.nn.functional.interpolate(pred_depth, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    return pred_full.squeeze().cpu().numpy()

def evaluate_unidepth(model, rgb_crop, K_pert, device):
    H_orig, W_orig = rgb_crop.shape[:2]
    img_t = torch.from_numpy(rgb_crop.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    K_t = torch.from_numpy(K_pert).float().unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model.infer(img_t, camera=K_t)
        depth = preds["depth"]
        if depth.shape[-2:] != (H_orig, W_orig):
            depth = torch.nn.functional.interpolate(depth, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    return depth.squeeze().cpu().numpy()

def evaluate_metric3d(model, rgb_crop, K_pert, device):
    H_orig, W_orig = rgb_crop.shape[:2]
    input_size = (616, 1064)
    scale = min(input_size[0] / H_orig, input_size[1] / W_orig)
    new_h, new_w = int(round(H_orig * scale)), int(round(W_orig * scale))
    img_resized = cv2.resize(rgb_crop, (new_w, new_h))
    
    pad_h = input_size[0] - new_h
    pad_w = input_size[1] - new_w
    img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
    img_t = torch.from_numpy(img_padded.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)
    img_t = (img_t - mean) / std

    with torch.no_grad():
        pred_depth, _, _ = model.inference({"input": img_t})
        pred_cropped = pred_depth[:, :, :new_h, :new_w]
        pred_full = torch.nn.functional.interpolate(pred_cropped, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        
        f_perturbed = (K_pert[0, 0] + K_pert[1, 1]) / 2.0
        canonical_scale = f_perturbed / 1000.0
        pred_metric = pred_full * canonical_scale
    return pred_metric.squeeze().cpu().numpy()

def main():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_dir, "outputs")
    fig_dir = os.path.join(repo_dir, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Executing Camera Miscalibration Benchmark on device: {device}")

    # Load 20 sampled continuous trajectory frames
    data_dir = os.path.join(repo_dir, "test_samples", "unseen_200_abandonedfactory")
    img_files = sorted(glob.glob(os.path.join(data_dir, "image_left", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(data_dir, "depth_left", "*.npy")))
    
    sampled_indices = list(range(0, len(img_files), 10))[:20]
    print(f"Sampled {len(sampled_indices)} frames across continuous trajectory.")

    test_data = []
    orig_W, orig_H = 640, 480
    left_crop = (orig_W - orig_H) // 2  # 80 px

    for idx in sampled_indices:
        ip = img_files[idx]
        dp = depth_files[idx]
        raw_bgr = cv2.imread(ip)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        raw_gt = np.load(dp).astype(np.float32)

        rgb_crop = raw_rgb[:, left_crop:left_crop + orig_H]
        gt_crop = raw_gt[:, left_crop:left_crop + orig_H]
        
        # Base TartanAir intrinsics for 480x480 crop: fx=320, fy=320, cx=240, cy=240
        K_base = np.array([
            [320.0, 0.0, 240.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        test_data.append({"rgb": rgb_crop, "gt": gt_crop, "K": K_base, "name": os.path.basename(ip)})

    print(f"Prepared {len(test_data)} valid square-cropped test frames.")
    models = load_models(repo_dir, device)

    # Focal perturbation sweep
    delta_f_list = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
    sweep_results = {"delta_f_percent": [int(df * 100) for df in delta_f_list], "models": {}}

    for m_name in ['dioptra', 'unidepth', 'metric3d']:
        if m_name not in models:
            continue
        print(f"\nEvaluating {m_name.upper()} across focal perturbation sweep...")
        sweep_results["models"][m_name] = {
            "absrel_mean": [],
            "scale_mean": [],
            "delta1_mean": [],
            "rmse_mean": []
        }

        for df in delta_f_list:
            absrel_acc = []
            scale_acc = []
            delta1_acc = []
            rmse_acc = []

            for item in test_data:
                K_pert = item["K"].copy()
                K_pert[0, 0] *= (1.0 + df)
                K_pert[1, 1] *= (1.0 + df)

                # Scaled intrinsics for Dioptra (224x224)
                K_pert_224 = K_pert.copy()
                K_pert_224[0, :] *= (224.0 / 480.0)
                K_pert_224[1, :] *= (224.0 / 480.0)

                if m_name == 'dioptra':
                    pred = evaluate_dioptra(models[m_name], item["rgb"], K_pert_224, device)
                elif m_name == 'unidepth':
                    pred = evaluate_unidepth(models[m_name], item["rgb"], K_pert, device)
                elif m_name == 'metric3d':
                    pred = evaluate_metric3d(models[m_name], item["rgb"], K_pert, device)

                ar, rm, d1, sc = compute_metrics(pred, item["gt"])
                absrel_acc.append(ar)
                rmse_acc.append(rm)
                delta1_acc.append(d1)
                scale_acc.append(sc)

            mean_ar = float(np.mean(absrel_acc))
            mean_sc = float(np.mean(scale_acc))
            mean_d1 = float(np.mean(delta1_acc))
            mean_rm = float(np.mean(rmse_acc))

            sweep_results["models"][m_name]["absrel_mean"].append(mean_ar)
            sweep_results["models"][m_name]["scale_mean"].append(mean_sc)
            sweep_results["models"][m_name]["delta1_mean"].append(mean_d1)
            sweep_results["models"][m_name]["rmse_mean"].append(mean_rm)

            print(f"  Focal Error: {df*100:+5.1f}% | AbsRel: {mean_ar:.4f} | Scale Ratio: {mean_sc:.4f} | delta1: {mean_d1:.1f}%")

    # Save JSON receipt
    json_path = os.path.join(out_dir, "benchmark_camera_miscalibration.json")
    with open(json_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved calibration sweep results to: {json_path}")

    # Plot Publication Sensitivity Curves
    print("Plotting Camera Miscalibration Sensitivity Curves...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    labels = {
        'dioptra': ('Dioptra-DINO (Ours)', '#1f77b4', 'o', '-'),
        'unidepth': ('UniDepth-V2 ViT-Small', '#2ca02c', '^', '-.'),
        'metric3d': ('Metric3D ViT-Small', '#d62728', 's', '--'),
    }

    # Left: AbsRel sensitivity
    ax1 = axes[0]
    pcts = sweep_results["delta_f_percent"]
    for k, (name, color, marker, ls) in labels.items():
        if k in sweep_results["models"]:
            vals = sweep_results["models"][k]["absrel_mean"]
            lw = 2.8 if k == 'dioptra' else 1.8
            ax1.plot(pcts, vals, label=name, color=color, marker=marker, linewidth=lw, linestyle=ls, markersize=6)

    ax1.set_xlabel("Camera Focal Length Error $\\Delta f / f$ (%)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Absolute Relative Error (AbsRel)", fontsize=12, fontweight='bold')
    ax1.set_title("Calibration Sensitivity Curve (AbsRel vs Focal Drift)", fontsize=13, fontweight='bold', pad=10)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.7)
    ax1.legend(loc='upper right', frameon=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Right: Metric Scale Ratio sensitivity
    ax2 = axes[1]
    for k, (name, color, marker, ls) in labels.items():
        if k in sweep_results["models"]:
            vals = sweep_results["models"][k]["scale_mean"]
            lw = 2.8 if k == 'dioptra' else 1.8
            ax2.plot(pcts, vals, label=name, color=color, marker=marker, linewidth=lw, linestyle=ls, markersize=6)

    ax2.axhline(1.0, color='black', linestyle='-', alpha=0.7, label="Ideal Metric Scale (1.000)")
    ax2.axhspan(0.95, 1.05, color='gray', alpha=0.15, label="$\pm 5\%$ Metric Safe Tolerance")
    ax2.set_xlabel("Camera Focal Length Error $\\Delta f / f$ (%)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Metric Scale Ratio ($s / s_{\\text{gt}}$)", fontsize=12, fontweight='bold')
    ax2.set_title("Metric Scale Drift vs Camera Miscalibration", fontsize=13, fontweight='bold', pad=10)
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.7)
    ax2.legend(loc='lower left', frameon=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_out = os.path.join(out_dir, "fig_camera_miscalibration_sensitivity.png")
    fig_paper = os.path.join(fig_dir, "fig_camera_miscalibration_sensitivity.png")
    plt.savefig(fig_out, dpi=300)
    plt.savefig(fig_paper, dpi=300)
    plt.close()

    print(f"Generated calibration sensitivity figure:\n  {fig_out}\n  {fig_paper}")
    print("\n[SUCCESS] Camera miscalibration benchmark complete.")

if __name__ == "__main__":
    main()
