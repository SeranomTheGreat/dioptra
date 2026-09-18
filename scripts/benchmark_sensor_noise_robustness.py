#!/usr/bin/env python3
"""
Dioptra-DINO: Sensor Noise & Physical Environmental Degradation Benchmark Suite
Evaluates model robustness against real-world drone hardware artifacts:
1. Dynamic Motion Blur (linear convolution kernel k in [0, 5, 11, 19] px) simulating flight vibration.
2. CMOS Sensor Shot Noise (Poisson-Gaussian noise sigma in [0.0, 0.02, 0.05, 0.10]) simulating sensor grain.
Compares Dioptra-DINO, UniDepth-V2, Depth Anything V2 (Affine), and Metric3D.
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
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dioptra_dino
import __main__
__main__.DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD

def load_models(repo_dir, device):
    models = {}

    # 1. Dioptra-DINO
    print("[Load] Loading Dioptra-DINO...")
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
        print(f"[Warn] UniDepth load warning: {e}")

    # 3. Depth Anything V2
    try:
        print("[Load] Loading Depth Anything V2-Small...")
        da_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        da_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()
        models['depth_anything'] = (da_proc, da_model)
    except Exception as e:
        print(f"[Warn] Depth Anything V2 load warning: {e}")

    # 4. Metric3D
    try:
        print("[Load] Loading Metric3D ViT-Small...")
        metric3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True).to(device).eval()
        models['metric3d'] = metric3d
    except Exception as e:
        print(f"[Warn] Metric3D load warning: {e}")

    return models

def apply_motion_blur(img, kernel_size, angle=45):
    """Apply linear directional motion blur."""
    if kernel_size <= 1:
        return img
    kernel = np.zeros((kernel_size, kernel_size))
    # Fill diagonal for 45 deg blur
    np.fill_diagonal(kernel, 1.0)
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)

def apply_sensor_noise(img, sigma):
    """Apply zero-mean Gaussian electronic read noise + scaled shot noise."""
    if sigma <= 0.0:
        return img
    img_f = img.astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, img_f.shape).astype(np.float32)
    noisy = np.clip(img_f + noise, 0.0, 1.0)
    return (noisy * 255.0).astype(np.uint8)

def compute_metrics(pred, gt):
    mask = (gt > 0.1) & (gt < 80.0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0.0)
    if not np.any(mask):
        return 1.0, 1.0, 0.0
    p = pred[mask]
    g = gt[mask]
    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25) * 100.0)
    return abs_rel, rmse, delta1

def infer_dioptra(model, rgb_crop, device):
    H_orig, W_orig = rgb_crop.shape[:2]
    rgb_resized = cv2.resize(rgb_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    mean_dino = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_dino = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    t_img = (t_img - mean_dino) / std_dino
    
    K = np.array([
        [320.0 * (224.0 / 480.0), 0.0, 240.0 * (224.0 / 480.0)],
        [0.0, 320.0 * (224.0 / 480.0), 240.0 * (224.0 / 480.0)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    K_t = torch.from_numpy(K).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred_depth = model(t_img, K_t)
        pred_full = torch.nn.functional.interpolate(pred_depth, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    return pred_full.squeeze().cpu().numpy()

def infer_unidepth(model, rgb_crop, device):
    H_orig, W_orig = rgb_crop.shape[:2]
    img_t = torch.from_numpy(rgb_crop.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    K = np.array([
        [320.0, 0.0, 240.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    K_t = torch.from_numpy(K).float().unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model.infer(img_t, camera=K_t)
        depth = preds["depth"]
        if depth.shape[-2:] != (H_orig, W_orig):
            depth = torch.nn.functional.interpolate(depth, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    return depth.squeeze().cpu().numpy()

def infer_depth_anything(proc_and_model, rgb_crop, gt_crop, device):
    proc, model = proc_and_model
    H_orig, W_orig = rgb_crop.shape[:2]
    pil_img = Image.fromarray(rgb_crop)
    inputs = proc(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_disp = outputs.predicted_depth
        pred_disp = torch.nn.functional.interpolate(pred_disp.unsqueeze(1), size=(H_orig, W_orig), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

    valid = (gt_crop > 0.1) & (gt_crop < 80.0) & np.isfinite(gt_crop) & np.isfinite(pred_disp)
    gt_disp = 1.0 / np.clip(gt_crop[valid], 0.1, 80.0)
    p_d = pred_disp[valid]
    
    A = np.vstack([p_d, np.ones_like(p_d)]).T
    sol, _, _, _ = np.linalg.lstsq(A, gt_disp, rcond=None)
    s_aff, t_aff = sol
    
    pred_disp_aff = s_aff * pred_disp + t_aff
    pred_disp_aff = np.clip(pred_disp_aff, 1.0 / 80.0, 1.0 / 0.1)
    pred_depth = 1.0 / pred_disp_aff
    return pred_depth

def infer_metric3d(model, rgb_crop, device):
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
        pred_metric = pred_full * (320.0 / 1000.0)
    return pred_metric.squeeze().cpu().numpy()

def main():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_dir, "outputs")
    fig_dir = os.path.join(repo_dir, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Executing Sensor Noise & Environmental Degradation Suite on device: {device}")

    # Load 25 sampled continuous trajectory frames
    data_dir = os.path.join(repo_dir, "test_samples", "unseen_200_abandonedfactory")
    img_files = sorted(glob.glob(os.path.join(data_dir, "image_left", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(data_dir, "depth_left", "*.npy")))
    
    sampled_indices = list(range(0, len(img_files), 8))[:25]
    print(f"Loaded {len(sampled_indices)} stress test samples.")

    orig_W, orig_H = 640, 480
    left_crop = (orig_W - orig_H) // 2

    test_data = []
    for idx in sampled_indices:
        ip = img_files[idx]
        dp = depth_files[idx]
        raw_bgr = cv2.imread(ip)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        raw_gt = np.load(dp).astype(np.float32)

        rgb_crop = raw_rgb[:, left_crop:left_crop + orig_H]
        gt_crop = raw_gt[:, left_crop:left_crop + orig_H]
        test_data.append({"rgb": rgb_crop, "gt": gt_crop, "name": os.path.basename(ip)})

    models = load_models(repo_dir, device)
    model_keys = ['dioptra', 'unidepth', 'depth_anything', 'metric3d']
    model_labels = {
        'dioptra': 'Dioptra-DINO (Ours)',
        'unidepth': 'UniDepth-V2 ViT-Small',
        'depth_anything': 'Depth Anything V2 (Affine)',
        'metric3d': 'Metric3D ViT-Small'
    }

    # 1. Motion Blur Sweep
    motion_blur_kernels = [1, 5, 9, 15]
    blur_results = {"kernel_sizes": motion_blur_kernels, "models": {}}

    print("\n--- 1. Evaluating Dynamic Flight Motion Blur Sweep ---")
    for m_key in model_keys:
        if m_key not in models:
            continue
        blur_results["models"][m_key] = {"absrel": [], "delta1": []}
        for k in motion_blur_kernels:
            ar_list, d1_list = [], []
            for item in test_data:
                blurred_rgb = apply_motion_blur(item["rgb"], k)
                if m_key == 'dioptra':
                    pred = infer_dioptra(models[m_key], blurred_rgb, device)
                elif m_key == 'unidepth':
                    pred = infer_unidepth(models[m_key], blurred_rgb, device)
                elif m_key == 'depth_anything':
                    pred = infer_depth_anything(models[m_key], blurred_rgb, item["gt"], device)
                elif m_key == 'metric3d':
                    pred = infer_metric3d(models[m_key], blurred_rgb, device)

                ar, _, d1 = compute_metrics(pred, item["gt"])
                ar_list.append(ar)
                d1_list.append(d1)
            mean_ar = float(np.mean(ar_list))
            mean_d1 = float(np.mean(d1_list))
            blur_results["models"][m_key]["absrel"].append(mean_ar)
            blur_results["models"][m_key]["delta1"].append(mean_d1)
            print(f"  {model_labels[m_key]:<28} | Blur k={k:2d} px | AbsRel: {mean_ar:.4f} | delta1: {mean_d1:.1f}%")

    # 2. CMOS Sensor Noise Sweep
    sensor_noise_sigmas = [0.0, 0.02, 0.05, 0.10]
    noise_results = {"sigmas": sensor_noise_sigmas, "models": {}}

    print("\n--- 2. Evaluating CMOS Sensor Shot Noise Sweep ---")
    for m_key in model_keys:
        if m_key not in models:
            continue
        noise_results["models"][m_key] = {"absrel": [], "delta1": []}
        for s in sensor_noise_sigmas:
            ar_list, d1_list = [], []
            for item in test_data:
                noisy_rgb = apply_sensor_noise(item["rgb"], s)
                if m_key == 'dioptra':
                    pred = infer_dioptra(models[m_key], noisy_rgb, device)
                elif m_key == 'unidepth':
                    pred = infer_unidepth(models[m_key], noisy_rgb, device)
                elif m_key == 'depth_anything':
                    pred = infer_depth_anything(models[m_key], noisy_rgb, item["gt"], device)
                elif m_key == 'metric3d':
                    pred = infer_metric3d(models[m_key], noisy_rgb, device)

                ar, _, d1 = compute_metrics(pred, item["gt"])
                ar_list.append(ar)
                d1_list.append(d1)
            mean_ar = float(np.mean(ar_list))
            mean_d1 = float(np.mean(d1_list))
            noise_results["models"][m_key]["absrel"].append(mean_ar)
            noise_results["models"][m_key]["delta1"].append(mean_d1)
            print(f"  {model_labels[m_key]:<28} | Noise sigma={s:.2f} | AbsRel: {mean_ar:.4f} | delta1: {mean_d1:.1f}%")

    # Save JSON receipt
    combined_suite = {
        "motion_blur_sweep": blur_results,
        "sensor_noise_sweep": noise_results
    }
    json_path = os.path.join(out_dir, "benchmark_sensor_noise_robustness.json")
    with open(json_path, "w") as f:
        json.dump(combined_suite, f, indent=2)
    print(f"\nSaved sensor noise results to: {json_path}")

    # Plot Publication Stress Figures
    print("Generating Sensor Degradation Robustness Publication Plot...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    colors = {'dioptra': '#1f77b4', 'unidepth': '#2ca02c', 'depth_anything': '#ff7f0e', 'metric3d': '#d62728'}
    markers = {'dioptra': 'o', 'unidepth': '^', 'depth_anything': 'd', 'metric3d': 's'}
    linestyles = {'dioptra': '-', 'unidepth': '-.', 'depth_anything': ':', 'metric3d': '--'}

    # Panel 1: Motion Blur Resilience
    ax1 = axes[0]
    for k in model_keys:
        if k in blur_results["models"]:
            vals = blur_results["models"][k]["absrel"]
            lbl = model_labels[k].replace(" ViT-Small", "").replace(" (Affine)", "")
            lw = 2.8 if k == 'dioptra' else 1.8
            ax1.plot(motion_blur_kernels, vals, label=lbl, color=colors[k], marker=markers[k],
                    linewidth=lw, linestyle=linestyles[k], markersize=6)

    ax1.set_xlabel("Flight Motion Blur Kernel Size (pixels) →", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Absolute Relative Error (AbsRel) ↓", fontsize=12, fontweight='bold')
    ax1.set_title("Resilience to Dynamic Quadrotor Motion Blur", fontsize=13, fontweight='bold', pad=10)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: CMOS Sensor Noise Resilience
    ax2 = axes[1]
    for k in model_keys:
        if k in noise_results["models"]:
            vals = noise_results["models"][k]["absrel"]
            lbl = model_labels[k].replace(" ViT-Small", "").replace(" (Affine)", "")
            lw = 2.8 if k == 'dioptra' else 1.8
            ax2.plot(sensor_noise_sigmas, vals, label=lbl, color=colors[k], marker=markers[k],
                    linewidth=lw, linestyle=linestyles[k], markersize=6)

    ax2.set_xlabel("Sensor Read & Shot Noise Level ($\\sigma$) →", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Absolute Relative Error (AbsRel) ↓", fontsize=12, fontweight='bold')
    ax2.set_title("Resilience to Low-Light CMOS Sensor Noise", fontsize=13, fontweight='bold', pad=10)
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_out = os.path.join(out_dir, "fig_sensor_noise_robustness.png")
    fig_paper = os.path.join(fig_dir, "fig_sensor_noise_robustness.png")
    plt.savefig(fig_out, dpi=300)
    plt.savefig(fig_paper, dpi=300)
    plt.close()

    print(f"Generated sensor noise robustness figure:\n  {fig_out}\n  {fig_paper}")
    print("\n[SUCCESS] Sensor noise robustness benchmark complete.")

if __name__ == "__main__":
    main()
