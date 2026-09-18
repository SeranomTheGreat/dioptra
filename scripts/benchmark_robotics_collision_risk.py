#!/usr/bin/env python3
"""
Dioptra-DINO: Downstream Robotics Flight Safety & Obstacle Collision Risk Suite
Quantifies autonomous flight hazard metrics:
1. Near-Field Critical Collision Miss Rate (CMR): % of pixels where D* < 2.0m but predicted > 2.5m (catastrophic crash risk).
2. False Alarm Emergency Braking Rate (FAR): % of pixels where D* > 4.0m but predicted < 2.0m (false stop risk).
3. Near-field Safety Conservatism Bias.
Compares Dioptra-DINO, UniDepth-V2, Depth Anything V2 (Affine), and Metric3D across 50 held-out frames.
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

def compute_robotics_safety_metrics(pred, gt, near_thresh=2.0, miss_buffer=2.5, far_thresh=4.0):
    """
    Computes robotics flight collision risk metrics.
    """
    valid = (gt > 0.1) & (gt < 80.0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0.0)
    
    # Near-field obstacle mask (< 2.0m)
    near_gt_mask = valid & (gt < near_thresh)
    n_near = np.sum(near_gt_mask)
    
    if n_near > 0:
        # Critical miss: obstacle is close (<2.0m) but model predicts > 2.5m (crash hazard)
        critical_misses = near_gt_mask & (pred > miss_buffer)
        cmr = float(np.sum(critical_misses) / n_near * 100.0)
    else:
        cmr = 0.0
        
    # Open-space mask (> 4.0m)
    open_gt_mask = valid & (gt > far_thresh)
    n_open = np.sum(open_gt_mask)
    
    if n_open > 0:
        # False alarm: space is open (>4.0m) but model hallucinates obstacle < 2.0m
        false_alarms = open_gt_mask & (pred < near_thresh)
        far = float(np.sum(false_alarms) / n_open * 100.0)
    else:
        far = 0.0

    # Near-field mean error bias (positive means dangerous overestimation, negative means safe buffer)
    if n_near > 0:
        near_bias = float(np.mean(pred[near_gt_mask] - gt[near_gt_mask]))
    else:
        near_bias = 0.0

    return cmr, far, near_bias, int(n_near), int(n_open)

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
    print(f"Executing Robotics Flight Safety & Collision Risk Suite on device: {device}")

    # Load 50 sampled continuous trajectory frames
    data_dir = os.path.join(repo_dir, "test_samples", "unseen_200_abandonedfactory")
    img_files = sorted(glob.glob(os.path.join(data_dir, "image_left", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(data_dir, "depth_left", "*.npy")))
    
    sampled_indices = list(range(0, len(img_files), 4))[:50]
    print(f"Evaluating {len(sampled_indices)} frames for collision safety.")

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

    safety_results = {}
    model_keys = ['dioptra', 'unidepth', 'depth_anything', 'metric3d']
    model_labels = {
        'dioptra': 'Dioptra-DINO (Ours)',
        'unidepth': 'UniDepth-V2 ViT-Small',
        'depth_anything': 'Depth Anything V2 (Affine)',
        'metric3d': 'Metric3D ViT-Small'
    }

    # Store first sample hazard maps for visualization
    sample_hazard_vis = {}
    first_sample = test_data[0]

    for m_key in model_keys:
        if m_key not in models:
            continue
        label = model_labels[m_key]
        print(f"\nEvaluating {label} for Critical Collision Risk & False Alarms...")
        
        cmr_list = []
        far_list = []
        bias_list = []

        for idx, item in enumerate(test_data):
            if m_key == 'dioptra':
                pred = infer_dioptra(models[m_key], item["rgb"], device)
            elif m_key == 'unidepth':
                pred = infer_unidepth(models[m_key], item["rgb"], device)
            elif m_key == 'depth_anything':
                pred = infer_depth_anything(models[m_key], item["rgb"], item["gt"], device)
            elif m_key == 'metric3d':
                pred = infer_metric3d(models[m_key], item["rgb"], device)

            cmr, far, bias, n_near, n_open = compute_robotics_safety_metrics(pred, item["gt"])
            if n_near > 0:
                cmr_list.append(cmr)
                bias_list.append(bias)
            if n_open > 0:
                far_list.append(far)

            if idx == 0:
                # Hazard map: red for critical miss, yellow for false alarm, green for safe
                h_map = np.zeros((*item["gt"].shape, 3), dtype=np.uint8)
                h_map[:, :] = [30, 160, 30] # default safe green
                crit = (item["gt"] < 2.0) & (pred > 2.5) & (item["gt"] > 0.1)
                fa = (item["gt"] > 4.0) & (pred < 2.0) & (item["gt"] > 0.1)
                h_map[fa] = [255, 200, 0] # Yellow false alarm
                h_map[crit] = [255, 30, 30] # Red critical collision hazard
                sample_hazard_vis[m_key] = h_map

        mean_cmr = float(np.mean(cmr_list)) if cmr_list else 0.0
        mean_far = float(np.mean(far_list)) if far_list else 0.0
        mean_bias = float(np.mean(bias_list)) if bias_list else 0.0

        safety_results[m_key] = {
            "model_label": label,
            "critical_miss_rate_percent": mean_cmr,
            "false_alarm_rate_percent": mean_far,
            "near_field_bias_meters": mean_bias
        }

        print(f"  {label:<28} | Critical Miss Rate: {mean_cmr:5.2f}% | False Alarm Rate: {mean_far:5.2f}% | Near Bias: {mean_bias:+6.3f} m")

    # Save JSON receipt
    json_path = os.path.join(out_dir, "benchmark_robotics_collision_risk.json")
    with open(json_path, "w") as f:
        json.dump(safety_results, f, indent=2)
    print(f"\nSaved robotics safety results to: {json_path}")

    # Plot Publication Safety Figure
    print("Generating Robotics Flight Safety Publication Figure...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Flight Hazard Tradeoff Scatter (Critical Miss Rate vs False Alarm Rate)
    ax1 = axes[0]
    colors = {'dioptra': '#1f77b4', 'unidepth': '#2ca02c', 'depth_anything': '#ff7f0e', 'metric3d': '#d62728'}
    markers = {'dioptra': 'o', 'unidepth': '^', 'depth_anything': 'd', 'metric3d': 's'}

    for k in model_keys:
        if k in safety_results:
            cmr = safety_results[k]["critical_miss_rate_percent"]
            far = safety_results[k]["false_alarm_rate_percent"]
            lbl = model_labels[k].replace(" ViT-Small", "").replace(" (Affine)", "")
            sz = 140 if k == 'dioptra' else 100
            ax1.scatter(cmr, far, color=colors[k], marker=markers[k], s=sz, label=lbl, zorder=5)
            ax1.annotate(lbl, xy=(cmr, far), xytext=(8, 4), textcoords="offset points",
                        fontsize=10, fontweight='bold', color=colors[k])

    ax1.set_xlabel("Critical Obstacle Miss Rate (% <2.0m obstacles missed as >2.5m) ↓ [Crash Risk]", fontsize=11, fontweight='bold')
    ax1.set_ylabel("False Alarm Braking Rate (% >4.0m open space flagged <2.0m) ↓ [Stall Risk]", fontsize=11, fontweight='bold')
    ax1.set_title("Downstream Robotics Flight Collision Safety Envelope", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlim(-0.5, max(15.0, max([v["critical_miss_rate_percent"] for v in safety_results.values()]) + 2.0))
    ax1.set_ylim(-0.5, max(15.0, max([v["false_alarm_rate_percent"] for v in safety_results.values()]) + 2.0))
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Shaded green safe zone at origin
    ax1.fill_between([-0.5, 2.0], -0.5, 2.0, color='green', alpha=0.15, label="High-Safety Autonomous Flight Zone")
    ax1.legend(loc='upper right', frameon=True, fontsize=9.5)

    # Panel 2: Grouped Bar Chart of Critical Miss Rate
    ax2 = axes[1]
    keys = [k for k in model_keys if k in safety_results]
    labels_clean = [model_labels[k].replace(" ViT-Small", "").replace(" (Affine)", "") for k in keys]
    cmr_vals = [safety_results[k]["critical_miss_rate_percent"] for k in keys]
    far_vals = [safety_results[k]["false_alarm_rate_percent"] for k in keys]

    x = np.arange(len(keys))
    w = 0.35
    r1 = ax2.bar(x - w/2, cmr_vals, w, label='Critical Collision Miss Rate (%)', color='#d62728', alpha=0.85)
    r2 = ax2.bar(x + w/2, far_vals, w, label='False Alarm Braking Rate (%)', color='#ff7f0e', alpha=0.85)

    ax2.set_ylabel('Percentage of Risk Events (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Robotic Flight Risk Metrics (Near Miss vs. False Brake)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_clean, fontsize=10.5)
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    for r in r1:
        h = r.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for r in r2:
        h = r.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    fig_out = os.path.join(out_dir, "fig_robotics_collision_risk.png")
    fig_paper = os.path.join(fig_dir, "fig_robotics_collision_risk.png")
    plt.savefig(fig_out, dpi=300)
    plt.savefig(fig_paper, dpi=300)
    plt.close()

    print(f"Generated robotics safety figure:\n  {fig_out}\n  {fig_paper}")
    print("\n[SUCCESS] Robotics collision risk benchmark complete.")

if __name__ == "__main__":
    main()
