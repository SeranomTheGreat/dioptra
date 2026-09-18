#!/usr/bin/env python3
"""
Dioptra-DINO: Occlusion Boundary & Depth Discontinuity Benchmark Suite (Edge-AbsRel)
Quantifies whether models suffer from edge-fattening, boundary bleeding, or flying pixels.
Extracts depth discontinuities via Sobel gradient (threshold > 1.0m step) dilated by 3x3 kernel.
Computes stratified AbsRel, RMSE, and delta1 on Boundary Transition Zone vs. Planar Interior Zone.
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

def extract_boundary_masks(gt, min_depth=0.1, max_depth=80.0, grad_thresh=1.0):
    """
    Extracts depth discontinuity boundary transition mask and planar interior mask.
    """
    valid_mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt)
    
    # Compute Sobel gradients in meters
    sobel_x = cv2.Sobel(gt, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gt, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Discontinuity edge mask
    edge_raw = (grad_mag > grad_thresh) & valid_mask
    
    # Dilate by 3x3 structuring element to capture transition zone (+/- 1 px)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary_mask = cv2.dilate(edge_raw.astype(np.uint8), kernel).astype(bool) & valid_mask
    
    # Planar interior mask is valid pixels excluding boundaries
    interior_mask = valid_mask & (~boundary_mask)
    
    return boundary_mask, interior_mask, valid_mask

def compute_stratified_metrics(pred, gt, mask):
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    p = pred[mask]
    g = gt[mask]
    valid_sub = (p > 0.0) & np.isfinite(p) & (g > 0.0)
    if not np.any(valid_sub):
        return 0.0, 0.0, 0.0
    p = p[valid_sub]
    g = g[valid_sub]
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
    
    # Base intrinsics scaled to 224x224
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

    # Oracle MiDaS Affine Alignment (s * d + t)
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
        pred_metric = pred_full * (320.0 / 1000.0) # Native focal scale
    return pred_metric.squeeze().cpu().numpy()

def main():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_dir, "outputs")
    fig_dir = os.path.join(repo_dir, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Executing Occlusion Boundary & Discontinuity Benchmark on device: {device}")

    # Load 50 sampled continuous trajectory frames (stride of 4 across 200 frames)
    data_dir = os.path.join(repo_dir, "test_samples", "unseen_200_abandonedfactory")
    img_files = sorted(glob.glob(os.path.join(data_dir, "image_left", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(data_dir, "depth_left", "*.npy")))
    
    sampled_indices = list(range(0, len(img_files), 4))[:50]
    print(f"Evaluating {len(sampled_indices)} frames with stratified edge boundaries.")

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
        
        b_mask, i_mask, v_mask = extract_boundary_masks(gt_crop)
        test_data.append({
            "rgb": rgb_crop,
            "gt": gt_crop,
            "b_mask": b_mask,
            "i_mask": i_mask,
            "v_mask": v_mask,
            "name": os.path.basename(ip)
        })

    print(f"Prepared {len(test_data)} samples with verified depth discontinuity masks.")
    models = load_models(repo_dir, device)

    eval_results = {}
    model_keys = ['dioptra', 'unidepth', 'depth_anything', 'metric3d']
    model_labels = {
        'dioptra': 'Dioptra-DINO (Ours)',
        'unidepth': 'UniDepth-V2 ViT-Small',
        'depth_anything': 'Depth Anything V2 (Affine)',
        'metric3d': 'Metric3D ViT-Small'
    }

    # Store sample predictions for visualization
    sample_vis = {}
    first_sample = test_data[0]

    for m_key in model_keys:
        if m_key not in models:
            continue
        label = model_labels[m_key]
        print(f"\nEvaluating {label} on Discontinuity Boundaries vs. Planar Interiors...")
        
        bound_absrel_list, bound_rmse_list, bound_d1_list = [], [], []
        inter_absrel_list, inter_rmse_list, inter_d1_list = [], [], []
        global_absrel_list = []

        for idx, item in enumerate(test_data):
            if m_key == 'dioptra':
                pred = infer_dioptra(models[m_key], item["rgb"], device)
            elif m_key == 'unidepth':
                pred = infer_unidepth(models[m_key], item["rgb"], device)
            elif m_key == 'depth_anything':
                pred = infer_depth_anything(models[m_key], item["rgb"], item["gt"], device)
            elif m_key == 'metric3d':
                pred = infer_metric3d(models[m_key], item["rgb"], device)

            if idx == 0:
                sample_vis[m_key] = pred.copy()

            # Boundary metrics
            b_ar, b_rm, b_d1 = compute_stratified_metrics(pred, item["gt"], item["b_mask"])
            bound_absrel_list.append(b_ar)
            bound_rmse_list.append(b_rm)
            bound_d1_list.append(b_d1)

            # Planar interior metrics
            i_ar, i_rm, i_d1 = compute_stratified_metrics(pred, item["gt"], item["i_mask"])
            inter_absrel_list.append(i_ar)
            inter_rmse_list.append(i_rm)
            inter_d1_list.append(i_d1)

            # Global valid metrics
            g_ar, _, _ = compute_stratified_metrics(pred, item["gt"], item["v_mask"])
            global_absrel_list.append(g_ar)

        mean_b_ar = float(np.mean(bound_absrel_list))
        mean_b_rm = float(np.mean(bound_rmse_list))
        mean_b_d1 = float(np.mean(bound_d1_list))

        mean_i_ar = float(np.mean(inter_absrel_list))
        mean_i_rm = float(np.mean(inter_rmse_list))
        mean_i_d1 = float(np.mean(inter_d1_list))

        mean_g_ar = float(np.mean(global_absrel_list))
        edge_degradation_ratio = float(mean_b_ar / (mean_i_ar + 1e-8))

        eval_results[m_key] = {
            "model_label": label,
            "boundary_metrics": {"absrel": mean_b_ar, "rmse": mean_b_rm, "delta1": mean_b_d1},
            "interior_metrics": {"absrel": mean_i_ar, "rmse": mean_i_rm, "delta1": mean_i_d1},
            "global_absrel": mean_g_ar,
            "edge_degradation_ratio": edge_degradation_ratio
        }

        print(f"  {label:<28} | Boundary AbsRel: {mean_b_ar:.4f} | Interior AbsRel: {mean_i_ar:.4f} | Ratio: {edge_degradation_ratio:.2f}x")

    # Save JSON receipt
    json_path = os.path.join(out_dir, "benchmark_edge_boundary_absrel.json")
    with open(json_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved boundary results to: {json_path}")

    # Plot Publication Comparison Figure
    print("Generating Boundary Discontinuity Publication Figure...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Grouped Bar Chart of Boundary vs. Interior AbsRel
    ax1 = axes[0]
    keys = [k for k in model_keys if k in eval_results]
    labels_clean = [model_labels[k].replace(" ViT-Small", "").replace(" (Affine)", "") for k in keys]
    b_vals = [eval_results[k]["boundary_metrics"]["absrel"] for k in keys]
    i_vals = [eval_results[k]["interior_metrics"]["absrel"] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    rects1 = ax1.bar(x - width/2, b_vals, width, label='Boundary Transition Zone (Edges)', color='#d62728', alpha=0.85)
    rects2 = ax1.bar(x + width/2, i_vals, width, label='Planar Interior Zone (Walls/Floor)', color='#1f77b4', alpha=0.85)

    ax1.set_ylabel('Absolute Relative Error (AbsRel)', fontsize=12, fontweight='bold')
    ax1.set_title('Occlusion Boundary vs. Planar Interior Error', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_clean, fontsize=10.5)
    ax1.legend(frameon=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Panel 2: Visual Boundary Extraction & Edge Overlay
    ax2 = axes[1]
    b_vis = first_sample["b_mask"].astype(float)
    rgb_show = first_sample["rgb"].copy()
    
    # Overlay boundary edges in red on RGB
    overlay = rgb_show.copy()
    overlay[first_sample["b_mask"]] = [255, 30, 30]
    blended = cv2.addWeighted(rgb_show, 0.65, overlay, 0.35, 0)
    
    ax2.imshow(blended)
    ax2.set_title("Ground-Truth Depth Discontinuity Mask Overlay (Red)", fontsize=13, fontweight='bold', pad=10)
    ax2.axis('off')

    plt.tight_layout()
    fig_out = os.path.join(out_dir, "fig_edge_boundary_eval.png")
    fig_paper = os.path.join(fig_dir, "fig_edge_boundary_eval.png")
    plt.savefig(fig_out, dpi=300)
    plt.savefig(fig_paper, dpi=300)
    plt.close()

    print(f"Generated boundary evaluation figure:\n  {fig_out}\n  {fig_paper}")
    print("\n[SUCCESS] Occlusion boundary benchmark complete.")

if __name__ == "__main__":
    main()
