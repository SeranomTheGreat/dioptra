"""
Benchmark 2: 3D Point Cloud Reconstruction & Surface Normal Fidelity
Evaluates 3D geometric reconstruction accuracy:
1. Surface Normal Mean Angular Error (MAE in degrees) and angular thresholds (< 11.25°, < 22.5°, < 30.0°)
2. 3D Chamfer Distance (CD in metres)
3. 3D F-Score at < 5cm, < 10cm, < 20cm precision thresholds

Evaluated across held-out continuous trajectory frames.
Saves to:
  outputs/benchmark_3d_pointcloud_normals.json
  outputs/3d_geometry_metrics.csv
  paper/figures/fig_pointcloud_normals.png
  outputs/fig_pointcloud_normals.png
"""

import os
import sys
import glob
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dioptra_dino
import __main__
__main__.DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def depth_to_points(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.stack([x, y, z], axis=-1)  # [H, W, 3]


def compute_surface_normals(points: np.ndarray, valid_mask: np.ndarray):
    """Compute surface normals from 3D points via central differences."""
    H, W, _ = points.shape
    du = np.zeros_like(points)
    dv = np.zeros_like(points)

    du[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dv[1:-1, :] = points[2:, :] - points[:-2, :]

    normals = np.cross(du, dv)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, norm + 1e-8)

    # Valid mask erosion to remove boundary artifacts
    kernel = np.ones((3, 3), np.uint8)
    eroded_valid = cv2.erode(valid_mask.astype(np.uint8), kernel).astype(bool)
    normals[~eroded_valid] = 0
    return normals, eroded_valid


def evaluate_normals(pred_normals: np.ndarray, gt_normals: np.ndarray, mask: np.ndarray):
    """Compute normal angular error in degrees."""
    p_n = pred_normals[mask]
    g_n = gt_normals[mask]

    dot = np.abs(np.sum(p_n * g_n, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    angle_deg = np.arccos(dot) * (180.0 / np.pi)

    mae = float(np.mean(angle_deg))
    med = float(np.median(angle_deg))
    a11 = float((angle_deg < 11.25).mean() * 100.0)
    a22 = float((angle_deg < 22.5).mean() * 100.0)
    a30 = float((angle_deg < 30.0).mean() * 100.0)

    return {
        "mae_deg": mae,
        "median_deg": med,
        "a_11_25": a11,
        "a_22_5": a22,
        "a_30_0": a30
    }


def compute_chamfer_and_fscore(p_pts: np.ndarray, g_pts: np.ndarray, max_samples: int = 5000):
    """Subsample points and compute 3D Chamfer Distance and F-Scores."""
    if len(p_pts) > max_samples:
        idx_p = np.random.choice(len(p_pts), max_samples, replace=False)
        p_sub = p_pts[idx_p]
    else:
        p_sub = p_pts

    if len(g_pts) > max_samples:
        idx_g = np.random.choice(len(g_pts), max_samples, replace=False)
        g_sub = g_pts[idx_g]
    else:
        g_sub = g_pts

    # Pairwise distances via PyTorch for edge GPU acceleration
    t_p = torch.from_numpy(p_sub).float()
    t_g = torch.from_numpy(g_sub).float()

    dists = torch.cdist(t_p, t_g)  # [Np, Ng]
    min_p2g, _ = torch.min(dists, dim=1)
    min_g2p, _ = torch.min(dists, dim=0)

    cd = float((torch.mean(min_p2g) + torch.mean(min_g2p)).item()) / 2.0

    # F-scores at 0.05m, 0.10m, 0.20m
    fscores = {}
    for d_th in [0.05, 0.10, 0.20]:
        prec = float((min_p2g < d_th).float().mean().item())
        rec = float((min_g2p < d_th).float().mean().item())
        if prec + rec > 0:
            f = 2.0 * prec * rec / (prec + rec) * 100.0
        else:
            f = 0.0
        fscores[f"f_{int(d_th*100)}cm"] = f

    return cd, fscores


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 85)
    print(f" BENCHMARK 2: 3D POINT CLOUD & SURFACE NORMAL EVALUATION ON {device}")
    print("=" * 85)

    # 1. Load Models
    print("[1/4] Loading Dioptra-DINO...")
    cfg = DioptraDINOConfig(freeze_backbone=False)
    model_dioptra = DioptraDINO(cfg).to(device)
    st = torch.load("outputs/dioptra_dino_best.pt", map_location=device, weights_only=False)
    sd = st.get("model_state_dict", st)
    model_dioptra.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    model_dioptra.eval()

    print("[2/4] Loading UniDepth-V2 ViT-Small...")
    model_unidepth = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version="v2", backbone="vits14", pretrained=True, trust_repo=True).to(device).eval()

    print("[3/4] Loading Metric3D ViT-Small...")
    model_m3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True).to(device).eval()

    print("[4/4] Loading Depth Anything V2-Small...")
    proc_da = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model_da = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()

    # Data
    data_dir = "test_samples/unseen_200_abandonedfactory"
    # Select 25 evenly spaced benchmark frames across the 200 continuous frames
    stride = 8
    frame_indices = list(range(0, 200, stride))
    print(f"\n[Evaluation] Evaluating across {len(frame_indices)} trajectory keyframes (stride={stride})...")

    orig_W, orig_H = 640, 480
    crop_size = 224

    orig_K = torch.tensor([
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    crop_fx = 320.0 * (crop_size / 480.0)
    crop_fy = 320.0 * (crop_size / 480.0)
    crop_cx = (320.0 - 80.0) * (crop_size / 480.0)
    crop_cy = 240.0 * (crop_size / 480.0)
    dioptra_K = torch.tensor([
        [crop_fx, 0.0, crop_cx],
        [0.0, crop_fy, crop_cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    mean_dino = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_dino = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    m3d_input_size = (616, 1064)
    mean_rgb = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1).to(device)
    std_rgb = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1).to(device)

    model_names = ["Dioptra-DINO (Ours)", "UniDepth-V2 ViT-S", "Metric3D ViT-S", "Depth Anything V2-S (Affine)"]
    metrics_acc = {m: {"mae_deg": [], "a_11_25": [], "a_22_5": [], "a_30_0": [], "cd": [], "f_5cm": [], "f_10cm": [], "f_20cm": []} for m in model_names}

    # Storing sample for visualization
    viz_sample = None

    for f_i, idx in enumerate(frame_indices):
        img_p = os.path.join(data_dir, "image_left", f"{idx:06d}_left.png")
        gt_p = os.path.join(data_dir, "depth_left", f"{idx:06d}_left_depth.npy")

        raw_bgr = cv2.imread(img_p)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        raw_pil = Image.fromarray(raw_rgb)
        raw_gt = np.load(gt_p).astype(np.float32)

        # Ground Truth Points & Normals on square crop area (or full frame)
        # We evaluate on the shared 480x480 crop where all models overlap
        left_crop = (orig_W - orig_H) // 2
        gt_crop = raw_gt[:, left_crop:left_crop + orig_H]
        rgb_crop = raw_rgb[:, left_crop:left_crop + orig_H]

        # 1. Dioptra-DINO
        rgb_resized = cv2.resize(rgb_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t_img = (t_img - mean_dino) / std_dino
        with torch.no_grad():
            pred_dioptra_224 = model_dioptra(t_img, dioptra_K).squeeze().cpu().numpy()
        pred_dioptra = cv2.resize(pred_dioptra_224, (orig_H, orig_H), interpolation=cv2.INTER_LINEAR)

        # 2. UniDepth-V2
        unidepth_rgb_tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_unidepth_full = model_unidepth.infer(unidepth_rgb_tensor, camera=orig_K)["depth"].squeeze().cpu().numpy()
        pred_unidepth = pred_unidepth_full[:, left_crop:left_crop + orig_H]

        # 3. Metric3D
        scale_m3d = min(m3d_input_size[0] / orig_H, m3d_input_size[1] / orig_W)
        rgb_m3d = cv2.resize(raw_rgb, (int(orig_W * scale_m3d), int(orig_H * scale_m3d)), interpolation=cv2.INTER_LINEAR)
        intrinsic_scaled = [320.0 * scale_m3d, 320.0 * scale_m3d, 320.0 * scale_m3d, 240.0 * scale_m3d]
        pad_h = m3d_input_size[0] - rgb_m3d.shape[0]
        pad_w = m3d_input_size[1] - rgb_m3d.shape[1]
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb_padded = cv2.copyMakeBorder(rgb_m3d, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
        tensor_rgb = torch.from_numpy(rgb_padded.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
        norm_rgb = (tensor_rgb - mean_rgb) / std_rgb
        with torch.no_grad():
            pred_m3d_raw, _, _ = model_m3d.inference({"input": norm_rgb})
        pred_m3d_raw = pred_m3d_raw.squeeze()
        pred_m3d_raw = pred_m3d_raw[pad_h_half : pred_m3d_raw.shape[0] - (pad_h - pad_h_half), pad_w_half : pred_m3d_raw.shape[1] - (pad_w - pad_w_half)]
        pred_m3d_full = F.interpolate(pred_m3d_raw[None, None, :, :], (orig_H, orig_W), mode="bilinear").squeeze().cpu().numpy()
        pred_m3d = np.clip(pred_m3d_full * (intrinsic_scaled[0] / 1000.0), 0.0, 80.0)[:, left_crop:left_crop + orig_H]

        # 4. Depth Anything V2 Relative (MiDaS Affine)
        da_inputs = proc_da(images=raw_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            pred_da_disp = model_da(**da_inputs).predicted_depth
            pred_da_disp = F.interpolate(pred_da_disp.unsqueeze(1), size=(orig_H, orig_W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        mask_da = (raw_gt > 0.1) & (raw_gt < 80.0) & np.isfinite(raw_gt) & (pred_da_disp > 0)
        d_vals = pred_da_disp[mask_da]
        inv_gt = 1.0 / raw_gt[mask_da]
        A = np.vstack([d_vals, np.ones_like(d_vals)]).T
        s_aff, t_aff = np.linalg.lstsq(A, inv_gt, rcond=None)[0]
        aligned_disp = np.clip(s_aff * pred_da_disp + t_aff, 1.0 / 80.0, 1.0 / 0.1)
        pred_da_full = 1.0 / aligned_disp
        pred_da = pred_da_full[:, left_crop:left_crop + orig_H]

        # Intrinsics for the 480x480 crop
        fx_crop = 320.0
        fy_crop = 320.0
        cx_crop = 320.0 - 80.0  # 240.0
        cy_crop = 240.0

        # Valid mask
        valid_gt = (gt_crop > 0.1) & (gt_crop < 60.0) & np.isfinite(gt_crop)

        # Ground truth points & normals
        gt_pts_grid = depth_to_points(gt_crop, fx_crop, fy_crop, cx_crop, cy_crop)
        gt_normals, valid_mask = compute_surface_normals(gt_pts_grid, valid_gt)

        # Compare each model
        preds = {
            "Dioptra-DINO (Ours)": pred_dioptra,
            "UniDepth-V2 ViT-S": pred_unidepth,
            "Metric3D ViT-S": pred_m3d,
            "Depth Anything V2-S (Affine)": pred_da
        }

        eval_mask = valid_mask & (gt_crop > 0.5)

        for m_name, pred_d in preds.items():
            pts_grid = depth_to_points(pred_d, fx_crop, fy_crop, cx_crop, cy_crop)
            p_normals, _ = compute_surface_normals(pts_grid, eval_mask)

            norm_m = evaluate_normals(p_normals, gt_normals, eval_mask)
            for k, v in norm_m.items():
                if k in metrics_acc[m_name]:
                    metrics_acc[m_name][k].append(v)

            # Chamfer & F-score
            cd, f_sc = compute_chamfer_and_fscore(pts_grid[eval_mask], gt_pts_grid[eval_mask])
            metrics_acc[m_name]["cd"].append(cd)
            metrics_acc[m_name]["f_5cm"].append(f_sc["f_5cm"])
            metrics_acc[m_name]["f_10cm"].append(f_sc["f_10cm"])
            metrics_acc[m_name]["f_20cm"].append(f_sc["f_20cm"])

        if f_i == 0:
            viz_sample = {
                "rgb": rgb_crop,
                "gt_normals": gt_normals,
                "dioptra_normals": compute_surface_normals(depth_to_points(pred_dioptra, fx_crop, fy_crop, cx_crop, cy_crop), eval_mask)[0],
                "unidepth_normals": compute_surface_normals(depth_to_points(pred_unidepth, fx_crop, fy_crop, cx_crop, cy_crop), eval_mask)[0],
                "m3d_normals": compute_surface_normals(depth_to_points(pred_m3d, fx_crop, fy_crop, cx_crop, cy_crop), eval_mask)[0],
                "da_normals": compute_surface_normals(depth_to_points(pred_da, fx_crop, fy_crop, cx_crop, cy_crop), eval_mask)[0],
                "mask": eval_mask
            }

        print(f"[{f_i+1:02d}/{len(frame_indices)}] Frame #{idx:03d}: Dioptra MAE={metrics_acc['Dioptra-DINO (Ours)']['mae_deg'][-1]:.2f}° | UniDepth={metrics_acc['UniDepth-V2 ViT-S']['mae_deg'][-1]:.2f}° | Metric3D={metrics_acc['Metric3D ViT-S']['mae_deg'][-1]:.2f}° | DA={metrics_acc['Depth Anything V2-S (Affine)']['mae_deg'][-1]:.2f}°")

    # Aggregate
    summary = {}
    print("\n" + "=" * 105)
    print(" SUMMARY 3D POINT CLOUD & SURFACE NORMAL BENCHMARK RESULTS")
    print("=" * 105)
    print(f"{'Model Name':<30} | {'Normal MAE':<11} | {'< 11.25°':<9} | {'< 22.5°':<9} | {'< 30.0°':<9} | {'Chamfer (m)':<12} | {'F @ 10cm':<8}")
    print("-" * 105)

    for m_name in model_names:
        mae = float(np.mean(metrics_acc[m_name]["mae_deg"]))
        a11 = float(np.mean(metrics_acc[m_name]["a_11_25"]))
        a22 = float(np.mean(metrics_acc[m_name]["a_22_5"]))
        a30 = float(np.mean(metrics_acc[m_name]["a_30_0"]))
        cd = float(np.mean(metrics_acc[m_name]["cd"]))
        f10 = float(np.mean(metrics_acc[m_name]["f_10cm"]))

        summary[m_name] = {
            "normal_mae_deg": mae,
            "normal_acc_11_25": a11,
            "normal_acc_22_5": a22,
            "normal_acc_30_0": a30,
            "chamfer_dist_m": cd,
            "f_score_10cm": f10
        }
        print(f"{m_name:<30} | {mae:<9.2f}° | {a11:<8.2f}% | {a22:<8.2f}% | {a30:<8.2f}% | {cd:<12.4f} | {f10:<7.2f}%")
    print("-" * 105)

    out_json = "outputs/benchmark_3d_pointcloud_normals.json"
    with open(out_json, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n[Saved] 3D geometry summary written to {out_json}")

    # Generate Surface Normal Visualization Figure
    print("\n[Viz] Generating surface normal comparison figure...")
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.8), constrained_layout=True)
    titles = [
        "Input RGB Crop",
        "Ground Truth Normals",
        "Dioptra-DINO (Ours)\n(Virtual Normal Loss)",
        "UniDepth-V2 ViT-S\n(Pseudo-Spherical)",
        "Metric3D ViT-S\n(Canonical)",
        "Depth Anything V2-S\n(Affine Aligned)"
    ]

    def normal_to_rgb(norm_map, m):
        rgb = (norm_map + 1.0) / 2.0
        rgb[~m] = 0.08
        return rgb

    m = viz_sample["mask"]
    axes[0].imshow(viz_sample["rgb"])
    axes[1].imshow(normal_to_rgb(viz_sample["gt_normals"], m))
    axes[2].imshow(normal_to_rgb(viz_sample["dioptra_normals"], m))
    axes[3].imshow(normal_to_rgb(viz_sample["unidepth_normals"], m))
    axes[4].imshow(normal_to_rgb(viz_sample["m3d_normals"], m))
    axes[5].imshow(normal_to_rgb(viz_sample["da_normals"], m))

    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")

    out_paper_fig = "paper/figures/fig_pointcloud_normals.png"
    out_outputs_fig = "outputs/fig_pointcloud_normals.png"
    os.makedirs(os.path.dirname(out_paper_fig), exist_ok=True)
    os.makedirs(os.path.dirname(out_outputs_fig), exist_ok=True)
    fig.savefig(out_paper_fig, dpi=300, bbox_inches="tight")
    fig.savefig(out_outputs_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Surface normal figure written to {out_paper_fig} and {out_outputs_fig}")


if __name__ == "__main__":
    main()
