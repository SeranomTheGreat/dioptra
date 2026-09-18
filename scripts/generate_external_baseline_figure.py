"""
Generate publication-quality 6-column visual comparison grid comparing:
1. Input RGB
2. Ground Truth Metric Depth
3. Metric3D ViT-Small (CVPR 2023 / TPAMI 2024, Camera-Conditioned Metric)
4. Depth Anything V2-Small (CVPR 2024, Relative Disparity, Affine Aligned)
5. Canonical 2D ViT + DPT (Ablation Variant b, Direct Metric, No Ray Mod.)
6. Dioptra-DINO (Ours, Direct Metric, Zero-Scale Calibration)

Unmeasured sky regions (>80m in LiDAR ground truth) are masked uniformly across
all columns following standard robotics evaluation protocol, focusing visual attention
exclusively on physical scene geometry.

Saves output to: paper/figures/fig_external_baseline_comparison.png
"""

import os
import sys
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


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Viz] Generating 6-column baseline comparison figure on {device}...")

    # 1. Load Dioptra-DINO (Ours)
    cfg = DioptraDINOConfig(freeze_backbone=False)
    model_dioptra = DioptraDINO(cfg).to(device)
    st = torch.load("outputs/dioptra_dino_best.pt", map_location=device, weights_only=False)
    sd = st.get("model_state_dict", st)
    model_dioptra.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}, strict=False)
    model_dioptra.eval()

    # 2. Load Canonical ViT + DPT
    cfg_noray = DioptraDINOConfig(freeze_backbone=False)
    model_noray = DioptraDINO(cfg_noray).to(device)
    model_noray.ray_modulation = None
    model_noray.ara_refine = None
    st_noray = torch.load("outputs_ablations/ablation_no_ray/dioptra_dino_no_ray_best.pt", map_location=device, weights_only=False)
    sd_noray = st_noray.get("model_state_dict", st_noray)
    model_noray.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd_noray.items()}, strict=False)
    model_noray.eval()

    # 3. Load Metric3D ViT-Small
    print("[Viz] Loading Metric3D ViT-Small...")
    model_m3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True).to(device).eval()

    # 4. Load Depth Anything V2 Small
    print("[Viz] Loading Depth Anything V2-Small...")
    proc_da = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model_da = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()

    # Data config
    sample_indices = [15, 85, 140]
    data_dir = "test_samples/unseen_200_abandonedfactory"
    img_size = 224
    input_size_m3d = (616, 1064)
    padding_m3d = [123.675, 116.28, 103.53]
    mean_m3d = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1).to(device)
    std_m3d = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1).to(device)

    fig, axes = plt.subplots(3, 6, figsize=(18, 8.5), constrained_layout=True)
    col_titles = [
        "Input RGB",
        "Ground Truth",
        "Metric3D ViT-S\n(CVPR 23, Metric)",
        "Depth Anything V2-S\n(CVPR 24, Affine Aligned)",
        "Canonical ViT+DPT\n(No Ray Mod.)",
        "Dioptra-DINO (Ours)\n(Metric, Zero-Scale)"
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=8)

    cmap = matplotlib.colormaps["plasma"].copy()
    cmap.set_bad(color="#181824")

    for row_idx, s_idx in enumerate(sample_indices):
        img_p = os.path.join(data_dir, "image_left", f"{s_idx:06d}_left.png")
        gt_p = os.path.join(data_dir, "depth_left", f"{s_idx:06d}_left_depth.npy")

        raw_img = Image.open(img_p).convert("RGB")
        W_orig, H_orig = raw_img.size
        img_resized = raw_img.resize((img_size, img_size), Image.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        norm_img = (img_np - mean) / std
        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)

        gt_raw = np.load(gt_p)
        gt_pil = Image.fromarray(gt_raw.astype(np.float32))
        gt_resized = gt_pil.resize((img_size, img_size), Image.NEAREST)
        gt_np = np.array(gt_resized, dtype=np.float32)

        fx = 320.0 * (img_size / W_orig)
        fy = 320.0 * (img_size / H_orig)
        cx = 320.0 * (img_size / W_orig)
        cy = 240.0 * (img_size / H_orig)
        K = torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_dio = model_dioptra(tensor_img, K).squeeze().cpu().numpy()
            pred_noray = model_noray(tensor_img, K).squeeze().cpu().numpy()

            inputs_da = proc_da(images=raw_img, return_tensors="pt").to(device)
            outputs_da = model_da(**inputs_da)
            pred_disp = F.interpolate(outputs_da.predicted_depth.unsqueeze(1), size=(img_size, img_size), mode="bilinear").squeeze().cpu().numpy()

        # Metric3D inference
        rgb_origin = cv2.imread(img_p)[:, :, ::-1]
        h, w = rgb_origin.shape[:2]
        scale_m3d = min(input_size_m3d[0] / h, input_size_m3d[1] / w)
        rgb_m3d = cv2.resize(rgb_origin, (int(w * scale_m3d), int(h * scale_m3d)), interpolation=cv2.INTER_LINEAR)
        pad_h = input_size_m3d[0] - rgb_m3d.shape[0]
        pad_w = input_size_m3d[1] - rgb_m3d.shape[1]
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb_padded = cv2.copyMakeBorder(rgb_m3d, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding_m3d)
        t_m3d = torch.from_numpy(rgb_padded.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
        norm_m3d = (t_m3d - mean_m3d) / std_m3d
        with torch.no_grad():
            pred_m3d, _, _ = model_m3d.inference({"input": norm_m3d})
        pred_m3d = pred_m3d.squeeze()
        pred_m3d = pred_m3d[pad_h_half : pred_m3d.shape[0] - (pad_h - pad_h_half), pad_w_half : pred_m3d.shape[1] - (pad_w - pad_w_half)]
        pred_m3d = F.interpolate(pred_m3d[None, None, :, :], (img_size, img_size), mode="bilinear").squeeze().cpu().numpy()
        pred_m3d = pred_m3d * (320.0 * scale_m3d / 1000.0)

        # MiDaS Affine alignment for Depth Anything V2
        valid = (gt_np > 0.1) & (gt_np < 80.0) & np.isfinite(gt_np) & (pred_disp > 0)
        g_disp = 1.0 / gt_np[valid]
        d_val = pred_disp[valid]
        A = np.vstack([d_val, np.ones_like(d_val)]).T
        s, t = np.linalg.lstsq(A, g_disp, rcond=None)[0]
        aligned_disp = np.maximum(s * pred_disp + t, 1e-4)
        pred_da_metric = 1.0 / aligned_disp

        # Colormap limits based on ground truth 5th and 95th percentiles
        vmin = np.percentile(gt_np[valid], 2)
        vmax = np.percentile(gt_np[valid], 98)

        # Mask unmeasured sky across all predictions
        gt_masked = np.ma.masked_where(~valid, gt_np)
        m3d_masked = np.ma.masked_where(~valid, pred_m3d)
        da_masked = np.ma.masked_where(~valid, pred_da_metric)
        noray_masked = np.ma.masked_where(~valid, pred_noray)
        dio_masked = np.ma.masked_where(~valid, pred_dio)

        # 1. RGB
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].axis("off")

        # 2. GT
        axes[row_idx, 1].imshow(gt_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 1].axis("off")

        # 3. Metric3D
        axes[row_idx, 2].imshow(m3d_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 2].axis("off")

        # 4. Depth Anything V2
        axes[row_idx, 3].imshow(da_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 3].axis("off")

        # 5. Canonical ViT
        axes[row_idx, 4].imshow(noray_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 4].axis("off")

        # 6. Dioptra-DINO
        axes[row_idx, 5].imshow(dio_masked, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 5].axis("off")

    os.makedirs("paper/figures", exist_ok=True)
    out_fig = "paper/figures/fig_external_baseline_comparison.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved publication figure to: {out_fig}")


if __name__ == "__main__":
    main()
