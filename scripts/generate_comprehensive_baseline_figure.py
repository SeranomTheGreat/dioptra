"""
Generate publication-quality 7-column visual comparison grid comparing:
1. Input RGB (Held-Out Benchmark Frames)
2. Ground Truth Metric Depth (LiDAR)
3. Dioptra-DINO (Ours, Direct Metric via Ray-FiLM, 27.5M, 38.0 FPS)
4. UniDepth-V2 ViT-Small (Camera-Conditioned Metric, 34.2M, 5.4 FPS)
5. Metric3D ViT-Small (Canonical Focal Transform, 37.5M, 1.8 FPS)
6. Depth Anything V2-Small (MiDaS Disparity Affine Aligned, 24.8M, 8.5 FPS)
7. ZoeDepth ZoeD_NK (BEiT-Large Metric Bins, 346.1M, 0.7 FPS)

Uniform masking of unmeasured sky/void (>80m in LiDAR ground truth) across all columns.
Saves output to:
  paper/figures/fig_external_baseline_comparison.png
  outputs/fig_comprehensive_baseline_comparison.png
"""

import os
import sys
import glob
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
    print(f"[Viz] Generating comprehensive multi-model baseline comparison on {device}...")

    # 1. Dioptra-DINO
    print("[1/5] Loading Dioptra-DINO (outputs/dioptra_dino_best.pt)...")
    cfg = DioptraDINOConfig(freeze_backbone=False)
    model_dioptra = DioptraDINO(cfg).to(device)
    st = torch.load("outputs/dioptra_dino_best.pt", map_location=device, weights_only=False)
    sd = st.get("model_state_dict", st)
    model_dioptra.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    model_dioptra.eval()

    # 2. UniDepth-V2 ViT-Small
    print("[2/5] Loading UniDepth-V2 ViT-Small...")
    model_unidepth = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version="v2", backbone="vits14", pretrained=True, trust_repo=True)
    model_unidepth.to(device).eval()

    # 3. Metric3D ViT-Small
    print("[3/5] Loading Metric3D ViT-Small...")
    model_m3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True).to(device).eval()

    # 4. Depth Anything V2 Small Relative
    print("[4/5] Loading Depth Anything V2-Small Relative...")
    proc_da = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model_da = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()

    # 5. ZoeDepth ZoeD_NK
    print("[5/5] Loading ZoeDepth ZoeD_NK...")
    model_zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True, trust_repo=True).to(device).eval()

    # Test frames to visualize
    sample_indices = [15, 85, 140]
    data_dir = "test_samples/unseen_200_abandonedfactory"
    orig_W, orig_H = 640, 480
    crop_size = 224

    # Camera Intrinsics
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

    fig, axes = plt.subplots(3, 7, figsize=(22, 8.5), constrained_layout=True)
    col_titles = [
        "Input RGB\n(Held-Out Frame)",
        "Ground Truth\n(LiDAR Metric)",
        "Dioptra-DINO (Ours)\n(27.5M, 38.0 FPS)",
        "UniDepth-V2 ViT-S\n(34.2M, 5.4 FPS)",
        "Metric3D ViT-S\n(37.5M, 1.8 FPS)",
        "Depth Anything V2-S\n(24.8M, Affine Aligned)",
        "ZoeDepth ZoeD_NK\n(346.1M, 0.7 FPS)"
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=8)

    cmap = matplotlib.colormaps["plasma"].copy()
    cmap.set_bad(color="#181824")

    for row_idx, s_idx in enumerate(sample_indices):
        img_p = os.path.join(data_dir, "image_left", f"{s_idx:06d}_left.png")
        gt_p = os.path.join(data_dir, "depth_left", f"{s_idx:06d}_left_depth.npy")

        raw_bgr = cv2.imread(img_p)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        raw_pil = Image.fromarray(raw_rgb)
        raw_gt = np.load(gt_p).astype(np.float32)

        # 1. Dioptra-DINO
        left_crop = (orig_W - orig_H) // 2
        rgb_crop = raw_rgb[:, left_crop:left_crop + orig_H]
        rgb_resized = cv2.resize(rgb_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t_img = (t_img - mean_dino) / std_dino

        with torch.no_grad():
            pred_dioptra_224 = model_dioptra(t_img, dioptra_K).squeeze().cpu().numpy()
        pred_dioptra_full = cv2.resize(pred_dioptra_224, (orig_H, orig_H), interpolation=cv2.INTER_LINEAR)
        pred_dioptra = np.full_like(raw_gt, np.nan)
        pred_dioptra[:, left_crop:left_crop + orig_H] = pred_dioptra_full

        # 2. UniDepth-V2
        unidepth_rgb_tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            unidepth_out = model_unidepth.infer(unidepth_rgb_tensor, camera=orig_K)
            pred_unidepth = unidepth_out["depth"].squeeze().cpu().numpy()

        # 3. Metric3D ViT-Small
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
        pred_m3d = np.clip(pred_m3d_full * (intrinsic_scaled[0] / 1000.0), 0.0, 300.0)

        # 4. Depth Anything V2 Small Relative (MiDaS Affine)
        da_inputs = proc_da(images=raw_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            pred_da_disp = model_da(**da_inputs).predicted_depth
            pred_da_disp = F.interpolate(pred_da_disp.unsqueeze(1), size=(orig_H, orig_W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        mask_da = (raw_gt > 0.1) & (raw_gt < 80.0) & np.isfinite(raw_gt) & (pred_da_disp > 0)
        d_vals = pred_da_disp[mask_da]
        inv_gt = 1.0 / raw_gt[mask_da]
        A = np.vstack([d_vals, np.ones_like(d_vals)]).T
        s_aff, t_aff = np.linalg.lstsq(A, inv_gt, rcond=None)[0]
        aligned_disp = np.maximum(s_aff * pred_da_disp + t_aff, 1e-4)
        pred_da = 1.0 / aligned_disp

        # 5. ZoeDepth
        with torch.no_grad():
            pred_zoe = model_zoe.infer_pil(raw_pil)

        # Sky mask and valid depth range
        valid_mask = (raw_gt > 0.1) & (raw_gt < 80.0) & np.isfinite(raw_gt)

        # Shared 480x480 square crop across ALL models for perfectly fair visual comparison
        crop_rgb = raw_rgb[:, left_crop:left_crop + orig_H]
        crop_gt = raw_gt[:, left_crop:left_crop + orig_H]
        crop_valid = valid_mask[:, left_crop:left_crop + orig_H]
        crop_gt_disp = np.where(crop_valid, crop_gt, np.nan)

        pred_dio_crop = np.where(crop_valid, pred_dioptra_full, np.nan)
        pred_uni_crop = np.where(crop_valid, pred_unidepth[:, left_crop:left_crop + orig_H], np.nan)
        pred_m3d_crop = np.where(crop_valid, pred_m3d[:, left_crop:left_crop + orig_H], np.nan)
        pred_da_crop = np.where(crop_valid, pred_da[:, left_crop:left_crop + orig_H], np.nan)
        pred_zoe_crop = np.where(crop_valid, pred_zoe[:, left_crop:left_crop + orig_H], np.nan)

        vmax = float(np.percentile(crop_gt[crop_valid], 98))
        vmin = float(np.percentile(crop_gt[crop_valid], 2))

        # Col 0: RGB
        axes[row_idx, 0].imshow(crop_rgb)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_ylabel(f"Frame #{s_idx}", fontsize=10, fontweight="bold", labelpad=5)

        # Col 1: GT
        axes[row_idx, 1].imshow(np.ma.masked_invalid(crop_gt_disp), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 1].axis("off")

        # Col 2: Dioptra-DINO
        axes[row_idx, 2].imshow(np.ma.masked_invalid(pred_dio_crop), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 2].axis("off")

        # Col 3: UniDepth-V2
        axes[row_idx, 3].imshow(np.ma.masked_invalid(pred_uni_crop), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 3].axis("off")

        # Col 4: Metric3D
        axes[row_idx, 4].imshow(np.ma.masked_invalid(pred_m3d_crop), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 4].axis("off")

        # Col 5: Depth Anything V2
        axes[row_idx, 5].imshow(np.ma.masked_invalid(pred_da_crop), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 5].axis("off")

        # Col 6: ZoeDepth
        axes[row_idx, 6].imshow(np.ma.masked_invalid(pred_zoe_crop), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 6].axis("off")

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=35))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Metric Depth (metres)", fontsize=10, fontweight="bold")

    out_paper = "paper/figures/fig_external_baseline_comparison.png"
    out_outputs = "outputs/fig_comprehensive_baseline_comparison.png"
    os.makedirs(os.path.dirname(out_paper), exist_ok=True)
    os.makedirs(os.path.dirname(out_outputs), exist_ok=True)

    fig.savefig(out_paper, dpi=300, bbox_inches="tight")
    fig.savefig(out_outputs, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved comprehensive comparison figures to {out_paper} and {out_outputs}.")


if __name__ == "__main__":
    main()
