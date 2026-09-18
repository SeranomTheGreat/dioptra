"""
Benchmark 4: Multi-FOV Camera Intrinsics Sweep Across Models
Evaluates model scale equivariance and robustness when subjected to varying 
optical Fields of View (FOV) ranging from telephoto (50°) to ultra-wide (100°).

Models Evaluated:
1. Dioptra-DINO (Ours) - conditioned via CAFM + ARA
2. UniDepth-V2 ViT-Small - camera conditioned via pseudo-spherical embedding
3. Metric3D ViT-Small - camera conditioned via canonical focal transform

Produces:
  outputs/benchmark_multi_fov_sweep.json
  paper/figures/fig_multi_fov_comparison.png
  outputs/fig_multi_fov_comparison.png
"""

import os
import sys
import json
import math
import cv2
import numpy as np
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


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth=0.1, max_depth=80.0):
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if mask.sum() == 0:
        return {}
    p = pred[mask]
    g = gt[mask]
    abs_rel = float(np.mean(np.abs(p - g) / g))
    sq_rel = float(np.mean(((p - g) ** 2) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    thresh = np.maximum(p / g, g / p)
    d1 = float((thresh < 1.25).mean() * 100.0)
    scale = float(np.median(p) / np.median(g))
    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "delta_1": d1,
        "scale_ratio": scale
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 85)
    print(f" BENCHMARK 4: MULTI-FOV INVARIANCE SWEEP ON {device}")
    print("=" * 85)

    # 1. Load Models
    print("[1/3] Loading Dioptra-DINO...")
    cfg = DioptraDINOConfig(freeze_backbone=False)
    model_dioptra = DioptraDINO(cfg).to(device)
    st = torch.load("outputs/dioptra_dino_best.pt", map_location=device, weights_only=False)
    sd = st.get("model_state_dict", st)
    model_dioptra.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    model_dioptra.eval()

    print("[2/3] Loading UniDepth-V2 ViT-Small...")
    model_unidepth = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version="v2", backbone="vits14", pretrained=True, trust_repo=True).to(device).eval()

    print("[3/3] Loading Metric3D ViT-Small...")
    model_m3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True).to(device).eval()

    # Preprocessing
    orig_W, orig_H = 640, 480
    crop_size = 224
    mean_dino = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_dino = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    m3d_input_size = (616, 1064)
    mean_rgb = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1).to(device)
    std_rgb = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1).to(device)

    # Test sample: representative held-out continuous trajectory frame
    img_p = "test_samples/unseen_200_abandonedfactory/image_left/000000_left.png"
    gt_p = "test_samples/unseen_200_abandonedfactory/depth_left/000000_left_depth.npy"

    raw_bgr = cv2.imread(img_p)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    raw_gt = np.load(gt_p).astype(np.float32)

    left_crop = (orig_W - orig_H) // 2
    rgb_crop = raw_rgb[:, left_crop:left_crop + orig_H]
    gt_crop = raw_gt[:, left_crop:left_crop + orig_H]

    rgb_resized = cv2.resize(rgb_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    t_img = (t_img - mean_dino) / std_dino

    fovs = [50.0, 60.0, 73.74, 85.0, 90.0, 100.0]
    results = {
        "Dioptra-DINO": {},
        "UniDepth-V2": {},
        "Metric3D": {}
    }

    viz_maps = {
        "Dioptra-DINO": [],
        "UniDepth-V2": [],
        "Metric3D": []
    }

    print("\nStarting FOV sweep across [50°, 60°, 73.74°, 85°, 90°, 100°]...")

    for fov_deg in fovs:
        fov_rad = math.radians(fov_deg)

        # 1. Dioptra-DINO camera matrix
        fx_dioptra = float((crop_size / 2.0) / math.tan(fov_rad / 2.0))
        fy_dioptra = fx_dioptra
        cx_dioptra = float(crop_size / 2.0)
        cy_dioptra = float(crop_size / 2.0)
        K_dioptra = torch.tensor([
            [fx_dioptra, 0.0, cx_dioptra],
            [0.0, fy_dioptra, cy_dioptra],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            pred_d_224 = model_dioptra(t_img, K_dioptra).squeeze().cpu().numpy()
        pred_dioptra = cv2.resize(pred_d_224, (orig_H, orig_H), interpolation=cv2.INTER_LINEAR)
        m_dioptra = compute_metrics(pred_dioptra, gt_crop)
        results["Dioptra-DINO"][str(fov_deg)] = m_dioptra
        viz_maps["Dioptra-DINO"].append(pred_dioptra)

        # 2. UniDepth-V2
        fx_unidepth = float((orig_W / 2.0) / math.tan(fov_rad / 2.0))
        fy_unidepth = fx_unidepth
        cx_unidepth = float(orig_W / 2.0)
        cy_unidepth = float(orig_H / 2.0)
        K_unidepth = torch.tensor([
            [fx_unidepth, 0.0, cx_unidepth],
            [0.0, fy_unidepth, cy_unidepth],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device).unsqueeze(0)

        unidepth_rgb_tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_unidepth_full = model_unidepth.infer(unidepth_rgb_tensor, camera=K_unidepth)["depth"].squeeze().cpu().numpy()
        pred_unidepth = pred_unidepth_full[:, left_crop:left_crop + orig_H]
        m_unidepth = compute_metrics(pred_unidepth, gt_crop)
        results["UniDepth-V2"][str(fov_deg)] = m_unidepth
        viz_maps["UniDepth-V2"].append(pred_unidepth)

        # 3. Metric3D
        fx_m3d = float((orig_W / 2.0) / math.tan(fov_rad / 2.0))
        scale_m3d = min(m3d_input_size[0] / orig_H, m3d_input_size[1] / orig_W)
        rgb_m3d = cv2.resize(raw_rgb, (int(orig_W * scale_m3d), int(orig_H * scale_m3d)), interpolation=cv2.INTER_LINEAR)
        intrinsic_scaled = [fx_m3d * scale_m3d, fx_m3d * scale_m3d, (orig_W / 2.0) * scale_m3d, (orig_H / 2.0) * scale_m3d]
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
        m_m3d = compute_metrics(pred_m3d, gt_crop)
        results["Metric3D"][str(fov_deg)] = m_m3d
        viz_maps["Metric3D"].append(pred_m3d)

        print(f"FOV {fov_deg:>5.1f}° | Dioptra: AbsRel={m_dioptra['abs_rel']:.4f}, s={m_dioptra['scale_ratio']:.4f} | UniDepth: AbsRel={m_unidepth['abs_rel']:.4f}, s={m_unidepth['scale_ratio']:.4f} | Metric3D: AbsRel={m_m3d['abs_rel']:.4f}, s={m_m3d['scale_ratio']:.4f}")

    # Output JSON
    out_json = "outputs/benchmark_multi_fov_sweep.json"
    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[Saved] Multi-FOV sweep results saved to {out_json}")

    # Plot Multi-Model Multi-FOV Comparison Figure
    fig, axes = plt.subplots(4, len(fovs), figsize=(3.2 * len(fovs), 11.5), constrained_layout=True)

    vmax = np.percentile(gt_crop[np.isfinite(gt_crop) & (gt_crop > 0)], 95)

    for c_idx, fov_deg in enumerate(fovs):
        # Row 0: Dioptra-DINO
        d_pred = viz_maps["Dioptra-DINO"][c_idx]
        s_d = results["Dioptra-DINO"][str(fov_deg)]["scale_ratio"]
        ar_d = results["Dioptra-DINO"][str(fov_deg)]["abs_rel"]
        axes[0, c_idx].imshow(d_pred, cmap="plasma", vmin=0, vmax=vmax)
        axes[0, c_idx].set_title(f"FOV {fov_deg}°\ns = {s_d:.3f} | AR = {ar_d:.3f}", fontsize=10, fontweight="bold")

        # Row 1: UniDepth-V2
        u_pred = viz_maps["UniDepth-V2"][c_idx]
        s_u = results["UniDepth-V2"][str(fov_deg)]["scale_ratio"]
        ar_u = results["UniDepth-V2"][str(fov_deg)]["abs_rel"]
        axes[1, c_idx].imshow(u_pred, cmap="plasma", vmin=0, vmax=vmax)
        axes[1, c_idx].set_title(f"s = {s_u:.3f} | AR = {ar_u:.3f}", fontsize=9)

        # Row 2: Metric3D
        m_pred = viz_maps["Metric3D"][c_idx]
        s_m = results["Metric3D"][str(fov_deg)]["scale_ratio"]
        ar_m = results["Metric3D"][str(fov_deg)]["abs_rel"]
        axes[2, c_idx].imshow(m_pred, cmap="plasma", vmin=0, vmax=vmax)
        axes[2, c_idx].set_title(f"s = {s_m:.3f} | AR = {ar_m:.3f}", fontsize=9)

    # Row 3: Quantitative Scale Ratio Curves and AbsRel Curves across FOVs
    axes[3, 0].remove()
    axes[3, 1].remove()
    axes[3, 2].remove()
    axes[3, 3].remove()
    axes[3, 4].remove()
    axes[3, 5].remove()

    gs = axes[3, 0].get_gridspec()
    ax_scale = fig.add_subplot(gs[3, :3])
    ax_err = fig.add_subplot(gs[3, 3:])

    d_scales = [results["Dioptra-DINO"][str(f)]["scale_ratio"] for f in fovs]
    u_scales = [results["UniDepth-V2"][str(f)]["scale_ratio"] for f in fovs]
    m_scales = [results["Metric3D"][str(f)]["scale_ratio"] for f in fovs]

    d_errors = [results["Dioptra-DINO"][str(f)]["abs_rel"] for f in fovs]
    u_errors = [results["UniDepth-V2"][str(f)]["abs_rel"] for f in fovs]
    m_errors = [results["Metric3D"][str(f)]["abs_rel"] for f in fovs]

    # Plot Scale Ratio
    ax_scale.plot(fovs, d_scales, "o-", color="#2563eb", linewidth=2.5, label="Dioptra-DINO (Ours)")
    ax_scale.plot(fovs, u_scales, "s--", color="#10b981", linewidth=2.0, label="UniDepth-V2 ViT-S")
    ax_scale.plot(fovs, m_scales, "^-.", color="#ef4444", linewidth=2.0, label="Metric3D ViT-S")
    ax_scale.axhline(1.0, color="#6b7280", linestyle=":", label="Ideal Scale (s=1.0)")
    ax_scale.set_title("Metric Scale Ratio vs Synthetic Camera FOV", fontsize=11, fontweight="bold")
    ax_scale.set_xlabel("Camera FOV (°)", fontsize=10)
    ax_scale.set_ylabel("Median Scale Ratio", fontsize=10)
    ax_scale.grid(True, linestyle="--", alpha=0.4)
    ax_scale.legend(fontsize=9)

    # Plot AbsRel Error
    ax_err.plot(fovs, d_errors, "o-", color="#2563eb", linewidth=2.5, label="Dioptra-DINO (Ours)")
    ax_err.plot(fovs, u_errors, "s--", color="#10b981", linewidth=2.0, label="UniDepth-V2 ViT-S")
    ax_err.plot(fovs, m_errors, "^-.", color="#ef4444", linewidth=2.0, label="Metric3D ViT-S")
    ax_err.axvline(73.74, color="#f59e0b", linestyle="--", alpha=0.7, label="Native Sensor FOV (73.7°)")
    ax_err.set_title("AbsRel Error vs Synthetic Camera FOV", fontsize=11, fontweight="bold")
    ax_err.set_xlabel("Camera FOV (°)", fontsize=10)
    ax_err.set_ylabel("Mean AbsRel", fontsize=10)
    ax_err.grid(True, linestyle="--", alpha=0.4)
    ax_err.legend(fontsize=9)

    axes[0, 0].set_ylabel("Dioptra-DINO\n(CAFM + ARA)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("UniDepth-V2\n(Spherical)", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("Metric3D\n(Canonical)", fontsize=10, fontweight="bold")

    for r in range(3):
        for c in range(len(fovs)):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    out_paper_fig = "paper/figures/fig_multi_fov_comparison.png"
    out_outputs_fig = "outputs/fig_multi_fov_comparison.png"
    os.makedirs(os.path.dirname(out_paper_fig), exist_ok=True)
    os.makedirs(os.path.dirname(out_outputs_fig), exist_ok=True)
    fig.savefig(out_paper_fig, dpi=300, bbox_inches="tight")
    fig.savefig(out_outputs_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Publication figure saved to {out_paper_fig} and {out_outputs_fig}")


if __name__ == "__main__":
    main()
