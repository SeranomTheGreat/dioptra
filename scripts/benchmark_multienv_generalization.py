"""
Benchmark 3: Multi-Environment Cross-Domain Generalization
Evaluates zero-shot domain transfer across 4 fundamentally distinct environments:
1. Industrial Factory (Daytime): Complex structural occlusion, scaffolding, rust textures
2. Outdoor Amusement Park: Bright sunlight, rollercoasters, foliage, deep open sky
3. Factory Night (Low-Light): Extreme dynamic range, flashlight cones, specular metal reflections
4. Hospital Indoor: Clean indoor rooms, linoleum flooring, specular walls, medical lamps

Evaluates models under identical fair-ground conditions:
- Dioptra-DINO (Ours): Camera-conditioned with ARA
- UniDepth-V2 ViT-Small: Exact camera intrinsics K provided
- Metric3D ViT-Small: Exact focal length fx and canonical scaling provided
- Depth Anything V2-Small (Affine): MiDaS least-squares disparity alignment

Saves results to:
  outputs/benchmark_multienv_generalization.json
  paper/figures/fig_multienv_generalization.png
  outputs/fig_multienv_generalization.png
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


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth=0.1, max_depth=80.0):
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if mask.sum() == 0:
        return {}
    p = pred[mask]
    g = gt[mask]

    thresh = np.maximum(p / g, g / p)
    d1 = (thresh < 1.25).mean() * 100.0
    d2 = (thresh < 1.25 ** 2).mean() * 100.0
    d3 = (thresh < 1.25 ** 3).mean() * 100.0

    abs_rel = np.mean(np.abs(p - g) / g)
    sq_rel = np.mean(((p - g) ** 2) / g)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    scale = np.median(p) / np.median(g)

    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "delta_1": float(d1),
        "delta_2": float(d2),
        "delta_3": float(d3),
        "scale_ratio": float(scale),
        "num_valid_px": int(mask.sum())
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 95)
    print(f" BENCHMARK 3: MULTI-ENVIRONMENT CROSS-DOMAIN GENERALIZATION ON {device}")
    print("=" * 95)

    # 1. Load Models
    print("\n[1/4] Loading Dioptra-DINO...")
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

    # Preprocessing constants
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

    left_crop = (orig_W - orig_H) // 2

    # Define environments
    environments = {
        "Industrial Factory (Day)": ["test_samples/abandonedfactory", "test_samples/abandonedfactory_hard"],
        "Amusement Park (Outdoors)": ["test_samples/amusement", "test_samples/amusement_hard"],
        "Factory Night (Low-Light)": ["test_samples/abandonedfactory_night", "test_samples/abandonedfactory_night_hard"],
        "Hospital (Indoor Specular)": ["test_samples/hospital"]
    }

    models = ["Dioptra-DINO (Ours)", "UniDepth-V2 ViT-S", "Metric3D ViT-S", "Depth Anything V2-S (Affine)"]
    env_results = {}
    viz_per_env = {}

    for env_name, paths in environments.items():
        print(f"\n" + "-" * 75)
        print(f"--> Evaluating Environment: {env_name}")
        print("-" * 75)

        pairs = []
        for p in paths:
            for img in sorted(glob.glob(f"{p}/*.png")):
                base = img[:-4]
                dep = base + "_depth.npy"
                if not os.path.exists(dep):
                    dep = base.replace("image_left", "depth_left") + "_depth.npy"
                if os.path.exists(dep):
                    pairs.append((img, dep))

        print(f"Found {len(pairs)} frames for {env_name}.")
        metrics_env = {m: [] for m in models}

        for idx, (img_p, gt_p) in enumerate(pairs):
            raw_bgr = cv2.imread(img_p)
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            raw_pil = Image.fromarray(raw_rgb)
            raw_gt = np.load(gt_p).astype(np.float32)

            gt_eval = raw_gt[:, left_crop:left_crop + orig_H]
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

            # 4. Depth Anything V2 (Affine aligned)
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

            preds = {
                "Dioptra-DINO (Ours)": pred_dioptra,
                "UniDepth-V2 ViT-S": pred_unidepth,
                "Metric3D ViT-S": pred_m3d,
                "Depth Anything V2-S (Affine)": pred_da
            }

            for m_name, p_val in preds.items():
                m_dict = compute_metrics(p_val, gt_eval)
                metrics_env[m_name].append(m_dict)

            if idx == 0:
                viz_per_env[env_name] = {
                    "rgb": rgb_crop,
                    "gt": gt_eval,
                    "dioptra": pred_dioptra,
                    "unidepth": pred_unidepth,
                    "m3d": pred_m3d,
                    "da": pred_da
                }

        # Summarize this environment
        env_summary = {}
        for m_name in models:
            res_list = metrics_env[m_name]
            env_summary[m_name] = {
                "abs_rel": float(np.mean([x["abs_rel"] for x in res_list])),
                "sq_rel": float(np.mean([x["sq_rel"] for x in res_list])),
                "rmse": float(np.mean([x["rmse"] for x in res_list])),
                "delta_1": float(np.mean([x["delta_1"] for x in res_list])),
                "scale_ratio": float(np.mean([x["scale_ratio"] for x in res_list]))
            }
            print(f"  {m_name:<28}: AbsRel={env_summary[m_name]['abs_rel']:.4f}, RMSE={env_summary[m_name]['rmse']:.3f}m, d1={env_summary[m_name]['delta_1']:.2f}%, Scale={env_summary[m_name]['scale_ratio']:.4f}")

        env_results[env_name] = env_summary

    # Save to JSON
    out_json = "outputs/benchmark_multienv_generalization.json"
    with open(out_json, "w") as fp:
        json.dump(env_results, fp, indent=2)
    print(f"\n[Saved] Multi-environment benchmark results saved to {out_json}")

    # Generate Publication Figure
    print("\n[Viz] Generating multi-environment 4-row figure...")
    fig, axes = plt.subplots(4, 6, figsize=(18, 12), constrained_layout=True)
    col_titles = [
        "Input RGB",
        "Ground Truth",
        "Dioptra-DINO (Ours)",
        "UniDepth-V2 ViT-S",
        "Metric3D ViT-S",
        "Depth Anything V2-S (Affine)"
    ]

    for row_idx, (env_name, sample) in enumerate(viz_per_env.items()):
        vmax = np.percentile(sample["gt"][np.isfinite(sample["gt"]) & (sample["gt"] > 0)], 95)
        vmax = max(vmax, 10.0)

        axes[row_idx, 0].imshow(sample["rgb"])
        axes[row_idx, 0].set_ylabel(env_name.split(" (")[0], fontsize=11, fontweight="bold")

        im1 = axes[row_idx, 1].imshow(sample["gt"], cmap="plasma", vmin=0, vmax=vmax)
        axes[row_idx, 2].imshow(sample["dioptra"], cmap="plasma", vmin=0, vmax=vmax)
        axes[row_idx, 3].imshow(sample["unidepth"], cmap="plasma", vmin=0, vmax=vmax)
        axes[row_idx, 4].imshow(sample["m3d"], cmap="plasma", vmin=0, vmax=vmax)
        axes[row_idx, 5].imshow(sample["da"], cmap="plasma", vmin=0, vmax=vmax)

        for col_idx in range(6):
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=11, fontweight="bold", pad=8)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    out_paper_fig = "paper/figures/fig_multienv_generalization.png"
    out_outputs_fig = "outputs/fig_multienv_generalization.png"
    os.makedirs(os.path.dirname(out_paper_fig), exist_ok=True)
    os.makedirs(os.path.dirname(out_outputs_fig), exist_ok=True)
    fig.savefig(out_paper_fig, dpi=300, bbox_inches="tight")
    fig.savefig(out_outputs_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Publication figure saved to {out_paper_fig} and {out_outputs_fig}")


if __name__ == "__main__":
    main()
