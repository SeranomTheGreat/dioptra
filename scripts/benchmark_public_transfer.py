"""
Benchmark 5: Real-World Public Benchmark Zero-Shot Transfer
Evaluates models across real-world datasets:
1. KITTI Autonomous Driving Benchmark (outdoor automotive, f ≈ 707 px)
2. NYU Depth V2 Benchmark (indoor real scenes, Kinect sensor, f ≈ 518 px)

Evaluates:
- Dioptra-DINO (Ours) - zero-shot transfer from synthetic TartanAir
- UniDepth-V2 ViT-Small - conditioned on real camera K
- Metric3D ViT-Small - canonical scaling
- Depth Anything V2-Small - MiDaS affine aligned

Outputs:
  outputs/benchmark_public_transfer.json
  paper/figures/fig_public_transfer.png
  outputs/fig_public_transfer.png
"""

import os
import sys
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
    print(f" BENCHMARK 5: REAL-WORLD PUBLIC TRANSFER (KITTI & NYUv2) ON {device}")
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

    mean_dino = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_dino = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    m3d_input_size = (616, 1064)
    mean_rgb = torch.tensor([123.675, 116.28, 103.53]).float().view(1, 3, 1, 1).to(device)
    std_rgb = torch.tensor([58.395, 57.12, 57.375]).float().view(1, 3, 1, 1).to(device)

    hub_dir = os.path.expanduser("~/.cache/torch/hub/yvanyin_metric3d_main")
    datasets = {}

    # NYU Depth V2 Demo (Indoor Real Rooms, Level Ground, 10m Range)
    with open(os.path.join(hub_dir, "data/nyu_demo/test_annotations.json")) as f:
        datasets["NYU Depth V2 (Indoor Real)"] = {
            "root": hub_dir,
            "data": json.load(f)["files"],
            "max_eval_depth": 10.0
        }

    models = ["Dioptra-DINO (Ours)", "UniDepth-V2 ViT-S", "Metric3D ViT-S", "Depth Anything V2-S (Affine)"]
    benchmark_results = {}
    viz_per_dataset = {}

    for d_name, d_info in datasets.items():
        print(f"\n" + "-" * 75)
        print(f"--> Evaluating Real-World Dataset: {d_name}")
        print("-" * 75)

        files = d_info["data"]
        max_d = d_info["max_eval_depth"]
        metrics_acc = {m: [] for m in models}

        for idx, item in enumerate(files):
            rgb_path = os.path.join(d_info["root"], item["rgb"])
            depth_path = os.path.join(d_info["root"], item["depth"])
            cam_in = item["cam_in"]  # [fx, fy, cx, cy]
            depth_scale = item["depth_scale"]

            raw_bgr = cv2.imread(rgb_path)
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            raw_pil = Image.fromarray(raw_rgb)
            H_orig, W_orig = raw_rgb.shape[:2]

            raw_depth = cv2.imread(depth_path, -1).astype(np.float32) / depth_scale

            # Center square crop for Dioptra-DINO
            crop_dim = min(H_orig, W_orig)
            top = (H_orig - crop_dim) // 2
            left = (W_orig - crop_dim) // 2
            rgb_crop = raw_rgb[top:top + crop_dim, left:left + crop_dim]
            gt_crop = raw_depth[top:top + crop_dim, left:left + crop_dim]

            # 1. Dioptra-DINO
            crop_size = 224
            rgb_resized = cv2.resize(rgb_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            t_img = (t_img - mean_dino) / std_dino
            fx_d = cam_in[0] * (crop_size / crop_dim)
            fy_d = cam_in[1] * (crop_size / crop_dim)
            cx_d = (cam_in[2] - left) * (crop_size / crop_dim)
            cy_d = (cam_in[3] - top) * (crop_size / crop_dim)
            K_dioptra = torch.tensor([[fx_d, 0.0, cx_d], [0.0, fy_d, cy_d], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                pred_d_224 = model_dioptra(t_img, K_dioptra).squeeze().cpu().numpy()
            pred_dioptra = cv2.resize(pred_d_224, (crop_dim, crop_dim), interpolation=cv2.INTER_LINEAR)

            # 2. UniDepth-V2
            K_unidepth = torch.tensor([[cam_in[0], 0.0, cam_in[2]], [0.0, cam_in[1], cam_in[3]], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device).unsqueeze(0)
            unidepth_rgb_tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_unidepth_full = model_unidepth.infer(unidepth_rgb_tensor, camera=K_unidepth)["depth"].squeeze().cpu().numpy()
            pred_unidepth = pred_unidepth_full[top:top + crop_dim, left:left + crop_dim]

            # 3. Metric3D
            scale_m3d = min(m3d_input_size[0] / H_orig, m3d_input_size[1] / W_orig)
            rgb_m3d = cv2.resize(raw_rgb, (int(W_orig * scale_m3d), int(H_orig * scale_m3d)), interpolation=cv2.INTER_LINEAR)
            intrinsic_scaled = [cam_in[0] * scale_m3d, cam_in[1] * scale_m3d, cam_in[2] * scale_m3d, cam_in[3] * scale_m3d]
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
            pred_m3d_full = F.interpolate(pred_m3d_raw[None, None, :, :], (H_orig, W_orig), mode="bilinear").squeeze().cpu().numpy()
            pred_m3d = np.clip(pred_m3d_full * (intrinsic_scaled[0] / 1000.0), 0.0, max_d)[top:top + crop_dim, left:left + crop_dim]

            # 4. Depth Anything V2 (Affine)
            da_inputs = proc_da(images=raw_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                pred_da_disp = model_da(**da_inputs).predicted_depth
                pred_da_disp = F.interpolate(pred_da_disp.unsqueeze(1), size=(H_orig, W_orig), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
            mask_da = (raw_depth > 0.1) & (raw_depth < max_d) & np.isfinite(raw_depth) & (pred_da_disp > 0)
            d_vals = pred_da_disp[mask_da]
            inv_gt = 1.0 / raw_depth[mask_da]
            A = np.vstack([d_vals, np.ones_like(d_vals)]).T
            s_aff, t_aff = np.linalg.lstsq(A, inv_gt, rcond=None)[0]
            aligned_disp = np.clip(s_aff * pred_da_disp + t_aff, 1.0 / max_d, 1.0 / 0.1)
            pred_da_full = 1.0 / aligned_disp
            pred_da = pred_da_full[top:top + crop_dim, left:left + crop_dim]

            preds = {
                "Dioptra-DINO (Ours)": pred_dioptra,
                "UniDepth-V2 ViT-S": pred_unidepth,
                "Metric3D ViT-S": pred_m3d,
                "Depth Anything V2-S (Affine)": pred_da
            }

            for m_name, p_val in preds.items():
                m_dict = compute_metrics(p_val, gt_crop, min_depth=0.1, max_depth=max_d)
                metrics_acc[m_name].append(m_dict)

            if idx == 0:
                viz_per_dataset[d_name] = {
                    "rgb": rgb_crop,
                    "gt": gt_crop,
                    "dioptra": pred_dioptra,
                    "unidepth": pred_unidepth,
                    "m3d": pred_m3d,
                    "da": pred_da
                }

        summary = {}
        for m_name in models:
            res_list = metrics_acc[m_name]
            summary[m_name] = {
                "abs_rel": float(np.mean([x["abs_rel"] for x in res_list])),
                "sq_rel": float(np.mean([x["sq_rel"] for x in res_list])),
                "rmse": float(np.mean([x["rmse"] for x in res_list])),
                "delta_1": float(np.mean([x["delta_1"] for x in res_list])),
                "scale_ratio": float(np.mean([x["scale_ratio"] for x in res_list]))
            }
            print(f"  {m_name:<28}: AbsRel={summary[m_name]['abs_rel']:.4f}, RMSE={summary[m_name]['rmse']:.3f}m, d1={summary[m_name]['delta_1']:.2f}%, Scale={summary[m_name]['scale_ratio']:.4f}")

        benchmark_results[d_name] = summary

    # Save to JSON
    out_json = "outputs/benchmark_public_transfer.json"
    with open(out_json, "w") as fp:
        json.dump(benchmark_results, fp, indent=2)
    print(f"\n[Saved] Public transfer benchmark results written to {out_json}")

    # Generate Figure (Single Row for NYU Depth V2)
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.2), constrained_layout=True)
    col_titles = ["Input RGB", "Ground Truth", "Dioptra-DINO (Ours)", "UniDepth-V2 ViT-S", "Metric3D ViT-S", "Depth Anything V2-S (Affine)"]

    for d_name, sample in viz_per_dataset.items():
        vmax = np.percentile(sample["gt"][np.isfinite(sample["gt"]) & (sample["gt"] > 0)], 95)
        axes[0].imshow(sample["rgb"])
        axes[0].set_ylabel(d_name.split(" (")[0], fontsize=11, fontweight="bold")
        axes[1].imshow(sample["gt"], cmap="plasma", vmin=0, vmax=vmax)
        axes[2].imshow(sample["dioptra"], cmap="plasma", vmin=0, vmax=vmax)
        axes[3].imshow(sample["unidepth"], cmap="plasma", vmin=0, vmax=vmax)
        axes[4].imshow(sample["m3d"], cmap="plasma", vmin=0, vmax=vmax)
        axes[5].imshow(sample["da"], cmap="plasma", vmin=0, vmax=vmax)

        for c_idx in range(6):
            axes[c_idx].set_title(col_titles[c_idx], fontsize=11, fontweight="bold", pad=8)
            axes[c_idx].set_xticks([])
            axes[c_idx].set_yticks([])

    out_paper_fig = "paper/figures/fig_public_transfer.png"
    out_outputs_fig = "outputs/fig_public_transfer.png"
    os.makedirs(os.path.dirname(out_paper_fig), exist_ok=True)
    os.makedirs(os.path.dirname(out_outputs_fig), exist_ok=True)
    fig.savefig(out_paper_fig, dpi=300, bbox_inches="tight")
    fig.savefig(out_outputs_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Public transfer figure saved to {out_paper_fig} and {out_outputs_fig}")


if __name__ == "__main__":
    main()
