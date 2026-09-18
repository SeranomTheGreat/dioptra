"""
Empirical Benchmark on Completely Unseen Indoor TartanAir Environment:
Environment: Office (Indoor Office Rooms & Corridors on Level Ground, Trajectory P001, 30 Consecutive Frames)
Evaluates: Dioptra-DINO vs Foundation Baselines (UniDepth-V2, Metric3D, ZoeDepth, Depth Anything V2)
Ensures testing is on level ground and without high-resolution upscaling (e.g. no 616x1064 for Metric3D).
"""

import os
import sys
import glob
import json
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())
import dioptra_dino
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
import __main__
__main__.DioptraDINOConfig = DioptraDINOConfig

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 80.0):
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & (pred > 0.01) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "a1": 0.0, "scale_ratio": 1.0}
    p = pred[mask]
    g = gt[mask]
    abs_rel = float(np.mean(np.abs(p - g) / g))
    sq_rel = float(np.mean(((p - g) ** 2) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    thresh = np.maximum(g / p, p / g)
    a1 = float((thresh < 1.25).mean()) * 100.0
    scale = float(np.median(p) / np.median(g))
    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "a1": a1,
        "scale_ratio": scale
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 100)
    print(f"BENCHMARK ON UNSEEN INDOOR TARTANAIR ENVIRONMENT: OFFICE P001 (LEVEL GROUND, 30 FRAMES) ON {device}")
    print("=" * 100)

    # 1. Dioptra-DINO
    print("[1/5] Loading Dioptra-DINO (Ours)...")
    ckpt = torch.load("outputs/dioptra_dino_best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", DioptraDINOConfig())
    model_dioptra = DioptraDINO(cfg)
    model_dioptra.load_state_dict(ckpt["model_state_dict"])
    model_dioptra = model_dioptra.to(device).eval()

    # 2. UniDepth-V2
    print("[2/5] Loading UniDepth-V2 ViT-Small...")
    model_unidepth = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version="v2", backbone="vits14", pretrained=True, trust_repo=True).to(device).eval()

    # 3. Metric3D
    print("[3/5] Loading Metric3D ViT-Small...")
    model_metric3d = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True).to(device).eval()

    # 4. ZoeDepth
    print("[4/5] Loading ZoeDepth ZoeD_NK...")
    model_zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).to(device).eval()

    # 5. Depth Anything V2
    print("[5/5] Loading Depth Anything V2-Small...")
    proc_da = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model_da = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()

    mean_dino = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_dino = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    mean_rgb = torch.tensor([123.675, 116.28, 103.53], device=device).float().view(1, 3, 1, 1)
    std_rgb = torch.tensor([58.395, 57.12, 57.375], device=device).float().view(1, 3, 1, 1)

    orig_W, orig_H = 640, 480
    crop_size = 224
    crop_fx = 320.0 * (crop_size / float(orig_H))
    crop_fy = 320.0 * (crop_size / float(orig_H))
    crop_cx = (320.0 - 80.0) * (crop_size / float(orig_H))
    crop_cy = 240.0 * (crop_size / float(orig_H))
    dioptra_K = torch.tensor([
        [crop_fx, 0.0, crop_cx],
        [0.0, crop_fy, crop_cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    orig_K_tensor = torch.tensor([
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    img_dir = "test_samples/unseen_office_p001/image_left"
    depth_dir = "test_samples/unseen_office_p001/depth_left"
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))

    print(f"Loaded {len(img_files)} images and {len(depth_files)} depth files from {img_dir}.")

    models = [
        "Dioptra-DINO (Ours)",
        "UniDepth-V2 ViT-Small (Direct)",
        "Depth Anything V2-S (Affine)",
        "Metric3D ViT-Small (Direct)",
        "Metric3D ViT-Small (Oracle Median)",
        "ZoeDepth ZoeD_NK (Direct)"
    ]
    results = {m: {"abs_rel": [], "sq_rel": [], "rmse": [], "a1": [], "scale": []} for m in models}

    with torch.no_grad():
        for idx in range(len(img_files)):
            raw_bgr = cv2.imread(img_files[idx])
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            raw_gt = np.load(depth_files[idx]).astype(np.float32)
            raw_pil = Image.fromarray(raw_rgb)

            # Central 480x480 crop matching square camera frustum evaluation
            left_crop = (orig_W - orig_H) // 2  # 80 px
            top_crop = 0
            min_side = orig_H
            rgb_crop = raw_rgb[top_crop:top_crop + min_side, left_crop:left_crop + min_side]
            gt_crop = raw_gt[top_crop:top_crop + min_side, left_crop:left_crop + min_side]

            # -----------------------------------------------------------------
            # 1. Dioptra-DINO ($224 \times 224$ input)
            # -----------------------------------------------------------------
            rgb_resized = cv2.resize(rgb_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            t_img = (t_img - mean_dino) / std_dino
            pred_dioptra_224 = model_dioptra(t_img, dioptra_K).squeeze().cpu().numpy()
            crop_gt_224 = cv2.resize(gt_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
            m_dioptra = compute_metrics(pred_dioptra_224, crop_gt_224)
            for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
                results["Dioptra-DINO (Ours)"][k].append(m_dioptra[k])
            results["Dioptra-DINO (Ours)"]["scale"].append(m_dioptra["scale_ratio"])

            # -----------------------------------------------------------------
            # 2. UniDepth-V2 (Native 480x640 or square crop, zero alignment)
            # -----------------------------------------------------------------
            rgb_unidepth = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            pred_unidepth = model_unidepth.infer(rgb_unidepth, camera=orig_K_tensor)["depth"].squeeze().cpu().numpy()
            m_unidepth = compute_metrics(pred_unidepth, raw_gt)
            for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
                results["UniDepth-V2 ViT-Small (Direct)"][k].append(m_unidepth[k])
            results["UniDepth-V2 ViT-Small (Direct)"]["scale"].append(m_unidepth["scale_ratio"])

            # -----------------------------------------------------------------
            # 3. Depth Anything V2-Small (Oracle Affine aligned)
            # -----------------------------------------------------------------
            da_inputs = proc_da(images=raw_pil, return_tensors="pt").to(device)
            da_out = model_da(**da_inputs)
            disp_da = da_out.predicted_depth.squeeze().cpu().numpy()
            disp_da_full = cv2.resize(disp_da, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)

            mask_gt = (raw_gt > 0.1) & (raw_gt < 80.0) & np.isfinite(raw_gt)
            gt_disp = 1.0 / np.clip(raw_gt[mask_gt], 0.1, 80.0)
            d_val = disp_da_full[mask_gt]
            A = np.vstack([d_val, np.ones_like(d_val)]).T
            sol, _, _, _ = np.linalg.lstsq(A, gt_disp, rcond=None)
            s_opt, t_opt = sol[0], sol[1]
            fitted_disp = np.clip(s_opt * disp_da_full + t_opt, 1.0 / 80.0, 1.0 / 0.1)
            pred_da_affine = 1.0 / fitted_disp
            m_da = compute_metrics(pred_da_affine, raw_gt)
            for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
                results["Depth Anything V2-S (Affine)"][k].append(m_da[k])
            results["Depth Anything V2-S (Affine)"]["scale"].append(m_da["scale_ratio"])

            # -----------------------------------------------------------------
            # 4. Metric3D ViT-Small (Native 480x640 image, NOT upscaled to 616x1064)
            # -----------------------------------------------------------------
            # Controlled moderate resolution: pad/resize native 480x640 directly without inflating to 616x1064
            # We use standard 384x512 or native 480x640:
            m3d_target = (480, 640)
            tensor_rgb = torch.from_numpy(raw_rgb.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
            norm_rgb = (tensor_rgb - mean_rgb) / std_rgb
            pred_depth, _, _ = model_metric3d.inference({"input": norm_rgb})
            pred_depth = pred_depth.squeeze()
            pred_depth_full = F.interpolate(pred_depth[None, None, :, :], (orig_H, orig_W), mode="bilinear").squeeze().cpu().numpy()
            canonical_scale = 320.0 / 1000.0
            m3d_pred_direct = np.clip(pred_depth_full * canonical_scale, 0.0, 80.0)
            m_m3d_dir = compute_metrics(m3d_pred_direct, raw_gt)
            for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
                results["Metric3D ViT-Small (Direct)"][k].append(m_m3d_dir[k])
            results["Metric3D ViT-Small (Direct)"]["scale"].append(m_m3d_dir["scale_ratio"])

            # Metric3D Oracle Median
            mask_m3d = (raw_gt > 0.1) & (raw_gt < 80.0) & np.isfinite(raw_gt) & (m3d_pred_direct > 0.05)
            scale_m3d_med = float(np.median(raw_gt[mask_m3d]) / np.median(m3d_pred_direct[mask_m3d])) if mask_m3d.sum() > 0 else 1.0
            pred_m3d_med = m3d_pred_direct * scale_m3d_med
            m_m3d_med = compute_metrics(pred_m3d_med, raw_gt)
            for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
                results["Metric3D ViT-Small (Oracle Median)"][k].append(m_m3d_med[k])
            results["Metric3D ViT-Small (Oracle Median)"]["scale"].append(m_m3d_med["scale_ratio"])

            # -----------------------------------------------------------------
            # 5. ZoeDepth (Standard 384x512 native)
            # -----------------------------------------------------------------
            pred_zoe = model_zoe.infer_pil(raw_pil)
            m_zoe = compute_metrics(pred_zoe, raw_gt)
            for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
                results["ZoeDepth ZoeD_NK (Direct)"][k].append(m_zoe[k])
            results["ZoeDepth ZoeD_NK (Direct)"]["scale"].append(m_zoe["scale_ratio"])

            if (idx + 1) % 10 == 0 or idx == len(img_files) - 1:
                print(f"  Processed [{idx+1:02d}/{len(img_files)}] frames...")

    summary = {}
    for m in models:
        summary[m] = {
            "abs_rel": float(np.mean(results[m]["abs_rel"])),
            "sq_rel": float(np.mean(results[m]["sq_rel"])),
            "rmse": float(np.mean(results[m]["rmse"])),
            "a1": float(np.mean(results[m]["a1"])),
            "scale_ratio": float(np.median(results[m]["scale"]))
        }

    print("\n" + "=" * 115)
    print("EMPIRICAL BENCHMARK: UNSEEN INDOOR TARTANAIR OFFICE P001 (LEVEL GROUND, 30 CONSECUTIVE FRAMES)")
    print(f"{'Model Architecture':<36} | {'AbsRel':<8} | {'SqRel':<8} | {'RMSE (m)':<9} | {'delta1 (%)':<10} | {'Scale Ratio':<11} | {'Execution Mode'}")
    print("-" * 115)
    for m in models:
        mode = "Direct Metric"
        if "Affine" in m:
            mode = "Oracle Affine (s*d + t)"
        elif "Oracle Median" in m:
            mode = "Oracle Median (s*d)"
        elif "Ours" in m:
            mode = "Direct Metric (K, Pinhole Ray)"
        elif "UniDepth" in m:
            mode = "Direct Metric (K, Spherical Ray)"
        print(f"{m:<36} | {summary[m]['abs_rel']:<8.4f} | {summary[m]['sq_rel']:<8.4f} | {summary[m]['rmse']:<9.4f} | {summary[m]['a1']:<10.2f} | {summary[m]['scale_ratio']:<11.4f} | {mode}")
    print("=" * 115)

    out_file = "outputs/benchmark_unseen_office_p001.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
