"""
Empirical Benchmark on Completely Unseen TartanAir Environment:
Environment: Gascola (Chemical Refinery Plant, Trajectory P001, 30 Frames)
Evaluates: Dioptra-DINO vs Foundation Baselines (UniDepth-V2, Metric3D, ZoeDepth, Depth Anything V2)
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
    print(f"BENCHMARK ON UNSEEN TARTANAIR ENVIRONMENT: GASCOLA P001 (30 CONSECUTIVE FRAMES) ON {device}")
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

    m3d_input_size = (616, 1064)
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

    data_dir = "test_samples/unseen_gascola_p001"
    img_files = sorted(glob.glob(os.path.join(data_dir, "image_left", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(data_dir, "depth_left", "*.npy")))

    assert len(img_files) > 0 and len(img_files) == len(depth_files), f"Mismatch: {len(img_files)} vs {len(depth_files)}"
    num_frames = len(img_files)
    print(f"\nDiscovered {num_frames} frames from completely unseen Gascola Chemical Refinery (P001).\n")

    models_to_test = [
        "Dioptra-DINO (Ours)",
        "UniDepth-V2 ViT-Small (Direct)",
        "Metric3D ViT-Small (Direct)",
        "Metric3D ViT-Small (Oracle Median)",
        "ZoeDepth ZoeD_NK (Metric Bins)",
        "Depth Anything V2-S (MiDaS Affine)",
        "Depth Anything V2-S (Oracle Median)",
    ]
    results = {m: {"abs_rel": [], "sq_rel": [], "rmse": [], "a1": [], "scale": []} for m in models_to_test}
    per_frame_records = []

    for idx in range(num_frames):
        fname = os.path.basename(img_files[idx])
        raw_bgr = cv2.imread(img_files[idx])
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        raw_gt = np.load(depth_files[idx]).astype(np.float32)
        raw_pil = Image.fromarray(raw_rgb)
        frame_rec = {"frame_idx": idx, "filename": fname}

        # 1. Dioptra-DINO
        left_crop = (orig_W - orig_H) // 2  # 80 px
        top_crop = 0
        min_side = orig_H
        rgb_crop = raw_rgb[top_crop:top_crop + min_side, left_crop:left_crop + min_side]
        rgb_resized = cv2.resize(rgb_crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t_img = (t_img - mean_dino) / std_dino
        with torch.no_grad():
            pred_dioptra_224 = model_dioptra(t_img, dioptra_K).squeeze().cpu().numpy()
        gt_crop = raw_gt[top_crop:top_crop + min_side, left_crop:left_crop + min_side]
        crop_gt_224 = cv2.resize(gt_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
        m_dioptra = compute_metrics(pred_dioptra_224, crop_gt_224)
        for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
            results["Dioptra-DINO (Ours)"][k].append(m_dioptra[k])
        results["Dioptra-DINO (Ours)"]["scale"].append(m_dioptra["scale_ratio"])
        frame_rec["dioptra_absrel"] = m_dioptra["abs_rel"]
        frame_rec["dioptra_rmse"] = m_dioptra["rmse"]
        frame_rec["dioptra_scale"] = m_dioptra["scale_ratio"]

        # 2. UniDepth-V2
        rgb_unidepth = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_unidepth = model_unidepth.infer(rgb_unidepth, camera=orig_K_tensor)["depth"].squeeze().cpu().numpy()
        m_unidepth = compute_metrics(pred_unidepth, raw_gt)
        for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
            results["UniDepth-V2 ViT-Small (Direct)"][k].append(m_unidepth[k])
        results["UniDepth-V2 ViT-Small (Direct)"]["scale"].append(m_unidepth["scale_ratio"])
        frame_rec["unidepth_absrel"] = m_unidepth["abs_rel"]
        frame_rec["unidepth_rmse"] = m_unidepth["rmse"]

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
            pred_depth, _, _ = model_metric3d.inference({"input": norm_rgb})
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_h_half : pred_depth.shape[0] - (pad_h - pad_h_half), pad_w_half : pred_depth.shape[1] - (pad_w - pad_w_half)]
        pred_depth_full = F.interpolate(pred_depth[None, None, :, :], (orig_H, orig_W), mode="bilinear").squeeze().cpu().numpy()
        canonical_scale = intrinsic_scaled[0] / 1000.0
        m3d_pred_direct = np.clip(pred_depth_full * canonical_scale, 0.0, 300.0)
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

        # 4. ZoeDepth
        with torch.no_grad():
            pred_zoe = model_zoe.infer_pil(raw_pil)
        m_zoe = compute_metrics(pred_zoe, raw_gt)
        for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
            results["ZoeDepth ZoeD_NK (Metric Bins)"][k].append(m_zoe[k])
        results["ZoeDepth ZoeD_NK (Metric Bins)"]["scale"].append(m_zoe["scale_ratio"])

        # 5. Depth Anything V2
        da_inputs = proc_da(images=raw_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            pred_disp = model_da(**da_inputs).predicted_depth
            pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=(orig_H, orig_W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

        # MiDaS affine fit
        mask_da = (raw_gt > 0.1) & (raw_gt < 80.0) & (pred_disp > 0.01)
        if mask_da.sum() > 50:
            inv_gt = 1.0 / np.clip(raw_gt[mask_da], 0.1, 80.0)
            d_pts = pred_disp[mask_da]
            A = np.vstack([d_pts, np.ones_like(d_pts)]).T
            s_fit, t_fit = np.linalg.lstsq(A, inv_gt, rcond=None)[0]
            pred_da_affine = 1.0 / np.clip(s_fit * pred_disp + t_fit, 1e-4, 10.0)
            pred_da_affine = np.clip(pred_da_affine, 0.1, 80.0)
        else:
            pred_da_affine = np.ones_like(pred_disp) * 10.0
        m_da_aff = compute_metrics(pred_da_affine, raw_gt)
        for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
            results["Depth Anything V2-S (MiDaS Affine)"][k].append(m_da_aff[k])
        results["Depth Anything V2-S (MiDaS Affine)"]["scale"].append(m_da_aff["scale_ratio"])

        # Oracle Median Scale
        rel_depth_da = 1.0 / np.clip(pred_disp, 1e-4, None)
        scale_da_med = float(np.median(raw_gt[mask_da]) / np.median(rel_depth_da[mask_da])) if mask_da.sum() > 50 else 1.0
        pred_da_med = rel_depth_da * scale_da_med
        m_da_med = compute_metrics(pred_da_med, raw_gt)
        for k in ["abs_rel", "sq_rel", "rmse", "a1"]:
            results["Depth Anything V2-S (Oracle Median)"][k].append(m_da_med[k])
        results["Depth Anything V2-S (Oracle Median)"]["scale"].append(m_da_med["scale_ratio"])

        per_frame_records.append(frame_rec)

        if (idx + 1) % 5 == 0 or (idx + 1) == num_frames:
            print(f"[{idx+1:02d}/{num_frames}] Dioptra AbsRel: {m_dioptra['abs_rel']:.4f} (scale {m_dioptra['scale_ratio']:.3f}) | UniDepth: {m_unidepth['abs_rel']:.4f} | M3D: {m_m3d_dir['abs_rel']:.4f} | Zoe: {m_zoe['abs_rel']:.4f} | DA (Aff): {m_da_aff['abs_rel']:.4f}")

    print("\n" + "=" * 105)
    print(" EMPIRICAL RESULTS: UNSEEN TARTANAIR GASCOLA P001 (30 CONSECUTIVE FRAMES)")
    print("=" * 105)
    header = f"{'Model Architecture':<36} | {'AbsRel':<7} | {'SqRel':<7} | {'RMSE (m)':<8} | {'δ1 < 1.25':<9} | {'Scale Ratio':<10}"
    print(header)
    print("-" * 105)

    summary = {}
    for m in models_to_test:
        mean_abs = float(np.mean(results[m]["abs_rel"]))
        mean_sq = float(np.mean(results[m]["sq_rel"]))
        mean_rmse = float(np.mean(results[m]["rmse"]))
        mean_a1 = float(np.mean(results[m]["a1"]))
        median_scale = float(np.median(results[m]["scale"]))
        summary[m] = {
            "abs_rel": round(mean_abs, 4),
            "sq_rel": round(mean_sq, 4),
            "rmse": round(mean_rmse, 3),
            "delta1": round(mean_a1, 2),
            "scale_ratio": round(median_scale, 4),
            "per_frame_absrel": results[m]["abs_rel"],
            "per_frame_scale": results[m]["scale"]
        }
        print(f"{m:<36} | {mean_abs:<7.4f} | {mean_sq:<7.4f} | {mean_rmse:<8.3f} | {mean_a1:<8.2f}% | {median_scale:<10.4f}")
    print("=" * 105)

    # Save outputs
    out_json = "outputs/benchmark_unseen_gascola_p001.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] Full empirical benchmark results written to {out_json}")


if __name__ == "__main__":
    main()
