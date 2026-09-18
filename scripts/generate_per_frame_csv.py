"""
Generate authoritative per-frame evaluation CSV for all 200 held-out benchmark frames.
Uses the exact official evaluation pipeline:
- Dioptra-DINO: Native pinhole square-aspect crop (480x480 -> 224x224), Direct Metric.
- Depth Anything V2 Small: Official MiDaS disparity-space affine alignment (1 / (s*d + t)) and median scaling.
- Metric3D ViT-Small: Native 616x1064 resolution, both Direct Metric and Oracle Median Scaling.

Saves to:
  outputs/benchmark_200_per_frame_comparison.csv
  eval_receipts/benchmark_200_per_frame_comparison.csv
"""

import os
import sys
import glob
import csv
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import dioptra_dino
import __main__
__main__.DioptraDINOConfig = dioptra_dino.DioptraDINOConfig
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_d=0.1, max_d=80.0):
    mask = (gt > min_d) & (gt < max_d) & np.isfinite(gt) & (pred > min_d) & (pred < max_d) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "delta1": 0.0, "scale": 1.0}
    p = pred[mask]
    g = gt[mask]
    thresh = np.maximum(g / p, p / g)
    d1 = float((thresh < 1.25).mean() * 100.0)
    abs_rel = float((np.abs(g - p) / g).mean())
    sq_rel = float((((g - p) ** 2) / g).mean())
    rmse = float(np.sqrt(((g - p) ** 2).mean()))
    scale = float(np.median(p) / np.median(g))
    return {"abs_rel": abs_rel, "sq_rel": sq_rel, "rmse": rmse, "delta1": d1, "scale": scale}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Authoritative CSV Generator] Device: {device}")

    data_dir = "test_samples/unseen_200_abandonedfactory"
    img_files = sorted(glob.glob(os.path.join(data_dir, "image_left", "*.png")))
    gt_files = sorted(glob.glob(os.path.join(data_dir, "depth_left", "*.npy")))

    assert len(img_files) == 200, f"Found {len(img_files)} images"
    assert len(gt_files) == 200, f"Found {len(gt_files)} depths"

    # 1. Load Dioptra-DINO
    print("[1/3] Loading Dioptra-DINO...")
    cfg_dino = DioptraDINOConfig(freeze_backbone=False)
    dioptra_model = DioptraDINO(cfg_dino).to(device)
    st = torch.load("outputs/dioptra_dino_best.pt", map_location=device, weights_only=False)
    sd = st.get("model_state_dict", st)
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    dioptra_model.load_state_dict(clean_sd, strict=False)
    dioptra_model.eval()

    # 2. Load Depth Anything V2 Small
    print("[2/3] Loading Depth Anything V2 Small...")
    da_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    da_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device)
    da_model.eval()

    # 3. Load Metric3D ViT-Small
    print("[3/3] Loading Metric3D ViT-Small...")
    metric3d_model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_small", pretrain=True).to(device)
    metric3d_model.eval()

    fieldnames = [
        "frame_idx",
        "filename",
        "dioptra_absrel",
        "dioptra_rmse",
        "dioptra_delta1",
        "dioptra_scale",
        "metric3d_direct_absrel",
        "metric3d_direct_rmse",
        "metric3d_direct_delta1",
        "metric3d_direct_scale",
        "metric3d_median_absrel",
        "metric3d_median_rmse",
        "metric3d_median_delta1",
        "da_v2_affine_absrel",
        "da_v2_affine_rmse",
        "da_v2_affine_delta1",
        "da_v2_median_absrel",
        "da_v2_median_rmse",
        "da_v2_median_delta1",
    ]

    # Pre-setup TartanAir crop geometry
    orig_w, orig_h = 640, 480
    min_side = 480
    left = (orig_w - min_side) // 2
    top = (orig_h - min_side) // 2
    img_size = 224

    fx_dino = 320.0 * (float(img_size) / float(min_side))
    K_dino = torch.tensor([[[fx_dino, 0.0, img_size / 2.0],
                            [0.0, fx_dino, img_size / 2.0],
                            [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

    mean_dino = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_dino = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # Metric3D padding constants
    input_size_m = (616, 1064)
    scale_m = min(input_size_m[0] / orig_h, input_size_m[1] / orig_w)
    nw_m, nh_m = int(orig_w * scale_m), int(orig_h * scale_m)
    pad_h_m = input_size_m[0] - nh_m
    pad_w_m = input_size_m[1] - nw_m
    pad_top_m, pad_left_m = pad_h_m // 2, pad_w_m // 2
    canonical_scale_m = (320.0 * scale_m) / 1000.0
    mean_m = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None].to(device)
    std_m = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None].to(device)

    rows = []
    print(f"[Authoritative CSV Generator] Processing {len(img_files)} frames...")
    t0 = time.time()

    for idx, (img_p, gt_p) in enumerate(zip(img_files, gt_files)):
        fname = os.path.basename(img_p)
        raw_pil = Image.open(img_p).convert("RGB")
        raw_np = np.array(raw_pil)
        gt_raw = np.load(gt_p).astype(np.float32)

        # ------------------------------------------------------------------
        # 1. Dioptra-DINO (Native Aspect Square Crop Protocol)
        # ------------------------------------------------------------------
        rgb_crop = raw_np[top : top + min_side, left : left + min_side]
        rgb_resized = cv2.resize(rgb_crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        t_img = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t_img = (t_img - mean_dino) / std_dino

        with torch.no_grad():
            pred_dino = dioptra_model(t_img, K_dino).squeeze().cpu().numpy()

        gt_crop = gt_raw[top : top + min_side, left : left + min_side]
        gt_dino = cv2.resize(gt_crop, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        m_dioptra = compute_metrics(pred_dino, gt_dino)

        # ------------------------------------------------------------------
        # 2. Depth Anything V2 Small (Disparity-Space Affine & Median Scaling)
        # ------------------------------------------------------------------
        gt_pil = Image.fromarray(gt_raw)
        gt_224 = np.array(gt_pil.resize((224, 224), Image.NEAREST))

        inputs_da = da_processor(images=raw_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_da = da_model(**inputs_da)
            pred_disp = F.interpolate(outputs_da.predicted_depth.unsqueeze(1), size=(224, 224),
                                      mode="bilinear", align_corners=False).squeeze().cpu().numpy()

        valid_da = (gt_224 > 0.1) & (gt_224 < 80.0) & np.isfinite(gt_224) & (pred_disp > 0)
        g_d = gt_224[valid_da]
        g_disp = 1.0 / g_d
        d_val = pred_disp[valid_da]

        # MiDaS Disparity Affine Alignment: s * d + t in disparity space
        A = np.vstack([d_val, np.ones_like(d_val)]).T
        s_aff, t_aff = np.linalg.lstsq(A, g_disp, rcond=None)[0]
        aligned_disp = np.maximum(s_aff * d_val + t_aff, 1e-4)
        aligned_depth_da = 1.0 / aligned_disp
        full_da_affine = np.zeros_like(gt_224)
        full_da_affine[valid_da] = aligned_depth_da
        m_da_affine = compute_metrics(full_da_affine, gt_224)

        # Median Scaling in Disparity Space
        inv_d = 1.0 / np.maximum(d_val, 1e-3)
        scale_med = np.median(g_d) / np.median(inv_d)
        med_depth_da = inv_d * scale_med
        full_da_median = np.zeros_like(gt_224)
        full_da_median[valid_da] = med_depth_da
        m_da_median = compute_metrics(full_da_median, gt_224)

        # ------------------------------------------------------------------
        # 3. Metric3D ViT-Small (Native 616x1064, Direct & Oracle Median)
        # ------------------------------------------------------------------
        rgb_m = cv2.resize(raw_np, (nw_m, nh_m), interpolation=cv2.INTER_LINEAR)
        rgb_padded = cv2.copyMakeBorder(rgb_m, pad_top_m, pad_h_m - pad_top_m,
                                        pad_left_m, pad_w_m - pad_left_m,
                                        cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
        t_m3d = torch.from_numpy(rgb_padded.transpose(2, 0, 1)).float().to(device)
        norm_rgb = torch.div((t_m3d - mean_m), std_m).unsqueeze(0)

        with torch.no_grad():
            pred_m_raw, _, _ = metric3d_model.inference({"input": norm_rgb})

        pred_crop_m = pred_m_raw.squeeze()[pad_top_m : pad_top_m + nh_m, pad_left_m : pad_left_m + nw_m]
        pred_res_m = F.interpolate(pred_crop_m.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w),
                                   mode="bilinear", align_corners=False).squeeze()
        m3d_direct = (pred_res_m * canonical_scale_m).clamp(0, 300).cpu().numpy()
        m_m3d_direct = compute_metrics(m3d_direct, gt_raw)

        # Oracle Median Scaling for Metric3D
        valid_m = (gt_raw > 0.1) & (gt_raw < 80.0) & np.isfinite(gt_raw) & (m3d_direct > 0.1) & np.isfinite(m3d_direct)
        s_m3d = np.median(gt_raw[valid_m]) / np.maximum(np.median(m3d_direct[valid_m]), 1e-6)
        m3d_median = s_m3d * m3d_direct
        m_m3d_median = compute_metrics(m3d_median, gt_raw)

        row = {
            "frame_idx": idx,
            "filename": fname,
            "dioptra_absrel": f"{m_dioptra['abs_rel']:.4f}",
            "dioptra_rmse": f"{m_dioptra['rmse']:.3f}",
            "dioptra_delta1": f"{m_dioptra['delta1']:.2f}",
            "dioptra_scale": f"{m_dioptra['scale']:.4f}",
            "metric3d_direct_absrel": f"{m_m3d_direct['abs_rel']:.4f}",
            "metric3d_direct_rmse": f"{m_m3d_direct['rmse']:.3f}",
            "metric3d_direct_delta1": f"{m_m3d_direct['delta1']:.2f}",
            "metric3d_direct_scale": f"{m_m3d_direct['scale']:.4f}",
            "metric3d_median_absrel": f"{m_m3d_median['abs_rel']:.4f}",
            "metric3d_median_rmse": f"{m_m3d_median['rmse']:.3f}",
            "metric3d_median_delta1": f"{m_m3d_median['delta1']:.2f}",
            "da_v2_affine_absrel": f"{m_da_affine['abs_rel']:.4f}",
            "da_v2_affine_rmse": f"{m_da_affine['rmse']:.3f}",
            "da_v2_affine_delta1": f"{m_da_affine['delta1']:.2f}",
            "da_v2_median_absrel": f"{m_da_median['abs_rel']:.4f}",
            "da_v2_median_rmse": f"{m_da_median['rmse']:.3f}",
            "da_v2_median_delta1": f"{m_da_median['delta1']:.2f}",
        }
        rows.append(row)

        if (idx + 1) % 25 == 0 or idx == len(img_files) - 1:
            elapsed = time.time() - t0
            print(f"  Processed {idx + 1}/{len(img_files)} frames ({elapsed:.1f}s)...")

    # Save to both locations
    for out_path in ["outputs/benchmark_200_per_frame_comparison.csv",
                     "eval_receipts/benchmark_200_per_frame_comparison.csv"]:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Authoritative CSV Generator] Wrote {len(rows)} rows to {out_path}")

    # Compute and print summary verification table
    print("\n" + "=" * 80)
    print(" 200-FRAME EVALUATION VERIFICATION SUMMARY")
    print("=" * 80)
    for c in fieldnames[2:]:
        vals = [float(r[c]) for r in rows]
        print(f"{c:24s}: Mean={np.mean(vals):.4f}, Median={np.median(vals):.4f}, Std={np.std(vals):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
