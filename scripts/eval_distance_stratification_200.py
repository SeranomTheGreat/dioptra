"""
Evaluate Operational Distance Stratification across all 200 frames of the
continuous benchmark trajectory (abandonedfactory/Easy/P010) using the 40-epoch
Dioptra-DINO checkpoint (outputs/dioptra_dino_best.pt).

Computes AbsRel, RMSE, and delta < 1.25 for:
- Near Navigation Zone: [0.1, 5.0] m
- Mid Object Interaction: [5.0, 15.0] m
- Far Architectural Zone: [15.0, 30.0] m
- Deep Background: [30.0, 80.0] m
"""

import os
import sys
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
import __main__
setattr(__main__, "DioptraDINOConfig", DioptraDINOConfig)


def evaluate_distance_stratification(ckpt_path: str = "outputs/dioptra_dino_best.pt"):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint: {ckpt_path} on {device}...")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", DioptraDINOConfig())
    model = DioptraDINO(cfg).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()

    img_dir = "test_samples/unseen_200_abandonedfactory/image_left"
    depth_dir = "test_samples/unseen_200_abandonedfactory/depth_left"

    pngs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not pngs:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")
    print(f"Found {len(pngs)} images to evaluate.")

    bands = [
        ("Near Navigation Zone", 0.1, 5.0),
        ("Mid Object Interaction", 5.0, 15.0),
        ("Far Architectural Zone", 15.0, 30.0),
        ("Deep Background", 30.0, 80.0),
    ]

    band_records = {b[0]: {"abs_rel": [], "rmse": [], "d1": []} for b in bands}

    for p in pngs:
        fname = os.path.basename(p)
        stem = fname.replace("_left.png", "")
        d_path = os.path.join(depth_dir, f"{stem}_left_depth.npy")
        if not os.path.exists(d_path):
            continue

        raw_img = Image.open(p).convert("RGB")
        W_orig, H_orig = raw_img.size
        min_side = min(H_orig, W_orig)
        top = (H_orig - min_side) // 2
        left = (W_orig - min_side) // 2
        raw_cropped = raw_img.crop((left, top, left + min_side, top + min_side))
        img_resized = raw_cropped.resize((224, 224), Image.Resampling.BILINEAR)

        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        input_t = ((img_t - mean) / std).unsqueeze(0).to(device)

        gt_arr = np.load(d_path).astype(np.float32)
        gt_t = torch.from_numpy(gt_arr)
        gt_cropped = gt_t[top:top + min_side, left:left + min_side]
        gt_depth = F.interpolate(gt_cropped.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="nearest").squeeze().numpy()

        fx_native = 320.0 * (224.0 / float(min_side))
        K = torch.tensor([[[fx_native, 0.0, 112.0], [0.0, fx_native, 112.0], [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

        with torch.no_grad():
            pred = model(input_t, K).squeeze().cpu().numpy()

        for b_name, d_min, d_max in bands:
            m = (gt_depth >= d_min) & (gt_depth < d_max) & np.isfinite(gt_depth) & (pred > 0.05) & np.isfinite(pred)
            if m.sum() > 10:
                p_sub = pred[m]
                g_sub = gt_depth[m]
                ar = float((np.abs(g_sub - p_sub) / g_sub).mean())
                rmse = float(np.sqrt(((g_sub - p_sub) ** 2).mean()))
                thresh = np.maximum(g_sub / p_sub, p_sub / g_sub)
                d1 = float((thresh < 1.25).mean()) * 100.0
                band_records[b_name]["abs_rel"].append(ar)
                band_records[b_name]["rmse"].append(rmse)
                band_records[b_name]["d1"].append(d1)

    results = {}
    print("=" * 80)
    print("OPERATIONAL DISTANCE STRATIFICATION ON 200 IMAGES (DIOPTRA-DINO EPOCH 40):")
    print(f"{'Distance Bracket':<25} | {'AbsRel':<10} | {'RMSE (m)':<12} | {'delta < 1.25':<12} | {'Valid Frames'}")
    print("-" * 80)
    for b_name, d_min, d_max in bands:
        ars = band_records[b_name]["abs_rel"]
        rmses = band_records[b_name]["rmse"]
        d1s = band_records[b_name]["d1"]
        mean_ar = float(np.mean(ars))
        mean_rmse = float(np.mean(rmses))
        mean_d1 = float(np.mean(d1s))
        count = len(ars)
        results[b_name] = {
            "span": [d_min, d_max],
            "abs_rel": round(mean_ar, 4),
            "rmse": round(mean_rmse, 3),
            "d1": round(mean_d1, 2),
            "frames": count,
        }
        print(f"{b_name:<25} | {mean_ar:<10.4f} | {mean_rmse:<10.3f} m | {mean_d1:<10.2f} % | {count}")
    print("=" * 80)

    out_json = "outputs/distance_stratification_200_verified.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved verified results to: {out_json}")
    return results


if __name__ == "__main__":
    evaluate_distance_stratification()
