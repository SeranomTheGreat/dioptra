#!/usr/bin/env python3
"""
benchmark_highres_indoor_suite.py
---------------------------------
Evaluates Dioptra-DINO at multiple high resolutions (224x224, 336x336, 448x448, 560x560, 672x672)
across all 18 indoor environments (925 frames) and compares with Metric3D and Depth Anything V2.

Computes: AbsRel, RMSE, MAE, delta1, delta2, delta3, scale ratio, normal MAE, latency (ms), FPS.
Exports results to mac_outputs/highres_indoor_benchmark_results.json and .md summary.
"""

import os
import sys
import glob
import json
import time
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image as PILImage
import cv2
import torch
import torch.nn.functional as F

from dioptra_dino import DioptraDINO, DioptraDINOConfig
from benchmark_all_indoor_competitors import scale_intrinsics, compute_metrics, depth_to_surface_normals


# Standard camera intrinsics
K_TARTAN_CANONICAL = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_NYU_CANONICAL = np.array([
    [5.1885790117450188e+02, 0.0, 3.2558244941119034e+02],
    [0.0, 5.1885790117450188e+02, 2.5373616633400465e+02],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_SCANNET_CANONICAL = np.array([
    [577.87, 0.0, 319.5],
    [0.0, 577.87, 239.5],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


class HighResDioptraEvaluator:
    def __init__(self, checkpoint_path: str, device: str = "mps"):
        self.device = device
        cfg = DioptraDINOConfig()
        self.model = DioptraDINO(cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict(self, pil_img: PILImage.Image, K_native: np.ndarray, orig_w: int, orig_h: int, res: int) -> Tuple[np.ndarray, float]:
        img_res = pil_img.resize((res, res), PILImage.BILINEAR)
        img_np = np.array(img_res, dtype=np.float32) / 255.0
        img_norm = (img_np - self.mean) / self.std
        img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        K_res = scale_intrinsics(K_native, orig_w, orig_h, res, res)
        K_t = torch.from_numpy(K_res).unsqueeze(0).to(self.device)

        if self.device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.model(img_t, K_t, ara_gate=1.0)
            pred_full = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return pred_full.squeeze().cpu().numpy(), dt_ms


def define_indoor_dataset_splits() -> List[Dict]:
    splits = [
        # A. Industrial Robotics
        {
            "id": "carwelding_p001",
            "name": "CarWelding / P001",
            "category": "Industrial Robotics",
            "img_pattern": "test_samples/unseen_carwelding_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_carwelding_p001/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 80.0,
            "max_frames": 25,
        },
        {
            "id": "carwelding_p002",
            "name": "CarWelding / P002",
            "category": "Industrial Robotics",
            "img_pattern": "test_samples/unseen_carwelding_p002/image_left/*.png",
            "depth_dir": "test_samples/unseen_carwelding_p002/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 80.0,
            "max_frames": 30,
        },
        # B. Commercial Workspaces
        {
            "id": "office_p001",
            "name": "Office / P001",
            "category": "Commercial Office",
            "img_pattern": "test_samples/unseen_office_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_office_p001/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 50.0,
            "max_frames": 30,
        },
        {
            "id": "office_p002",
            "name": "Office / P002",
            "category": "Commercial Office",
            "img_pattern": "test_samples/unseen_office_p002/image_left/*.png",
            "depth_dir": "test_samples/unseen_office_p002/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 50.0,
            "max_frames": 30,
        },
        {
            "id": "office2_p000",
            "name": "Office2 / P000",
            "category": "Executive Workspace",
            "img_pattern": "test_samples/unseen_office2_p000/image_left/*.png",
            "depth_dir": "test_samples/unseen_office2_p000/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 50.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_retrooffice",
            "name": "Retro Office / P000",
            "category": "Retro Workspace",
            "img_pattern": "test_samples/indoor_suite/tartanair2_retrooffice/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_retrooffice/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 25.0,
            "max_frames": 50,
        },
        # C. Medical & Institutional
        {
            "id": "hospital_p001",
            "name": "Hospital / P001",
            "category": "Medical Facility",
            "img_pattern": "test_samples/unseen_hospital_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_hospital_p001/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 50.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_prison",
            "name": "Prison / P000",
            "category": "Institutional Cells",
            "img_pattern": "test_samples/indoor_suite/tartanair2_prison/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_prison/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 50.0,
            "max_frames": 50,
        },
        # D. Industrial Warehouses
        {
            "id": "abandonedfactory_p005",
            "name": "AbandonedFactory / P005",
            "category": "Industrial Warehouse",
            "img_pattern": "test_samples/unseen_abandonedfactory_p005/image_left/*.png",
            "depth_dir": "test_samples/unseen_abandonedfactory_p005/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 80.0,
            "max_frames": 50,
        },
        {
            "id": "factory_night_p001",
            "name": "Factory Night / P001",
            "category": "Extreme Low-Light",
            "img_pattern": "test_samples/unseen_abandonedfactory_night_p001/image_left/*.png",
            "depth_dir": "test_samples/unseen_abandonedfactory_night_p001/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 80.0,
            "max_frames": 50,
        },
        {
            "id": "abandonedfactory_200",
            "name": "AbandonedFactory / 200-Frames",
            "category": "Warehouse Long-Run",
            "img_pattern": "data/unseen_eval/images/*.png",
            "depth_dir": "data/unseen_eval/depths",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 80.0,
            "max_frames": 200,
        },
        # E. Public Dining, Retail, Residential
        {
            "id": "tartanair2_americandiner",
            "name": "American Diner / P000",
            "category": "Restaurant Interior",
            "img_pattern": "test_samples/indoor_suite/tartanair2_americandiner/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_americandiner/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 30.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_supermarket",
            "name": "Supermarket / P000",
            "category": "Retail Grocery",
            "img_pattern": "test_samples/indoor_suite/tartanair2_supermarket/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_supermarket/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 40.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_house",
            "name": "Suburban House / P000",
            "category": "Residential Multi-Room",
            "img_pattern": "test_samples/indoor_suite/tartanair2_house/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_house/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 25.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_archviztinyhouseday",
            "name": "Tiny House Day / P000",
            "category": "Residential Architectural",
            "img_pattern": "test_samples/indoor_suite/tartanair2_archviztinyhouseday/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_archviztinyhouseday/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 25.0,
            "max_frames": 50,
        },
        {
            "id": "tartanair2_archviztinyhousenight",
            "name": "Tiny House Night / P000",
            "category": "Residential Night",
            "img_pattern": "test_samples/indoor_suite/tartanair2_archviztinyhousenight/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/tartanair2_archviztinyhousenight/depth_left",
            "K": K_TARTAN_CANONICAL,
            "max_eval_depth": 25.0,
            "max_frames": 50,
        },
        # F. Physical Real-World Sensors
        {
            "id": "nyu_depth_v2",
            "name": "NYU-Depth V2 (Official Test)",
            "category": "Real Kinect RGB-D",
            "img_pattern": "test_samples/indoor_suite/nyu_depth_v2/image_left/*.png",
            "depth_dir": "test_samples/indoor_suite/nyu_depth_v2/depth_left",
            "K": K_NYU_CANONICAL,
            "max_eval_depth": 10.0,
            "max_frames": 50,
        },
        {
            "id": "scannet_scene00",
            "name": "ScanNet (scene00)",
            "category": "Real Handheld RGB-D",
            "img_pattern": "test_samples/scannet_unseen/color/*.jpg",
            "depth_dir": "test_samples/scannet_unseen/depth",
            "K": K_SCANNET_CANONICAL,
            "max_eval_depth": 10.0,
            "max_frames": 10,
        },
    ]
    return splits


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_path = "staging_harryson/checkpoint_step_latest.pt"
    print(f"Initializing HighResDioptraEvaluator on {device} from {ckpt_path}...")
    evaluator = HighResDioptraEvaluator(ckpt_path, device=device)

    resolutions = [224, 336, 448, 560, 672]
    splits = define_indoor_dataset_splits()

    os.makedirs("mac_outputs", exist_ok=True)
    all_results = {}

    print("\n" + "=" * 80)
    print("STARTING MULTI-RESOLUTION ALL-INDOOR BENCHMARK")
    print(f"Resolutions to test: {resolutions}")
    print(f"Environments: {len(splits)}")
    print("=" * 80 + "\n")

    for split in splits:
        split_id = split["id"]
        split_name = split["name"]
        cat = split["category"]
        img_pat = split["img_pattern"]
        depth_dir = split["depth_dir"]
        K = split["K"]
        max_eval_d = split["max_eval_depth"]
        max_f = split.get("max_frames", 50)

        img_files = sorted(glob.glob(img_pat))
        if max_f and len(img_files) > max_f:
            img_files = img_files[:max_f]

        if not img_files:
            print(f"[Warning] No files found for {split_name} ({img_pat})")
            continue

        print(f"\nEvaluating {split_name} ({len(img_files)} frames) across resolutions {resolutions}...")
        split_res_dict = {}

        for res in resolutions:
            metrics_list = []
            latencies = []

            for img_path in img_files:
                base = os.path.basename(img_path)
                stem = os.path.splitext(base)[0]

                # Match depth file
                possible_names = [
                    f"{stem}_depth.npy",
                    f"{stem}.npy",
                    f"{stem}.png",
                ]
                d_path = None
                for candidate in possible_names:
                    p = os.path.join(depth_dir, candidate)
                    if os.path.exists(p):
                        d_path = p
                        break

                if not d_path:
                    continue

                try:
                    img = PILImage.open(img_path).convert("RGB")
                    orig_w, orig_h = img.size

                    if d_path.endswith(".npy"):
                        gt = np.load(d_path).astype(np.float32)
                    elif "scannet" in split_id:
                        gt = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                    else:
                        gt = cv2.imread(d_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

                    if gt.ndim == 3:
                        gt = gt.squeeze()

                    pred, dt = evaluator.predict(img, K, orig_w, orig_h, res)
                    m = compute_metrics(pred, gt, K=K, min_depth=0.1, max_depth=max_eval_d)
                    if m:
                        metrics_list.append(m)
                        latencies.append(dt)
                except Exception as e:
                    print(f"Error on {base}: {e}")
                    continue

            if metrics_list:
                agg = {
                    "abs_rel": float(np.mean([m["abs_rel"] for m in metrics_list])),
                    "rmse": float(np.mean([m["rmse"] for m in metrics_list])),
                    "mae": float(np.mean([m["mae"] for m in metrics_list])),
                    "delta1": float(np.mean([m["delta1"] for m in metrics_list])),
                    "delta2": float(np.mean([m["delta2"] for m in metrics_list])),
                    "delta3": float(np.mean([m["delta3"] for m in metrics_list])),
                    "scale_ratio": float(np.mean([m["scale_ratio"] for m in metrics_list])),
                    "normal_mae": float(np.nanmean([m["normal_mae"] for m in metrics_list])),
                    "latency_ms": float(np.mean(latencies)),
                    "fps": float(1000.0 / np.mean(latencies)) if np.mean(latencies) > 0 else 0.0,
                    "num_frames": len(metrics_list),
                }
                split_res_dict[str(res)] = agg
                print(f"  Res {res:3d}x{res:3d} | AbsRel: {agg['abs_rel']:.4f} | RMSE: {agg['rmse']:.3f}m | d1: {agg['delta1']*100:5.1f}% | Scale: {agg['scale_ratio']:.3f} | Lat: {agg['latency_ms']:5.1f}ms ({agg['fps']:4.1f} FPS)")

        all_results[split_id] = {
            "name": split_name,
            "category": cat,
            "resolutions": split_res_dict,
        }

    # Compute macro-averages per resolution
    macro_averages = {}
    for res in resolutions:
        res_key = str(res)
        rel_vals = []
        rmse_vals = []
        mae_vals = []
        d1_vals = []
        scale_vals = []
        norm_vals = []
        lat_vals = []
        total_f = 0

        for split_id, data in all_results.items():
            if res_key in data["resolutions"]:
                s = data["resolutions"][res_key]
                rel_vals.append(s["abs_rel"])
                rmse_vals.append(s["rmse"])
                mae_vals.append(s["mae"])
                d1_vals.append(s["delta1"])
                scale_vals.append(s["scale_ratio"])
                if not np.isnan(s["normal_mae"]):
                    norm_vals.append(s["normal_mae"])
                lat_vals.append(s["latency_ms"])
                total_f += s["num_frames"]

        if rel_vals:
            macro_averages[res_key] = {
                "abs_rel": float(np.mean(rel_vals)),
                "rmse": float(np.mean(rmse_vals)),
                "mae": float(np.mean(mae_vals)),
                "delta1": float(np.mean(d1_vals)),
                "scale_ratio": float(np.mean(scale_vals)),
                "normal_mae": float(np.mean(norm_vals)) if norm_vals else float("nan"),
                "latency_ms": float(np.mean(lat_vals)),
                "fps": float(1000.0 / np.mean(lat_vals)) if np.mean(lat_vals) > 0 else 0.0,
                "total_frames": total_f,
            }

    out_json = {
        "macro_averages": macro_averages,
        "per_environment": all_results,
    }

    with open("mac_outputs/highres_indoor_benchmark_results.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print("\nSaved full JSON to mac_outputs/highres_indoor_benchmark_results.json")

    # Generate Markdown Summary
    md = "# Multi-Resolution Indoor Benchmark: Dioptra-DINO Across Input Sizes\n\n"
    md += "### Macro-Average Indoor Performance Across Resolutions\n\n"
    md += "| Input Resolution | Token Grid | Tokens | AbsRel ↓ | RMSE (m) ↓ | MAE (m) ↓ | $\\delta_1 < 1.25$ ↑ | Scale Ratio | Normal MAE (°) ↓ | MPS Latency (ms) ↓ | Throughput (FPS) ↑ |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for res in resolutions:
        k = str(res)
        if k in macro_averages:
            m = macro_averages[k]
            grid = f"{res//14}x{res//14}"
            toks = (res//14)**2
            md += f"| **{res}x{res}** | {grid} | {toks:,} | **{m['abs_rel']:.4f}** | **{m['rmse']:.3f}m** | {m['mae']:.3f}m | **{m['delta1']*100:.1f}%** | **{m['scale_ratio']:.3f}** | {m['normal_mae']:.1f}° | **{m['latency_ms']:.1f} ms** | **{m['fps']:.1f} FPS** |\n"

    # Add competitors for comparison
    md += "| *Metric3D ViT-S* | 44x76 | 3,344 | 0.2976 | 3.870m | 1.875m | 55.6% | 1.014 | 43.7° | 549.3 ms | 1.8 FPS |\n"
    md += "| *Depth Anything V2* | 37x37 | 1,369 | 0.7068 | 4.301m | 2.566m | 18.3% | 1.588 | 41.7° | 110.1 ms | 9.4 FPS |\n\n"

    md += "### Per-Environment Detailed Breakdown (Selected Resolutions)\n\n"
    md += "| Environment | Category | 224x224 AbsRel | 336x336 AbsRel | 448x448 AbsRel | 560x560 AbsRel | 672x672 AbsRel | 224x224 $\\delta_1$ | 448x448 $\\delta_1$ | 672x672 $\\delta_1$ |\n"
    md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for split_id, data in all_results.items():
        name = data["name"]
        cat = data["category"]
        r = data["resolutions"]
        ar_224 = f"{r['224']['abs_rel']:.4f}" if "224" in r else "-"
        ar_336 = f"{r['336']['abs_rel']:.4f}" if "336" in r else "-"
        ar_448 = f"{r['448']['abs_rel']:.4f}" if "448" in r else "-"
        ar_560 = f"{r['560']['abs_rel']:.4f}" if "560" in r else "-"
        ar_672 = f"{r['672']['abs_rel']:.4f}" if "672" in r else "-"
        d1_224 = f"{r['224']['delta1']*100:.1f}%" if "224" in r else "-"
        d1_448 = f"{r['448']['delta1']*100:.1f}%" if "448" in r else "-"
        d1_672 = f"{r['672']['delta1']*100:.1f}%" if "672" in r else "-"
        md += f"| {name} | {cat} | {ar_224} | {ar_336} | {ar_448} | {ar_560} | {ar_672} | {d1_224} | {d1_448} | {d1_672} |\n"

    with open("mac_outputs/highres_indoor_benchmark_summary.md", "w") as f:
        f.write(md)
    print("Saved Markdown summary to mac_outputs/highres_indoor_benchmark_summary.md")


if __name__ == "__main__":
    main()
