#!/usr/bin/env python3
"""
benchmark_unidepth_pure_indoor_3000.py
Evaluates UniDepth V2 ViT-Small (CVPR 2024) across the 3,000-frame Pure Photorealistic
True Indoor Benchmark (InteriorNet + Apple Hypersim).

Supports resume capability, streaming JSONL writes, and progress reporting.
"""

import os
import sys
import io
import time
import glob
import math
import json
import zipfile
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
import cv2

# Import benchmark components
from benchmark_pure_indoor_3000 import (
    gather_benchmark_pairs,
    compute_metrics,
    depth_to_surface_normals,
    K_INTERIORNET,
    K_HYPERSIM,
)

sys.path.append(os.path.expanduser("~/.cache/torch/hub/lpiccinelli-eth_UniDepth_main"))


class UniDepthV2Evaluator:
    def __init__(self, device: str = "mps"):
        self.device = device
        print("[UniDepth V2] Loading UniDepth V2 ViT-Small (DINOv2 backbone)...", flush=True)
        self.model = torch.hub.load(
            "lpiccinelli-eth/UniDepth",
            "UniDepth",
            version="v2",
            backbone="vits14",
            pretrained=True,
        )
        self.model = self.model.to(device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        print(f"[UniDepth V2] Ready! ({self.params/1e6:.2f}M params, device={device})", flush=True)

    def predict(self, pil_img: PILImage.Image, K_native: np.ndarray, orig_w: int, orig_h: int) -> Tuple[np.ndarray, float]:
        rgb_t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0).float()
        K_t = torch.from_numpy(K_native.astype(np.float32)).unsqueeze(0).float()

        if self.device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.infer(rgb_t, K_t)
        if self.device == "mps":
            torch.mps.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        depth = out["depth"].squeeze().cpu().numpy()
        del rgb_t, K_t, out
        return depth, dt_ms


def run_unidepth_3000(
    output_dir: str = "outputs_pure_indoor_3000",
    device: str = "mps",
    max_frames: int = 3000,
    stride: int = 1,
):
    os.makedirs(output_dir, exist_ok=True)
    entries_file = os.path.join(output_dir, "unidepth_3000_entries.jsonl")
    results_file = os.path.join(output_dir, "unidepth_3000_results.json")

    print("=" * 80, flush=True)
    print("UNIDEPTH V2 (CVPR 2024) 3,000-FRAME PURE INDOOR BENCHMARK", flush=True)
    print(f"Device:      {device}", flush=True)
    print(f"Target:      {max_frames} frames (InteriorNet + Hypersim)", flush=True)
    print(f"Output File: {entries_file}", flush=True)
    print("=" * 80, flush=True)

    int_pairs, hyp_pairs = gather_benchmark_pairs(total_target=max_frames)
    all_pairs = (int_pairs + hyp_pairs)[:max_frames]
    if stride > 1:
        all_pairs = all_pairs[::stride]
        print(f"Applying stride {stride} -> {len(all_pairs)} frames to evaluate", flush=True)

    # Load already completed indices
    completed = {}
    if os.path.exists(entries_file):
        with open(entries_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        completed[d["idx"]] = d
                    except Exception:
                        pass
        print(f"[Resume Engine] Loaded {len(completed)} already completed frames!", flush=True)

    evaluator = UniDepthV2Evaluator(device=device)
    hyp_zip_obj = None

    t_start = time.time()
    for idx, item in enumerate(all_pairs):
        if idx in completed:
            continue

        dom_type = item["type"]
        scene_name = item["scene"]
        K = item["K"]

        # Load RGB & GT Depth
        if dom_type == "interiornet":
            pil_img = PILImage.open(item["rgb_path"]).convert("RGB")
            orig_w, orig_h = pil_img.size
            rgb_np = np.array(pil_img)
            gt_depth = np.array(PILImage.open(item["depth_path"]), dtype=np.float32) / 1000.0
        else:
            if hyp_zip_obj is None:
                hyp_zip_obj = zipfile.ZipFile(item["zip_file"], "r")
            img_bytes = hyp_zip_obj.read(item["rgb_path"])
            dep_bytes = hyp_zip_obj.read(item["depth_path"])
            pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            orig_w, orig_h = pil_img.size
            rgb_np = np.array(pil_img)

            import h5py
            with h5py.File(io.BytesIO(dep_bytes), "r") as hf:
                raw_d = np.array(hf["dataset"][:], dtype=np.float32)
            dh, dw = raw_d.shape[:2]
            fx_h = K[0, 0] * (float(dw) / 1024.0)
            fy_h = K[1, 1] * (float(dh) / 768.0)
            cx_h = K[0, 2] * (float(dw) / 1024.0)
            cy_h = K[1, 2] * (float(dh) / 768.0)
            gy, gx = np.indices((dh, dw), dtype=np.float32)
            ray_scale = np.sqrt(1.0 + ((gx - cx_h) / fx_h) ** 2 + ((gy - cy_h) / fy_h) ** 2)
            gt_depth = raw_d / ray_scale

        # Predict UniDepth V2
        try:
            pred_uni, dt_uni = evaluator.predict(pil_img, K, orig_w, orig_h)
            m_uni = compute_metrics(pred_uni, gt_depth, K=K, min_depth=0.1, max_depth=10.0)
        except Exception as e:
            print(f"Error predicting frame {idx}: {e}", flush=True)
            continue

        if m_uni:
            entry = {
                "idx": idx,
                "type": dom_type,
                "scene": scene_name,
                "unidepth": m_uni,
                "latency_ms": dt_uni,
            }
            completed[idx] = entry
            with open(entries_file, "a") as f_out:
                f_out.write(json.dumps(entry) + "\n")

        # Periodic log
        if (idx + 1) % 25 == 0 or (idx + 1) == len(all_pairs):
            cur_entries = list(completed.values())
            cur_ar = np.mean([x["unidepth"]["abs_rel"] for x in cur_entries])
            cur_d1 = np.mean([x["unidepth"]["delta1"] for x in cur_entries]) * 100
            cur_scale = np.mean([x["unidepth"]["scale_ratio"] for x in cur_entries])
            elapsed = time.time() - t_start
            rate = len(completed) / max(1.0, elapsed)
            rem = (len(all_pairs) - len(completed)) / max(0.01, rate)
            print(
                f"[{idx+1:4d}/{len(all_pairs):4d}] UniDepth AbsRel: {cur_ar:.4f} | "
                f"d1: {cur_d1:.1f}% | Scale: {cur_scale:.3f} | Lat: {dt_uni:.1f}ms | "
                f"ETA: {rem/60.0:.1f} min",
                flush=True,
            )

        # Periodic garbage collection
        if (idx + 1) % 50 == 0:
            import gc
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

    if hyp_zip_obj is not None:
        hyp_zip_obj.close()

    # Aggregate final metrics
    entries = list(completed.values())
    if not entries:
        print("No entries evaluated!", flush=True)
        return

    def aggregate(subset):
        if not subset:
            return {}
        m = [x["unidepth"] for x in subset]
        return {
            "frames": len(subset),
            "abs_rel": float(np.mean([x["abs_rel"] for x in m])),
            "sq_rel": float(np.mean([x["sq_rel"] for x in m])),
            "rmse": float(np.mean([x["rmse"] for x in m])),
            "mae": float(np.mean([x["mae"] for x in m])),
            "delta1": float(np.mean([x["delta1"] for x in m])),
            "delta2": float(np.mean([x["delta2"] for x in m])),
            "delta3": float(np.mean([x["delta3"] for x in m])),
            "scale_ratio": float(np.mean([x["scale_ratio"] for x in m])),
            "abs_rel_aligned": float(np.mean([x["abs_rel_aligned"] for x in m])),
            "rmse_aligned": float(np.mean([x["rmse_aligned"] for x in m])),
            "delta1_aligned": float(np.mean([x["delta1_aligned"] for x in m])),
            "normal_mae": float(np.nanmean([x["normal_mae"] for x in m])),
            "normal_acc11": float(np.nanmean([x["normal_acc11"] for x in m])),
            "avg_latency_ms": float(np.mean([x["latency_ms"] for x in subset])),
        }

    overall = aggregate(entries)
    int_agg = aggregate([x for x in entries if x["type"] == "interiornet"])
    hyp_agg = aggregate([x for x in entries if x["type"] == "hypersim"])

    results = {
        "metadata": {
            "model": "UniDepth V2 ViT-Small (CVPR 2024)",
            "checkpoint": "lpiccinelli-eth/UniDepth:v2:vits14",
            "parameters": "34.18M",
            "device": device,
            "total_frames_evaluated": len(entries),
        },
        "overall": overall,
        "interiornet": int_agg,
        "hypersim": hyp_agg,
    }

    with open(results_file, "w") as f_res:
        json.dump(results, f_res, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("FINAL UNIDEPTH V2 3,000-FRAME BENCHMARK SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Overall ({len(entries)} frames): AbsRel={overall.get('abs_rel', 0):.4f}, RMSE={overall.get('rmse', 0):.3f}m, d1={overall.get('delta1', 0)*100:.1f}%, Scale={overall.get('scale_ratio', 0):.3f}")
    if int_agg:
        print(f"InteriorNet ({int_agg.get('frames', 0)} frames): AbsRel={int_agg.get('abs_rel', 0):.4f}, RMSE={int_agg.get('rmse', 0):.3f}m, d1={int_agg.get('delta1', 0)*100:.1f}%, Scale={int_agg.get('scale_ratio', 0):.3f}")
    if hyp_agg:
        print(f"Hypersim ({hyp_agg.get('frames', 0)} frames): AbsRel={hyp_agg.get('abs_rel', 0):.4f}, RMSE={hyp_agg.get('rmse', 0):.3f}m, d1={hyp_agg.get('delta1', 0)*100:.1f}%, Scale={hyp_agg.get('scale_ratio', 0):.3f}")
    print("=" * 80, flush=True)
    print(f"Saved to: {results_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frames", type=int, default=3000)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    run_unidepth_3000(device=args.device, max_frames=args.max_frames, stride=args.stride)
