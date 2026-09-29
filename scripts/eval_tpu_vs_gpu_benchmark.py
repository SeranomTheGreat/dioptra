#!/usr/bin/env python3
"""
Comprehensive Benchmarking Harness: TPU (Step Latest 4) vs GPU (Step Latest 3)
Evaluates Dioptra-DINO checkpoints on unseen indoor benchmarks:
  1. ScanNet (unseen scene00, Structure sensor, K fx=577.87)
  2. NYU-Depth-v2 (unseen Kinect RGB-D, K fx=518.86)
  3. InteriorNet (unseen synthetic photoreal, K f=600px)
  4. Indoor Suite (8 environments x 50 frames = 400 frames, 0.1-10m)
"""

import os
import sys
import glob
import json
import time
import csv
import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import dioptra_dino
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
import __main__
__main__.DioptraDINOConfig = dioptra_dino.DioptraDINOConfig

# Canonical Intrinsics
K_SCANNET = np.array([
    [577.87, 0.0, 319.5],
    [0.0, 577.87, 239.5],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_NYU_V2 = np.array([
    [518.8579, 0.0, 325.5824],
    [0.0, 518.8579, 253.7362],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_TARTAN_SQ = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 320.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

K_INTERIORNET = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_d: float = 0.1, max_d: float = 10.0) -> Dict[str, float]:
    mask = (gt > min_d) & (gt <= max_d) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if mask.sum() == 0:
        return {
            "abs_rel": float("nan"), "sq_rel": float("nan"), "rmse": float("nan"),
            "rmse_log": float("nan"), "a1": float("nan"), "a2": float("nan"),
            "a3": float("nan"), "scale_ratio": float("nan"), "valid_pixels": 0
        }
    p = pred[mask]
    g = gt[mask]
    thresh = np.maximum(g / np.maximum(p, 1e-6), p / np.maximum(g, 1e-6))
    
    abs_rel = float(np.mean(np.abs(g - p) / g))
    sq_rel = float(np.mean(((g - p) ** 2) / g))
    rmse = float(np.sqrt(np.mean((g - p) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(np.maximum(g, 1e-6)) - np.log(np.maximum(p, 1e-6))) ** 2)))
    a1 = float((thresh < 1.25).mean())
    a2 = float((thresh < 1.25 ** 2).mean())
    a3 = float((thresh < 1.25 ** 3).mean())
    scale_ratio = float(np.median(p) / np.median(g))
    
    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "scale_ratio": scale_ratio,
        "valid_pixels": int(mask.sum())
    }


def load_checkpoint_model(ckpt_path: str, device: str = "mps") -> Tuple[DioptraDINO, Dict[str, Any]]:
    print(f"[Model Loader] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    cfg = ckpt.get("cfg", DioptraDINOConfig(image_size=336, grid_size=24, freeze_backbone=False))
    model = DioptraDINO(cfg)
    
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Model Loader] Loaded successfully. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    model = model.to(device).eval()
    meta = {
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "mean_loss": ckpt.get("mean_loss"),
        "device": device,
    }
    return model, meta


def run_scannet_eval(model: DioptraDINO, device: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    scan_dir = os.path.join(REPO_ROOT, "data/scannet_tiny/scene00")
    color_files = sorted(glob.glob(os.path.join(scan_dir, "color", "*.jpg")))
    assert color_files, f"No ScanNet images found in {scan_dir}"
    
    img_size = 336
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    rows = []
    print(f"\n---> Evaluating ScanNet Unseen (scene00, {len(color_files)} frames) on {device}...")
    
    for i, cp in enumerate(color_files):
        stem = os.path.splitext(os.path.basename(cp))[0]
        dp = os.path.join(scan_dir, "depth", f"{stem}.png")
        assert os.path.exists(dp), f"Missing depth for {cp}"
        
        img_pil = Image.open(cp).convert("RGB")
        W, H = img_pil.size
        img_res = img_pil.resize((img_size, img_size), Image.BILINEAR)
        img_arr = np.array(img_res, dtype=np.float32) / 255.0
        t_img = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
        t_norm = (t_img - mean_t) / std_t
        
        K = K_SCANNET.copy()
        K[0, 0] *= img_size / W
        K[1, 1] *= img_size / H
        K[0, 2] *= img_size / W
        K[1, 2] *= img_size / H
        t_K = torch.from_numpy(K).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(t_norm, t_K, ara_gate=1.0).squeeze().cpu().numpy()
            
        p_full = np.array(Image.fromarray(pred).resize((W, H), Image.BILINEAR), dtype=np.float32)
        gt = np.array(Image.open(dp), dtype=np.float32) / 1000.0  # 16-bit mm -> meters
        
        m = compute_metrics(p_full, gt, min_d=0.1, max_d=10.0)
        m["file"] = os.path.basename(cp)
        rows.append(m)
        print(f"  [{i+1:2d}/{len(color_files)}] {m['file']}: AbsRel={m['abs_rel']:.4f} | RMSE={m['rmse']:.3f}m | δ1={m['a1']*100:.1f}% | Scale={m['scale_ratio']:.3f} | PredMed={np.median(p_full):.2f}m")
        
    with open(os.path.join(out_dir, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale_ratio", "valid_pixels"])
        w.writeheader()
        w.writerows(rows)
        
    avg = {k: float(np.mean([r[k] for r in rows])) for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale_ratio"]}
    avg["n"] = len(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(avg, f, indent=2)
        
    print(f"✓ ScanNet Summary: AbsRel={avg['abs_rel']:.4f}, RMSE={avg['rmse']:.3f}m, δ1={avg['a1']*100:.1f}%, Scale={avg['scale_ratio']:.3f}")
    return avg


def run_indoor_suite_eval(model: DioptraDINO, device: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    suite_dir = os.path.join(REPO_ROOT, "test_samples/indoor_suite")
    
    envs = {
        "tartanair2_americandiner": ("tartan", 0.1, 10.0),
        "tartanair2_archviztinyhouseday": ("tartan", 0.1, 10.0),
        "tartanair2_archviztinyhousenight": ("tartan", 0.1, 10.0),
        "tartanair2_house": ("tartan", 0.1, 10.0),
        "tartanair2_prison": ("tartan", 0.1, 10.0),
        "tartanair2_retrooffice": ("tartan", 0.1, 10.0),
        "tartanair2_supermarket": ("tartan", 0.1, 10.0),
        "nyu_depth_v2": ("nyu", 0.1, 10.0),
    }
    
    img_size = 336
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    all_rows = []
    summary = {}
    print(f"\n---> Evaluating Indoor Suite (8 environments x 50 frames = 400 frames, 0.1-10m) on {device}...")
    
    for env, (kind, min_d, max_d) in envs.items():
        idir = os.path.join(suite_dir, env, "image_left")
        ddir = os.path.join(suite_dir, env, "depth_left")
        imgs = sorted(glob.glob(os.path.join(idir, "*.png")))
        if not imgs:
            print(f"⚠️  Missing images for {env}")
            continue
            
        K_base = K_TARTAN_SQ.copy() if kind == "tartan" else K_NYU_V2.copy()
        t0 = time.time()
        env_rows = []
        
        batch_size = 4
        for i in range(0, len(imgs), batch_size):
            chunk = imgs[i:i + batch_size]
            ib, Kb, shapes, gts, names = [], [], [], [], []
            
            for cp in chunk:
                stem = os.path.basename(cp)[:-4]
                cand = [
                    os.path.join(ddir, stem + "_depth.npy"),
                    os.path.join(ddir, stem + ".npy"),
                    os.path.join(ddir, stem + ".png")
                ]
                dp = next((c for c in cand if os.path.exists(c)), None)
                assert dp, f"Missing depth for {cp}"
                
                pil = Image.open(cp).convert("RGB")
                W, H = pil.size
                img_res = pil.resize((img_size, img_size), Image.BILINEAR)
                arr = np.array(img_res, dtype=np.float32) / 255.0
                t_arr = torch.from_numpy(arr).permute(2, 0, 1)
                
                K = K_base.copy()
                K[0, 0] *= img_size / W
                K[1, 1] *= img_size / H
                K[0, 2] *= img_size / W
                K[1, 2] *= img_size / H
                
                ib.append(t_arr)
                Kb.append(torch.from_numpy(K).float())
                shapes.append((W, H))
                names.append(os.path.basename(cp))
                
                if dp.endswith(".npy"):
                    d = np.load(dp).astype(np.float32)
                else:
                    d = np.array(Image.open(dp), dtype=np.float32)
                    if d.max() > 250:
                        d = d / 1000.0
                gts.append(d)
                
            t_batch = torch.stack(ib).to(device)
            t_batch_norm = (t_batch - mean_t) / std_t
            t_K_batch = torch.stack(Kb).to(device)
            
            with torch.no_grad():
                preds = model(t_batch_norm, t_K_batch, ara_gate=1.0).squeeze(1).cpu().numpy()
                
            for j in range(len(chunk)):
                W, H = shapes[j]
                p_full = np.array(Image.fromarray(preds[j]).resize((W, H), Image.BILINEAR), dtype=np.float32)
                gt = gts[j]
                if gt.shape[:2] != (H, W):
                    gt = np.array(Image.fromarray(gt).resize((W, H), Image.NEAREST), dtype=np.float32)
                    
                m = compute_metrics(p_full, gt, min_d, max_d)
                m["env"] = env
                m["file"] = names[j]
                env_rows.append(m)
                all_rows.append(m)
                
        avg = {k: float(np.mean([r[k] for r in env_rows if not math.isnan(r[k])]))
               for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale_ratio"]}
        avg["n"] = len(env_rows)
        summary[env] = avg
        print(f"  ✓ {env:32s}: n={avg['n']:2d} | AbsRel: {avg['abs_rel']:.4f} | RMSE: {avg['rmse']:.3f}m | δ1: {avg['a1']*100:.1f}% | Scale: {avg['scale_ratio']:.3f} ({time.time()-t0:.1f}s)")
        
    with open(os.path.join(out_dir, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "file", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale_ratio", "valid_pixels"])
        w.writeheader()
        w.writerows(all_rows)
        
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary


def run_interiornet_eval(model: DioptraDINO, device: str, out_dir: str) -> Optional[Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    seq_dir = os.path.join(REPO_ROOT, "data/interiornet/3FO4MMTWI01K_Guest_room/3FO4MMTWI01K_Guest_room")
    if not os.path.exists(seq_dir):
        print(f"⚠️  InteriorNet sequence not found at {seq_dir}, skipping.")
        return None
        
    rgb_files = sorted(glob.glob(os.path.join(seq_dir, "cam0/data/*.png")),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not rgb_files:
        print(f"⚠️  No frames found in {seq_dir}/cam0/data")
        return None
        
    img_size = 336
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    rows = []
    print(f"\n---> Evaluating InteriorNet Guest_room ({len(rgb_files)} frames, synthetic photoreal, 0.1-10m) on {device}...")
    
    for i, cp in enumerate(rgb_files):
        dp = os.path.join(seq_dir, "depth0/data", os.path.basename(cp))
        assert os.path.exists(dp), f"Missing depth: {dp}"
        
        pil = Image.open(cp).convert("RGB")
        W, H = pil.size
        img_res = pil.resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img_res, dtype=np.float32) / 255.0
        t_img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        t_norm = (t_img - mean_t) / std_t
        
        K = K_INTERIORNET.copy()
        K[0, 0] *= img_size / W
        K[1, 1] *= img_size / H
        K[0, 2] *= img_size / W
        K[1, 2] *= img_size / H
        t_K = torch.from_numpy(K).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(t_norm, t_K, ara_gate=1.0).squeeze().cpu().numpy()
            
        p_full = np.array(Image.fromarray(pred).resize((W, H), Image.BILINEAR), dtype=np.float32)
        gt = np.array(Image.open(dp), dtype=np.float32) / 1000.0  # 16-bit mm -> meters
        
        m = compute_metrics(p_full, gt, min_d=0.1, max_d=10.0)
        m["file"] = os.path.basename(cp)
        rows.append(m)
        print(f"  [{i+1:2d}/{len(rgb_files)}] {m['file']}: AbsRel={m['abs_rel']:.4f} | RMSE={m['rmse']:.3f}m | δ1={m['a1']*100:.1f}% | Scale={m['scale_ratio']:.3f}")
        
    with open(os.path.join(out_dir, "per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale_ratio", "valid_pixels"])
        w.writeheader()
        w.writerows(rows)
        
    avg = {k: float(np.mean([r[k] for r in rows])) for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale_ratio"]}
    avg["n"] = len(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(avg, f, indent=2)
        
    print(f"✓ InteriorNet Summary: AbsRel={avg['abs_rel']:.4f}, RMSE={avg['rmse']:.3f}m, δ1={avg['a1']*100:.1f}%, Scale={avg['scale_ratio']:.3f}")
    return avg


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using compute device: {device}")
    
    tpu_zip = os.path.expanduser("~/Downloads/checkpoint_step_latest (4).zip")
    if not os.path.exists(tpu_zip):
        raise FileNotFoundError(f"TPU checkpoint zip not found at {tpu_zip}")
        
    model, meta = load_checkpoint_model(tpu_zip, device=device)
    print(f"Model metadata: {meta}")
    
    # Run evaluations
    scannet_res = run_scannet_eval(
        model=model,
        device=device,
        out_dir=os.path.join(REPO_ROOT, "eval_receipts/scannet_tiny_tpu_ckpt")
    )
    
    suite_res = run_indoor_suite_eval(
        model=model,
        device=device,
        out_dir=os.path.join(REPO_ROOT, "eval_receipts/indoor_suite_tpu_10m")
    )
    
    interior_res = run_interiornet_eval(
        model=model,
        device=device,
        out_dir=os.path.join(REPO_ROOT, "eval_receipts/interiornet_guestroom_tpu")
    )
    
    print("\n" + "=" * 90)
    print("TPU (STEP LATEST 4) EVALUATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
