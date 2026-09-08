import os
import time
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import tesseract_v1_patched as tess

def evaluate_all():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    # 1. Load Model
    cfg = tess.TesseractConfig()
    model = tess.Tesseract(cfg.model).to(device)
    ckpt_path = 'outputs/checkpoint.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'outputs/checkpoint_epoch24.pt'
    
    print(f"Loading checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    model_state = state.get("tesseract", state)
    if any(k.startswith("module.") for k in model_state):
        model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=False)
    model.eval()
    
    os.makedirs('test_outputs', exist_ok=True)
    os.makedirs('paper/figures', exist_ok=True)
    
    # TartanAir camera intrinsics (90 deg FOV, 640x480 native)
    # fx = 320.0, fy = 320.0, cx = 320.0, cy = 240.0
    # Rescaled to 224x224:
    fx_224 = 320.0 * (224.0 / 640.0)
    fy_224 = 320.0 * (224.0 / 480.0)
    cx_224 = 320.0 * (224.0 / 640.0)
    cy_224 = 240.0 * (224.0 / 480.0)
    K_224 = torch.tensor([[[fx_224, 0.0, cx_224],
                           [0.0, fy_224, cy_224],
                           [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)
    
    mean = torch.tensor(tess.IMAGENET_MEAN).view(3, 1, 1).to(device)
    std = torch.tensor(tess.IMAGENET_STD).view(3, 1, 1).to(device)
    
    env_results = {}
    
    sample_dirs = sorted(glob.glob('test_samples/*'))
    print(f"\nFound {len(sample_dirs)} environment folders to test.")
    
    for edir in sample_dirs:
        env_name = os.path.basename(edir)
        img_files = sorted(glob.glob(os.path.join(edir, '*_left.png')))
        if not img_files:
            continue
            
        print(f"\n=======================================================")
        print(f"  Evaluating Environment: {env_name} ({len(img_files)} samples)")
        print(f"=======================================================")
        
        env_results[env_name] = []
        
        for img_path in img_files:
            stem = os.path.basename(img_path).replace('.png', '')
            depth_path = os.path.join(edir, f"{stem}_depth.npy")
            if not os.path.exists(depth_path):
                print(f"Skipping {stem}: depth .npy not found")
                continue
                
            raw_rgb = Image.open(img_path).convert('RGB')
            gt_depth = np.load(depth_path).astype(np.float32) # (480, 640)
            
            # Preprocess to 224x224
            img_resized = raw_rgb.resize((224, 224), Image.Resampling.BILINEAR)
            img_t = TF.to_tensor(img_resized).to(device)
            input_t = ((img_t - mean) / std).unsqueeze(0)
            
            # Forward pass
            t0 = time.perf_counter()
            with torch.no_grad():
                pred_depth_224, points, _ = model(input_t, K_224, irer_gate=1.0)
            dt_ms = (time.perf_counter() - t0) * 1000
            
            # Upsample prediction back to native (480, 640)
            pred_full = torch.nn.functional.interpolate(
                pred_depth_224, size=(480, 640), mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
            
            # Valid mask: 0.2m to 80.0m
            valid_mask = (gt_depth >= 0.2) & (gt_depth <= 80.0) & np.isfinite(gt_depth) & (gt_depth > 0)
            if valid_mask.sum() < 100:
                print(f"Skipping {stem}: too few valid pixels")
                continue
                
            gt_v = gt_depth[valid_mask]
            pred_v = pred_full[valid_mask]
            
            # Scale alignment via median
            scale = np.median(gt_v) / (np.median(pred_v) + 1e-8)
            pred_aligned = pred_v * scale
            
            # Aligned Metrics
            abs_rel = np.mean(np.abs(pred_aligned - gt_v) / gt_v)
            rmse = np.sqrt(np.mean((pred_aligned - gt_v) ** 2))
            log10 = np.mean(np.abs(np.log10(np.clip(pred_aligned, 1e-3, None)) - np.log10(np.clip(gt_v, 1e-3, None))))
            
            ratio = np.maximum(pred_aligned / gt_v, gt_v / pred_aligned)
            d1 = np.mean(ratio < 1.25) * 100.0
            d2 = np.mean(ratio < 1.25 ** 2) * 100.0
            d3 = np.mean(ratio < 1.25 ** 3) * 100.0
            
            # Unaligned Raw Metric
            m_abs_rel = np.mean(np.abs(pred_v - gt_v) / gt_v)
            m_rmse = np.sqrt(np.mean((pred_v - gt_v) ** 2))
            m_ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
            m_d1 = np.mean(m_ratio < 1.25) * 100.0
            
            res = {
                'frame': stem,
                'time_ms': dt_ms,
                'abs_rel': abs_rel,
                'rmse': rmse,
                'log10': log10,
                'd1': d1,
                'd2': d2,
                'd3': d3,
                'metric_abs_rel': m_abs_rel,
                'metric_rmse': m_rmse,
                'metric_d1': m_d1
            }
            env_results[env_name].append(res)
            
            print(f"[{env_name}] {stem} | Latency: {dt_ms:.1f}ms | AbsRel: {abs_rel:.4f} | δ₁: {d1:.1f}% | δ₂: {d2:.1f}% | Metric AbsRel: {m_abs_rel:.4f} | Metric RMSE: {m_rmse:.2f}m")
            
            # Generate 4-panel publication visualization
            pred_aligned_full = pred_full * scale
            error_map = np.abs(np.log(np.clip(pred_aligned_full, 0.1, 80.0)) - np.log(np.clip(gt_depth, 0.1, 80.0)))
            error_map[~valid_mask] = 0.0
            
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=200)
            
            axes[0].imshow(raw_rgb)
            axes[0].set_title(f"Input RGB ({env_name})", fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            im1 = axes[1].imshow(gt_depth, cmap='viridis', vmin=0.5, vmax=min(40.0, np.percentile(gt_v, 98)))
            axes[1].set_title(f"Ground Truth Depth", fontsize=12, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Metres')
            
            im2 = axes[2].imshow(pred_aligned_full, cmap='viridis', vmin=0.5, vmax=min(40.0, np.percentile(gt_v, 98)))
            axes[2].set_title(f"Dioptra Pred (AbsRel={abs_rel:.3f})", fontsize=12, fontweight='bold')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Metres')
            
            im3 = axes[3].imshow(error_map, cmap='magma', vmin=0.0, vmax=1.0)
            axes[3].set_title(f"Log Error Heatmap (δ₁={d1:.1f}%)", fontsize=12, fontweight='bold')
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label='|log(pred) - log(gt)|')
            
            plt.tight_layout()
            out_png = f"test_outputs/{env_name}_{stem}_benchmark.png"
            plt.savefig(out_png, bbox_inches='tight')
            plt.close()
            
            # Export 3D Mesh
            mesh_out = f"test_outputs/{env_name}_{stem}_mesh.ply"
            depth_224_t = torch.from_numpy(pred_depth_224.squeeze().cpu().numpy()).to(device)
            img_224_t = img_t.cpu()
            tess.export_mesh_ply(img_224_t, depth_224_t, K_224[0], mesh_out)

    print("\n" + "="*95)
    print("  SUMMARY: DIVERSE ENVIRONMENT EVALUATION RESULTS ON APPLE SILICON M3")
    print("="*95)
    held_out_items = []
    
    for env_name, items in env_results.items():
        if not items:
            continue
        if env_name != 'hospital':
            held_out_items.extend(items)
            
        mean_absrel = np.mean([x['abs_rel'] for x in items])
        mean_rmse = np.mean([x['rmse'] for x in items])
        mean_d1 = np.mean([x['d1'] for x in items])
        mean_d2 = np.mean([x['d2'] for x in items])
        mean_d3 = np.mean([x['d3'] for x in items])
        mean_m_absrel = np.mean([x['metric_abs_rel'] for x in items])
        mean_m_rmse = np.mean([x['metric_rmse'] for x in items])
        mean_m_d1 = np.mean([x['metric_d1'] for x in items])
        mean_ms = np.mean([x['time_ms'] for x in items])
        
        # Best frame by AbsRel
        best_item = min(items, key=lambda x: x['abs_rel'])
        
        print(f"{env_name:<26} (N={len(items):<2}) | Mean AbsRel: {mean_absrel:.4f} | RMSE: {mean_rmse:.2f}m | δ₁: {mean_d1:.1f}% | δ₂: {mean_d2:.1f}% | Metric AbsRel: {mean_m_absrel:.4f} | Metric RMSE: {mean_m_rmse:.2f}m | {mean_ms:.1f}ms")
        print(f"   ↳ Best Frame [{best_item['frame']}]: AbsRel: {best_item['abs_rel']:.4f} | RMSE: {best_item['rmse']:.2f}m | δ₁: {best_item['d1']:.1f}% | δ₂: {best_item['d2']:.1f}% | Metric AbsRel: {best_item['metric_abs_rel']:.4f} | Metric RMSE: {best_item['metric_rmse']:.2f}m")

    if held_out_items:
        print("\n" + "-"*95)
        ho_absrel = np.mean([x['abs_rel'] for x in held_out_items])
        ho_rmse = np.mean([x['rmse'] for x in held_out_items])
        ho_d1 = np.mean([x['d1'] for x in held_out_items])
        ho_d2 = np.mean([x['d2'] for x in held_out_items])
        ho_d3 = np.mean([x['d3'] for x in held_out_items])
        ho_m_absrel = np.mean([x['metric_abs_rel'] for x in held_out_items])
        ho_m_rmse = np.mean([x['metric_rmse'] for x in held_out_items])
        ho_m_d1 = np.mean([x['metric_d1'] for x in held_out_items])
        ho_ms = np.mean([x['time_ms'] for x in held_out_items])
        print(f"HELD-OUT TOTAL (N={len(held_out_items):<2})      | Mean AbsRel: {ho_absrel:.4f} | RMSE: {ho_rmse:.2f}m | δ₁: {ho_d1:.1f}% | δ₂: {ho_d2:.1f}% | Metric AbsRel: {ho_m_absrel:.4f} | Metric RMSE: {ho_m_rmse:.2f}m | {ho_ms:.1f}ms")
        
        # Robust mean excluding extreme outliers (> 2.0 AbsRel, e.g. catwalk close-up)
        robust_ho = [x for x in held_out_items if x['abs_rel'] < 2.0]
        if len(robust_ho) < len(held_out_items):
            r_absrel = np.mean([x['abs_rel'] for x in robust_ho])
            r_rmse = np.mean([x['rmse'] for x in robust_ho])
            r_d1 = np.mean([x['d1'] for x in robust_ho])
            r_d2 = np.mean([x['d2'] for x in robust_ho])
            r_m_absrel = np.mean([x['metric_abs_rel'] for x in robust_ho])
            r_m_rmse = np.mean([x['metric_rmse'] for x in robust_ho])
            print(f"HELD-OUT ROBUST (N={len(robust_ho):<2})     | Mean AbsRel: {r_absrel:.4f} | RMSE: {r_rmse:.2f}m | δ₁: {r_d1:.1f}% | δ₂: {r_d2:.1f}% | Metric AbsRel: {r_m_absrel:.4f} | Metric RMSE: {r_m_rmse:.2f}m")
    print("="*95)

if __name__ == '__main__':
    evaluate_all()

