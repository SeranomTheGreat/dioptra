import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dioptra_dino import DioptraDINO, DioptraDINOConfig
from scripts.eval_dino import load_model, preprocess_sample

def generate():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = load_model('outputs_dino/dioptra_dino_epoch26_best.pt', device=device)

    img_p = 'test_samples/abandonedfactory/000100_left.png'
    gt_p = 'test_samples/abandonedfactory/000100_left_depth.npy'

    inp, gt, img_t, K, fx = preprocess_sample(img_p, gt_p, device=device)

    with torch.no_grad():
        pred = model(inp, K, ara_gate=1.0)
        pred_np = pred.squeeze().cpu().numpy()

    # Compute 3D surface normals
    B, _, H, W = pred.shape
    cx, cy = 112.0, 112.0
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    pts = torch.cat([(x_grid - cx) / fx * pred, (y_grid - cy) / fx * pred, pred], dim=1)
    s = 2
    vx = pts[:, :, s:-s, 2*s:] - pts[:, :, s:-s, :-2*s]
    vy = pts[:, :, 2*s:, s:-s] - pts[:, :, :-2*s, s:-s]
    normal = torch.cross(vx, vy, dim=1)
    normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(min=1e-6)
    norm_np = normal.squeeze().permute(1, 2, 0).cpu().numpy()
    norm_rgb = ((norm_np + 1.0) / 2.0).clip(0, 1)

    norm_full = np.zeros((224, 224, 3), dtype=np.float32)
    norm_full[s:-s, s:-s] = norm_rgb

    disp_np = 1.0 / np.clip(pred_np, 0.5, 50.0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=300)

    # 1. RGB
    rgb_disp = img_t.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    axes[0].imshow(rgb_disp)
    axes[0].set_title('(a) Input RGB (224x224)', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # 2. Predicted Metric Depth
    d_mask = (gt > 0.1) & (gt < 50.0)
    vmin = np.percentile(gt[d_mask], 2)
    vmax = np.percentile(gt[d_mask], 98)
    im1 = axes[1].imshow(pred_np, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[1].set_title('(b) Dioptra-DINO Metric Depth (m)', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Metres (m)', fontsize=10)

    # 3. 3D Surface Normals (VNL)
    axes[2].imshow(norm_full)
    axes[2].set_title('(c) 3D Virtual Normals [nx, ny, nz]', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    # 4. Proximity / Disparity Map
    im3 = axes[3].imshow(disp_np, cmap='magma')
    axes[3].set_title('(d) Obstacle Proximity (1 / Depth)', fontsize=13, fontweight='bold')
    axes[3].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.set_label('Inverse Depth (1/m)', fontsize=10)

    plt.tight_layout()
    out_p = 'paper/figures/fig_dino_method_analysis.png'
    plt.savefig(out_p, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Successfully saved: {out_p}')

if __name__ == '__main__':
    generate()
