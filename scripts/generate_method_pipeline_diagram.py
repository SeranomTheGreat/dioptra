"""
Generate Publication-Grade Figure 2: Dioptra-DINO Architecture and Geometry-Coupled Training Framework.
Features:
1. End-to-end architectural flow diagram (Trivision Ray Encoding, FiLM, DINOv2 ViT, ARA, DPT Decoder, Loss).
2. Live qualitative output panels evaluated directly on the 40-epoch Dioptra-DINO checkpoint (outputs/dioptra_dino_best.pt).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
import __main__
setattr(__main__, "DioptraDINOConfig", DioptraDINOConfig)


def generate_pipeline_figure(
    ckpt_path: str = "outputs/dioptra_dino_best.pt",
    sample_img: str = "test_samples/abandonedfactory/000100_left.png",
    sample_gt: str = "test_samples/abandonedfactory/000100_left_depth.npy",
    out_path: str = "paper/figures/fig_method_pipeline.png"
):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading checkpoint {ckpt_path} on {device}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", DioptraDINOConfig())
    model = DioptraDINO(cfg).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()

    # Process sample
    raw_img = Image.open(sample_img).convert("RGB")
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

    gt_arr = np.load(sample_gt).astype(np.float32)
    gt_t = torch.from_numpy(gt_arr)
    gt_cropped = gt_t[top:top + min_side, left:left + min_side]
    gt_depth = F.interpolate(gt_cropped.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="nearest").squeeze().numpy()

    fx = 320.0 * (224.0 / float(min_side))
    K = torch.tensor([[[fx, 0.0, 112.0], [0.0, fx, 112.0], [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

    with torch.no_grad():
        pred = model(input_t, K)
        pred_np = pred.squeeze().cpu().numpy()

    # Compute 3D surface normals
    cx, cy = 112.0, 112.0
    y_grid, x_grid = torch.meshgrid(
        torch.arange(224, device=device, dtype=torch.float32),
        torch.arange(224, device=device, dtype=torch.float32),
        indexing="ij"
    )
    pts = torch.cat([(x_grid - cx) / fx * pred, (y_grid - cy) / fx * pred, pred], dim=1)
    s = 2
    vx = pts[:, :, s:-s, 2 * s:] - pts[:, :, s:-s, :-2 * s]
    vy = pts[:, :, 2 * s:, s:-s] - pts[:, :, :-2 * s, s:-s]
    normal = torch.cross(vx, vy, dim=1)
    normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(min=1e-6)
    norm_np = normal.squeeze().permute(1, 2, 0).cpu().numpy()
    norm_rgb = ((norm_np + 1.0) / 2.0).clip(0, 1)
    norm_full = np.zeros((224, 224, 3), dtype=np.float32)
    norm_full[s:-s, s:-s] = norm_rgb

    # Now create composite figure
    fig = plt.figure(figsize=(16, 8.8), dpi=300)
    gs = GridSpec(2, 4, height_ratios=[1.3, 1.0], hspace=0.35, wspace=0.25)

    # Top panel: Architectural Schema
    ax_arch = fig.add_subplot(gs[0, :])
    ax_arch.set_xlim(0, 100)
    ax_arch.set_ylim(0, 48)
    ax_arch.axis("off")

    def draw_box(ax, x, y, w, h, title, subtitle, color, text_color="white", border="#2c3e50"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            linewidth=1.5, edgecolor=border, facecolor=color
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.62, title, color=text_color, fontsize=10, fontweight="bold", ha="center", va="center")
        ax.text(x + w / 2, y + h * 0.28, subtitle, color=text_color, fontsize=8, ha="center", va="center")

    def draw_arrow(ax, x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#2c3e50", edgecolor="#2c3e50", width=1.5, headwidth=6, headlength=7)
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.2, label, fontsize=7.5, fontweight="bold", color="#2c3e50", ha="center")

    # Draw Architecture Blocks
    # Block 1: Input Frame & Intrinsics
    draw_box(ax_arch, 1.5, 27, 15, 17, "Input & Intrinsics", "RGB (3x224x224)\nK' (Dynamic Pinhole)", "#1f77b4")

    # Block 2: Trivision Ray PE & FiLM
    draw_box(ax_arch, 20.5, 27, 16.5, 17, "Trivision Ray PE", "K'⁻¹ Cramer Unproject\nFourier γ(Rᵢ) → MLP", "#2ca02c")

    # Block 3: DINOv2-Small ViT Backbone
    draw_box(ax_arch, 41, 27, 16.5, 17, "DINOv2-Small ViT", "14x14 Patches (256)\n12 Transformer Layers", "#d95f02")

    # Block 4: Angular Residual Attention (ARA)
    draw_box(ax_arch, 61.5, 27, 16.5, 17, "ARA Mechanism", "sin²(θ_qk) Ray Penalty\nCosine Warmup Γ(t)", "#7570b3")

    # Block 5: DPT Decoder
    draw_box(ax_arch, 82, 27, 16.5, 17, "DPT Decoder", "Multi-Scale Fusion\nFull 224x224 Metric Depth", "#e7298a")

    # Arrows between main blocks
    draw_arrow(ax_arch, 16.5, 38, 20.5, 38, "K'")
    draw_arrow(ax_arch, 16.5, 29.5, 41, 29.5, "")
    ax_arch.text(28.7, 27.5, "Tokens X₀", fontsize=8, fontweight="bold", color="#1f77b4", ha="center")

    draw_arrow(ax_arch, 37, 38, 41, 38, "FiLM")
    draw_arrow(ax_arch, 57.5, 35.5, 61.5, 35.5, "ARA Bias")
    draw_arrow(ax_arch, 77.5, 35.5, 82, 35.5, "Tokens")

    # Lower supervision flow
    rect_loss = patches.FancyBboxPatch(
        (16, 2.5), 68, 17,
        boxstyle="round,pad=0.5,rounding_size=1.2",
        linewidth=1.5, edgecolor="#7f7f7f", facecolor="#f8f9fa"
    )
    ax_arch.add_patch(rect_loss)
    ax_arch.text(50, 16.2, "Decoupled Multi-Task Geometric Supervision", fontsize=11, fontweight="bold", color="#1a252f", ha="center")

    loss_items = [
        ("Raw Physical SiLog", "L_silog (λ = 0.85)", "#34495e"),
        ("Log-Median Scale", "L_scale (|log med(D)|)", "#2980b9"),
        ("Multi-Scale Sobel Edge", "L_edge (Occlusion)", "#8e44ad"),
        ("3D Virtual Normal Loss", "L_vnl (Planar Geometry)", "#c0392b"),
    ]
    for idx, (lt, ls, lc) in enumerate(loss_items):
        lx = 18 + idx * 16.2
        draw_box(ax_arch, lx, 4.0, 14.5, 10, lt, ls, lc, text_color="white")

    draw_arrow(ax_arch, 90.25, 27, 90.25, 11, "")
    draw_arrow(ax_arch, 90.25, 11, 84.5, 11, "")
    ax_arch.text(87.5, 12.5, "Supervision", fontsize=8, fontweight="bold", color="#2c3e50", ha="center")

    ax_arch.set_title("Dioptra-DINO: End-to-End Geometry-Coupled Architecture & Supervision", fontsize=13, fontweight="bold", pad=12)

    # Bottom 4 Panels: Live Outputs from 40-Epoch Model
    ax_rgb = fig.add_subplot(gs[1, 0])
    ax_rgb.imshow(img_np)
    ax_rgb.set_title("(a) Input RGB (224x224)", fontsize=11, fontweight="bold")
    ax_rgb.axis("off")

    ax_gt = fig.add_subplot(gs[1, 1])
    vmax = min(max(float(gt_depth[gt_depth < 80].max()), 10.0), 30.0)
    im_gt = ax_gt.imshow(gt_depth, cmap="plasma", vmin=0.2, vmax=vmax)
    ax_gt.set_title("(b) Ground Truth Metric Depth", fontsize=11, fontweight="bold")
    ax_gt.axis("off")
    cbar_gt = plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
    cbar_gt.set_label("Metres (m)", fontsize=9)

    ax_pred = fig.add_subplot(gs[1, 2])
    im_pred = ax_pred.imshow(pred_np, cmap="plasma", vmin=0.2, vmax=vmax)
    ax_pred.set_title("(c) Dioptra-DINO Epoch 40 (Ours)", fontsize=11, fontweight="bold")
    ax_pred.axis("off")
    cbar_p = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
    cbar_p.set_label("Metres (m)", fontsize=9)

    ax_norm = fig.add_subplot(gs[1, 3])
    ax_norm.imshow(norm_full)
    ax_norm.set_title("(d) 3D Virtual Normals (VNL)", fontsize=11, fontweight="bold")
    ax_norm.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated clean Figure 2 to: {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_pipeline_figure()
