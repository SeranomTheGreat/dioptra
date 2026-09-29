#!/usr/bin/env python3
"""
generate_publication_figures.py
Generates publication-quality figures for the Dioptra-DINO research paper:
1. fig1_architecture.png: Visual system architecture flow diagram.
2. fig6_error_distribution.png: Performance trade-off & accuracy bar charts.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("research_paper/figures", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Figure 1: Architecture Diagram
# -----------------------------------------------------------------------------
def make_architecture_figure():
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#ffffff", dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def draw_box(ax, x, y, w, h, title, subtitle, color, text_color="#1a1a1a"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=color, edgecolor="#333333", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.15, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=text_color)
        if subtitle:
            ax.text(x + w/2, y + h/2 - 0.22, subtitle, ha="center", va="center",
                    fontsize=8.5, color="#444444")

    def draw_arrow(ax, x1, y1, x2, y2, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.8, shrinkA=3, shrinkB=3)
        )
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.15, label, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#1f4e79")

    # Boxes
    draw_box(ax, 0.5, 3.2, 1.8, 1.4, "Input RGB", "Image (H x W x 3)\nAny Aspect Ratio", "#e8f4f8")
    draw_box(ax, 0.5, 1.0, 1.8, 1.4, "Camera Intrinsics", "K = [fx, fy, cx, cy]\nPinhole Matrix", "#fef3d6")

    draw_box(ax, 3.0, 3.2, 1.8, 1.4, "Resize & Norm", "Bilinear (336 x 336)\nImageNet Mean/Std", "#e8f4f8")
    draw_box(ax, 3.0, 1.0, 2.0, 1.4, "Canonical Virtual\nCamera Normalization", "F_canon = 1000 px\ns = f_x / F_canon", "#fef3d6")

    draw_box(ax, 5.5, 3.0, 2.2, 1.8, "DINOv2-Small Backbone", "ViT-S/14 (21.7M Params)\nTokens: 24x24 = 576", "#d1e7dd")
    draw_box(ax, 8.4, 3.0, 2.0, 1.8, "Multi-Scale FPN\nDecoder", "Features from\nBlocks [3, 6, 9, 12]", "#d1e7dd")

    draw_box(ax, 6.2, 0.8, 3.4, 1.5, "Adaptive Receptive\nAlignment (ARA)", "Intrinsics conditioning\nGating multi-scale tokens", "#f8d7da")

    draw_box(ax, 11.0, 2.6, 2.5, 1.6, "Metric Depth Head", "Scale & Shift Decoupled\nd_metric = d_canon * (fx / F_canon)", "#cfe2ff")

    # Arrows
    draw_arrow(ax, 2.3, 3.9, 3.0, 3.9)
    draw_arrow(ax, 2.3, 1.7, 3.0, 1.7)
    draw_arrow(ax, 4.8, 3.9, 5.5, 3.9, "336x336x3")
    draw_arrow(ax, 7.7, 3.9, 8.4, 3.9, "4-Scale Feats")
    draw_arrow(ax, 5.0, 1.7, 6.2, 1.55, "Scale Ratio")
    draw_arrow(ax, 9.4, 3.0, 8.5, 2.3)
    draw_arrow(ax, 8.5, 2.3, 8.5, 2.3)
    draw_arrow(ax, 9.6, 1.55, 11.0, 3.0, "ARA Features")
    draw_arrow(ax, 10.4, 3.9, 11.0, 3.6, "FPN Features")

    # Title
    ax.text(7.0, 5.6, "Dioptra-DINO: Architectural Overview & Canonical Virtual Camera Normalization",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#111111")
    ax.text(7.0, 5.25, "Compact 27.51M parameter metric depth architecture running at native 336x336 resolution for real-time edge robotics",
            ha="center", va="center", fontsize=9.5, style="italic", color="#555555")

    plt.tight_layout()
    fig.savefig("research_paper/figures/fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved research_paper/figures/fig1_architecture.png")


# -----------------------------------------------------------------------------
# 2. Figure 6: Comparative Charts
# -----------------------------------------------------------------------------
def make_comparison_figure():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6), dpi=300, facecolor="#ffffff")

    models = ["Dioptra-DINO\n(Ours, 336)", "Metric3D\n(ViT-S, 336)", "UniDepth V2\n(ViT-S, 336)", "Metric3D\n(Native 616x1064)", "UniDepth V2\n(Native Full-Res)"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e6ab02", "#a6761d"]

    # Chart 1: Overall AbsRel
    abs_rels = [0.2290, 0.3997, 0.2643, 0.2360, 0.2357]
    bars1 = ax1.bar(models, abs_rels, color=colors, width=0.55, edgecolor="#222222", linewidth=1.1)
    ax1.set_ylabel("Direct AbsRel (Lower is Better)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Direct Absolute Relative Error", fontsize=11, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 0.46)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"{yval:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Chart 2: Inlier Accuracy delta1
    delta1 = [72.5, 17.3, 51.0, 72.6, 60.0]
    bars2 = ax2.bar(models, delta1, color=colors, width=0.55, edgecolor="#222222", linewidth=1.1)
    ax2.set_ylabel("Inlier Accuracy δ < 1.25 % (Higher is Better)", fontsize=10.5, fontweight="bold")
    ax2.set_title("Inlier Precision (δ₁ Coverage)", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.set_ylim(0, 85)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Chart 3: Latency vs AbsRel Pareto Curve
    latencies = [62.9, 76.1, 108.2, 753.5, 421.6]
    scatter_labels = ["Dioptra-DINO (336)", "Metric3D (336)", "UniDepth V2 (336)", "Metric3D (616x1064)", "UniDepth V2 (Native)"]
    for i in range(len(models)):
        ax3.scatter(latencies[i], abs_rels[i], color=colors[i], s=140, edgecolor="#111111", zorder=4, label=scatter_labels[i])
        offset_y = 0.015 if i != 1 else -0.025
        offset_x = 10 if i < 3 else -80
        ax3.annotate(scatter_labels[i], (latencies[i], abs_rels[i]),
                     textcoords="offset points", xytext=(offset_x, offset_y*500),
                     fontsize=8.5, fontweight="bold", color=colors[i])

    ax3.set_xlabel("Inference Latency (ms) on Apple Silicon MPS", fontsize=10.5, fontweight="bold")
    ax3.set_ylabel("Direct AbsRel", fontsize=10.5, fontweight="bold")
    ax3.set_title("Latency vs. Error Trade-off", fontsize=11, fontweight="bold", pad=10)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.set_xlim(0, 850)
    ax3.set_ylim(0.18, 0.45)

    for ax in (ax1, ax2):
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)

    plt.tight_layout()
    fig.savefig("research_paper/figures/fig6_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved research_paper/figures/fig6_error_distribution.png")

if __name__ == "__main__":
    make_architecture_figure()
    make_comparison_figure()
