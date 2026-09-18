import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=300)

variants = [
    'Full Dioptra-DINO\n(Headline)',
    'w/o 3D VNL\n(no-vnl)',
    'Canonical 2D ViT\n(no-ray)',
    'Center-Ray PE\n(center-ray)',
    'w/o ARA\n(no-ara)'
]

colors = ['#1a73e8', '#f2994a', '#eb5757', '#bb6bd9', '#e056fd']

# 1. AbsRel Error (lower is better)
abs_rels = [0.0542, 0.4852, 0.5056, 0.5454, 0.5516]
bars1 = axes[0].bar(range(len(variants)), abs_rels, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
axes[0].set_title(r'(a) Absolute Relative Error $\downarrow$', fontweight='bold', pad=12)
axes[0].set_ylabel('AbsRel (Lower is Better)')
axes[0].set_xticks(range(len(variants)))
axes[0].set_xticklabels(variants, rotation=25, ha='right', fontsize=9)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars1:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., h + 0.015, f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].set_ylim(0, 0.65)

# 2. RMSE Error (lower is better)
rmses = [2.385, 10.378, 10.250, 10.711, 11.099]
bars2 = axes[1].bar(range(len(variants)), rmses, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
axes[1].set_title(r'(b) Root Mean Squared Error $\downarrow$', fontweight='bold', pad=12)
axes[1].set_ylabel('RMSE (Metres)')
axes[1].set_xticks(range(len(variants)))
axes[1].set_xticklabels(variants, rotation=25, ha='right', fontsize=9)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars2:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., h + 0.25, f'{h:.2f}m', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].set_ylim(0, 13)

# 3. Accuracy Threshold delta1 < 1.25 (higher is better)
d1s = [97.09, 15.96, 12.31, 11.58, 12.47]
bars3 = axes[2].bar(range(len(variants)), d1s, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
axes[2].set_title(r'(c) Threshold Accuracy $\delta_1 < 1.25$ $\uparrow$', fontweight='bold', pad=12)
axes[2].set_ylabel(r'$\delta_1$ Accuracy (%)')
axes[2].set_xticks(range(len(variants)))
axes[2].set_xticklabels(variants, rotation=25, ha='right', fontsize=9)
axes[2].grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars3:
    h = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., h + 2.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[2].set_ylim(0, 115)

# 4. Metric Scale Ratio (closer to 1.0 is better)
scales = [1.0060, 0.5718, 0.5582, 0.5789, 0.4901]
bars4 = axes[3].bar(range(len(variants)), scales, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
axes[3].axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Ideal Unity (1.000)')
axes[3].set_title(r'(d) Metric Scale Ratio ($s / s_{gt}$)', fontweight='bold', pad=12)
axes[3].set_ylabel('Scale Ratio (Target = 1.0)')
axes[3].set_xticks(range(len(variants)))
axes[3].set_xticklabels(variants, rotation=25, ha='right', fontsize=9)
axes[3].grid(axis='y', linestyle='--', alpha=0.5)
axes[3].legend(loc='lower left', fontsize=9)
for bar in bars4:
    h = bar.get_height()
    axes[3].text(bar.get_x() + bar.get_width()/2., h + 0.025, f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[3].set_ylim(0, 1.2)

plt.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
out_fig = 'paper/figures/fig_ablation_component_breakdown.png'
plt.savefig(out_fig, bbox_inches='tight')
print(f'Saved high-resolution ablation figure: {out_fig}')
