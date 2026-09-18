"""
Benchmark 1: Temporal Scale Stability & Video Trajectory Drift Analysis
Evaluates the continuous 200-frame trajectory (TartanAir AbandonedFactory P010) to quantify:
1. Frame-to-frame scale volatility: Delta s_t = |s_{t+1} - s_t|
2. Trajectory scale standard deviation: sigma(s)
3. Maximum excursion from physical reality: max |s_t - 1.0|
4. Percentage of frames within +/- 5% of ideal scale (s in [0.95, 1.05])

Generates:
  outputs/benchmark_temporal_stability.json
  outputs/temporal_scale_drift.csv
  paper/figures/fig_temporal_scale_stability.png
  outputs/fig_temporal_scale_stability.png
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

def main():
    print("=" * 80)
    print(" BENCHMARK 1: TEMPORAL SCALE STABILITY & TRAJECTORY DRIFT ANALYSIS")
    print("=" * 80)

    csv_path = "outputs/comprehensive_per_frame_comparison.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run comprehensive metric suite first.")

    df = pd.read_csv(csv_path)
    print(f"[Data] Loaded {len(df)} frames from {csv_path}")

    # Models to analyze
    # For models with explicit scale ratio:
    models_scale = {
        "Dioptra-DINO (Ours)": "dioptra_scale",
        "UniDepth-V2 ViT-S": "unidepth_scale",
        "Metric3D ViT-S": "m3d_direct_scale",
        "Depth Anything V2 (Indoor)": "da_indoor_scale",
        "ZoeDepth ZoeD_NK": "zoe_scale",
    }

    results = {}
    time_series = {}

    for name, col in models_scale.items():
        s = df[col].values
        # Temporal derivative
        ds = np.abs(np.diff(s))
        
        mean_scale = float(np.mean(s))
        median_scale = float(np.median(s))
        std_scale = float(np.std(s))
        mean_jitter = float(np.mean(ds))
        max_jitter = float(np.max(ds))
        max_excursion = float(np.max(np.abs(s - 1.0)))
        pct_within_5pct = float((np.abs(s - 1.0) <= 0.05).mean() * 100.0)
        pct_within_10pct = float((np.abs(s - 1.0) <= 0.10).mean() * 100.0)

        results[name] = {
            "mean_scale": mean_scale,
            "median_scale": median_scale,
            "std_scale": std_scale,
            "mean_interframe_jitter": mean_jitter,
            "max_interframe_jitter": max_jitter,
            "max_scale_excursion": max_excursion,
            "pct_within_5pct": pct_within_5pct,
            "pct_within_10pct": pct_within_10pct,
        }
        time_series[name] = s

    # Print summary table
    print("\n" + "-" * 105)
    print(f"{'Model Name':<28} | {'Median Scale':<12} | {'Std Dev σ(s)':<12} | {'Mean Jitter |Δs|':<16} | {'Max Excursion':<14} | {'In ±5% Band':<10}")
    print("-" * 105)
    for name, r in results.items():
        print(f"{name:<28} | {r['median_scale']:<12.4f} | {r['std_scale']:<12.4f} | {r['mean_interframe_jitter']:<16.4f} | {r['max_scale_excursion']:<14.4f} | {r['pct_within_5pct']:<9.1f}%")
    print("-" * 105)

    # Save to JSON
    out_json = "outputs/benchmark_temporal_stability.json"
    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[Saved] Temporal stability metrics saved to {out_json}")

    # Generate Publication Figure
    print("\n[Viz] Generating temporal scale stability trajectory figure...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    frames = np.arange(len(df))
    colors = {
        "Dioptra-DINO (Ours)": "#00E676",       # Vibrant Emerald Green
        "UniDepth-V2 ViT-S": "#2979FF",         # Bright Blue
        "Metric3D ViT-S": "#FF9100",            # Amber Orange
        "Depth Anything V2 (Indoor)": "#D500F9", # Magenta
        "ZoeDepth ZoeD_NK": "#78909C"           # Slate Grey
    }

    # Top plot: Scale ratio over time
    ax1.axhline(1.0, color="#FFFFFF", linestyle="--", linewidth=1.5, alpha=0.9, label="Ideal Physical Scale (s = 1.0)")
    ax1.axhspan(0.95, 1.05, color="#00E676", alpha=0.10, label="±5% Operational Tolerance Band")

    for name, s in time_series.items():
        lw = 2.4 if "Ours" in name else 1.4
        alpha = 1.0 if "Ours" in name else 0.8
        ax1.plot(frames, s, label=f"{name} (σ={results[name]['std_scale']:.3f})", color=colors[name], linewidth=lw, alpha=alpha)

    ax1.set_ylabel("Metric Scale Ratio ($s = \\hat{D} / D^*$)", fontsize=11, fontweight="bold")
    ax1.set_title("Temporal Metric Scale Stability Across Continuous 200-Frame Drone Trajectory", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylim(0.4, 2.2)
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    # Bottom plot: Instantaneous inter-frame scale derivative |s_{t+1} - s_t|
    for name, s in time_series.items():
        ds = np.abs(np.diff(s))
        lw = 2.0 if "Ours" in name else 1.0
        alpha = 0.9 if "Ours" in name else 0.6
        ax2.plot(frames[1:], ds, label=name, color=colors[name], linewidth=lw, alpha=alpha)

    ax2.set_xlabel("Trajectory Frame Index ($t$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Inter-Frame Jitter\n$|s_{t+1} - s_t|$", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 0.25)
    ax2.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()

    out_paper_fig = "paper/figures/fig_temporal_scale_stability.png"
    out_outputs_fig = "outputs/fig_temporal_scale_stability.png"
    os.makedirs(os.path.dirname(out_paper_fig), exist_ok=True)
    os.makedirs(os.path.dirname(out_outputs_fig), exist_ok=True)

    fig.savefig(out_paper_fig, dpi=300, bbox_inches="tight")
    fig.savefig(out_outputs_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Temporal stability figures saved to {out_paper_fig} and {out_outputs_fig}")


if __name__ == "__main__":
    main()
