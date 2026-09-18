#!/usr/bin/env python3
"""
Dioptra-DINO: Statistical Significance, Bootstrap 95% Confidence Intervals & Error Percentile Suite
Evaluates paired Wilcoxon signed-rank tests, paired t-tests, Cohen's d effect sizes,
10,000-iteration non-parametric bootstrapping for 95% CIs, error percentiles (P50-P99),
and empirical Cumulative Distribution Functions (CDFs) across all 200 benchmark frames.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def run_bootstrap_ci(data, n_boot=10000, ci=95, seed=42):
    """Compute non-parametric bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    
    alpha = (100 - ci) / 2.0
    low = np.percentile(boot_means, alpha)
    high = np.percentile(boot_means, 100 - alpha)
    return float(low), float(high), float(np.std(boot_means))

def compute_percentiles(data):
    """Compute standard robotics error percentiles."""
    return {
        "p50_median": float(np.percentile(data, 50)),
        "p75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data))
    }

def compute_paired_tests(dioptra_vals, baseline_vals, name):
    """Compute paired Wilcoxon signed-rank test, paired t-test, and Cohen's d."""
    diff = dioptra_vals - baseline_vals
    
    # Wilcoxon signed-rank test (two-sided)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(dioptra_vals, baseline_vals, alternative='two-sided')
    except Exception as e:
        w_stat, p_wilcoxon = float('nan'), float('nan')
        
    # Paired Student's t-test
    t_stat, p_ttest = stats.ttest_rel(dioptra_vals, baseline_vals)
    
    # Cohen's d for paired samples
    cohen_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
    
    # Superiority win rate (fraction of frames where Dioptra is better)
    win_rate = float(np.mean(dioptra_vals < baseline_vals) * 100.0)
    
    return {
        "baseline_name": name,
        "mean_diff": float(np.mean(diff)),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(p_wilcoxon),
        "ttest_stat": float(t_stat),
        "ttest_p_value": float(p_ttest),
        "cohens_d": cohen_d,
        "dioptra_win_rate_percent": win_rate
    }

def main():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(repo_dir, "outputs", "comprehensive_per_frame_comparison.csv")
    out_dir = os.path.join(repo_dir, "outputs")
    fig_dir = os.path.join(repo_dir, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[1/4] Loading continuous 200-frame CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    n_frames = len(df)
    print(f"Loaded {n_frames} evaluated frames.")

    models_meta = [
        {"key": "dioptra", "col_absrel": "dioptra_abs_rel", "col_rmse": "dioptra_rmse", "col_delta1": "dioptra_delta1", "label": "Dioptra-DINO (Ours)", "color": "#1f77b4"},
        {"key": "unidepth", "col_absrel": "unidepth_abs_rel", "col_rmse": "unidepth_rmse", "col_delta1": "unidepth_delta1", "label": "UniDepth-V2 ViT-Small", "color": "#2ca02c"},
        {"key": "da_midas", "col_absrel": "da_midas_abs_rel", "col_rmse": "da_midas_rmse", "col_delta1": "da_midas_delta1", "label": "Depth Anything V2 (Affine)", "color": "#ff7f0e"},
        {"key": "da_median", "col_absrel": "da_median_abs_rel", "col_rmse": "da_median_rmse", "col_delta1": "da_median_delta1", "label": "Depth Anything V2 (Median)", "color": "#9467bd"},
        {"key": "m3d_direct", "col_absrel": "m3d_direct_abs_rel", "col_rmse": "m3d_direct_rmse", "col_delta1": "m3d_direct_delta1", "label": "Metric3D ViT-Small (Direct)", "color": "#d62728"},
        {"key": "m3d_median", "col_absrel": "m3d_median_abs_rel", "col_rmse": "m3d_median_rmse", "col_delta1": "m3d_median_delta1", "label": "Metric3D ViT-Small (Median)", "color": "#8c564b"},
        {"key": "zoe", "col_absrel": "zoe_abs_rel", "col_rmse": "zoe_rmse", "col_delta1": "zoe_delta1", "label": "ZoeDepth ZoeD_NK", "color": "#e377c2"},
    ]

    # 1. 10,000-sample Non-Parametric Bootstrap 95% Confidence Intervals
    print("\n[2/4] Computing 10,000-Iteration Bootstrap 95% Confidence Intervals...")
    bootstrap_results = {}
    percentile_results = {}

    for m in models_meta:
        k = m["key"]
        absrel_vals = df[m["col_absrel"]].to_numpy()
        rmse_vals = df[m["col_rmse"]].to_numpy()
        delta1_vals = df[m["col_delta1"]].to_numpy()

        low_abs, high_abs, se_abs = run_bootstrap_ci(absrel_vals, n_boot=10000, ci=95)
        low_rmse, high_rmse, se_rmse = run_bootstrap_ci(rmse_vals, n_boot=10000, ci=95)
        low_d1, high_d1, se_d1 = run_bootstrap_ci(delta1_vals, n_boot=10000, ci=95)

        bootstrap_results[k] = {
            "model_label": m["label"],
            "absrel": {"mean": float(np.mean(absrel_vals)), "ci95_low": low_abs, "ci95_high": high_abs, "std_err": se_abs},
            "rmse": {"mean": float(np.mean(rmse_vals)), "ci95_low": low_rmse, "ci95_high": high_rmse, "std_err": se_rmse},
            "delta1": {"mean": float(np.mean(delta1_vals)), "ci95_low": low_d1, "ci95_high": high_d1, "std_err": se_d1},
        }

        percentile_results[k] = {
            "model_label": m["label"],
            "absrel_percentiles": compute_percentiles(absrel_vals),
            "rmse_percentiles": compute_percentiles(rmse_vals),
            "delta1_percentiles": compute_percentiles(delta1_vals),
        }
        print(f"  {m['label']:<30} | AbsRel CI95: [{low_abs:.4f}, {high_abs:.4f}] | RMSE CI95: [{low_rmse:.3f}, {high_rmse:.3f}] m")

    # 2. Paired Hypothesis Tests
    print("\n[3/4] Computing Paired Hypothesis Tests (Wilcoxon Signed-Rank & Paired t-Test)...")
    dioptra_absrel = df["dioptra_abs_rel"].to_numpy()
    paired_tests = {}

    for m in models_meta:
        if m["key"] == "dioptra":
            continue
        base_absrel = df[m["col_absrel"]].to_numpy()
        res = compute_paired_tests(dioptra_absrel, base_absrel, m["label"])
        paired_tests[m["key"]] = res
        print(f"  Dioptra vs {m['label']:<28} | Wilcoxon p={res['wilcoxon_p_value']:.2e} | t-test p={res['ttest_p_value']:.2e} | Cohen's d={res['cohens_d']:.3f} | Win Rate={res['dioptra_win_rate_percent']:.1f}%")

    statistical_suite = {
        "n_frames": n_frames,
        "n_bootstrap_iterations": 10000,
        "confidence_level": 95,
        "bootstrap_confidence_intervals": bootstrap_results,
        "error_percentiles": percentile_results,
        "paired_significance_tests": paired_tests
    }

    stat_json_path = os.path.join(out_dir, "benchmark_statistical_significance.json")
    perc_json_path = os.path.join(out_dir, "benchmark_error_percentiles.json")
    with open(stat_json_path, "w") as f:
        json.dump(statistical_suite, f, indent=2)
    with open(perc_json_path, "w") as f:
        json.dump(percentile_results, f, indent=2)
    print(f"\nSaved statistical results to: {stat_json_path}")
    print(f"Saved percentile results to: {perc_json_path}")

    # 3. Generate Publication CDF & Statistical Distribution Plot
    print("\n[4/4] Generating Publication CDF and Distribution Figures...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Empirical CDF of AbsRel
    ax1 = axes[0]
    for m in models_meta:
        if m["key"] in ["zoe", "da_median"]:
            continue  # Keep figure clean with core models
        vals = df[m["col_absrel"]].to_numpy()
        sorted_v = np.sort(vals)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        lw = 2.8 if m["key"] == "dioptra" else 1.8
        ls = '-' if m["key"] == "dioptra" else ('--' if 'm3d' in m["key"] else '-.')
        ax1.plot(sorted_v, cdf * 100.0, label=m["label"], color=m["color"], linewidth=lw, linestyle=ls)

    ax1.set_xlim(0.0, 0.35)
    ax1.set_ylim(0, 102)
    ax1.set_xlabel("Absolute Relative Error threshold ($\\tau$)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Percentage of Frames with AbsRel $\\leq \\tau$ (%)", fontsize=12, fontweight='bold')
    ax1.set_title("Empirical Cumulative Error Distribution (AbsRel CDF)", fontsize=13, fontweight='bold', pad=10)
    ax1.axvline(0.06, color='gray', linestyle=':', alpha=0.7, label="Rigorous 6% AbsRel Target")
    ax1.legend(loc='lower right', frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Error Percentile Progression (P50 to P99)
    ax2 = axes[1]
    percentile_labels = ['P50\n(Median)', 'P75', 'P90', 'P95', 'P99\n(Worst-Case)']
    percentile_keys = ['p50_median', 'p75', 'p90', 'p95', 'p99']

    for m in models_meta:
        if m["key"] in ["zoe", "da_median"]:
            continue
        p_vals = [percentile_results[m["key"]]["absrel_percentiles"][pk] for pk in percentile_keys]
        lw = 2.8 if m["key"] == "dioptra" else 1.8
        marker = 'o' if m["key"] == "dioptra" else ('s' if 'm3d' in m["key"] else '^')
        ax2.plot(range(len(percentile_keys)), p_vals, label=m["label"], color=m["color"], linewidth=lw, marker=marker, markersize=6)

    ax2.set_xticks(range(len(percentile_keys)))
    ax2.set_xticklabels(percentile_labels, fontsize=11)
    ax2.set_ylabel("Absolute Relative Error (AbsRel)", fontsize=12, fontweight='bold')
    ax2.set_title("Worst-Case Error Percentile Tail (P50 to P99)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(0.0, 0.50)
    ax2.legend(loc='upper left', frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig_out1 = os.path.join(out_dir, "fig_statistical_distributions_cdf.png")
    fig_paper1 = os.path.join(fig_dir, "fig_statistical_distributions_cdf.png")
    plt.savefig(fig_out1, dpi=300)
    plt.savefig(fig_paper1, dpi=300)
    plt.close()

    print(f"Generated CDF plot at:\n  {fig_out1}\n  {fig_paper1}")
    print("\n[SUCCESS] Statistical significance and distribution suite finished.")

if __name__ == "__main__":
    main()
