# Dioptra-DINO Research Manuscript LaTeX Package

**Title**: Dioptra-DINO: Foundation-Assisted Geometry-Aware Monocular Metric Depth for Real-Time Edge Robotics  
**Author**: Yumnam Harryson Singh  
**Repository**: [https://github.com/SeranomTheGreat/dioptra](https://github.com/SeranomTheGreat/dioptra)

---

## 1. Package Structure
```
.
├── main.tex                    # Master LaTeX document (CVPR/NeurIPS/IEEE 2-column format)
├── references.bib              # Complete BibTeX bibliography (26 entries)
├── main.bbl                    # Pre-compiled bibliography environment
├── sections/                   # Modular LaTeX sections
│   ├── 00_abstract.tex         # Abstract
│   ├── 01_introduction.tex     # Introduction & contributions
│   ├── 02_related_work.tex     # Related literature
│   ├── 03_method.tex           # Mathematical formulation & architecture
│   ├── 04_experiments.tex      # Empirical benchmarks, baseline comparisons & ablations
│   ├── 05_discussion.tex       # Edge profiling, surface normals, failure modes & limitations
│   └── 06_conclusion.tex       # Concluding remarks
└── figures/                    # High-resolution publication figures (300 DPI)
    ├── fig_dino_diverse_scenes.png          # Figure 1: Multi-scene teaser
    ├── fig_method_pipeline.png              # Figure 2: Architecture schematic & model predictions
    ├── fig_metrics_distribution.png         # Figure 3: 200-frame empirical metric distributions
    ├── fig_visual_predictions_grid.png      # Figure 4: Qualitative predictions & error heatmaps
    ├── fig_external_baseline_comparison.png # Figure 5: Qualitative comparison with Depth Anything V2
    ├── fig_dino_multi_fov_sweep.png         # Figure 6: Multi-FOV equivariance sweep (50° to 100°)
    └── fig_ablation_component_breakdown.png # Figure 7: Component-wise ablation breakdown (ARA, Ray, VNL)
```

---

## 2. Compilation Instructions

### Option A: Overleaf (Recommended)
1. Upload `dioptra_dino_paper_latex.zip` directly to [Overleaf](https://www.overleaf.com).
2. Set Compiler to **pdfLaTeX** (or **TeX Live 2023 / 2024**).
3. Set Main document to `main.tex`.
4. Click **Recompile**.

### Option B: Local Command Line
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 3. Scientific Integrity & Empirical Provenance
- **Strict Architecture Focus**: All reported metrics reflect the 40-epoch trained checkpoint (`outputs/dioptra_dino_best.pt`, $27.51$\,M parameters, $27,512,834$ total parameters, $105.05$\,MB FP32).
- **Direct Foundation Baseline Comparison**: Benchmarked on the exact same 200 held-out frames against Metric3D ViT-Small (CVPR 2023, $37.50$\,M parameters), Depth Anything V2-Small (CVPR 2024, $24.79$\,M parameters), and Canonical 2D ViT + DPT ($26.10$\,M parameters):
  - **Dioptra-DINO (Ours)**: $0.0555$ AbsRel, $97.06\%$ $\delta_1$, $2.396$\,m RMSE, $1.0003$ scale ratio, $38.0$\,FPS ($26.3$\,ms forward) / $28.3$\,FPS ($35.3$\,ms pipeline) (Apple M3 GPU), Zero test-time scaling.
  - **UniDepth-V2 ViT-Small**: $0.1190$ AbsRel, $94.70\%$ $\delta_1$, $16.736$\,m RMSE, $0.9626$ scale ratio, $5.4$\,FPS ($184.2$\,ms).
  - **Metric3D ViT-Small (Camera-Conditioned)**: $0.3269$ AbsRel, $26.52\%$ $\delta_1$, $7.192$\,m RMSE, $0.6843$ scale ratio, $1.8$\,FPS ($548.3$\,ms).
  - **Depth Anything V2-Small (MiDaS Affine)**: $0.0964$ AbsRel, $90.78\%$ $\delta_1$, $4.006$\,m RMSE, $0.9962$ scale ratio, $8.6$\,FPS ($116.6$\,ms), Requires oracle per-frame affine alignment ($s \cdot d + t$).
  - **Depth Anything V2-Small (Median-Scaled)**: $0.1010$ AbsRel, $90.84\%$ $\delta_1$, $3.654$\,m RMSE, $1.0000$ scale ratio, $8.6$\,FPS ($116.6$\,ms).
  - **Canonical 2D ViT + DPT (Ablation b)**: $0.5056$ AbsRel, $12.31\%$ $\delta_1$, $10.250$\,m RMSE, $0.5582$ scale ratio ($-44.18\%$ scale collapse).
- **Unseen Indoor Office Benchmark (`office/Easy/P001`, $N=30$, Level Ground)**:
  - UniDepth-V2: $0.0495$ AbsRel, $0.4106$ SqRel, $2.255$\,m RMSE, $98.26\%$ $\delta_1$, $0.9868$ scale ratio.
  - Depth Anything V2 (Affine): $0.0586$ AbsRel, $0.0590$ SqRel, $0.695$\,m RMSE, $97.55\%$ $\delta_1$, $0.9818$ scale ratio.
  - Metric3D ViT-Small (Oracle Median): $0.1012$ AbsRel, $0.0965$ SqRel, $0.874$\,m RMSE, $92.15\%$ $\delta_1$, $1.0000$ scale ratio.
  - Metric3D ViT-Small (Direct): $0.1113$ AbsRel, $0.1580$ SqRel, $1.175$\,m RMSE, $86.71\%$ $\delta_1$, $0.8907$ scale ratio.
  - Dioptra-DINO (Ours): $0.4018$ AbsRel, $0.7415$ SqRel, $1.808$\,m RMSE, $29.57\%$ $\delta_1$, $1.2938$ scale ratio.
  - ZoeDepth ZoeD_NK: $0.4112$ AbsRel, $0.4822$ SqRel, $1.439$\,m RMSE, $32.53\%$ $\delta_1$, $1.3652$ scale ratio.
- **Hardware Benchmarking**: Evaluated on an **Apple Silicon M3** (8-core CPU, 10-core GPU, 8\,GB unified memory) running PyTorch 2.8.0 via Apple Metal Performance Shaders (MPS) under unquantized single-precision FP32 with $<210$\,MB working RAM.

