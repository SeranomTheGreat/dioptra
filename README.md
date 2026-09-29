# Dioptra-DINO: Real-Time Monocular Metric Depth Estimation via Canonical Virtual Camera Normalization for Edge Robotics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Preprint](https://img.shields.io/badge/Paper-PDF-red.svg)](research_paper/dioptra_dino_paper.pdf)

Official PyTorch implementation, evaluation benchmarks, and publication code for **Dioptra-DINO**, an efficient 27.51M-parameter vision transformer tailored for real-time monocular metric depth estimation on edge robotics platforms.

Coupling a self-supervised DINOv2-Small (`vits14`) visual backbone with a **Canonical Virtual Camera Transformation** ($F_{\text{canon}} = 1000.0\,\text{px}$) and an **Adaptive Receptive Alignment (ARA)** module, Dioptra-DINO delivers metric depth estimation directly at native $336 \times 336$ resolution at **16–17 FPS (58.2–62.9 ms)** on Apple Silicon MPS with $<240\text{ MB}$ memory footprint.

---

## Key Highlights

- **Photorealistic Ray-Traced Superiority (Apple Hypersim, 2,744 Frames)**: Dioptra-DINO achieves **0.1477 AbsRel** and **84.3% inliers ($\delta_1$)**, outperforming Metric3D ViT-Small (0.2259 AbsRel, 73.4% inliers) by **34.6% in relative error**.
- **Robustness Under Equal Resolution (100 Frames @ 336×336)**: When constrained to an identical $336 \times 336$ budget, Metric3D undergoes severe degradation (inliers collapse to **17.3%**, and **0.3% on real ScanNet iPad data**, underestimating scale by $>30\%$). Dioptra-DINO preserves solid inlier precision (**72.5%**), exact metric scale (**1.019×**), and runs in **62.9 ms**.
- **Real-Time Edge Efficiency**: Runs 12.6× faster than Metric3D (62.9 ms vs. 753.5 ms) and 6.2× faster than UniDepth V2 (62.9 ms vs. 421.6 ms) on edge accelerators.
- **Zero Metric Hallucination**: All metrics cited in the paper and benchmark tables are programmatically verifiable against serialized JSON evaluation receipts via `python research_paper/verify_paper_metrics.py`.

---

## Benchmark Results

### 1. Equal-Resolution Foundation Benchmark (All Models @ 336×336)
Constraining all foundation models to an identical $336 \times 336$ resolution budget across 100 indoor frames:

| Model Architecture | Parameters | Resolution | Direct AbsRel $\downarrow$ | RMSE (m) $\downarrow$ | Inlier $\delta_1$ $\uparrow$ | Scale Ratio | Edge Latency (MPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dioptra-DINO (Ours)** | **27.51M** | **336×336** | **0.2290** | **1.085 m** | **72.5%** | **1.019** | **62.9 ms (15.9 FPS)** |
| **UniDepth V2 (CVPR '24)** | 34.18M | 336×336 | 0.2643 | 1.448 m | 51.0% | 0.892 | 108.2 ms (9.2 FPS) |
| **Metric3D ViT-Small** | 37.50M | 336×336 | 0.3997 | 2.266 m | 17.3% | 0.698 | 76.1 ms (13.1 FPS) |

*Key finding on real handheld sensor data (ScanNet Scene00)*: Metric3D collapses to **0.3% inliers** (scale ratio 0.619×), whereas Dioptra-DINO maintains **90.9% inliers** (scale ratio 1.077×).

### 2. 3,000-Frame Pure Photorealistic True Indoor Metric Benchmark ($0.1\text{m} - 10.0\text{m}$)
Evaluated under each model's native configuration across Apple Hypersim (2,744 ray-traced frames) and InteriorNet (240 multi-room residential frames):

| Model Architecture | Direct AbsRel $\downarrow$ | RMSE (m) $\downarrow$ | Inlier $\delta_1$ $\uparrow$ | Scale Ratio | Normal MAE (°) $\downarrow$ | Native Resolution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dioptra-DINO (Ours)** | **0.1658** | **0.689 m** | **82.5%** | **1.040** | **27.4°** | $336 \times 336$ |
| **Metric3D ViT-Small** | 0.2360 | 0.955 m | 72.6% | 1.041 | 29.3° | $616 \times 1064$ |
| **Depth Anything V2 Metric** | 0.0902* | 0.425 m | 90.7% | 1.024 | 28.1° | $518 \times 518$ |

*\*Note on Depth Anything V2*: Depth Anything V2 achieves lower AbsRel on smooth synthetic walls, but lacks focal length conditioning and suffers catastrophic scale collapse ($>1.28$ AbsRel) when transferred to wider room topologies (e.g., TartanAir indoor enclosures).

---

## Quick Start

### Installation
```bash
git clone https://github.com/SeranomTheGreat/dioptra.git
cd dioptra

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Smoke Test and Unit Verification
Verify model instantiation, parameter footprint (27.51M parameters), and forward pass:
```bash
# Verify parameter count and memory envelope
python dioptra_dino.py --count

# Run forward pass smoke test
python dioptra_dino.py --smoke

# Run geometric unit tests
python dioptra_dino.py --test
```

### Single-Image Inference
Run metric depth prediction on a custom input image:
```bash
python scripts/eval_dino.py \
  --checkpoint outputs/dioptra_dino_best.pt \
  --image assets/sample.jpg \
  --fx 500.0 --fy 500.0 \
  --output test_outputs/depth_preview.png
```

---

## Evaluation Benchmarks

To reproduce the benchmark comparisons reported in the paper:

```bash
# 1. Run Equal-Resolution comparison (Dioptra vs. Metric3D vs. UniDepth @ 336x336)
python benchmark_equal_resolution_336.py

# 2. Run 3,000-frame Pure Photorealistic Indoor Benchmark
python benchmark_pure_indoor_3000.py

# 3. Verify zero-hallucination metric consistency across the paper
python research_paper/verify_paper_metrics.py
```

---

## Research Paper & Artifacts

The complete preprint manuscript, figures, and verification audit are located in `research_paper/`:
- **Preprint PDF**: [`research_paper/dioptra_dino_paper.pdf`](research_paper/dioptra_dino_paper.pdf)
- **LaTeX Source**: [`research_paper/dioptra_dino_paper.tex`](research_paper/dioptra_dino_paper.tex)
- **Markdown Manuscript**: [`research_paper/dioptra_dino_paper.md`](research_paper/dioptra_dino_paper.md)
- **BibTeX Citations**: [`research_paper/references.bib`](research_paper/references.bib)

---

## Repository Structure

```
dioptra/
├── dioptra_dino.py                 # Core Dioptra-DINO architecture (27.51M parameters)
├── dioptra.py                      # Original lightweight 8.1M architecture
├── benchmark_equal_resolution_336.py # Equal-resolution 336x336 benchmark runner
├── benchmark_pure_indoor_3000.py   # 3,000-frame Apple Hypersim & InteriorNet suite
├── benchmark_strictly_indoor.py    # Multi-environment stress-test evaluation
├── eval.py                         # Standard evaluation routines
├── scripts/
│   ├── eval_dino.py                # Dioptra-DINO inference and benchmarking
│   └── generate_publication_figures.py # Publication figure generation script
├── research_paper/                 # Complete preprint manuscript and assets
│   ├── dioptra_dino_paper.pdf      # Compiled 5-page publication PDF
│   ├── dioptra_dino_paper.tex      # IEEE/CVPR format LaTeX source
│   ├── dioptra_dino_paper.md       # Standalone Markdown paper
│   ├── references.bib              # 15 BibTeX citations
│   ├── verify_paper_metrics.py     # Programmatic metric audit script
│   └── figures/                    # 300 DPI high-resolution figures
└── assets/                         # Test imagery and sample inputs
```

---

## Citation

If you find Dioptra-DINO useful in your robotics or computer vision research, please cite our preprint:

```bibtex
@article{singh2026dioptradino,
  title={Dioptra-DINO: Real-Time Monocular Metric Depth Estimation via Canonical Virtual Camera Normalization for Edge Robotics},
  author={Singh, Yumnam Harryson},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
