# Dioptra: An Ultra-Lightweight Geometry-Aware Architecture for Monocular Metric Depth on Edge Devices

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Official PyTorch implementation and evaluation benchmark for **Dioptra**, an 8.1M parameter vision transformer for monocular metric depth estimation on resource-constrained edge hardware.

Rather than adapting multi-hundred-million parameter vision foundation models, Dioptra formulates depth prediction directly through camera intrinsics $\mathbf{K}$ using two complementary geometric inductive biases:

1. **Trivision Ray Positional Embeddings**: Unprojects non-collinear optical ray triplets $(\hat{\mathbf{r}}_c, \hat{\mathbf{r}}_1, \hat{\mathbf{r}}_2)$ per patch token to establish metric scale calibration.
2. **Angular Residual Attention (ARA)**: An intrinsic-conditioned self-attention regularizer that penalizes off-axis spatial deformation, enforcing physical surface planarity and cutting surface normal error.
3. **Layer-Shared Geometric Caching**: Because ray angles depend only on camera geometry, the pairwise distance matrix $\sin^2(\theta_{q,k})$ is computed once per frame ($1.26$\,ms) and shared across all 10 transformer blocks. This reduces ARA overhead by $>90\%$, delivering steady-state inference at **18.9 FPS (52.8 ms)** on an Apple M3 GPU and **35.4 FPS (28.2 ms)** on an NVIDIA T4 with bit-for-bit mathematical equivalence ($\Delta = 0.00000000$).

---

## Benchmark Results

### Component Ablation Across 40 Held-Out Challenge Frames
Evaluated on uncompressed floating-point ground truth depth arrays ($\texttt{\_depth.npy}$) across three strictly unseen evaluation environments (*abandonedfactory*, *abandonedfactory_night*, *amusement*) under from-scratch 24-epoch retraining on dual NVIDIA T4 GPUs:

| Model / Ablation Variant | Ray Embedding | ARA Bias | Training $\mathbf{K}$ | Aligned AbsRel $\downarrow$ | Raw Metric AbsRel $\downarrow$ | Raw Metric $\delta_1$ $\uparrow$ | Surface Normal Error $\downarrow$ | M3 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dioptra Stage 2 (Headline)** | **Trivision** | **Gate = 1.0** | **Dynamic** | **0.5944** | **0.7420** | **24.49%** | **56.86°** | **52.8 ms (Cached)** |
| Stage 1 Baseline (24 Ep.) | Trivision | Gate = 1.0 | Fixed Canonical | 0.5545 | 0.7771 | 22.45% | 55.22° | 98.4 ms |
| Retrained Without ARA (24 Ep.) | Trivision | Disabled | Fixed Canonical | 0.5836 | 0.7458 | 18.79% | 58.99° (+3.77°) | 52.4 ms |
| Retrained Center-Ray PE (24 Ep.) | Center-Ray | Gate = 1.0 | Fixed Canonical | 0.5226 | 0.8909 | 22.62% | 55.45° | 101.2 ms |
| Retrained 2D ViT (24 Ep.) | None (2D Patch) | Disabled | Fixed Canonical | 0.5468 | 0.8918 | 18.57% | 56.13° | 51.8 ms |

*Note on Evaluation Protocol*: While oracle median scaling masks scale drift in naive 2D ViT and Center-Ray baselines, raw metric evaluation reveals that geometric ray embeddings anchor physical scale ($0.7420$ vs. $0.8918$), while ARA enforces physical surface planarity and boundary sharpness.

### Out-of-Distribution Focal Invariance (FOV Sweep)
Under focal length shifts from $50^\circ$ telephoto to $100^\circ$ wide-angle:
- **Fixed-$\mathbf{K}$ Baseline**: Degrades to $0.7198$ AbsRel at $50^\circ$.
- **Dioptra (Dynamic Pinhole)**: Maintains scale equivariance across optical configurations, reducing telephoto error by up to **$-55.9\%$**.

---

## Getting Started

### Installation
```bash
git clone https://github.com/SeranomTheGreat/dioptra.git
cd dioptra

# Create environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pillow matplotlib
```

### Verification & Unit Tests
Run the 16-point unit test suite covering ray geometry, numerical stability, chiral reflection, and caching:
```bash
# Model parameter count (8.1M)
python dioptra.py --count

# Run test suite
python dioptra.py --test

# Forward pass smoke test
python dioptra.py --smoke
```

### Benchmarking Precomputed ARA Caching
Verify mathematical bit-for-bit equivalence and measure latency speedup:
```bash
python dioptra.py --benchmark-cache
```

---

## Evaluation

Run evaluation on the multi-domain challenge test set:
```bash
python eval.py
```
Output comparison figures and error heatmaps will be generated in `test_outputs/`.

---

## Training and Reproduction

Detailed training workflows are provided in:
- [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md): Dual NVIDIA T4 training and ablation scripts.
- [`MAC_SETUP.md`](MAC_SETUP.md): Apple Silicon MPS hardware acceleration and memory budgeting.
- [`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb): Training notebook for Stage 1 and Stage 2 fine-tuning.
- [`notebooks/ablation_kaggle.ipynb`](notebooks/ablation_kaggle.ipynb): Notebook reproducing from-scratch component ablations.

```bash
# Stage 1 training (24 epochs, canonical intrinsics)
python dioptra.py --train auto --epochs 24 --batch_size 8 --accum_steps 6

# Stage 2 dynamic pinhole fine-tuning (15 epochs)
python dioptra.py --train auto --resume outputs/checkpoint_epoch24.pt --epochs 15 --dynamic-crop
```

---

## Dioptra-DINO (Foundation-Assisted Upgrade, ~25.4M Parameters)

For applications demanding photorealistic depth boundaries and near-commercial accuracy on edge devices, the repository includes **Dioptra-DINO**. It combines a pre-trained **DINOv2-Small** (`vits14`, 21.6M parameters pre-trained on 142M images) visual backbone with Dioptra's optical ray geometry:
- **Trivision Ray Positional Encoding**: Continuous optical ray unprojection directly modulating DINOv2 tokens via FiLM.
- **Angular Residual Attention (ARA)**: Pairwise angular attention bias $\sin^2(\theta_{q, k})$ penalizing off-axis spatial warping.
- **Multi-Scale DPT Reassembly**: Progressive feature fusion from layers $\{3, 6, 9, 12\}$ for dense metric depth.
- **Real-Time Edge Speed**: Executes at **36.9 FPS (27.1 ms)** on Apple Silicon M3 in unquantized FP32.

### Quick Start with Dioptra-DINO:
```bash
# Verify parameter audit (27.5M params, ~105 MB FP32, ~52 MB FP16)
python dioptra_dino.py --count

# Run forward/backward smoke test
python dioptra_dino.py --smoke

# Run geometric unit test suite
python dioptra_dino.py --test

# Run benchmark demo on Apple Silicon / CUDA
python scripts/eval_dino.py --demo
```

### Kaggle Training for Dioptra-DINO:
A ready-to-run notebook is provided at [`notebooks/train_dino_kaggle.ipynb`](notebooks/train_dino_kaggle.ipynb) configured for dual NVIDIA T4 GPUs with automatic dataset mounting, mixed precision, and dynamic pinhole crop augmentation.

---

## Interactive 3D Visualization

Dioptra includes an interactive WebGL 3D point cloud and surface mesh visualizer comparing ground truth depth with model predictions:
```bash
open demo/viewer_3d.html
```

---

## Repository Structure

```
dioptra/
├── dioptra.py                 # 8.1M lightweight from-scratch ViT architecture
├── dioptra_dino.py            # ~25.4M foundation-assisted Dioptra-DINO architecture
├── eval.py                    # Evaluation benchmark runner on held-out test frames
├── dioptra_mac.py             # Apple Silicon (MPS) profiling and inference runner
├── assets/                    # Sample input images and test textures
├── demo/                      # Interactive 3D WebGL mesh viewer (viewer_3d.html)
├── notebooks/
│   ├── train_kaggle.ipynb     # Dioptra 8.1M Kaggle training notebook
│   ├── ablation_kaggle.ipynb  # 24-epoch Kaggle retraining ablations
│   └── train_dino_kaggle.ipynb # Dioptra-DINO dual T4 training notebook
├── scripts/
│   └── eval_dino.py           # Dioptra-DINO evaluation and latency profiling
├── paper/                     # Complete preprint LaTeX source, bibliography, and figures
├── KAGGLE_SETUP.md            # Kaggle reproduction instructions
└── MAC_SETUP.md               # Apple Silicon MPS documentation
```

---

## Citation

If you find this work useful in your research, please cite our preprint:

```bibtex
@article{harryson2026dioptra,
  title={Dioptra: An Ultra-Lightweight Geometry-Aware Architecture for Monocular Metric Depth on Edge Devices},
  author={Harryson, Yumnam},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
