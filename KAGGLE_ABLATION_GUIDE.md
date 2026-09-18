# Dioptra-DINO: Complete Guide to Retraining Ablations from Scratch

This guide details how to train **100% genuine, from-scratch ablation models** (Epoch 0 to 40) on Kaggle GPUs or any multi-GPU environment.

---

## 1. Supported Ablation Variants

Each variant tests an isolated architectural component or loss term:

| Variant Key | Name & Description | Architectural Change | Expected Parameters |
| :--- | :--- | :--- | :---: |
| **`full`** | **Full Headline Baseline** | All components active | **$27.513$\,M** |
| **`no-ara`** | **Without Angular Residual Attention** | `enable_ara=False` (vanilla DPT reassembly) | **$26.329$\,M** |
| **`center-ray`** | **Center-Ray PE Only** | `ray_mode="center_ray"` (1 ray/patch, 36 dims) | **$27.494$\,M** |
| **`no-ray`** | **Canonical 2D ViT + DPT** | `enable_trivision=False` (no ray unprojection) | **$26.103$\,M** |
| **`no-vnl`** | **Without Virtual Normal Loss** | `weight_normal=0.0` (no 3D surface loss) | **$27.513$\,M** |
| **`no-scale-loss`** | **Without Log-Median Scale Loss** | `weight_scale=0.0` (unsupervised scale) | **$27.513$\,M** |
| **`no-dynamic-crop`** | **Without Dynamic Pinhole Crop** | Fixed camera calibration matrix $K$ | **$27.513$\,M** |

---

## 2. Kaggle Setup Instructions

1. **Create a Kaggle Notebook**:
   * Navigate to [kaggle.com/code](https://www.kaggle.com/code).
   * Settings $\to$ **Accelerator: GPU T4 x2** (Dual NVIDIA T4 GPUs, 32 GB total VRAM).
   * Settings $\to$ **Internet: ON**.

2. **Attach Input Datasets** (Right Sidebar $\to$ `+ Add Input`):
   * Search and attach: **`tartanair-warehouse-stereo-suite`** (by `yumnamharryson`).
   * (Optional) Attach this repository code or clone via GitHub.

---

## 3. Launching Retraining (CLI Commands)

Open a terminal or notebook cell in Kaggle:

### A. Retrain Without Angular Residual Attention (ARA)
```bash
python scripts/train_dino_ablation.py --ablation no-ara --epochs 40 --batch-size 8 --accum-steps 4
```
* **Goal**: Verifies the exact contribution of the geometric attention bias ($\sin^2 \theta_{qk}$) to structural edge sharpness and boundary delineation.

### B. Retrain Center-Ray PE Only
```bash
python scripts/train_dino_ablation.py --ablation center-ray --epochs 40 --batch-size 8 --accum-steps 4
```
* **Goal**: Measures the difference between a single optical coordinate ray vs. the full Trivision frustum triplet capturing patch solid angle.

### C. Retrain Without Ray Positional Modulation (Canonical 2D ViT)
```bash
python scripts/train_dino_ablation.py --ablation no-ray --epochs 40 --batch-size 8 --accum-steps 4
```
* **Goal**: Verifies that self-supervised DINOv2 visual features alone cannot establish calibrated physical metric depth without projective camera ray unprojection.

### D. Retrain Without 3D Virtual Normal Loss (VNL)
```bash
python scripts/train_dino_ablation.py --ablation no-vnl --epochs 40 --batch-size 8 --accum-steps 4
```
* **Goal**: Isolates the impact of 3D planar surface regularization on floor and wall surface normal angular errors.

### E. Retrain Without Dynamic Pinhole Crop Augmentation
```bash
python scripts/train_dino_ablation.py --ablation no-dynamic-crop --epochs 40 --batch-size 8 --accum-steps 4
```
* **Goal**: Demonstrates whether fixed-intrinsics training causes the model to overfit to static pixel positions rather than internalizing camera-intrinsic equivariance.

---

## 4. Or Run via Jupyter Notebook

You can open and execute [`notebooks/ablation_dino_kaggle.ipynb`](file:///Users/krishnakant/Downloads/tesseract_kaggle_code_v16/notebooks/ablation_dino_kaggle.ipynb). Each ablation is organized into a single executable cell.

---

## 5. Output Checkpoints and Logs

All models save automatically to:
```
outputs_ablations/
├── ablation_no_ara/
│   ├── checkpoint_epoch_40.pt
│   ├── dioptra_dino_no_ara_best.pt
│   ├── training.log
│   └── ablation_history.json
├── ablation_center_ray/
│   ├── dioptra_dino_center_ray_best.pt
│   └── ...
└── ...
```

Once training is complete, download the checkpoints or run:
```bash
python scripts/download_and_eval_200.py --checkpoint outputs_ablations/ablation_no_ara/dioptra_dino_no_ara_best.pt
```
This produces 100% genuine, independently retrained numbers for your final manuscript table!
