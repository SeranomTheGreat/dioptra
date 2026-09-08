# Training Dioptra on Kaggle (Dual NVIDIA T4 GPUs)

This guide explains how to reproduce Dioptra training and ablation experiments on Kaggle's dual NVIDIA T4 environment.

---

## 1. Datasets and Setup

Training is conducted on the TartanAir benchmark using the cross-environment split:
- **15 Training Environments**: `carwelding`, `endofworld`, `gascola`, `hospital`, `japanesealley`, `neighborhood`, `ocean`, `office`, `office2`, `oldtown`, `seasidetown`, `seasonsforest`, `seasonsforest_winter`, `soulcity`, `westerndesert`.
- **3 Strictly Held-Out Validation Environments**: `abandonedfactory`, `abandonedfactory_night`, `amusement`.

### Attaching Inputs on Kaggle
1. Create a new notebook on Kaggle (Settings $\to$ **Accelerator: GPU T4 x2**).
2. Attach the code dataset containing `dioptra.py` (or upload this repository).
3. Attach the TartanAir validation split dataset (`pandrii000/dasvo-tartanair-rgb-d-validation-split`).

---

## 2. Running Training

You can launch training directly using the provided Jupyter notebook in `notebooks/train_kaggle.ipynb` or via CLI:

```bash
# Stage 1 Baseline Training (24 Epochs, Canonical Intrinsics)
python dioptra.py --train auto --epochs 24 --batch_size 8 --accum_steps 6

# Stage 2 Dynamic Pinhole Fine-Tuning (15 Epochs)
python dioptra.py --train auto --resume outputs/checkpoint_epoch24.pt --epochs 15 --dynamic-crop
```

---

## 3. Running Component Ablations

To retrain the architectural ablation controls from scratch (Epoch 0), run `notebooks/ablation_kaggle.ipynb` or use the `--ablation` flag:

```bash
# 1. Retrain Without ARA (Vanilla Multi-Head Attention)
python dioptra.py --train auto --ablation trivision-no-irer --epochs 24

# 2. Retrain Center-Ray PE (Single ray per token, no corner aperture)
python dioptra.py --train auto --ablation center-ray --epochs 24

# 3. Retrain Canonical 2D ViT (Standard 2D patch coordinate grid)
python dioptra.py --train auto --ablation 2d-vit --epochs 24
```

Checkpoints, evaluation metrics, and logs will be saved to `outputs/`.
