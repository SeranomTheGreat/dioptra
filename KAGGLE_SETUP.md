# TESSERACT v1 — Kaggle Setup Guide

Single-file trainer for monocular depth estimation on TartanAir, packaged
for Kaggle. This zip contains everything you need; the only other thing to
attach is the dataset.

## What's in this zip

| File | Purpose |
|------|---------|
| `tesseract_v1_patched.py` | The complete trainer (model + data + losses + CLI) — single file, no other imports beyond torch/torchvision/numpy/PIL (all preinstalled on Kaggle) |
| `run_tesseract.ipynb` | Ready-to-import notebook that finds the script and launches training |
| `KAGGLE_SETUP.md` | This guide |

## Step 1 — Upload this zip as a Kaggle dataset

1. Go to https://www.kaggle.com/datasets → **New Dataset** (top right).
2. Drag `tesseract_kaggle_code_v16.zip` in (Kaggle unpacks it server-side).
3. Title it e.g. **tesseract-v16** → the notebook below finds the script
   automatically, whatever you name it.
4. Click **Create**. Wait for processing to finish.

> **Which zip is this?**
> - **v17** is the major academic redesign addressing all peer-review / CVPR rigor requirements:
>   - **True Metric Depth Supervision**: Unnormalized raw depth Log-L1 + raw depth SiLog + batch log-median scale consistency loss ($L_{\text{scale}} = |\log(\text{med}(\hat{D})) - \log(\text{med}(D))|$). The model learns absolute physical scale (metres).
>   - **Strict Cross-Environment Split**: Zero-shot environment partitioning (15 train environments, 3 completely held-out validation environments: `abandonedfactory`, `abandonedfactory_night`, `amusement`). Eliminates intra-environment visual feature leakage.
>   - **Physical Chiral Ray Reflection**: Data augmentations track horizontal flips (`is_flipped=True`), reflecting corner rays to maintain true chirality ($c_1$ top-right, $c_2$ bottom-left).
>   - **Dual Metric Reporting**: Reports both pure unaligned metric depth numbers (for robotics/navigation) and median-scaled Eigen numbers (for benchmark comparisons).
>   - **80m Evaluation Cap**: Filters sky and non-finite outliers ($0.2\text{m} \le D \le 80.0\text{m}$) matching established outdoor benchmark protocols.
>   - **Ablation Suite**: Built-in CLI flag `--ablation [full|trivision-no-irer|center-ray|2d-vit]` allows running paper ablations with a single argument.
>   - **14 Unit Tests**: Full verification suite covering geometry, numerical stability, gradients, chirality, and split isolation.
> - **v16** fixed the dual-GPU crash `RuntimeError: lazy wrapper should
>   be called at most once` (PyTorch's `torch.linalg` backend is not
>   thread-safe under `nn.DataParallel`; the ray computation uses an
>   analytic matrix inverse instead — no linalg library calls).

## Step 2 — Create the notebook and attach BOTH datasets

1. https://www.kaggle.com/code → **New Notebook**.
2. Settings → **Accelerator**: GPU T4 ×2 (or P100).
3. In the right sidebar **Input**, click **+ Add Input** twice:
   - **Your Datasets → tesseract-v16** (the zip you just uploaded)
   - Search → **DASVO TartanAir RGB-D Validation Split**
     (`pandrii000/dasvo-tartanair-rgb-d-validation-split`)

Both must be attached at the same time — the trainer reads its code from
your code dataset and trains on the TartanAir dataset. Optionally import
`run_tesseract.ipynb` (File → Import Notebook) — or just use the
commands below.

> **Where things mount** (Kaggle changed this!):
> - older notebooks: `/kaggle/input/<dataset-slug>/…`
> - current notebooks: `/kaggle/input/datasets/<owner>/<slug>/…`
>
> `--train auto` scans recursively and handles both, so you normally
> don't need to care. The run log prints every mounted input it sees.

## Step 3 — Run

The script auto-detects the dataset and its own location. Easiest: import
`run_tesseract.ipynb` (File → Import Notebook) and run its cells — it
globs for the script in any mount layout and pre-checks that both inputs
are attached.

Or from any notebook cell (layout-agnostic one-liner):

```bash
!python $(find /kaggle/input -name tesseract_v1_patched.py | head -1) --train auto
```

`--train auto` recursively scans `/kaggle/input`, finds the TartanAir
mount in either layout, and resolves the real root. Equivalently, pass
the mount dir or the archive directly — both of these work on current
Kaggle notebooks:

```bash
!python /kaggle/input/datasets/volsiai/tesseract-v16/tesseract_v1_patched.py \
    --train /kaggle/input/datasets/pandrii000/dasvo-tartanair-rgb-d-validation-split

!python /kaggle/input/datasets/volsiai/tesseract-v16/tesseract_v1_patched.py \
    --train /kaggle/input/datasets/pandrii000/dasvo-tartanair-rgb-d-validation-split/tartanair
```

(adjust `volsiai/tesseract-v16` to your username/dataset slug, or just
use the `find` one-liner / the notebook)

Useful flags:

| Flag | Effect |
|------|--------|
| `--ablation MODE` | Architecture ablation: `full` (default), `trivision-no-irer`, `center-ray`, `2d-vit` |
| `--split-mode MODE` | Dataset split: `cross_env` (default, zero-shot 15 train / 3 val) or `cross_traj` |
| `--epochs N` | Override total epochs (default: 30) |
| `--batch-size N` | Override batch size (default 8; dual T4 handles it) |
| `--lr F` | Override base learning rate |
| `--num-workers N` | DataLoader workers (default: min(8, CPUs)) |
| `--tensorboard` | Log scalars to TensorBoard |
| `--seed N` | Reproducibility seed |
| `--output-dir PATH` | Defaults to `/kaggle/working/outputs` on Kaggle |

## Step 4 — Running Full Architectural Retraining Ablations (Option B)

To run the complete from-scratch retraining ablations for the paper:
1. Import **`run_ablation_kaggle.ipynb`** into Kaggle (File → Import Notebook).
2. Attach both inputs:
   - Your code dataset (`tesseract_v17_kaggle.zip`)
   - `pandrii000/dasvo-tartanair-rgb-d-validation-split`
3. Each ablation runs from Epoch 0 across 24 epochs and saves its own checkpoint:

```bash
# 1. Without Angular Residual Attention (Trivision PE + Vanilla Attention)
!python -u "$SCRIPT_PATH" --train auto --ablation trivision-no-irer --output-dir /kaggle/working/outputs_no_irer

# 2. Single Center-Ray PE (No Aperture Triplets)
!python -u "$SCRIPT_PATH" --train auto --ablation center-ray --output-dir /kaggle/working/outputs_center_ray

# 3. Canonical 2D Vision Transformer (No 3D Ray PE, No ARA)
!python -u "$SCRIPT_PATH" --train auto --ablation 2d-vit --output-dir /kaggle/working/outputs_2d_vit
```

Cell [5] in the notebook automatically evaluates all finished checkpoints on the held-out validation set and formats the final retrained Table 6 for the paper!
Cell [6] packages `/kaggle/working/tesseract_ablations.zip` for direct download.

## How the dataset is structured (verified against the actual upload)

```
<kaggle mount>/          ← /kaggle/input/<slug> (older) OR
                          ← /kaggle/input/datasets/pandrii000/<slug> (current)
└── tartanair/
    ├── {environment}/            ← 18 envs (abandonedfactory, gascola, …)
    │   ├── Easy/                 ← difficulty (Easy + Hard only in this split)
    │   │   └── P0xx/             ← trajectory folder
    │   │       ├── image_left/   ← 000000_left.png, 000001_left.png, …
    │   │       ├── depth_left/   ← 000000_left_depth.npy (float32, metres)
    │   │       └── pose_left.txt ← ground-truth poses (not used by training)
    │   └── Hard/
    └── …
```

- 32 trajectories / 18 environments / 26,650 RGB-D frames, 640×480.
- Intrinsics: the dataset ships no `cam_left.json`; the script's default
  intrinsics for 640×480 (fx=fy=320, cx=320, cy=240) exactly match the
  dataset's documented calibration — nothing to configure.
- The loader pairs `000000_left.png ↔ 000000_left_depth.npy` automatically
  (it also still understands the official `000000_og.png ↔ 000000.npy`
  TartanAir naming, so other TartanAir datasets work too).
- Train/val is split by trajectory hash (~80/20 → ≈22,084 train /
  ≈4,566 val frames on this dataset; the same environment's Easy and Hard
  runs always land in the same split — no leakage).
- **If Kaggle mounts the dataset as a raw `archive.zip` instead of the
  extracted tree** (large-dataset behaviour), the script detects it and
  reads frames directly out of the zip — no extraction, no setup. Both
  cases are handled by `--train auto`.

## Outputs

Everything lands in `/kaggle/working/outputs/` (the only writable dir):

```
/kaggle/working/outputs/
├── checkpoint.pt     ← resume with --resume
├── tesseract.log     ← full training log
├── viz/              ← depth-prediction visualizations (PNG)
├── per_scene.txt     ← per-scene validation metrics
└── eval_viz/         ← (only with --evaluate)
```

Click **Save Version → Save & Run All** to persist outputs with the
notebook version; or download files directly from the Output panel.

## Training length & the 9-hour limit

The script runs a 13-epoch schedule with an internal **8.5 h budget**
(Kaggle's 9 h limit minus a buffer). When the budget is exhausted it
stops gracefully and saves the checkpoint. To continue in a new session:

1. Save the notebook version (outputs include `checkpoint.pt`), or
   download `checkpoint.pt` and upload it as a small dataset
   (e.g. `tesseract-ckpt`).
2. In the next session run (the notebook's last cell auto-finds the
   checkpoint wherever it mounted):

   ```bash
   !python $(find /kaggle/input -name tesseract_v1_patched.py | head -1) \
       --train auto \
       --resume $(find /kaggle/input -name checkpoint.pt | head -1) \
       --epochs 26
   ```

   (raise `--epochs` so there is something left to train; the scheduler
   continues from the resumed epoch).

## Evaluating a checkpoint

```bash
!python $(find /kaggle/input -name tesseract_v1_patched.py | head -1) \
    --evaluate $(find /kaggle/input -name checkpoint.pt | head -1) \
    --train auto
```

Prints validation metrics (abs_rel, sq_rel, rmse, δ<1.25 …), the top-5
best/worst scenes, and writes `eval_viz/` depth renders.

## Stage 2: Camera-Invariant Dynamic Pinhole Fine-Tuning

TartanAir frames share a fixed virtual camera ($f_x = f_y = 320$, $\text{FOV} = 90^\circ$). Under fixed $K$, ray unprojection reduces to a static coordinate embedding.
**Stage 2 Fine-Tuning** trains the model under **Dynamic Pinhole Intrinsics Crop Augmentation**:
- Window size $L \in [0.55 \cdot S_{\max}, S_{\max}]$ at random crop offsets $(x_0, y_0)$
- Dynamic focal lengths $f_x \in [140, 260]$ px (effective horizontal FOV $\sim 45^\circ$ to $74^\circ$)
- Preserves physical metric depth $Z$ in metres while forcing the network to dynamically decode variable camera intrinsics $K$

**Launch Stage 2 Fine-Tuning on Kaggle (Dual T4 GPUs)**:
```bash
!python $(find /kaggle/input -name tesseract_v1_patched.py | head -1) \
    --train auto \
    --finetune $(find /kaggle/input -name checkpoint.pt | head -1) \
    --epochs 15 \
    --batch-size 8 \
    --finetune-lr 5e-5 \
    --pinhole-aug \
    --pinhole-min-scale 0.55 \
    --output-dir /kaggle/working/outputs
```

## Multi-FOV Synthetic Camera Benchmark

Evaluate the model across a synthetic focal length / FOV sweep ($50^\circ$ to $100^\circ$):
```bash
!python $(find /kaggle/input -name tesseract_v1_patched.py | head -1) \
    --eval-multi-fov /kaggle/input/datasets/pandrii000/dasvo-tartanair-rgb-d-validation-split/tartanair/abandonedfactory/Easy/P000/image_left/000000_left.png \
    --resume /kaggle/working/outputs/checkpoint.pt \
    --output-dir /kaggle/working/outputs
```
Prints an ASCII metric table across FOVs and generates a multi-panel visualizer (`*_multi_fov_sweep.png`) showing depth maps and 3D camera ray unprojections.

## Sanity checks (optional)

```bash
!python $(find /kaggle/input -name tesseract_v1_patched.py | head -1) --config   # config + param count (8.3M)
!python $(find /kaggle/input -name tesseract_v1_patched.py | head -1) --smoke    # forward-pass test
!python $(find /kaggle/input -name tesseract_v1_patched.py | head -1) --test     # 14 unit tests
```

## Troubleshooting

- **"RuntimeError: lazy wrapper should be called at most once"** — a
  PyTorch bug where `torch.linalg` (solve/inv/pinv) crashes when two
  DataParallel replica threads call it concurrently (pytorch/pytorch#90613).
  **Fixed in v16**: the model computes rays with an analytic 3x3 inverse
  (pure elementwise math, no linalg library), plus a main-thread linalg
  warm-up before the replicas spawn. If you see this error, you are
  running an older zip — update to v16.
- **"Could not resolve a TartanAir dataset from: auto"** — the error now
  lists every mounted input the scan saw. Read that line:
  - If it shows only your code dataset (e.g. `datasets/volsiai/tesseract-v16`),
    the TartanAir dataset is **not attached** → right sidebar → Input →
    **+ Add Input** → search `dasvo-tartanair-rgb-d-validation-split`, then
    re-run.
  - If it shows the TartanAir mount but still fails, inspect it:
    `!ls /kaggle/input/datasets/pandrii000/dasvo-tartanair-rgb-d-validation-split`
    and pass the inner directory explicitly via `--train`.
- **CUDA out of memory** — the script auto-halves the batch size on OOM
  and rebuilds the loader; if it still fails, pass `--batch-size 4`.
- **Nothing written?** — `/kaggle/input` is read-only; outputs always go
  to `/kaggle/working` (the script enforces this automatically).
- **Old vs new mount paths** — if `/kaggle/input` contains a `datasets/`
  folder, your notebook uses the new namespaced layout
  (`/kaggle/input/datasets/<owner>/<slug>/`); everything above handles
  both. `!ls /kaggle/input` shows what you actually have.
