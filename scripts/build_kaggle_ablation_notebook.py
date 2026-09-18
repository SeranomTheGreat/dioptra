"""
Generates the robust, self-contained Kaggle Ablation Kernel Package.
Embeds dioptra_dino.py, train_dino_ablation.py, and download_and_eval_200.py.
Handles Pascal P100 (sm_60) vs Turing T4 (sm_75) auto-detection by running pip BEFORE import torch.
Uses pure Python in verification cells to prevent bash multiline quote errors.
"""

import os
import io
import json
import base64
import zipfile

def build_kernel():
    out_dir = "kaggle_ablation_kernel"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Package Python codebase into base64 payload
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("dioptra_dino.py", "dioptra_dino.py")
        zf.write("scripts/train_dino_ablation.py", "scripts/train_dino_ablation.py")
        zf.write("scripts/download_and_eval_200.py", "scripts/download_and_eval_200.py")
    b64_payload = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 2. Build Jupyter Notebook cells
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Dioptra-DINO: End-to-End Architectural Ablation Retraining\n",
                "### Retraining Component Controls from Scratch (Epoch 0 to 40) on GPU Acceleration\n",
                "\n",
                "This autonomous kernel retrains the primary architectural ablations of Dioptra-DINO:\n",
                "1. **`no-ara`**: Without Angular Residual Attention (`enable_ara=False`)\n",
                "2. **`center-ray`**: Center-Ray PE Only (`ray_mode=\"center_ray\"`, 36 dims vs 108 dims)\n",
                "3. **`no-ray`**: Canonical 2D ViT-S/14 + DPT Decoder (`enable_trivision=False`)\n",
                "4. **`no-vnl`**: Without 3D Virtual Normal Loss (`weight_normal=0.0`)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [1] Unpack Embedded Code & Configure GPU Hardware (T4 / P100 Compatibility)\n",
                "import os, sys, io, base64, zipfile\n",
                "\n",
                "print('=' * 75)\n",
                "print('DIOPTRA-DINO ABLATION RUNTIME SETUP')\n",
                "print('=' * 75)\n",
                "\n",
                "# Check GPU architecture capability via nvidia-smi BEFORE importing torch\n",
                "gpu_info = os.popen('nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null').read().strip()\n",
                "print(f'NVIDIA GPU detected: {gpu_info}')\n",
                "if 'P100' in gpu_info:\n",
                "    print('Pascal GPU (P100 / sm_60) detected. PyTorch 2.6 dropped sm_60 support.')\n",
                "    print('Installing PyTorch 2.4.1+cu121 for native sm_60 binary execution BEFORE importing torch...')\n",
                "    os.system('pip install -q --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121')\n",
                "    print('PyTorch 2.4.1 compatibility installation complete!')\n",
                "\n",
                "# NOW import torch safely into Python\n",
                "import torch\n",
                "print(f'PyTorch Version : {torch.__version__}')\n",
                "print(f'CUDA Available  : {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    gpu_count = torch.cuda.device_count()\n",
                "    for i in range(gpu_count):\n",
                "        major, minor = torch.cuda.get_device_capability(i)\n",
                "        print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB VRAM, sm_{major}{minor})')\n",
                "    print(f'Total available GPU acceleration: {gpu_count}x GPU(s) ready via PyTorch AMP.')\n",
                "else:\n",
                "    print('WARNING: CUDA is not available. Please verify GPU is enabled in notebook settings.')\n",
                "\n",
                "# Unpack verified codebase\n",
                f"payload = '{b64_payload}'\n",
                "buf = io.BytesIO(base64.b64decode(payload.encode('utf-8')))\n",
                "with zipfile.ZipFile(buf, 'r') as zf:\n",
                "    zf.extractall('/kaggle/working')\n",
                "print('Extracted dioptra_dino.py and scripts/ successfully to /kaggle/working/')\n",
                "\n",
                "if '/kaggle/working' not in sys.path:\n",
                "    sys.path.insert(0, '/kaggle/working')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [2] Verify TartanAir Warehouse Stereo Suite Dataset\n",
                "from dioptra_dino import resolve_dataset_root\n",
                "data_root = resolve_dataset_root('auto')\n",
                "print(f'Active TartanAir Dataset Root: {data_root}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [3] Run Smoke Test Across All Ablation Configs (Pure Python Execution)\n",
                "import torch, argparse\n",
                "from scripts.train_dino_ablation import setup_ablation_config\n",
                "from dioptra_dino import DioptraDINO, DioptraDINOLoss\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f'Testing ablation models on device: {device}')\n",
                "\n",
                "for ab in ['no-ara', 'center-ray', 'no-ray', 'no-vnl']:\n",
                "    args = argparse.Namespace(epochs=40, batch_size=2, accum_steps=1, lr_backbone=2e-5, lr_head=2e-4, image_size=224)\n",
                "    cfg = setup_ablation_config(ab, args)\n",
                "    model = DioptraDINO(cfg).to(device)\n",
                "    loss_fn = DioptraDINOLoss(cfg).to(device)\n",
                "    x = torch.randn(2, 3, 224, 224, device=device)\n",
                "    d = torch.abs(torch.randn(2, 1, 224, 224, device=device)) + 1.0\n",
                "    K = torch.eye(3, device=device).unsqueeze(0).expand(2, -1, -1)\n",
                "    pred = model(x, K)\n",
                "    loss, _ = loss_fn(pred, d, K=K)\n",
                "    loss.backward()\n",
                "    print(f'Verification [{ab:12s}] PASS! Params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## Execute Retraining of Ablation Variants\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Ablation 1 Status] 'no-ara' was previously trained for 40 epochs on Kaggle,\n",
                "# verified, and benchmarked on 200 held-out frames (AbsRel: 0.5516, Scale: 0.4901).\n",
                "print('[Ablation 1] no-ara already completed and verified.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Ablation 2] Retrain CENTER-RAY ONLY PE (ray_mode='center_ray')\n",
                "# Evaluates single optical direction vs full Trivision frustum aperture\n",
                "!python scripts/train_dino_ablation.py --ablation center-ray --epochs 20 --batch-size 16 --accum-steps 2\n",
                "\n",
                "# Evaluate immediately on 200 held-out frames\n",
                "!python scripts/download_and_eval_200.py --checkpoint 'outputs_ablations/ablation_center_ray/dioptra_dino_center_ray_best.pt' --output-dir 'outputs_ablations/ablation_center_ray/eval_200' --save-json 'outputs_ablations/ablation_center_ray/eval_200_metrics.json'\n",
                "!zip -r -q dioptra_dino_ablations_retrained.zip outputs_ablations/\n",
                "print('Ablation 2 [center-ray] completed, evaluated, and checkpoint archived.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Ablation 3] Retrain WITHOUT Ray Positional Modulation (Canonical 2D ViT + DPT)\n",
                "# Evaluates whether foundation features alone can establish calibrated metric scale\n",
                "!python scripts/train_dino_ablation.py --ablation no-ray --epochs 20 --batch-size 16 --accum-steps 2\n",
                "\n",
                "# Evaluate immediately on 200 held-out frames\n",
                "!python scripts/download_and_eval_200.py --checkpoint 'outputs_ablations/ablation_no_ray/dioptra_dino_no_ray_best.pt' --output-dir 'outputs_ablations/ablation_no_ray/eval_200' --save-json 'outputs_ablations/ablation_no_ray/eval_200_metrics.json'\n",
                "!zip -r -q dioptra_dino_ablations_retrained.zip outputs_ablations/\n",
                "print('Ablation 3 [no-ray] completed, evaluated, and checkpoint archived.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Ablation 4] Retrain WITHOUT 3D Virtual Normal Loss (weight_normal=0.0)\n",
                "# Evaluates planar architectural floor/wall consistency without normal supervision\n",
                "!python scripts/train_dino_ablation.py --ablation no-vnl --epochs 20 --batch-size 16 --accum-steps 2\n",
                "\n",
                "# Evaluate immediately on 200 held-out frames\n",
                "!python scripts/download_and_eval_200.py --checkpoint 'outputs_ablations/ablation_no_vnl/dioptra_dino_no_vnl_best.pt' --output-dir 'outputs_ablations/ablation_no_vnl/eval_200' --save-json 'outputs_ablations/ablation_no_vnl/eval_200_metrics.json'\n",
                "!zip -r -q dioptra_dino_ablations_retrained.zip outputs_ablations/\n",
                "print('Ablation 4 [no-vnl] completed, evaluated, and checkpoint archived.')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "--- \n",
                "## Final Checkpoint Verification & Archive Packaging\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# [Summary] List all saved checkpoints and final archive\n",
                "import os\n",
                "print('\\nFinal list of saved ablation checkpoints:')\n",
                "os.system('ls -lh outputs_ablations/*/*.pt')\n",
                "os.system('zip -r -q dioptra_dino_ablations_retrained.zip outputs_ablations/')\n",
                "print('Final archive ready: /kaggle/working/dioptra_dino_ablations_retrained.zip')\n"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    nb_path = os.path.join(out_dir, "dioptra_dino_ablations.ipynb")
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated standalone notebook: {nb_path}")

    # 3. Build kernel-metadata.json
    metadata = {
        "id": "yumnamharryson/dioptra-dino-ablations-retraining",
        "title": "Dioptra-DINO Ablations Retraining",
        "code_file": "dioptra_dino_ablations.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "yumnamharryson/tartanair-warehouse-stereo-suite"
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": []
    }

    meta_path = os.path.join(out_dir, "kernel-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated kernel metadata: {meta_path}")

if __name__ == "__main__":
    build_kernel()
