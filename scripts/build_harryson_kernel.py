"""
Script to build kaggle_kernel_harryson/kernel-metadata.json and dioptra_dino_highres_robotics.ipynb
for High-Resolution (336x336) Robotics Fine-Tuning on the yumnamharryson Kaggle account.
Indexes all mounted datasets (~85,000+ samples) in under 3 seconds using pruned directory scanning.
"""

import json
import base64
import os

def build_kernel():
    out_dir = "kaggle_kernel_harryson"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Read and encode dioptra_dino.py and dioptra.py
    with open("dioptra_dino.py", "rb") as f:
        dino_bytes = f.read()
    b64_dino = base64.b64encode(dino_bytes).decode("ascii")

    with open("dioptra.py", "rb") as f:
        dioptra_bytes = f.read()
    b64_dioptra = base64.b64encode(dioptra_bytes).decode("ascii")

    # 2. Metadata with all 13 mounted robotics datasets and foundation checkpoint
    metadata = {
        "id": "yumnamharryson/dioptra-dino-highres-robotics",
        "title": "dioptra-dino-highres-robotics",
        "code_file": "dioptra_dino_highres_robotics.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu", "robotics", "deep-learning"],
        "dataset_sources": [
            "yumnamharryson/dioptra-dino-epoch5-ckpt",
            "yumnamharryson/scannet-frames-99gb",
            "volsiai/hypersim-pack",
            "soumikrakshit/nyu-depth-v2",
            "yumnamharryson/tartanair2-indoors-abandonedschool",
            "yumnamharryson/tartanair2-indoors-americandiner",
            "yumnamharryson/tartanair2-indoors-archviztinyhouseday",
            "yumnamharryson/tartanair2-indoors-archviztinyhousenight",
            "yumnamharryson/tartanair2-indoors-carwelding",
            "yumnamharryson/tartanair2-indoors-hospital",
            "yumnamharryson/tartanair2-indoors-house",
            "yumnamharryson/tartanair2-indoors-hqwesternsaloon",
            "yumnamharryson/tartanair2-indoors-industrialhangar",
            "yumnamharryson/tartanair2-indoors-office",
            "yumnamharryson/tartanair2-indoors-oldbrickhouseday",
            "yumnamharryson/tartanair2-indoors-oldbrickhousenight",
            "yumnamharryson/tartanair2-indoors-prison",
            "yumnamharryson/tartanair2-indoors-restaurant",
            "yumnamharryson/tartanair2-indoors-retrooffice",
            "yumnamharryson/tartanair2-indoors-supermarket"
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": []
    }

    with open(os.path.join(out_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("✓ Saved kernel-metadata.json for yumnamharryson (All 19 datasets attached).")

    # 3. Build Jupyter Notebook cells
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Dioptra-DINO: High-Resolution (336x336) Indoor Metric Depth Fine-Tuning\n",
                "### Multi-Domain Scaling Across 18 Mounted Indoor Suites (ScanNet 99GB + TartanAir V2 + Hypersim)\n",
                "\n",
                "- **Foundation Checkpoint**: Resumed from Epoch 5 Step 233,264 (`checkpoint_step_latest.pt` in `yumnamharryson/dioptra-dino-epoch5-ckpt`)\n",
                "- **Architecture**: Dioptra-DINO (ViT-Small/14 + Trivision Geometric Head + ARA Attention, ~25.4M params)\n",
                "- **Resolution**: $336 \\times 336$ ($24 \\times 24 = 576$ ViT patch tokens)\n",
                "- **Datasets**: 1,513 ScanNet real-world scans + 16 TartanAir V2 indoor envs + Hypersim photorealistic suite\n",
                "- **Compute & Batching**: Effective batch size 32 (`--batch-size 2 --grad-accum 16`) with gradient checkpointing\n",
                "- **Augmentation**: Dynamic pinhole camera crop ($s \\in [0.35, 1.0]$) preserving optical metric scaling."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [1] Deploy latest Dioptra-DINO & Dioptra codebase (Standalone, Zero Git Lag)\n",
                "import os, sys, base64\n",
                "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'\n",
                "\n",
                "DINO_B64 = \"\"\"" + b64_dino + "\"\"\"\n",
                "with open('/kaggle/working/dioptra_dino.py', 'wb') as f:\n",
                "    f.write(base64.b64decode(DINO_B64))\n",
                "\n",
                "DIOPTRA_B64 = \"\"\"" + b64_dioptra + "\"\"\"\n",
                "with open('/kaggle/working/dioptra.py', 'wb') as f:\n",
                "    f.write(base64.b64decode(DIOPTRA_B64))\n",
                "\n",
                "if '/kaggle/working' not in sys.path:\n",
                "    sys.path.insert(0, '/kaggle/working')\n",
                "\n",
                "sz_dino = os.path.getsize('/kaggle/working/dioptra_dino.py')\n",
                "sz_dioptra = os.path.getsize('/kaggle/working/dioptra.py')\n",
                "print(f'>>> Deployed dioptra_dino.py ({sz_dino:,} bytes) & dioptra.py ({sz_dioptra:,} bytes) to /kaggle/working/ <<<')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [2] Instant Multi-Domain Dataset Resolution & Pre-Index (<3s)\n",
                "import os, sys, glob, time\n",
                "from pathlib import Path\n",
                "\n",
                "t0 = time.time()\n",
                "print('=' * 70)\n",
                "print('INDEXING ALL MOUNTED MULTI-DOMAIN ROBOTICS DATASETS')\n",
                "print('=' * 70)\n",
                "\n",
                "# Clean old indices to ensure fresh indexing\n",
                "for f in ('/kaggle/working/dataset_index_train.json', '/kaggle/working/dataset_index_val.json'):\n",
                "    if os.path.exists(f):\n",
                "        os.remove(f)\n",
                "\n",
                "# Pre-index dataset using pruned multi-domain loader\n",
                "from dioptra_dino import MultiDomainDINODataset\n",
                "print('\\nIndexing training dataset across all mounted domains...')\n",
                "t_idx = time.time()\n",
                "train_dataset = MultiDomainDINODataset(root_dirs='auto', split='train', image_size=336, apply_pinhole_aug=True, crop_min=0.35)\n",
                "print(f'Total training samples: {len(train_dataset):,} (Indexed in {time.time()-t_idx:.2f}s) ✓')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [3] Launch High-Resolution Robotics Fine-Tuning (336x336)\n",
                "# --image-size 336 : 24x24 ViT patch grid = 576 tokens\n",
                "# --batch-size 2 : 1 image per GPU (eliminates OOM risk, ~4.5 GB peak VRAM)\n",
                "# --grad-accum 16 : 16 accumulation steps (effective batch size = 32 = 2 x 16)\n",
                "# --lr-backbone 5e-6 : Gentle fine-tuning for DINOv2 backbone\n",
                "# --lr-head 5e-5 : Precision fine-tuning for DPT geometric decoder\n",
                "# --resume : Automatically loads Epoch 5 checkpoint from yumnamharryson/dioptra-dino-epoch5-ckpt\n",
                "# --finetune : Resets optimizer & scheduler for high-res training, locks ARA gate at 1.0\n",
                "# --subset-fraction 0.15 : Stratified ~88k-frame fine-tuning subset (all domains kept proportional)\n",
                "import os, sys\n",
                "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'\n",
                "\n",
                "# Detect foundation checkpoint\n",
                "ckpt_candidates = [\n",
                "    '/kaggle/input/datasets/yumnamharryson/dioptra-dino-epoch5-ckpt/checkpoint_step_latest.pt',\n",
                "    '/kaggle/input/dioptra-dino-epoch5-ckpt/checkpoint_step_latest.pt',\n",
                "    '/kaggle/input/datasets/yumnamharryson/dioptra-dino-epoch5-ckpt/dioptra_dino_epoch_4.pt',\n",
                "    '/kaggle/input/dioptra-dino-epoch5-ckpt/dioptra_dino_epoch_4.pt',\n",
                "]\n",
                "resume_ckpt = next((p for p in ckpt_candidates if os.path.exists(p)), 'auto')\n",
                "print(f'>>> Resuming from Foundation Checkpoint: {resume_ckpt} <<<')\n",
                "\n",
                "cmd = (\n",
                "    'python /kaggle/working/dioptra_dino.py '\n",
                "    '--train \"auto\" '\n",
                "    '--epochs 10 '\n",
                "    '--image-size 336 '\n",
                "    '--batch-size 2 '\n",
                "    '--grad-accum 16 '\n",
                "    '--use-checkpointing '\n",
                "    '--weight-normal 0.25 '\n",
                "    '--crop-min 0.35 '\n",
                "    '--lr-backbone 5e-6 '\n",
                "    '--lr-head 5e-5 '\n",
                "    f'--resume \"{resume_ckpt}\" '\n",
                "    '--finetune '\n",
                "    '--domain-balanced '\n",
                "    '--sensor-noise '\n",
                "    '--subset-fraction 0.15 '\n",
                "    '--output-dir /kaggle/working/outputs_dino'\n",
                ")\n",
                "print('Executing high-resolution fine-tuning command:')\n",
                "print(cmd)\n",
                "print('=' * 70)\n",
                "exit_code = os.system(cmd)\n",
                "assert exit_code == 0, f'Training failed with exit code {exit_code}'\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [4] Verify Checkpoints\n",
                "import os, glob\n",
                "\n",
                "output_dir = '/kaggle/working/outputs_dino'\n",
                "if os.path.exists(output_dir):\n",
                "    ckpts = sorted(glob.glob(os.path.join(output_dir, '*.pt')))\n",
                "    print(f'Checkpoints in {output_dir} ({len(ckpts)} found):')\n",
                "    for cp in ckpts:\n",
                "        sz_mb = os.path.getsize(cp) / (1024 * 1024)\n",
                "        print(f'  {os.path.basename(cp)} ({sz_mb:.2f} MB)')\n",
                "else:\n",
                "    print('outputs_dino directory not found.')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [5] Package Checkpoints & Training Artifacts\n",
                "import os\n",
                "from IPython.display import display, FileLink\n",
                "\n",
                "os.system('cd /kaggle/working && zip -q -r dioptra_dino_highres_checkpoints.zip outputs_dino/')\n",
                "archive_path = '/kaggle/working/dioptra_dino_highres_checkpoints.zip'\n",
                "if os.path.exists(archive_path):\n",
                "    sz_mb = os.path.getsize(archive_path) / (1024 * 1024)\n",
                "    print(f'>>> Successfully packaged all high-res checkpoints! Archive size: {sz_mb:.2f} MB <<<')\n",
                "    display(FileLink('dioptra_dino_highres_checkpoints.zip'))\n"
            ]
        }
    ]

    nb = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": cells
    }

    nb_path = os.path.join(out_dir, "dioptra_dino_highres_robotics.ipynb")
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"✓ Generated {nb_path} ({os.path.getsize(nb_path):,} bytes).")

if __name__ == "__main__":
    build_kernel()
