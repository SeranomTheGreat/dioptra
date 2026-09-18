"""
Script to build kaggle_kernel_volsiai/kernel-metadata.json and notebookff7e204f6e.ipynb
with all 13 dataset sources and embedded multi-domain Dioptra-DINO code.
"""

import json
import base64
import os

def build_kernel():
    out_dir = "kaggle_kernel_volsiai"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Read and encode dioptra_dino.py
    with open("dioptra_dino.py", "rb") as f:
        dino_bytes = f.read()
    b64_code = base64.b64encode(dino_bytes).decode("ascii")

    # 2. Metadata with all 13 datasets
    metadata = {
        "id": "volsiai/notebookff7e204f6e",
        "id_no": 133507982,
        "title": "notebookff7e204f6e",
        "code_file": "notebookff7e204f6e.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [
            "yumnamharryson/tartanair-warehouse-stereo-suite",
            "yumnamharryson/tartanground-office-stereo-depth",
            "yumnamharryson/tartanground-oldindustrialcity-stereo-depth",
            "yumnamharryson/tartanground-hospital-stereo-depth",
            "yumnamharryson/tartanair-indoors-hospital",
            "yumnamharryson/tartanair-indoors-abandonedschool",
            "yumnamharryson/tartanair-indoors-restaurant",
            "yumnamharryson/tartanair-indoors-office2",
            "volsiai/hypersim-pack",
            "pandrii000/dasvo-tartanair-rgb-d-validation-split",
            "soumikrakshit/nyu-depth-v2",
            "alextitto/kitti-rgb-depth-20k-subset",
            "volsiai/dioptra-dino-epoch-13",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": []
    }

    with open(os.path.join(out_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("✓ Saved kernel-metadata.json with 13 dataset sources.")

    # 3. Build Jupyter Notebook cells
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Dioptra-DINO: Multi-Domain Large-Scale Metric Depth Training\n",
                "\n",
                "Training **Dioptra-DINO** (~27.5M params) on dual NVIDIA T4 GPUs across an aggregated multi-domain corpus:\n",
                "- **TartanAir & TartanGround AMR**: 8 public stereo suites (Warehouse, Hospital, Office, OldIndustrialCity, Restaurant, School)\n",
                "- **Apple Hypersim**: 191 indoor environments (`volsiai/hypersim-pack`)\n",
                "- **NYU-Depth-v2**: Official RGB-D indoor benchmark (`soumikrakshit/nyu-depth-v2`)\n",
                "- **KITTI**: Eigen metric depth benchmark (`alextitto/kitti-rgb-depth-20k-subset`)\n",
                "- **DASVO TartanAir**: Benchmark validation split (`pandrii000/dasvo-tartanair-rgb-d-validation-split`)\n",
                "- **Automatic Resume**: Preemption-resistant state restoration from attached checkpoint (`volsiai/dioptra-dino-epoch-13`) or local checkpoints"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [1] Deploy latest Multi-Domain Dioptra-DINO codebase\n",
                "import os, sys, base64\n",
                "\n",
                "CODE_B64 = \"\"\"" + b64_code + "\"\"\"\n",
                "with open('/kaggle/working/dioptra_dino.py', 'wb') as f:\n",
                "    f.write(base64.b64decode(CODE_B64))\n",
                "\n",
                "if '/kaggle/working' not in sys.path:\n",
                "    sys.path.insert(0, '/kaggle/working')\n",
                "\n",
                "sz = os.path.getsize('/kaggle/working/dioptra_dino.py')\n",
                "print(f'>>> Deployed multi-domain dioptra_dino.py to /kaggle/working/ ({sz:,} bytes) <<<')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [2] Multi-Domain Dataset Discovery & Audit\n",
                "import os, sys, glob\n",
                "from dioptra_dino import resolve_all_dataset_roots, MultiDomainDINODataset\n",
                "\n",
                "print('Scanning all mounted input datasets in /kaggle/input/...')\n",
                "roots = resolve_all_dataset_roots('auto')\n",
                "print(f'Discovered {len(roots)} multi-domain dataset roots:')\n",
                "for r, dom in roots:\n",
                "    print(f'  [{dom.upper():8s}] {r}')\n",
                "\n",
                "print('\\nIndexing training dataset across all domains...')\n",
                "train_dataset = MultiDomainDINODataset(root_dirs='auto', split='train', image_size=224, apply_pinhole_aug=True, crop_min=0.35)\n",
                "print(f'Total training samples: {len(train_dataset):,}')\n",
                "\n",
                "dom_counts = {}\n",
                "for s in train_dataset.samples:\n",
                "    dom_counts[s[2]] = dom_counts.get(s[2], 0) + 1\n",
                "print('\\nTraining domain breakdown:')\n",
                "for dom, cnt in sorted(dom_counts.items()):\n",
                "    print(f'  {dom.upper():8s}: {cnt:,} samples')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [3] Launch Training with Freely Resumable Engine\n",
                "# --train auto : Auto-scans all 12 mounted datasets across TartanAir, Hypersim, NYUv2, and KITTI\n",
                "# --resume auto : Automatically restores weights, optimizer, scheduler, scaler from /kaggle/input or local\n",
                "# --weight-normal 0.25 : 3D Virtual Normal Loss enforcing surface planarity & boundary sharpness\n",
                "# --crop-min 0.35 : Wide optical zoom crop for camera-intrinsic equivariance\n",
                "# --batch-size 8 : Batch size per GPU (DataParallel multi-GPU acceleration across 2x T4s)\n",
                "import os\n",
                "\n",
                "cmd = (\n",
                "    'python /kaggle/working/dioptra_dino.py '\n",
                "    '--train auto '\n",
                "    '--epochs 40 '\n",
                "    '--batch-size 8 '\n",
                "    '--weight-normal 0.25 '\n",
                "    '--crop-min 0.35 '\n",
                "    '--lr-backbone 2e-5 '\n",
                "    '--lr-head 2e-4 '\n",
                "    '--resume auto '\n",
                "    '--output-dir /kaggle/working/outputs_dino'\n",
                ")\n",
                "print('Executing training command:')\n",
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
                "os.system('cd /kaggle/working && zip -q -r dioptra_dino_multidomain_checkpoints.zip outputs_dino/')\n",
                "archive_path = '/kaggle/working/dioptra_dino_multidomain_checkpoints.zip'\n",
                "if os.path.exists(archive_path):\n",
                "    sz_mb = os.path.getsize(archive_path) / (1024 * 1024)\n",
                "    print(f'>>> Successfully packaged all checkpoints! Archive size: {sz_mb:.2f} MB <<<')\n",
                "    display(FileLink('dioptra_dino_multidomain_checkpoints.zip'))\n"
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

    nb_path = os.path.join(out_dir, "notebookff7e204f6e.ipynb")
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"✓ Generated {nb_path} ({os.path.getsize(nb_path):,} bytes).")

if __name__ == "__main__":
    build_kernel()
