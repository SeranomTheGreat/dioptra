"""Build & push Dioptra-DINO TPU v5e-8 Kaggle kernel.

- Base64-encodes the TPU-optimized dioptra_dino.py (xmp.spawn, bf16, xm.optimizer_step).
- Writes kaggle_kernel_tpuv5e/kernel-metadata.json (TPU enabled, 20 dataset mounts).
- Writes kaggle_kernel_tpuv5e/dioptra_dino_tpu_launcher.ipynb (3 cells: deploy, train, package).
- Pushes via Kaggle CLI.

Usage:
    KAGGLE_USERNAME=yumnamharryson KAGGLE_KEY=... python scripts/build_and_push_tpu_kernel.py [--no-push]
"""

import base64
import json
import os
import subprocess
import sys

DATASET_SOURCES = [
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
    "yumnamharryson/tartanair2-indoors-supermarket",
]


def generate_and_push(do_push: bool = True):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(repo_root, "dioptra_dino.py")
    kernel_dir = os.path.join(repo_root, "kaggle_kernel_tpuv5e")
    os.makedirs(kernel_dir, exist_ok=True)

    # 1. Base64 encode the TPU-optimized dioptra_dino.py
    with open(src_path, "rb") as f:
        code_b64 = base64.b64encode(f.read()).decode("ascii")

    # Sanity: Reason-A fix + XLA symbols must be baked in before push.
    raw = base64.b64decode(code_b64).decode("utf-8", errors="ignore")
    required = [
        'if domain.startswith("tartan") and H == W:',
        "K[1, 2] = K[0, 2]",
        "tpu_static_loss",
        "DistributedSampler",
        "ParallelLoader",
        "xm.optimizer_step",
        'torch.autocast(device_type="xla", dtype=torch.bfloat16)',
        "xm.is_master_ordinal",
        "xm.save",
    ]
    missing = [c for c in required if c not in raw]
    if missing:
        raise RuntimeError(f"TPU-optimized dioptra_dino.py missing: {missing}")
    print(f"Encoded dioptra_dino.py ({len(raw):,} chars) with Reason-A + XLA symbols verified.")

    # 2. Build kernel-metadata.json with TPU enabled
    metadata = {
        "id": "yumnamharryson/dioptra-dino-tpuv5e-training",
        "title": "dioptra-dino-tpuv5e-training",
        "code_file": "dioptra_dino_tpu_launcher.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": True,
        "keywords": ["tpu", "robotics", "depth-estimation"],
        "dataset_sources": DATASET_SOURCES,
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved kernel-metadata.json ({len(DATASET_SOURCES)} dataset mounts, TPU enabled).")

    # 3. Create Notebook launcher cells:
    # Cell 1: Environment check & deployment
    # Cell 2: Launch 8-core distributed training via PyTorch/XLA (inner xmp.spawn)
    # Cell 3: Checkpoint packaging into zip
    notebook = {
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# [1] Setup TPU v5e Environment & Extract Standalone Code\n",
                    "import os, sys, base64\n",
                    "os.environ['PJRT_DEVICE'] = 'TPU'\n",
                    "print('PJRT_DEVICE =', os.environ.get('PJRT_DEVICE'))\n",
                    "try:\n",
                    "    import torch, torch_xla\n",
                    "    print('torch:', torch.__version__)\n",
                    "    print('torch_xla import OK (device query deferred: training workers own the single TPU client).')\n",
                    "except Exception as e:\n",
                    "    print('torch_xla import note (installs on demand):', e)\n",
                    "    os.system('pip install -q torch_xla')\n",
                    "CODE_B64 = \"\"\"" + code_b64 + "\"\"\"\n",
                    "with open('/kaggle/working/dioptra_dino.py', 'wb') as f:\n",
                    "    f.write(base64.b64decode(CODE_B64))\n",
                    "src = open('/kaggle/working/dioptra_dino.py').read()\n",
                    "assert 'tpu_static_loss' in src and 'xm.optimizer_step' in src, 'TPU symbols missing!'\n",
                    "assert 'if domain.startswith(\"tartan\") and H == W:' in src, 'Reason-A fix missing!'\n",
                    "print('TPU v5e Codebase Deployed with Reason-A + XLA verified.')\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# [2] Pre-index datasets ONCE + warm DINO weights (avoids 8-worker FUSE stampede & TPU idle kill)\n",
                    "# Spawned workers reuse /kaggle/working/dataset_index_train.json in ~0.05s instead of\n",
                    "# each walking 100GB+ of FUSE mounts while the TPU sits idle (idle >2h => exit 137).\n",
                    "import os, sys, time\n",
                    "if '/kaggle/working' not in sys.path:\n",
                    "    sys.path.insert(0, '/kaggle/working')\n",
                    "t0 = time.time()\n",
                    "from dioptra_dino import MultiDomainDINODataset, DINOV2_VITS14_URL\n",
                    "ds = MultiDomainDINODataset(root_dirs='auto', split='train', image_size=336, apply_pinhole_aug=True, crop_min=0.35)\n",
                    "print(f'Pre-indexed {len(ds):,} train samples in {time.time()-t0:.1f}s -> cache warmed.')\n",
                    "del ds\n",
                    "for p in ('/kaggle/working/dataset_index_train.json', '/kaggle/working/dataset_index_train.pkl'):\n",
                    "    print(p, 'size:', os.path.getsize(p) if os.path.exists(p) else 'MISSING')\n",
                    "assert any(os.path.exists(p) and os.path.getsize(p) > 1024 for p in ('/kaggle/working/dataset_index_train.json', '/kaggle/working/dataset_index_train.pkl')), 'index cache save failed!'\n",
                    "print('Index cache verified on disk.')\n",
                    "t0 = time.time()\n",
                    "try:\n",
                    "    import torch\n",
                    "    torch.hub.load_state_dict_from_url(DINOV2_VITS14_URL, map_location='cpu')\n",
                    "    print(f'DINOv2 weights cached in {time.time()-t0:.1f}s.')\n",
                    "except Exception as e:\n",
                    "    print('DINO weights warmup note:', e)\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# [3] Launch 8-Core Distributed Training via PyTorch/XLA (inner xmp.spawn, all devices)\n",
                    "import os\n",
                    "os.environ['PJRT_DEVICE'] = 'TPU'\n",
                    "# Foundation checkpoint: direct paths only (NO recursive glob over 100GB+ FUSE mounts).\n",
                    "ckpt_cands = [\n",
                    "    '/kaggle/input/dioptra-dino-epoch5-ckpt/checkpoint_step_latest.pt',\n",
                    "    '/kaggle/input/datasets/yumnamharryson/dioptra-dino-epoch5-ckpt/checkpoint_step_latest.pt',\n",
                    "]\n",
                    "ckpt = next((p for p in ckpt_cands if os.path.exists(p)), 'auto')\n",
                    "print('using ckpt:', ckpt)\n",
                    "print('mounted inputs:', sorted(os.listdir('/kaggle/input'))[:30])\n",
                    "cmd = (\n",
                    "    'python -u /kaggle/working/dioptra_dino.py '\n",
                    "    '--train \"auto\" '\n",
                    "    '--epochs 10 '\n",
                    "    '--image-size 336 '\n",
                    "    '--batch-size 8 '\n",
                    "    '--grad-accum 4 '\n",
                    "    '--lr-head 5e-5 '\n",
                    "    '--freeze-backbone '\n",
                    "    f'--resume \"{ckpt}\" '\n",
                    "    '--finetune '\n",
                    "    '--domain-balanced '\n",
                    "    '--sensor-noise '\n",
                    "    '--subset-fraction 0.15 '\n",
                    "    '--tpu '\n",
                    "    '--use-bfloat16 '\n",
                    "    '--tpu-num-cores 8 '\n",
                    "    '--num-workers 4 '\n",
                    "    '--save-interval 500 '\n",
                    "    '--output-dir /kaggle/working/outputs_dino'\n",
                    ")\n",
                    "print(cmd)\n",
                    "exit_code = os.system(cmd)\n",
                    "assert exit_code == 0, f'TPU training failed with exit code {exit_code}'\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# [4] Package Checkpoints\n",
                    "import os\n",
                    "os.system('cd /kaggle/working && zip -q -r dioptra_dino_tpuv5e_checkpoints.zip outputs_dino/')\n",
                    "print('Packaged checkpoints successfully.')\n",
                ],
            },
        ],
    }

    nb_path = os.path.join(kernel_dir, "dioptra_dino_tpu_launcher.ipynb")
    with open(nb_path, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Saved notebook {nb_path} ({os.path.getsize(nb_path):,} bytes).")

    if not do_push:
        print("Skipping push (--no-push).")
        return kernel_dir

    print("Pushing kernel to Kaggle via REST API...")
    import requests
    TOKEN = os.environ.get("KAGGLE_KEY", "KGAT_2d953a3819e6d87511286f79d2e294cf")
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    nb_text = json.dumps(notebook, indent=2)
    payload = {
        "slug": "yumnamharryson/dioptra-dino-tpuv5e-training",
        "newTitle": "dioptra-dino-tpuv5e-training",
        "text": nb_text,
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": False,
        "enableTpu": True,
        "enableInternet": True,
        "datasetDataSources": DATASET_SOURCES,
        "kernelDataSources": [],
        "competitionDataSources": [],
        "modelDataSources": [],
        "categoryIds": ["tpu", "robotics", "depth-estimation"],
    }
    resp = requests.post("https://www.kaggle.com/api/v1/kernels/push", headers=headers, json=payload)
    print("Push response status:", resp.status_code)
    res = resp.json()
    print("Kernel URL:", res.get("url"))
    print("Version:", res.get("versionNumber"))
    print("Error:", res.get("error"))
    if res.get("hasError"):
        raise RuntimeError(f"TPU kernel push error: {res.get('error')}")
    return kernel_dir


if __name__ == "__main__":
    generate_and_push(do_push="--no-push" not in sys.argv)
