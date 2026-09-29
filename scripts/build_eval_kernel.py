"""
Script to build kaggle_kernel_eval/kernel-metadata.json and dioptra_dino_rigorous_evaluation.ipynb
for rigorous unseen benchmark evaluation on VolsiAI (GPU enabled).
"""

import os
import json
import base64

def build_eval_kernel():
    out_dir = "kaggle_kernel_eval"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Encode dioptra_dino.py
    with open("dioptra_dino.py", "rb") as f:
        dino_bytes = f.read()
    b64_dino = base64.b64encode(dino_bytes).decode("ascii")

    # 2. Encode evaluate_unseen.py
    with open("evaluate_unseen.py", "rb") as f:
        eval_bytes = f.read()
    b64_eval = base64.b64encode(eval_bytes).decode("ascii")

    # 3. Kernel Metadata
    metadata = {
        "id": "yumnamharryson/dioptra-dino-rigorous-eval",
        "title": "dioptra-dino-rigorous-eval",
        "code_file": "dioptra_dino_rigorous_evaluation.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu", "computer-vision"],
        "dataset_sources": [
            "yumnamharryson/dioptra-dino-epoch5-ckpt",
            "volsiai/hypersim-pack",
            "soumikrakshit/nyu-depth-v2",
            "alextitto/kitti-rgb-depth-20k-subset",
            "pandrii000/dasvo-tartanair-rgb-d-validation-split"
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": []
    }

    with open(os.path.join(out_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("✓ Saved kernel-metadata.json for yumnamharryson rigorous evaluation.")

    # 4. Notebook cells
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Dioptra-DINO: Rigorous Unseen Multi-Domain Evaluation Benchmark\n",
                "\n",
                "Comprehensive academic and robotics evaluation of trained **Dioptra-DINO** checkpoints on **UNSEEN** test datasets:\n",
                "1. **NYU Depth v2 Test Split**: 654 real-world indoor RGB-D scenes (Eigen standard crop, 0.5m - 10.0m)\n",
                "2. **KITTI Autonomous Driving Eigen Test Split**: Real-world outdoor LiDAR scenes (Garg crop, 0.001m - 80.0m)\n",
                "3. **TartanAir Validation Trajectories**: Unseen robotics environments (gascola, japanesealley, carwelding)\n",
                "4. **Apple Hypersim Validation Split**: Photorealistic architectural geometries\n",
                "\n",
                "### Metrics Evaluated:\n",
                "- **Error**: AbsRel, SqRel, RMSE (m), RMSE log, SiLog\n",
                "- **Accuracy Thresholds**: $\\delta < 1.25$, $\\delta < 1.25^2$, $\\delta < 1.25^3$\n",
                "- **Boundary Sharpness**: Edge RMSE on physical depth discontinuities\n",
                "- **Surface Planarity**: Normal Mean Angular Error (MAE in deg), Normal Accuracy ($<11.25^\\circ$, $<22.5^\\circ$, $<30^\\circ$)\n",
                "- **Multi-Resolution Generality**: Evaluated at standard $224 \\times 224$ and high-resolution $392 \\times 392$"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [1] Deploy Model Codebase & Evaluation Harness\n",
                "import os, sys, base64\n",
                "\n",
                "DINO_B64 = \"\"\"" + b64_dino + "\"\"\"\n",
                "EVAL_B64 = \"\"\"" + b64_eval + "\"\"\"\n",
                "\n",
                "with open('/kaggle/working/dioptra_dino.py', 'wb') as f:\n",
                "    f.write(base64.b64decode(DINO_B64))\n",
                "with open('/kaggle/working/evaluate_unseen.py', 'wb') as f:\n",
                "    f.write(base64.b64decode(EVAL_B64))\n",
                "\n",
                "if '/kaggle/working' not in sys.path:\n",
                "    sys.path.insert(0, '/kaggle/working')\n",
                "\n",
                "print(f'>>> Deployed dioptra_dino.py ({os.path.getsize(\"/kaggle/working/dioptra_dino.py\"):,} bytes) <<<')\n",
                "print(f'>>> Deployed evaluate_unseen.py ({os.path.getsize(\"/kaggle/working/evaluate_unseen.py\"):,} bytes) <<<')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [2] Audit GPU & Discover Mounted Datasets\n",
                "import torch, glob, os\n",
                "\n",
                "print(f'PyTorch Version : {torch.__version__}')\n",
                "print(f'CUDA Available  : {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'Device Name     : {torch.cuda.get_device_name(0)}')\n",
                "    print(f'Device Count    : {torch.cuda.device_count()}')\n",
                "\n",
                "print('\\nDiscovered Checkpoints in /kaggle/input:')\n",
                "ckpts = sorted(glob.glob('/kaggle/input/**/*.pt', recursive=True))\n",
                "for c in ckpts:\n",
                "    sz_mb = os.path.getsize(c) / (1024 * 1024)\n",
                "    print(f'  {c} ({sz_mb:.1f} MB)')\n",
                "\n",
                "print('\\nDiscovered Input Benchmark Roots:')\n",
                "for root in sorted(glob.glob('/kaggle/input/*')):\n",
                "    print(f'  {root}')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [3] Run Rigorous Evaluation on Epoch 5 Step 37,500 Checkpoint\n",
                "# Evaluates across 224x224 and 392x392 resolutions\n",
                "# Computes all 11 metrics + generates 20 visual panels per domain\n",
                "import os, glob\n",
                "\n",
                "step_ckpts = sorted(glob.glob('/kaggle/input/**/checkpoint_step_latest.pt', recursive=True))\n",
                "target_ckpt = step_ckpts[0] if step_ckpts else '/kaggle/input/dioptra-dino-checkpoint-latest/checkpoint_step_latest.pt'\n",
                "\n",
                "cmd = (\n",
                "    f'python /kaggle/working/evaluate_unseen.py '\n",
                "    f'--checkpoint {target_ckpt} '\n",
                "    f'--output-dir /kaggle/working/outputs_eval_step_latest '\n",
                "    f'--resolutions 224 392 '\n",
                "    f'--max-samples 400 '\n",
                "    f'--save-visuals 15'\n",
                ")\n",
                "print(f'Executing command: {cmd}')\n",
                "exit_code = os.system(cmd)\n",
                "print(f'Evaluation finished with exit code: {exit_code}')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [4] Run Comparative Evaluation on Epoch 4 Complete Checkpoint\n",
                "import os, glob\n",
                "\n",
                "ep4_ckpts = sorted(glob.glob('/kaggle/input/**/dioptra_dino_epoch_4.pt', recursive=True))\n",
                "if ep4_ckpts:\n",
                "    cmd = (\n",
                "        f'python /kaggle/working/evaluate_unseen.py '\n",
                "        f'--checkpoint {ep4_ckpts[0]} '\n",
                "        f'--output-dir /kaggle/working/outputs_eval_epoch_4 '\n",
                "        f'--resolutions 224 392 '\n",
                "        f'--max-samples 400 '\n",
                "        f'--save-visuals 10'\n",
                "    )\n",
                "    print(f'Executing command: {cmd}')\n",
                "    os.system(cmd)\n",
                "else:\n",
                "    print('Epoch 4 checkpoint not found; skipping comparative run.')\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [5] Cross-Checkpoint Comparative Synthesis & Presentation\n",
                "import os, json, glob\n",
                "from IPython.display import display, Markdown\n",
                "\n",
                "summaries = glob.glob('/kaggle/working/**/benchmark_metrics_summary.json', recursive=True)\n",
                "print(f'Found {len(summaries)} evaluation summaries:')\n",
                "\n",
                "all_data = {}\n",
                "for s in sorted(summaries):\n",
                "    run_label = os.path.basename(os.path.dirname(s))\n",
                "    with open(s) as f:\n",
                "        all_data[run_label] = json.load(f)\n",
                "\n",
                "# Construct Markdown Comparison Table\n",
                "md_lines = [\n",
                "    '### Dioptra-DINO Rigorous Benchmark Evaluation Results\\n',\n",
                "    '| Checkpoint Run | Benchmark Domain | Resolution | AbsRel (↓) | SqRel (↓) | RMSE (m) (↓) | SiLog (↓) | δ < 1.25 (↑) | Normal MAE (↓) |',\n",
                "    '| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |'\n",
                "]\n",
                "\n",
                "for run_label, benchmarks in sorted(all_data.items()):\n",
                "    for dom_res, m in sorted(benchmarks.items()):\n",
                "        parts = dom_res.rsplit('_', 1)\n",
                "        dom = parts[0]\n",
                "        res = parts[1] if len(parts) > 1 else 'N/A'\n",
                "        norm_str = f\"{m.get('normal_mae', float('nan')):.1f}°\" if not math.isnan(m.get('normal_mae', float('nan'))) else 'N/A'\n",
                "        md_lines.append(\n",
                "            f\"| {run_label} | {dom.upper()} | {res} | {m['abs_rel']:.4f} | {m['sq_rel']:.4f} | {m['rmse']:.3f}m | {m['silog']:.4f} | {m['delta1']*100:.1f}% | {norm_str} |\"\n",
                "        )\n",
                "\n",
                "display(Markdown('\\n'.join(md_lines)))\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [6] Package Evaluation Report & Qualitative Visual Panels\n",
                "import os, glob\n",
                "from IPython.display import display, FileLink\n",
                "\n",
                "os.system('cd /kaggle/working && zip -q -r dioptra_dino_evaluation_full_package.zip outputs_eval_*/')\n",
                "archive_path = '/kaggle/working/dioptra_dino_evaluation_full_package.zip'\n",
                "if os.path.exists(archive_path):\n",
                "    sz_mb = os.path.getsize(archive_path) / (1024 * 1024)\n",
                "    print(f'>>> Complete evaluation package ready: {sz_mb:.2f} MB <<<')\n",
                "    display(FileLink('dioptra_dino_evaluation_full_package.zip'))\n"
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

    nb_path = os.path.join(out_dir, "dioptra_dino_rigorous_evaluation.ipynb")
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"✓ Generated {nb_path} ({os.path.getsize(nb_path):,} bytes).")

if __name__ == "__main__":
    build_eval_kernel()
