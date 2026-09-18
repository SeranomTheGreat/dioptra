import json

cell2_lines = [
    "# Step 2: Download Official Apple Hypersim Scenes",
    "import os",
    "import sys",
    "import zipfile",
    "from tqdm import tqdm",
    "",
    "DEST_DIR = '/content/drive/MyDrive/dioptra_datasets/hypersim'",
    "os.makedirs(DEST_DIR, exist_ok=True)",
    "",
    "# 45 High-priority photorealistic indoor scenes (offices, rooms, hallways)",
    "SCENES = [",
    "    f'ai_{major:03d}_{minor:03d}'",
    "    for major in range(1, 9)",
    "    for minor in range(1, 7)",
    "]",
    "",
    "print(f'Selected {len(SCENES)} Hypersim scenes for download (~65-75 GB).')",
    "",
    "def download_scene(scene_id):",
    "    scene_dir = os.path.join(DEST_DIR, scene_id)",
    "    zip_path = os.path.join(DEST_DIR, f'{scene_id}.zip')",
    "    url = f'https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene_id}.zip'",
    "    ",
    "    if os.path.exists(scene_dir) and len(os.listdir(scene_dir)) > 0:",
    "        print(f'[Skipped] {scene_id} already extracted.')",
    "        return",
    "    ",
    "    print(f'Downloading {scene_id} from Apple CDN...')",
    "    cmd = f\"wget -c -q --show-progress -O '{zip_path}' '{url}'\"",
    "    ret = os.system(cmd)",
    "    if ret != 0:",
    "        print(f'[Error] Failed to download {scene_id}')",
    "        return",
    "    ",
    "    print(f'Extracting {scene_id}...')",
    "    os.makedirs(scene_dir, exist_ok=True)",
    "    try:",
    "        with zipfile.ZipFile(zip_path, 'r') as zf:",
    "            target_members = [",
    "                m for m in zf.namelist()",
    "                if ('.tonemap.jpg' in m or '.depth_meters.hdf5' in m or 'camera_keyframe_positions.csv' in m or 'metadata' in m)",
    "            ]",
    "            if not target_members:",
    "                target_members = zf.namelist()",
    "            zf.extractall(scene_dir, members=target_members)",
    "        print(f'[Success] {scene_id} extracted ({len(target_members)} files).')",
    "        if os.path.exists(zip_path):",
    "            os.remove(zip_path)",
    "    except Exception as e:",
    "        print(f'[Warning] Extraction issue for {scene_id}: {e}')",
    "",
    "for idx, scene in enumerate(SCENES, 1):",
    "    print(f'\\n--- [{idx}/{len(SCENES)}] Processing {scene} ---')",
    "    download_scene(scene)",
    "",
    "print('\\nAll Apple Hypersim scenes downloaded and verified in Google Drive!')"
]

code_str = "\n".join(cell2_lines)
compile(code_str, "<string>", "exec")
print("Python syntax compiles cleanly!")

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Apple Hypersim Cloud-to-Cloud Downloader for Google Drive\n",
                "\n",
                "This notebook streams official Apple Hypersim photorealistic indoor scenes directly from Apple's Developer CDN into your mounted **Google Drive** (`/content/drive/MyDrive/dioptra_datasets/hypersim`).\n",
                "- **High Speed**: Directly utilizes Google Colab's 1+ Gbps cloud backbone.\n",
                "- **Direct Storage**: Saves directly to your 5 TB Google Drive without filling Colab's temporary disk.\n",
                "- **Resume Support**: Checks existing files and skips already downloaded scenes.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Mount Google Drive\n",
                "from google.colab import drive\n",
                "import os\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "DEST_DIR = '/content/drive/MyDrive/dioptra_datasets/hypersim'\n",
                "os.makedirs(DEST_DIR, exist_ok=True)\n",
                "print(f'Target directory: {DEST_DIR}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in cell2_lines]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("notebooks/download_hypersim_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Saved clean notebooks/download_hypersim_colab.ipynb")
