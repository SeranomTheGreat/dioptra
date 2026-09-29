import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Upload ScanNet & TartanAir V2 to Kaggle (Authenticated: yumnamharryson)\n"
        ]
    },
    # Step 1: Mount Google Drive
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 1: Mount Google Drive"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from google.colab import drive\n",
            "import os, sys, json, subprocess, shutil, time\n",
            "\n",
            "drive.mount('/content/drive')\n",
            "DIO = '/content/drive/MyDrive/dioptra_datasets'\n",
            "assert os.path.exists(DIO), f'Missing folder: {DIO}'\n",
            "print('✓ Google Drive mounted. Datasets directory found.')\n"
        ]
    },
    # Step 2: Configure Kaggle API with KGAT Token
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 2: Configure Kaggle API (Auto-Authenticated)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q -U kaggle\n",
            "\n",
            "import os, json\n",
            "\n",
            "KAGGLE_USER = 'yumnamharryson'\n",
            "KGAT_TOKEN  = 'KGAT_c4347b9105f102d13cfd827733121a36'\n",
            "\n",
            "# 1. Set environment variables (highest priority for Kaggle CLI)\n",
            "os.environ['KAGGLE_USERNAME']  = KAGGLE_USER\n",
            "os.environ['KAGGLE_KEY']       = KGAT_TOKEN\n",
            "os.environ['KAGGLE_API_TOKEN'] = KGAT_TOKEN\n",
            "\n",
            "# 2. Configure ~/.kaggle files for both legacy and new token specs\n",
            "kaggle_dir = os.path.expanduser('~/.kaggle')\n",
            "os.makedirs(kaggle_dir, exist_ok=True)\n",
            "\n",
            "# New-style KGAT access token file\n",
            "with open(os.path.join(kaggle_dir, 'access_token'), 'w') as f:\n",
            "    f.write(KGAT_TOKEN.strip())\n",
            "os.chmod(os.path.join(kaggle_dir, 'access_token'), 0o600)\n",
            "\n",
            "# Legacy kaggle.json config\n",
            "with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:\n",
            "    json.dump({'username': KAGGLE_USER, 'key': KGAT_TOKEN}, f)\n",
            "os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)\n",
            "\n",
            "print('✓ Kaggle authentication files configured for yumnamharryson.')\n",
            "\n",
            "# Verify authentication\n",
            "!kaggle datasets list --mine | head -n 10\n"
        ]
    },
    # Step 3: ScanNet Upload
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Upload ScanNet (99.1 GB Direct Archive)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "scannet_dir = f'{DIO}/scannet'\n",
            "scannet_file = f'{scannet_dir}/scannet-frames.zip'\n",
            "assert os.path.exists(scannet_file), f'Missing {scannet_file}'\n",
            "sz_gb = os.path.getsize(scannet_file) / (1024**3)\n",
            "print(f'Found ScanNet file: {sz_gb:.2f} GB')\n",
            "\n",
            "# Clean any lingering aria2 temp file\n",
            "aria_file = f'{scannet_dir}/scannet-frames.zip.aria2'\n",
            "if os.path.exists(aria_file):\n",
            "    os.remove(aria_file)\n",
            "\n",
            "meta = {\n",
            "    'title': 'ScanNet Frames 99GB RGB-D',\n",
            "    'id': 'yumnamharryson/scannet-frames-99gb',\n",
            "    'licenses': [{'name': 'CC0-1.0'}]\n",
            "}\n",
            "with open(f'{scannet_dir}/dataset-metadata.json', 'w') as f:\n",
            "    json.dump(meta, f, indent=2)\n",
            "\n",
            "print('\\nStarting direct upload of ScanNet (99.1 GB) to Kaggle...')\n",
            "!kaggle datasets create -p \"$scannet_dir\" --public\n"
        ]
    },
    # Step 4: TartanAir V2 Indoors (Fast Tar Stream)
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Upload TartanAir V2 Indoors (Fast Tar Stream)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "tartan_path = f'{DIO}/tartanair2_indoors'\n",
            "envs = sorted([d for d in os.listdir(tartan_path) if os.path.isdir(os.path.join(tartan_path, d))])\n",
            "\n",
            "STAGE_DIR = '/tmp/tartan_upload'\n",
            "\n",
            "for env_name in envs:\n",
            "    slug = f'tartanair2-indoors-{env_name.lower()}'.replace('_', '-')\n",
            "    stage_env = os.path.join(STAGE_DIR, slug)\n",
            "    os.makedirs(stage_env, exist_ok=True)\n",
            "    \n",
            "    print(f'\\n======================================================')\n",
            "    print(f'Processing: {env_name} -> yumnamharryson/{slug}')\n",
            "    print(f'======================================================')\n",
            "    \n",
            "    # 1. Check if already uploaded\n",
            "    check = subprocess.run(['kaggle', 'datasets', 'status', f'yumnamharryson/{slug}'], capture_output=True, text=True)\n",
            "    if 'ready' in check.stdout.lower():\n",
            "        print(f'✓ Dataset {slug} already exists and is ready on Kaggle. Skipping.')\n",
            "        continue\n",
            "        \n",
            "    # 2. Write metadata\n",
            "    meta = {\n",
            "        'title': f'TartanAir2 Indoors {env_name}',\n",
            "        'id': f'yumnamharryson/{slug}',\n",
            "        'licenses': [{'name': 'CC0-1.0'}]\n",
            "    }\n",
            "    with open(os.path.join(stage_env, 'dataset-metadata.json'), 'w') as f:\n",
            "        json.dump(meta, f, indent=2)\n",
            "    \n",
            "    # 3. Fast packaging into uncompressed tar\n",
            "    tar_file = os.path.join(stage_env, f'{env_name}.tar')\n",
            "    if not os.path.exists(tar_file):\n",
            "        print(f'  Creating fast archive {env_name}.tar...')\n",
            "        t0 = time.time()\n",
            "        !tar -cf \"$tar_file\" -C \"$tartan_path\" \"$env_name\"\n",
            "        sz_gb = os.path.getsize(tar_file) / (1024**3)\n",
            "        print(f'  ✓ Archive ready: {sz_gb:.2f} GB in {time.time()-t0:.1f}s')\n",
            "    \n",
            "    # 4. Upload single tar file directly\n",
            "    print(f'  Uploading {env_name}.tar to Kaggle...')\n",
            "    !kaggle datasets create -p \"$stage_env\" --public\n",
            "    \n",
            "    # 5. Clean up local tar file immediately\n",
            "    shutil.rmtree(stage_env, ignore_errors=True)\n",
            "    print(f'✓ Cleaned up local archive for {env_name}.')\n",
            "\n",
            "print('\\n✓ All TartanAir V2 indoor environments complete!')\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("notebooks/upload_scannet_and_tartanair2_kaggle.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("✓ Updated notebooks/upload_scannet_and_tartanair2_kaggle.ipynb with hardwired KGAT token")
