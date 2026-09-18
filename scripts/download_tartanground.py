#!/usr/bin/env python3
"""
TartanGround Ground Robotics Dataset Downloader (Local Network -> Google Drive)
Downloads front stereo image pairs & metric depth from CMU AirLab's TartanGround (IROS 2025):
  - Environments: Office, Hospital, ConstructionSite, OldIndustrialCity, ModularNeighborhoodIntExt
  - Robot Platforms: Data_diff (differential drive AMR), Data_omni (omnidirectional AMR)
  - Modalities: image_lcam_front.zip, image_rcam_front.zip, depth_lcam_front.zip

Saves directly to:
  /Users/krishnakant/Library/CloudStorage/GoogleDrive-harryson424242@gmail.com/My Drive/dioptra_datasets/tartanground/
"""

import os
import sys
import time
import json
import urllib.request
from typing import List, Dict

GDRIVE_ROOT = "/Users/krishnakant/Library/CloudStorage/GoogleDrive-harryson424242@gmail.com/My Drive/dioptra_datasets/tartanground"
HF_API_BASE = "https://huggingface.co/api/datasets/theairlabcmu/TartanGround/tree/main"
HF_RAW_BASE = "https://huggingface.co/datasets/theairlabcmu/TartanGround/resolve/main"

TARGET_ENVIRONMENTS = [
    "Office",
    "Hospital",
    "ConstructionSite",
    "OldIndustrialCity",
    "ModularNeighborhoodIntExt"
]

TARGET_MODALITIES = [
    "image_lcam_front.zip",
    "image_rcam_front.zip",
    "depth_lcam_front.zip"
]


def list_hf_dir(path: str) -> List[Dict]:
    """Fetch file and directory listings from Hugging Face API."""
    url = f"{HF_API_BASE}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  [Error listing {path}]: {e}")
        return []


def download_file_with_resume(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> bool:
    """Download with HTTP Range resume support."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_path = dest_path + ".part"

    initial_pos = 0
    if os.path.exists(dest_path):
        print(f"    [Exists] {os.path.basename(dest_path)} is already complete.")
        return True
    if os.path.exists(temp_path):
        initial_pos = os.path.getsize(temp_path)

    headers = {"User-Agent": "Mozilla/5.0"}
    if initial_pos > 0:
        headers["Range"] = f"bytes={initial_pos}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_range = resp.headers.get("Content-Range")
            if content_range:
                total_size = int(content_range.split("/")[-1])
            else:
                total_size = int(resp.headers.get("Content-Length", 0)) + initial_pos

            mode = "ab" if initial_pos > 0 else "wb"
            fname = os.path.basename(dest_path)
            print(f"    Fetching {fname}: {initial_pos / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({total_size / (1024**3):.2f} GB)")

            downloaded = initial_pos
            start_time = time.time()
            last_print = 0

            with open(temp_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_print > 1.0:
                        elapsed = now - start_time
                        speed_mb = (downloaded - initial_pos) / (1024 * 1024) / max(elapsed, 1e-4)
                        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                        print(f"      {percent:5.1f}% | {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB @ {speed_mb:.2f} MB/s", end="\r", flush=True)
                        last_print = now

            print(f"\n    [Complete] Finished {fname}.", flush=True)
            os.rename(temp_path, dest_path)
            return True
    except Exception as e:
        print(f"\n    [Error] Failed downloading {url}: {e}", flush=True)
        return False


def main():
    os.makedirs(GDRIVE_ROOT, exist_ok=True)
    print("=" * 80, flush=True)
    print(" TARTANGROUND GROUND ROBOTICS DATASET DOWNLOADER", flush=True)
    print(" Targets: AMR Front Stereo Pairs (image_lcam_front, image_rcam_front, depth_lcam_front)", flush=True)
    print(f" Target Directory: {GDRIVE_ROOT}", flush=True)
    print("=" * 80, flush=True)

    for env in TARGET_ENVIRONMENTS:
        print(f"\n>>> Checking TartanGround Environment: {env}", flush=True)
        env_items = list_hf_dir(env)
        platform_dirs = [item["path"] for item in env_items if item["type"] == "directory"]

        for pdir in platform_dirs:
            pname = os.path.basename(pdir)
            traj_items = list_hf_dir(pdir)
            traj_dirs = [item["path"] for item in traj_items if item["type"] == "directory"]

            print(f"  Platform: {pname} ({len(traj_dirs)} trajectories)", flush=True)
            for traj in traj_dirs:
                tname = os.path.basename(traj)
                dest_traj_dir = os.path.join(GDRIVE_ROOT, env, pname, tname)
                os.makedirs(dest_traj_dir, exist_ok=True)

                for modality in TARGET_MODALITIES:
                    mod_url = f"{HF_RAW_BASE}/{traj}/{modality}"
                    dest_file = os.path.join(dest_traj_dir, modality)
                    download_file_with_resume(mod_url, dest_file)

    print("\n" + "=" * 80, flush=True)
    print(" TartanGround Download Complete!", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
