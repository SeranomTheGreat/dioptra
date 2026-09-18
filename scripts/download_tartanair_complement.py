#!/usr/bin/env python3
"""
TartanAir Indoor & Level-Ground Complement Downloader (Local Network -> Google Drive)
Target Environments (Strictly excludes Kaggle Warehouse datasets):
  - office (TartanAir 1)
  - office2 (TartanAir 1)
  - hospital (TartanAir 1)
  - Restaurant (TartanAir 2)
  - AbandonedSchool (TartanAir 2)

Saves directly to:
  /Users/krishnakant/Library/CloudStorage/GoogleDrive-harryson424242@gmail.com/My Drive/dioptra_datasets/tartanair_indoor_complement/
"""

import os
import sys
import time
import zipfile
import urllib.request
from typing import Dict, List

GDRIVE_ROOT = "/Users/krishnakant/Library/CloudStorage/GoogleDrive-harryson424242@gmail.com/My Drive/dioptra_datasets/tartanair_indoor_complement"

# Explicit registry with URLs
TARTANAIR_INDOOR_REGISTRY: Dict[str, Dict[str, any]] = {
    "office": {
        "version": "v1",
        "files": {
            "image_left": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/office/Easy/image_left.zip",
            "image_right": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/office/Easy/image_right.zip",
            "depth_left": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/office/Easy/depth_left.zip",
        }
    },
    "office2": {
        "version": "v1",
        "files": {
            "image_left": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/office2/Easy/image_left.zip",
            "image_right": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/office2/Easy/image_right.zip",
            "depth_left": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/office2/Easy/depth_left.zip",
        }
    },
    "hospital": {
        "version": "v1",
        "files": {
            "image_left": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/hospital/Easy/image_left.zip",
            "image_right": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/hospital/Easy/image_right.zip",
            "depth_left": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/hospital/Easy/depth_left.zip",
        }
    },
    "Restaurant": {
        "version": "v2",
        "files": {
            "image_lcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/Restaurant/Data_easy/image_lcam_front.zip",
            "image_rcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/Restaurant/Data_easy/image_rcam_front.zip",
            "depth_lcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/Restaurant/Data_easy/depth_lcam_front.zip",
        }
    },
    "AbandonedSchool": {
        "version": "v2",
        "files": {
            "image_lcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/AbandonedSchool/Data_easy/image_lcam_front.zip",
            "image_rcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/AbandonedSchool/Data_easy/image_rcam_front.zip",
            "depth_lcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/AbandonedSchool/Data_easy/depth_lcam_front.zip",
        }
    }
}

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
                        print(f"      {percent:5.1f}% | {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB @ {speed_mb:.2f} MB/s", end="\r")
                        last_print = now

            print(f"\n    [Complete] Finished {fname}.")
            os.rename(temp_path, dest_path)
            return True
    except Exception as e:
        print(f"\n    [Error] Failed downloading {url}: {e}")
        return False


def main():
    os.makedirs(GDRIVE_ROOT, exist_ok=True)
    print("=" * 80)
    print(" TARTANAIR INDOOR & LEVEL-GROUND COMPLEMENT DOWNLOADER")
    print(f" Target Directory: {GDRIVE_ROOT}")
    print(" (Excludes Kaggle Warehouse datasets: carwelding, abandonedfactory, IndustrialHangar, Supermarket)")
    print("=" * 80)

    for env_name, meta in TARTANAIR_INDOOR_REGISTRY.items():
        print(f"\n>>> Environment: {env_name} ({meta['version']})")
        env_dir = os.path.join(GDRIVE_ROOT, env_name)
        os.makedirs(env_dir, exist_ok=True)

        for modality_key, url in meta["files"].items():
            dest_file = os.path.join(env_dir, f"{modality_key}.zip")
            download_file_with_resume(url, dest_file)

    print("\n" + "=" * 80)
    print(" TartanAir Complement Download Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
