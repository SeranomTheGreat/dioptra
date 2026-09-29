#!/usr/bin/env python3
"""
InteriorNet Official Dataset Downloader & Extractor (Python 3)
Downloads official InteriorNet HD7 sequence packages (ETH Zurich / Imperial College distribution):
  - cam0/data/*.png: Original full color RGB frames
  - depth0/data/*.png: Original 16-bit millimeter metric depth maps
  - cam0.render, cam0.txt: 4x4 camera trajectory poses & intrinsics
  - velocity_angular.csv, velocity_linear.csv: Ground truth IMU measurements

Supports:
  - High-speed direct streaming via gdown
  - In-place extraction to target directory
  - Automatic cleanup of large temporary archives
"""

import os
import sys
import argparse
import zipfile
import tarfile
from typing import Dict, List, Optional

# Verified direct Google Drive file IDs for InteriorNet HD7 sequences
VERIFIED_INTERIORNET_HD7: Dict[str, str] = {
    "3FO4MMTWI01K_Guest_room": "1FYiAvuqW2BwlcWmJanxWqfl-FhiLj2om",
    "3FO4MMTVR3VT_Guest_room": "1OKabst84M46VO4YSieTWDQidUUqG1Xkl",
    "3FO4MMN8Q7SM_Guest_room": "1m2tei_coUUiDYECmxgL53B20JhfvtRFb",
    "3FO4MMLDUKDJ_Living_room": "1mwy2BV4UV78Nw9bzm0EBjQjTagJwKk5F",
    "3FO4MMLDUKDJ_Dining_room": "1dRF0zuCZBdF1HOz3XHSnYlaKoS_SplgI",
    "3FO4MMLD2AN3_Living_room": "18mvQAQVRWl7KGwiPNka59KsroQOohlFX",
    "3FO4MMLCRTM9_Living_room": "1A5mkE0mI_kn0oQDbflRO7ZZMUAdRGxlS",
    "3FO4MMKKYVFQ_Living_room": "1UIsLlUc9qGI54eLYHAzJe3GvE4zRreOM",
    "3FO4MMJXJS4X_Living_room": "1r6jhV2gC1f7yp6LognUA9xDASrXlpv-z",
    "3FO4MMJVI5DI_Living_room": "11-tCFdalpPAlvbV491ya4zuoTx1oRX_V",
    "3FO4MMJSDU7T_Bedroom": "1h66Y37W9IlM9L8TEWmy1JJL-RoCD3b8d",
    "3FO4MMJO698O_Bedroom": "1WoZVZ7NG1X_Mv6kWDN3KBYFVAG2MXIxj",
}


def download_and_extract_interiornet(
    seq_name: str,
    drive_id_or_url: str,
    dest_dir: str,
    clean_zip: bool = True,
) -> bool:
    """Download an InteriorNet sequence package and unpack full RGB-D data."""
    try:
        import gdown
    except ImportError:
        print("[Error] 'gdown' is required. Install via: pip install gdown")
        return False

    target_dir = os.path.join(dest_dir, seq_name)
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 5:
        print(f"[Exists] {seq_name} already extracted in {target_dir}.")
        return True

    os.makedirs(target_dir, exist_ok=True)
    tmp_archive = os.path.join(dest_dir, f"{seq_name}.zip")

    print(f"\n{'='*70}")
    print(f"STREAMING INTERIORNET SEQUENCE: {seq_name}")
    print(f"Target Directory: {target_dir}")
    print(f"{'='*70}")

    if drive_id_or_url.startswith("http"):
        gdown.download(url=drive_id_or_url, output=tmp_archive, quiet=False, fuzzy=True)
    else:
        gdown.download(id=drive_id_or_url, output=tmp_archive, quiet=False, fuzzy=True)

    if os.path.exists(tmp_archive) and os.path.getsize(tmp_archive) > 1024:
        print(f"[InteriorNet] Extracting archive {os.path.basename(tmp_archive)} to {target_dir}...")
        if zipfile.is_zipfile(tmp_archive):
            with zipfile.ZipFile(tmp_archive, "r") as zf:
                zf.extractall(target_dir)
        elif tarfile.is_tarfile(tmp_archive):
            with tarfile.open(tmp_archive, "r:*") as tf:
                tf.extractall(target_dir)
        else:
            print(f"[Warning] Unknown archive format: {tmp_archive}")
            return False

        if clean_zip:
            os.remove(tmp_archive)
            print(f"Cleaned up {os.path.basename(tmp_archive)} to conserve storage.")

        print(f"✓ Successfully extracted {seq_name} into {target_dir}")
        return True
    else:
        print(f"[Error] Failed to download {seq_name} from {drive_id_or_url}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Official InteriorNet HD7 Sequence Downloader")
    parser.add_argument("--scene", type=str, default="3FO4MMTWI01K_Guest_room", help="Sequence name, comma-separated list, or 'all'")
    parser.add_argument("--output-dir", type=str, default="data/interiornet", help="Destination directory for unpacked sequences")
    parser.add_argument("--clean-archive", action="store_true", default=True, help="Delete temporary zip archives after extraction")
    parser.add_argument("--keep-archive", dest="clean_archive", action="store_false", help="Retain downloaded zip archives")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scene.lower() == "all":
        targets = list(VERIFIED_INTERIORNET_HD7.items())
    else:
        targets = []
        for s in args.scene.split(","):
            s = s.strip()
            if s in VERIFIED_INTERIORNET_HD7:
                targets.append((s, VERIFIED_INTERIORNET_HD7[s]))
            else:
                print(f"[Notice] Scene '{s}' not in default presets; checking if it's a direct URL or ID...")
                targets.append((s, s))

    for name, link in targets:
        download_and_extract_interiornet(
            seq_name=name,
            drive_id_or_url=link,
            dest_dir=args.output_dir,
            clean_zip=args.clean_archive,
        )


if __name__ == "__main__":
    main()
