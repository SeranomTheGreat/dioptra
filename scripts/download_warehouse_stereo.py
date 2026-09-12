import os
import sys
import time
import argparse
import urllib.request
from typing import Dict, List, Tuple

# Exact verified URLs and byte sizes for Warehouse/Factory Stereo Suite
DATASET_REGISTRY: Dict[str, Dict[str, any]] = {
    "carwelding": {
        "dataset_version": "tartanair1",
        "split": "Easy",
        "description": "Industrial automotive assembly floor with robot welding arms and safety bays",
        "files": {
            "image_left": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/carwelding/Easy/image_left.zip",
            "image_right": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/carwelding/Easy/image_right.zip",
            "depth_left": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/carwelding/Easy/depth_left.zip",
        },
        "approx_gb": 4.48,
    },
    "abandonedfactory": {
        "dataset_version": "tartanair1",
        "split": "Easy",
        "description": "High-bay industrial warehouse, concrete floors, steel columns, and catwalks",
        "files": {
            "image_left": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/abandonedfactory/Easy/image_left.zip",
            "image_right": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/abandonedfactory/Easy/image_right.zip",
            "depth_left": "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/abandonedfactory/Easy/depth_left.zip",
        },
        "approx_gb": 15.96,
    },
    "IndustrialHangar": {
        "dataset_version": "tartanair2",
        "split": "Data_easy",
        "description": "Large-scale open warehouse / hangar, steel structural trusses, industrial lighting",
        "files": {
            "image_lcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/IndustrialHangar/Data_easy/image_lcam_front.zip",
            "image_rcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/IndustrialHangar/Data_easy/image_rcam_front.zip",
            "depth_lcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/IndustrialHangar/Data_easy/depth_lcam_front.zip",
        },
        "approx_gb": 16.05,
    },
    "Supermarket": {
        "dataset_version": "tartanair2",
        "split": "Data_easy",
        "description": "Tall multi-tier metal shelving, long straight aisles, boxed pallet organization",
        "files": {
            "image_lcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/Supermarket/Data_easy/image_lcam_front.zip",
            "image_rcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/Supermarket/Data_easy/image_rcam_front.zip",
            "depth_lcam_front": "https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/Supermarket/Data_easy/depth_lcam_front.zip",
        },
        "approx_gb": 8.96,
    },
}

def download_file_with_resume(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> bool:
    """Download file with HTTP Range resume support and clean terminal progress reporting."""
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
        with urllib.request.urlopen(req, timeout=20) as resp:
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
                    if now - last_print > 0.5:
                        elapsed = now - start_time
                        speed_mb = (downloaded - initial_pos) / (1024 * 1024) / max(elapsed, 1e-4)
                        pct = (downloaded / total_size * 100) if total_size > 0 else 0
                        sys.stdout.write(f"\r      Progress: {downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB [{pct:.1f}%] @ {speed_mb:.1f} MB/s   ")
                        sys.stdout.flush()
                        last_print = now

            print(f"\r      Completed: {fname} ({total_size / (1024**3):.2f} GB)                                ")
            os.rename(temp_path, dest_path)
            return True
    except Exception as e:
        print(f"\n    [Error] Failed to download {url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download TartanAir Warehouse & Factory Stereo Dataset (~45 GB total)")
    parser.add_argument("--out-dir", type=str, default="data/warehouse_stereo", help="Destination base directory")
    parser.add_argument(
        "--env",
        type=str,
        default="all",
        choices=["all", "carwelding", "abandonedfactory", "IndustrialHangar", "Supermarket"],
        help="Select specific environment or 'all'",
    )
    args = parser.parse_args()

    target_envs = list(DATASET_REGISTRY.keys()) if args.env == "all" else [args.env]
    total_gb = sum(DATASET_REGISTRY[e]["approx_gb"] for e in target_envs)

    print("==========================================================================")
    print("  TartanAir Warehouse & Factory Pure Stereo Downloader")
    print(f"  Target Environments: {', '.join(target_envs)}")
    print(f"  Estimated Download  : ~{total_gb:.2f} GB")
    print(f"  Destination Path    : {os.path.abspath(args.out_dir)}")
    print("==========================================================================\n")

    for env_name in target_envs:
        meta = DATASET_REGISTRY[env_name]
        print(f"\n>>> Environment: {env_name} (~{meta['approx_gb']:.2f} GB)")
        print(f"    Description: {meta['description']}")
        env_dir = os.path.join(args.out_dir, env_name, meta["split"])

        for mod_name, url in meta["files"].items():
            dest = os.path.join(env_dir, f"{mod_name}.zip")
            success = download_file_with_resume(url, dest)
            if not success:
                print(f"    [Warning] Could not fetch {mod_name}. Retrying once...")
                time.sleep(2)
                download_file_with_resume(url, dest)

    print("\n==========================================================================")
    print(f"  Download process finished! All archives stored in: {os.path.abspath(args.out_dir)}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
