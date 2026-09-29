#!/usr/bin/env python3
"""
Download and Assemble Diverse Indoor Benchmark Suite
=====================================================
Downloads and organizes unseen indoor frames from:
1. TartanAir v2:
   - AmericanDiner (Restaurant interior)
   - ArchVizTinyHouseDay (Scandinavian residential day)
   - ArchVizTinyHouseNight (Residential night)
   - Prison (Correctional cell block / concrete interior)
   - RetroOffice (Retro corporate interior)
   - House (Suburban house interior)
   - Supermarket (Retail aisles and checkout)
2. NYU-Depth V2:
   - Official validation set RGB-D frames (Real-world Microsoft Kinect)
"""

import os
import io
import sys
import time
import zlib
import json
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image

sys.path.append("scripts")
from download_unseen_benchmarks import RemoteZipIndex, fetch_single_file


# ---------------------------------------------------------------------------
# TartanAir2 Downloader
# ---------------------------------------------------------------------------

def download_tartanair2_indoor(
    env_name: str,
    traj_id: str = "P000",
    difficulty: str = "Data_easy",
    num_frames: int = 50,
    out_base: str = "test_samples/indoor_suite",
):
    target_dir = os.path.join(out_base, f"tartanair2_{env_name.lower()}")
    img_dir = os.path.join(target_dir, "image_left")
    depth_dir = os.path.join(target_dir, "depth_left")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    existing_imgs = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    existing_depths = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
    if len(existing_imgs) >= num_frames and len(existing_depths) >= num_frames:
        print(f"[{env_name}] Already present ({len(existing_imgs)} frames). Skipping.")
        return target_dir

    print(f"\n{'='*75}")
    print(f"DOWNLOADING TARTANAIR2 INDOOR: {env_name} ({traj_id}) -> {num_frames} frames")
    print(f"{'='*75}")

    img_url = f"https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/{env_name}/{difficulty}/image_lcam_front.zip"
    depth_url = f"https://huggingface.co/datasets/theairlabcmu/tartanair2/resolve/main/{env_name}/{difficulty}/depth_lcam_front.zip"

    t0 = time.time()
    idx_img = RemoteZipIndex(img_url)
    idx_depth = RemoteZipIndex(depth_url)
    print(f"✓ Parsed remote indices in {time.time() - t0:.2f}s ({len(idx_img.entries)} images, {len(idx_depth.entries)} depths)")

    # Filter keys matching traj_id
    img_entries = {k: v for k, v in idx_img.entries.items() if f"/{traj_id}/image_lcam_front/" in k and k.endswith(".png")}
    depth_entries = {k: v for k, v in idx_depth.entries.items() if f"/{traj_id}/depth_lcam_front/" in k and k.endswith(".png")}

    # Match common frame stems
    # stem: e.g. 000018_lcam_front
    img_map = {os.path.basename(k).replace(".png", ""): (k, v) for k, v in img_entries.items()}
    depth_map = {os.path.basename(k).replace("_depth.png", ""): (k, v) for k, v in depth_entries.items()}

    common_stems = sorted(list(set(img_map.keys()) & set(depth_map.keys())))
    if len(common_stems) > num_frames:
        step = max(1, len(common_stems) // num_frames)
        selected_stems = common_stems[::step][:num_frames]
    else:
        selected_stems = common_stems[:num_frames]

    print(f"Fetching {len(selected_stems)} frames concurrently...")

    def fetch_and_save_pair(stem: str) -> bool:
        img_k, img_e = img_map[stem]
        depth_k, depth_e = depth_map[stem]

        img_dest = os.path.join(img_dir, f"{stem}.png")
        depth_dest = os.path.join(depth_dir, f"{stem}_depth.npy")

        if os.path.exists(img_dest) and os.path.exists(depth_dest):
            return True

        try:
            raw_img = fetch_single_file(img_url, img_e)
            raw_depth = fetch_single_file(depth_url, depth_e)

            # Write RGB
            with open(img_dest, "wb") as f:
                f.write(raw_img)

            # Decode 4-channel depth PNG to float32 metres
            depth_rgba = cv2.imdecode(np.frombuffer(raw_depth, np.uint8), cv2.IMREAD_UNCHANGED)
            depth_m = np.squeeze(depth_rgba.view("<f4"), axis=-1)
            np.save(depth_dest, depth_m.astype(np.float32))
            return True
        except Exception as e:
            print(f"Error fetching {stem}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(fetch_and_save_pair, selected_stems))

    success_cnt = sum(results)
    print(f"✓ Saved {success_cnt}/{len(selected_stems)} frames to {target_dir}")
    return target_dir


# ---------------------------------------------------------------------------
# NYU-Depth V2 Downloader
# ---------------------------------------------------------------------------

def download_nyu_depth_v2(num_frames: int = 50, out_base: str = "test_samples/indoor_suite"):
    import h5py
    target_dir = os.path.join(out_base, "nyu_depth_v2")
    img_dir = os.path.join(target_dir, "image_left")
    depth_dir = os.path.join(target_dir, "depth_left")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    existing_imgs = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    existing_depths = [f for f in os.listdir(depth_dir) if f.endswith(".npy")]
    if len(existing_imgs) >= num_frames and len(existing_depths) >= num_frames:
        print(f"[NYU-Depth V2] Already present ({len(existing_imgs)} frames). Skipping.")
        return target_dir

    print(f"\n{'='*75}")
    print(f"DOWNLOADING NYU-DEPTH V2 TEST FRAMES -> {num_frames} frames")
    print(f"{'='*75}")

    tar_url = "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/val-000000.tar"
    # Fetch first 75MB buffer containing ~51 h5 files
    req = urllib.request.Request(tar_url, headers={"Range": "bytes=0-78643200", "User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=45) as resp:
        buf = resp.read()
    print(f"✓ Downloaded {len(buf)/(1024*1024):.1f} MB in {time.time()-t0:.2f}s")

    pos = 0
    saved = 0
    while pos + 512 <= len(buf) and saved < num_frames:
        h = buf[pos:pos+512]
        name = h[:100].split(b"\x00")[0].decode("utf-8", errors="ignore")
        if not name:
            break
        try:
            sz = int(h[124:136].split(b"\x00")[0].strip(), 8)
        except:
            break

        file_data = buf[pos+512:pos+512+sz]
        blocks = (sz + 511) // 512
        pos += 512 + blocks * 512

        if not name.endswith(".h5"):
            continue

        stem = os.path.basename(name).replace(".h5", "")
        img_dest = os.path.join(img_dir, f"{stem}.png")
        depth_dest = os.path.join(depth_dir, f"{stem}_depth.npy")

        try:
            with h5py.File(io.BytesIO(file_data), "r") as h5:
                rgb = np.array(h5["rgb"]) # (3, 480, 640) uint8
                depth = np.array(h5["depth"]) # (480, 640) float32 in metres

            # Transpose rgb to (H, W, 3)
            rgb = np.transpose(rgb, (1, 2, 0))
            PILImage = Image.fromarray(rgb)
            PILImage.save(img_dest)
            np.save(depth_dest, depth.astype(np.float32))
            saved += 1
        except Exception as e:
            print(f"Error extracting {name}: {e}")

    print(f"✓ Saved {saved} official NYU-Depth V2 test frames to {target_dir}")
    return target_dir


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = "test_samples/indoor_suite"
    os.makedirs(out_dir, exist_ok=True)

    t_start = time.time()
    tartanair2_envs = [
        "AmericanDiner",
        "ArchVizTinyHouseDay",
        "ArchVizTinyHouseNight",
        "Prison",
        "RetroOffice",
        "House",
        "Supermarket",
    ]

    for env in tartanair2_envs:
        download_tartanair2_indoor(env, traj_id="P000", num_frames=50, out_base=out_dir)

    download_nyu_depth_v2(num_frames=50, out_base=out_dir)

    print(f"\n{'='*75}")
    print(f"INDOOR SUITE DOWNLOAD COMPLETE IN {time.time() - t_start:.2f}s")
    print(f"{'='*75}")
