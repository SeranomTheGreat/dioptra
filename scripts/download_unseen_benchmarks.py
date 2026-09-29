#!/usr/bin/env python3
"""
High-Speed Concurrent Downloader for Unseen TartanAir Benchmarks:
  1. JapaneseAlley (Easy / P001 - Outdoor Narrow Street)
  2. CarWelding (Easy / P001 - Industrial Robotics Workcell)

Uses RemoteZipIndex with parallel HTTP Range chunk extraction.
"""

import os
import io
import sys
import time
import zlib
import glob
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple


class RemoteZipIndex:
    """Reads central directory of a remote zip file via HTTP range requests with Zip64 support."""
    def __init__(self, url: str):
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            self.url = resp.geturl()
            self.length = int(resp.headers.get("Content-Length"))

        scan_size = min(131072, self.length)
        start = self.length - scan_size
        req = urllib.request.Request(self.url, headers={
            "Range": f"bytes={start}-{self.length - 1}",
            "User-Agent": "Mozilla/5.0"
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            tail = resp.read()

        # Check for Zip64 Locator
        z64_loc = tail.rfind(b"\x50\x4b\x06\x07")
        if z64_loc != -1:
            z64_eocd_offset = int.from_bytes(tail[z64_loc + 8:z64_loc + 16], "little")
            req = urllib.request.Request(self.url, headers={
                "Range": f"bytes={z64_eocd_offset}-{z64_eocd_offset + 64}",
                "User-Agent": "Mozilla/5.0"
            })
            with urllib.request.urlopen(req, timeout=20) as resp:
                z64_rec = resp.read()
            cd_size = int.from_bytes(z64_rec[40:48], "little")
            cd_offset = int.from_bytes(z64_rec[48:56], "little")
        else:
            eocd_pos = tail.rfind(b"\x50\x4b\x05\x06")
            if eocd_pos == -1:
                raise ValueError(f"Could not find EOCD in {url}")
            cd_size = int.from_bytes(tail[eocd_pos + 12:eocd_pos + 16], "little")
            cd_offset = int.from_bytes(tail[eocd_pos + 16:eocd_pos + 20], "little")

        # Read the central directory
        req = urllib.request.Request(self.url, headers={
            "Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}",
            "User-Agent": "Mozilla/5.0"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            cd_data = resp.read()

        # Parse central directory entries with Zip64 extra field support
        self.entries = {}
        pos = 0
        while pos < len(cd_data):
            if cd_data[pos:pos + 4] != b"\x50\x4b\x01\x02":
                break
            compress_type = int.from_bytes(cd_data[pos + 10:pos + 12], "little")
            compress_size = int.from_bytes(cd_data[pos + 20:pos + 24], "little")
            file_size = int.from_bytes(cd_data[pos + 24:pos + 28], "little")
            fname_len = int.from_bytes(cd_data[pos + 28:pos + 30], "little")
            extra_len = int.from_bytes(cd_data[pos + 30:pos + 32], "little")
            comment_len = int.from_bytes(cd_data[pos + 32:pos + 34], "little")
            header_offset = int.from_bytes(cd_data[pos + 42:pos + 46], "little")

            fname = cd_data[pos + 46:pos + 46 + fname_len].decode("utf-8", errors="ignore")
            extra = cd_data[pos + 46 + fname_len:pos + 46 + fname_len + extra_len]

            # Parse zip64 extra field (tag 0x0001)
            epos = 0
            while epos + 4 <= len(extra):
                etag = int.from_bytes(extra[epos:epos + 2], "little")
                esize = int.from_bytes(extra[epos + 2:epos + 4], "little")
                if etag == 1:
                    zpos = epos + 4
                    if file_size == 0xFFFFFFFF and zpos + 8 <= len(extra):
                        file_size = int.from_bytes(extra[zpos:zpos + 8], "little")
                        zpos += 8
                    if compress_size == 0xFFFFFFFF and zpos + 8 <= len(extra):
                        compress_size = int.from_bytes(extra[zpos:zpos + 8], "little")
                        zpos += 8
                    if header_offset == 0xFFFFFFFF and zpos + 8 <= len(extra):
                        header_offset = int.from_bytes(extra[zpos:zpos + 8], "little")
                        zpos += 8
                    break
                epos += 4 + esize

            self.entries[fname] = {
                "compress_type": compress_type,
                "compress_size": compress_size,
                "file_size": file_size,
                "header_offset": header_offset,
                "fname_len": fname_len,
            }
            pos += 46 + fname_len + extra_len + comment_len


def fetch_single_file(url: str, entry: dict, retries: int = 3) -> bytes:
    """Fetch and decompress an individual file from remote zip using byte ranges."""
    start = entry["header_offset"]
    end = start + 30 + entry["fname_len"] + 256 + entry["compress_size"]
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "Mozilla/5.0"
            })
            with urllib.request.urlopen(req, timeout=25) as resp:
                buf = resp.read()

            extra_len = int.from_bytes(buf[28:30], "little")
            data_start = 30 + entry["fname_len"] + extra_len
            raw = buf[data_start:data_start + entry["compress_size"]]

            if entry["compress_type"] == 8:
                return zlib.decompress(raw, -15)
            return raw
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(1.0 * (attempt + 1))


def download_trajectory_fast(
    env_name: str,
    traj_id: str = "P001",
    difficulty: str = "Easy",
    num_frames: int = 25,
    save_dir: str = "test_samples",
):
    target_dir = os.path.join(save_dir, f"unseen_{env_name}_{traj_id.lower()}")
    img_dir = os.path.join(target_dir, "image_left")
    depth_dir = os.path.join(target_dir, "depth_left")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"FAST DOWNLOADING UNSEEN BENCHMARK: {env_name.upper()} ({difficulty}/{traj_id})")
    print(f"Destination: {target_dir}")
    print(f"{'='*75}")

    img_url = f"https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/{env_name}/{difficulty}/image_left.zip"
    depth_url = f"https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/{env_name}/{difficulty}/depth_left.zip"

    print("Fetching remote zip indexes...")
    t0 = time.time()
    idx_img = RemoteZipIndex(img_url)
    idx_depth = RemoteZipIndex(depth_url)
    print(f"✓ Parsed remote indices in {time.time() - t0:.2f}s ({len(idx_img.entries)} images, {len(idx_depth.entries)} depth files)")

    # Match files containing trajectory ID and correct file extension
    img_entries = {k: v for k, v in idx_img.entries.items() if f"/{traj_id}/image_left/" in k or k.startswith(f"{traj_id}/image_left/")}
    if not img_entries:
        # Fallback to any trajectory available
        available_trajs = sorted(list(set([k.split("/image_left/")[0].split("/")[-1] for k in idx_img.entries if "/image_left/" in k])))
        print(f"Warning: {traj_id} not found. Available trajectories: {available_trajs}")
        if available_trajs:
            traj_id = available_trajs[0]
            img_entries = {k: v for k, v in idx_img.entries.items() if f"/{traj_id}/image_left/" in k or k.startswith(f"{traj_id}/image_left/")}

    depth_entries = {k: v for k, v in idx_depth.entries.items() if f"/{traj_id}/depth_left/" in k or k.startswith(f"{traj_id}/depth_left/")}

    # Match common frame stems
    img_map = {os.path.basename(k).replace(".png", ""): (k, v) for k, v in img_entries.items() if k.endswith(".png")}
    depth_map = {os.path.basename(k).replace("_depth.npy", ""): (k, v) for k, v in depth_entries.items() if k.endswith("_depth.npy")}

    common_stems = sorted(list(set(img_map.keys()) & set(depth_map.keys())))[:num_frames]
    print(f"Discovered {len(common_stems)} matched image/depth pairs. Commencing parallel download...")

    def _fetch_pair(stem: str) -> str:
        img_dest = os.path.join(img_dir, f"{stem}.png")
        depth_dest = os.path.join(depth_dir, f"{stem}_depth.npy")

        if os.path.exists(img_dest) and os.path.exists(depth_dest) and os.path.getsize(img_dest) > 0 and os.path.getsize(depth_dest) > 0:
            return stem

        k_img, e_img = img_map[stem]
        k_dep, e_dep = depth_map[stem]

        if not os.path.exists(img_dest) or os.path.getsize(img_dest) == 0:
            data_img = fetch_single_file(idx_img.url, e_img)
            with open(img_dest, "wb") as f:
                f.write(data_img)

        if not os.path.exists(depth_dest) or os.path.getsize(depth_dest) == 0:
            data_dep = fetch_single_file(idx_depth.url, e_dep)
            with open(depth_dest, "wb") as f:
                f.write(data_dep)

        return stem

    t_fetch = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        completed = list(pool.map(_fetch_pair, common_stems))

    elapsed = time.time() - t_fetch
    print(f"✓ Successfully downloaded and verified {len(completed)} pairs for {env_name} in {elapsed:.2f}s ({len(completed)/max(0.1, elapsed):.1f} pairs/sec)!")


def main():
    # 1. Download JapaneseAlley (Outdoor narrow street, held-out validation)
    download_trajectory_fast(
        env_name="japanesealley",
        traj_id="P001",
        difficulty="Easy",
        num_frames=25,
    )

    # 2. Download CarWelding (Industrial robotics workcell, held-out validation)
    download_trajectory_fast(
        env_name="carwelding",
        traj_id="P001",
        difficulty="Easy",
        num_frames=25,
    )

    print("\n" + "=" * 75)
    print("ALL UNSEEN TEST BENCHMARKS READY FOR EVALUATION!")
    print("=" * 75)


if __name__ == "__main__":
    main()
