#!/usr/bin/env python3
"""
Stream Missing Indoor & Level Ground Environments to Google Drive:
  Target Environments (NOT present in Kaggle Warehouse dataset):
    - office
    - office2
    - hospital
    - neighborhood

  Target Directory:
    /Users/krishnakant/Library/CloudStorage/GoogleDrive-harryson424242@gmail.com/My Drive/dioptra_datasets/tartanair_indoor_complement
"""

import os
import sys
import io
import time
import zipfile
import urllib.request
from tqdm import tqdm

GDRIVE_ROOT = "/Users/krishnakant/Library/CloudStorage/GoogleDrive-harryson424242@gmail.com/My Drive/dioptra_datasets"

class HttpRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            self.length = int(resp.headers.get('Content-Length', 0))

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1 or self.pos + size > self.length:
            size = self.length - self.pos
        if size <= 0:
            return b''
        req = urllib.request.Request(
            self.url,
            headers={'Range': f'bytes={self.pos}-{self.pos+size-1}', 'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def download_tartanair_env(env_name: str, max_frames: int = None):
    """Stream and extract TartanAir frames to Google Drive."""
    out_dir = os.path.join(GDRIVE_ROOT, "tartanair_indoor_complement", env_name)
    img_dir = os.path.join(out_dir, "image_left")
    depth_dir = os.path.join(out_dir, "depth_left")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    frame_str = "ALL available" if max_frames is None else f"{max_frames}"
    print(f"\nStreaming Indoor Complement: {env_name} ({frame_str} frames)...")
    img_url = f"https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/{env_name}/Easy/image_left.zip"
    depth_url = f"https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/{env_name}/Easy/depth_left.zip"

    try:
        zf_img = zipfile.ZipFile(HttpRangeFile(img_url))
        zf_depth = zipfile.ZipFile(HttpRangeFile(depth_url))

        img_names = sorted([n for n in zf_img.namelist() if n.endswith(".png")])
        depth_names = sorted([n for n in zf_depth.namelist() if n.endswith(".npy")])

        if max_frames is not None:
            img_names = img_names[:max_frames]
            depth_names = depth_names[:max_frames]

        for img_n, depth_n in zip(tqdm(img_names, desc=f"{env_name} Frames"), depth_names):
            fname = os.path.basename(img_n)
            dname = os.path.basename(depth_n)
            img_dest = os.path.join(img_dir, fname)
            depth_dest = os.path.join(depth_dir, dname)

            if not os.path.exists(img_dest):
                data = zf_img.read(img_n)
                with open(img_dest, "wb") as f:
                    f.write(data)
            if not os.path.exists(depth_dest):
                data = zf_depth.read(depth_n)
                with open(depth_dest, "wb") as f:
                    f.write(data)

        print(f"[SUCCESS] {env_name}: Saved {len(img_names)} frames to {out_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to stream TartanAir {env_name}: {e}")


def main():
    print("=" * 78)
    print(" STREAMING TARTANAIR INDOOR COMPLEMENT TO GOOGLE DRIVE (ALL FRAMES)")
    print(" (Excludes environments already present in Kaggle Warehouse dataset)")
    print(f" Target Directory: {GDRIVE_ROOT}")
    print("=" * 78)

    # Missing environments NOT in your Kaggle warehouse dataset
    indoor_complement_envs = [
        "office",
        "office2",
        "hospital",
        "neighborhood"
    ]
    for env in indoor_complement_envs:
        download_tartanair_env(env, max_frames=None)

    print("\n" + "=" * 78)
    print(" INDOOR COMPLEMENT DATASET COMPLETE! Ready in Google Drive.")
    print("=" * 78)


if __name__ == "__main__":
    main()
