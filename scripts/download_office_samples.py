"""
Download 30 consecutive frames from TartanAir indoor Office environment (office/Easy/P001).
Indoor environment on level ground (office rooms, desks, hallways, 0.5 - 15m range).
Uses HTTP range requests to extract frames without downloading the multi-GB archives.
"""

import os
import sys
import io
import urllib.request
import zipfile

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


def download_office_subset(num_frames=30, out_dir="test_samples/unseen_office_p001"):
    os.makedirs(os.path.join(out_dir, "image_left"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth_left"), exist_ok=True)

    img_url = "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/office/Easy/image_left.zip"
    depth_url = "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/office/Easy/depth_left.zip"

    print("[1/4] Connecting to TartanAir office image archive...")
    zf_img = zipfile.ZipFile(HttpRangeFile(img_url))
    print("[2/4] Connecting to TartanAir office depth archive...")
    zf_depth = zipfile.ZipFile(HttpRangeFile(depth_url))

    img_names = sorted([n for n in zf_img.namelist() if "P001/image_left/" in n and n.endswith(".png")])[:num_frames]
    depth_names = sorted([n for n in zf_depth.namelist() if "P001/depth_left/" in n and n.endswith(".npy")])[:num_frames]

    print(f"[3/4] Downloading {len(img_names)} consecutive RGB and Depth pairs from office/Easy/P001...")
    for i, (img_path, depth_path) in enumerate(zip(img_names, depth_names)):
        fname_img = os.path.basename(img_path)
        fname_depth = os.path.basename(depth_path)

        local_img = os.path.join(out_dir, "image_left", fname_img)
        local_depth = os.path.join(out_dir, "depth_left", fname_depth)

        if not os.path.exists(local_img):
            data = zf_img.read(img_path)
            with open(local_img, "wb") as f:
                f.write(data)

        if not os.path.exists(local_depth):
            data = zf_depth.read(depth_path)
            with open(local_depth, "wb") as f:
                f.write(data)

        print(f"  [{i+1:02d}/{num_frames}] Saved: {fname_img} + {fname_depth}")

    print(f"[4/4] Successfully downloaded {num_frames} frames to {out_dir}!")


if __name__ == "__main__":
    download_office_subset(num_frames=30)
