#!/usr/bin/env python3
"""
ScanNet Official Dataset Downloader & SensReader Extractor (Python 3)
Downloads official raw ScanNet streams from TUM server (kaldir.vc.in.tum.de):
  - v1 raw .sens video streams (full RGB, 16-bit metric depth, camera poses, intrinsics)
  - v2 3D meshes (_vh_clean_2.ply, _vh_clean_2.labels.ply, .aggregation.json)
  - Camera metadata (.txt)

Supports:
  - HTTP Range header resumable chunked downloads
  - In-place SensorData decompression to color images, 16-bit depth PNGs, poses, intrinsics
  - Automatic cleanup of large temporary .sens files
"""

import os
import sys
import io
import time
import struct
import zlib
import argparse
import urllib.request
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

TUM_BASE_URL = "https://kaldir.vc.in.tum.de/scannet"
COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


class RGBDFrame:
    """Individual RGB-D frame extracted from ScanNet .sens stream."""

    def __init__(self):
        self.camera_to_world = None
        self.timestamp_color = None
        self.timestamp_depth = None
        self.color_size_bytes = None
        self.depth_size_bytes = None
        self.color_data = None
        self.depth_data = None

    def load(self, f):
        self.camera_to_world = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", f.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", f.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", f.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", f.read(8))[0]
        self.color_data = f.read(self.color_size_bytes)
        self.depth_data = f.read(self.depth_size_bytes)

    def decompress_depth(self, comp_type: str, height: int, width: int) -> np.ndarray:
        if comp_type == "zlib_ushort":
            raw = zlib.decompress(self.depth_data)
            return np.frombuffer(raw, dtype=np.uint16).reshape(height, width)
        elif comp_type == "raw_ushort":
            return np.frombuffer(self.depth_data, dtype=np.uint16).reshape(height, width)
        raise ValueError(f"Unsupported depth compression: {comp_type}")

    def decompress_color(self, comp_type: str) -> Image.Image:
        if comp_type in ("jpeg", "png"):
            return Image.open(io.BytesIO(self.color_data))
        raise ValueError(f"Unsupported color compression: {comp_type}")


class SensorData:
    """Native Python 3 parser for binary ScanNet .sens files."""

    def __init__(self, filename: str):
        self.filename = filename
        self.version = 4
        self.frames = []
        self._load()

    def _load(self):
        print(f"[SensReader] Loading metadata from {os.path.basename(self.filename)}...")
        with open(self.filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert version == self.version, f"Unsupported .sens version {version} (expected {self.version})"
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = f.read(strlen).decode("utf-8", errors="ignore")
            self.color_compression_type = COMPRESSION_TYPE_COLOR.get(struct.unpack("i", f.read(4))[0], "unknown")
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH.get(struct.unpack("i", f.read(4))[0], "unknown")
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            self.calibration_color = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.calibration_depth = np.asarray(struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32).reshape(4, 4)
            self.num_frames = struct.unpack("Q", f.read(8))[0]

            print(f"  • Sensor:        {self.sensor_name}")
            print(f"  • Color Stream:  {self.color_width}x{self.color_height} ({self.color_compression_type})")
            print(f"  • Depth Stream:  {self.depth_width}x{self.depth_height} ({self.depth_compression_type}, shift={self.depth_shift})")
            print(f"  • Frame Count:   {self.num_frames:,}")

            for _ in tqdm(range(self.num_frames), desc="Indexing Frames", leave=False):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_full_rgbd(self, output_dir: str, frame_skip: int = 1):
        """Export color PNGs, 16-bit depth PNGs, 4x4 poses, and intrinsics."""
        color_dir = os.path.join(output_dir, "color")
        depth_dir = os.path.join(output_dir, "depth")
        pose_dir = os.path.join(output_dir, "pose")
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)

        # Save camera intrinsic matrices
        np.savetxt(os.path.join(output_dir, "intrinsic_color.txt"), self.calibration_color, fmt="%.8f")
        np.savetxt(os.path.join(output_dir, "intrinsic_depth.txt"), self.calibration_depth, fmt="%.8f")

        print(f"[SensReader] Unpacking {len(self.frames) // frame_skip:,} frames to {output_dir}...")
        for i in tqdm(range(0, len(self.frames), frame_skip), desc="Exporting RGB-D"):
            frame = self.frames[i]
            frame_id = f"{i:06d}"

            # 1. Color frame
            color_img = frame.decompress_color(self.color_compression_type)
            color_img.save(os.path.join(color_dir, f"{frame_id}.jpg"), quality=95)

            # 2. 16-bit Metric depth frame (in millimeters)
            depth_map = frame.decompress_depth(self.depth_compression_type, self.depth_height, self.depth_width)
            depth_img = Image.fromarray(depth_map)
            depth_img.save(os.path.join(depth_dir, f"{frame_id}.png"))

            # 3. 4x4 Camera-to-world pose
            pose_path = os.path.join(pose_dir, f"{frame_id}.txt")
            np.savetxt(pose_path, frame.camera_to_world, fmt="%.8f")

        print(f"✓ Export complete: {len(self.frames) // frame_skip:,} frames in {output_dir}")


def download_file_with_resume(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> bool:
    """Download large binary files with HTTP Range resume support and live progress bar."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_path = dest_path + ".part"

    initial_pos = 0
    if os.path.exists(dest_path):
        print(f"  [Exists] {os.path.basename(dest_path)} already downloaded.")
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
            with open(temp_path, mode) as out_f:
                with tqdm(
                    total=total_size,
                    initial=initial_pos,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(dest_path),
                ) as pbar:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        pbar.update(len(chunk))

        os.rename(temp_path, dest_path)
        return True
    except Exception as e:
        print(f"  [Download Error]: {e}")
        return False


def download_scannet_scan(
    scan_id: str,
    output_dir: str,
    file_types: Optional[List[str]] = None,
    extract_sens: bool = True,
    frame_skip: int = 1,
    clean_sens_after_extract: bool = False,
):
    """Download official ScanNet files for a given scan ID and optionally unpack."""
    if file_types is None:
        file_types = [".sens", ".txt"]

    scan_dir = os.path.join(output_dir, scan_id)
    os.makedirs(scan_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"DOWNLOADING SCANNET SCAN: {scan_id}")
    print(f"Target Directory: {scan_dir}")
    print(f"{'='*70}")

    for ft in file_types:
        if ft == ".sens":
            # Raw .sens stream resides in v1/scans on TUM server
            url = f"{TUM_BASE_URL}/v1/scans/{scan_id}/{scan_id}.sens"
            dest = os.path.join(scan_dir, f"{scan_id}.sens")
        elif ft == ".txt":
            # Metadata file with camera details
            url = f"{TUM_BASE_URL}/v1/scans/{scan_id}/{scan_id}.txt"
            dest = os.path.join(scan_dir, f"{scan_id}.txt")
        elif ft in ("_vh_clean_2.ply", "_vh_clean_2.labels.ply", ".aggregation.json"):
            # v2 reconstructed mesh and annotations
            url = f"{TUM_BASE_URL}/v2/scans/{scan_id}/{scan_id}{ft}"
            dest = os.path.join(scan_dir, f"{scan_id}{ft}")
        else:
            url = f"{TUM_BASE_URL}/v2/scans/{scan_id}/{scan_id}{ft}"
            dest = os.path.join(scan_dir, f"{scan_id}{ft}")

        print(f"\nFetching {ft} from {url}...")
        success = download_file_with_resume(url, dest)
        if not success:
            print(f"Failed to fetch {ft} for {scan_id}")

    sens_path = os.path.join(scan_dir, f"{scan_id}.sens")
    if extract_sens and os.path.exists(sens_path) and os.path.getsize(sens_path) > 1024 * 1024:
        print(f"\nExtracting full RGB-D data from {sens_path}...")
        sd = SensorData(sens_path)
        sd.export_full_rgbd(scan_dir, frame_skip=frame_skip)
        if clean_sens_after_extract:
            os.remove(sens_path)
            print(f"Cleaned up {os.path.basename(sens_path)} to conserve storage.")


def main():
    parser = argparse.ArgumentParser(description="Official ScanNet Dataset Downloader & SensReader")
    parser.add_argument("--scan-id", type=str, default="scene0000_00", help="Scan ID(s), comma-separated (e.g. 'scene0000_00')")
    parser.add_argument("--output-dir", type=str, default="data/scannet", help="Destination directory for extracted scenes")
    parser.add_argument("--types", type=str, default=".sens,.txt", help="Comma-separated file extensions to download (.sens, .txt, _vh_clean_2.ply)")
    parser.add_argument("--extract", action="store_true", default=True, help="Extract RGB-D frames, poses, and intrinsics (default: True)")
    parser.add_argument("--no-extract", dest="extract", action="store_false", help="Only download raw files without extraction")
    parser.add_argument("--frame-skip", type=int, default=1, help="Subsampling stride (e.g. 1=all frames, 2=every 2nd frame)")
    parser.add_argument("--clean-sens", action="store_true", default=False, help="Delete .sens after extraction to save disk space")
    args = parser.parse_args()

    scans = [s.strip() for s in args.scan_id.split(",") if s.strip()]
    file_types = [t.strip() for t in args.types.split(",") if t.strip()]

    for scan in scans:
        download_scannet_scan(
            scan_id=scan,
            output_dir=args.output_dir,
            file_types=file_types,
            extract_sens=args.extract,
            frame_skip=args.frame_skip,
            clean_sens_after_extract=args.clean_sens,
        )


if __name__ == "__main__":
    main()
