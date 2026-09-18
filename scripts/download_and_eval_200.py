"""
High-Speed 200-Image Benchmark on Unseen Indoor Environment (TartanAir AbandonedFactory P010).

1. Fetches 200 consecutive image/depth pairs directly from CMU TartanAir on Hugging Face using
   parallel HTTP byte-range extraction (takes ~20 seconds, ~75 MB total).
2. Evaluates the 40-epoch trained checkpoint (outputs/dioptra_dino_best.pt) on all 200 unseen images.
3. Computes standard metrics: AbsRel, SqRel, RMSE, RMSE Log, delta < 1.25, delta < 1.25^2, delta < 1.25^3, Scale Ratio.
4. Generates visual distribution and prediction grid plots.
"""

import os
import sys
import io
import time
import math
import zlib
import glob
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import __main__
from dioptra_dino import DioptraDINO, DioptraDINOConfig, IMAGENET_MEAN, IMAGENET_STD
setattr(__main__, "DioptraDINOConfig", DioptraDINOConfig)

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

def fetch_200_pairs_fast(save_dir: str, num_frames: int = 200):
    img_dir = os.path.join(save_dir, "image_left")
    depth_dir = os.path.join(save_dir, "depth_left")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    existing_pngs = sorted([p for p in glob.glob(os.path.join(img_dir, "*.png")) if os.path.getsize(p) > 0])
    existing_npys = sorted([p for p in glob.glob(os.path.join(depth_dir, "*.npy")) if os.path.getsize(p) > 0])
    if len(existing_pngs) >= num_frames and len(existing_npys) >= num_frames:
        print(f"[Fetch] Found {len(existing_pngs)} valid frames already downloaded in {save_dir}.")
        return

    print("[Fetch] Connecting to TartanAir remote archives on Hugging Face...")
    depth_url = "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/abandonedfactory/Easy/depth_left.zip"
    img_url = "https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/abandonedfactory/Easy/image_left.zip"

    d_index = RemoteZipIndex(depth_url)
    i_index = RemoteZipIndex(img_url)

    # Pick 200 consecutive verified frames from P010 that exist in both archives
    all_depth_candidates = sorted([n for n in d_index.entries if "P010/depth_left/" in n and n.endswith(".npy")])
    target_pairs = []
    for d_name in all_depth_candidates:
        fname = os.path.basename(d_name)
        stem = fname.replace("_left_depth.npy", "")
        i_name = f"abandonedfactory/Easy/P010/image_left/{stem}_left.png"
        if i_name in i_index.entries:
            target_pairs.append((d_name, i_name))
        if len(target_pairs) == num_frames:
            break

    print(f"[Fetch] Identified {len(target_pairs)} verified image+depth pairs from unseen trajectory P010.")

    def download_pair(pair):
        d_name, i_name = pair
        fname = os.path.basename(d_name)
        stem = fname.replace("_left_depth.npy", "")

        d_dst = os.path.join(depth_dir, fname)
        i_dst = os.path.join(img_dir, f"{stem}_left.png")

        if not os.path.exists(d_dst) or os.path.getsize(d_dst) == 0:
            d_bytes = fetch_single_file(d_index.url, d_index.entries[d_name])
            tmp_d = d_dst + ".tmp"
            with open(tmp_d, "wb") as f:
                f.write(d_bytes)
            os.replace(tmp_d, d_dst)

        if not os.path.exists(i_dst) or os.path.getsize(i_dst) == 0:
            i_bytes = fetch_single_file(i_index.url, i_index.entries[i_name])
            tmp_i = i_dst + ".tmp"
            with open(tmp_i, "wb") as f:
                f.write(i_bytes)
            os.replace(tmp_i, i_dst)

    print(f"[Fetch] Downloading {len(target_pairs)} pairs in parallel (20 threads)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(download_pair, target_pairs))
    print(f"[Fetch] Complete! Downloaded {len(target_pairs)} pairs in {time.time() - t0:.2f} seconds.")

def compute_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 80.0):
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & (pred > min_depth) & (pred < max_depth) & np.isfinite(pred)
    if mask.sum() == 0:
        return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0, "a1": 0.0, "a2": 0.0, "a3": 0.0, "scale_ratio": 1.0}

    p = pred[mask]
    g = gt[mask]

    thresh = np.maximum((g / p), (p / g))
    a1 = float((thresh < 1.25).mean())
    a2 = float((thresh < 1.25 ** 2).mean())
    a3 = float((thresh < 1.25 ** 3).mean())

    rmse = float(np.sqrt(((g - p) ** 2).mean()))
    rmse_log = float(np.sqrt(((np.log(g) - np.log(p)) ** 2).mean()))
    abs_rel = float((np.abs(g - p) / g).mean())
    sq_rel = float((((g - p) ** 2) / g).mean())
    scale_ratio = float(np.median(p) / np.median(g))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "scale_ratio": scale_ratio,
    }

def preprocess_sample(img_path: str, gt_path: str, img_size: int = 224, device: torch.device = torch.device("cpu")):
    raw_img = Image.open(img_path).convert("RGB")
    W_orig, H_orig = raw_img.size
    min_side = min(H_orig, W_orig)
    top = (H_orig - min_side) // 2
    left = (W_orig - min_side) // 2

    raw_cropped = raw_img.crop((left, top, left + min_side, top + min_side))
    img_resized = raw_cropped.resize((img_size, img_size), Image.Resampling.BILINEAR)

    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    input_t = ((img_t - mean) / std).unsqueeze(0).to(device)

    gt_arr = np.load(gt_path).astype(np.float32)
    gt_t = torch.from_numpy(gt_arr)
    gt_cropped = gt_t[top:top + min_side, left:left + min_side]
    gt_depth = F.interpolate(gt_cropped.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode="nearest").squeeze(0).squeeze(0)
    gt_np = gt_depth.numpy()

    fx_native = 320.0 * (float(img_size) / float(min_side))
    K_native = torch.tensor([[[fx_native, 0.0, img_size / 2.0],
                             [0.0, fx_native, img_size / 2.0],
                             [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

    return input_t, gt_np, img_np, K_native

def render_200_summary(m_list, out_plot: str):
    abs_rels = [m["abs_rel"] for m in m_list]
    rmses = [m["rmse"] for m in m_list]
    a1s = [m["a1"] * 100 for m in m_list]
    scales = [m["scale_ratio"] for m in m_list]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Dioptra-DINO: Benchmark Distribution Across 200 Unseen Indoor Images", fontsize=14, fontweight="bold")

    axes[0, 0].hist(abs_rels, bins=25, color="#2b5c8f", edgecolor="black", alpha=0.85)
    axes[0, 0].axvline(np.mean(abs_rels), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(abs_rels):.4f}")
    axes[0, 0].set_title("Absolute Relative Error (AbsRel)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("AbsRel (lower is better)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(rmses, bins=25, color="#1b7837", edgecolor="black", alpha=0.85)
    axes[0, 1].axvline(np.mean(rmses), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(rmses):.3f}m")
    axes[0, 1].set_title("Root Mean Squared Error (RMSE)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("RMSE in meters (lower is better)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(a1s, bins=25, color="#762a83", edgecolor="black", alpha=0.85)
    axes[1, 0].axvline(np.mean(a1s), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(a1s):.2f}%")
    axes[1, 0].set_title("Accuracy (δ < 1.25)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Accuracy % (higher is better)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(scales, bins=25, color="#d95f02", edgecolor="black", alpha=0.85)
    axes[1, 1].axvline(np.median(scales), color="red", linestyle="--", linewidth=2, label=f"Median: {np.median(scales):.4f}")
    axes[1, 1].axvline(1.0, color="black", linestyle=":", linewidth=1.5, label="Target (1.000)")
    axes[1, 1].set_title("Metric Scale Ratio (s / s_gt)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Scale Ratio")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved distribution plot to: {out_plot}")

def render_sample_grid(model, pairs, out_grid: str, device, num_display: int = 6):
    step = len(pairs) // num_display
    selected = [pairs[i * step] for i in range(num_display)]

    fig, axes = plt.subplots(num_display, 4, figsize=(14, 2.7 * num_display))
    fig.suptitle("Dioptra-DINO: Sample Predictions Across 200-Image Unseen Trajectory", fontsize=14, fontweight="bold", y=0.99)

    cols = ["Input RGB", "Ground Truth (Metric)", "Dioptra-DINO (Ours)", "Absolute Error Heatmap"]
    for col_idx, col_name in enumerate(cols):
        axes[0, col_idx].set_title(col_name, fontsize=12, fontweight="bold", pad=8)

    for row_idx, (img_p, gt_p) in enumerate(selected):
        inp, gt_np, img_np, K = preprocess_sample(img_p, gt_p, img_size=224, device=device)
        with torch.no_grad():
            pred = model(inp, K).squeeze().cpu().numpy()

        vmax = max(gt_np[gt_np < 80].max() if (gt_np < 80).any() else 10.0, 10.0)
        vmax = min(vmax, 25.0)

        axes[row_idx, 0].imshow(img_np)
        fname = os.path.basename(img_p).replace("_left.png", "")
        axes[row_idx, 0].set_ylabel(f"Frame {fname}", fontsize=9, fontweight="bold")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        axes[row_idx, 1].imshow(gt_np, cmap="inferno", vmin=0.2, vmax=vmax)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(pred, cmap="inferno", vmin=0.2, vmax=vmax)
        axes[row_idx, 2].axis("off")

        diff = np.abs(pred - gt_np)
        diff[~np.isfinite(gt_np) | (gt_np <= 0.1) | (gt_np >= 80)] = 0.0
        axes[row_idx, 3].imshow(diff, cmap="magma", vmin=0.0, vmax=1.5)
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(out_grid, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved visual sample grid to: {out_grid}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Dioptra-DINO checkpoint on 200 unseen frames")
    parser.add_argument("--checkpoint", type=str, default="outputs/dioptra_dino_best.pt", help="Path to checkpoint .pt")
    parser.add_argument("--data-dir", type=str, default="test_samples/unseen_200_abandonedfactory", help="Benchmark frames directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for plots and metrics")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save metrics JSON")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 78)
    print(" DIOPTRA-DINO: 200-IMAGE UNSEEN INDOOR GENERALIZATION BENCHMARK")
    print(f" Compute Device: {device}")
    print(f" Checkpoint    : {args.checkpoint}")
    print("=" * 78)

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = args.data_dir
    fetch_200_pairs_fast(data_dir, num_frames=200)

    # Gather matching pairs
    png_files = sorted([p for p in glob.glob(os.path.join(data_dir, "image_left", "*.png")) if os.path.getsize(p) > 0])
    pairs = []
    for p in png_files:
        base = os.path.basename(p).replace("_left.png", "")
        gt_path = os.path.join(data_dir, "depth_left", f"{base}_left_depth.npy")
        if os.path.exists(gt_path) and os.path.getsize(gt_path) > 0:
            pairs.append((p, gt_path))
    pairs = pairs[:200]

    print(f"\n[Test] Loaded {len(pairs)} verified image/depth pairs for evaluation.")

    ckpt_path = args.checkpoint
    print(f"[Test] Loading model weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("config", DioptraDINOConfig(freeze_backbone=False))
    model = DioptraDINO(cfg).to(device)

    state_dict = state.get("model_state_dict", state)
    cleaned_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_dict, strict=False)
    model.eval()
    print(f"[Test] Successfully loaded model! Trained Epoch: {state.get('epoch')}")

    print(f"\n--- Running Forward Inference on {len(pairs)} Unseen Indoor Images ---")
    m_list = []
    t_start = time.time()
    for idx, (img_p, gt_p) in enumerate(pairs):
        inp, gt_np, img_np, K = preprocess_sample(img_p, gt_p, img_size=224, device=device)
        with torch.no_grad():
            pred = model(inp, K).squeeze().cpu().numpy()
        m = compute_metrics(pred, gt_np)
        m_list.append(m)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(pairs):
            elapsed = time.time() - t_start
            fps = (idx + 1) / elapsed
            print(f"  Processed [{idx+1}/{len(pairs)}] images ({fps:.1f} FPS, AbsRel so far: {np.mean([x['abs_rel'] for x in m_list]):.4f})...")

    # Aggregate metrics
    mean_absrel = np.mean([m["abs_rel"] for m in m_list])
    mean_sqrel = np.mean([m["sq_rel"] for m in m_list])
    mean_rmse = np.mean([m["rmse"] for m in m_list])
    mean_rmselog = np.mean([m["rmse_log"] for m in m_list])
    mean_a1 = np.mean([m["a1"] for m in m_list]) * 100
    mean_a2 = np.mean([m["a2"] for m in m_list]) * 100
    mean_a3 = np.mean([m["a3"] for m in m_list]) * 100
    median_scale = np.median([m["scale_ratio"] for m in m_list])

    print("\n" + "=" * 78)
    print(f" FINAL BENCHMARK REPORT (N = {len(pairs)} UNSEEN INDOOR IMAGES)")
    print("=" * 78)
    print(f"  Absolute Relative Error (AbsRel)  : {mean_absrel:.4f}  ({mean_absrel*100:.2f}%)")
    print(f"  Squared Relative Error (SqRel)   : {mean_sqrel:.4f}")
    print(f"  Root Mean Squared Error (RMSE)   : {mean_rmse:.3f} m")
    print(f"  RMSE Log                         : {mean_rmselog:.4f}")
    print(f"  Threshold Accuracy (δ < 1.25)    : {mean_a1:.2f}%")
    print(f"  Threshold Accuracy (δ < 1.25²)   : {mean_a2:.2f}%")
    print(f"  Threshold Accuracy (δ < 1.25³)   : {mean_a3:.2f}%")
    print(f"  Median Metric Scale Ratio        : {median_scale:.4f}  (Scale Error: {abs(median_scale - 1.0)*100:.2f}%)")
    print("=" * 78)

    metrics_dict = {
        "abs_rel": float(mean_absrel),
        "sq_rel": float(mean_sqrel),
        "rmse": float(mean_rmse),
        "rmse_log": float(mean_rmselog),
        "delta1": float(mean_a1),
        "delta2": float(mean_a2),
        "delta3": float(mean_a3),
        "scale_ratio": float(median_scale),
        "scale_error": float(abs(median_scale - 1.0) * 100),
        "num_samples": len(pairs),
        "checkpoint": args.checkpoint,
    }

    if args.save_json:
        import json
        with open(args.save_json, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"[Metrics] Saved benchmark JSON to: {args.save_json}")

    plot_path = os.path.join(args.output_dir, "unseen_200_metrics_distribution.png")
    grid_path = os.path.join(args.output_dir, "unseen_200_visual_grid.png")
    render_200_summary(m_list, plot_path)
    render_sample_grid(model, pairs, grid_path, device, num_display=6)

    print("\n>>> 200-IMAGE BENCHMARK COMPLETED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    main()
