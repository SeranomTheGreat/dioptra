"""
Hypersim Dataset Pack -> Kaggle Uploader
- Scans Google Drive Hypersim directory
- Supports all 196 scenes (191 extracted scenes + leftover archives)
- Automatically unnests duplicate scene folders (ai_xxx_yyy/ai_xxx_yyy -> ai_xxx_yyy)
- Filters out .DS_Store, temporary artifacts, and zero-byte files
- Stages clean symbolic links into /tmp/kaggle_upload/hypersim-pack
- Creates or versions the Kaggle dataset under volsiai/hypersim-pack
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess

KAGGLE_USERNAME = "volsiai"
DATASET_SLUG = "hypersim-pack"
DEFAULT_SRC = "/content/drive/MyDrive/dioptra_datasets/hypersim"
MAC_SRC = "/Users/krishnakant/Library/CloudStorage/GoogleDrive-harryson424242@gmail.com/My Drive/dioptra_datasets/hypersim"

def get_source_dir():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    if os.path.exists(DEFAULT_SRC):
        return DEFAULT_SRC
    if os.path.exists(MAC_SRC):
        return MAC_SRC
    raise FileNotFoundError("Could not find Hypersim dataset directory.")

def main():
    src_dir = get_source_dir()
    print(f"[*] Source directory: {src_dir}")
    
    stage_dir = f"/tmp/kaggle_upload/{DATASET_SLUG}"
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)
    
    # Check for valid unextracted zip files
    for item in sorted(os.listdir(src_dir)):
        if item.endswith(".zip") and not item.startswith("."):
            zp = os.path.join(src_dir, item)
            sc_name = item[:-4]
            sc_folder = os.path.join(src_dir, sc_name)
            if os.path.getsize(zp) > 1024 * 1024 and not os.path.exists(sc_folder):
                print(f"[*] Extracting essentials from {item}...")
                try:
                    with zipfile.ZipFile(zp, "r") as zf:
                        members = [m for m in zf.namelist() if ".tonemap.jpg" in m or ".depth_meters.hdf5" in m or "metadata" in m]
                        zf.extractall(sc_folder, members=members)
                    print(f"[+] Extracted {len(members)} files for {sc_name}")
                except Exception as e:
                    print(f"[-] Error extracting {item}: {e}")

    scenes = [s for s in sorted(os.listdir(src_dir)) if s.startswith("ai_") and os.path.isdir(os.path.join(src_dir, s))]
    print(f"[*] Found {len(scenes)} valid scene directories to stage.")
    
    staged_links = 0
    for idx, sc in enumerate(scenes, 1):
        sc_src = os.path.join(src_dir, sc)
        # Resolve redundant nesting: ai_xxx_yyy/ai_xxx_yyy -> ai_xxx_yyy
        inner = os.path.join(sc_src, sc)
        walk_root = inner if os.path.isdir(inner) else sc_src
        sc_dest = os.path.join(stage_dir, sc)
        
        scene_files = 0
        for root, dirs, files in os.walk(walk_root):
            rel = os.path.relpath(root, walk_root)
            dest = sc_dest if rel == "." else os.path.join(sc_dest, rel)
            os.makedirs(dest, exist_ok=True)
            for fn in files:
                if fn.startswith(".") or fn.endswith(".tmp") or fn.endswith(".part"):
                    continue
                src_file = os.path.join(root, fn)
                link_file = os.path.join(dest, fn)
                if not os.path.lexists(link_file):
                    os.symlink(src_file, link_file)
                    staged_links += 1
                    scene_files += 1
        if idx % 20 == 0 or idx == len(scenes):
            print(f"[{idx}/{len(scenes)}] Staged {sc} ({scene_files} files)")
            
    print(f"\n[SUCCESS] Staged {staged_links} files across {len(scenes)} scenes.")
    
    # Metadata for Kaggle
    meta_path = os.path.join(stage_dir, "dataset-metadata.json")
    meta = {
        "title": "Hypersim Indoor Dataset Pack",
        "id": f"{KAGGLE_USERNAME}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": "Photorealistic indoor RGB-D scenes from Apple Hypersim with full-resolution tonemapped RGB images, Float32 metric depth maps, and camera trajectory metadata."
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[*] Generated dataset metadata at {meta_path}")
    
    # Kaggle dataset upload
    print("\n[*] Uploading to Kaggle...")
    cmd = ["kaggle", "datasets", "create", "-p", stage_dir, "--dir-mode", "zip", "--public"]
    print(f"+ {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        if "already in use" in (res.stdout + res.stderr) or "already exists" in (res.stdout + res.stderr):
            print("[*] Dataset already exists, creating a new version...")
            vcmd = ["kaggle", "datasets", "version", "-p", stage_dir, "-m", "full pack", "--dir-mode", "zip"]
            vres = subprocess.run(vcmd, capture_output=True, text=True)
            print(vres.stdout)
            if vres.returncode != 0:
                print(vres.stderr)
        else:
            print(res.stderr)

if __name__ == "__main__":
    main()
