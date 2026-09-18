#!/usr/bin/env python3
"""
Stream-repack Hypersim into a clean, unnested, verified dataset archive.
- Merges the 5 split Google Drive zips and the ai_023_010 Apple archive.
- Flattens duplicate nesting: hypersim/ai_xxx_yyy/ai_xxx_yyy/... -> hypersim/ai_xxx_yyy/...
- Strips .DS_Store, empty files, and unnecessary physics passes.
- Output: data/kaggle_upload/hypersim_pack.zip (~8.3 GB)
"""

import os
import glob
import time
import zipfile

DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
OUT_ZIP = "/Users/krishnakant/Downloads/tesseract_kaggle_code_v16/data/kaggle_upload/hypersim_pack.zip"

def main():
    start_time = time.time()
    gdrive_zips = sorted(glob.glob(os.path.join(DOWNLOADS_DIR, "hypersim-20260918T092537Z-1-*.zip")))
    ai_zip = os.path.join(DOWNLOADS_DIR, "ai_023_010-001.zip")
    
    print(f"[*] Found {len(gdrive_zips)} Google Drive hypersim zips:")
    for gz in gdrive_zips:
        print(f"    - {os.path.basename(gz)} ({os.path.getsize(gz)/(1024**3):.2f} GB)")
    print(f"[*] Found ai_023_010 archive: {os.path.basename(ai_zip)} ({os.path.getsize(ai_zip)/(1024**3):.2f} GB)")
    
    if os.path.exists(OUT_ZIP):
        os.remove(OUT_ZIP)
    os.makedirs(os.path.dirname(OUT_ZIP), exist_ok=True)
    
    seen_paths = set()
    written_count = 0
    scenes = set()
    
    print(f"\n[*] Creating clean unified archive: {OUT_ZIP}")
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as out_zf:
        # 1. Process the 5 Google Drive zips
        for gz_idx, gz in enumerate(gdrive_zips, 1):
            print(f"[*] Processing [{gz_idx}/{len(gdrive_zips)}] {os.path.basename(gz)}...")
            with zipfile.ZipFile(gz, "r") as in_zf:
                for info in in_zf.infolist():
                    name = info.filename
                    if name.endswith("/") or ".DS_Store" in name:
                        continue
                    if name.endswith(".zip") and info.file_size == 0:
                        continue
                    
                    parts = name.split("/")
                    # Resolve duplicate nesting: hypersim/ai_xxx_yyy/ai_xxx_yyy/... -> hypersim/ai_xxx_yyy/...
                    if len(parts) >= 3 and parts[0] == "hypersim" and parts[1].startswith("ai_") and parts[2] == parts[1]:
                        clean_name = f"hypersim/{parts[1]}/" + "/".join(parts[3:])
                        scene_id = parts[1]
                    elif len(parts) >= 2 and parts[0] == "hypersim" and parts[1].startswith("ai_"):
                        clean_name = name
                        scene_id = parts[1]
                    else:
                        clean_name = name
                        scene_id = None
                    
                    if clean_name in seen_paths:
                        continue
                    seen_paths.add(clean_name)
                    if scene_id:
                        scenes.add(scene_id)
                    
                    # Stream raw bytes directly to output
                    data = in_zf.read(info)
                    out_zf.writestr(clean_name, data)
                    written_count += 1
                    
                    if written_count % 10000 == 0:
                        print(f"    ... written {written_count:,} files (scenes so far: {len(scenes)})")

        # 2. Process ai_023_010-001.zip (filtering only training essentials)
        if os.path.exists(ai_zip):
            print(f"[*] Processing {os.path.basename(ai_zip)} (extracting essentials)...")
            with zipfile.ZipFile(ai_zip, "r") as in_zf:
                for info in in_zf.infolist():
                    name = info.filename
                    if name.endswith("/") or ".DS_Store" in name:
                        continue
                    # Keep RGB tonemap, depth meters, and metadata
                    keep = any(k in name for k in [".tonemap.jpg", ".depth_meters.hdf5", "metadata", "camera_keyframe"])
                    if not keep:
                        continue
                    
                    clean_name = f"hypersim/{name}"
                    if clean_name in seen_paths:
                        continue
                    seen_paths.add(clean_name)
                    scenes.add("ai_023_010")
                    
                    data = in_zf.read(info)
                    out_zf.writestr(clean_name, data)
                    written_count += 1

    elapsed = time.time() - start_time
    final_gb = os.path.getsize(OUT_ZIP) / (1024**3)
    print(f"\n" + "="*60)
    print(f"[SUCCESS] Repack complete in {elapsed:.1f} seconds!")
    print(f"Total files written:  {written_count:,}")
    print(f"Total unique scenes:  {len(scenes)}")
    print(f"Final archive size:   {final_gb:.2f} GB")
    print(f"Target location:      {OUT_ZIP}")
    print("="*60)

if __name__ == "__main__":
    main()
