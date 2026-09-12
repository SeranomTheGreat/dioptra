#!/usr/bin/env python3
import os
import sys
import time
import json
import tempfile
import urllib.request
import urllib.error
import subprocess

FALLBACK_URL = "https://www.googleapis.com/upload/storage/v1/b/kaggle-data-sets/o?uploadType=resumable&upload_id=AJjja9aVX2eCux_OiREBV5GqFJpalE_8YsBzAbQTa85t6QBWJRu8OcLPlfn_LJmEFCirMpKun0glA052a1BhMUTzNuIKAxHIDeAh9c-OWHfF9IA"
FILE_PATH = "data/kaggle_upload/tartanair_warehouse_stereo.zip"
if not os.path.exists(FILE_PATH):
    FILE_PATH = "data/tartanair_warehouse_stereo.zip"

CHUNK_SIZE = 16 * 1024 * 1024  # 16 MB chunks (multiple of 256 KB)
STATUS_FILE = "logs/kaggle_upload_status.json"

def find_kaggle_upload_info():
    temp_dir = os.path.join(tempfile.gettempdir(), '.kaggle/uploads')
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            if f.endswith('.json'):
                path = os.path.join(temp_dir, f)
                try:
                    with open(path) as fp:
                        data = json.load(fp)
                        if 'start_blob_upload_response' in data:
                            url = data['start_blob_upload_response'].get('createUrl')
                            if url:
                                return url, path
                except Exception:
                    pass
    return FALLBACK_URL, None

def get_current_offset(url, file_size):
    req = urllib.request.Request(
        url,
        headers={"Content-Length": "0", "Content-Range": f"bytes */{file_size}"},
        method="PUT"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status in (200, 201):
                return file_size
    except urllib.error.HTTPError as e:
        if e.code == 308:
            range_hdr = e.headers.get("Range")
            if range_hdr:
                last_byte = int(range_hdr.split("-")[1])
                return last_byte + 1
            return 0
        raise
    return 0

def format_time(seconds):
    if seconds <= 0 or seconds > 86400 * 7:
        return "--:--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"

def save_status(curr, file_size, speed_mb, eta_sec, status="uploading"):
    os.makedirs("logs", exist_ok=True)
    pct = (curr / file_size) * 100
    data = {
        "status": status,
        "uploaded_bytes": curr,
        "uploaded_gb": round(curr / (1024**3), 2),
        "total_bytes": file_size,
        "total_gb": round(file_size / (1024**3), 2),
        "progress_pct": round(pct, 2),
        "speed_mb_s": round(speed_mb, 2),
        "eta_seconds": round(eta_sec, 1),
        "eta_formatted": format_time(eta_sec),
        "timestamp": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(STATUS_FILE, "w") as fp:
        json.dump(data, fp, indent=2)

def finalize_kaggle_dataset(info_path):
    print("\n========================================================")
    print(" Finalizing Dataset Registration on Kaggle...")
    print("========================================================")
    sys.stdout.flush()

    if info_path and os.path.exists(info_path):
        try:
            with open(info_path, "r") as fp:
                d = json.load(fp)
            d["upload_complete"] = True
            with open(info_path, "w") as fp:
                json.dump(d, fp, indent=1)
            print("[INFO] Updated Kaggle local cache upload_complete=True")
        except Exception as e:
            print(f"[WARN] Could not update cache file: {e}")

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = "KGAT_606ae9d5fc752194af3ad226030798e0"
    cmd = ["/Users/krishnakant/Library/Python/3.9/bin/kaggle", "datasets", "create", "-p", "data/kaggle_upload/"]
    print(f"Running: {' '.join(cmd)}")
    sys.stdout.flush()

    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("Output:\n", res.stdout)
    if res.stderr:
        print("Stderr:\n", res.stderr)
    if res.returncode == 0:
        print("\n🎉 SUCCESS: Kaggle dataset created successfully!")
    else:
        print(f"\n[ERROR] Kaggle CLI returned exit code {res.returncode}")
    sys.stdout.flush()

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Error: Target file not found: {FILE_PATH}")
        sys.exit(1)

    file_size = os.path.getsize(FILE_PATH)
    url, info_path = find_kaggle_upload_info()

    print("========================================================")
    print(" Robust Resumable Kaggle Dataset Uploader")
    print(f" Target File: {FILE_PATH} ({file_size / (1024**3):.2f} GB)")
    print(" Chunk Size:  16 MB (with auto-retry and timeout protection)")
    print("========================================================")
    sys.stdout.flush()

    offset = get_current_offset(url, file_size)
    pct = (offset / file_size) * 100
    print(f"[SERVER CHECK] Current confirmed offset: {offset / (1024**3):.2f} GB / {file_size / (1024**3):.2f} GB ({pct:.2f}%)")
    sys.stdout.flush()

    if offset >= file_size:
        print("[INFO] Upload is already 100% complete on GCS!")
        save_status(file_size, file_size, 0, 0, status="completed")
        finalize_kaggle_dataset(info_path)
        return

    curr = offset
    start_session_time = time.time()
    uploaded_in_session = 0

    with open(FILE_PATH, "rb") as f:
        f.seek(curr)

        while curr < file_size:
            chunk_len = min(CHUNK_SIZE, file_size - curr)
            data = f.read(chunk_len)
            end_byte = curr + chunk_len - 1

            headers = {
                "Content-Length": str(chunk_len),
                "Content-Range": f"bytes {curr}-{end_byte}/{file_size}",
                "User-Agent": "Mozilla/5.0 KaggleResumer/2.0"
            }

            chunk_uploaded = False
            for attempt in range(1, 11):
                chunk_start = time.time()
                req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
                try:
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        if resp.status in (200, 201):
                            curr = file_size
                            chunk_uploaded = True
                            save_status(curr, file_size, 0, 0, status="completed")
                            print(f"\n[100.00%] Final chunk acknowledged by server!")
                            sys.stdout.flush()
                            finalize_kaggle_dataset(info_path)
                            return
                except urllib.error.HTTPError as e:
                    if e.code == 308:
                        range_hdr = e.headers.get("Range")
                        if range_hdr:
                            curr = int(range_hdr.split("-")[1]) + 1
                        else:
                            curr += chunk_len
                        chunk_uploaded = True
                        break
                    else:
                        err_body = e.read().decode("utf-8", errors="ignore")[:200]
                        print(f"  [ATTEMPT {attempt}/10] HTTP {e.code}: {err_body}. Retrying in {attempt*2}s...")
                        sys.stdout.flush()
                        time.sleep(attempt * 2)
                except Exception as e:
                    print(f"  [ATTEMPT {attempt}/10] Network error: {e}. Retrying in {attempt*2}s...")
                    sys.stdout.flush()
                    time.sleep(attempt * 2)

                # Re-query offset on retry
                try:
                    curr = get_current_offset(url, file_size)
                    f.seek(curr)
                except Exception:
                    pass

            if not chunk_uploaded:
                print(f"[FATAL] Failed to upload chunk after 10 attempts. Exiting.")
                save_status(curr, file_size, 0, 0, status="failed")
                sys.exit(1)

            uploaded_in_session += chunk_len
            elapsed = max(time.time() - start_session_time, 0.001)
            avg_speed = (uploaded_in_session / (1024 * 1024)) / elapsed
            rem_bytes = file_size - curr
            eta_sec = rem_bytes / (avg_speed * 1024 * 1024) if avg_speed > 0 else 0
            curr_pct = (curr / file_size) * 100

            save_status(curr, file_size, avg_speed, eta_sec, status="uploading")

            log_line = (
                f"[{curr_pct:5.2f}%] "
                f"{curr / (1024**3):5.2f} / {file_size / (1024**3):5.2f} GB | "
                f"Speed: {avg_speed:5.2f} MB/s | "
                f"ETA: {format_time(eta_sec)}"
            )
            print(log_line)
            sys.stdout.flush()

    save_status(file_size, file_size, 0, 0, status="completed")
    finalize_kaggle_dataset(info_path)

if __name__ == "__main__":
    main()
