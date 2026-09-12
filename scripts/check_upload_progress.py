#!/usr/bin/env python3
import json
import os
import sys
import subprocess

STATUS_FILE = "logs/kaggle_upload_status.json"
LOG_FILE = "logs/kaggle_upload.log"

def main():
    print("=" * 60)
    print(" Kaggle Dataset Upload Progress Monitor")
    print("=" * 60)

    # Check process
    try:
        ps = subprocess.check_output(["ps", "aux"]).decode()
        running = any("resume_kaggle_upload.py" in line for line in ps.splitlines())
    except Exception:
        running = False

    state_str = "🟢 RUNNING" if running else "⚪ NOT RUNNING (or finished)"
    print(f"Process Status: {state_str}")

    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE) as fp:
                data = json.load(fp)
            print(f"Upload State:   {data.get('status', 'unknown').upper()}")
            print(f"Progress:       {data.get('uploaded_gb', 0):.2f} GB / {data.get('total_gb', 0):.2f} GB ({data.get('progress_pct', 0):.2f}%)")
            print(f"Current Speed:  {data.get('speed_mb_s', 0):.2f} MB/s")
            print(f"Estimated ETA:  {data.get('eta_formatted', '--:--:--')}")
            print(f"Last Update:    {data.get('time_str', 'unknown')}")
        except Exception as e:
            print(f"Error reading status json: {e}")
    else:
        print("Status file not created yet.")

    print("-" * 60)
    print("Recent Log lines:")
    if os.path.exists(LOG_FILE):
        try:
            lines = subprocess.check_output(["tail", "-n", "5", LOG_FILE]).decode().strip().splitlines()
            for l in lines:
                print(" ", l)
        except Exception:
            pass
    else:
        print(" (No log file yet)")

    print("=" * 60)
    print("Commands:")
    print(" • Live stream logs: tail -f logs/kaggle_upload.log")
    print(" • Quick status:     python3 scripts/check_upload_progress.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
