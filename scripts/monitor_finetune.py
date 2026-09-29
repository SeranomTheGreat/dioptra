#!/usr/bin/env python3
"""Monitor Kaggle kernel fine-tuning execution status and logs."""

import os
import sys
import time
import json
import requests

TOKEN = os.environ.get("KAGGLE_KEY", "KGAT_2d953a3819e6d87511286f79d2e294cf")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
API = "https://www.kaggle.com/api/v1"


def monitor(slug: str = "dioptra-dino-highres-robotics", user: str = "yumnamharryson", follow: bool = False, interval: int = 15):
    print(f"Monitoring Kaggle kernel: {user}/{slug}")
    print("=" * 60)

    last_log_len = 0

    while True:
        resp = requests.get(f"{API}/kernels/status", headers=HEADERS, params={"userName": user, "kernelSlug": slug})
        if resp.status_code != 200:
            print(f"Status check error ({resp.status_code}): {resp.text[:300]}")
            if not follow:
                break
            time.sleep(interval)
            continue

        data = resp.json()
        status = data.get("status", "unknown")
        failure = data.get("failureMessage", "")

        t_str = time.strftime("%H:%M:%S")
        print(f"[{t_str}] Kernel: {user}/{slug} | Status: {status.upper()}")
        if failure:
            print(f"  Failure Message: {failure}")

        # Fetch output logs
        resp_out = requests.get(f"{API}/kernels/output", headers=HEADERS, params={"userName": user, "kernelSlug": slug})
        if resp_out.status_code == 200:
            out_data = resp_out.json()
            log = out_data.get("log", "")
            if len(log) > last_log_len:
                new_content = log[last_log_len:]
                last_log_len = len(log)
                # Print clean stdout lines
                lines = [l for l in new_content.splitlines() if l.strip()]
                for l in lines[-10:]:
                    print(f"  >> {l}")

            if status == "complete":
                files = out_data.get("files", [])
                print("\nExecution Complete! Output files generated:")
                for f in files:
                    print(f"  - {f.get('fileName')} ({f.get('totalBytes', 0)/(1024*1024):.2f} MB)")
                break

        if status in ("complete", "error", "cancelAcknowledged"):
            break

        if not follow:
            break

        time.sleep(interval)


if __name__ == "__main__":
    follow_flag = "--follow" in sys.argv or "-f" in sys.argv
    target_slug = next((a for a in sys.argv[1:] if not a.startswith("-")), "dioptra-dino-highres-robotics")
    monitor(slug=target_slug, follow=follow_flag)
