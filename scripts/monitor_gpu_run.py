#!/usr/bin/env python3
"""Monitor Kaggle GPU fine-tuning run for Dioptra-DINO."""

import json
import os
import sys
import time
import requests

TOKEN = os.environ.get("KAGGLE_KEY", "KGAT_16572d790f89195f4a7d8b2521ae067d")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
API = "https://www.kaggle.com/api/v1"

def check_run(user: str = "volsiai", slug: str = "dioptra-dino-highres-level1"):
    st_resp = requests.get(
        f"{API}/kernels/status",
        params={"userName": user, "kernelSlug": slug},
        headers=HEADERS,
    )
    if st_resp.status_code != 200:
        print(f"Failed to get status: {st_resp.status_code} {st_resp.text}")
        return False, None
    status_info = st_resp.json()
    status = status_info.get("status")
    failure = status_info.get("failureMessage", "")

    out_resp = requests.get(
        f"{API}/kernels/output",
        params={"userName": user, "kernelSlug": slug},
        headers=HEADERS,
    )
    log_text = ""
    if out_resp.status_code == 200:
        raw_log = out_resp.json().get("log", "")
        if raw_log:
            try:
                entries = json.loads(raw_log)
                lines = []
                for e in entries:
                    stream = e.get("stream_name", "")
                    data = e.get("data", "")
                    lines.append(f"[{stream}] {data}")
                log_text = "".join(lines)
            except Exception:
                log_text = raw_log

    return True, {"status": status, "failure": failure, "log": log_text}

if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "dioptra-dino-highres-level1"
    user = sys.argv[2] if len(sys.argv) > 2 else "volsiai"
    ok, info = check_run(user, slug)
    if ok:
        print(f"Kernel: {user}/{slug}")
        print(f"Status: {info['status']}")
        if info['failure']:
            print(f"Failure: {info['failure']}")
        print(f"Log size: {len(info['log'])} chars")
        if info['log']:
            print("\n--- Recent Log Output ---")
            print(info['log'][-2500:])
