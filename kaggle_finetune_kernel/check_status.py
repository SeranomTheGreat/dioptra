#!/usr/bin/env python3
"""Monitor Kaggle kernel execution status."""
import requests, json, sys, time

TOKEN = "KGAT_2d953a3819e6d87511286f79d2e294cf"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
API = "https://www.kaggle.com/api/v1"

KERNEL = "dioptra-dino-fine-tune-resume"
USER = "yumnamharryson"

resp = requests.get(f"{API}/kernels/status",
    headers=HEADERS,
    params={"userName": USER, "kernelSlug": KERNEL})

if resp.status_code == 200:
    data = resp.json()
    status = data.get("status", "unknown")
    failure = data.get("failureMessage", "")
    print(f"Kernel: {USER}/{KERNEL}")
    print(f"Status: {status}")
    if failure:
        print(f"Failure: {failure}")
    
    if status == "complete":
        # Try to get output
        resp2 = requests.get(f"{API}/kernels/output",
            headers=HEADERS,
            params={"userName": USER, "kernelSlug": KERNEL})
        if resp2.status_code == 200:
            output = resp2.json()
            print(f"\nOutput files: {json.dumps(output.get('files', []), indent=2)[:1000]}")
            if output.get("log"):
                print(f"\nLast log lines:")
                lines = output["log"].strip().split("\n")
                for line in lines[-30:]:
                    print(f"  {line}")
    elif status == "running":
        print("Training is actively running on Kaggle GPU...")
else:
    print(f"Error: {resp.status_code} {resp.text[:300]}")
