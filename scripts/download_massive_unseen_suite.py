#!/usr/bin/env python3
"""
High-Speed Concurrent Downloader for 10+ Unseen TartanAir Benchmarks
Downloads matched image/depth pairs across multiple unseen domains.
"""

import os
import sys
import time
import glob
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.download_unseen_benchmarks import download_trajectory_fast

BENCHMARKS = [
    {"env": "amusement", "traj": "P001", "frames": 50},
    {"env": "hospital", "traj": "P001", "frames": 50},
    {"env": "ocean", "traj": "P000", "frames": 50},
    {"env": "seasonsforest", "traj": "P000", "frames": 50},
    {"env": "soulcity", "traj": "P000", "frames": 50},
    {"env": "oldtown", "traj": "P000", "frames": 50},
    {"env": "neighborhood", "traj": "P000", "frames": 50},
    {"env": "westerndesert", "traj": "P000", "frames": 50},
    {"env": "office2", "traj": "P000", "frames": 50},
    {"env": "abandonedfactory_night", "traj": "P001", "frames": 50},
]

def main():
    print("=" * 80)
    print("COMMENCING LARGE-SCALE UNSEEN BENCHMARK ACQUISITION")
    print(f"Targeting {len(BENCHMARKS)} diverse unseen environments (50 frames each)")
    print("=" * 80)

    t0 = time.time()
    success_count = 0

    for b in BENCHMARKS:
        try:
            print(f"\n>>> Downloading {b['env'].upper()} ({b['traj']}) - {b['frames']} frames...")
            download_trajectory_fast(
                env_name=b["env"],
                traj_id=b["traj"],
                difficulty="Easy",
                num_frames=b["frames"],
                save_dir="test_samples",
            )
            success_count += 1
        except Exception as e:
            print(f"Failed {b['env']}: {e}")

    print("\n" + "=" * 80)
    print(f"LARGE-SCALE UNSEEN DATASET ACQUISITION COMPLETE in {time.time() - t0:.1f}s")
    print(f"Successfully processed {success_count}/{len(BENCHMARKS)} environments.")
    print("=" * 80)

if __name__ == "__main__":
    main()
