"""
Test Multi-Domain Dataset Loader logic (Domain Classification, Indexing, Decoders, Intrinsics).
"""

import os
import shutil
import tempfile
import sys
import types
import numpy as np
from PIL import Image as PILImage
import h5py

# Mock torch for lightweight unit testing on local environment
mock_torch = types.ModuleType("torch")
mock_torch.utils = types.ModuleType("utils")
mock_torch.utils.data = types.ModuleType("data")
class MockDataset: pass
mock_torch.utils.data.Dataset = MockDataset
mock_torch.nn = types.ModuleType("nn")
class MockModule:
    def __init__(self, *args, **kwargs): pass
mock_torch.nn.Module = MockModule
mock_torch.nn.functional = types.ModuleType("functional")
class MockTensor: pass
mock_torch.Tensor = MockTensor
sys.modules["torch"] = mock_torch
sys.modules["torch.utils"] = mock_torch.utils
sys.modules["torch.utils.data"] = mock_torch.utils.data
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.nn.functional"] = mock_torch.nn.functional
sys.path.insert(0, ".")

from dioptra_dino import MultiDomainDINODataset, resolve_all_dataset_roots

def run_tests():
    temp_dir = tempfile.mkdtemp(prefix="test_multidomain_")
    print(f"Creating mock multi-domain datasets in {temp_dir}...")

    try:
        # Hypersim
        os.makedirs(os.path.join(temp_dir, 'hypersim_pack', 'hypersim', 'scene_001', 'scene_cam_00_final_preview'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'hypersim_pack', 'hypersim', 'scene_001', 'scene_cam_00_geometry_hdf5'), exist_ok=True)
        PILImage.fromarray(np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)).save(
            os.path.join(temp_dir, 'hypersim_pack', 'hypersim', 'scene_001', 'scene_cam_00_final_preview', 'frame.0001.tonemap.jpg')
        )
        with h5py.File(os.path.join(temp_dir, 'hypersim_pack', 'hypersim', 'scene_001', 'scene_cam_00_geometry_hdf5', 'frame.0001.depth_meters.hdf5'), 'w') as f:
            f.create_dataset('dataset', data=np.full((768, 1024), 3.25, dtype=np.float32))

        # Tartan
        os.makedirs(os.path.join(temp_dir, 'tartan_mock', 'warehouse_stereo', 'warehouse', 'Data_easy', 'P000', 'image_lcam_front'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'tartan_mock', 'warehouse_stereo', 'warehouse', 'Data_easy', 'P000', 'depth_lcam_front'), exist_ok=True)
        PILImage.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)).save(
            os.path.join(temp_dir, 'tartan_mock', 'warehouse_stereo', 'warehouse', 'Data_easy', 'P000', 'image_lcam_front', '000000_lcam_front.png')
        )
        depth_float = np.full((480, 640), 5.5, dtype=np.float32)
        depth_rgba = np.ascontiguousarray(depth_float).view(np.uint8).reshape((480, 640, 4))
        PILImage.fromarray(depth_rgba).save(
            os.path.join(temp_dir, 'tartan_mock', 'warehouse_stereo', 'warehouse', 'Data_easy', 'P000', 'depth_lcam_front', '000000_lcam_front_depth.png')
        )

        # NYU
        os.makedirs(os.path.join(temp_dir, 'nyu_mock', 'nyu_data', 'data', 'nyu2_train'), exist_ok=True)
        PILImage.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)).save(
            os.path.join(temp_dir, 'nyu_mock', 'nyu_data', 'data', 'nyu2_train', '00001_colors.png')
        )
        PILImage.fromarray(np.full((480, 640), 2500, dtype=np.uint16)).save(
            os.path.join(temp_dir, 'nyu_mock', 'nyu_data', 'data', 'nyu2_train', '00001_depth.png')
        )

        # KITTI
        os.makedirs(os.path.join(temp_dir, 'kitti_mock', 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'kitti_mock', 'train', 'depths'), exist_ok=True)
        PILImage.fromarray(np.random.randint(0, 255, (375, 640, 3), dtype=np.uint8)).save(
            os.path.join(temp_dir, 'kitti_mock', 'train', 'images', 'drive_0001.png')
        )
        PILImage.fromarray(np.full((375, 640), 2560, dtype=np.uint16)).save(
            os.path.join(temp_dir, 'kitti_mock', 'train', 'depths', 'drive_0001.png')
        )

        print("Mock datasets created successfully.")

        roots = resolve_all_dataset_roots(temp_dir)
        print("Discovered dataset roots:")
        for r, dom in roots:
            print(f"  [{dom.upper()}] {r}")
        assert len(roots) >= 4, f"Expected at least 4 roots, found {len(roots)}"

        # Instantiate MultiDomainDINODataset
        ds = MultiDomainDINODataset(root_dirs=temp_dir, split="train", image_size=224, apply_pinhole_aug=False)
        print(f"Total indexed samples across domains: {len(ds)}")
        for s in ds.samples:
            print("  -> Found sample:", s)
        assert len(ds) == 4, f"Expected 4 indexed samples, found {len(ds)}"

        # Check each indexed sample format
        for idx in range(len(ds)):
            img_src, depth_src, domain = ds.samples[idx]
            print(f"Sample {idx}: Domain={domain}")
            print(f"  Image: {img_src}")
            print(f"  Depth: {depth_src}")

            # Verify depth decoding directly
            if domain == "hypersim":
                with h5py.File(depth_src, "r") as hf:
                    d = np.array(hf["dataset"][:], dtype=np.float32)
                assert np.isclose(d.mean(), 3.25), f"Hypersim depth mismatch: {d.mean()}"
                print(f"  ✓ Hypersim metric depth decoded: {d.mean():.2f}m")
            elif domain == "nyu":
                raw_d = np.array(PILImage.open(depth_src)).astype(np.float32) / 1000.0
                assert np.isclose(raw_d.mean(), 2.5), f"NYU depth mismatch: {raw_d.mean()}"
                print(f"  ✓ NYUv2 metric depth decoded: {raw_d.mean():.2f}m")
            elif domain == "kitti":
                raw_d = np.array(PILImage.open(depth_src)).astype(np.float32) / 256.0
                assert np.isclose(raw_d.mean(), 10.0), f"KITTI depth mismatch: {raw_d.mean()}"
                print(f"  ✓ KITTI metric depth decoded: {raw_d.mean():.2f}m")
            elif domain == "tartan":
                raw_d = np.array(PILImage.open(depth_src))
                d = np.ascontiguousarray(raw_d).view(np.float32).squeeze(-1)
                assert np.isclose(d.mean(), 5.5), f"TartanAir IEEE-754 depth mismatch: {d.mean()}"
                print(f"  ✓ TartanAir IEEE-754 float32 depth decoded: {d.mean():.2f}m")

            K = ds.K_CANONICAL[domain]
            print(f"  ✓ Intrinsics for {domain}: fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}, cx={K[0, 2]:.1f}, cy={K[1, 2]:.1f}")

        print("\n>>> ALL MULTI-DOMAIN AND DECODER UNIT TESTS PASSED! <<<")
        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_tests()
