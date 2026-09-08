# Running Dioptra on Apple Silicon (macOS)

This guide outlines setup, hardware acceleration, and profiling for Dioptra on Apple Silicon Macs (M1/M2/M3/M4) via Apple's Metal Performance Shaders (MPS) backend.

---

## 1. Requirements and Installation

Dioptra requires Python 3.10+ and standard PyTorch with MPS support (included by default in official PyTorch macOS arm64 wheels).

```bash
# Clone repository and enter directory
git clone https://github.com/SeranomTheGreat/dioptra.git
cd dioptra

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision numpy pillow matplotlib
```

---

## 2. Hardware Diagnostics and Model Checks

Run the built-in diagnostic suite to inspect device capabilities, memory allocation, and layer profiles:

```bash
# Verify MPS hardware acceleration and memory budget
python3 dioptra_mac.py --specs

# Forward pass smoke test (random input tensor)
python3 dioptra.py --smoke

# Run full 16-point unit test suite
python3 dioptra.py --test
```

---

## 3. Inference and 3D Mesh Export

Run depth prediction on a sample image and export a textured 3D mesh:

```bash
# Predict depth map
python3 dioptra_mac.py --predict assets/test_lobby.png

# Predict and export textured 3D mesh (.ply format)
python3 dioptra_mac.py --predict assets/test_lobby.png --export-ply output_mesh.ply
```

---

## 4. Benchmark Evaluation on Held-Out Challenge Frames

Evaluate the model against uncompressed ground-truth depth arrays across held-out environments:

```bash
python3 eval.py
```

Evaluation artifacts and comparison figures will be generated in `test_outputs/`.
