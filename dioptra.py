"""
═══════════════════════════════════════════════════════════════════════════════
  DIOPTRA v1.1 — Single-File Implementation
  An Ultra-Lightweight Geometry-Aware Architecture for Monocular Metric Depth
═══════════════════════════════════════════════════════════════════════════════

Dioptra (formerly codenamed Tesseract) is an ultra-lightweight (8.1M parameter)
geometry-aware foundation model for monocular metric depth estimation on edge devices.
It incorporates Angular Residual Attention (ARA) and continuous trivision ray
positional embeddings to eliminate depth-scale ambiguity without needing massive
pretrained backbones.

Usage:
    python dioptra.py                # print config summary
    python dioptra.py --smoke        # forward pass smoke test
    python dioptra.py --count        # parameter count breakdown
    python dioptra.py --test         # run full unit test suite
"""

from tesseract import *
from tesseract import Dioptra, DioptraConfig, Tesseract, TesseractConfig

if __name__ == "__main__":
    from tesseract import main
    main()
