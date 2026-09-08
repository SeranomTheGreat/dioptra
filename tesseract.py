"""
Backwards-compatibility shim for Dioptra.
Please import from `dioptra` directly:
    from dioptra import Dioptra, DioptraConfig
"""
from dioptra import *
from dioptra import Dioptra, DioptraConfig, Tesseract, TesseractConfig

if __name__ == "__main__":
    from dioptra import main
    main()
