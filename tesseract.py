"""
═══════════════════════════════════════════════════════════════════════════════
  DIOPTRA (formerly Tesseract) v1.1 (v17 Refinement) — Single-File Implementation
  An Ultra-Lightweight Geometry-Aware Architecture for Monocular Metric Depth
═══════════════════════════════════════════════════════════════════════════════

  Architecture (~8.1M parameters, no pretrained backbone):
    Input: image (B,3,224,224) + intrinsics K (B,3,3)
      │
      ├─ ConvStem (2-layer): 3→128 (k4s4) → 256 (k2s2) → 784 tokens @ D=256
      ├─ Trivision Rays: 3 rays per patch (centre/top-left/bottom-right)
      ├─ FiLM Ray PE: tokens = γ(rays) ⊙ tokens + β(rays)
      ├─ Transformer: 10 blocks, 8 heads, head_dim=32, MLP ratio 3.0
      │     Heads 0-3: Continuous Angular Relative Positional Encoding (CARPE / IRER)
      │     Heads 4-7: Standard attention (appearance)
      │     LayerScale(1e-5) + DropPath(0.0→0.1) + gradient checkpointing
      ├─ DPT Head: sparse reassembly (layers 3,6,9 with decoupled LayerNorms) → merge → 3× upsample
      └─ 3D Points: ray_centre × depth_at_patches

  Output: depth (B,1,224,224), points (B,784,3), rays (B,784,3,3)

  Training:
    - 8.5h time budget (Kaggle dual T4)
    - Batch 8 × accumulate 6 = effective 48
    - AdamW: base_lr=1e-4, head_lr=5e-4, weight_decay=0.05
    - Warm restarts: T₀=8, T_mult=1
    - IRER warmup gate (epochs 1→4, smoothstep)
    - Loss: SiLog + Log-L1 + multi-scale Sobel edge + image-aware smoothness + planarity
            (all uncertainty-weighted, Kendall 2018)

  Usage:
    python tesseract_v1.py                # print config summary
    python tesseract_v1.py --train /path/to/tartanair
    python tesseract_v1.py --train auto   # auto-find dataset under /kaggle/input
    python tesseract_v1.py --train /kaggle/input/dasvo-tartanair-rgb-d-validation-split
    python tesseract_v1.py --train /path/to/tartanair.zip  # zip-backed, no extraction
    python tesseract_v1.py --count        # parameter count
    python tesseract_v1.py --smoke        # forward pass smoke test

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations




# ════════════════════════════════════════════════════════════════════════════
# INLINED MODULES BELOW
# All relative imports (from .X import Y) have been inlined — each module's
# symbols are now available at the top level of this file.
# ════════════════════════════════════════════════════════════════════════════

#════════════════════════════════════════════════════════════════════════════#
# MODULE: config.py
#────────────────────────────────────────────────────────────────────────────#
"""
Configuration module for Tesseract v1.

All hyperparameters are centralized here for reproducibility and easy
experimentation. Values are drawn from the v1 build specification,
incorporating fixes from the v5.2 review.

References:
    - Default values follow the Tesseract v1 Architecture Specification
    - Optimizer settings follow AdamW (Loshchilov & Hutter, 2019)
    - Uncertainty weighting follows Kendall et al. (CVPR 2018)
"""


from dataclasses import dataclass, field, replace
from typing import Final, Tuple


# ---------------------------------------------------------------------------
# Physical constants & fixed architectural choices (not intended to change)
# ---------------------------------------------------------------------------

IMAGENET_MEAN: Final[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
"""ImageNet channel-wise mean for RGB normalization."""

IMAGENET_STD: Final[Tuple[float, float, float]] = (0.229, 0.224, 0.225)
"""ImageNet channel-wise std for RGB normalization."""

VALID_DEPTH_MIN: Final[float] = 0.1
"""Minimum physically valid depth in metres (excludes sky / LiDAR misses)."""

VALID_DEPTH_MAX: Final[float] = 200.0
"""Maximum physically valid depth in metres. Raised from 80 to 200 for TartanAir outdoor scenes."""

EIGEN_DELTA: Final[float] = 1.25
"""Eigen threshold base for delta_1/delta_2/delta_3 accuracies."""

GROUP_NORM_NUM_GROUPS: Final[int] = 8
"""Number of groups for GroupNorm layers (small-batch-friendly)."""

LOG_CLAMP_MIN: Final[float] = 1e-6
"""Lower clamp for log() arguments to avoid -inf."""

MIN_VALID_PIXELS: Final[int] = 10
"""Minimum valid pixels per sample for loss computation."""


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNG sources for reproducibility.

    Args:
        seed: Random seed.
        deterministic: If True, use deterministic algorithms (slower but reproducible).
    """
    import os
    import random
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Model architecture hyperparameters.

    The default configuration targets ~10.7M parameters (encoder 7.9M +
    DPT head 2.7M + Ray PE 0.1M + stem 0.05M), roughly half the size
    of DPT-ViT-S (~22M), suitable for dual T4 training.

    Attributes:
        img_size: Input image resolution (square).
        patch_size: ConvStem patch size. 8 → 28×28 = 784 tokens,
            enabling flash attention and deeper models (vs 3136 tokens
            at patch_size=4).
        embed_dim: Transformer embedding dimension. 256 with 8 heads
            gives head_dim=32.
        num_heads: Number of attention heads.
        num_layers: Transformer encoder depth. 10 layers (vs 6 in v5.2)
            for stronger 3D reasoning from scratch.
        mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio.
        num_freqs: Number of sinusoidal frequency bands for ray PE.
            6 (was 4 in v5.2) captures finer angular variation.
        drop_path_max: Maximum stochastic depth drop probability.
            Linear schedule from 0 at block 0 to this value at the
            last block.
        layer_scale_init: Initial value for LayerScale diagonal.
            1e-5 ensures near-identity residual at init, enabling
            stable deep training (DeRi / CaiT technique).
        num_reassemble_blocks: Number of DPT reassembly blocks (one
            per transformer layer).
        reassemble_mid_channels: Intermediate channels in each
            ReassembleBlock.
        merge_out_channels: Output channels after multi-scale merge.
        up_channels: Channel dims for the 3 progressive 2× upsample
            stages (28→56, 56→112, 112→224).
        disp_min: Minimum disparity offset. depth_max ≈ 1/disp_min.
        disp_max: Maximum disparity clamp. depth_min ≈ 1/disp_max.
        depth_min: Hard physical clamp on depth.
        depth_max: Hard physical clamp on depth.
        chunk_size: Chunk size for IRER attention bias computation.
            With N=784, chunk_size=256 gives only 3-4 loop iterations.
        use_flash_attention: Prefer SDPA flash attention backend when
            available.
        gradient_checkpointing: Enable per-block gradient checkpointing
            to trade compute for VRAM.
    """

    img_size: int = 224
    patch_size: int = 8
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 10
    mlp_ratio: float = 3.0
    num_freqs: int = 6
    drop_path_max: float = 0.1
    layer_scale_init: float = 1e-5
    num_reassemble_blocks: int = 10
    reassemble_mid_channels: int = 128
    merge_out_channels: int = 256
    up_channels: Tuple[int, int, int] = (128, 64, 32)
    # disp_min=0.0125 → depth_max=80m too aggressive for outdoor
    # TartanAir scenes which routinely reach 100-200m. Lowered to allow depth_max=200.
    disp_min: float = 0.005
    disp_max: float = 10.0
    depth_min: float = 0.1
    depth_max: float = 200.0
    chunk_size: int = 256
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    # --- Geometric/appearance head split ---
    num_geometric_heads: int = 4
    # --- IRER warmup schedule ---
    # original 10→40 unreachable under 13-epoch budget
    # (gate stayed ~0 the entire run). Set to 1→4 so IRER actually trains.
    irer_warmup_epochs: int = 1
    irer_full_epochs: int = 4
    # --- Sparse reassembly layers ---
    reassemble_layers: Tuple[int, ...] = (3, 6, 9)
    # --- FiLM modulation for positional encoding ---
    use_film_pe: bool = True
    # --- Ablation flags ---
    enable_irer: bool = True
    enable_trivision: bool = True
    pe_mode: str = "trivision"  # "trivision", "center_ray", or "none"

    def __post_init__(self) -> None:
        """Validate inter-field constraints. 
        so invalid combinations crashed later with cryptic errors."""
        if self.patch_size != 8:
            raise ValueError(f"patch_size must be 8 (ConvStem 2-layer), got {self.patch_size}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim={self.embed_dim} not divisible by num_heads={self.num_heads}")
        if self.img_size % self.patch_size != 0:
            raise ValueError(f"img_size={self.img_size} not divisible by patch_size={self.patch_size}")
        if not (0 < self.disp_min < self.disp_max):
            raise ValueError(f"Require 0 < disp_min < disp_max; got {self.disp_min}, {self.disp_max}")
        if max(self.reassemble_layers) >= self.num_layers:
            raise ValueError(
                f"reassemble_layers max {max(self.reassemble_layers)} >= num_layers {self.num_layers}"
            )
        if self.num_geometric_heads > self.num_heads:
            raise ValueError(
                f"num_geometric_heads {self.num_geometric_heads} > num_heads {self.num_heads}"
            )

    @property
    def head_dim(self) -> int:
        """Per-head dimension: embed_dim // num_heads."""
        assert self.embed_dim % self.num_heads == 0, (
            f"embed_dim={self.embed_dim} must be divisible by "
            f"num_heads={self.num_heads}"
        )
        return self.embed_dim // self.num_heads

    @property
    def num_patches(self) -> int:
        """Number of spatial tokens: (img_size / patch_size)²."""
        g = self.img_size // self.patch_size
        assert self.img_size % self.patch_size == 0, (
            f"img_size={self.img_size} must be divisible by "
            f"patch_size={self.patch_size}"
        )
        return g * g

    @property
    def grid_size(self) -> int:
        """Spatial grid dimension per side."""
        return self.img_size // self.patch_size


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        batch_size: Total batch across all GPUs. 8 → 4 per T4.
        accumulate_steps: Gradient accumulation steps. Effective
            batch = batch_size × accumulate_steps = 48.
        base_lr: Base learning rate for encoder weights.
        depth_head_lr: Learning rate multiplier for depth head
            (5× base, was 10× in v5.2 — too aggressive).
        min_lr: Minimum LR after cosine annealing.
        weight_decay: AdamW weight decay for weight tensors (ndim ≥ 2).
        weight_decay_bias: Weight decay for bias/norm params (ndim ≤ 1).
            Always 0.
        warmup_epochs: Linear LR warmup duration.
        total_epochs: Maximum training epochs.
        patience: Early stopping patience (improved from 10).
        min_delta: Minimum improvement to reset patience counter.
        grad_clip_norm: Per-group gradient clipping norm.
        grad_scaler_init: Initial scale for AMP GradScaler.
            1024 is conservative — prevents fp16 overflow in log ops.
        ema_decay: EMA decay (N/A for 3D-only model, no JEPA).
    """

    batch_size: int = 8
    accumulate_steps: int = 6
    base_lr: float = 1e-4
    depth_head_lr: float = 5e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    weight_decay_bias: float = 0.0
    # 30-epoch training budget
    warmup_epochs: int = 2
    total_epochs: int = 30
    patience: int = 8
    min_delta: float = 1e-4
    grad_clip_norm: float = 1.0
    # Conservative scaler init preventing FP16 underflow
    grad_scaler_init: float = 65536.0
    # DEPRECATED: not used anywhere; kept for backward compat.
    ema_decay: float = 0.0
    # --- Time-based early stopping ---
    time_limit_hours: float = 12.0
    # DEPRECATED: head_lr_* fields are unused (get_head_lr_multiplier is dead code).
    # Kept for backward compat; depth head uses constant cfg.depth_head_lr.
    head_lr_start: float = 10.0
    head_lr_end: float = 3.0
    head_lr_transition_epoch: int = 30
    # --- Warm restarts ---
    # T_0=15 gives two full 15-epoch cosine annealing cycles over 30 epochs.
    use_warm_restarts: bool = True
    restart_t0: int = 15
    restart_t_mult: int = 1
    # --- Validation interval ---
    val_interval: int = 2

    @property
    def effective_batch_size(self) -> int:
        """Effective batch = batch_size × accumulate_steps."""
        return self.batch_size * self.accumulate_steps


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossConfig:
    """Loss function hyperparameters.

    Attributes:
        silog_lambda_w: Lambda weighting in SiLog variance term.
            0.85 is effectively inert after median normalization
            (since mean(d) ≈ 0), so the loss degenerates to
            sqrt(Var[d]) — a valid loss.
        edge_threshold: Threshold for edge-aware masking. Skip
            loss computation where GT gradient exceeds this
            (occlusion boundaries).
        smooth_threshold: Laplacian threshold for planarity loss.
            Only penalize predicted high-frequency noise where
            GT is smooth (small Laplacian).
        charbonnier_eps: Epsilon for Charbonnier robust L2:
            sqrt(x² + eps²).
        num_tasks: Number of loss terms for uncertainty weighting.
    """

    silog_lambda_w: float = 0.85
    edge_threshold: float = 1.0
    smooth_threshold: float = 0.5
    charbonnier_eps: float = 1e-3
    num_tasks: int = 5
    # --- L1 auxiliary loss ---
    l1_weight: float = 0.1
    # --- Image-aware smoothness loss ---
    smooth_weight: float = 0.05
    smooth_image_grad_weight: float = 10.0  # exponential decay weight for image gradients
    # --- Multi-scale edge supervision ---
    edge_scales: Tuple[float, ...] = (0.25, 0.5, 1.0)
    # --- True Metric Supervision (v2) ---
    use_metric_supervision: bool = True
    scale_loss_weight: float = 1.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Dataset and augmentation hyperparameters.

    Attributes:
        val_batches: Number of validation batches per epoch.
        num_viz_samples: Number of samples for visualization.
        color_jitter_brightness: Brightness jitter factor.
        color_jitter_contrast: Contrast jitter factor.
        color_jitter_saturation: Saturation jitter factor.
        color_jitter_hue: Hue jitter factor.
        random_crop_scale: Scale range for random resized crop.
        split_mode: "cross_env" (strict zero-shot cross-environment split)
            or "cross_traj" (intra-environment cross-trajectory split).
        eval_min_depth: Minimum depth threshold for benchmark evaluation (metres).
        eval_max_depth: Maximum depth threshold for benchmark evaluation (metres).
    """

    val_batches: int = 200
    num_viz_samples: int = 16
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.05
    random_crop_scale: Tuple[float, float] = (0.6, 1.0)  # was (0.6, 1.4); >1.0 silently truncates via TF.crop
    # --- Intrinsics augmentation (K-jitter & Dynamic Pinhole Crop) ---
    enable_pinhole_crop_aug: bool = True  # dynamic optical crop forcing variable K
    pinhole_crop_min_scale: float = 0.55  # crop scale in [0.55, 1.0], giving FOV range ~45° to ~74°
    k_jitter_focal: float = 0.05  # subtle residual focal length jitter ±5%
    k_jitter_principal: float = 0.02  # principal point jitter ±2%
    # --- Photometric augmentations ---
    gamma_aug_prob: float = 0.5
    fog_aug_prob: float = 0.3
    # --- Geometric augmentations ---
    rotation_aug_prob: float = 0.3
    rotation_max_deg: float = 5.0
    # --- Aspect ratio ---
    preserve_aspect_ratio: bool = True
    # --- Cross-environment split mode ---
    split_mode: str = "cross_env"
    # --- Benchmark evaluation caps ---
    eval_min_depth: float = 0.2
    eval_max_depth: float = 80.0


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurriculumConfig:
    """Curriculum learning schedule by difficulty.

    Attributes:
        phase1_end: Epoch where Phase 1 ends (Easy only).
        phase2_end: Epoch where Phase 2 ends (Easy + Medium).
        phase1_easy: Fraction of Easy samples in Phase 1.
        phase2_easy: Fraction of Easy samples in Phase 2.
        phase2_medium: Fraction of Medium samples in Phase 2.
        phase3_easy: Fraction of Easy in Phase 3.
        phase3_medium: Fraction of Medium in Phase 3.
        phase3_hard: Fraction of Hard in Phase 3.
    """

    # 30-epoch curriculum schedule
    phase1_end: int = 10
    phase2_end: int = 20
    phase1_easy: float = 1.0
    phase2_easy: float = 0.5
    phase2_medium: float = 0.5
    phase3_easy: float = 0.3
    phase3_medium: float = 0.3
    phase3_hard: float = 0.4


# ---------------------------------------------------------------------------
# Combined configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TesseractConfig:
    """Top-level configuration container.

    Aggregates all sub-configs for a single entry point. All fields
    are frozen (immutable) after construction to prevent accidental
    mutation during training.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def summary(self) -> str:
        """Return a human-readable summary of all hyperparameters."""
        lines = [
            "=" * 60,
            "Tesseract v1 — Configuration Summary",
            "=" * 60,
            "",
            "── Model ──",
            f"  Image size:        {self.model.img_size}×{self.model.img_size}",
            f"  Patch size:        {self.model.patch_size}",
            f"  Tokens:            {self.model.num_patches} "
            f"({self.model.grid_size}×{self.model.grid_size})",
            f"  Embed dim:         {self.model.embed_dim}",
            f"  Heads:             {self.model.num_heads} "
            f"(head_dim={self.model.head_dim})",
            f"  Layers:            {self.model.num_layers}",
            f"  MLP ratio:         {self.model.mlp_ratio}",
            f"  Ray PE freqs:      {self.model.num_freqs}",
            f"  Drop path max:     {self.model.drop_path_max}",
            f"  LayerScale init:   {self.model.layer_scale_init}",
            f"  Disp range:        [{self.model.disp_min}, {self.model.disp_max}]",
            f"  Depth range:       [{self.model.depth_min}, {self.model.depth_max}]",
            f"  Trivision PE:      {self.model.enable_trivision} (mode={self.model.pe_mode})",
            f"  IRER Attention:    {self.model.enable_irer}",
            "",
            "── Training ──",
            f"  Batch size:        {self.train.batch_size}",
            f"  Accumulate:        {self.train.accumulate_steps}",
            f"  Effective batch:   {self.train.effective_batch_size}",
            f"  Base LR:           {self.train.base_lr:.1e}",
            f"  Depth head LR:     {self.train.depth_head_lr:.1e}",
            f"  Weight decay:      {self.train.weight_decay}",
            f"  Warmup:            {self.train.warmup_epochs} epochs",
            f"  Total epochs:      {self.train.total_epochs}",
            f"  Patience:          {self.train.patience}",
            f"  Grad clip:         {self.train.grad_clip_norm}",
            "",
            "── Loss ──",
            f"  SiLog λ_w:         {self.loss.silog_lambda_w}",
            f"  Metric supervision: {self.loss.use_metric_supervision}",
            f"  Scale loss weight:  {self.loss.scale_loss_weight}",
            f"  Edge threshold:    {self.loss.edge_threshold}",
            f"  Smooth threshold:  {self.loss.smooth_threshold}",
            f"  Uncertainty tasks:  {self.loss.num_tasks}",
            "",
            "── Data ──",
            f"  Split mode:        {self.data.split_mode}",
            f"  Pinhole crop aug:  {self.data.enable_pinhole_crop_aug} (min_scale={self.data.pinhole_crop_min_scale})",
            f"  Eval depth range:  [{self.data.eval_min_depth}m, {self.data.eval_max_depth}m]",
            f"  Val batches:       {self.data.val_batches}",
            f"  Viz samples:       {self.data.num_viz_samples}",
            "=" * 60,
        ]
        return "\n".join(lines)


# Primary alias: DioptraConfig
DioptraConfig = TesseractConfig


#════════════════════════════════════════════════════════════════════════════#
# MODULE: geometry.py
#────────────────────────────────────────────────────────────────────────────#
"""
Geometry primitives for Tesseract v1.

Provides:
    - Trivision 3-ray positional encoding computation
    - IRER (Inverse Epipolar Residual) bias for Continuous 3D Attention
    - SE(3) exponential / logarithmic maps (retained for future 4D extension)

The trivision ray encoding is the core geometric inductive bias, retained
from v5.2. For each 8×8 patch, we compute 3 camera rays (centre,
top-left, bottom-right) that sample the continuous epipolar geometry
within the patch.

References:
    - Epipolar geometry: Hartley & Zisserman, "Multiple View Geometry",
      Cambridge UP, 2004
    - IRER bias design: derived from the property that points on an
      epipolar line satisfy r × p = 0 for ray direction r and 3D point p
"""


import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# Trivision ray computation
# ======================================================================

def _inv3x3(m: Tensor) -> Tensor:
    """Analytic inverse of a batch of 3x3 matrices (adjugate / det).

    Mathematically identical to torch.linalg.inv for invertible inputs,
    but built only from elementwise ops and stacking:

      * No LAPACK / cuSOLVER backend → immune to the lazily-initialised
        linalg runtime whose init is NOT thread-safe: under
        nn.DataParallel's replica threads, a concurrent first call to
        torch.linalg.solve / inv / pinv crashes with
        "RuntimeError: lazy wrapper should be called at most once"
        (pytorch/pytorch#90613; observed on Kaggle dual-T4).
      * Fully differentiable and thread-safe by construction.
      * ~30 FLOPs per matrix — cheaper than a library call for 3x3.

    Degenerate matrices (|det| < eps, det == 0) get a sign-preserving
    clamped determinant so the result stays finite; camera intrinsics
    always have det(K) = fx*fy > 0, so this guard is a non-case in
    practice (it merely mirrors the old pinv fallback's "finite but
    geometrically meaningless" behaviour for pathological inputs).
    """
    a, b, c = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    d, e, f = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    g, h, i = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    # Adjugate (transposed cofactor matrix), row by row
    adj = torch.stack([
        torch.stack([e * i - f * h, c * h - b * i, b * f - c * e], dim=-1),
        torch.stack([f * g - d * i, a * i - c * g, c * d - a * f], dim=-1),
        torch.stack([d * h - e * g, b * g - a * h, a * e - b * d], dim=-1),
    ], dim=-2)  # (B, 3, 3)

    # det = a(ei − fh) − b(di − fg) + c(dh − eg), expanded along row 0
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)  # (B,)

    # Sign-preserving |det| >= eps clamp (handles det == 0 exactly)
    eps = det.new_tensor(1e-8)
    det = torch.where(
        det.abs() >= eps, det, torch.where(det < 0, -eps, eps)
    )

    return adj / det.view(-1, 1, 1)  # (B, 3, 3)


def compute_trivision_rays(
    intrinsics: Tensor,
    grid_size: int,
    patch_size: int,
    img_size: int,
    is_flipped: Optional[Tensor] = None,
) -> Tensor:
    """Compute 3 camera rays per patch for Trivision positional encoding.

    For each patch, we sample 3 pixel locations — centre, corner 1,
    and corner 2 — and unproject them to unit ray directions
    using the camera intrinsics K. Under standard orientation, corner 1
    is top-left and corner 2 is bottom-right. When is_flipped is True
    (horizontal reflection), corner 1 reflects to top-right and corner 2
    reflects to bottom-left to preserve physical chiral correspondence.

    Args:
        intrinsics: Camera intrinsics matrix (B, 3, 3).
            Expected format:
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        grid_size: Number of patches per side (img_size / patch_size).
        patch_size: Spatial patch size in pixels.
        img_size: Input image size in pixels.
        is_flipped: Optional boolean tensor of shape (B,) or bool indicating
            which batch elements underwent horizontal mirroring.

    Returns:
        rays: Unit ray directions (B, N, 3, 3) where N = grid_size².
            rays[:, :, 0, :] = centre ray
            rays[:, :, 1, :] = corner 1 ray (top-left, or top-right if flipped)
            rays[:, :, 2, :] = corner 2 ray (bottom-right, or bottom-left if flipped)

    Shape notes:
        - For img_size=224, patch_size=8: grid_size=28, N=784
        - Each of the 3 rays is a 3D unit vector (x, y, z) in camera frame
    """
    B = intrinsics.shape[0]
    device = intrinsics.device
    original_dtype = intrinsics.dtype
    # Force float32 for intrinsics precision (camera params need high precision)
    intrinsics = intrinsics.float()
    dtype = torch.float32

    # Build patch grid in pixel coordinates
    # Patch centres: centre of each patch_size × patch_size cell
    patch_centres = torch.arange(grid_size, device=device, dtype=dtype) * patch_size + patch_size / 2.0
    # Shape: (grid_size,)

    cv, cu = torch.meshgrid(patch_centres, patch_centres, indexing="ij")
    # cu, cv: (grid_size, grid_size) — column and row coords

    half_p = patch_size / 2.0

    # Top corner: cv - half_p
    c1_v = cv - half_p
    # Bottom corner: cv + half_p
    c2_v = cv + half_p

    # Standard (unflipped): c1 is top-left (cu - half_p), c2 is bottom-right (cu + half_p)
    # Flipped (chiral reflection): c1 is top-right (cu + half_p), c2 is bottom-left (cu - half_p)
    if is_flipped is not None:
        if isinstance(is_flipped, bool):
            is_flipped_t = torch.tensor([is_flipped] * B, device=device, dtype=torch.bool)
        elif is_flipped.dim() == 0:
            is_flipped_t = is_flipped.expand(B)
        else:
            is_flipped_t = is_flipped.to(device=device, dtype=torch.bool)

        # Build per-batch u coordinates: (B, G, G, 3)
        cu_b = cu.unsqueeze(0).expand(B, -1, -1)
        c1_u_unflip = cu_b - half_p
        c1_u_flip = cu_b + half_p
        c2_u_unflip = cu_b + half_p
        c2_u_flip = cu_b - half_p

        mask_b = is_flipped_t.view(B, 1, 1)
        c1_u = torch.where(mask_b, c1_u_flip, c1_u_unflip)
        c2_u = torch.where(mask_b, c2_u_flip, c2_u_unflip)

        u_all = torch.stack([cu_b, c1_u, c2_u], dim=-1)  # (B, G, G, 3)
        v_all = torch.stack([cv, c1_v, c2_v], dim=-1).unsqueeze(0).expand(B, -1, -1, -1) # (B, G, G, 3)
    else:
        tl_u = cu - half_p
        br_u = cu + half_p
        u_all = torch.stack([cu, tl_u, br_u], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, G, G, 3)
        v_all = torch.stack([cv, c1_v, c2_v], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # (B, G, G, 3)

    N = grid_size * grid_size
    # Flatten spatial dims: (B, G, G, 3) → (B, N, 3)
    u_flat = u_all.reshape(B, N, 3)
    v_flat = v_all.reshape(B, N, 3)

    # Unproject pixels to rays: r = K^{-1} @ p.
    # this used torch.linalg.solve with a
    # torch.linalg.pinv fallback — both hit the lazily-loaded LAPACK /
    # cuSOLVER backend whose one-time init is not thread-safe. Under
    # nn.DataParallel's replica threads (Kaggle T4 x2), the concurrent
    # first call crashed with "RuntimeError: lazy wrapper should be
    # called at most once" (pytorch/pytorch#90613). The analytic 3x3
    # inverse below is pure elementwise tensor math: thread-safe by
    # construction, differentiable, and numerically equivalent for the
    # well-conditioned pinhole K used here (det = fx*fy > 0; a
    # sign-preserving determinant clamp keeps degenerate inputs finite,
    # replacing the old pinv fallback).
    ones = torch.ones(B, N, 3, device=device, dtype=dtype)
    pixel_homo = torch.stack([
        u_flat,  # (B, N, 3)
        v_flat,  # (B, N, 3)
        ones,    # (B, N, 3)
    ], dim=-1)  # (B, N, 3_points, 3)

    # Reshape for batched matmul: (B, 3, N*3) against (B, 3, 3)
    p_flat = pixel_homo.reshape(B, N * 3, 3).transpose(1, 2)  # (B, 3, N*3)
    K_inv = _inv3x3(intrinsics)               # (B, 3, 3)
    rays_flat = torch.bmm(K_inv, p_flat)      # (B, 3, N*3)
    rays = rays_flat.transpose(1, 2).reshape(B, N, 3, 3)  # (B, N, 3_points, 3)

    # Normalize to unit length
    rays = F.normalize(rays, p=2.0, dim=-1)

    # Cast back to original dtype
    return rays.to(dtype=original_dtype)  # (B, N, 3, 3)


# ======================================================================
# IRER (Inverse Epipolar Residual) bias
# ======================================================================

@torch.compiler.disable
def compute_irer_bias(
    query_rays: Tensor,
    key_points: Tensor,
    alpha: Tensor,
    sigma_sq: Tensor,
    chunk_size: int = 256,
) -> Tensor:
    """Compute continuous angular relative positional bias (CARPE / IRER) for 3D Attention.

    In a single perspective camera frame, camera rays originate from the optical
    center (0, 0, 0). The angular separation residual between a query ray r_q and a
    key ray r_k is:

        d²(r_q, r_k) = 1 - (r_q · r_k)² = sin²(θ_{q, k})  (for unit rays)

    The geometric attention bias is:
        bias(i, j) = -α / σ² × Σ_m sin²(θ(r_{i, m}, r_j))

    where m indexes the 3 trivision rays per query patch (centre, top-left, bottom-right),
    and α, σ² are learnable per-head scaling parameters.

    Args:
        query_rays: 3 camera rays per query token (B, H, N_q, 3_rays, 3).
        key_points: Key ray directions or 3D points (B, H, N_k, 3).
            If not already unit length, normalized internally.
        alpha: Learnable per-head IRER strength (H,).
            Positive values enforce angular localization.
        sigma_sq: Learnable per-head IRER variance (H,).
            Controls the softness of the angular constraint.
        chunk_size: Chunk size for memory-efficient computation.
            With N=784, chunk_size=256 gives 3-4 iterations.

    Returns:
        bias: Attention bias (B, H, N_q, N_k) to be added to QK^T/√d before softmax.
            More negative = stronger suppression of off-axis attention.
    """
    B, H, N_q, _, _ = query_rays.shape
    N_k = key_points.shape[2]
    device = query_rays.device
    original_dtype = query_rays.dtype

    # Force float32 for geometric angular residuals (O(1e-4))
    query_rays = query_rays.float()
    key_points = key_points.float()
    alpha = alpha.float()
    sigma_sq = sigma_sq.float()
    dtype = torch.float32

    # Normalize key vectors to unit length (guarantees exact cos(theta) dot products)
    key_points_norm = F.normalize(key_points, p=2.0, dim=-1)  # (B, H, N_k, 3)

    # Reshape alpha and sigma_sq for broadcasting: (1, H, 1, 1)
    alpha_bc = alpha.view(1, H, 1, 1)
    sigma_sq_bc = sigma_sq.view(1, H, 1, 1).clamp(min=1e-6)  # avoid division by zero

    # Compute bias in chunks to manage memory
    bias = torch.zeros(B, H, N_q, N_k, device=device, dtype=dtype)

    for q_start in range(0, N_q, chunk_size):
        q_end = min(q_start + chunk_size, N_q)
        # rays_chunk: (B, H, chunk, 3_rays, 3)
        rays_chunk = query_rays[:, :, q_start:q_end]

        for k_start in range(0, N_k, chunk_size):
            k_end = min(k_start + chunk_size, N_k)
            # keys_chunk: (B, H, chunk, 3)
            keys_chunk = key_points_norm[:, :, k_start:k_end]

            # Dot product: cos(theta) between query ray and key ray
            dot = torch.einsum("bhqrc,bhkc->bhqrk", rays_chunk, keys_chunk)

            # d² = 1 - (r·p)² = sin²(theta)  (since ||r||=||p||=1)
            d_sq = 1.0 - dot * dot  # (B, H, Cq, 3r, Ck)

            # Clamp for numerical stability (d² should be in [0, 1])
            d_sq = d_sq.clamp(min=0.0, max=1.0)

            # Sum over 3 query rays: Σ_m d²(r_{q,m}, r_k)
            d_sq_sum = d_sq.sum(dim=3)  # (B, H, Cq, Ck)

            # CARPE / IRER angular bias: -α / σ² × d²_sum
            chunk_bias = -alpha_bc / sigma_sq_bc * d_sq_sum

            bias[:, :, q_start:q_end, k_start:k_end] = chunk_bias

    # Cast back to original dtype
    return bias.to(dtype=original_dtype)


def compute_pairwise_sin2_theta(
    query_rays: Tensor,
    key_points: Tensor,
    chunk_size: int = 256,
) -> Tensor:
    """Compute layer-shared pairwise angular residual matrix D(i, j).

    D(i, j) = sum_{m=1}^3 sin^2(theta(r_{i, m}, r_j))

    Since ray directions depend only on camera calibration K and token patch coordinates,
    this (B, N_q, N_k) matrix is strictly layer-invariant and head-invariant.
    Precomputing it once per forward pass and reusing across all transformer blocks
    reduces ARA overhead by >90% (from ~50 ms down to ~1.26 ms on Apple Silicon M3),
    yielding bit-for-bit identical attention biases.

    Args:
        query_rays: 3 camera rays per query token (B, N_q, 3, 3).
        key_points: Center key ray direction per key token (B, N_k, 3).
        chunk_size: Chunk size for memory-efficient block matrix multiplication.

    Returns:
        sin2_theta: (B, N_q, N_k) tensor of pairwise angular residuals.
    """
    B, N_q, _, _ = query_rays.shape
    N_k = key_points.shape[1]
    device = query_rays.device

    key_points_norm = F.normalize(key_points.float(), p=2.0, dim=-1)
    rays_f = query_rays.float()
    d_sq_sum = torch.zeros(B, N_q, N_k, device=device, dtype=torch.float32)

    for q_start in range(0, N_q, chunk_size):
        q_end = min(q_start + chunk_size, N_q)
        rays_chunk = rays_f[:, q_start:q_end]
        for k_start in range(0, N_k, chunk_size):
            k_end = min(k_start + chunk_size, N_k)
            keys_chunk = key_points_norm[:, k_start:k_end]
            dot = torch.einsum("bqrc,bkc->bqrk", rays_chunk, keys_chunk)
            d_sq = (1.0 - dot * dot).clamp(min=0.0, max=1.0)
            d_sq_sum[:, q_start:q_end, k_start:k_end] = d_sq.sum(dim=2)

    return d_sq_sum


# ======================================================================
# SE(3) exponential and logarithmic maps
# ======================================================================

def se3_exp(twist: Tensor) -> Tensor:
    """Compute the exponential map se(3) → SE(3).

    Converts a 6-DOF twist vector (v, ω) to a 4×4 rigid-body
    transformation matrix using the Rodrigues formula.

    Retained from v5.2 for future 4D extension (TemporalLoom).

    Args:
        twist: Twist vector (..., 6) where twist[..., :3] = v
            (linear velocity) and twist[..., 3:] = ω (angular velocity).

    Returns:
        T: SE(3) transformation matrix (..., 4, 4).

    References:
        - Murray, Li & Sastry, "A Mathematical Introduction to Robotic
          Manipulation", CRC Press, 1994
        - Implementation follows the numerically stable form with
          Taylor expansion for small angles.
    """
    v = twist[..., :3]   # (..., 3) linear
    w = twist[..., 3:]   # (..., 3) angular

    # Angle of rotation
    theta = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (..., 1)
    theta_sq = theta * theta

    # Skew-symmetric matrix [w]×
    wx = _skew_symmetric(w)  # (..., 3, 3)

    # Rodrigues formula for SO(3) exponential
    # R = I + sin(θ)/θ [w]× + (1-cos(θ))/θ² [w]×²
    I3 = torch.eye(3, device=twist.device, dtype=twist.dtype)
    I3 = I3.expand(wx.shape[:-2] + (3, 3))

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    wx_sq = torch.bmm(wx.view(-1, 3, 3), wx.view(-1, 3, 3)).view(wx.shape)

    R = I3 + (sin_theta / theta).unsqueeze(-1) * wx + \
        ((1.0 - cos_theta) / theta_sq).unsqueeze(-1) * wx_sq

    # V matrix for translation component
    # V = I + (1-cos(θ))/θ² [w]× + (θ-sin(θ))/θ³ [w]×²
    V = I3 + ((1.0 - cos_theta) / theta_sq).unsqueeze(-1) * wx + \
        ((theta - sin_theta) / (theta_sq * theta)).unsqueeze(-1) * wx_sq

    t = torch.einsum("...ij,...j->...i", V, v)  # (..., 3)

    # Assemble 4×4 matrix
    T = torch.zeros(twist.shape[:-1] + (4, 4), device=twist.device, dtype=twist.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0

    return T


def se3_log(T: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute the logarithmic map SE(3) → se(3).

    Converts a 4×4 rigid-body transformation to a 6-DOF twist vector.
    Retained from v5.2 for future 4D extension.

    Args:
        T: SE(3) transformation matrix (..., 4, 4).
        eps: Small constant for numerical stability.

    Returns:
        twist: Twist vector (..., 6).

    Raises:
        ValueError: If rotation matrix trace < -1 (invalid rotation).
    """
    R = T[..., :3, :3]  # (..., 3, 3)
    t = T[..., :3, 3]   # (..., 3)

    # acos is numerically unstable near θ=0 and θ=π (derivative → inf).
    # Use atan2(sin_θ, cos_θ) which is stable everywhere.
    trace = R.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (...)
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    # Stable sin_θ: for a rotation matrix ||R - R^T||_F = 2√2·sin(θ).
    # original divided by √2 only — a factor of 2
    # too large (clamped away for θ > π/6, corrupting θ thereafter).
    sin_theta = (R - R.transpose(-2, -1)).norm(dim=(-2, -1)) / (2.0 * 2.0 ** 0.5)
    sin_theta = sin_theta.clamp(min=0.0, max=1.0)
    theta = torch.atan2(sin_theta, cos_theta)  # stable everywhere
    theta_sq = theta * theta

    # Logarithmic map for SO(3): [w]× = θ / (2 sin(θ)) (R - R^T)
    # Safe clamp on sin_theta to avoid division by zero near θ=0 or π
    sin_theta_safe = sin_theta.clamp(min=eps)

    wx = (theta / (2.0 * sin_theta_safe)).unsqueeze(-1).unsqueeze(-1) * (R - R.transpose(-2, -1))

    # Extract ω from skew-symmetric [w]×
    w = torch.stack([wx[..., 2, 1], wx[..., 0, 2], wx[..., 1, 0]], dim=-1)  # (..., 3)

    # V inverse for translation
    # V^{-1} = I - 0.5 [w]× + (1 - θ sin(θ)/(2(1-cos(θ))))/θ² [w]×²
    I3 = torch.eye(3, device=T.device, dtype=T.dtype).expand(R.shape)
    wx_sq = torch.bmm(wx.view(-1, 3, 3), wx.view(-1, 3, 3)).view(wx.shape)

    # Handle small angles with Taylor expansion (use 1e-3 threshold for fp32).
    # the old code unsqueezed small_angle and A
    # independently, producing (…,1,1,1,1) broadcasts that CRASHED with
    # "Tensors must have same number of dimensions" on unbatched input and
    # silently mis-broadcast on batched input. Keep everything at (…,1,1)
    # so it broadcasts cleanly against the (…,3,3) matrices.
    small = theta < 1e-3                       # (…)
    small_angle = small[..., None, None]       # (…, 1, 1)
    # Stable A: piecewise Taylor below 1e-3 rad
    A_small = 1.0 / 12.0 - theta_sq / 720.0
    A_general = (1.0 - theta * sin_theta_safe / (2.0 * (1.0 - cos_theta + 1e-30))) / (theta_sq + 1e-30)
    A = torch.where(small, A_small, A_general)[..., None, None]  # (…, 1, 1)

    V_inv = I3 - 0.5 * wx + A * wx_sq
    # For small angles, V_inv → I (identity)
    V_inv = torch.where(small_angle, I3, V_inv)

    v = torch.einsum("...ij,...j->...i", V_inv, t)  # (..., 3)

    return torch.cat([v, w], dim=-1)  # (..., 6)


def _skew_symmetric(v: Tensor) -> Tensor:
    """Construct skew-symmetric matrix [v]× from 3-vector v.

    [v]× = [[ 0, -v3,  v2],
             [v3,  0,  -v1],
             [-v2, v1,  0 ]]

    Args:
        v: Input vector (..., 3).

    Returns:
        Skew-symmetric matrix (..., 3, 3).
    """
    zeros = torch.zeros_like(v[..., 0])
    M = torch.stack([
        zeros,    -v[..., 2],  v[..., 1],
        v[..., 2],  zeros,    -v[..., 0],
        -v[..., 1], v[..., 0],  zeros,
    ], dim=-1)
    return M.reshape(v.shape[:-1] + (3, 3))


#════════════════════════════════════════════════════════════════════════════#
# MODULE: metrics.py
#────────────────────────────────────────────────────────────────────────────#
"""
Evaluation metrics for Tesseract v1.

Implements the 7 standard Eigen metrics for monocular depth estimation,
all computed after per-sample median normalization.

Key fix from v5.2: RMSE computed as sqrt(sum((pred-gt)²) / count)
NOT mean(sqrt(mean((pred-gt)²))) — the latter gives a different (wrong)
result because it averages per-sample RMSEs instead of computing RMSE
over all pixels.

Metrics:
    1. abs_rel  — Absolute relative difference: |pred - gt| / gt
    2. rmse     — Root mean squared error
    3. log10    — Mean log10 ratio
    4. silog    — Scale-invariant log error (same as loss but without sqrt)
    5. d1       — Threshold accuracy δ₁: % of pixels with max(pred/gt, gt/pred) < 1.25
    6. d2       — Threshold accuracy δ₂: % with max < 1.25²
    7. d3       — Threshold accuracy δ₃: % with max < 1.25³

References:
    - Eigen et al., NIPS 2014
    - Ranftl et al., "Towards Robust Monocular Depth Estimation", TPAMI 2022
"""


from typing import Dict, Tuple

import torch
from torch import Tensor


def compute_eigen_metrics(
    pred: Tensor,
    gt: Tensor,
    valid: Tensor,
    min_depth: float = 0.2,
    max_depth: float = 80.0,
) -> Dict[str, float]:
    """Compute benchmark metrics with dual reporting (metric unaligned & scale-aligned).

    Depth evaluation range is capped to [min_depth, max_depth] (standard 0.2m - 80m
    for TartanAir/outdoor benchmark protocols).

    Args:
        pred: Predicted depth (B, 1, H, W).
        gt: Ground-truth depth (B, 1, H, W).
        valid: Valid pixel mask (B, 1, H, W).
        min_depth: Minimum depth evaluation threshold in metres (default: 0.2).
        max_depth: Maximum depth evaluation threshold in metres (default: 80.0).

    Returns:
        Dict with keys:
            - Aligned relative metrics: "abs_rel", "rmse", "log10", "silog", "d1", "d2", "d3"
            - Unaligned metric metrics: "metric_abs_rel", "metric_rmse", "metric_d1"
    """
    # Squeeze channel dim
    pred_s = pred.squeeze(1)   # (B, H, W)
    gt_s = gt.squeeze(1)
    valid_s = valid.squeeze(1)

    B = pred_s.shape[0]

    # Accumulators for scale-aligned metrics
    sum_sq_err = 0.0
    total_pixels = 0
    sum_abs_rel = 0.0
    sum_log10 = 0.0
    sum_silog_sq = 0.0
    sum_silog_mean = 0.0
    count_d1 = 0
    count_d2 = 0
    count_d3 = 0

    # Accumulators for unaligned true metric metrics
    sum_metric_abs_rel = 0.0
    sum_metric_sq_err = 0.0
    count_metric_d1 = 0

    n_valid_samples = 0

    for b in range(B):
        # Apply evaluation depth cap [min_depth, max_depth] and valid finite mask
        v = (
            valid_s[b].bool()
            & (gt_s[b] >= min_depth)
            & (gt_s[b] <= max_depth)
            & (pred_s[b] > 0)
            & torch.isfinite(gt_s[b])
            & torch.isfinite(pred_s[b])
        )
        if v.sum() == 0:
            continue
        n_valid_samples += 1

        p_raw = pred_s[b][v].float()
        g = gt_s[b][v].float()
        n = p_raw.numel()
        total_pixels += n

        # 1. Unaligned Metric Metrics (zero test-time alignment)
        sum_metric_abs_rel += ((p_raw - g).abs() / g.clamp(min=1e-6)).sum().item()
        sum_metric_sq_err += ((p_raw - g) ** 2).sum().item()
        thresh_raw = torch.max(p_raw / g.clamp(min=1e-6), g / p_raw.clamp(min=1e-6))
        count_metric_d1 += (thresh_raw < EIGEN_DELTA).sum().item()

        # 2. Per-sample median normalization (Eigen relative protocol)
        scale = (g.median() / p_raw.median()).clamp(1e-4, 1e4)
        p = p_raw * scale

        # Absolute relative
        sum_abs_rel += ((p - g).abs() / g.clamp(min=1e-6)).sum().item()

        # RMSE accumulator (proper: sum of squared errors, then single sqrt)
        sum_sq_err += ((p - g) ** 2).sum().item()

        # log10
        ratio_log10 = torch.log10(p.clamp(min=1e-6)) - torch.log10(g.clamp(min=1e-6))
        sum_log10 += ratio_log10.abs().sum().item()

        # SiLog
        d = torch.log(p.clamp(min=1e-6)) - torch.log(g.clamp(min=1e-6))
        d_mean = d.mean()
        sum_silog_sq += (d ** 2).mean().item()
        sum_silog_mean += d_mean.item()

        # Threshold accuracies (Eigen δ₁/δ₂/δ₃)
        thresh = torch.max(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
        count_d1 += (thresh < EIGEN_DELTA).sum().item()
        count_d2 += (thresh < EIGEN_DELTA ** 2).sum().item()
        count_d3 += (thresh < EIGEN_DELTA ** 3).sum().item()

    if total_pixels == 0:
        return {
            k: float("nan")
            for k in [
                "abs_rel", "rmse", "log10", "silog", "d1", "d2", "d3",
                "metric_abs_rel", "metric_rmse", "metric_d1",
            ]
        }

    # Compute final metrics
    metrics = {
        # Aligned Eigen metrics (standard relative evaluation)
        "abs_rel": sum_abs_rel / total_pixels,
        "rmse": (sum_sq_err / total_pixels) ** 0.5,
        "log10": sum_log10 / total_pixels,
        "silog": max(0.0, sum_silog_sq / n_valid_samples - (sum_silog_mean / n_valid_samples) ** 2) ** 0.5,
        "d1": count_d1 / total_pixels,
        "d2": count_d2 / total_pixels,
        "d3": count_d3 / total_pixels,
        # Unaligned Metric metrics (pure metric navigation evaluation)
        "metric_abs_rel": sum_metric_abs_rel / total_pixels,
        "metric_rmse": (sum_metric_sq_err / total_pixels) ** 0.5,
        "metric_d1": count_metric_d1 / total_pixels,
    }

    return metrics


def compute_per_scene_metrics(
    pred: Tensor,
    gt: Tensor,
    valid: Tensor,
    scene_names: Tuple[str, ...],
    min_depth: float = 0.2,
    max_depth: float = 80.0,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compute metrics overall and per-scene with dual reporting and depth cap.

    Also identifies top-5 best and worst scenes by abs_rel, grouped
    by environment name.

    Args:
        pred: Predicted depth (B, 1, H, W).
        gt: Ground-truth depth (B, 1, H, W).
        valid: Valid pixel mask (B, 1, H, W).
        scene_names: Scene name for each sample in the batch.
        min_depth: Minimum depth evaluation threshold in metres.
        max_depth: Maximum depth evaluation threshold in metres.

    Returns:
        overall: Dict of overall metrics.
        per_scene: Dict mapping scene name to per-scene metrics.
    """
    overall = compute_eigen_metrics(pred, gt, valid, min_depth=min_depth, max_depth=max_depth)

    B = pred.shape[0]
    per_scene: Dict[str, Dict[str, float]] = {}

    for b in range(B):
        scene = scene_names[b] if b < len(scene_names) else f"scene_{b}"
        p_b = pred[b:b+1]
        g_b = gt[b:b+1]
        v_b = valid[b:b+1]
        per_scene[scene] = compute_eigen_metrics(p_b, g_b, v_b, min_depth=min_depth, max_depth=max_depth)

    return overall, per_scene


#════════════════════════════════════════════════════════════════════════════#
# MODULE: viz.py
#────────────────────────────────────────────────────────────────────────────#
"""
Visualization utilities for Tesseract v1.

4-panel visualization: RGB | GT Depth | Predicted Depth | Log-Ratio Error

Key fixes from v5.2:
    - Median-normalized predictions for display (raw predictions are at
      arbitrary scale and misleading)
    - vmax from GT only (not polluted by prediction outliers)
    - Log-ratio error map: |log(pred) - log(gt)| reveals multiplicative
      errors better than absolute difference

References:
    - Eigen visualization protocol
"""


import logging
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def save_visualization(
    rgb: Tensor,
    gt_depth: Tensor,
    pred_depth: Tensor,
    save_path: str,
    valid: Optional[Tensor] = None,
) -> None:
    """Save 4-panel visualization: RGB | GT | Pred | Error.

    All depth maps use the same colormap range (vmax from GT only)
    and median-normalized predictions for fair visual comparison.

    Args:
        rgb: RGB image (3, H, W) in [0, 1] or ImageNet-normalized.
        gt_depth: Ground-truth depth (H, W) or (1, H, W).
        pred_depth: Predicted depth (H, W) or (1, H, W).
        save_path: Output file path (e.g., "viz_001.png").
        valid: Valid pixel mask (H, W) or None.
    """
    import matplotlib
    # force=True — in a Kaggle/Jupyter session pyplot
    # may already be loaded with an interactive backend; without force the
    # switch is ignored and figure rendering can try to hit a display.
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Ensure 2D depth maps
    if gt_depth.dim() == 3:
        gt_depth = gt_depth.squeeze(0)
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(0)

    # --- Median normalization for display ---
    if valid is not None:
        if valid.dim() == 3:
            valid = valid.squeeze(0)
        v = valid.bool()
    else:
        v = gt_depth > 0

    if v.sum() > 0:
        scale = (gt_depth[v].median() / pred_depth[v].median().clamp(min=1e-6)).clamp(1e-4, 1e4)
        pred_display = pred_depth * scale
    else:
        pred_display = pred_depth

    # --- Robust vmin / vmax using percentiles to prevent sky outliers from crushing contrast ---
    if v.sum() > 0:
        valid_vals = gt_depth[v].float()
        vmin = max(0.0, torch.quantile(valid_vals, 0.01).item())
        vmax = torch.quantile(valid_vals, 0.95).item()
        if vmax <= vmin:
            vmax = vmin + 5.0
    else:
        vmin, vmax = 0.0, 10.0

    # --- Denormalize RGB if ImageNet-normalized ---
    rgb_display = rgb.clone()
    if rgb.min() < 0:  # likely ImageNet-normalized
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(rgb.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(rgb.device)
        rgb_display = rgb * std + mean
    rgb_display = rgb_display.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    # --- Log-ratio error ---
    error = torch.abs(
        torch.log(pred_display.clamp(min=1e-4)) - torch.log(gt_depth.clamp(min=1e-4))
    )
    error = error.cpu().numpy()

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb_display)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(gt_depth.cpu().numpy(), cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title("GT Depth")
    axes[1].axis("off")

    axes[2].imshow(pred_display.cpu().numpy(), cmap="magma", vmin=vmin, vmax=vmax)
    axes[2].set_title("Pred Depth (median-norm)")
    axes[2].axis("off")

    im = axes[3].imshow(error, cmap="inferno", vmin=0, vmax=2.0)
    axes[3].set_title("Log-Ratio Error")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_per_scene_breakdown(
    per_scene: dict,
    save_path: str,
) -> None:
    """Save per-scene metric breakdown with top-5 best/worst scenes.

    Args:
        per_scene: Dict mapping scene name to metric dict.
        save_path: Output text file path.
    """
    if not per_scene:
        return

    # Sort by abs_rel
    sorted_scenes = sorted(per_scene.items(), key=lambda x: x[1].get("abs_rel", float("inf")))

    lines = [
        "Per-Scene Breakdown (sorted by abs_rel)",
        "=" * 60,
        "",
        "Top-5 Best Scenes:",
    ]
    for name, metrics in sorted_scenes[:5]:
        lines.append(f"  {name}: abs_rel={metrics.get('abs_rel', float('nan')):.4f}")

    lines.append("")
    lines.append("Top-5 Worst Scenes:")
    for name, metrics in sorted_scenes[-5:]:
        lines.append(f"  {name}: abs_rel={metrics.get('abs_rel', float('nan')):.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def export_point_cloud_ply(
    image: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
    save_path: str,
    valid: Optional[Tensor] = None,
) -> int:
    """Export unprojected 3D point cloud as a colored ASCII .ply file.

    Compatible with MeshLab, CloudCompare, Blender, and macOS QuickLook / 3D viewers.

    Args:
        image: RGB image (3, H, W) in [0, 1] or ImageNet-normalized.
        depth: Depth map (H, W) in metres.
        intrinsics: Camera intrinsics K (3, 3).
        save_path: Destination .ply file path.
        valid: Optional valid pixel mask (H, W), bool.

    Returns:
        vertex_count: Number of exported 3D points.
    """
    import numpy as np

    if image.dim() == 4:
        image = image.squeeze(0)
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    if valid is not None and valid.dim() == 3:
        valid = valid.squeeze(0)

    H, W = depth.shape[-2:]

    # Denormalize image to [0, 255] uint8 RGB
    rgb = image.detach().float().cpu()
    if rgb.min() < 0:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = rgb * std + mean
    rgb = (rgb.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).numpy()

    # Unproject pixels to 3D camera coordinates: p = d * K^{-1} [u, v, 1]^T
    depth_np = depth.detach().float().cpu().numpy()
    K_np = intrinsics.detach().float().cpu().numpy()
    fx, fy = K_np[0, 0], K_np[1, 1]
    cx, cy = K_np[0, 2], K_np[1, 2]

    # Coordinate grids
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Valid mask
    if valid is not None:
        mask = valid.detach().bool().cpu().numpy() & np.isfinite(depth_np) & (depth_np > 0.05)
    else:
        mask = np.isfinite(depth_np) & (depth_np > 0.05) & (depth_np < 200.0)

    z = depth_np[mask]
    x = (uu[mask] - cx) * z / fx
    y = -(vv[mask] - cy) * z / fy  # Upright: +Y is UP in standard 3D graphics (OpenGL/QuickLook/Blender/Three.js)
    colors = rgb[mask]

    n_vertices = int(z.shape[0])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Created by Tesseract 3D Foundation Model (Upright +Y)\n")
        f.write(f"element vertex {n_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n_vertices):
            f.write(f"{x[i]:.4f} {y[i]:.4f} {z[i]:.4f} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")

    return n_vertices


def export_mesh_ply(
    image: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
    save_path: str,
    valid: Optional[Tensor] = None,
    max_depth_ratio: float = 1.18,
) -> int:
    """Export unprojected 3D scene as a solid triangular mesh (.ply) with faces.

    Connects adjacent depth pixels into triangular polygons, filtering out
    steep depth discontinuities (occlusion boundaries) so objects do not
    smear into the background. Compatible with macOS QuickLook, Blender, MeshLab.

    Args:
        image: RGB image (3, H, W) in [0, 1] or ImageNet-normalized.
        depth: Depth map (H, W) in metres.
        intrinsics: Camera intrinsics K (3, 3).
        save_path: Destination .ply file path.
        valid: Optional valid pixel mask (H, W), bool.
        max_depth_ratio: Maximum allowable depth ratio between adjacent vertices
                         to form a triangular face (default 1.18).

    Returns:
        face_count: Number of exported triangular faces.
    """
    import numpy as np

    if image.dim() == 4:
        image = image.squeeze(0)
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    if valid is not None and valid.dim() == 3:
        valid = valid.squeeze(0)

    H, W = depth.shape[-2:]

    # Denormalize image to [0, 255] uint8 RGB
    rgb = image.detach().float().cpu()
    if rgb.min() < 0:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = rgb * std + mean
    rgb_np = (rgb.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).numpy()

    depth_np = depth.detach().float().cpu().numpy()
    K_np = intrinsics.detach().float().cpu().numpy()
    fx, fy = K_np[0, 0], K_np[1, 1]
    cx, cy = K_np[0, 2], K_np[1, 2]

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Valid mask
    if valid is not None:
        mask = valid.detach().bool().cpu().numpy() & np.isfinite(depth_np) & (depth_np > 0.05)
    else:
        mask = np.isfinite(depth_np) & (depth_np > 0.05) & (depth_np < 200.0)

    x = (uu - cx) * depth_np / fx
    y = -(vv - cy) * depth_np / fy  # Upright: +Y is UP
    z = depth_np

    # Build vertex list and 2D index lookup
    v_idx = np.full((H, W), -1, dtype=int)
    vertices = []
    colors = []
    cur = 0
    for r in range(H):
        for c in range(W):
            if mask[r, c]:
                vertices.append((x[r, c], y[r, c], z[r, c]))
                colors.append(rgb_np[r, c])
                v_idx[r, c] = cur
                cur += 1

    # Connect adjacent pixels into triangular faces
    faces = []
    for r in range(H - 1):
        for c in range(W - 1):
            i00 = v_idx[r, c]
            i10 = v_idx[r + 1, c]
            i01 = v_idx[r, c + 1]
            i11 = v_idx[r + 1, c + 1]
            if i00 >= 0 and i10 >= 0 and i01 >= 0 and i11 >= 0:
                z_quad = [z[r, c], z[r + 1, c], z[r, c + 1], z[r + 1, c + 1]]
                min_z, max_z = min(z_quad), max(z_quad)
                if min_z > 0 and (max_z / min_z) < max_depth_ratio:
                    faces.append((i00, i10, i01))
                    faces.append((i01, i10, i11))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Created by Tesseract 3D Foundation Model (Solid Mesh)\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for (vx, vy, vz), (cr, cg, cb) in zip(vertices, colors):
            f.write(f"{vx:.4f} {vy:.4f} {vz:.4f} {cr} {cg} {cb}\n")
        for f0, f1, f2 in faces:
            f.write(f"3 {f0} {f1} {f2}\n")

    return len(faces)


#════════════════════════════════════════════════════════════════════════════#
# MODULE: checkpoint.py
#────────────────────────────────────────────────────────────────────────────#
"""
Checkpoint management for Tesseract v1.

Implements atomic save/load with full RNG state for exact reproducibility.

Key features:
    - Saves ALL GPU RNG states (torch.cuda.get_rng_state_all()),
      not just GPU 0 
    - Saves CPU, Python, and NumPy RNG states
    - Corruption recovery: if checkpoint is corrupted, rename it and
      start from scratch instead of crashing
    - Atomic writes: write to temp file, then rename (prevents
      half-written checkpoints on crash)

References:
    - PyTorch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
"""


import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    batch_size: int,
    best_metric: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save training checkpoint atomically with full RNG state.

    Writes to a temporary file first, then renames to the target path.
    This prevents half-written checkpoints if the process is killed
    during save.

    Args:
        path: Target checkpoint path.
        model: Model to save.
        optimizer: Optimizer to save.
        scheduler: LR scheduler to save.
        scaler: AMP GradScaler to save.
        epoch: Current epoch.
        batch_size: Current batch size (may have been halved by OOM).
        best_metric: Best validation metric so far.
        extra: Additional state to include.
    """
    # Handle DataParallel — save the underlying model's state_dict
    # to avoid 'module.' prefix mismatch on load. When model is wrapped in
    # nn.DataParallel, model.state_dict() has keys prefixed with 'module.'
    # which won't match the unwrapped model during load_checkpoint.
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    state: Dict[str, Any] = {
        "tesseract": model_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "batch_size": batch_size,
        "best_metric": best_metric,
        # RNG states (fix from v5.2 — include ALL GPUs)
        "rng_cpu": torch.random.get_rng_state(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
    }

    # Save ALL GPU RNG states, not just GPU 0
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()

    if extra is not None:
        state.update(extra)

    # Atomic write: temp file → rename
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(
            dir=str(Path(path).parent),
            suffix=".tmp",
            delete=False,
        ) as tmp:
            torch.save(state, tmp.name)
            # fsync while the file is still open for
            # WRITING (POSIX fsync may fail with EBADF on read-only fds) and
            # BEFORE the rename, so data is durable before the file becomes
            # the checkpoint.
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name

        # Atomic rename (POSIX)
        os.replace(tmp_path, path)
        logger.info(f"Checkpoint saved: {path} (epoch={epoch})")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        # Clean up temp file if it exists
        if "tmp_path" in locals() and os.path.isfile(tmp_path):
            os.unlink(tmp_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Any = None,
    device: str = "cpu",
) -> Tuple[int, int, float]:
    """Load training checkpoint with corruption recovery.

    If the checkpoint file is corrupted (e.g., truncated due to OOM
    during save), rename it with a ".corrupt" suffix and return
    default values to start from scratch.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        optimizer: Optimizer to load state into (None → skip).
        scheduler: LR scheduler to load state into (None → skip).
        scaler: AMP GradScaler to load state into (None → skip).
        device: Device to map tensors to.

    Returns:
        epoch: Resume epoch (0 if starting from scratch).
        batch_size: Resume batch size.
        best_metric: Resume best metric (inf if starting from scratch).
    """
    path = str(path)

    # clean up orphan .tmp files from previous SIGKILL mid-saves
    parent = Path(path).parent
    if parent.is_dir():
        for orphan in parent.glob("*.tmp"):
            try:
                orphan.unlink()
                logger.info(f"Removed orphan tmp file: {orphan}")
            except OSError:
                pass

    if not os.path.isfile(path):
        logger.info(f"No checkpoint found at {path}, starting from scratch")
        return 0, 8, float("inf")

    try:
        # weights_only=True prevents arbitrary code execution via pickle
        # (security risk for untrusted checkpoints). Fall back to weights_only=False
        # only if the checkpoint contains non-tensor state (older format).
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except Exception as e_wo:
            logger.warning(f"weights_only=True failed ({e_wo}); trying without (less secure)")
            state = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        logger.warning(f"Checkpoint corrupted: {e}. Starting from scratch.")
        # Rename corrupted file to prevent future load attempts
        corrupt_path = path + ".corrupt"
        if os.path.isfile(path):
            os.rename(path, corrupt_path)
            logger.info(f"Renamed corrupted checkpoint to {corrupt_path}")
        return 0, 8, float("inf")

    # Load model weights
    # Strip 'module.' prefix from checkpoint keys if the model
    # is NOT wrapped in DataParallel. This happens when the checkpoint
    # was saved from a DataParallel-wrapped model but loaded into an
    # unwrapped model (the normal flow: load → then wrap in DataParallel).
    saved_state = state["tesseract"]
    if not isinstance(model, nn.DataParallel):
        needs_strip = any(k.startswith("module.") for k in saved_state.keys())
        if needs_strip:
            saved_state = {k.replace("module.", "", 1): v for k, v in saved_state.items()}
            logger.info("Stripped 'module.' prefix from checkpoint keys (DataParallel → unwrapped)")

    try:
        model.load_state_dict(saved_state)
    except RuntimeError as e:
        logger.warning(f"Partial model load (key mismatch): {e}")
        # Use strict=False for partial loads (e.g., after architecture change)
        model.load_state_dict(saved_state, strict=False)

    # Load optimizer state
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    # Load scheduler state
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    # Load scaler state
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    # Restore RNG states for exact reproducibility
    if "rng_cpu" in state:
        torch.random.set_rng_state(state["rng_cpu"])
    if "rng_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    if "rng_python" in state:
        random.setstate(state["rng_python"])
    if "rng_numpy" in state:
        np.random.set_state(state["rng_numpy"])

    epoch = state.get("epoch", 0)
    batch_size = state.get("batch_size", 8)
    best_metric = state.get("best_metric", float("inf"))

    logger.info(
        f"Checkpoint loaded: {path} "
        f"(resume at epoch {epoch}, batch_size={batch_size}, best_metric={best_metric:.4f})"
    )

    return epoch, batch_size, best_metric


def load_pretrained_weights(
    path: str,
    model: nn.Module,
    device: str = "cpu",
) -> Tuple[int, float]:
    """Load model weights from checkpoint for Stage 2 fine-tuning or transfer learning.

    Leaves optimizer, scheduler, and scaler untouched so training starts with
    a fresh fine-tuning schedule without stale momentum states.

    Args:
        path: Checkpoint file path.
        model: Tesseract model instance.
        device: Device to map tensors to.

    Returns:
        (source_epoch, source_best_metric)
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    try:
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint at {path}: {e}")

    saved_state = state.get("tesseract", state)
    if not isinstance(model, nn.DataParallel):
        saved_state = {k.replace("module.", "", 1): v for k, v in saved_state.items()}

    # Strip auxiliary loss_fn keys if present (managed separately in training loop)
    saved_state = {k: v for k, v in saved_state.items() if not k.startswith("loss_fn.")}

    incompatible = model.load_state_dict(saved_state, strict=False)
    source_epoch = state.get("epoch", 0)
    source_best = state.get("best_metric", float("inf"))

    logger.info(
        f"Stage 2 Pretrained weights loaded: {path} (epoch {source_epoch}, "
        f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
    )
    return source_epoch, source_best


#════════════════════════════════════════════════════════════════════════════#
# MODULE: model.py
#────────────────────────────────────────────────────────────────────────────#
"""
Neural network modules for Tesseract v1.

Architecture overview:
    Input: (B, 3, 224, 224) + K (B, 3, 3)
      │
      ├─ ConvStem: Conv2d(3, 256, k=8, s=8) + GELU + LayerNorm → (B, 784, 256)
      ├─ Trivision PE: 3 rays per patch → sinusoidal(freq=6) → MLP → (B, 784, 256)
      │  tokens = tokens + ray_pe
      ├─ Transformer Encoder: 10 blocks (PreNorm + IRER Attention + LayerScale + DropPath)
      │  Extract intermediate features from ALL 10 blocks
      ├─ DPT Depth Head: Multi-scale reassembly → progressive 2× upsample → disparity
      └─ 3D Point Reconstruction: rays × depth_at_patches

Output: depth (B, 1, 224, 224), points (B, 784, 3)

Key design decisions (from v5.2 hard-won experience):
    - patch_size=8 → 784 tokens (vs 3136) enables flash attention
    - Pre-norm + LayerScale(1e-5) for stable deep training
    - GroupNorm instead of BatchNorm (small batch sizes)
    - Progressive 2× upsample (NOT single 8× ConvTranspose2d)
    - Bilinear-like init for all ConvTranspose2d layers

References:
    - DPT: Ranftl et al., ICCV 2021
    - LayerScale: Touvron et al., "Going deeper with Image Transformers", ICCV 2022
    - Stochastic depth: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016
"""


import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from .config import ModelConfig
# from .geometry import compute_irer_bias, compute_trivision_rays


# ======================================================================
# ConvStem — ViT-style single conv patchifier
# ======================================================================

class ConvStem(nn.Module):
    """Two-layer convolution patchifier with gradual downsample.

    Replaces single Conv2d(3, 256, k=8, s=8) with a 2-layer stem:
      Conv2d(3, 128, k=4, s=4) + GELU + LN → Conv2d(128, 256, k=2, s=2) + GELU + LN
    This gives 4×2=8 effective patch size with smoother feature extraction.
    stem_dim = embed_dim // 2 (128 for embed_dim=256).

    Args:
        in_channels: Number of input image channels (3 for RGB).
        embed_dim: Token embedding dimension.
        patch_size: Spatial patch size (must be 8 for 4×2 factorization).
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 256, patch_size: int = 8):
        super().__init__()
        if patch_size != 8:
            raise ValueError(f"Two-layer stem requires patch_size=8, got {patch_size}")
        stem_dim = embed_dim // 2  # 128
        self.proj1 = nn.Conv2d(in_channels, stem_dim, kernel_size=4, stride=4)
        self.norm1 = nn.GroupNorm(GROUP_NORM_NUM_GROUPS, stem_dim)
        self.proj2 = nn.Conv2d(stem_dim, embed_dim, kernel_size=2, stride=2)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tuple[Tensor, int]:
        """Patchify image into token sequence.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            tokens: Patch embeddings (B, N, D) where N = (H/p) × (W/p).
            grid_size: Patches per side (H / patch_size).
        """
        B, C, H, W = x.shape
        # assert is stripped under python -O; use explicit raise
        if H != W:
            raise ValueError(f"Expected square image, got {H}×{W}")
        if H % self.patch_size != 0:
            raise ValueError(
                f"Image size {H} must be divisible by patch_size {self.patch_size}"
            )

        # Layer 1: (B, 3, H, W) → (B, stem_dim, H/4, W/4)
        x = self.proj1(x)
        x = self.norm1(x)
        x = F.gelu(x)

        # Layer 2: (B, stem_dim, H/4, W/4) → (B, embed_dim, H/8, W/8)
        x = self.proj2(x)

        grid_size = H // self.patch_size

        # Flatten spatial dims: (B, D, G, G) → (B, D, N) → (B, N, D)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm2(tokens)

        return tokens, grid_size


# ======================================================================
# Trivision Ray Positional Encoding
# ======================================================================

class RayPositionalEncoding(nn.Module):
    """Trivision 3-ray positional encoding with sinusoidal frequency mapping.

    For each patch, 3 camera rays (centre, top-left, bottom-right) are
    encoded via sinusoidal positional encoding at multiple frequency bands,
    then projected through an MLP to match the token embedding dimension.

    Improvement from v5.2 review: num_freqs increased from 4 to 6,
    adding bands at frequencies 16 and 32 for finer angular resolution.

    Architecture:
        Input: 9 raw ray coords + 9×2×6=108 freq features = 117 total
        MLP: Linear(117, D) → GELU → Linear(D, D)

    Args:
        embed_dim: Token embedding dimension.
        num_freqs: Number of sinusoidal frequency bands (6 in v1, was 4).
    """

    def __init__(self, embed_dim: int = 256, num_freqs: int = 6, use_film_pe: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.use_film_pe = use_film_pe
        self.raw_dim = 9  # 3 rays × 3 coordinates
        self.freq_dim = self.raw_dim * 2 * num_freqs  # 9 × 2 × 6 = 108
        self.mlp_input_dim = self.raw_dim + self.freq_dim  # 9 + 108 = 117

        if use_film_pe:
            # FiLM modulation: MLP outputs 2*embed_dim, split into γ and β
            # tokens = γ(rays) * tokens + β(rays)
            #
            # Identity init — zero the last Linear's weights and
            # set γ-slice of bias to 1.0 so FiLM starts as identity (γ=1, β=0).
            # Default PyTorch init produces γ ~ N(0, 0.04^2) which destroys the
            # token signal at init, especially under LayerScale(1e-5).
            self.film_mlp = nn.Sequential(
                nn.Linear(self.mlp_input_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 2 * embed_dim),
            )
            nn.init.zeros_(self.film_mlp[-1].weight)
            nn.init.zeros_(self.film_mlp[-1].bias)
            with torch.no_grad():
                self.film_mlp[-1].bias[:embed_dim].fill_(1.0)  # γ slice = 1
        else:
            # Original additive PE: MLP outputs embed_dim
            self.mlp = nn.Sequential(
                nn.Linear(self.mlp_input_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Precompute frequency bands: 2^0, 2^1, ..., 2^(L-1)
        self.register_buffer(
            "freq_bands",
            torch.tensor([2.0 ** i for i in range(num_freqs)]),
        )

    def forward(self, rays: Tensor) -> Tuple[Tensor, bool]:
        """Compute ray positional encoding.

        Args:
            rays: Unit ray directions (B, N, 3, 3) from compute_trivision_rays.

        Returns:
            If use_film_pe=True:
                film_params: (γ, β) tuple, each (B, N, D), for FiLM modulation.
            Else:
                ray_pe: Positional encoding (B, N, D) to be added to token embeddings.
            is_film: Boolean indicating if FiLM mode is used.
        """
        B, N, _, _ = rays.shape
        device = rays.device
        dtype = rays.dtype

        # Flatten 3 rays × 3 coords → (B, N, 9)
        raw = rays.reshape(B, N, self.raw_dim)

        # Sinusoidal encoding at multiple frequency bands
        # For each freq band f: [sin(2π f x), cos(2π f x)] for each of 9 coords
        # freq_bands: (num_freqs,)
        # raw: (B, N, 9) → (B, N, 9, 1) × (1, 1, 1, num_freqs) → (B, N, 9, num_freqs)
        raw_unsqueezed = raw.unsqueeze(-1)  # (B, N, 9, 1)
        angles = 2.0 * math.pi * raw_unsqueezed * self.freq_bands.to(device=device, dtype=dtype)

        # sin and cos: (B, N, 9, num_freqs) each
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)

        # Interleave: (B, N, 9, 2, num_freqs) → (B, N, 9 × 2 × num_freqs)
        freq_features = torch.stack([sin_features, cos_features], dim=-2)
        freq_features = freq_features.reshape(B, N, self.freq_dim)

        # Concatenate raw + freq: (B, N, 117)
        encoded = torch.cat([raw, freq_features], dim=-1)

        if self.use_film_pe:
            # FiLM: output 2*D, split into γ and β
            film_out = self.film_mlp(encoded)  # (B, N, 2*D)
            gamma, beta = film_out.chunk(2, dim=-1)  # each (B, N, D)
            return (gamma, beta), True
        else:
            # Additive PE: project through MLP
            return self.mlp(encoded), False


# ======================================================================
# LayerScale
# ======================================================================

class LayerScale(nn.Module):
    """Learnable diagonal scale on residual path.

    Initializes near-zero (1e-5) so the network starts as a near-identity
    map, enabling stable training of very deep transformers. This follows
    the CaiT / DeRi technique.

    Args:
        dim: Embedding dimension.
        init_value: Initial value for the diagonal scale (typically 1e-5).
    """

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply diagonal scaling: x * gamma (element-wise)."""
        return self.gamma * x


# ======================================================================
# DropPath (Stochastic Depth)
# ======================================================================

class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample.

    During training, randomly drops entire residual branches with
    probability drop_prob. During evaluation, acts as identity.

    The drop probability follows a linear schedule from 0 at block 0
    to drop_path_max at the last block.

    Args:
        drop_prob: Probability of dropping the path (0 = no drop).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        # Random tensor: keep if > drop_prob
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast over all dims except batch
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# ======================================================================
# Continuous 3D Attention with IRER
# ======================================================================

class Continuous3DAttention(nn.Module):
    """Multi-head attention with IRER (Inverse Epipolar Residual) bias.

    Combines standard scaled dot-product attention with a geometric bias
    derived from epipolar constraints. The IRER bias suppresses attention
    between tokens that are NOT on each other's epipolar lines.

    Key features:
        - Per-head learnable α and σ² for IRER (fix from v5.2)
        - Falls back to flash attention when IRER bias is not needed
        - Chunked computation for memory efficiency

    Args:
        embed_dim: Total embedding dimension across all heads.
        num_heads: Number of attention heads.
        chunk_size: Chunk size for IRER bias computation.
        use_flash: Whether to use flash attention when possible.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 chunk_size: int = 256, use_flash: bool = True,
                 num_geometric_heads: int = 4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} not divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size
        self.use_flash = use_flash
        self.num_geometric_heads = num_geometric_heads

        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)

        # original 'consolidated λ' made α=σ²=softplus(λ), so the
        # ratio α/σ²=1 always → the learnable parameter had no effect on the
        # bias magnitude. Restored separate α (strength) and σ² (sharpness)
        # per geometric head, so each head can independently tune how
        # aggressively it suppresses off-epipolar attention.
        # Only the first num_geometric_heads entries are used.
        self.irer_alpha_raw = nn.Parameter(torch.zeros(num_geometric_heads))
        self.irer_sigma_sq_raw = nn.Parameter(torch.zeros(num_geometric_heads))
        # irer_gate is NOT a learnable buffer — passed as forward arg.

    def forward(
        self,
        x: Tensor,
        query_rays: Optional[Tensor] = None,
        key_points: Optional[Tensor] = None,
        irer_gate: float = 1.0,
        cached_sin2_theta: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute attention with optional IRER bias.

        Args:
            x: Input tokens (B, N, D).
            query_rays: Camera rays per query token (B, N, 3, 3).
                None → standard attention (no IRER bias).
            key_points: 3D key points (B, N, 3).
                None → standard attention (no IRER bias).
            irer_gate: Multiplicative gate for IRER bias (0.0 = disabled, 1.0 = full).
            cached_sin2_theta: Precomputed layer-shared pairwise sin²(θ) residual matrix (B, N, N).

        Returns:
            Output tokens (B, N, D).
        """
        B, N, D = x.shape

        # QKV projection: (B, N, 3D) → 3 × (B, N, D)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)  # each (B, H, N, head_dim)

        # Build attention mask (IRER bias) if applicable
        attn_mask = None
        if (query_rays is not None or cached_sin2_theta is not None) and irer_gate > 0.0:
            attn_mask = self._compute_irer(
                query_rays, key_points, B, N, irer_gate,
                cached_sin2_theta=cached_sin2_theta,
            )
            # SDPA expects attn_mask broadcastable to (B, H, N, N)
            # and is added to the (Q @ K^T / sqrt(d)) prior to softmax.
            # under AMP autocast q/k/v are fp16 while
            # the IRER bias is computed in fp32; CUDA SDPA kernels require a
            # float attn_mask to MATCH the query dtype. Cast here (also halves
            # mask memory).
            if attn_mask.dtype != q.dtype:
                attn_mask = attn_mask.to(q.dtype)

        # use F.scaled_dot_product_attention (routes to flash/memory-efficient
        # backend) instead of manual matmul+softmax+matmul that materializes
        # the full (B, H, N, N) attention matrix (~320 MB at B=8, N=784, fp16).
        # SDPA handles scale internally (1/sqrt(head_dim)).
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        )  # (B, H, N, head_dim)

        # Concatenate heads: (B, N, D)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        return self.proj(attn_output)

    def _compute_irer(
        self,
        query_rays: Optional[Tensor],
        key_points: Optional[Tensor],
        B: int,
        N: int,
        irer_gate: float = 1.0,
        cached_sin2_theta: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute IRER bias with geometric/appearance head split.

        Only the first num_geometric_heads heads receive IRER bias;
        the remaining appearance heads get zero bias. Supports layer-shared
        caching via cached_sin2_theta to avoid redundant pairwise ray recalculation.

        Args:
            query_rays: (B, N, 3, 3) — 3 rays per token (or None if cached).
            key_points: (B, N, 3) — 3D point per token (or None if cached).
            B, N: Batch and sequence lengths.
            irer_gate: Multiplicative gate for IRER bias.
            cached_sin2_theta: Optional precomputed (B, N, N) matrix.

        Returns:
            bias: (B, H, N, N) attention bias.
        """
        # use separate α (strength) and σ² (sharpness) per head.
        # softplus enforces positivity; +1e-3 avoids degenerate σ²=0.
        alpha = F.softplus(self.irer_alpha_raw) + 1e-3  # (H_geo,)
        sigma_sq = F.softplus(self.irer_sigma_sq_raw) + 1e-3  # (H_geo,)

        if cached_sin2_theta is not None:
            alpha_bc = alpha.view(1, self.num_geometric_heads, 1, 1)
            sigma_sq_bc = sigma_sq.view(1, self.num_geometric_heads, 1, 1).clamp(min=1e-6)
            geo_bias = - (alpha_bc / sigma_sq_bc) * cached_sin2_theta.unsqueeze(1)
        else:
            # Expand rays for geometric heads only: (B, H_geo, N, 3, 3)
            rays_expanded = query_rays.unsqueeze(1).expand(-1, self.num_geometric_heads, -1, -1, -1)
            # Expand key points for geometric heads only: (B, H_geo, N, 3)
            keys_expanded = key_points.unsqueeze(1).expand(-1, self.num_geometric_heads, -1, -1)

            # Compute bias for geometric heads only
            geo_bias = compute_irer_bias(
                query_rays=rays_expanded,
                key_points=keys_expanded,
                alpha=alpha,
                sigma_sq=sigma_sq,
                chunk_size=self.chunk_size,
            )  # (B, H_geo, N, N)

        # Apply gate and assemble full bias (H_geo heads + H_app appearance heads)
        full_bias = torch.zeros(B, self.num_heads, N, N,
                                device=geo_bias.device, dtype=geo_bias.dtype)
        full_bias[:, :self.num_geometric_heads] = geo_bias * irer_gate

        # fp16 overflow guard. The bias is cast to
        # fp16 for SDPA on T4; if α/σ² diverges during training (σ² is
        # clamped only at 1e-6, α unbounded) the bias can exceed fp16 max
        # (65504) → inf → NaN after softmax. Softmax saturates long before
        # -1e4, so clamping there is lossless for attention but prevents
        # NaN blowups mid-run.
        full_bias = full_bias.clamp(min=-1.0e4)

        return full_bias


# ======================================================================
# Transformer Block
# ======================================================================

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with LayerScale and stochastic depth.

    Architecture:
        x = x + DropPath(LS1(Attention(LayerNorm(x))))
        x = x + DropPath(LS2(MLP(LayerNorm(x))))

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio.
        layer_scale_init: Initial value for LayerScale (1e-5).
        drop_path: Stochastic depth drop probability.
        chunk_size: Chunk size for IRER computation.
        use_flash: Whether to use flash attention.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 3.0,
        layer_scale_init: float = 1e-5,
        drop_path: float = 0.0,
        chunk_size: int = 256,
        use_flash: bool = True,
        num_geometric_heads: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Continuous3DAttention(embed_dim, num_heads, chunk_size, use_flash, num_geometric_heads)
        self.ls1 = LayerScale(embed_dim, layer_scale_init)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        self.ls2 = LayerScale(embed_dim, layer_scale_init)
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self,
        x: Tensor,
        query_rays: Optional[Tensor] = None,
        key_points: Optional[Tensor] = None,
        irer_gate: float = 1.0,
        cached_sin2_theta: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with optional IRER bias.

        Args:
            x: Input tokens (B, N, D).
            query_rays: Camera rays (B, N, 3, 3) or None.
            key_points: 3D key points (B, N, 3) or None.
            irer_gate: Multiplicative gate for IRER bias.
            cached_sin2_theta: Optional precomputed pairwise sin²(θ) residual matrix (B, N, N).

        Returns:
            Output tokens (B, N, D).
        """
        # Attention residual
        x = x + self.drop_path1(self.ls1(
            self.attn(self.norm1(x), query_rays, key_points, irer_gate, cached_sin2_theta=cached_sin2_theta)
        ))
        # MLP residual
        x = x + self.drop_path2(self.ls2(
            self.mlp(self.norm2(x))
        ))
        return x


# ======================================================================
# Transformer Encoder
# ======================================================================

class TransformerEncoder(nn.Module):
    """Stack of TransformerBlocks with linear stochastic depth schedule.

    Extracts intermediate features from ALL blocks for the multi-scale
    depth head.

    Args:
        num_layers: Number of transformer blocks (10 in v1, was 6 in v5.2).
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        layer_scale_init: LayerScale initial value.
        drop_path_max: Maximum drop path probability (linear schedule).
        chunk_size: Chunk size for IRER.
        use_flash: Use flash attention.
        gradient_checkpointing: Enable gradient checkpointing per block.
    """

    def __init__(
        self,
        num_layers: int = 10,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 3.0,
        layer_scale_init: float = 1e-5,
        drop_path_max: float = 0.1,
        chunk_size: int = 256,
        use_flash: bool = True,
        gradient_checkpointing: bool = True,
        num_geometric_heads: int = 4,
        reassemble_layers: Tuple[int, ...] = (3, 6, 9),
    ):
        super().__init__()
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.reassemble_layers = reassemble_layers

        # Linear stochastic depth schedule: 0 at block 0, drop_path_max at last
        dpr = [x * drop_path_max / max(num_layers - 1, 1) for x in range(num_layers)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                layer_scale_init=layer_scale_init,
                drop_path=dpr[i],
                chunk_size=chunk_size,
                use_flash=use_flash,
                num_geometric_heads=num_geometric_heads,
            )
            for i in range(num_layers)
        ])
        # UPGRADE (v17): Decouple LayerNorm per tapped reassembly layer.
        # Each layer (e.g. 3, 6, 9) learns its own scale and bias for the DPT head,
        # preventing coupling between shallow spatial and deep semantic representations.
        self.reassemble_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(len(self.reassemble_layers))
        ])
        # Fallback alias for backward compatibility with older checkpoints
        self.norm = self.reassemble_norms[-1]

    def forward(
        self,
        x: Tensor,
        query_rays: Optional[Tensor] = None,
        key_points: Optional[Tensor] = None,
        irer_gate: float = 1.0,
        enable_irer: bool = True,
        enable_trivision: bool = True,
        reassemble_layers: Optional[Tuple[int, ...]] = None,
    ) -> List[Tensor]:
        """Forward pass, returning features (only at reassemble_layers).

        Extract features from designated reassembly layers.
        but DPT only uses layers {3, 6, 9}. Now only stores those.

        Args:
            x: Input tokens (B, N, D).
            query_rays: Camera rays (B, N, 3, 3) or None.
            key_points: 3D key points (B, N, 3) or None.
            irer_gate: Multiplicative gate for IRER bias.
            enable_irer: If False, pass None rays/key_points (no IRER).
            enable_trivision: If False, skip ray PE (tokens unchanged).
            reassemble_layers: Tuple of layer indices to retain (None = all).

        Returns:
            intermediate_features: List of tensors, each (B, N, D).
        """
        if reassemble_layers is None:
            reassemble_list = list(self.reassemble_layers)
        else:
            reassemble_list = list(reassemble_layers)
        reassemble_set = set(reassemble_list)
        layer_to_idx = {l: idx for idx, l in enumerate(reassemble_list)}

        features = []

        # If IRER disabled, pass None for rays/key_points
        eff_rays = query_rays if enable_irer else None
        eff_keys = key_points if enable_irer else None
        eff_gate = irer_gate if enable_irer else 0.0

        # Layer-shared geometric caching: pairwise sin²(θ) residual matrix D(i, j)
        # depends purely on camera calibration K and token coordinates, making it
        # strictly layer- and head-invariant. Precomputing once per forward pass
        # cuts ARA forward latency by >90% without loss of precision.
        cached_sin2 = None
        if eff_rays is not None and eff_keys is not None and eff_gate > 0.0:
            chunk_sz = self.blocks[0].attn.chunk_size
            cached_sin2 = compute_pairwise_sin2_theta(eff_rays, eff_keys, chunk_size=chunk_sz)

        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing: trade compute for VRAM
                x = torch.utils.checkpoint.checkpoint(
                    block, x, eff_rays, eff_keys, eff_gate, cached_sin2,
                    use_reentrant=False,
                )
            else:
                x = block(x, eff_rays, eff_keys, eff_gate, cached_sin2_theta=cached_sin2)
            if i in reassemble_set:
                idx = layer_to_idx[i]
                if idx < len(self.reassemble_norms):
                    features.append(self.reassemble_norms[idx](x))
                else:
                    features.append(self.norm(x))

        return features


# ======================================================================
# DPT Depth Head
# ======================================================================

class ReassembleBlock(nn.Module):
    """Per-scale feature reassembly for DPT depth head.

    Projects token features at 28×28 resolution through learned convolutions
    to produce a 2D feature map at intermediate channel width.

    Uses GroupNorm (NOT BatchNorm) because batch sizes are too small for
    reliable BN statistics.

    Args:
        embed_dim: Token embedding dimension.
        mid_channels: Intermediate channel width.
    """

    def __init__(self, embed_dim: int = 256, mid_channels: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(GROUP_NORM_NUM_GROUPS, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(GROUP_NORM_NUM_GROUPS, mid_channels),
            nn.GELU(),
        )

    def forward(self, tokens: Tensor, grid_size: int) -> Tensor:
        """Reshape tokens to 2D and apply convolution.

        Args:
            tokens: Token features (B, N, D).
            grid_size: Spatial grid dimension (e.g., 28).

        Returns:
            features: 2D feature map (B, mid_channels, grid_size, grid_size).
        """
        B, N, D = tokens.shape
        # Reshape: (B, N, D) → (B, D, G, G)
        features = tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        return self.proj(features)


class DPTDepthHead(nn.Module):
    """DPT-inspired depth head with progressive 2× upsample.

    Complete redesign from v5.2 (which was rated WEAK/CRITICAL by review).

    Architecture:
        1. Multi-scale reassembly: each transformer block's tokens → 2D features
        2. Concatenation + merge: all scales → single feature map
        3. Progressive 2× upsample: 28→56→112→224 with refinement at each stage
        4. Disparity → depth conversion with softplus + clamping

    Critical improvements over v5.2:
        - DPT-style concat+merge (vs weak element-wise summation)
        - Progressive 2× upsample (vs single 4× ConvTranspose2d)
        - GroupNorm (vs BatchNorm with small batches)
        - Bilinear-like init for ConvTranspose2d (vs default kaiming)

    Args:
        cfg: ModelConfig with all architecture parameters.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # --- Sparse reassembly: only reassemble from specified layers ---
        # Instead of num_reassemble_blocks (10), use only len(reassemble_layers) (3)
        # This saves ~1.49M params
        self.reassemble_layers = cfg.reassemble_layers
        num_reassemble = len(self.reassemble_layers)

        self.reassemble = nn.ModuleList([
            ReassembleBlock(cfg.embed_dim, cfg.reassemble_mid_channels)
            for _ in range(num_reassemble)
        ])

        # --- Concat + merge ---
        concat_channels = cfg.reassemble_mid_channels * num_reassemble
        self.merge = nn.Sequential(
            nn.Conv2d(concat_channels, cfg.merge_out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(GROUP_NORM_NUM_GROUPS, cfg.merge_out_channels),
            nn.GELU(),
        )

        # --- Progressive 2× upsample stages ---
        # Stage 1: 28→56 (merge_out_channels → up_channels[0])
        self.up1 = nn.ConvTranspose2d(cfg.merge_out_channels, cfg.up_channels[0], kernel_size=2, stride=2)
        self.refine1 = self._make_refine_block(cfg.up_channels[0])

        # Stage 2: 56→112
        self.up2 = nn.ConvTranspose2d(cfg.up_channels[0], cfg.up_channels[1], kernel_size=2, stride=2)
        self.refine2 = self._make_refine_block(cfg.up_channels[1])

        # Stage 3: 112→224
        self.up3 = nn.ConvTranspose2d(cfg.up_channels[1], cfg.up_channels[2], kernel_size=2, stride=2)
        self.refine3 = self._make_refine_block(cfg.up_channels[2])

        # --- Disparity head at full resolution ---
        self.disp_head = nn.Sequential(
            nn.Conv2d(cfg.up_channels[2], 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        # --- Critical initialization ---
        self._init_weights()

    @staticmethod
    def _make_refine_block(channels: int) -> nn.Sequential:
        """Create a refinement block: Conv3×3 + GroupNorm + GELU × 2."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(GROUP_NORM_NUM_GROUPS, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(GROUP_NORM_NUM_GROUPS, channels),
            nn.GELU(),
        )

    def _init_weights(self) -> None:
        """Initialize weights with critical fixes from v5.2.

        1. Disparity head last conv: small init to prevent disparity explosion
        2. ConvTranspose2d layers: bilinear-like init (NOT default kaiming)
           Default init produces near-zero/negative output that kills depth signal.
        """
        # --- Disp head: small init ---
        with torch.no_grad():
            nn.init.zeros_(self.disp_head[-1].bias)
            self.disp_head[-1].weight.data.mul_(0.01)

        # --- ConvTranspose2d: bilinear-like init ---
        for up_layer in [self.up1, self.up2, self.up3]:
            self._init_upsample_bilinear(up_layer)

    @staticmethod
    def _init_upsample_bilinear(layer: nn.ConvTranspose2d) -> None:
        """Initialize ConvTranspose2d with bilinear-like kernel.

        For ConvTranspose2d(C_in, C_out, k=2, s=2):
        1. Zero all weights and biases
        2. Set diagonal channels (c_in == c_out) to identity-like 2×2 kernel
        3. Normalize so constant input → constant output

        This prevents the near-zero/negative output problem with default
        kaiming initialization that kills the depth signal.

        ConvTranspose2d weight shape convention (C_in, C_out, kH, kW).
        The normalization loop must index weight[:, c_out] (all input
        channels for a given output channel), NOT weight[c_out] (which
        indexes C_in dimension, treating it as output channels).
        """
        with torch.no_grad():
            nn.init.normal_(layer.weight, 0, 0.01)  # small random base
            nn.init.zeros_(layer.bias)

            # Set identity-like mapping for min(C_in, C_out) channels
            C_in = layer.in_channels
            C_out = layer.out_channels
            n_id = min(C_in, C_out)

            for c in range(n_id):
                layer.weight.data[c, c, :, :] = 0.25  # each input channel → one output

            # Normalize per OUTPUT channel.
            # weight shape: (C_in, C_out, kH, kW)
            # For output channel c_out, we need sum over ALL input channels
            # and spatial positions: sum(weight[:, c_out, :, :]) = 1
            # Previously indexed weight[c_out] which selects C_in dim — wrong!
            for c_out in range(C_out):
                kernel_sum = layer.weight.data[:, c_out].sum()
                if kernel_sum > 0:
                    layer.weight.data[:, c_out].div_(kernel_sum)

    def forward(self, intermediate_features: List[Tensor], grid_size: int) -> Tensor:
        """Predict depth from multi-scale transformer features.

        Args:
            intermediate_features: Features from each transformer block,
                list of (B, N, D) tensors.
            grid_size: Spatial grid dimension.

        Returns:
            depth: Predicted depth map (B, 1, 224, 224) in metres.
        """
        B = intermediate_features[0].shape[0]

        # --- Sparse multi-scale reassembly ---
        # after P1-BB encoder returns only len(reassemble_layers) features
        # (in order), not num_layers features. Iterate by enumerate, not by index.
        reassembled = []
        if len(intermediate_features) != len(self.reassemble_layers):
            # Backward compat: encoder returned all features — select by index
            for i, layer_idx in enumerate(self.reassemble_layers):
                feat = intermediate_features[layer_idx]
                reassembled.append(self.reassemble[i](feat, grid_size))
        else:
            # New (sparse) path: features are already at the right indices
            for i, feat in enumerate(intermediate_features):
                reassembled.append(self.reassemble[i](feat, grid_size))
        # Each: (B, 128, 28, 28)

        # --- Concat + merge ---
        multi_scale = torch.cat(reassembled, dim=1)  # (B, 1280, 28, 28)
        fused = self.merge(multi_scale)  # (B, 256, 28, 28)

        # --- Progressive 2× upsample ---
        # Stage 1: 28→56
        x = self.up1(fused)       # (B, 128, 56, 56)
        x = self.refine1(x)

        # Stage 2: 56→112
        x = self.up2(x)           # (B, 64, 112, 112)
        x = self.refine2(x)

        # Stage 3: 112→224
        x = self.up3(x)           # (B, 32, 224, 224)
        x = self.refine3(x)

        # --- Disparity → depth ---
        disp_raw = self.disp_head(x)  # (B, 1, 224, 224)
        disp = F.softplus(disp_raw) + self.cfg.disp_min  # softplus(0) ≈ 0.693
        disp = disp.clamp(max=self.cfg.disp_max)

        depth = 1.0 / disp
        depth = depth.clamp(min=self.cfg.depth_min, max=self.cfg.depth_max)

        # --- NaN safety ---
        # Replace non-finite values with median of finite values
        finite_mask = torch.isfinite(depth)
        if not finite_mask.all():
            for b in range(B):
                finite_vals = depth[b][finite_mask[b]]
                if finite_vals.numel() > 0:
                    median_val = finite_vals.median()
                else:
                    median_val = torch.tensor(1.0, device=depth.device, dtype=depth.dtype)
                depth[b] = torch.where(finite_mask[b], depth[b], median_val)

        return depth


# ======================================================================
# Full Tesseract Model
# ======================================================================

class Tesseract(nn.Module):
    """Tesseract v1: 3D Spatial Foundation Model for Monocular Depth Estimation.

    Full architecture:
        1. ConvStem: image → patch tokens
        2. Trivision PE: camera rays → positional encoding
        3. Transformer Encoder: 10 blocks with IRER attention
        4. DPT Depth Head: multi-scale → progressive upsample → depth
        5. 3D Point Reconstruction: ray × depth

    Target: ~10.7M parameters, enabled by geometric priors (IRER + Trivision).

    Args:
        cfg: ModelConfig with architecture hyperparameters.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ConvStem (2-layer)
        self.conv_stem = ConvStem(3, cfg.embed_dim, cfg.patch_size)
        # Ray positional encoding (FiLM or additive)
        self.ray_pe = RayPositionalEncoding(cfg.embed_dim, cfg.num_freqs, cfg.use_film_pe)

        # Transformer encoder
        self.encoder = TransformerEncoder(
            num_layers=cfg.num_layers,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            layer_scale_init=cfg.layer_scale_init,
            drop_path_max=cfg.drop_path_max,
            chunk_size=cfg.chunk_size,
            use_flash=cfg.use_flash_attention,
            gradient_checkpointing=cfg.gradient_checkpointing,
            num_geometric_heads=cfg.num_geometric_heads,
            reassemble_layers=cfg.reassemble_layers,
        )

        # Depth head
        self.depth_head = DPTDepthHead(cfg)

        # Standard 2D positional embedding for ablation against 2D ViT baseline
        if not cfg.enable_trivision:
            self.pos_embed_2d = nn.Parameter(torch.zeros(1, cfg.grid_size * cfg.grid_size, cfg.embed_dim))
            nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)
        else:
            self.pos_embed_2d = None

        # --- Weight initialization (BUG FIX: original used PyTorch defaults) ---
        # ViT-style trunc_normal for all Linear/Conv, zero-init residual projections
        self.apply(self._init_weights)

        # the global pass above RE-INITIALIZES every
        # nn.Linear/nn.Conv2d — including the two layers that were given
        # specialized initialization inside their own constructors:
        #   (a) ray_pe.film_mlp[-1]       — FiLM identity init (γ=1, β=0)
        #   (b) depth_head.disp_head[-1]  — small init to tame initial disparity
        # Without re-applying them here, FiLM starts with γ≈0 (random std=0.02
        # weights, zero bias) and zeroes the token stream — the EXACT bug the
        # identity init was meant to fix — and the disparity head loses its
        # small init. Verified at runtime before fixing.
        if cfg.use_film_pe:
            nn.init.zeros_(self.ray_pe.film_mlp[-1].weight)
            nn.init.zeros_(self.ray_pe.film_mlp[-1].bias)
            with torch.no_grad():
                self.ray_pe.film_mlp[-1].bias[:cfg.embed_dim].fill_(1.0)  # γ = 1
        with torch.no_grad():
            nn.init.zeros_(self.depth_head.disp_head[-1].bias)
            self.depth_head.disp_head[-1].weight.data.mul_(0.01)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """ViT-style init: trunc_normal(std=0.02) for Linear/Conv, zero bias.
        Residual-projection outputs are zero-init for clean identity residual at start
        (combined with LayerScale(1e-5), gives near-identity init).
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        image: Tensor,
        intrinsics: Tensor,
        irer_gate: float = 1.0,
        is_flipped: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass: image + intrinsics → depth + 3D points.

        Args:
            image: RGB image (B, 3, 224, 224), assumed ImageNet-normalized.
            intrinsics: Camera intrinsics K (B, 3, 3).
            irer_gate: Multiplicative gate for IRER bias (0.0 = disabled, 1.0 = full).
            is_flipped: Optional boolean tensor indicating horizontally mirrored inputs.

        Returns:
            depth: Predicted depth map (B, 1, 224, 224) in metres.
            points: 3D points per patch (B, 784, 3) in camera frame.
            rays: Trivision rays (B, 784, 3, 3) for downstream use.
        """
        # --- ConvStem ---
        tokens, grid_size = self.conv_stem(image)  # (B, 784, 256)

        # --- Trivision PE (with FiLM or additive) ---
        if self.cfg.enable_trivision:
            rays = compute_trivision_rays(
                intrinsics, grid_size, self.cfg.patch_size, self.cfg.img_size,
                is_flipped=is_flipped,
            )  # (B, 784, 3, 3)
            if self.cfg.pe_mode == "center_ray":
                # Ablation: collapse 3 rays into single center ray (no aperture/corner geometry)
                rays_for_pe = rays[:, :, 0:1, :].expand(-1, -1, 3, -1)
            else:
                rays_for_pe = rays
            pe_output, is_film = self.ray_pe(rays_for_pe)
            if is_film:
                # FiLM modulation: tokens = γ * tokens + β
                gamma, beta = pe_output
                tokens = gamma * tokens + beta
            else:
                # Additive PE: tokens = tokens + ray_pe
                tokens = tokens + pe_output
        else:
            # Baseline 2D ViT ablation: standard additive 2D positional embedding
            tokens = tokens + self.pos_embed_2d
            rays = None

        # --- Key geometry for Continuous 3D Attention (CARPE / IRER) ---
        # In a single perspective camera frame, camera rays originate from the
        # optical center (0, 0, 0). The center ray direction per patch provides
        # the exact reference vector for continuous angular distance attention.
        key_points = rays[:, :, 0, :] if rays is not None else None

        intermediate_features = self.encoder(
            tokens, rays, key_points,
            irer_gate=irer_gate,
            enable_irer=self.cfg.enable_irer,
            enable_trivision=self.cfg.enable_trivision,
            reassemble_layers=self.cfg.reassemble_layers,
        )
        # Sparse list of len(reassemble_layers) × (B, 784, 256)

        # --- Depth Head ---
        depth = self.depth_head(intermediate_features, grid_size)  # (B, 1, 224, 224)

        # --- 3D Point Reconstruction ---
        # Sample depth at patch centres for 3D point computation
        depth_at_patches = F.avg_pool2d(
            depth,
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        )  # (B, 1, 28, 28)
        depth_at_patches = depth_at_patches.squeeze(1).reshape(image.shape[0], -1)  # (B, 784)

        # points = ray_direction * depth (only if rays available)
        if rays is not None:
            points = rays[:, :, 0, :] * depth_at_patches.unsqueeze(-1)  # (B, 784, 3)
        else:
            # Trivision disabled: return zero points (no 3D reconstruction)
            points = torch.zeros(image.shape[0], depth_at_patches.shape[1], 3,
                                 device=depth.device, dtype=depth.dtype)

        return depth, points, rays

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component for architecture verification.

        Returns:
            Dict mapping component name to parameter count.
        """
        counts = {}
        for name, module in [
            ("conv_stem", self.conv_stem),
            ("ray_pe", self.ray_pe),
            ("encoder", self.encoder),
            ("depth_head", self.depth_head),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts


# Primary alias: Dioptra
Dioptra = Tesseract


#════════════════════════════════════════════════════════════════════════════#
# MODULE: losses.py
#────────────────────────────────────────────────────────────────────────────#
"""
Loss functions for Tesseract v1.

Pure depth estimation losses — NO JEPA loss, NO VISReg loss, NO temporal
component. This is a 3D-only model.

Provides:
    1. SiLog Loss with Median Normalization (Eigen protocol)
    2. Edge Gradient Loss with Sobel Filters (O(h²) accuracy)
    3. Planarity Loss (replaces JEPA's geometric supervision)
    4. UncertaintyWeightedLoss (Kendall et al. 2018)

Key fixes from v5.2:
    - Median normalization applied to edge loss (was missing → scale bug)
    - Sobel filters replace finite-difference gradients (O(h²) vs O(h) error)
    - Edge-aware masking to skip occlusion boundaries
    - Uncertainty weighting replaces static manual weights
    - Depth warmup NOT needed (no JEPA competing)

References:
    - Eigen et al., "Depth Map Prediction from a Single Image using
      Multi-Scale Deep Networks", NIPS 2014
    - Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
      Losses for Geometric and Optical Problems", CVPR 2018
"""


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from .config import LossConfig, ModelConfig


# ======================================================================
# Module-level filter caches (BUG FIX: avoid re-allocating every forward call)
# ======================================================================

_SOBEL_CACHE: dict = {}
"""Cache for Sobel filter kernels, keyed by (device, dtype)."""

_LAPLACIAN_CACHE: dict = {}
"""Cache for Laplacian filter kernels, keyed by (device, dtype)."""


# ======================================================================
# Median Normalization (Eigen Protocol)
# ======================================================================

def median_normalize(
    pred: Tensor,
    gt: Tensor,
    valid: Tensor,
) -> Tensor:
    """Per-sample median normalization following the Eigen protocol.

    Scales predicted depth so that its median matches the GT median.
    This makes the loss invariant to global scale, which is the standard
    evaluation protocol for monocular depth estimation.

    Args:
        pred: Predicted depth (B, 1, H, W) or (B, H, W).
        gt: Ground-truth depth (B, 1, H, W) or (B, H, W).
        valid: Valid pixel mask (B, 1, H, W) or (B, H, W), bool.

    Returns:
        pred_normalized: Scale-normalized predictions, same shape as pred.
    """
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred_s = pred.squeeze(1)
        gt_s = gt.squeeze(1)
        valid_s = valid.squeeze(1)
        squeeze = True
    else:
        pred_s, gt_s, valid_s = pred, gt, valid
        squeeze = False

    B = pred_s.shape[0]
    pred_norm = pred_s.clone()

    for b in range(B):
        mask = (
            (valid_s[b] > 0)
            & torch.isfinite(gt_s[b])
            & (gt_s[b] > 0)
            & torch.isfinite(pred_s[b])
            & (pred_s[b] > 0)
        )
        if mask.sum() == 0:
            continue
        pred_vals = pred_s[b][mask]
        gt_vals = gt_s[b][mask]

        med_pred = pred_vals.median()
        med_gt = gt_vals.median()

        if med_pred.abs() < 1e-8 or not torch.isfinite(med_pred) or not torch.isfinite(med_gt):
            continue

        scale = (med_gt / med_pred).clamp(1e-4, 1e4)
        pred_norm[b] = pred_s[b] * scale

    if squeeze:
        pred_norm = pred_norm.unsqueeze(1)
    return pred_norm


# ======================================================================
# SiLog Loss
# ======================================================================

def silog_loss(
    pred: Tensor,
    gt: Tensor,
    valid: Tensor,
    lambda_w: float = 0.85,
    pred_norm: Optional[Tensor] = None,
    use_median_norm: bool = False,
) -> Tensor:
    """Scale-Invariant Log (SiLog) loss.

    Formula:
        d = log(p) - log(gt)
        SiLog = sqrt(mean(d²) - λ_w × mean(d)²)

    When use_median_norm=True (or pred_norm is passed), predictions are median-normalized.
    When use_median_norm=False (default for metric supervision), SiLog evaluates directly
    on raw predictions, penalizing structural shape error while retaining metric scale
    gradient via the (1 - λ_w) residual.

    Args:
        pred: Predicted depth (B, 1, H, W).
        gt: Ground-truth depth (B, 1, H, W).
        valid: Valid pixel mask (B, 1, H, W).
        lambda_w: Variance weighting (0.85).
        pred_norm: Pre-normalized predictions. If provided, used directly.
        use_median_norm: If True and pred_norm is None, normalize predictions by median.

    Returns:
        Scalar loss value.
    """
    if pred_norm is not None:
        p = pred_norm
    elif use_median_norm:
        p = median_normalize(pred, gt, valid)
    else:
        p = pred

    p = p.float()
    gt = gt.float()

    # Per-sample SiLog, evaluated strictly on finite positive valid pixels
    B = pred.shape[0]
    losses = []
    for b in range(B):
        v = (
            (valid[b] > 0)
            & torch.isfinite(gt[b])
            & (gt[b] > 0)
            & torch.isfinite(p[b])
            & (p[b] > 0)
        )
        if v.sum() < 10:
            continue

        p_b = p[b][v].clamp(min=1e-6)
        g_b = gt[b][v].clamp(min=1e-6)
        d_b = torch.log(p_b) - torch.log(g_b)

        d_mean = d_b.mean()
        d_sq_mean = (d_b * d_b).mean()

        silog_sq = d_sq_mean - lambda_w * d_mean * d_mean
        losses.append(torch.sqrt(silog_sq.clamp(min=1e-10)))

    if len(losses) == 0:
        return torch.tensor(0.0, device=pred.device, dtype=torch.float32, requires_grad=True)
    return torch.stack(losses).mean()


# ======================================================================
# Edge Gradient Loss with Sobel Filters
# ======================================================================

def edge_gradient_loss(
    pred: Tensor,
    gt: Tensor,
    valid: Tensor,
    cfg: LossConfig = LossConfig(),
    pred_norm: Optional[Tensor] = None,
    scales: Optional[Tuple[float, ...]] = None,
) -> Tensor:
    """Edge-aware gradient loss using Sobel filters with Charbonnier penalty.

    Numerical Robustness (v17.1):
        1. Fully executed in float32 to prevent FP16 overflow (>65504 -> inf).
        2. Replaces non-finite values in gt and pred with 0.0 before convolution
           to prevent NaN propagation across neighboring pixels.
        3. Convolves valid mask with 3x3 ones kernel to ensure Sobel gradients
           are only evaluated on continuous surface patches, never on sky/occlusions.
        4. Uses boolean indexing instead of zero-multiplication, avoiding
           IEEE 754 inf * 0.0 -> NaN and NaN * 0.0 -> NaN traps.

    Args:
        pred: Predicted depth (B, 1, H, W).
        gt: Ground-truth depth (B, 1, H, W).
        valid: Valid pixel mask (B, 1, H, W).
        cfg: LossConfig with edge_threshold and charbonnier_eps.
        pred_norm: Pre-normalized predictions. If provided, skips
            internal median_normalize.
        scales: Multi-scale factors for edge supervision.

    Returns:
        Scalar loss value (float32).
    """
    pred = pred.float()
    gt = gt.float()
    valid_mask = (valid > 0)
    valid_f = valid_mask.float()

    if pred_norm is None:
        pred_norm = median_normalize(pred, gt, valid_mask)
    else:
        pred_norm = pred_norm.float()

    if scales is None:
        scales = (1.0,)

    # Replace any non-finite values in gt and pred before convolution
    gt_clean = torch.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
    pred_clean = torch.nan_to_num(pred_norm, nan=0.0, posinf=0.0, neginf=0.0)

    total_loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)

    for scale in scales:
        if scale != 1.0:
            H, W = pred_clean.shape[-2:]
            new_H, new_W = int(H * scale), int(W * scale)
            if new_H < 3 or new_W < 3:
                continue
            pred_s = F.interpolate(pred_clean, size=(new_H, new_W), mode="bilinear", align_corners=False)
            gt_s = F.interpolate(gt_clean, size=(new_H, new_W), mode="nearest")
            valid_s = F.interpolate(valid_f, size=(new_H, new_W), mode="nearest")
        else:
            pred_s = pred_clean
            gt_s = gt_clean
            valid_s = valid_f

        # Get Sobel filters from cache (float32)
        _cache_key = (pred_s.device, torch.float32)
        if _cache_key not in _SOBEL_CACHE:
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32, device=pred_s.device,
            ).reshape(1, 1, 3, 3) / 8.0
            _SOBEL_CACHE[_cache_key] = (sobel_x, sobel_x.transpose(-2, -1))
        sobel_x, sobel_y = _SOBEL_CACHE[_cache_key]

        # Compute gradients via 2D convolution
        pred_gx = F.conv2d(pred_s, sobel_x, padding=1)
        pred_gy = F.conv2d(pred_s, sobel_y, padding=1)
        gt_gx = F.conv2d(gt_s, sobel_x, padding=1)
        gt_gy = F.conv2d(gt_s, sobel_y, padding=1)

        # A 3x3 Sobel gradient is only valid if all 3x3 neighbors are valid
        ones_3x3 = torch.ones(1, 1, 3, 3, device=valid_s.device, dtype=torch.float32)
        valid_patch = F.conv2d(valid_s, ones_3x3, padding=1) > 8.5

        # Edge-aware masking: only penalize continuous surfaces (skip occlusion boundaries)
        edge_mask = (
            (gt_gx.abs() < cfg.edge_threshold) &
            (gt_gy.abs() < cfg.edge_threshold) &
            valid_patch
        )

        n_valid = edge_mask.float().sum()
        if n_valid < 1.0:
            continue

        # Extract only valid edge pixels via boolean indexing (prevents nan/inf contamination)
        diff_x = pred_gx[edge_mask] - gt_gx[edge_mask]
        diff_y = pred_gy[edge_mask] - gt_gy[edge_mask]

        eps = cfg.charbonnier_eps
        loss_x = torch.sqrt(diff_x * diff_x + eps * eps)
        loss_y = torch.sqrt(diff_y * diff_y + eps * eps)

        total_loss = total_loss + (loss_x.sum() + loss_y.sum()) / n_valid.clamp(min=1.0)

    return total_loss / len(scales)


# ======================================================================
# Planarity Loss
# ======================================================================

def planarity_loss(
    pred: Tensor,
    gt: Tensor,
    valid: Tensor,
    cfg: LossConfig = LossConfig(),
    pred_norm: Optional[Tensor] = None,
) -> Tensor:
    """Planarity loss: penalize non-planar noise in smooth GT regions.

    Since v1 no longer has JEPA providing implicit geometric structure,
    this explicit loss encourages smooth depth within planar regions.
    Uses a Laplacian filter to detect high-frequency noise in the
    prediction that corresponds to smooth (small-Laplacian) regions in GT.

    This effectively says: "where the ground truth is smooth, the
    prediction should also be smooth." It does NOT force the prediction
    to be planar everywhere — only where GT evidence supports it.

    Args:
        pred: Predicted depth (B, 1, H, W).
        gt: Ground-truth depth (B, 1, H, W).
        valid: Valid pixel mask (B, 1, H, W).
        cfg: LossConfig with smooth_threshold.
        pred_norm: Pre-normalized predictions. If provided, skips
            internal median_normalize (avoids 3× redundant computation).

    Returns:
        Scalar loss value.
    """
    # Apply median normalization for consistent scale
    # Accept pre-normalized pred to avoid 3× redundant calls
    if pred_norm is None:
        pred_norm = median_normalize(pred, gt, valid)

    pred_clean = torch.nan_to_num(pred_norm.float(), nan=0.0, posinf=0.0, neginf=0.0)
    _cache_key = (pred.device, torch.float32)
    if _cache_key not in _LAPLACIAN_CACHE:
        _LAPLACIAN_CACHE[_cache_key] = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32, device=pred.device,
        ).reshape(1, 1, 3, 3)
    laplacian = _LAPLACIAN_CACHE[_cache_key]
    gt_clean = torch.nan_to_num(gt.float(), nan=0.0, posinf=0.0, neginf=0.0)

    pred_lap = F.conv2d(pred_clean, laplacian, padding=1)
    gt_lap = F.conv2d(gt_clean, laplacian, padding=1)

    # Valid interior: erode mask with 3x3 kernel so boundary pixels of sky/occlusions don't corrupt Laplacian
    _ones3x3 = torch.ones(1, 1, 3, 3, device=valid.device, dtype=torch.float32)
    valid_f = (valid.float() > 0).float()
    valid_interior = (F.conv2d(valid_f, _ones3x3, padding=1) >= 8.5)

    # Smooth mask: only penalize where GT is smooth (small Laplacian) and completely within valid interior
    smooth_mask = (gt_lap.abs() < cfg.smooth_threshold) & valid_interior & torch.isfinite(gt_clean) & torch.isfinite(pred_clean)

    # Penalize prediction Laplacian magnitude in smooth GT regions
    n_smooth = smooth_mask.float().sum().clamp(min=1.0)
    return (pred_lap[smooth_mask].abs().sum()) / n_smooth


# ======================================================================
# L1 Loss (auxiliary, pins absolute scale)
# ======================================================================

def l1_loss(
    pred: Tensor,
    gt: Tensor,
    valid: Tensor,
    pred_norm: Optional[Tensor] = None,
) -> Tensor:
    """Log-L1 loss on RAW depth (balanced metric scale pinning).

    UPGRADE (v17): In outdoor TartanAir scenes reaching 200m, linear meter L1
    (|pred - gt|) is heavily dominated by distant geometry (e.g. 20m error on
    far background creates loss=20, whereas 0.2m error on close object creates
    loss=0.2). This distorts near-field metric navigation.

    Log-L1 computes |log(pred) - log(gt)| on unnormalized raw depth. This ensures
    equal gradient magnitude for equal relative errors (e.g. 10% error at 1m is
    penalized equally to 10% error at 100m) while rigidly pinning absolute scale
    (unlike SiLog, which is scale-invariant).

    Args:
        pred: Predicted depth (B, 1, H, W).
        gt: Ground-truth depth (B, 1, H, W).
        valid: Valid pixel mask (B, 1, H, W).
        pred_norm: Ignored (kept for backward API compat).

    Returns:
        Scalar loss value.
    """
    pred_f = pred.float()
    gt_f = gt.float()
    mask = (
        (valid > 0)
        & torch.isfinite(gt_f)
        & (gt_f > 0)
        & torch.isfinite(pred_f)
        & (pred_f > 0)
    )
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=torch.float32, requires_grad=True)

    p_val = pred_f[mask].clamp(min=1e-6)
    g_val = gt_f[mask].clamp(min=1e-6)
    diff = (torch.log(p_val) - torch.log(g_val)).abs()
    return diff.mean()


# ======================================================================
# Image-Aware Smoothness Loss
# ======================================================================

def image_aware_smoothness_loss(
    pred: Tensor,
    gt: Tensor,
    valid: Tensor,
    image: Tensor,
    cfg: LossConfig = LossConfig(),
    pred_norm: Optional[Tensor] = None,
) -> Tensor:
    """Smooth depth where image is smooth; allow discontinuities at image edges.

    Uses exponential decay on image gradients to weight depth smoothness:
    where the image has strong gradients (edges), the depth is allowed
    to be discontinuous; where the image is smooth, the depth is
    encouraged to be smooth.

    Args:
        pred: Predicted depth (B, 1, H, W).
        gt: Ground-truth depth (B, 1, H, W).
        valid: Valid pixel mask (B, 1, H, W).
        image: RGB image (B, 3, H, W) in [0, 1] (before ImageNet normalization).
        cfg: LossConfig with smooth_image_grad_weight.
        pred_norm: Pre-normalized predictions. If provided, skips
            internal median_normalize.

    Returns:
        Scalar loss value.
    """
    if pred_norm is None:
        pred_norm = median_normalize(pred, gt, valid)

    pred_f = pred_norm.float()

    # ImageNet-normalized images have Sobel gradients ~4× larger than
    # [0,1] images, so exp(-10 * |grad|) ≈ 0 in textured regions, silently
    # disabling this loss. Un-normalize back to [0,1] first.
    img_raw = image.float()
    # detection heuristic extended. Checking only
    # min < -0.5 missed BRIGHT normalized scenes (fog/snow/sky), whose minimum
    # normalized pixel stays above -0.5 — silently disabling this loss again.
    # Raw [0,1] images never exceed 1.0; normalized bright pixels exceed 1.5.
    if img_raw.min() < -0.5 or img_raw.max() > 1.5:  # looks ImageNet-normalized
        mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=image.device).view(1, 3, 1, 1)
        img_raw = (img_raw * std + mean).clamp(0.0, 1.0)

    # Get Sobel filters from cache
    _cache_key = (pred_f.device, pred_f.dtype)
    if _cache_key not in _SOBEL_CACHE:
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=pred_f.dtype, device=pred_f.device,
        ).reshape(1, 1, 3, 3) / 8.0
        _SOBEL_CACHE[_cache_key] = (sobel_x, sobel_x.transpose(-2, -1))
    sobel_x, sobel_y = _SOBEL_CACHE[_cache_key]

    # Depth gradients (Sobel) with NaN sanitization
    pred_clean = torch.nan_to_num(pred_f, nan=0.0, posinf=0.0, neginf=0.0)
    pred_gx = F.conv2d(pred_clean, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_clean, sobel_y, padding=1)

    # Image gradients (on un-normalized [0,1] image)
    # original called F.conv2d(img_f, sobel_x) with img_f shape
    # (B, 3, H, W) and sobel_x shape (1, 1, 3, 3) → channel mismatch error. Use
    # grouped conv with the kernel repeated per-channel to get (B, 3, H, W) output.
    sobel_x_rgb = sobel_x.expand(3, 1, 3, 3)  # (3, 1, 3, 3) — same kernel per channel
    sobel_y_rgb = sobel_y.expand(3, 1, 3, 3)
    img_gx = F.conv2d(img_raw, sobel_x_rgb, padding=1, groups=3)  # (B, 3, H, W)
    img_gy = F.conv2d(img_raw, sobel_y_rgb, padding=1, groups=3)

    # Exponential decay: smooth where image is smooth, sharp at edges
    weight_x = torch.exp(-cfg.smooth_image_grad_weight * img_gx.abs().mean(dim=1, keepdim=True))
    weight_y = torch.exp(-cfg.smooth_image_grad_weight * img_gy.abs().mean(dim=1, keepdim=True))

    # Mask to valid interior so boundary steps aren't penalized
    _ones3x3 = torch.ones(1, 1, 3, 3, device=valid.device, dtype=torch.float32)
    valid_f = (valid.float() > 0).float()
    valid_interior = (F.conv2d(valid_f, _ones3x3, padding=1) >= 8.5)

    loss_x = (pred_gx.abs() * weight_x * valid_interior.float()).sum()
    loss_y = (pred_gy.abs() * weight_y * valid_interior.float()).sum()
    n_valid = valid_interior.float().sum().clamp(min=1.0)
    return (loss_x + loss_y) / n_valid


# ======================================================================
# Uncertainty-Weighted Loss (Kendall et al. 2018)
# ======================================================================

class UncertaintyWeightedLoss(nn.Module):
    """Learnable uncertainty-based multi-task loss weighting.

    Instead of manually tuning loss weights (which are fragile and
    task-dependent), this module learns per-task log-variance parameters
    that automatically balance the losses based on their noise levels.

    For task i with loss L_i and learnable log-variance σ_i²:
        total = Σ_i [exp(-log σ_i²) × L_i + log σ_i²]

    The first term is the precision-weighted loss, and the second is a
    regularizer that prevents the network from making all variances
    infinite (trivial solution).

    With 5 loss terms, this adds only 5 learnable parameters — negligible cost.

    Depth warmup: NOT needed (no JEPA competing). All losses start at
    full strength from epoch 0.

    Args:
        num_tasks: Number of loss terms (5: silog + l1 + edge + smoothness + planarity).

    References:
        Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
        Losses for Geometric and Optical Problems", CVPR 2018.
    """

    def __init__(self, num_tasks: int = 5):
        super().__init__()
        # Learnable log-variance parameters (initialized to 0 → equal weighting)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: Tuple[Tensor, ...]) -> Tensor:
        """Compute uncertainty-weighted total loss.

        Args:
            losses: Tuple of individual loss values (silog, edge, planarity).
                Each is a scalar tensor.

        Returns:
            total: Scalar total loss value.

        Raises:
            ValueError: If number of losses doesn't match num_tasks.
        """
        if len(losses) != self.log_vars.shape[0]:
            raise ValueError(
                f"Expected {self.log_vars.shape[0]} losses, got {len(losses)}"
            )

        # Clamped to [-2.0, 4.0] to prevent negative loss drift while allowing flexible weighting
        s = self.log_vars.clamp(-2.0, 4.0)
        precision = torch.exp(-s)  # (T,)
        losses_t = torch.stack(losses)  # (T,)
        return (precision * losses_t).sum() + s.sum()


#════════════════════════════════════════════════════════════════════════════#
# MODULE: dataset.py
#────────────────────────────────────────────────────────────────────────────#
"""
TartanAir dataset loader for Tesseract.

Key fixes from v5.2:
    1. Horizontal flip: negate v_x, omega_y, AND omega_z
       (omega_z was missing in v5.2 under reflection M = diag(-1,1,1))
    2. Color jitter temporal consistency: draw params ONCE, apply
       identically to both frames
    3. Curriculum difficulty schedule

NOTE: Since this is a 3D-only model (no TemporalLoom), the action
twist is NOT used during training. However, the hflip fix is retained
for future 4D extension and for valid mask / pose tracking.

References:
    - TartanAir: Wang et al., "TartanAir: A Large-Scale Data Generator
      for Navigation in Complex Environments", IROS 2020
    - SE(3) reflection: Under M = diag(-1,1,1):
      v' = Mv, ω' = (ω_x, -ω_y, -ω_z)
"""


import logging
import math
import hashlib
import os
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# from .config import DataConfig, ModelConfig, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


# ======================================================================
# Dataset root resolution (Kaggle-aware)
# ======================================================================

_TARTANAIR_DIFFS = {"Easy", "Medium", "Hard"}


def _dir_looks_like_tartanair(root: Path) -> bool:
    """Cheap structural probe: does root match either TartanAir layout?

    Layout A: root/{env}/{Easy|Medium|Hard}/P000/...
    Layout B: root/{Easy|Medium|Hard}/{env}/P000/...
    Probes at most two directory levels (listings stay small).
    """
    try:
        tops = [d for d in root.iterdir() if d.is_dir()]
    except OSError:
        return False
    if not tops:
        return False
    if any(t.name in _TARTANAIR_DIFFS for t in tops):
        return True  # layout B root
    for t in tops[:64]:
        try:
            if any(c.name in _TARTANAIR_DIFFS for c in t.iterdir() if c.is_dir()):
                return True  # layout A root
        except OSError:
            continue
    return False


def _zip_looks_like_tartanair(zip_path: Path) -> bool:
    """True if the archive contains image_left/ + depth_left/ members."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return False
    has_img = has_depth = False
    for n in names:
        if "/image_left/" in n:
            has_img = True
        elif "/depth_left/" in n:
            has_depth = True
        if has_img and has_depth:
            return True
    return False


def _enumerate_input_mounts(base: Path) -> List[str]:
    """List human-readable dataset-mount identifiers under ``base``.

    Understands both Kaggle input layouts:
      * classic:            /kaggle/input/<slug>/
      * namespaced (2025+): /kaggle/input/datasets/<owner>/<slug>/
    Purely informational — used for log lines and error messages.
    """
    try:
        children = sorted(d for d in base.iterdir() if d.is_dir())
    except OSError:
        return []
    mounts: List[str] = []
    for c in children:
        if c.name == "datasets":
            found = False
            try:
                owners = sorted(d for d in c.iterdir() if d.is_dir())
            except OSError:
                owners = []
            for o in owners:
                try:
                    slugs = sorted(d for d in o.iterdir() if d.is_dir())
                except OSError:
                    slugs = []
                for s in slugs:
                    mounts.append(f"datasets/{o.name}/{s.name}")
                    found = True
            if not found:
                mounts.append("datasets/")
        else:
            mounts.append(c.name)
    return mounts


# Depth budget for the recursive scan: /kaggle/input (0) → datasets (1)
# → owner (2) → dataset slug (3) → tartanair root (4); one spare level.
_MAX_SCAN_DEPTH = 5
_SCAN_SKIP_DIRS = {".ipynb_checkpoints", "__pycache__", ".git", "__MACOSX"}


def _deep_find_tartanair(
    base: Path,
) -> Tuple[List[Path], List[Tuple[Path, int]]]:
    """Depth-limited search for TartanAir roots/archives under ``base``.

    Returns ``(dir_hits, zip_hits)``; zip hits carry the archive size for
    ranking (largest first). Never descends into a match (its subtree IS
    the dataset — ~53k entries, pointless to walk) and prunes junk dirs,
    so the scan stays cheap even with several datasets mounted. Symlinks
    are followed (Kaggle mounts may be symlinked) with cycle protection.
    """
    dir_hits: List[Path] = []
    zip_hits: List[Tuple[Path, int]] = []
    seen = set()
    stack: List[Tuple[Path, int]] = [(base, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            key = str(d.resolve())
        except OSError:
            key = str(d)
        if key in seen:
            continue
        seen.add(key)
        if _dir_looks_like_tartanair(d):
            dir_hits.append(d)
            continue  # match: the subtree below is the dataset itself
        if depth >= _MAX_SCAN_DEPTH:
            continue
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name in _SCAN_SKIP_DIRS:
                    continue
                stack.append((e, depth + 1))
            elif e.is_file() and e.suffix.lower() == ".zip":
                try:
                    size = e.stat().st_size
                except OSError:
                    continue
                if size < 1_048_576:  # < 1 MB → junk archive, skip
                    continue
                if _zip_looks_like_tartanair(e):
                    zip_hits.append((e, size))
    dir_hits.sort(key=str)
    zip_hits.sort(key=lambda t: t[1], reverse=True)
    return dir_hits, zip_hits


def _resolve_single_mount(cand: Path) -> Optional[Path]:
    """Resolve one candidate mount/dir/file to a usable dataset root.

    Returns None when the candidate is not TartanAir-like (so 'auto' can
    skip unrelated mounted datasets, e.g. a code-only upload).
    """
    # 1. Explicit zip file
    if cand.is_file() and cand.suffix.lower() == ".zip":
        return cand
    if not cand.is_dir():
        return None
    # 2. Directory that IS a dataset root
    if _dir_looks_like_tartanair(cand):
        return cand
    # 3. Dataset nested one level down — Kaggle auto-extracted mounts keep
    #    the uploader's top folder, e.g. .../dasvo-tartanair-.../tartanair/
    try:
        subdirs = [d for d in sorted(cand.iterdir()) if d.is_dir()]
    except OSError:
        subdirs = []
    for sd in subdirs[:16]:
        if _dir_looks_like_tartanair(sd):
            return sd
    # 4. Dataset shipped as a zip inside the dir (Kaggle zip-mounted
    #    datasets); pick the largest matching archive.
    try:
        zips = sorted(
            (f for f in cand.glob("*.zip") if f.is_file()),
            key=lambda f: f.stat().st_size, reverse=True,
        )
    except OSError:
        zips = []
    for z in zips:
        if _zip_looks_like_tartanair(z):
            return z
    # 5. Deeper nesting — current Kaggle notebooks mount inputs as
    #    /kaggle/input/datasets/<owner>/<slug>/…, so the dataset root can
    #    sit several levels below the given dir. Bounded recursive search
    #    (also makes any ancestor dir of the dataset work with --train).
    dir_hits, zip_hits = _deep_find_tartanair(cand)
    if dir_hits:
        return dir_hits[0]
    if zip_hits:
        return zip_hits[0][0]
    return None


def resolve_dataset_root(path: str) -> str:
    """Resolve a --train argument into a usable dataset root (str).

    Accepts:
      * "auto" — recursively scan the Kaggle input mounts (/kaggle/input,
        or the dir in $TESSERACT_INPUT_DIR) for a TartanAir dataset. Both
        Kaggle layouts are supported:
          classic:    /kaggle/input/<slug>/tartanair/…
          namespaced: /kaggle/input/datasets/<owner>/<slug>/tartanair/…
      * a dataset root directory (used as-is)
      * a directory CONTAINING the dataset at any depth ≤ 5, e.g. the
        Kaggle mount /kaggle/input/dasvo-tartanair-rgb-d-validation-split
        → .../dasvo-tartanair-rgb-d-validation-split/tartanair
      * a .zip archive path (zip-backed lazy mode — no extraction needed)
      * a directory containing the dataset as a zip (Kaggle zip-mounted
        datasets); picks the largest matching archive

    Raises FileNotFoundError with a descriptive message (listing every
    mounted input it saw) when nothing TartanAir-like is found.
    """
    if str(path).lower() == "auto":
        scan_roots: List[Path] = []
        env_dir = os.environ.get("TESSERACT_INPUT_DIR")
        if env_dir:
            scan_roots.append(Path(env_dir))
        if Path("/kaggle/input").is_dir():
            scan_roots.append(Path("/kaggle/input"))
        if not scan_roots:
            raise FileNotFoundError(
                "--train auto: no /kaggle/input mount found. Attach the "
                "dataset to the notebook, or set TESSERACT_INPUT_DIR to a "
                "directory containing dataset mounts, or pass the dataset "
                "directory/zip path explicitly."
            )
        mount_names: List[str] = []
        for r in scan_roots:
            mount_names.extend(_enumerate_input_mounts(r))
        if not mount_names:
            raise FileNotFoundError(
                f"--train auto: no dataset mounts under "
                f"{' or '.join(str(r) for r in scan_roots)}"
            )
        logger.info(
            "--train auto: mounted input(s): " + ", ".join(mount_names)
        )
        dir_hits: List[Path] = []
        zip_hits: List[Tuple[Path, int]] = []
        for r in scan_roots:
            dh, zh = _deep_find_tartanair(r)
            dir_hits.extend(dh)
            zip_hits.extend(zh)
        if dir_hits:  # prefer an extracted tree over any archive
            root = dir_hits[0]
            logger.info(f"Dataset root resolved: auto -> {root}")
            return str(root)
        if zip_hits:
            zip_hits.sort(key=lambda t: t[1], reverse=True)
            root = zip_hits[0][0]
            logger.info(f"Dataset root resolved: auto -> {root} (zip-backed)")
            return str(root)
        blob = " ".join(mount_names).lower()
        if not any(k in blob for k in ("tartanair", "dasvo")):
            hint = (
                "None of the mounted inputs looks like the TartanAir dataset "
                "— it is probably not attached to this notebook. Add it via "
                "the right sidebar → Input → '+ Add Input' → search "
                "'dasvo-tartanair-rgb-d-validation-split' (by pandrii000), "
                "then re-run. On current Kaggle notebooks it mounts under "
                "/kaggle/input/datasets/pandrii000/"
                "dasvo-tartanair-rgb-d-validation-split/."
            )
        else:
            hint = (
                "A TartanAir-named input is mounted, but no "
                "{env}/{Easy|Hard}/P00x/image_left + depth_left tree (or a "
                ".zip of it) was found inside it — inspect it with "
                "!ls <mount> and pass the inner directory via --train."
            )
        raise FileNotFoundError(
            "Could not resolve a TartanAir dataset from: auto\n"
            f"  scanned (recursively): "
            f"{'; '.join(str(r) for r in scan_roots)}\n"
            f"  mounted input(s): {', '.join(mount_names)}\n"
            f"{hint}"
        )

    # Explicit path: dataset root / mount dir / ancestor dir / zip archive
    resolved = _resolve_single_mount(Path(path))
    if resolved is not None:
        if str(resolved) != str(path):
            logger.info(f"Dataset root resolved: {path} -> {resolved}")
        return str(resolved)
    raise FileNotFoundError(
        "Could not resolve a TartanAir dataset from: "
        f"{path}\n  scanned: {path} (recursively)\n"
        "Expected a directory with {env}/{Easy|Hard}/P00x/image_left + "
        "depth_left (optionally nested a few levels down), or a .zip "
        "archive of that tree."
    )


# ======================================================================
# TartanAir Dataset
# ======================================================================

class TartanAirDataset(torch.utils.data.Dataset):
    """TartanAir dataset for monocular depth estimation.

    Loads RGB images and corresponding depth maps from the TartanAir
    synthetic navigation dataset. Handles multiple difficulty levels
    with curriculum scheduling.

    Directory layout (expected):
        root/
        ├── {env_name}/
        │   ├── Easy/
        │   │   ├── P000/
        │   │   │   ├── image_left/
        │   │   │   │   ├── 000000.png
        │   │   │   │   ├── 000001.png
        │   │   │   │   └── ...
        │   │   │   ├── depth_left/
        │   │   │   │   ├── 000000.npy
        │   │   │   │   └── ...
        │   │   │   └── cam_left.json  (intrinsics)
        │   │   └── ...
        │   ├── Medium/
        │   └── Hard/
        └── ...

    Both {env}/{difficulty}/P000 (layout A above) and the official
    {difficulty}/{env}/P000 (layout B) are supported, and the root may
    alternatively be a .zip archive of either tree (members are read
    lazily per __getitem__ — no extraction, so 57 GB datasets mounted
    unextracted by Kaggle still work).

    Image/depth filename pairings (normalized-stem matching strips the
    _depth/_og/_left camera tags):
        000000.png         ↔ 000000.npy             (plain)
        000000_og.png      ↔ 000000.npy             (official TartanAir)
        000000_left.png    ↔ 000000_left_depth.npy  (DASVO/Kaggle split)

    Args:
        root: Root directory of TartanAir dataset, or a .zip archive
            containing that directory tree.
        difficulty: Difficulty level(s) to sample from.
            One of "easy", "medium", "hard", or "all".
        split: Dataset split ("train" or "val").
        img_size: Target image size (square).
        augment: Whether to apply data augmentation.
        cfg: DataConfig for augmentation parameters.
    """

    # Difficulty level mapping
    DIFFICULTY_MAP = {
        "easy": "Easy",
        "medium": "Medium",
        "hard": "Hard",
    }

    def __init__(
        self,
        root: str,
        difficulty: str = "easy",
        split: str = "train",
        img_size: int = 224,
        augment: bool = True,
        cfg: DataConfig = DataConfig(),
    ):
        super().__init__()
        self.root = Path(root)
        # fail fast with a clear message instead of
        # a bare FileNotFoundError from iterdir() when the root is missing.
        # the root may also be a .zip archive
        # (Kaggle mounts very large datasets unextracted; a 57 GB payload
        # cannot be unpacked into the 20 GB /kaggle/working quota). Zip
        # members are then read lazily per __getitem__ — no extraction.
        self._zip_path: Optional[Path] = None
        if str(self.root).lower().endswith(".zip"):
            if not self.root.is_file():
                raise FileNotFoundError(
                    f"Dataset zip does not exist or is not a file: {self.root}"
                )
            self._zip_path = self.root
        elif not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset root does not exist or is not a directory: {self.root}"
            )
        # Per-process zip handles (see _zip_read): forked DataLoader workers
        # must NOT share a parent's OS file descriptor (zipfile seeks on it).
        self._zip_cache: Dict[str, "zipfile.ZipFile"] = {}
        self._zip_cache_pid: Optional[int] = None
        self.difficulty = difficulty
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.cfg = cfg

        # Build sample index
        self.samples: List[Dict] = []
        self._build_index()

        logger.info(
            f"TartanAirDataset: {len(self.samples)} samples, "
            f"difficulty={difficulty}, split={split}"
        )

    def _build_index(self) -> None:
        """Build the sample index for the dataset root.

        Dispatches to the directory-tree scanner or the zip-archive scanner
        depending on how --train was resolved (see resolve_dataset_root).
        """
        if self._zip_path is not None:
            self._build_index_zip()
        else:
            self._build_index_tree()

    def _build_index_tree(self) -> None:
        """Scan directory tree and build sample index.

        Supports BOTH directory layouts (BUG FIX, review pass 2 — the official
        TartanAir release nests difficulty first, which the old scanner never
        found, silently producing an empty dataset):
          A) root/{env}/{Easy|Medium|Hard}/P000/...   (assumed layout)
          B) root/{Easy|Medium|Hard}/{env}/P000/...   (official TartanAir)

        Also handles the official TartanAir image naming 000000_og.png with
        depth files named 000000.png (old exact-stem matching dropped every
        frame of such trajectories).

        Splits train/val by hashing (env/traj) so the same trajectory's
        frames don't appear in both splits. BUG FIX: original ignored
        self.split entirely → train and val were byte-for-byte identical.

        Intrinsics are parsed ONCE per trajectory here (BUG FIX: the old
        code re-read cam_left.json in every __getitem__ call) and cached in
        the sample dicts.
        """
        difficulties = (
            ["Easy", "Medium", "Hard"] if self.difficulty == "all"
            else [self.DIFFICULTY_MAP.get(self.difficulty.lower(), "Easy")]
        )
        diff_set = set(difficulties)

        # Collect (traj_dir, env_name, diff_name) across both layouts.
        traj_dirs: List[Tuple[Path, str, str]] = []
        for top_dir in sorted(self.root.iterdir()):
            if not top_dir.is_dir():
                continue
            if top_dir.name in diff_set:
                # Layout B (official): root/{difficulty}/{env}/P000/
                for env_dir in sorted(top_dir.iterdir()):
                    if not env_dir.is_dir():
                        continue
                    for traj_dir in sorted(env_dir.iterdir()):
                        if traj_dir.is_dir():
                            traj_dirs.append((traj_dir, env_dir.name, top_dir.name))
            else:
                # Layout A (assumed): root/{env}/{difficulty}/P000/
                for diff_name in difficulties:
                    diff_path = top_dir / diff_name
                    if not diff_path.is_dir():
                        continue
                    for traj_dir in sorted(diff_path.iterdir()):
                        if traj_dir.is_dir():
                            traj_dirs.append((traj_dir, top_dir.name, diff_name))

        # TartanAir official uses 'Easy' and 'Hard' (no 'Medium' folder)
        if not traj_dirs and self.difficulty.lower() == "medium":
            logger.info(
                "TartanAir uses 'Easy' and 'Hard' levels (no 'Medium' directory); "
                "aliasing requested 'medium' -> 'Hard'"
            )
            for top_dir in sorted(self.root.iterdir()):
                if not top_dir.is_dir():
                    continue
                if top_dir.name == "Hard":
                    for env_dir in sorted(top_dir.iterdir()):
                        if not env_dir.is_dir():
                            continue
                        for traj_dir in sorted(env_dir.iterdir()):
                            if traj_dir.is_dir():
                                traj_dirs.append((traj_dir, env_dir.name, "Hard"))
                else:
                    diff_path = top_dir / "Hard"
                    if diff_path.is_dir():
                        for traj_dir in sorted(diff_path.iterdir()):
                            if traj_dir.is_dir():
                                traj_dirs.append((traj_dir, top_dir.name, "Hard"))

        # Per-trajectory intrinsics cache
        K_cache: Dict[str, Tensor] = {}
        depth_scale_cache: Dict[str, float] = {}

        for traj_dir, env_name, diff_name in traj_dirs:
            img_dir = traj_dir / "image_left"
            depth_dir = traj_dir / "depth_left"

            if not img_dir.is_dir() or not depth_dir.is_dir():
                continue

            traj_key = f"{env_name}/{traj_dir.name}"

            # Academic zero-shot cross-environment split (v2):
            # Split strictly by environment name so train and val NEVER share 3D worlds.
            # Prevents intra-scene data leakage.
            if getattr(self.cfg, "split_mode", "cross_env") == "cross_env":
                env_key = env_name.lower().strip()
                env_hash = int(hashlib.md5(env_key.encode()).hexdigest(), 16)
                is_val = (env_hash % 10) >= 8  # 80% train envs, 20% held-out val envs
            else:
                # Legacy / intra-environment cross-trajectory split
                traj_hash = int(hashlib.md5(traj_key.encode()).hexdigest(), 16)
                is_val = (traj_hash % 10) >= 8  # 80% train, 20% val
            if self.split == "train" and is_val:
                continue
            if self.split == "val" and not is_val:
                continue

            # --- Intrinsics: parse once per trajectory ---
            cam_path = traj_dir / "cam_left.json"
            if cam_path.is_file():
                cam_key = str(cam_path)
                if cam_key not in K_cache:
                    K, ds = self._load_intrinsics_full(cam_key)
                    K_cache[cam_key] = K
                    depth_scale_cache[cam_key] = ds
                traj_K = K_cache[cam_key]
                traj_depth_scale = depth_scale_cache[cam_key]
            else:
                logger.warning(
                    f"cam_left.json not found for {traj_key}; using 90° FOV "
                    f"default intrinsics (may be wrong for non-TartanAir data)."
                )
                traj_K = None
                traj_depth_scale = 1000.0

            # List image files; support .npy/.png/.tiff depth.
            # the DASVO TartanAir validation
            # split (kaggle.com/datasets/pandrii000/…) names images
            # 000000_left.png with depth files 000000_left_depth.npy — the
            # old exact-stem matcher (plus its _og fallback) dropped EVERY
            # frame of that dataset, yielding an empty index. Depth files
            # are now indexed by NORMALIZED stem (camera tags stripped), so
            # all known TartanAir conventions pair up:
            #   000000.png ↔ 000000.npy                 (assumed layout)
            #   000000_og.png ↔ 000000.npy              (official release)
            #   000000_left.png ↔ 000000_left_depth.npy (DASVO/Kaggle split)
            depth_index: Dict[str, Tuple[Path, str]] = {}
            for d_path in sorted(depth_dir.iterdir()):
                d_ext = d_path.suffix.lower()
                if d_ext not in (".npy", ".png", ".tiff"):
                    continue  # skips pose_left.txt and anything unrelated
                key = self._normalize_stem(d_path.stem)
                if key in depth_index:
                    logger.warning(
                        f"{traj_key}: ambiguous depth stems normalize to "
                        f"'{key}' ({depth_index[key][0].name} vs "
                        f"{d_path.name}); keeping the first."
                    )
                    continue
                depth_index[key] = (d_path, d_ext.lstrip("."))

            for img_path in sorted(img_dir.glob("*.png")):
                stem = img_path.stem
                key = self._normalize_stem(stem)
                entry = depth_index.get(key)
                if entry is None:
                    continue
                depth_path, depth_format = entry

                self.samples.append({
                    "image": str(img_path),
                    "depth": str(depth_path),
                    "depth_format": depth_format,
                    "intrinsics": traj_K,          # cached (3, 3) tensor or None
                    "depth_scale": traj_depth_scale,
                    "difficulty": diff_name,
                    "env": env_name,
                    "traj": traj_dir.name,
                    "scene": f"{env_name}/{traj_dir.name}/{stem}",
                })

        if not self.samples:
            raise FileNotFoundError(
                f"No samples found under {self.root} "
                f"(difficulty={self.difficulty}, split={self.split}). "
                f"Expected layout: {{env}}/{{Easy|Medium|Hard}}/P000/ or "
                f"{{Easy|Medium|Hard}}/{{env}}/P000/ with image_left/*.png + "
                f"depth_left/*.(npy|png)"
            )

    def _build_index_zip(self) -> None:
        """Index a TartanAir dataset shipped as a ZIP archive (no extraction).

        Kaggle mounts very large datasets as a raw archive.zip; with a ~57 GB
        uncompressed payload that cannot be extracted into the 20 GB
        /kaggle/working quota. Members are read lazily per __getitem__ call
        via zipfile random access (each member decompresses independently).

        Mirrors _build_index_tree exactly: same layout A/B scan, same
        (env/traj) hash split, same normalized-stem image↔depth pairing,
        same per-trajectory intrinsics caching.
        """
        with zipfile.ZipFile(self._zip_path, "r") as zf:
            raw_names = [n for n in zf.namelist() if not n.endswith("/")]

        # Strip a single common top-level folder (e.g. 'tartanair/') for the
        # layout scan, but keep the prefix — member reads need ORIGINAL names.
        prefix = ""
        if raw_names:
            tops = {n.split("/", 1)[0] for n in raw_names}
            if len(tops) == 1 and "/" in next(iter(raw_names)):
                prefix = next(iter(tops)) + "/"
                logger.info(
                    f"Zip dataset: scanning under top-level folder "
                    f"'{prefix.rstrip('/')}'; member reads keep the prefix."
                )
        names = [n[len(prefix):] for n in raw_names if n.startswith(prefix)]

        # Virtual directory tree, built in one pass.
        dirs = set()
        by_parent: Dict[str, List[str]] = {}
        for n in names:
            parent = n.rsplit("/", 1)[0] if "/" in n else ""
            by_parent.setdefault(parent, []).append(n)
            parts = n.split("/")
            for i in range(1, len(parts)):
                dirs.add("/".join(parts[:i]))

        difficulties = (
            ["Easy", "Medium", "Hard"] if self.difficulty == "all"
            else [self.DIFFICULTY_MAP.get(self.difficulty.lower(), "Easy")]
        )
        diff_set = set(difficulties)

        # Collect (traj_prefix, env_name, diff_name) across both layouts.
        traj_entries: List[Tuple[str, str, str]] = []
        top_names = sorted({n.split("/")[0] for n in names})
        for top_name in top_names:
            if top_name in diff_set:
                # Layout B: {difficulty}/{env}/P00x/
                envs = sorted({
                    d.split("/")[1] for d in dirs
                    if d.startswith(top_name + "/") and d.count("/") == 1
                })
                for env_name in envs:
                    for tr in sorted({
                        d.split("/")[2] for d in dirs
                        if d.startswith(f"{top_name}/{env_name}/")
                        and d.count("/") == 2
                    }):
                        traj_entries.append(
                            (f"{top_name}/{env_name}/{tr}", env_name, top_name)
                        )
            else:
                # Layout A: {env}/{difficulty}/P00x/
                for diff_name in difficulties:
                    dp = f"{top_name}/{diff_name}"
                    if dp not in dirs:
                        continue
                    for tr in sorted({
                        d.split("/")[2] for d in dirs
                        if d.startswith(dp + "/") and d.count("/") == 2
                    }):
                        traj_entries.append((f"{dp}/{tr}", top_name, diff_name))

        # TartanAir official uses 'Easy' and 'Hard' (no 'Medium' folder)
        if not traj_entries and self.difficulty.lower() == "medium":
            logger.info(
                "TartanAir zip uses 'Easy' and 'Hard' levels (no 'Medium' directory); "
                "aliasing requested 'medium' -> 'Hard'"
            )
            for top_name in top_names:
                if top_name == "Hard":
                    envs = sorted({
                        d.split("/")[1] for d in dirs
                        if d.startswith("Hard/") and d.count("/") == 1
                    })
                    for env_name in envs:
                        for tr in sorted({
                            d.split("/")[2] for d in dirs
                            if d.startswith(f"Hard/{env_name}/") and d.count("/") == 2
                        }):
                            traj_entries.append((f"Hard/{env_name}/{tr}", env_name, "Hard"))
                else:
                    dp = f"{top_name}/Hard"
                    if dp in dirs:
                        for tr in sorted({
                            d.split("/")[2] for d in dirs
                            if d.startswith(dp + "/") and d.count("/") == 2
                        }):
                            traj_entries.append((f"{dp}/{tr}", top_name, "Hard"))

        file_set = set(names)
        kept: List[Tuple[str, str, str]] = []
        for traj_prefix, env_name, diff_name in traj_entries:
            if f"{traj_prefix}/image_left" not in dirs:
                continue
            if f"{traj_prefix}/depth_left" not in dirs:
                continue

            # Academic zero-shot cross-environment split (v2):
            # Split strictly by environment name so train and val NEVER share 3D worlds.
            # Prevents intra-scene data leakage.
            if getattr(self.cfg, "split_mode", "cross_env") == "cross_env":
                env_key = env_name.lower().strip()
                env_hash = int(hashlib.md5(env_key.encode()).hexdigest(), 16)
                is_val = (env_hash % 10) >= 8  # 80% train envs, 20% held-out val envs
            else:
                # Legacy / intra-environment cross-trajectory split
                traj_key = f"{env_name}/{traj_prefix.rsplit('/', 1)[-1]}"
                traj_hash = int(hashlib.md5(traj_key.encode()).hexdigest(), 16)
                is_val = (traj_hash % 10) >= 8
            if self.split == "train" and is_val:
                continue
            if self.split == "val" and not is_val:
                continue
            kept.append((traj_prefix, env_name, diff_name))

        # Read all cam_left.json members in one zip open (prefix restored —
        # 'names' are layout-scan names, zip members keep the top folder).
        cam_members = [f"{tp}/cam_left.json" for tp, _, _ in kept
                       if f"{tp}/cam_left.json" in file_set]
        cam_data: Dict[str, bytes] = {}
        if cam_members:
            with zipfile.ZipFile(self._zip_path, "r") as zf:
                for member in cam_members:
                    cam_data[member] = zf.read(prefix + member)

        for traj_prefix, env_name, diff_name in kept:
            traj_key = f"{env_name}/{traj_prefix.rsplit('/', 1)[-1]}"
            img_prefix = traj_prefix + "/image_left/"
            depth_prefix = traj_prefix + "/depth_left/"

            # --- Intrinsics: parse once per trajectory ---
            cam_member = traj_prefix + "/cam_left.json"
            if cam_member in cam_data:
                traj_K, traj_depth_scale = self._parse_cam_json(cam_data[cam_member])
            else:
                logger.warning(
                    f"cam_left.json not found for {traj_key}; using 90° FOV "
                    f"default intrinsics (may be wrong for non-TartanAir data)."
                )
                traj_K = None
                traj_depth_scale = 1000.0

            # --- Depth index by normalized stem (same as dir mode) ---
            depth_index: Dict[str, Tuple[str, str]] = {}
            for member in sorted(by_parent.get(depth_prefix[:-1], [])):
                ext = member.rsplit(".", 1)[-1].lower()
                if ext not in ("npy", "png", "tiff"):
                    continue
                stem = member[len(depth_prefix):-len(ext) - 1]
                key = self._normalize_stem(stem)
                if key in depth_index:
                    logger.warning(
                        f"{traj_key}: ambiguous depth stems normalize to "
                        f"'{key}' ({depth_index[key][0]} vs {member}); "
                        f"keeping the first."
                    )
                    continue
                depth_index[key] = (member, ext)

            for member in sorted(by_parent.get(img_prefix[:-1], [])):
                if not member.lower().endswith(".png"):
                    continue
                stem = member[len(img_prefix):-4]
                key = self._normalize_stem(stem)
                entry = depth_index.get(key)
                if entry is None:
                    continue
                depth_member, depth_format = entry

                self.samples.append({
                    "image": prefix + member,
                    "depth": prefix + depth_member,
                    "depth_format": depth_format,
                    "intrinsics": traj_K,          # cached (3, 3) tensor or None
                    "depth_scale": traj_depth_scale,
                    "difficulty": diff_name,
                    "env": env_name,
                    "traj": traj_prefix.rsplit("/", 1)[-1],
                    "scene": f"{env_name}/{traj_prefix.rsplit('/', 1)[-1]}/{stem}",
                    "zip": str(self._zip_path),
                })

        if not self.samples:
            raise FileNotFoundError(
                f"No samples found inside zip {self._zip_path} "
                f"(difficulty={self.difficulty}, split={self.split}). "
                f"Expected layout: {{env}}/{{Easy|Medium|Hard}}/P000/ or "
                f"{{Easy|Medium|Hard}}/{{env}}/P000/ with image_left/*.png + "
                f"depth_left/*.(npy|png)"
            )

    @staticmethod
    def _normalize_stem(stem: str) -> str:
        """Strip camera/suffix tags so image and depth stems pair up across
        TartanAir naming conventions.

        One pass over the known tags handles every real convention:
          000000_left_depth -> 000000   (DASVO/Kaggle depth naming)
          000000_left       -> 000000   (DASVO/Kaggle image naming)
          000000_og         -> 000000   (official TartanAir image naming)
          000000            -> 000000   (plain naming)
        """
        for tag in ("_depth", "_og", "_left"):
            if stem.endswith(tag):
                stem = stem[: -len(tag)]
        return stem

    def _zip_read(self, zip_path: str, member: str) -> bytes:
        """Read one member from the dataset zip via a per-process handle.

        The handle cache is keyed by PID: DataLoader workers fork the parent
        process, and a fork would share the OS file descriptor — zipfile
        seeks on it, so interleaved reads from two processes would corrupt
        each other. Re-opening per PID is safe (the zip central directory
        is re-parsed once per worker, ~7 MB for the full 53k-file dataset).
        """
        pid = os.getpid()
        if self._zip_cache_pid != pid:
            self._zip_cache = {}
            self._zip_cache_pid = pid
        zf = self._zip_cache.get(zip_path)
        if zf is None:
            zf = zipfile.ZipFile(zip_path, "r")
            self._zip_cache[zip_path] = zf
        with zf.open(member) as f:
            return f.read()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Load and augment a single sample.

        Returns:
            Dict with keys:
                - "image": RGB image (3, H, W), float32 in [0, 1]
                - "depth": Depth map (1, H, W), float32 in metres
                - "intrinsics": Camera intrinsics (3, 3)
                - "valid": Valid depth mask (1, H, W), bool
        """
        sample = self.samples[idx]

        # --- Load image ---
        # Zip-backed samples carry raw member names; bytes are read through
        # a per-process handle (see _zip_read).
        zip_path = sample.get("zip")
        if zip_path:
            image = self._load_image(self._zip_read(zip_path, sample["image"]))
        else:
            image = self._load_image(sample["image"])

        # --- Load depth ---
        if zip_path:
            depth = self._load_depth(
                self._zip_read(zip_path, sample["depth"]),
                sample.get("depth_format", "npy"),
                depth_scale=sample.get("depth_scale", 1000.0),
            )
        else:
            depth = self._load_depth(
                sample["depth"],
                sample.get("depth_format", "npy"),
                depth_scale=sample.get("depth_scale", 1000.0),
            )

        # --- Intrinsics (cached at index time; clone before any mutation) ---
        # the fallback intrinsics are now built in the
        # ORIGINAL image pixel coordinate frame (using the loaded image size),
        # NOT in post-resize 224×224 coords. The old default (fx=112, cx=112)
        # was then run through the center-crop and resize transforms a SECOND
        # time, producing garbage on 640×480 inputs (measured: fx=52, cx=15
        # instead of the correct fx=149.3, cx=112 — a 65% focal error).
        # A 90°-FOV default in original coords (fx=W/2, fy=fx, cx=W/2, cy=H/2)
        # exactly reproduces TartanAir's true K (fx=fy=320, cx=320, cy=240 for
        # 640×480) when cam_left.json is absent.
        cached_K = sample.get("intrinsics")
        if isinstance(cached_K, Tensor):
            intrinsics = cached_K.clone()
        else:
            intrinsics = self._default_intrinsics(image.shape[1], image.shape[2])

        # --- Valid mask --- (BUG FIX: use named constants, not magic numbers)
        valid = (depth > VALID_DEPTH_MIN) & (depth < VALID_DEPTH_MAX) & torch.isfinite(depth)

        # --- Dynamic Pinhole Crop & Exact Intrinsics Tracking ---
        # TartanAir frames are 480×640. If K never varies, ray unprojection reduces
        # to a static 2D coordinate embedding. Pinhole crop augmentation dynamically
        # samples a square window of side L ∈ [min_scale * S_max, S_max], translating
        # (cx, cy) and scaling (fx, fy) uniformly to img_size (224×224).
        # This exercises real optical focal length and FOV variations (~45° to ~74° FOV)
        # while keeping metric depth Z in metres physically invariant.
        import torchvision.transforms.functional as TF

        H_orig, W_orig = image.shape[1], image.shape[2]
        S_max = min(H_orig, W_orig)

        if self.augment and getattr(self.cfg, "enable_pinhole_crop_aug", True):
            min_scale = getattr(self.cfg, "pinhole_crop_min_scale", 0.55)
            scale = random.uniform(min_scale, 1.0)
            L = int(round(S_max * scale))
            L = max(L, 64)
            x0 = random.randint(0, max(0, W_orig - L))
            y0 = random.randint(0, max(0, H_orig - L))
        else:
            # Deterministic isotropic center-crop (canonical pinhole sensor)
            L = S_max
            x0 = (W_orig - S_max) // 2
            y0 = (H_orig - S_max) // 2

        # Crop image, depth, valid
        image = image[:, y0:y0 + L, x0:x0 + L]
        depth = depth[y0:y0 + L, x0:x0 + L]
        valid = valid[y0:y0 + L, x0:x0 + L]

        # Shift optical center for crop window position
        intrinsics = intrinsics.clone()
        intrinsics[0, 2] -= float(x0)
        intrinsics[1, 2] -= float(y0)

        # Scale focal length and optical center for uniform resize to (img_size, img_size)
        s_scale = float(self.img_size) / float(L)
        intrinsics[0, 0] *= s_scale
        intrinsics[1, 1] *= s_scale
        intrinsics[0, 2] *= s_scale
        intrinsics[1, 2] *= s_scale

        # Resize tensors to (img_size, img_size)
        image = TF.resize(image, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        depth = TF.resize(depth.unsqueeze(0), [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        valid = TF.resize(valid.unsqueeze(0).float(), [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST).squeeze(0).bool()

        # --- Photometric and Horizontal Reflection Augmentation ---
        is_flipped = False
        if self.augment:
            image, depth, valid, intrinsics, is_flipped = self._augment(
                image, depth, valid, intrinsics
            )

        # --- ImageNet normalization ---
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image = (image - mean) / std

        return {
            "image": image,
            "depth": depth.unsqueeze(0),   # (1, H, W)
            "intrinsics": intrinsics,
            "valid": valid.unsqueeze(0),   # (1, H, W)
            "is_flipped": torch.tensor(is_flipped, dtype=torch.bool),
            "scene": sample.get("scene", ""),  # str: env/traj/stem for per-scene metrics
            "depth_format": sample.get("depth_format", "npy"),
        }

    # ----- I/O helpers -----

    @staticmethod
    def _load_image(src) -> Tensor:
        """Load RGB image as (3, H, W) float32 in [0, 1].

        Accepts a filesystem path or raw bytes (zip-backed datasets).
        """
        import io
        import torchvision.transforms.functional as TF
        from PIL import Image
        if isinstance(src, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(src))).convert("RGB")
        else:
            img = Image.open(src).convert("RGB")
        return TF.to_tensor(img)

    @staticmethod
    def _load_depth(src, depth_format: str = "npy",
                    depth_scale: float = 1000.0) -> Tensor:
        """Load depth map as (H, W) float32 in metres.

        Handles both 16-bit PNG and raw float32 .npy formats.
        (value = depth_metres × 1000), so .npy assumption silently produced
        an empty dataset on the real release. Now supports both formats.
        Accepts a filesystem path or raw bytes (zip-backed datasets).
        """
        import io
        if depth_format == "npy":
            if isinstance(src, (bytes, bytearray)):
                depth = np.load(io.BytesIO(bytes(src))).astype(np.float32)
            else:
                depth = np.load(src).astype(np.float32)
        elif depth_format == "png":
            from PIL import Image
            if isinstance(src, (bytes, bytearray)):
                depth = np.array(Image.open(io.BytesIO(bytes(src)))).astype(np.float32) / depth_scale
            else:
                depth = np.array(Image.open(src)).astype(np.float32) / depth_scale
        elif depth_format == "tiff":
            from PIL import Image
            if isinstance(src, (bytes, bytearray)):
                depth = np.array(Image.open(io.BytesIO(bytes(src)))).astype(np.float32) / depth_scale
            else:
                depth = np.array(Image.open(src)).astype(np.float32) / depth_scale
        else:
            raise ValueError(f"Unknown depth_format: {depth_format}")
        return torch.from_numpy(depth)

    @staticmethod
    def _load_intrinsics_full(path: str) -> Tuple[Optional[Tensor], float]:
        """Parse a TartanAir cam_left.json file → (K, depth_scale).

        See _parse_cam_json for the format details; this is the file-path
        wrapper (the zip-backed scanner calls _parse_cam_json on bytes).
        """
        with open(path, "rb") as f:
            return TartanAirDataset._parse_cam_json(f.read())

    @staticmethod
    def _parse_cam_json(data: bytes) -> Tuple[Optional[Tensor], float]:
        """Parse cam_left.json bytes → (K, depth_scale).

        BUG FIX (review pass 2): the original only looked for "K"/"intrinsic"
        keys, which do NOT exist in stock TartanAir cam_left.json — the real
        file uses "cam_K" (flattened row-major 3×3) and "cam_D" (distortion).
        Every sample therefore silently fell back to the default 90° FOV
        guess. Also returns depth_scale when present.

        Args:
            data: raw bytes of a cam_left.json file.

        Returns:
            (K, depth_scale): 3×3 intrinsics tensor (or None when no
            recognizable key exists) and depth scale factor.
        """
        import json
        cam = json.loads(data)

        # depth_scale first (needed by the no-K early return below)
        depth_scale = float(cam.get("depth_scale", 1000.0))
        if depth_scale <= 0 or not math.isfinite(depth_scale):
            depth_scale = 1000.0

        K = None
        for key in ("cam_K", "K", "intrinsic"):
            if key in cam:
                K = torch.tensor(cam[key], dtype=torch.float32)
                break
        # if the file exists but has no recognizable
        # key, return None so the caller falls back to the image-size-based
        # default in ORIGINAL pixel coords (the old code returned a 224×224-
        # coords matrix here too, causing the same double-transform garbage).
        if K is None:
            logger.warning(
                "cam_left.json has no cam_K/K/intrinsic key; using 90° FOV "
                "image-size-based default intrinsics."
            )
            return None, depth_scale
        if K.shape == (4, 4):
            K = K[:3, :3]
        elif K.numel() == 9:
            K = K.reshape(3, 3)
        else:
            raise ValueError(f"Unrecognized intrinsics shape in cam_left.json: {tuple(K.shape)}")

        return K, depth_scale

    @staticmethod
    def _default_intrinsics(height: int, width: int) -> Tensor:
        """Last-resort intrinsics when cam_left.json is missing or unusable.

        Expressed in the ORIGINAL image pixel coordinate frame (before any
        crop/resize), assuming a 90° horizontal FOV and square pixels:
            fx = fy = W/2,  cx = W/2,  cy = H/2
        For TartanAir's 640×480 frames this yields exactly the dataset's true
        calibration (fx=fy=320, cx=320, cy=240), so the downstream crop/resize
        transforms produce the correct 224×224 K (fx≈149.3, cx=cy=112).

        BUG FIX (Kaggle pass): the old default was hardcoded in POST-resize
        coords (fx=112) but was still run through the crop+resize transforms,
        double-transforming it into garbage on non-224 inputs.

        Args:
            height, width: Original image dimensions in pixels.
        """
        fx = fy = width / 2.0
        return torch.tensor([
            [fx,    0.0,   width / 2.0],
            [0.0,   fy,    height / 2.0],
            [0.0,   0.0,   1.0],
        ], dtype=torch.float32)

    # ----- Augmentation -----

    def _augment(
        self,
        image: Tensor,
        depth: Tensor,
        valid: Tensor,
        intrinsics: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, bool]:
        """Apply spatial and photometric data augmentations.

        Augmentations:
            1. Random horizontal flip (with chiral ray tracking)
            2. Color jitter (temporal consistency — draw params once)
            3. Random resized crop

        Args:
            image: (3, H, W)
            depth: (H, W)
            valid: (H, W)
            intrinsics: (3, 3)

        Returns:
            Augmented (image, depth, valid, intrinsics, is_flipped).
        """
        import torchvision.transforms.functional as TF

        # --- Random horizontal flip ---
        is_flipped = False
        if random.random() < 0.5:
            image = TF.hflip(image)
            depth = TF.hflip(depth)
            valid = TF.hflip(valid)
            is_flipped = True

            # BUG FIX from v5.2: Under reflection M = diag(-1,1,1):
            # SE(3) twist transforms as: v' = Mv, ω' = (ω_x, -ω_y, -ω_z)
            # The omega_z negation was MISSING in v5.2.
            # Since this is a 3D-only model, action is not used in training,
            # but we keep the fix for future 4D extension.
            # Intrinsics: flip cx → W - cx (exact pinhole principal point reflection)
            W_img = image.shape[2]
            intrinsics = intrinsics.clone()
            intrinsics[0, 2] = float(W_img) - intrinsics[0, 2]

        # --- Color jitter (with temporal consistency fix) ---
        # Draw jitter params ONCE, apply identically to both frames
        # (Fix from v5.2 where independent jitter broke temporal consistency)
        if random.random() < 0.5:
            image = self._apply_color_jitter(image)


        # --- Photometric augmentations ---
        # Gamma correction
        if random.random() < self.cfg.gamma_aug_prob:
            gamma = random.uniform(0.7, 1.5)
            image = image.pow(gamma).clamp(0.0, 1.0)

        # Fog simulation: image = image * exp(-β*depth) + fog_color * (1 - exp(-β*depth))
        if random.random() < self.cfg.fog_aug_prob:
            beta_fog = random.uniform(0.01, 0.1)
            fog_color = random.uniform(0.7, 1.0)
            depth_clamped = depth.clamp(min=0.0)
            transmission = torch.exp(-beta_fog * depth_clamped).unsqueeze(0)  # (1, H, W)
            image = image * transmission + fog_color * (1.0 - transmission)
            image = image.clamp(0.0, 1.0)

        # Small rotation with intrinsics update
        if random.random() < self.cfg.rotation_aug_prob:
            max_rad = math.radians(self.cfg.rotation_max_deg)
            angle = random.uniform(-max_rad, max_rad)
            image = TF.rotate(image, math.degrees(angle),
                              interpolation=TF.InterpolationMode.BILINEAR)
            depth = TF.rotate(depth.unsqueeze(0), math.degrees(angle),
                              interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
            valid = TF.rotate(valid.unsqueeze(0).float(), math.degrees(angle),
                              interpolation=TF.InterpolationMode.NEAREST).squeeze(0).bool()
            # Update intrinsics principal point for rotation
            # For small rotations, the principal point rotates with the image
            cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()
            img_cx, img_cy = image.shape[2] / 2.0, image.shape[1] / 2.0
            # Offset from center, rotate, add back
            dx, dy = cx - img_cx, cy - img_cy
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            intrinsics = intrinsics.clone()
            intrinsics[0, 2] = img_cx + dx * cos_a - dy * sin_a
            intrinsics[1, 2] = img_cy + dx * sin_a + dy * cos_a

        # --- Additional photometric augmentations ---
        # Gaussian blur (simulates focus/defocus — TartanAir renders are sharp)
        # kernel_size=3 badly truncates σ > 0.5 (a 3-tap
        # kernel can't represent a wide Gaussian — the blur silently becomes
        # much weaker than sampled). Use a 5-tap kernel with σ ∈ [0.1, 1.0].
        if random.random() < 0.2:
            sigma = random.uniform(0.1, 1.0)
            import torchvision.transforms.functional as TF
            image = TF.gaussian_blur(image, kernel_size=5, sigma=sigma)

        # Gaussian noise (simulates sensor noise absent in synthetic)
        if random.random() < 0.2:
            sigma_noise = random.uniform(0.01, 0.05)
            noise = torch.randn_like(image) * sigma_noise
            image = (image + noise).clamp(0.0, 1.0)

        # Random erasing (regularizer)
        if random.random() < 0.25:
            area = image.shape[1] * image.shape[2]
            target_area = random.uniform(0.02, 0.2) * area
            aspect = random.uniform(0.3, 3.0)
            eh = int(round((target_area * aspect) ** 0.5))
            ew = int(round((target_area / aspect) ** 0.5))
            if eh < image.shape[1] and ew < image.shape[2]:
                top = random.randint(0, image.shape[1] - eh)
                left = random.randint(0, image.shape[2] - ew)
                image[:, top:top+eh, left:left+ew] = torch.rand(3, device=image.device).view(3, 1, 1)

        # --- Intrinsics augmentation (K-jitter) ---
        intrinsics = self._augment_intrinsics(intrinsics, image.shape[1:])

        return image, depth, valid, intrinsics, is_flipped

    def _apply_color_jitter(self, image: Tensor) -> Tensor:
        """Apply color jitter with parameters drawn once (temporal consistency).

        Args:
            image: (3, H, W) in [0, 1] (before ImageNet normalization).

        Returns:
            Jittered image (3, H, W).
        """
        import torchvision.transforms.functional as TF

        # Draw random factors
        brightness = 1.0 + random.uniform(-self.cfg.color_jitter_brightness,
                                           self.cfg.color_jitter_brightness)
        contrast = 1.0 + random.uniform(-self.cfg.color_jitter_contrast,
                                         self.cfg.color_jitter_contrast)
        saturation = 1.0 + random.uniform(-self.cfg.color_jitter_saturation,
                                           self.cfg.color_jitter_saturation)
        hue = random.uniform(-self.cfg.color_jitter_hue, self.cfg.color_jitter_hue)

        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)
        image = TF.adjust_hue(image, hue)

        return image.clamp(0.0, 1.0)

    def _random_crop(
        self,
        image: Tensor,
        depth: Tensor,
        valid: Tensor,
        intrinsics: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Random resized crop to img_size with intrinsics update.

        Args:
            image: (3, H, W)
            depth: (H, W)
            valid: (H, W)
            intrinsics: (3, 3)

        Returns:
            Cropped (image, depth, valid, intrinsics) at img_size.
        """
        import torchvision.transforms.functional as TF

        H, W = image.shape[1:]
        scale = random.uniform(*self.cfg.random_crop_scale)

        # Compute crop size
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        crop_h = max(crop_h, self.img_size)
        crop_w = max(crop_w, self.img_size)

        # Random top-left corner
        top = random.randint(0, max(0, H - crop_h))
        left = random.randint(0, max(0, W - crop_w))

        # Update intrinsics for crop offset
        K = intrinsics.clone()
        K[0, 2] -= left  # cx shifts by crop offset
        K[1, 2] -= top   # cy shifts by crop offset

        image = TF.crop(image, top, left, crop_h, crop_w)
        depth = TF.crop(depth.unsqueeze(0), top, left, crop_h, crop_w).squeeze(0)
        valid = TF.crop(valid.unsqueeze(0).float(), top, left, crop_h, crop_w).squeeze(0).bool()

        # Update intrinsics for resize
        scale_h = self.img_size / crop_h
        scale_w = self.img_size / crop_w
        K[0, 0] *= scale_w  # fx
        K[1, 1] *= scale_h  # fy
        K[0, 2] *= scale_w  # cx
        K[1, 2] *= scale_h  # cy

        # Resize to target
        image = TF.resize(image, [self.img_size, self.img_size])
        depth = TF.resize(depth.unsqueeze(0), [self.img_size, self.img_size],
                          interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        valid = TF.resize(valid.unsqueeze(0).float(), [self.img_size, self.img_size],
                          interpolation=TF.InterpolationMode.NEAREST).squeeze(0).bool()

        return image, depth, valid, K

    # ----- Resize helpers -----

    def _augment_intrinsics(self, intrinsics: Tensor, img_shape: Tuple[int, int]) -> Tensor:
        """Jitter camera intrinsics (K-jitter).

        Jitter focal length by ±k_jitter_focal and principal point by
        ±k_jitter_principal of image size.

        Args:
            intrinsics: (3, 3) camera intrinsics.
            img_shape: (H, W) image shape.

        Returns:
            Jittered intrinsics (3, 3).
        """
        K = intrinsics.clone()
        H, W = img_shape

        # independent fx/fy jitter creates non-square pixels
        # (rare in real cameras, confuses IRER). Use a single focal_scale
        # plus a small separate pixel-aspect jitter.
        focal_scale = 1.0 + random.uniform(-self.cfg.k_jitter_focal, self.cfg.k_jitter_focal)
        pixel_aspect = 1.0 + random.uniform(-0.02, 0.02)  # ±2% pixel aspect
        K[0, 0] *= focal_scale
        K[1, 1] *= focal_scale * pixel_aspect

        # Jitter principal point
        cx_offset = random.uniform(-self.cfg.k_jitter_principal, self.cfg.k_jitter_principal) * W
        cy_offset = random.uniform(-self.cfg.k_jitter_principal, self.cfg.k_jitter_principal) * H
        K[0, 2] += cx_offset
        K[1, 2] += cy_offset

        return K

    def _resize_image(self, image: Tensor) -> Tensor:
        """Resize image to (3, img_size, img_size)."""
        import torchvision.transforms.functional as TF
        if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
            image = TF.resize(image, [self.img_size, self.img_size])
        return image

    def _resize_depth(self, depth: Tensor) -> Tensor:
        """Resize depth map to (img_size, img_size) using nearest interpolation."""
        import torchvision.transforms.functional as TF
        if depth.shape[0] != self.img_size or depth.shape[1] != self.img_size:
            depth = TF.resize(
                depth.unsqueeze(0), [self.img_size, self.img_size],
                interpolation=TF.InterpolationMode.NEAREST,
            ).squeeze(0)
        return depth

    def _resize_valid(self, valid: Tensor) -> Tensor:
        """Resize valid mask to (img_size, img_size) using nearest interpolation."""
        import torchvision.transforms.functional as TF
        if valid.shape[0] != self.img_size or valid.shape[1] != self.img_size:
            valid = TF.resize(
                valid.unsqueeze(0).float(), [self.img_size, self.img_size],
                interpolation=TF.InterpolationMode.NEAREST,
            ).squeeze(0).bool()
        return valid


# ======================================================================
# Curriculum difficulty sampler
# ======================================================================

class CurriculumSampler:
    """Curriculum learning sampler that adjusts difficulty over epochs.

    Phase 1 (epochs 0-15):  Easy only
    Phase 2 (epochs 15-30): 70% Easy + 30% Medium
    Phase 3 (epochs 30+):   50% Easy + 30% Medium + 20% Hard

    Args:
        cfg: CurriculumConfig with phase boundaries and proportions.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or CurriculumConfig()

    def get_difficulty(self, epoch: int) -> str:
        """Get dataset difficulty for the given epoch.

        Args:
            epoch: Current training epoch.

        Returns:
            Difficulty string: "easy", "medium", or "hard".
        """
        r = random.random()

        if epoch < self.cfg.phase1_end:
            # Phase 1: Easy only
            return "easy"
        elif epoch < self.cfg.phase2_end:
            # Phase 2: Easy + Medium
            if r < self.cfg.phase2_easy:
                return "easy"
            else:
                return "medium"
        else:
            # Phase 3: Easy + Medium + Hard
            if r < self.cfg.phase3_easy:
                return "easy"
            elif r < self.cfg.phase3_easy + self.cfg.phase3_medium:
                return "medium"
            else:
                return "hard"

    def get_difficulty_distribution(self, epoch: int) -> Dict[str, float]:
        """Get the difficulty distribution for the given epoch.

        Returns:
            Dict mapping difficulty name to proportion.
        """
        if epoch < self.cfg.phase1_end:
            return {"easy": 1.0}
        elif epoch < self.cfg.phase2_end:
            return {"easy": self.cfg.phase2_easy, "medium": self.cfg.phase2_medium}
        else:
            return {
                "easy": self.cfg.phase3_easy,
                "medium": self.cfg.phase3_medium,
                "hard": self.cfg.phase3_hard,
            }


#════════════════════════════════════════════════════════════════════════════#
# MODULE: train.py
#────────────────────────────────────────────────────────────────────────────#
"""
Training loop for Tesseract v1.

Implements the complete training pipeline with all bug fixes from v5.2:
    - Two parameter groups with different weight decay
    - Depth head at 5× LR (was 10× — too aggressive)
    - Linear warmup → cosine annealing
    - Mixed precision with conservative GradScaler
    - Gradient accumulation with TAIL FLUSH (bug fix)
    - Per-group gradient clipping (fix for 5× LR mismatch)
    - Even-only OOM batch halving (8→4→2, never 3)
    - Early stopping with correct logic (fix: don't update best in elif)
    - nn.DataParallel default (DDD note for ~15% speedup)

References:
    - AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", 2019
    - AMP: PyTorch automatic mixed precision
    - Gradient accumulation: effective batch = batch_size × accumulate_steps
"""


import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR

# from .config import TesseractConfig, ModelConfig, TrainConfig
# from .losses import silog_loss, edge_gradient_loss, planarity_loss, l1_loss, image_aware_smoothness_loss, UncertaintyWeightedLoss
# from .metrics import compute_eigen_metrics
# from .viz import save_visualization

logger = logging.getLogger(__name__)


def compute_irer_gate(epoch: int, irer_warmup_epochs: int = 10, irer_full_epochs: int = 40) -> float:
    """Compute IRER gate value for the given epoch.

    Smoothstep schedule:
        - epoch < irer_warmup_epochs: gate = 0.0 (IRER disabled)
        - irer_warmup_epochs <= epoch < irer_full_epochs: smoothstep ramp
        - epoch >= irer_full_epochs: gate = 1.0 (IRER at full strength)

    Args:
        epoch: Current training epoch.
        irer_warmup_epochs: Epochs before IRER activates.
        irer_full_epochs: Epochs when IRER is at full strength.

    Returns:
        Gate value in [0.0, 1.0].
    """
    if epoch < irer_warmup_epochs:
        return 0.0
    elif epoch < irer_full_epochs:
        t = (epoch - irer_warmup_epochs) / (irer_full_epochs - irer_warmup_epochs)
        return 2 * t * t if t < 0.5 else 1 - 2 * (1 - t) ** 2  # smoothstep
    else:
        return 1.0


def get_head_lr_multiplier(epoch: int, head_lr_start: float = 10.0,
                           head_lr_end: float = 3.0, transition_epoch: int = 30) -> float:
    """Compute staged head LR multiplier.

    Linearly decreases from head_lr_start to head_lr_end over
    transition_epoch epochs, then stays at head_lr_end.

    Args:
        epoch: Current training epoch.
        head_lr_start: Initial head LR multiplier.
        head_lr_end: Final head LR multiplier.
        transition_epoch: Epoch to reach head_lr_end.

    Returns:
        Head LR multiplier.
    """
    if epoch < transition_epoch:
        return head_lr_start - (head_lr_start - head_lr_end) * epoch / transition_epoch
    return head_lr_end


class StepLogger:
    """Simple per-step logger that accumulates loss values and logs every N steps.

    Args:
        log_interval: Number of steps between log emissions.
    """

    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self.accumulated: Dict[str, float] = {}
        self.count = 0

    def step(self, loss_dict: Dict[str, float], global_step: int) -> None:
        """Accumulate a loss dict and log if at interval.

        Args:
            loss_dict: Dict of loss name → value.
            global_step: Current global step.
        """
        for k, v in loss_dict.items():
            self.accumulated[k] = self.accumulated.get(k, 0.0) + v
        self.count += 1

        # Log early steps (1, 5, 10, 25) so progress is immediately visible without waiting 50 steps
        is_early = global_step in (1, 5, 10, 25)
        if self.count % self.log_interval == 0 or is_early:
            avg = {k: v / max(1, self.count) for k, v in self.accumulated.items()}
            logger.info(
                f"Step {global_step} | " +
                " | ".join(f"{k}: {v:.4f}" for k, v in avg.items())
            )
            self.accumulated.clear()
            self.count = 0


# ======================================================================
# Optimizer & Scheduler Construction
# ======================================================================

def build_optimizer(
    model: nn.Module,
    cfg: TrainConfig,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with separate parameter groups.

    Two main groups:
        1. Encoder: base LR, weight_decay=0.05 for weights, 0.0 for biases
        2. Depth head: 5× base LR (was 10× in v5.2 — too aggressive)

    This separation is critical because the depth head trains from a
    random initialization while the encoder benefits from slower, more
    stable learning.

    Args:
        model: Tesseract model.
        cfg: TrainConfig.

    Returns:
        AdamW optimizer with configured parameter groups.
    """
    # Separate depth head parameters
    depth_head_params = set(id(p) for p in model.depth_head.parameters())
    # Actually USE uncertainty_params for separate group.
    # Previously, uncertainty_params was computed but never checked,
    # so log_vars got weight_decay=0.05 which is wrong — it would
    # regularize the log-variances toward zero, biasing loss weighting.
    uncertainty_params = set(id(p) for p in model.loss_fn.parameters()) if hasattr(model, 'loss_fn') else set()

    param_groups = []
    weights, biases = [], []
    dh_weights, dh_biases = [], []
    unc_params = []  # uncertainty log_vars — no weight decay!

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check uncertainty params FIRST (they must have weight_decay=0)
        if id(param) in uncertainty_params:
            unc_params.append(param)
            continue

        is_depth_head = id(param) in depth_head_params
        is_bias_or_norm = (param.ndim <= 1) or ("norm" in name.lower())

        if is_depth_head:
            if is_bias_or_norm:
                dh_biases.append(param)
            else:
                dh_weights.append(param)
        else:
            if is_bias_or_norm:
                biases.append(param)
            else:
                weights.append(param)

    # Encoder weights: weight_decay=0.05
    if weights:
        param_groups.append({"params": weights, "lr": cfg.base_lr, "weight_decay": cfg.weight_decay, "name": "encoder_weights"})
    # Encoder biases/norms: weight_decay=0.0
    if biases:
        param_groups.append({"params": biases, "lr": cfg.base_lr, "weight_decay": cfg.weight_decay_bias, "name": "encoder_biases"})
    # Depth head weights: 5× LR, weight_decay=0.05
    if dh_weights:
        param_groups.append({"params": dh_weights, "lr": cfg.depth_head_lr, "weight_decay": cfg.weight_decay, "name": "depth_head_weights"})
    # Depth head biases/norms: 5× LR, weight_decay=0.0
    if dh_biases:
        param_groups.append({"params": dh_biases, "lr": cfg.depth_head_lr, "weight_decay": cfg.weight_decay_bias, "name": "depth_head_biases"})
    # Uncertainty log_vars — base LR, NO weight decay
    # Weight decay on log_vars would shrink them to zero, destroying
    # the learned loss balance (Kendall 2018).
    if unc_params:
        param_groups.append({"params": unc_params, "lr": cfg.base_lr, "weight_decay": 0.0, "name": "uncertainty_log_vars"})

    return torch.optim.AdamW(param_groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build LR scheduler: warmup → cosine (or warm restarts).

    Schedule:
        - Epochs 0 to warmup_epochs: linear ramp from 0.01× to 1.0× LR
        - After warmup:
            - If use_warm_restarts: CosineAnnealingWarmRestarts
            - Else: cosine annealing to min_lr

    Args:
        optimizer: AdamW optimizer.
        cfg: TrainConfig.
        steps_per_epoch: Number of optimizer steps per epoch.

    Returns:
        LR scheduler.
    """
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.total_epochs * steps_per_epoch

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)

    if cfg.use_warm_restarts:
        cosine = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.restart_t0 * steps_per_epoch,
            T_mult=cfg.restart_t_mult,
            eta_min=cfg.min_lr,
        )
    else:
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=cfg.min_lr,
        )

    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


# ======================================================================
# Training Step
# ======================================================================

def train_step(
    model: nn.Module,
    batch: Dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: UncertaintyWeightedLoss,
    cfg: TesseractConfig,
    accumulate_step: int,
    device: torch.device,
    epoch: int = 0,
) -> Tuple[Tensor, Dict[str, float]]:
    """Single training step with gradient accumulation and AMP.

    Args:
        model: Tesseract model (possibly wrapped in DataParallel).
        batch: Dict with "image", "depth", "intrinsics", "valid".
        optimizer: AdamW optimizer.
        scaler: AMP GradScaler.
        loss_fn: UncertaintyWeightedLoss.
        cfg: Full configuration.
        accumulate_step: Current step within accumulation cycle.
        device: Target device.
        epoch: Current epoch (for IRER gate computation).

    Returns:
        total_loss: Scalar loss for this step.
        loss_dict: Dict of individual loss values for logging.
    """
    image = batch["image"].to(device)
    gt_depth = batch["depth"].to(device)
    intrinsics = batch["intrinsics"].to(device)
    valid = batch["valid"].to(device)
    is_flipped = batch.get("is_flipped", None)
    if is_flipped is not None:
        is_flipped = is_flipped.to(device)

    # Compute IRER gate for this epoch
    irer_gate = compute_irer_gate(
        epoch,
        irer_warmup_epochs=cfg.model.irer_warmup_epochs,
        irer_full_epochs=cfg.model.irer_full_epochs,
    )

    # Forward pass with AMP (enabled only on CUDA; no-op on MPS/CPU)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(amp_device, enabled=scaler.is_enabled()):
        pred_depth, _, _ = model(image, intrinsics, irer_gate=irer_gate, is_flipped=is_flipped)

    # Cast outputs to float32 outside autocast.
    # Loss functions must execute in full float32 to prevent
    # FP16 overflow (>65504 -> +inf), which triggered inf * 0.0 -> NaN in edge loss.
    pred_depth = pred_depth.float()
    gt_depth = gt_depth.float()
    # Mask invalid, non-finite, and non-positive pixels across pred and gt
    valid = (valid > 0) & torch.isfinite(gt_depth) & (gt_depth > 0) & torch.isfinite(pred_depth) & (pred_depth > 0)

    # Normalize once for relative shape and edge supervision
    pred_norm = median_normalize(pred_depth, gt_depth, valid)

    # Compute individual losses (in float32 for absolute numerical stability)
    if cfg.loss.use_metric_supervision:
        # Metric supervision: SiLog and Log-L1 operate on raw metric depth
        l_silog = silog_loss(pred_depth, gt_depth, valid, cfg.loss.silog_lambda_w, use_median_norm=False)
        l_l1 = l1_loss(pred_depth, gt_depth, valid)
    else:
        # Relative depth mode: SiLog operates on median-normalized depth
        l_silog = silog_loss(pred_depth, gt_depth, valid, cfg.loss.silog_lambda_w, pred_norm=pred_norm)
        l_l1 = l1_loss(pred_depth, gt_depth, valid, pred_norm=pred_norm)

    l_edge = edge_gradient_loss(pred_depth, gt_depth, valid, cfg.loss, pred_norm=pred_norm,
                                 scales=cfg.loss.edge_scales)
    l_smooth = image_aware_smoothness_loss(pred_depth, gt_depth, valid, image, cfg.loss, pred_norm=pred_norm)
    l_planar = planarity_loss(pred_depth, gt_depth, valid, cfg.loss, pred_norm=pred_norm)

    # Uncertainty-weighted total (5 losses)
    total = loss_fn((l_silog, l_l1, l_edge, l_smooth, l_planar))

    # Scale consistency loss: penalize discrepancy between batch predicted vs GT medians
    l_scale = torch.tensor(0.0, device=device)
    if cfg.loss.use_metric_supervision and cfg.loss.scale_loss_weight > 0:
        scale_losses = []
        B = pred_depth.shape[0]
        for b in range(B):
            v_b = valid[b, 0]
            if v_b.sum() > 10:
                p_med = pred_depth[b, 0][v_b].median().clamp(min=1e-4)
                g_med = gt_depth[b, 0][v_b].median().clamp(min=1e-4)
                scale_losses.append(torch.abs(torch.log(p_med) - torch.log(g_med)))
        if scale_losses:
            l_scale = torch.stack(scale_losses).mean()
            total = total + cfg.loss.scale_loss_weight * l_scale

    # Anomaly guard: if total loss is non-finite, fall back strictly to finite loss terms
    if not torch.isfinite(total):
        logger.warning(
            f"train_step: non-finite loss detected (silog={l_silog.item():.4f}, "
            f"l1={l_l1.item():.4f}, edge={l_edge.item():.4f}, "
            f"smooth={l_smooth.item():.4f}, planar={l_planar.item():.4f}, "
            f"scale={l_scale.item():.4f}). "
            "Falling back to finite loss terms."
        )
        finite_losses = [l for l in (l_silog, l_l1, l_edge, l_smooth, l_planar, l_scale) if torch.isfinite(l)]
        if finite_losses:
            total = torch.stack(finite_losses).sum()
        else:
            total = torch.tensor(0.0, device=device, requires_grad=True)

    # Scale by accumulation factor
    total = total / cfg.train.accumulate_steps

    # Backward pass with GradScaler
    scaler.scale(total).backward()

    loss_dict = {
        "silog": l_silog.item(),
        "l1": l_l1.item(),
        "edge": l_edge.item(),
        "smoothness": l_smooth.item(),
        "planarity": l_planar.item(),
        "scale": l_scale.item(),
        "total": total.item() * cfg.train.accumulate_steps,
        "irer_gate": irer_gate,
    }

    return total.detach(), loss_dict


# ======================================================================
# Optimizer Step (with all fixes)
# ======================================================================

def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: SequentialLR,
    cfg: TrainConfig,
) -> None:
    """Take an optimizer step with per-group gradient clipping.

    Critical fixes from v5.2:
        1. Per-group clipping (uniform clip with 5× LR mismatch caused
           the depth head to be under-constrained)
        2. Proper scaler.unscale_ → clip → scaler.step ordering

    Args:
        optimizer: AdamW optimizer.
        scaler: AMP GradScaler.
        scheduler: LR scheduler.
        cfg: TrainConfig.
    """
    # Unscale gradients before clipping
    scaler.unscale_(optimizer)

    # single global gradient clipping across ALL parameters
    # (original used per-group clip with same max_norm — doesn't fix the
    # 5× LR mismatch, just clips each group tighter and starves the encoder)
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(all_params, max_norm=cfg.grad_clip_norm)

    # Step optimizer and update scaler
    scaler.step(optimizer)
    scaler.update()

    # Step scheduler
    scheduler.step()

    # Zero gradients for next accumulation cycle
    optimizer.zero_grad(set_to_none=True)


# ======================================================================
# Validation
# ======================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 200,
    num_viz: int = 16,
    viz_dir: Optional[str] = None,
    irer_gate: float = 1.0,
    use_amp: bool = False,
    eval_min_depth: float = 0.2,
    eval_max_depth: float = 80.0,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Run validation and compute Eigen metrics.

    BUG FIX: pass `irer_gate` so validation matches training regime.
    Originally validate used irer_gate=1.0 unconditionally while training
    ran at gate ~0.02, so val metrics weren't representative of training.

    BUG FIX (review pass 2): AMP is now opt-in via `use_amp` so fp32
    debugging runs validate in fp32 (previously autocast was hardcoded on
    whenever CUDA was present, regardless of the training regime).

    Args:
        model: Tesseract model.
        val_loader: Validation data loader.
        device: Target device.
        num_batches: Number of validation batches (200 → 800 samples).
        num_viz: Number of visualization samples.
        viz_dir: Directory for visualization outputs (None → skip viz).
        irer_gate: IRER gate value matching the current training epoch.
        use_amp: Run the forward pass under autocast (matches AMP training).
        eval_min_depth: Minimum evaluation depth threshold (m).
        eval_max_depth: Maximum evaluation depth threshold (m).

    Returns:
        metrics: Dict of overall Eigen metrics.
        per_scene: Dict of per-scene metrics.
    """
    model.eval()

    all_pred = []
    all_gt = []
    all_valid = []
    all_scene_names = []
    viz_count = 0

    for i, batch in enumerate(val_loader):
        if i >= num_batches:
            break

        image = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        valid = batch["valid"].to(device)
        scene_names = batch.get("scene", [f"val_{i}_{j}" for j in range(image.shape[0])])

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred_depth, _, _ = model(image, intrinsics, irer_gate=irer_gate)

        # Cast to fp32 immediately to halve CPU memory pressure
        all_pred.append(pred_depth.float().cpu())
        all_gt.append(gt_depth.float().cpu())
        all_valid.append(valid.cpu())
        all_scene_names.extend(scene_names)

        # Visualization
        if viz_dir is not None and viz_count < num_viz:
            for b in range(min(image.shape[0], num_viz - viz_count)):
                save_visualization(
                    image[b].cpu(),
                    gt_depth[b, 0].cpu(),
                    pred_depth[b, 0].cpu(),
                    str(Path(viz_dir) / f"val_{viz_count:04d}.png"),
                    valid[b, 0].cpu(),
                )
                viz_count += 1

        del image, gt_depth, intrinsics, valid, pred_depth

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Concatenate all validation data
    # guard against zero processed batches (empty
    # loader / num_batches=0) — torch.cat([]) raises and kills training.
    if not all_pred:
        logger.warning("validate: no batches were processed; returning NaN metrics")
        model.train()
        return (
            {
                k: float("nan")
                for k in (
                    "abs_rel", "rmse", "log10", "silog", "d1", "d2", "d3",
                    "metric_abs_rel", "metric_rmse", "metric_d1",
                )
            },
            {},
        )
    pred_all = torch.cat(all_pred, dim=0)
    gt_all = torch.cat(all_gt, dim=0)
    valid_all = torch.cat(all_valid, dim=0)

    # Compute metrics with evaluation depth bounds
    metrics = compute_eigen_metrics(
        pred_all, gt_all, valid_all,
        min_depth=eval_min_depth, max_depth=eval_max_depth,
    )
    per_scene: Dict[str, Dict[str, float]] = {}

    # Simple per-scene computation
    scene_names_tuple = tuple(all_scene_names)
    unique_scenes = set(scene_names_tuple)
    if len(unique_scenes) > 1:
        _, per_scene = compute_per_scene_metrics(
            pred_all, gt_all, valid_all, scene_names_tuple,
            min_depth=eval_min_depth, max_depth=eval_max_depth,
        )

    model.train()
    return metrics, per_scene


# ======================================================================
# OOM Handler
# ======================================================================

def handle_oom(batch_size: int) -> int:
    """Handle OOM by halving batch size (kept even for DataParallel).

    BUG FIX: original formula `max(2, (bs//2)*2)` returned bs unchanged
    for any even input (8→8, 4→4, 2→2), causing an OOM death-loop.
    Correct: `max(2, bs // 2)`, then snap down to even if needed.

    Args:
        batch_size: Current batch size.

    Returns:
        New batch size (halved, even, at least 2).
    """
    new_bs = max(2, batch_size // 2)
    if new_bs % 2:  # keep even for DataParallel on 2 GPUs
        new_bs -= 1
    logger.warning(f"OOM detected, halving batch_size: {batch_size} → {new_bs}")
    return new_bs


def _rebuild_loader(
    loader: torch.utils.data.DataLoader,
    batch_size: int,
    dataset: Optional[torch.utils.data.Dataset] = None,
) -> torch.utils.data.DataLoader:
    """Clone a DataLoader's settings with a new batch size (and optional
    dataset override).

    BUG FIX (review pass 2): OOM batch-halving previously updated only a
    local variable — the DataLoader kept its original batch size forever,
    so every subsequent batch OOM'd again (death loop). The Curriculum
    sampler also needs dataset swaps at phase boundaries.
    """
    ds = dataset if dataset is not None else loader.dataset
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler),
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        persistent_workers=getattr(loader, "persistent_workers", False) and loader.num_workers > 0,
        prefetch_factor=(loader.prefetch_factor if loader.num_workers > 0 else None),
        collate_fn=loader.collate_fn,
        # preserve the per-worker seeding function so
        # augmented samples stay reproducible after curriculum/OOM rebuilds.
        worker_init_fn=getattr(loader, "worker_init_fn", None),
    )


# ======================================================================
# Early Stopping
# ======================================================================

class EarlyStopping:
    """Early stopping with correct logic (fix from v5.2).

    Bug fix: In v5.2, the elif branch updated best_metric, which
    prevented the patience counter from properly incrementing for
    small improvements. The correct logic:
        - Significant improvement (< best - min_delta): reset patience
        - Small improvement (< best but ≥ best - min_delta): DON'T reset,
          increment patience (the improvement isn't enough to justify
          continuing)
        - No improvement: increment patience

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum improvement threshold.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.counter = 0

    def step(self, current_metric: float) -> bool:
        """Check if training should stop.

        Args:
            current_metric: Current validation metric (lower is better).

        Returns:
            True if training should stop (patience exceeded).
        """
        if current_metric < self.best_metric - self.min_delta:
            # Significant improvement
            self.best_metric = current_metric
            self.counter = 0
            return False
        elif current_metric < self.best_metric:
            # Small improvement: DON'T update best_metric
            # (fix from v5.2 — this was incorrectly updating best)
            self.counter += 1
        else:
            # No improvement
            self.counter += 1

        return self.counter >= self.patience


# ======================================================================
# Main Training Loop
# ======================================================================

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: TesseractConfig,
    device: torch.device,
    checkpoint_path: str = "checkpoint.pt",
    resume: bool = True,
    finetune_from: Optional[str] = None,
    tb_writer: Optional[object] = None,
    wandb_run: Optional[object] = None,
    start_time: Optional[float] = None,
) -> None:
    """Main training loop for Tesseract v1.

    Implements the complete training pipeline:
        1. Build optimizer with separate param groups
        2. Build scheduler (warmup → cosine)
        3. Initialize AMP GradScaler
        4. Load checkpoint if resuming or pretrained weights if fine-tuning
        5. Train with gradient accumulation + tail flush
        6. Validate periodically
        7. Early stopping

    Args:
        model: Tesseract model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        cfg: Full configuration.
        device: Target device.
        checkpoint_path: Path for checkpoint save/load.
        resume: Whether to resume from checkpoint.
        finetune_from: Optional checkpoint path to initialize weights for Stage 2.
        tb_writer: Optional torch.utils.tensorboard.SummaryWriter.
        wandb_run: Optional wandb.Run for experiment tracking.
        start_time: Optional launch timestamp (time.time()) from which the
            time budget is counted. BUG FIX (Kaggle pass): on Kaggle the
            dataset scan + worker spawn before train() can eat 10-30 min of
            the 9h session; counting from launch keeps the budget honest.
    """

    # --- Loss function ---
    # Create loss_fn BEFORE optimizer so its log_vars parameters
    # are included in the optimizer. Previously, loss_fn was created after
    # the optimizer, so log_vars were frozen at initialization and
    # uncertainty weighting was completely non-functional.
    loss_fn = UncertaintyWeightedLoss(num_tasks=cfg.loss.num_tasks)
    model.loss_fn = loss_fn  # attach for parameter group detection

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, cfg.train)
    # guard against tiny datasets where
    # len(loader) < accumulate_steps would give steps_per_epoch=0 and crash
    # LinearLR(total_iters=0).
    steps_per_epoch = max(1, len(train_loader) // cfg.train.accumulate_steps)
    scheduler = build_scheduler(optimizer, cfg.train, steps_per_epoch)
    is_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda, init_scale=cfg.train.grad_scaler_init)

    # --- Early stopping ---
    early_stopper = EarlyStopping(cfg.train.patience, cfg.train.min_delta)

    # --- Load checkpoint or Stage 2 Pretrained Weights ---
    start_epoch = 0
    batch_size = cfg.train.batch_size
    best_metric = float("inf")

    if finetune_from:
        src_epoch, src_metric = load_pretrained_weights(finetune_from, model, str(device))
        logger.info(
            f"=== STAGE 2: CAMERA-INVARIANT PINHOLE FINE-TUNING ===\n"
            f"  Initialized weights from: {finetune_from} (source trained to epoch {src_epoch})\n"
            f"  Dynamic Pinhole Crop Aug: {'ACTIVE' if getattr(cfg.data, 'enable_pinhole_crop_aug', True) else 'DISABLED'}\n"
            f"  Min crop scale:           {getattr(cfg.data, 'pinhole_crop_min_scale', 0.55):.2f} (~45° to ~74° FOV)\n"
            f"  Fine-tuning schedule:     {cfg.train.total_epochs} epochs with base_lr={cfg.train.base_lr:.2e}\n"
            f"  Fresh optimizer & Cosine LR schedule active."
        )
        early_stopper.best_metric = float("inf")
    elif resume:
        start_epoch, batch_size, best_metric = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, str(device),
        )
        early_stopper.best_metric = best_metric
        # checkpoints saved at the END of epoch E now
        # store E+1 (the next epoch to run), so a resumed session no longer
        # re-trains an already-completed epoch (~8% of the 8.5h budget wasted
        # per restart on Kaggle). Time-break saves still store E so a
        # partially-trained epoch is redone from scratch (its tail gradients
        # were dropped, so redoing is the correct behaviour).
        if start_epoch >= cfg.train.total_epochs:
            logger.info(
                f"Checkpoint is at epoch {start_epoch} >= total_epochs "
                f"{cfg.train.total_epochs}; nothing left to train."
            )
            # close experiment trackers before the
            # early return (otherwise wandb_run / tb_writer leak an open run).
            if tb_writer is not None:
                tb_writer.close()
            if wandb_run is not None:
                wandb_run.finish()
            return

    # --- DataParallel ---
    # warm up torch.linalg's LAPACK/cuSOLVER
    # runtime ONCE on the main thread BEFORE the DataParallel replica
    # threads exist. Its lazy init is not thread-safe — two replicas
    # hitting any linalg op concurrently on first use crash with
    # "RuntimeError: lazy wrapper should be called at most once"
    # (pytorch/pytorch#90613). The model itself no longer calls
    # torch.linalg anywhere, but this guards future edits and any
    # library code that does.
    if device.type == "cuda":
        try:
            for dev in range(torch.cuda.device_count()):
                eye3 = torch.eye(3, device=f"cuda:{dev}")
                torch.linalg.solve(eye3, eye3)
            torch.cuda.synchronize()
            logger.debug("torch.linalg CUDA warm-up complete")
        except Exception as exc:  # never let the guard itself crash training
            logger.debug(f"torch.linalg warm-up skipped: {exc}")

    if torch.cuda.device_count() > 1:
        logger.info(f"Using nn.DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    # removed the unconditional
    # `torch.backends.cudnn.benchmark = True` — it silently overrode the
    # --deterministic flag set in set_seed(). set_seed() already configures
    # benchmark based on the deterministic choice.

    # --- Curriculum difficulty schedule ---
    # CurriculumSampler was defined but never wired
    # into training. Rebuild the train loader when the desired difficulty
    # changes (only 2-3 rebuilds across the whole schedule).
    curriculum = CurriculumSampler(cfg.curriculum)
    current_difficulty = "all"
    # OOM bookkeeping: consecutive OOMs at the minimum batch size
    oom_strikes = 0

    # --- Training loop ---
    logger.info(cfg.summary())

    # --- Time-based early stopping ---
    TIME_BUDGET = cfg.train.time_limit_hours * 3600  # total seconds
    # count the budget from process launch (dataset
    # indexing + worker spawn included) when start_time is provided.
    train_start = start_time if start_time is not None else time.time()
    # 15-min safety buffer (validation + save can take 5-10 min combined)
    TIME_BUFFER = 900

    # --- SIGTERM handler: Kaggle kills at 9h with SIGTERM ---
    import signal
    _stop_requested = [False]
    def _sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM/SIGINT; will save checkpoint and exit.")
        _stop_requested[0] = True
    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
        signal.signal(signal.SIGINT, _sigterm_handler)
    except (ValueError, RuntimeError):
        pass  # not main thread or signal not supported

    # --- Per-step logger ---
    step_logger = StepLogger(log_interval=50)
    global_step = 0
    # explicitly init val_abs_rel to avoid fragile `dir()` check
    val_abs_rel = float("inf")
    val_metric_abs_rel = float("inf")
    val_rmse = float("inf")
    val_metric_rmse = float("inf")
    # on resume, seed val_abs_rel with the checkpoint's
    # best metric so skipped-validation epochs (odd epochs under val_interval=2)
    # don't log/reuse inf right after a restart.
    # guard against NaN too — NaN != inf is True, so
    # the old check would seed NaN and silently corrupt early-stopping comparisons.
    import math as _math
    if resume and _math.isfinite(best_metric):
        val_abs_rel = best_metric
        val_metric_abs_rel = best_metric

    for epoch in range(start_epoch, cfg.train.total_epochs):
        model.train()

        # --- Curriculum difficulty for this epoch ---
        desired = curriculum.get_difficulty(epoch)
        if desired != current_difficulty:
            base_ds = train_loader.dataset
            underlying = base_ds.dataset if isinstance(base_ds, torch.utils.data.Subset) else base_ds
            diff_name = TartanAirDataset.DIFFICULTY_MAP.get(desired, "Easy")
            indices = [i for i, s in enumerate(underlying.samples)
                       if s.get("difficulty") == diff_name]
            # TartanAir official uses 'Easy' and 'Hard' levels (no 'Medium' folder)
            if not indices and desired == "medium":
                diff_name = "Hard"
                indices = [i for i, s in enumerate(underlying.samples)
                           if s.get("difficulty") == diff_name]
                if indices:
                    logger.info(
                        f"Curriculum: epoch {epoch} requested 'medium' → using 'Hard' "
                        f"(TartanAir standard levels: 'Easy' and 'Hard') ({len(indices)} samples)"
                    )
            if indices:
                train_loader = _rebuild_loader(
                    train_loader, batch_size,
                    dataset=torch.utils.data.Subset(underlying, indices),
                )
                current_difficulty = desired
                if desired != "medium":
                    logger.info(
                        f"Curriculum: epoch {epoch} → difficulty '{desired}' "
                        f"({len(indices)} samples)"
                    )
            elif current_difficulty != "all":
                # Requested difficulty unavailable → fall back to full set
                train_loader = _rebuild_loader(train_loader, batch_size, dataset=underlying)
                current_difficulty = "all"
                logger.info(
                    f"Curriculum: no '{diff_name}' samples available; "
                    f"using full dataset"
                )

        epoch_loss = 0.0
        step_count = 0
        epoch_start = time.time()
        # track whether the epoch fully completed
        # (for-else ran) vs was broken mid-epoch. The time-break save uses this
        # to decide whether to store epoch (partial → redo) or epoch+1 (done → skip).
        epoch_completed = False

        for batch_idx, batch in enumerate(train_loader):
            # in-epoch time check — if Kaggle will kill mid-epoch,
            # break out of both loops cleanly so save_checkpoint runs.
            elapsed = time.time() - train_start
            if elapsed > TIME_BUDGET - TIME_BUFFER or _stop_requested[0]:
                logger.info(
                    f"Time budget / SIGTERM at epoch {epoch} batch {batch_idx} "
                    f"({elapsed/3600:.1f}h elapsed)"
                )
                break

            try:
                # Training step
                _, loss_dict = train_step(
                    model, batch, optimizer, scaler, loss_fn,
                    cfg, step_count % cfg.train.accumulate_steps, device,
                    epoch=epoch,
                )
                epoch_loss += loss_dict["total"]
                step_count += 1
                global_step += 1
                # a successful batch resets the OOM
                # strike counter — previously 3 OOMs spread over many
                # successful batches would abort even though they weren't
                # 3 *consecutive* failures.
                oom_strikes = 0

                # Per-step logging
                step_logger.step(loss_dict, global_step)

                # Optimizer step at accumulation boundary
                if step_count % cfg.train.accumulate_steps == 0:
                    optimizer_step(optimizer, scaler, scheduler, cfg.train)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    new_bs = handle_oom(batch_size)
                    if new_bs < batch_size:
                        # actually REBUILD the loader
                        # at the smaller batch size (the old code only updated
                        # a local variable — the smaller batch never took
                        # effect and OOM repeated indefinitely). Scale
                        # gradient accumulation so the effective batch stays
                        # constant. The new batch size takes effect from the
                        # next epoch (the current iterator is already bound).
                        new_accum = int(max(1, min(64, round(
                            cfg.train.accumulate_steps * batch_size / new_bs))))
                        cfg = replace(cfg, train=replace(
                            cfg.train, batch_size=new_bs, accumulate_steps=new_accum))
                        train_loader = _rebuild_loader(train_loader, new_bs)
                        batch_size = new_bs
                        oom_strikes = 0
                        logger.warning(
                            f"OOM: rebuilt DataLoader with batch_size={new_bs}, "
                            f"accumulate_steps={new_accum} (effective batch "
                            f"{new_bs * new_accum} unchanged; applies next epoch)"
                        )
                    else:
                        # Already at the minimum batch size — count strikes
                        # and abort rather than looping forever.
                        oom_strikes += 1
                        if oom_strikes >= 3:
                            raise RuntimeError(
                                "Repeated OOM at minimum batch size 2 — "
                                "reduce image size or check for leaks."
                            ) from e
                        logger.warning(
                            f"Skipping batch {batch_idx} after OOM "
                            f"(at minimum batch size; strike {oom_strikes}/3)"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()  # clear scaler state
                else:
                    raise
        else:
            # --- TAIL GRADIENT FLUSH  ---
            # Don't drop remaining gradients at epoch end
            epoch_completed = True  # for-else ran → full epoch
            if step_count % cfg.train.accumulate_steps != 0:
                optimizer_step(optimizer, scaler, scheduler, cfg.train)

        # Check time budget / SIGTERM after epoch (or after break)
        elapsed = time.time() - train_start
        if elapsed > TIME_BUDGET - TIME_BUFFER or _stop_requested[0]:
            logger.info(
                f"Time budget exhausted at epoch {epoch} "
                f"({elapsed/3600:.1f}h / {cfg.train.time_limit_hours:.1f}h)"
            )
            # save epoch+1 if the epoch COMPLETED
            # (tail flush ran) so resume doesn't redo a fully-trained epoch.
            # Mid-epoch breaks keep epoch (partial → correctly redone).
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler,
                epoch + 1 if epoch_completed else epoch,
                batch_size, early_stopper.best_metric,
            )
            break

        # --- Validation (every val_interval epochs) ---
        if epoch % cfg.train.val_interval == 0:
            # pass irer_gate so val matches training regime
            current_irer_gate = compute_irer_gate(
                epoch,
                irer_warmup_epochs=cfg.model.irer_warmup_epochs,
                irer_full_epochs=cfg.model.irer_full_epochs,
            )
            if is_cuda:
                torch.cuda.empty_cache()

            val_metrics, per_scene = validate(
                model, val_loader, device,
                num_batches=cfg.data.val_batches,
                num_viz=cfg.data.num_viz_samples,
                viz_dir=str(Path(checkpoint_path).parent / "viz"),
                irer_gate=current_irer_gate,
                use_amp=scaler.is_enabled(),
                eval_min_depth=cfg.data.eval_min_depth,
                eval_max_depth=cfg.data.eval_max_depth,
            )

            if is_cuda:
                torch.cuda.empty_cache()

            # per_scene was computed but discarded.
            # Write the breakdown file .
            if per_scene:
                save_per_scene_breakdown(
                    per_scene,
                    str(Path(checkpoint_path).parent / "per_scene.txt"),
                )
        else:
            # Skip validation, reuse previous metrics
            val_metrics = {
                "abs_rel": val_abs_rel,
                "metric_abs_rel": val_metric_abs_rel,
                "rmse": val_rmse,
                "metric_rmse": val_metric_rmse,
                "d1": val_d1,
                "metric_d1": val_metric_d1,
            }
            per_scene = {}

        val_abs_rel = val_metrics.get("abs_rel", float("inf"))
        val_rmse = val_metrics.get("rmse", float("nan"))
        val_metric_abs_rel = val_metrics.get("metric_abs_rel", float("nan"))
        val_metric_rmse = val_metrics.get("metric_rmse", float("nan"))
        val_d1 = val_metrics.get("d1", float("nan"))
        val_metric_d1 = val_metrics.get("metric_d1", float("nan"))

        # --- Logging ---
        epoch_time = time.time() - epoch_start
        irer_gate_val = compute_irer_gate(
            epoch,
            irer_warmup_epochs=cfg.model.irer_warmup_epochs,
            irer_full_epochs=cfg.model.irer_full_epochs,
        )
        logger.info(
            f"Epoch {epoch}/{cfg.train.total_epochs} | "
            f"Loss: {epoch_loss/max(step_count,1):.4f} | "
            f"Metric AbsRel: {val_metric_abs_rel:.4f} (RMSE: {val_metric_rmse:.2f}m, d1: {val_metric_d1*100:.1f}%) | "
            f"Aligned AbsRel: {val_abs_rel:.4f} (RMSE: {val_rmse:.2f}m, d1: {val_d1*100:.1f}%) | "
            f"IRER gate: {irer_gate_val:.3f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # --- TensorBoard / WandB logging ---
        epoch_loss_avg = epoch_loss / max(step_count, 1)
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", epoch_loss_avg, epoch)
            tb_writer.add_scalar("val/aligned_abs_rel", val_abs_rel, epoch)
            tb_writer.add_scalar("val/metric_abs_rel", val_metric_abs_rel, epoch)
            tb_writer.add_scalar("train/irer_gate", irer_gate_val, epoch)
            for gi, group in enumerate(optimizer.param_groups):
                tb_writer.add_scalar(f"lr/group_{gi}", group["lr"], epoch)
        if wandb_run is not None:
            wandb_run.log({
                "train/loss": epoch_loss_avg,
                "val/aligned_abs_rel": val_abs_rel,
                "val/metric_abs_rel": val_metric_abs_rel,
                "train/irer_gate": irer_gate_val,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            })

        # --- Save checkpoint ---
        # store epoch + 1 (next epoch to run) so resume
        # doesn't re-train this completed epoch.
        save_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler,
            epoch + 1, batch_size, early_stopper.best_metric,
        )

        # --- Early stopping ---
        # only tick the patience counter on epochs
        # that actually validated. With val_interval=2, patience=3, the old
        # code stopped at epoch 2 (only 2 validations!) because skipped odd
        # epochs reused the previous val_abs_rel and ticked the counter.
        if epoch % cfg.train.val_interval == 0:
            if early_stopper.step(val_abs_rel):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(best abs_rel: {early_stopper.best_metric:.4f})"
                )
                break

    # --- Clean up experiment trackers ---
    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    logger.info(
        f"Training complete. Best val abs_rel: {early_stopper.best_metric:.4f}"
    )



def run_multi_fov_eval(
    image_path: str,
    model: nn.Module,
    device: torch.device,
    gt_depth_path: Optional[str] = None,
    output_dir: str = "outputs",
    img_size: int = 224,
) -> Dict[str, Any]:
    """Evaluate depth predictions and 3D bounding geometry across synthetic FOVs.

    Sweeps horizontal field-of-view from 50° (narrow/telephoto) to 100° (wide angle),
    passing exact pinhole camera intrinsics K(FOV) to model(img, K).
    Demonstrates camera-grounded geometric conditioning and metric scale invariance.
    """
    import numpy as np
    from PIL import Image
    import torchvision.transforms.functional as TF
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    raw_img = Image.open(image_path).convert("RGB")
    W_orig, H_orig = raw_img.size

    # Center-crop to square
    min_side = min(H_orig, W_orig)
    top = (H_orig - min_side) // 2
    left = (W_orig - min_side) // 2
    raw_img_cropped = raw_img.crop((left, top, left + min_side, top + min_side))
    img_resized = raw_img_cropped.resize((img_size, img_size), Image.Resampling.BILINEAR)

    img_t = TF.to_tensor(img_resized)  # (3, H, W)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    input_t = ((img_t - mean) / std).unsqueeze(0).to(device)

    # Resolve GT depth if available
    gt_depth = None
    if gt_depth_path is None:
        p = Path(image_path)
        candidate = p.parent / f"{p.stem}_depth.npy"
        if candidate.is_file():
            gt_depth_path = str(candidate)
        else:
            cand2 = p.parent / f"{p.stem.replace('_left', '')}_depth.npy"
            if cand2.is_file():
                gt_depth_path = str(cand2)

    if gt_depth_path and os.path.isfile(gt_depth_path):
        try:
            gt_arr = np.load(gt_depth_path).astype(np.float32)
            gt_t = torch.from_numpy(gt_arr)
            # Center crop and resize GT depth to match image transform
            gt_cropped = gt_t[top:top + min_side, left:left + min_side]
            gt_depth = TF.resize(
                gt_cropped.unsqueeze(0), [img_size, img_size],
                interpolation=TF.InterpolationMode.NEAREST
            ).squeeze(0)
        except Exception as e:
            logger.warning(f"Could not load GT depth from {gt_depth_path}: {e}")

    # Sweep horizontal FOVs
    fovs = [50.0, 60.0, 73.74, 85.0, 90.0, 100.0]
    results = []

    model.eval()
    with torch.no_grad():
        for fov in fovs:
            fov_rad = fov * np.pi / 180.0
            fx = (img_size / 2.0) / np.tan(fov_rad / 2.0)
            fy = fx
            cx = img_size / 2.0
            cy = img_size / 2.0
            K = torch.tensor([[[fx, 0.0, cx],
                               [0.0, fy, cy],
                               [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

            pred_depth, points, _ = model(input_t, K, irer_gate=1.0)
            d = pred_depth.squeeze().cpu()
            pts = points.squeeze().cpu()  # (H, W, 3)

            mean_z = d.mean().item()
            med_z = d.median().item()
            x_span = (pts[..., 0].min().item(), pts[..., 0].max().item())
            z_span = (pts[..., 2].min().item(), pts[..., 2].max().item())

            res = {
                "fov": fov,
                "fx": fx,
                "mean_z": mean_z,
                "med_z": med_z,
                "x_span": x_span,
                "z_span": z_span,
                "depth": d,
                "abs_rel": None,
                "rmse": None,
            }

            if gt_depth is not None:
                v = (gt_depth > 0.1) & (gt_depth < 80.0) & torch.isfinite(gt_depth)
                if v.sum() > 0:
                    diff = (d[v] - gt_depth[v]).abs()
                    abs_rel = (diff / gt_depth[v].clamp(min=1e-3)).mean().item()
                    rmse = torch.sqrt(((d[v] - gt_depth[v]) ** 2).mean()).item()
                    res["abs_rel"] = abs_rel
                    res["rmse"] = rmse

            results.append(res)

    # Print summary table
    print("\n" + "=" * 96)
    print(" TESSERACT MULTI-FOV SYNTHETIC CAMERA EVALUATION")
    print(f" Image:    {image_path}")
    print(f" GT Depth: {gt_depth_path if gt_depth is not None else 'None'}")
    print("=" * 96)
    hdr = f" {'FOV (deg)':<10} | {'Focal fx (px)':<14} | {'Mean Z (m)':<11} | {'Med Z (m)':<10} | {'X-Span (m)':<18} | {'Z-Span (m)':<16}"
    if gt_depth is not None:
        hdr += f" | {'AbsRel':<8} | {'RMSE (m)':<8}"
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        star = "*" if abs(r["fov"] - 73.74) < 0.1 else " "
        row = f" {r['fov']:<4.1f}°{star}    | {r['fx']:<14.2f} | {r['mean_z']:<11.3f} | {r['med_z']:<10.3f} | [{r['x_span'][0]:>6.2f}, {r['x_span'][1]:>6.2f}]  | [{r['z_span'][0]:>5.2f}, {r['z_span'][1]:>5.2f}]"
        if gt_depth is not None:
            ar_str = f"{r['abs_rel']:.4f}" if r['abs_rel'] is not None else "N/A"
            rmse_str = f"{r['rmse']:.3f}" if r['rmse'] is not None else "N/A"
            row += f" | {ar_str:<8} | {rmse_str:<8}"
        print(row)
    print("=" * len(hdr))
    print(" * Canonical TartanAir sensor calibration (fx = 149.33 px at 224x224)\n")

    # Generate multi-panel figure
    os.makedirs(output_dir, exist_ok=True)
    n_fovs = len(fovs)
    n_cols = n_fovs + (2 if gt_depth is not None else 1)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.5), constrained_layout=True)

    # Panel 0: RGB
    rgb_disp = img_t.permute(1, 2, 0).numpy()
    axes[0].imshow(rgb_disp)
    axes[0].set_title(f"Input RGB\n({img_size}×{img_size})", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    col_idx = 1
    # Panel 1: GT Depth (if present)
    if gt_depth is not None:
        vmax = np.percentile(gt_depth.numpy(), 98) if gt_depth.max() > 0 else 10.0
        im_gt = axes[col_idx].imshow(gt_depth.numpy(), cmap="magma", vmin=0, vmax=max(vmax, 1.0))
        axes[col_idx].set_title("Ground Truth Depth\n(Metric)", fontsize=10, fontweight="bold")
        axes[col_idx].axis("off")
        plt.colorbar(im_gt, ax=axes[col_idx], fraction=0.046, pad=0.04, label="Metres")
        col_idx += 1

    # FOV sweep panels
    vmax_pred = max(r["depth"].max().item() for r in results)
    vmax_pred = min(vmax_pred, 30.0)
    for r in results:
        ax = axes[col_idx]
        im = ax.imshow(r["depth"].numpy(), cmap="magma", vmin=0, vmax=vmax_pred)
        title = f"FOV {r['fov']:.0f}° (fx={r['fx']:.0f}px)\nMean: {r['mean_z']:.2f}m"
        if r["abs_rel"] is not None:
            title += f" | AR: {r['abs_rel']:.3f}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        col_idx += 1

    stem = Path(image_path).stem
    out_path = os.path.join(output_dir, f"{stem}_multi_fov_sweep.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Multi-FOV comparison visualization to: {out_path}")

    return {"results": results, "out_plot": out_path}


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    import sys
    import os
    import argparse
    import logging
    import time
    import torch

    parser = argparse.ArgumentParser(description="Tesseract v1 — single-file")
    parser.add_argument("--train", type=str, default=None,
                        help="TartanAir dataset: root dir, Kaggle mount dir, "
                             ".zip archive, or 'auto' (recursive scan of "
                             "/kaggle/input — handles both the classic "
                             "/kaggle/input/<slug> and the namespaced "
                             "/kaggle/input/datasets/<owner>/<slug> mount "
                             "layouts) — launches training")
    parser.add_argument("--count", action="store_true",
                        help="Print parameter count by component")
    parser.add_argument("--smoke", action="store_true",
                        help="Run forward-pass smoke test (requires torch + CUDA)")
    parser.add_argument("--config", action="store_true",
                        help="Print configuration summary")
    parser.add_argument("--test", action="store_true",
                        help="Run unit tests (geometry, losses, checkpoint round-trip)")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="Path to checkpoint to evaluate (loads + runs validation)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Override checkpoint path (default: outputs/checkpoint.pt)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start training from scratch (ignore existing checkpoint)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override total_epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override base_lr")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory for checkpoints, logs, viz (default: outputs)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic algorithms (slower but reproducible)")
    parser.add_argument("--wandb", type=str, default=None,
                        help="WandB project name (optional, enables WandB logging)")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers (default: min(8, CPU count))")
    parser.add_argument("--predict", type=str, default=None,
                        help="Path to an RGB image to run depth prediction on")
    parser.add_argument("--export-ply", type=str, default=None,
                        help="Optional output path for colored 3D point cloud (.ply)")
    parser.add_argument("--device", type=str, default=None,
                        help="Override compute device (cuda, mps, or cpu)")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "trivision-no-irer", "center-ray", "2d-vit"],
                        help="Architecture ablation mode: full, trivision-no-irer, center-ray, 2d-vit")
    parser.add_argument("--split-mode", type=str, default="cross_env",
                        choices=["cross_env", "cross_traj"],
                        help="Dataset split mode: cross_env (zero-shot) or cross_traj (intra-env)")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to checkpoint weights to initialize for Stage 2 fine-tuning")
    parser.add_argument("--finetune-lr", type=float, default=5e-5,
                        help="Learning rate for Stage 2 fine-tuning (default: 5e-5)")
    parser.add_argument("--pinhole-aug", dest="pinhole_aug", action="store_true", default=True,
                        help="Enable dynamic pinhole crop augmentation (default: True)")
    parser.add_argument("--no-pinhole-aug", dest="pinhole_aug", action="store_false",
                        help="Disable dynamic pinhole crop augmentation")
    parser.add_argument("--pinhole-min-scale", type=float, default=0.55,
                        help="Minimum crop scale for pinhole augmentation (default: 0.55)")
    parser.add_argument("--eval-multi-fov", type=str, default=None,
                        help="Run multi-FOV synthetic camera evaluation on an image across FOV sweep (50° to 100°)")
    parser.add_argument("--gt-depth", type=str, default=None,
                        help="Optional path to ground-truth depth (.npy) for --eval-multi-fov or --predict")
    args = parser.parse_args()

    def _get_device(requested: Optional[str] = None) -> torch.device:
        if requested:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_config() -> TesseractConfig:
        from dataclasses import replace as _dc_replace
        model_kwargs = {}
        if args.ablation == "full":
            model_kwargs = {"enable_trivision": True, "enable_irer": True, "pe_mode": "trivision"}
        elif args.ablation == "trivision-no-irer":
            model_kwargs = {"enable_trivision": True, "enable_irer": False, "pe_mode": "trivision"}
        elif args.ablation == "center-ray":
            model_kwargs = {"enable_trivision": True, "enable_irer": True, "pe_mode": "center_ray"}
        elif args.ablation == "2d-vit":
            model_kwargs = {"enable_trivision": False, "enable_irer": False, "pe_mode": "none"}

        data_kwargs = {
            "split_mode": args.split_mode,
            "enable_pinhole_crop_aug": args.pinhole_aug,
            "pinhole_crop_min_scale": args.pinhole_min_scale,
        }
        base_cfg = TesseractConfig()
        return _dc_replace(
            base_cfg,
            model=_dc_replace(base_cfg.model, **model_kwargs),
            data=_dc_replace(base_cfg.data, **data_kwargs),
        )

    # /kaggle/input is READ-ONLY and other
    # working directories (/, /kaggle) may not be writable either — the
    # default output dir must land inside /kaggle/working on Kaggle.
    # Only remap when the user left the default (explicit paths win).
    if args.output_dir == "outputs" and Path("/kaggle/working").is_dir():
        args.output_dir = "/kaggle/working/outputs"

    # Configure logging (BUG FIX: original had no basicConfig → all loggers silent)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "tesseract.log")),
        ],
    )
    # Set seed for reproducibility
    set_seed(args.seed, deterministic=args.deterministic)

    # Resolve the dataset root BEFORE any dataset construction below.
    # Supports "auto" (recursive scan of /kaggle/input — both the classic
    # /kaggle/input/<slug> and namespaced /kaggle/input/datasets/<owner>/
    # <slug> layouts), Kaggle mount dirs (…/dasvo-tartanair-…/tartanair),
    # and .zip archives (lazy zip-backed mode, no extraction). See
    # resolve_dataset_root.
    if args.train:
        args.train = resolve_dataset_root(args.train)

    # Default: print config summary
    if not any([args.train, args.count, args.smoke, args.config, args.test, args.evaluate, args.predict, args.eval_multi_fov]):
        args.config = True

    if args.config:
        cfg = _build_config()
        print(cfg.summary())
        print()
        print("Parameter count by component:")
        model = Tesseract(cfg.model)
        counts = model.count_parameters()
        for name, n in counts.items():
            print(f"  {name:15s}: {n:>12,}  ({n/1e6:.3f}M)")
        print()

    if args.count:
        cfg = _build_config()
        model = Tesseract(cfg.model)
        counts = model.count_parameters()
        print("Parameter count by component:")
        print("-" * 50)
        for name, n in counts.items():
            if name in ("total", "trainable"):
                print(f"  {name.upper():15s}: {n:>12,}  ({n/1e6:.3f}M)")
            else:
                print(f"  {name:15s}: {n:>12,}  ({n/1e6:.3f}M)")
        print("-" * 50)

    if args.smoke:
        print("Running forward-pass smoke test...")
        import torch
        from dataclasses import replace
        set_seed(args.seed, deterministic=args.deterministic)
        device = _get_device(args.device)
        print(f"  device: {device}")
        cfg = _build_config()
        # Disable gradient checkpointing for smoke test (faster, no backward needed)
        # removed dead cfg_dict computation
        new_model_cfg = replace(cfg.model, gradient_checkpointing=False)
        model = Tesseract(new_model_cfg).to(device)
        model.eval()

        B = 2
        image = torch.randn(B, 3, 224, 224, device=device)
        # original K used fx=320 for 224×224 image, giving a 38° FOV
        # (TartanAir is ~90°). Correct post-resize K for 224×224 with 90° FOV
        # is fx=fy~112 (half image width).
        K = torch.tensor([[[112., 0, 112.], [0, 112., 112.], [0, 0, 1.]]] * B,
                          device=device)
        with torch.no_grad():
            depth, points, rays = model(image, K, irer_gate=0.5)
        print(f"  depth:  {tuple(depth.shape)}  range=[{depth.min():.3f}, {depth.max():.3f}]")
        print(f"  points: {tuple(points.shape)}")
        print(f"  rays:   {tuple(rays.shape) if rays is not None else None}")
        # Sanity checks (assertions on key invariants)
        assert depth.shape == (B, 1, 224, 224), f"depth shape wrong: {depth.shape}"
        assert points.shape == (B, 784, 3), f"points shape wrong: {points.shape}"
        if rays is not None:
            assert rays.shape == (B, 784, 3, 3), f"rays shape wrong: {rays.shape}"
            # Rays should be unit length
            ray_norms = rays.norm(dim=-1)
            assert torch.allclose(ray_norms, torch.ones_like(ray_norms), atol=1e-4), \
                f"Rays not unit length: max dev = {(ray_norms - 1).abs().max():.4e}"
        assert torch.isfinite(depth).all(), "depth has non-finite values!"
        print("  SMOKE TEST PASSED ✓")

    if args.test:
        print("Running unit tests...")
        _run_unit_tests()

    if args.evaluate:
        print(f"Evaluating checkpoint: {args.evaluate}")
        import torch
        from torch.utils.data import DataLoader
        device = _get_device(args.device)
        print(f"  device: {device}")
        cfg = _build_config()
        model = Tesseract(cfg.model).to(device)
        # Load checkpoint (just model weights)
        # try the secure weights_only=True load path
        # first (matches load_checkpoint) instead of blindly deserializing.
        try:
            state = torch.load(args.evaluate, map_location=str(device), weights_only=True)
        except Exception:
            state = torch.load(args.evaluate, map_location=str(device), weights_only=False)
        model_state = state.get("tesseract", state)
        # Strip module. prefix if present
        if any(k.startswith("module.") for k in model_state):
            model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
        model.load_state_dict(model_state, strict=False)
        model.eval()
        # Need a val dataset — require user to also pass --train path
        if not args.train:
            print("ERROR: --evaluate requires --train PATH for the val set.")
            sys.exit(1)
        val_ds = TartanAirDataset(
            root=args.train, difficulty="all", split="val",
            img_size=cfg.model.img_size, augment=False, cfg=cfg.data,
        )
        def _collate(batch):
            return {
                "image": torch.stack([b["image"] for b in batch]),
                "depth": torch.stack([b["depth"] for b in batch]),
                "intrinsics": torch.stack([b["intrinsics"] for b in batch]),
                "valid": torch.stack([b["valid"] for b in batch]),
                "scene": [b.get("scene", "") for b in batch],
                "is_flipped": torch.tensor([b.get("is_flipped", False) for b in batch], dtype=torch.bool),
            }
        # respect --num-workers / CPU count here too
        # (was hardcoded 4, which oversubscribes small machines).
        eval_workers = args.num_workers if args.num_workers is not None else \
            min(4, os.cpu_count() or 1)
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                                num_workers=eval_workers, collate_fn=_collate)
        metrics, per_scene = validate(
            model, val_loader, device,
            num_batches=cfg.data.val_batches,
            num_viz=cfg.data.num_viz_samples,
            viz_dir=os.path.join(args.output_dir, "eval_viz"),
            irer_gate=1.0,
            use_amp=(device.type == "cuda"),
            eval_min_depth=cfg.data.eval_min_depth,
            eval_max_depth=cfg.data.eval_max_depth,
        )
        print("Validation metrics:")
        for k, v in metrics.items():
            print(f"  {k:10s}: {v:.4f}")
        if per_scene:
            top5_best = sorted(per_scene.items(), key=lambda x: x[1].get("abs_rel", float("inf")))[:5]
            top5_worst = sorted(per_scene.items(), key=lambda x: -x[1].get("abs_rel", 0))[:5]
            print("\nTop-5 best scenes:")
            for name, m in top5_best:
                print(f"  {name}: abs_rel={m.get('abs_rel', float('nan')):.4f}")
            print("\nTop-5 worst scenes:")
            for name, m in top5_worst:
                print(f"  {name}: abs_rel={m.get('abs_rel', float('nan')):.4f}")
        # --evaluate used to fall through and ALSO
        # launch training (it needs --train as the dataset path). Return here.
        return

    if args.predict:
        print(f"Running prediction on: {args.predict}")
        import torch
        from PIL import Image
        import torchvision.transforms.functional as TF

        device = _get_device(args.device)
        print(f"Compute device: {device}")
        cfg = _build_config()
        model = Tesseract(cfg.model).to(device)

        if args.evaluate:
            ckpt_p = args.evaluate
        elif args.resume:
            ckpt_p = args.resume
        elif args.finetune:
            ckpt_p = args.finetune
        else:
            ckpt_p = os.path.join(args.output_dir, "checkpoint.pt")

        if os.path.isfile(ckpt_p):
            print(f"Loading checkpoint from: {ckpt_p}")
            try:
                state = torch.load(ckpt_p, map_location=str(device), weights_only=True)
            except Exception:
                state = torch.load(ckpt_p, map_location=str(device), weights_only=False)
            model_state = state.get("tesseract", state)
            if any(k.startswith("module.") for k in model_state):
                model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
            model.load_state_dict(model_state, strict=False)
        else:
            print("No checkpoint found; running with initial model weights.")

        model.eval()

        raw_img = Image.open(args.predict).convert("RGB")
        W_orig, H_orig = raw_img.size

        # Center-crop to square
        if H_orig != W_orig:
            min_side = min(H_orig, W_orig)
            top = (H_orig - min_side) // 2
            left = (W_orig - min_side) // 2
            raw_img = raw_img.crop((left, top, left + min_side, top + min_side))

        img_resized = raw_img.resize((cfg.model.img_size, cfg.model.img_size), Image.Resampling.BILINEAR)
        img_t = TF.to_tensor(img_resized)  # (3, 224, 224)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        input_t = ((img_t - mean) / std).unsqueeze(0).to(device)

        # 90-deg FOV default camera intrinsics in 224x224 coordinates
        K = torch.tensor([[[112.0, 0.0, 112.0],
                           [0.0, 112.0, 112.0],
                           [0.0, 0.0, 1.0]]], device=device, dtype=torch.float32)

        with torch.no_grad():
            depth, points, _ = model(input_t, K, irer_gate=1.0)

        depth_2d = depth.squeeze(0).squeeze(0)  # (224, 224)
        print(f"Predicted depth range: [{depth_2d.min():.2f}m, {depth_2d.max():.2f}m]")

        # Save visualization
        stem = Path(args.predict).stem
        viz_out = os.path.join(args.output_dir, f"{stem}_depth_pred.png")
        save_visualization(
            img_t.cpu(),
            depth_2d.cpu(),  # use pred as reference
            depth_2d.cpu(),
            viz_out,
        )
        print(f"Depth visualization saved: {viz_out}")

        ply_out = args.export_ply or os.path.join(args.output_dir, f"{stem}_pointcloud.ply")
        v_count = export_point_cloud_ply(img_t, depth_2d, K[0], ply_out)
        print(f"Colored 3D point cloud ({v_count:,} vertices) saved: {ply_out}")
        return

    if args.eval_multi_fov:
        print(f"Running Multi-FOV Evaluation on: {args.eval_multi_fov}")
        device = _get_device(args.device)
        print(f"Compute device: {device}")
        cfg = _build_config()
        model = Tesseract(cfg.model).to(device)

        if args.evaluate:
            ckpt_p = args.evaluate
        elif args.resume:
            ckpt_p = args.resume
        elif args.finetune:
            ckpt_p = args.finetune
        else:
            ckpt_p = os.path.join(args.output_dir, "checkpoint.pt")

        if os.path.isfile(ckpt_p):
            load_pretrained_weights(ckpt_p, model, str(device))
        else:
            print(f"NOTE: No checkpoint found at {ckpt_p}; evaluating with initial model weights.")

        run_multi_fov_eval(
            image_path=args.eval_multi_fov,
            model=model,
            device=device,
            gt_depth_path=args.gt_depth,
            output_dir=args.output_dir,
            img_size=cfg.model.img_size,
        )
        return

    if args.train:
        print(f"Launching training on dataset: {args.train}")
        import torch
        from torch.utils.data import DataLoader
        from dataclasses import replace as dc_replace

        # budget counts from launch — on Kaggle the
        # dataset scan + DataLoader spawn before train() can consume a
        # nontrivial slice of the 9h session.
        t_launch = time.time()

        device = _get_device(args.device)
        print(f"Compute device: {device}", flush=True)
        cfg = _build_config()

        # Apply CLI overrides
        overrides = {}
        if args.epochs is not None:
            overrides["total_epochs"] = args.epochs
        elif args.finetune:
            overrides["total_epochs"] = 15  # default 15 fine-tuning epochs

        if args.batch_size is not None:
            overrides["batch_size"] = args.batch_size

        if args.lr is not None:
            overrides["base_lr"] = args.lr
        elif args.finetune and args.finetune_lr is not None:
            overrides["base_lr"] = args.finetune_lr

        if overrides:
            cfg = dc_replace(cfg, train=dc_replace(cfg.train, **overrides))

        checkpoint_path = args.resume or os.path.join(args.output_dir, "checkpoint.pt")

        print("Scanning and indexing TartanAir dataset (this takes 1-3 mins on Kaggle network mount)...", flush=True)
        train_ds = TartanAirDataset(
            root=args.train,
            difficulty="all",
            split="train",
            img_size=cfg.model.img_size,
            augment=True,
            cfg=cfg.data,
        )
        val_ds = TartanAirDataset(
            root=args.train,
            difficulty="all",
            split="val",
            img_size=cfg.model.img_size,
            augment=False,
            cfg=cfg.data,
        )
        print(f"Dataset indexed: {len(train_ds):,} train samples, {len(val_ds):,} val samples ✓", flush=True)
        # missing persistent_workers (re-spawns workers each epoch,
        # wasting 5-20s/epoch × 13 = 1-4 min lost) and prefetch_factor.
        # Also use custom collate_fn to keep "scene" as a list (default
        # collate would try to stack str into a tensor).
        def _collate(batch):
            images = torch.stack([b["image"] for b in batch])
            depths = torch.stack([b["depth"] for b in batch])
            Ks = torch.stack([b["intrinsics"] for b in batch])
            valids = torch.stack([b["valid"] for b in batch])
            scenes = [b.get("scene", "") for b in batch]
            is_flipped = torch.tensor([b.get("is_flipped", False) for b in batch], dtype=torch.bool)
            return {"image": images, "depth": depths, "intrinsics": Ks,
                    "valid": valids, "scene": scenes, "is_flipped": is_flipped}

        def _worker_init_fn(worker_id: int):
            import numpy as _np
            import random as _r
            worker_seed = (args.seed + worker_id) % (2 ** 31)
            _np.random.seed(worker_seed)
            _r.seed(worker_seed)

        # num_workers was hardcoded (8 train / 4 val),
        # which oversubscribes small machines and can exceed container memory
        # limits (each worker is a full fork of the parent process). Default
        # to min(8, cpu_count) with a CLI override.
        n_workers = args.num_workers if args.num_workers is not None else \
            min(8, os.cpu_count() or 1)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=True, drop_last=True,
            persistent_workers=n_workers > 0,
            prefetch_factor=4 if n_workers > 0 else None,
            collate_fn=_collate, worker_init_fn=_worker_init_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=max(2, n_workers // 2), pin_memory=True,
            persistent_workers=n_workers > 1,
            prefetch_factor=2 if n_workers > 1 else None,
            collate_fn=_collate,
        )

        model = Tesseract(cfg.model)

        # --- Experiment trackers (BUG FIX review pass 2: --tensorboard and
        #     --wandb flags were parsed but never used) ---
        # catch ALL exceptions, not just ImportError —
        # `wandb init` with no API key raises UsageError (not ImportError) and
        # would otherwise crash the training launch on a Kaggle session that
        # hasn't run `wandb login`.
        tb_writer = None
        if args.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_writer = SummaryWriter(
                    log_dir=os.path.join(args.output_dir, "tensorboard"))
            except Exception as e:
                print(f"NOTE: TensorBoard unavailable ({e}); --tensorboard ignored.")
        wandb_run = None
        if args.wandb:
            try:
                import wandb
                wandb_run = wandb.init(
                    project=args.wandb,
                    config={
                        "total_epochs": cfg.train.total_epochs,
                        "batch_size": cfg.train.batch_size,
                        "effective_batch": cfg.train.effective_batch_size,
                        "base_lr": cfg.train.base_lr,
                        "seed": args.seed,
                    },
                )
            except Exception as e:
                print(f"NOTE: WandB unavailable ({e}); --wandb ignored.")

        print(f"Spawning DataLoader workers and launching training loop ({cfg.train.total_epochs} epochs)...", flush=True)
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            checkpoint_path=checkpoint_path,
            resume=(not args.no_resume) if not args.finetune else False,
            finetune_from=args.finetune,
            tb_writer=tb_writer,
            wandb_run=wandb_run,
            start_time=t_launch,
        )


def _run_unit_tests() -> None:
    """Run a small set of unit tests for geometry/losses/checkpoint.

    Not a full pytest suite — just sanity checks that the major components
    work as expected. Useful for catching regressions after refactoring.
    Tests 5-6 are regression tests for review-pass-2 bug fixes;
    test 8 for the DataParallel linalg crash (Kaggle DP pass).
    """
    import torch
    print("  [1/8] compute_trivision_rays: shapes, unit length, orientation...")
    K = torch.tensor([[[112., 0, 112.], [0, 112., 112.], [0, 0, 1.]]])
    rays = compute_trivision_rays(K, 28, 8, 224)
    assert rays.shape == (1, 784, 3, 3), f"shape: {rays.shape}"
    norms = rays.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), \
        f"rays not unit: max dev {(norms - 1).abs().max():.4e}"
    # Regression (review pass 2): with an ASYMMETRIC K the rays must not be
    # transposed w.r.t. the token grid. Token (row=0, col=27) is the
    # top-right corner → its centre ray must point right of centre (+x).
    K_asym = torch.tensor([[[112., 0, 180.], [0, 112., 60.], [0, 0, 1.]]])
    rays_asym = compute_trivision_rays(K_asym, 28, 8, 224)
    n = 0 * 28 + 27  # row 0, col 27
    assert rays_asym[0, n, 0, 0] > 0.2, (
        f"top-right token ray must point +x, got {rays_asym[0, n, 0, 0]:.4f} "
        f"(u/v transpose regression?)"
    )
    # Symmetric-K invariance: token (0, 27) vs (27, 0) must differ now
    assert not torch.allclose(rays_asym[0, 27, 0], rays_asym[0, 27 * 28, 0]), \
        "asymmetric K produced transpose-symmetric rays"

    print("  [2/8] compute_irer_bias: shapes and finite...")
    q_rays = rays.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # (1, 4, 784, 3, 3)
    k_pts = rays[:, :, 0, :].unsqueeze(1).expand(-1, 4, -1, -1)  # (1, 4, 784, 3)
    alpha = torch.ones(4)
    sigma_sq = torch.ones(4)
    bias = compute_irer_bias(q_rays, k_pts, alpha, sigma_sq, chunk_size=256)
    assert bias.shape == (1, 4, 784, 784), f"bias shape: {bias.shape}"
    assert torch.isfinite(bias).all(), "bias not finite"
    # Note: diagonal isn't exactly 0 because query has 3 rays (centre, TL, BR)
    # but key is just the centre ray. TL and BR rays give small non-zero d².
    # Verify the diagonal is small in absolute terms.
    diag = bias[0, 0].diagonal()
    assert diag.abs().max() < 0.1, f"self-bias too large: {diag.abs().max():.4e}"

    print("  [3/8] SiLog loss: gradient flow...")
    # Use a leaf tensor (multiplication makes it non-leaf; use requires_grad on result)
    pred = (torch.ones(2, 1, 16, 16) * 5.0).requires_grad_(True)
    gt = torch.ones(2, 1, 16, 16) * 4.0
    valid = torch.ones(2, 1, 16, 16).bool()
    loss = silog_loss(pred, gt, valid)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all(), "no finite grad"

    print("  [4/8] handle_oom: even halving...")
    assert handle_oom(8) == 4, f"8→4 expected, got {handle_oom(8)}"
    assert handle_oom(4) == 2, f"4→2 expected, got {handle_oom(4)}"
    assert handle_oom(2) == 2, f"2→2 expected, got {handle_oom(2)}"
    assert handle_oom(3) == 2, f"3→2 expected, got {handle_oom(3)}"

    print("  [5/8] FiLM identity init survives global init pass...")
    # Regression (review pass 2): the global trunc_normal pass in
    # Tesseract.__init__ used to clobber the FiLM identity init (γ≈0 →
    # tokens zeroed) and the DPT disp-head small init.
    cfg = TesseractConfig()
    model = Tesseract(cfg.model)
    w = model.ray_pe.film_mlp[-1].weight
    b = model.ray_pe.film_mlp[-1].bias
    assert w.abs().max().item() == 0.0, \
        f"FiLM output weight must be zero-init, got max {w.abs().max().item():.2e}"
    assert torch.allclose(b[:cfg.model.embed_dim], torch.ones(cfg.model.embed_dim)), \
        "FiLM gamma bias must be 1.0"
    assert torch.allclose(b[cfg.model.embed_dim:], torch.zeros(cfg.model.embed_dim)), \
        "FiLM beta bias must be 0.0"
    dw = model.depth_head.disp_head[-1].weight
    assert dw.abs().max().item() < 1e-3, \
        f"disp_head small init clobbered: max {dw.abs().max().item():.2e}"

    print("  [6/8] se3_exp/se3_log round-trip (unbatched + batched)...")
    # Regression (review pass 2): se3_log crashed on unbatched input and
    # recovered |ω| with a factor-of-2 error.
    import math as _math
    theta = 0.7
    c, s = _math.cos(theta), _math.sin(theta)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = torch.tensor([0.3, -0.2, 0.5])
    twist = se3_log(T)                       # unbatched — used to crash
    assert twist.shape == (6,), f"unbatched twist shape: {twist.shape}"
    assert abs(twist[3:].norm().item() - theta) < 1e-4, \
        f"recovered |ω|={twist[3:].norm().item():.4f}, expected {theta}"
    T_rt = se3_exp(twist)
    assert torch.allclose(T_rt, T, atol=1e-5), "se3 round-trip failed (unbatched)"
    T_b = torch.stack([T, torch.eye(4)])     # batched
    twist_b = se3_log(T_b)
    assert twist_b.shape == (2, 6), f"batched twist shape: {twist_b.shape}"
    assert torch.allclose(se3_exp(twist_b), T_b, atol=1e-5), "se3 round-trip failed (batched)"

    print("  [7/10] Tesseract forward + count_parameters...")
    cfg = TesseractConfig()
    model = Tesseract(cfg.model)
    counts = model.count_parameters()
    assert counts["total"] > 1_000_000, f"total params too small: {counts['total']}"
    print(f"    Total params: {counts['total']:,}")

    print("  [8/10] analytic _inv3x3 == torch.linalg.inv + rays equivalence...")
    # Regression (Kaggle DP pass): torch.linalg.solve/pinv in the model
    # forward crashed under nn.DataParallel replica threads with
    # "lazy wrapper should be called at most once" (pytorch/pytorch#90613)
    # because torch.linalg's lazily-initialised LAPACK/cuSOLVER backend is
    # not thread-safe. _inv3x3 replaces it with pure elementwise math.
    Ms = torch.randn(32, 3, 3, dtype=torch.float64) + 3.0 * torch.eye(3, dtype=torch.float64)
    assert torch.allclose(_inv3x3(Ms), torch.linalg.inv(Ms), rtol=1e-6, atol=1e-8), \
        "_inv3x3 disagrees with torch.linalg.inv"
    # Pinhole K: exact closed form
    K = torch.tensor([[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1.0]])
    K_inv = _inv3x3(K.unsqueeze(0))[0]
    assert torch.allclose(K_inv, torch.tensor([[1 / 320, 0, -1.0],
                                               [0, 1 / 320, -240 / 320],
                                               [0, 0, 1.0]]), atol=1e-9), \
        "pinhole K inverse != closed form"
    # Rays from the analytic path == rays from the old linalg.solve path
    Kb = torch.stack([K] * 4)
    rays = compute_trivision_rays(Kb, grid_size=8, patch_size=8, img_size=64)
    p = torch.stack([torch.tensor([8.0 * j + 4.0, 8.0 * k + 4.0, 1.0])
                     for k in range(8) for j in range(8)])  # (N, 3)
    ref = torch.nn.functional.normalize(
        (K.inverse() @ p.T).T.reshape(1, 64, 3), dim=-1)
    assert torch.allclose(rays[0, :, 0, :], ref[0], atol=1e-5), \
        "analytic rays != direct K^-1 unprojection"
    # Degenerate K stays finite (old pinv-fallback behaviour)
    bad = torch.tensor([[[0.0, 0, 320.0], [0, 0, 240.0], [0, 0, 1.0]]])
    assert torch.isfinite(compute_trivision_rays(bad, 4, 8, 32)).all(), \
        "det=0 K produced non-finite rays"

    print("  [9/10] Log-L1 loss: relative invariance and scale gradient...")
    # Verify that a 10% error at 1m vs 100m produces identical loss
    pred_near = (torch.ones(1, 1, 4, 4) * 1.1).requires_grad_(True)
    gt_near = torch.ones(1, 1, 4, 4) * 1.0
    valid_mask = torch.ones(1, 1, 4, 4).bool()
    loss_near = l1_loss(pred_near, gt_near, valid_mask)

    pred_far = (torch.ones(1, 1, 4, 4) * 110.0).requires_grad_(True)
    gt_far = torch.ones(1, 1, 4, 4) * 100.0
    loss_far = l1_loss(pred_far, gt_far, valid_mask)

    assert torch.allclose(loss_near, loss_far, atol=1e-5), \
        f"Log-L1 must yield identical loss for equal relative error: {loss_near.item():.4f} vs {loss_far.item():.4f}"
    loss_near.backward()
    assert pred_near.grad is not None and torch.isfinite(pred_near.grad).all(), "Log-L1 gradient failed"

    print("  [10/11] Decoupled reassembly LayerNorms...")
    assert len(model.encoder.reassemble_norms) == len(cfg.model.reassemble_layers), \
        "Each reassembly tap must have its own LayerNorm"
    # Ensure they are separate parameter instances
    assert model.encoder.reassemble_norms[0] is not model.encoder.reassemble_norms[1], \
        "Reassembly norms must not share the same LayerNorm instance"

    print("  [11/14] Edge gradient loss: NaN immunity with sky / invalid pixels...")
    pred_edge = (torch.ones(1, 1, 16, 16) * 5.0).requires_grad_(True)
    gt_edge = torch.ones(1, 1, 16, 16) * 5.0
    valid_edge = torch.ones(1, 1, 16, 16).bool()
    # Inject TartanAir sky representation (10000.0, inf, nan) in masked region
    gt_edge[0, 0, :4, :] = 10000.0
    gt_edge[0, 0, 0, 0] = float("nan")
    gt_edge[0, 0, 0, 1] = float("inf")
    valid_edge[0, 0, :4, :] = False

    loss_edge = edge_gradient_loss(pred_edge, gt_edge, valid_edge)
    assert torch.isfinite(loss_edge), f"edge_gradient_loss produced non-finite: {loss_edge}"
    loss_edge.backward()
    assert pred_edge.grad is not None and torch.isfinite(pred_edge.grad).all(), "edge_gradient_loss grad is not finite"

    print("  [12/14] Chiral ray reflection & horizontal flip correspondence...")
    K_test = torch.tensor([[[112., 0, 112.], [0, 112., 112.], [0, 0, 1.]]])
    # Compute unflipped rays vs flipped rays
    rays_unflipped = compute_trivision_rays(K_test, 28, 8, 224, is_flipped=False)
    rays_flipped = compute_trivision_rays(K_test, 28, 8, 224, is_flipped=True)
    # Center ray of middle patch should have identical coordinates
    assert torch.allclose(rays_unflipped[:, :, 0, :], rays_flipped[:, :, 0, :], atol=1e-5)
    # Corner 1 reflects from left (-x relative to center) to right (+x relative to center):
    # c1_flipped_x > c1_unflipped_x for every patch
    assert (rays_flipped[0, :, 1, 0] > rays_unflipped[0, :, 1, 0]).all(), "c1 must reflect to right (+x)"
    # Corner 2 reflects from right (+x relative to center) to left (-x relative to center):
    # c2_flipped_x < c2_unflipped_x for every patch
    assert (rays_flipped[0, :, 2, 0] < rays_unflipped[0, :, 2, 0]).all(), "c2 must reflect to left (-x)"
    # Unit length preserved
    assert torch.allclose(rays_flipped.norm(dim=-1), torch.ones_like(rays_flipped.norm(dim=-1)), atol=1e-4)

    print("  [13/14] Metric supervision scale gradient & dual metric reporting...")
    # Pred depth is 2× scaled compared to GT
    pred_scale = (torch.ones(1, 1, 8, 8) * 10.0).requires_grad_(True)
    gt_scale = torch.ones(1, 1, 8, 8) * 5.0
    valid_scale = torch.ones(1, 1, 8, 8).bool()
    # In relative mode (use_median_norm=True), median normalization cancels scale difference
    loss_rel = silog_loss(pred_scale, gt_scale, valid_scale, use_median_norm=True)
    assert abs(loss_rel.item()) < 1e-4, "Relative SiLog should be zero for uniform scaling"
    # In metric mode (use_median_norm=False), SiLog retains scale penalty via 1 - lambda_w
    loss_metric = silog_loss(pred_scale, gt_scale, valid_scale, use_median_norm=False)
    assert loss_metric.item() > 0.1, f"Metric SiLog must penalize 2x global scale: {loss_metric.item()}"
    loss_metric.backward()
    assert pred_scale.grad is not None and pred_scale.grad.abs().sum() > 0, "Metric loss must provide scale gradient"

    # Dual reporting in compute_eigen_metrics
    eval_metrics = compute_eigen_metrics(pred_scale.detach(), gt_scale, valid_scale, min_depth=0.2, max_depth=80.0)
    assert "metric_abs_rel" in eval_metrics and "abs_rel" in eval_metrics
    # Aligned abs_rel is 0.0 (perfect relative alignment)
    assert eval_metrics["abs_rel"] < 1e-4
    # Metric abs_rel is (10 - 5)/5 = 1.0 (unaligned raw error)
    assert abs(eval_metrics["metric_abs_rel"] - 1.0) < 1e-4

    print("  [14/15] Cross-environment hashing isolation...")
    import hashlib
    all_envs = [
        "abandonedfactory", "abandonedfactory_night", "amusement", "carwelding",
        "endofworld", "gascola", "hospital", "japanesealley", "neighborhood",
        "ocean", "office", "office2", "oldtown", "seasidetown", "seasonsforest",
        "seasonsforest_winter", "soulcity", "westerndesert",
    ]
    val_envs = [e for e in all_envs if (int(hashlib.md5(e.encode("utf-8")).hexdigest(), 16) % 10) >= 8]
    train_envs = [e for e in all_envs if e not in val_envs]
    # Check disjoint sets
    assert set(train_envs).isdisjoint(set(val_envs)), "Train and Val environments must be strictly disjoint"
    assert len(val_envs) > 0 and len(train_envs) > 0, "Both train and val must have environments"
    print(f"    Train environments ({len(train_envs)}): {', '.join(train_envs[:4])}...")
    print(f"    Val environments   ({len(val_envs)}): {', '.join(val_envs)}")

    print("  [15/15] IEEE 754 NaN & Inf immunity across all loss functions...")
    p_test = (torch.ones(2, 1, 32, 32) * 2.0).requires_grad_(True)
    g_test = torch.ones(2, 1, 32, 32) * 4.0
    v_test = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    # Inject NaN and Inf into invalid/sky regions (standard in raw TartanAir)
    g_test[:, :, :10, :10] = float("nan")
    g_test[:, :, 10:20, :10] = float("inf")
    v_test[:, :, :20, :10] = False

    # Median normalize must stay finite
    p_norm_test = median_normalize(p_test, g_test, v_test)
    assert torch.isfinite(p_norm_test).all(), "median_normalize must be finite despite NaN in GT"

    # All 5 loss functions must stay strictly finite and backprop without NaN
    t_l1 = l1_loss(p_test, g_test, v_test)
    assert torch.isfinite(t_l1), f"l1_loss must be finite: {t_l1.item()}"
    t_silog = silog_loss(p_test, g_test, v_test)
    assert torch.isfinite(t_silog), f"silog_loss must be finite: {t_silog.item()}"
    t_edge = edge_gradient_loss(p_test, g_test, v_test)
    assert torch.isfinite(t_edge), f"edge_gradient_loss must be finite: {t_edge.item()}"
    t_smooth = image_aware_smoothness_loss(p_test, g_test, v_test, torch.rand(2, 3, 32, 32))
    assert torch.isfinite(t_smooth), f"smoothness_loss must be finite: {t_smooth.item()}"
    t_planar = planarity_loss(p_test, g_test, v_test)
    assert torch.isfinite(t_planar), f"planarity_loss must be finite: {t_planar.item()}"

    # Gradient flow test: backprop through sum of all losses
    t_total = t_l1 + t_silog + t_edge + t_smooth + t_planar
    t_total.backward()
    assert p_test.grad is not None, "Gradient must be computed"
    assert torch.isfinite(p_test.grad).all(), "Gradient must be finite across all parameters (no NaN/Inf)"
    assert not torch.isnan(p_test.grad).any(), "Gradient must contain no NaN"

    print("  [16/16] TartanAirDataset indexing without cam_left.json (cross_env)...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        # carwelding is in train_envs
        mock_traj = tmp_root / "carwelding" / "Easy" / "P000"
        (mock_traj / "image_left").mkdir(parents=True)
        (mock_traj / "depth_left").mkdir(parents=True)
        from PIL import Image
        import numpy as np
        Image.new("RGB", (64, 48)).save(mock_traj / "image_left" / "000000_left.png")
        np.save(mock_traj / "depth_left" / "000000_left_depth.npy", np.ones((48, 64), dtype=np.float32))

        cfg_test = TesseractConfig()
        ds_test = TartanAirDataset(str(tmp_root), split="train", cfg=cfg_test.data)
        assert len(ds_test) == 1, f"Expected 1 indexed sample, got {len(ds_test)}"
        item = ds_test[0]
        assert "image" in item and "depth" in item and "intrinsics" in item

    print("  ALL 16 TESTS PASSED ✓")


if __name__ == "__main__":
    main()
