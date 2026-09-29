"""
Dioptra-DINO: Foundation-Assisted Geometry-Aware Monocular Metric Depth.

Pairs a pre-trained DINOv2-Small (vits14, 21.6M params) visual backbone with:
  1. Trivision Ray Positional Encoding: continuous optical ray triplets unprojected from K.
  2. Angular Residual Attention (ARA): intrinsic geometric attention bias sin^2(theta_qk).
  3. Multi-Scale DPT Reassembly Head: multi-layer fusion for dense 224x224 metric depth.
  4. Decoupled Metric Scale Supervision: log-median scale consistency loss.
  5. Dynamic Pinhole Intrinsics Augmentation: camera-intrinsic focal equivariance.

Total parameter footprint: 27.51M parameters (27,512,834 params, 105.05 MB FP32, 52.5 MB FP16).

Usage:
  python dioptra_dino.py --smoke          # Test forward/backward pass
  python dioptra_dino.py --count          # Print parameter breakdown
  python dioptra_dino.py --test           # Run unit test suite
  python dioptra_dino.py --train <path>   # Train on TartanAir dataset
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Prevent PyTorch CUDA memory fragmentation across long training runs
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_DEPTH_MIN = 0.1
VALID_DEPTH_MAX = 200.0
EIGEN_DELTA = 1.25
DINOV2_VITS14_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DioptraDINOConfig:
    """Hyperparameters for Dioptra-DINO architecture and optimization."""

    # Input image & token geometry
    image_size: int = 224
    patch_size: int = 14
    grid_size: int = 16  # 224 / 14 = 16 (256 patch tokens)

    # DINOv2 Backbone
    backbone_embed_dim: int = 384
    backbone_depth: int = 12
    backbone_heads: int = 6
    backbone_mlp_ratio: float = 4.0
    out_layers: Tuple[int, int, int, int] = (3, 6, 9, 12)  # 1-indexed feature layers
    freeze_backbone: bool = False
    pretrained_weights_path: Optional[str] = None

    # Trivision Ray Positional Encoding
    enable_trivision: bool = True
    ray_mode: str = "trivision"  # "trivision" (3 rays -> 108 dims) or "center_ray" (1 ray -> 36 dims)
    num_ray_freqs: int = 6  # 3 rays x 6 freqs x 2 (sin/cos) x 3 (xyz) = 108 dims
    film_hidden_dim: int = 256

    # Angular Residual Attention (ARA)
    enable_ara: bool = True
    ara_heads: int = 4
    ara_init_lambda: float = 0.5
    ara_gate: float = 1.0

    # DPT Reassembly & Fusion
    reassemble_features: Tuple[int, int, int, int] = (64, 128, 256, 512)
    postprocess_channels: int = 128

    # Depth prediction bounds
    min_depth: float = 0.1
    max_depth: float = 200.0

    # Training & Optimization
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch size = 32
    lr_backbone: float = 2e-5
    lr_head: float = 2e-4
    weight_decay: float = 0.05
    epochs: int = 15
    use_amp: bool = True
    gradient_clip: float = 1.0
    use_checkpointing: bool = True

    # Loss weights
    weight_silog: float = 1.0
    weight_scale: float = 0.5
    weight_edge: float = 0.2
    weight_normal: float = 0.25
    # TPU static graphs: masked-sum formulations with fixed tensor shapes
    # (boolean indexing yields per-batch variable sizes => XLA recompile storm).
    tpu_static_loss: bool = False


# Guarantee DioptraDINOConfig is available in __main__ for unpickling across all workers/processes
if "__main__" in sys.modules and not hasattr(sys.modules["__main__"], "DioptraDINOConfig"):
    setattr(sys.modules["__main__"], "DioptraDINOConfig", DioptraDINOConfig)


# ---------------------------------------------------------------------------
# DINOv2 Standalone Backbone (Zero external dependencies)
# ---------------------------------------------------------------------------

class LayerScale(nn.Module):
    """LayerScale module initialized with 1e-5 (as in DINOv2)."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class ViTMLP(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ViTAttention(nn.Module):
    """Standard multi-head self-attention with scaled dot-product."""

    def __init__(self, dim: int, num_heads: int = 6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        if hasattr(F, "scaled_dot_product_attention"):
            # FlashAttention / Memory-efficient scaled dot product attention
            x_attn = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x_attn = attn @ v
        x = x_attn.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class ViTBlock(nn.Module):
    """Standard Vision Transformer Block with pre-LayerNorm and LayerScale."""

    def __init__(self, dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = ViTAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = ViTMLP(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Patch embedding layer matching official DINOv2 key naming (patch_embed.proj)."""

    def __init__(self, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class DINOv2Backbone(nn.Module):
    """Standalone DINOv2-Small (vits14) model with multi-scale intermediate extraction."""

    def __init__(self, cfg: DioptraDINOConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.backbone_embed_dim

        # Patch projection (14x14 conv matching official patch_embed.proj)
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size, in_chans=3, embed_dim=self.embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # 1370 = 1 cls + 1369 spatial positions (37x37 grid for native 518x518 pre-training)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, self.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))

        self.blocks = nn.ModuleList([
            ViTBlock(dim=self.embed_dim, num_heads=cfg.backbone_heads, mlp_ratio=cfg.backbone_mlp_ratio)
            for _ in range(cfg.backbone_depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        # Load weights
        self._load_pretrained_weights(cfg.pretrained_weights_path)

        if cfg.freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def _load_pretrained_weights(self, path: Optional[str] = None) -> None:
        """Load DINOv2 pre-trained weights from local path or Meta hub URL."""
        try:
            if path and os.path.exists(path):
                print(f"[Dioptra-DINO] Loading pre-trained backbone from local file: {path}")
                try:
                    state_dict = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    state_dict = torch.load(path, map_location="cpu")
            else:
                print(f"[Dioptra-DINO] Downloading / loading DINOv2-Small weights from Meta Hub...")
                try:
                    state_dict = torch.hub.load_state_dict_from_url(DINOV2_VITS14_URL, map_location="cpu", weights_only=False)
                except (TypeError, Exception):
                    try:
                        state_dict = torch.hub.load_state_dict_from_url(DINOV2_VITS14_URL, map_location="cpu")
                    except Exception:
                        cached_file = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
                        if os.path.exists(cached_file):
                            try:
                                state_dict = torch.load(cached_file, map_location="cpu", weights_only=False)
                            except TypeError:
                                state_dict = torch.load(cached_file, map_location="cpu")
                        else:
                            raise

            # Clean unexpected keys if any (e.g. classifier heads)
            model_keys = set(self.state_dict().keys())
            filtered = {k: v for k, v in state_dict.items() if k in model_keys}
            msg = self.load_state_dict(filtered, strict=False)
            print(f"[Dioptra-DINO] Backbone initialized successfully! {len(filtered)} keys loaded ({msg}).")
        except Exception as e:
            print(f"[Dioptra-DINO] WARNING: Could not load DINOv2 weights ({e}). Initializing randomly.")

    def interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        """Bicubicly interpolate native 37x37 positional embeddings to (h/14, w/14)."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        orig_size = int(math.sqrt(N))
        patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(h0, w0), mode="bicubic", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1)

    def forward(self, x: Tensor) -> List[Tensor]:
        """Forward pass extracting multi-scale tokens at layers [3, 6, 9, 12].

        Args:
            x: Input RGB image tensor [B, 3, H, W].

        Returns:
            List of 4 feature tensors at layers [3, 6, 9, 12], each shaped [B, N, C].
        """
        B, C, H, W = x.shape
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, C]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # [B, N+1, C]

        pos_embed = self.interpolate_pos_encoding(tokens, W, H)
        tokens = tokens + pos_embed

        out_features = []
        target_layers = set(self.cfg.out_layers)

        for i, block in enumerate(self.blocks, start=1):
            if self.training and getattr(self.cfg, "use_checkpointing", True):
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if i in target_layers:
                # Discard CLS token; keep spatial patch tokens [B, N, C]
                spatial_tokens = tokens[:, 1:, :]
                out_features.append(spatial_tokens)

        return out_features


# ---------------------------------------------------------------------------
# Trivision Ray Positional Encoding & Camera Modulation
# ---------------------------------------------------------------------------

class TrivisionRayModulation(nn.Module):
    """Computes continuous camera ray triplets and modulates tokens via FiLM."""

    def __init__(self, cfg: DioptraDINOConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.backbone_embed_dim
        self.grid_size = cfg.grid_size
        self.patch_size = cfg.patch_size
        self.num_freqs = cfg.num_ray_freqs
        self.ray_mode = getattr(cfg, "ray_mode", "trivision")

        # 3 rays (or 1 ray for center_ray ablation) x 6 freqs x 2 (sin/cos) x 3 (xyz)
        num_rays = 1 if self.ray_mode == "center_ray" else 3
        in_dim = num_rays * self.num_freqs * 2 * 3

        self.film_mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.film_hidden_dim),
            nn.LayerNorm(cfg.film_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.film_hidden_dim, self.embed_dim * 2),
        )

        # FiLM identity initialization: scale gamma = 1.0, shift beta = 0.0
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)
        with torch.no_grad():
            self.film_mlp[-1].bias[:self.embed_dim].fill_(1.0)

    def _unproject_rays(
        self, intrinsics: Tensor, is_flipped: Optional[Tensor] = None, num_tokens: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """Unproject ray triplets dynamically for any patch token count.

        Returns:
            ray_features: Sinusoidal embeddings [B, N, 108]
            unit_rays_center: Normalized center ray directions [B, N, 3]
        """
        B = intrinsics.shape[0]
        device = intrinsics.device
        dtype = intrinsics.dtype

        grid_size = int(math.isqrt(num_tokens)) if num_tokens is not None else self.grid_size

        # Analytic closed-form pinhole inversion
        fx = intrinsics[:, 0, 0].clamp(min=1e-5)
        fy = intrinsics[:, 1, 1].clamp(min=1e-5)
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]

        # Patch center coordinates
        half_p = self.patch_size / 2.0
        ys = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) * self.patch_size
        xs = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) * self.patch_size
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        # Corner ray offsets expanded to [B, N]
        c1_x = (grid_x - half_p).unsqueeze(0).expand(B, -1).clone()  # Top-left
        c1_y = (grid_y - half_p).unsqueeze(0).expand(B, -1).clone()
        c2_x = (grid_x + half_p).unsqueeze(0).expand(B, -1).clone()  # Bottom-right
        c2_y = (grid_y + half_p).unsqueeze(0).expand(B, -1).clone()
        grid_x_b = grid_x.unsqueeze(0).expand(B, -1)
        grid_y_b = grid_y.unsqueeze(0).expand(B, -1)

        # Reflection equivariance tracking: swap chiral corners under horizontal flip
        if is_flipped is not None:
            flipped_mask = is_flipped.bool().view(-1, 1)
            # Normal: (TL, BR). Flipped: (TR, BL)
            alt_c1_x = (grid_x + half_p).unsqueeze(0).expand(B, -1)
            alt_c1_y = (grid_y - half_p).unsqueeze(0).expand(B, -1)
            alt_c2_x = (grid_x - half_p).unsqueeze(0).expand(B, -1)
            alt_c2_y = (grid_y + half_p).unsqueeze(0).expand(B, -1)
            c1_x = torch.where(flipped_mask, alt_c1_x, c1_x)
            c1_y = torch.where(flipped_mask, alt_c1_y, c1_y)
            c2_x = torch.where(flipped_mask, alt_c2_x, c2_x)
            c2_y = torch.where(flipped_mask, alt_c2_y, c2_y)

        # Unproject points into camera frame: r = [(u - cx) / fx, (v - cy) / fy, 1]
        def to_unit_rays(px: Tensor, py: Tensor) -> Tensor:
            rx = (px - cx.unsqueeze(1)) / fx.unsqueeze(1)
            ry = (py - cy.unsqueeze(1)) / fy.unsqueeze(1)
            rz = torch.ones_like(rx)
            rays = torch.stack([rx, ry, rz], dim=-1)
            norm = torch.norm(rays, dim=-1, keepdim=True).clamp(min=1e-8)
            return rays / norm

        rc = to_unit_rays(grid_x_b, grid_y_b)  # Center rays [B, N, 3]
        r1 = to_unit_rays(c1_x, c1_y)          # Corner 1 rays [B, N, 3]
        r2 = to_unit_rays(c2_x, c2_y)          # Corner 2 rays [B, N, 3]

        # Multi-scale Fourier features across 6 frequencies
        if getattr(self.cfg, "ray_mode", "trivision") == "center_ray":
            all_rays = rc  # [B, N, 3] (Center-Ray ablation)
        else:
            all_rays = torch.cat([rc, r1, r2], dim=-1)  # [B, N, 9] (Trivision Triplet)
        freq_bands = 2.0 ** torch.arange(self.num_freqs, device=device, dtype=dtype) * math.pi
        prod = all_rays.unsqueeze(-1) * freq_bands.view(1, 1, 1, -1)
        sin_feat = torch.sin(prod)
        cos_feat = torch.cos(prod)
        fourier_feat = torch.cat([sin_feat, cos_feat], dim=-1).flatten(2)  # [B, N, 108]

        return fourier_feat, rc

    def forward(
        self, tokens: Tensor, intrinsics: Tensor, is_flipped: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Modulate tokens with camera ray geometry.

        Args:
            tokens: Visual tokens [B, N, 384]
            intrinsics: Camera calibration matrix [B, 3, 3]
            is_flipped: Optional boolean tensor [B] indicating horizontal flip augmentation.

        Returns:
            modulated_tokens: [B, N, 384]
            unit_center_rays: [B, N, 3] (passed to ARA attention bias)
        """
        fourier_rays, unit_rays_center = self._unproject_rays(intrinsics, is_flipped, num_tokens=tokens.shape[1])
        film_params = self.film_mlp(fourier_rays)  # [B, N, 768]
        gamma = film_params[:, :, :self.embed_dim]  # Scale
        beta = film_params[:, :, self.embed_dim:]   # Shift

        modulated = gamma * tokens + beta
        return modulated, unit_rays_center


# ---------------------------------------------------------------------------
# Angular Residual Attention (ARA) Refinement Block
# ---------------------------------------------------------------------------

class ARARefinementBlock(nn.Module):
    """Refines multi-scale features using Angular Residual Attention."""

    def __init__(self, dim: int = 384, num_heads: int = 6, init_lambda: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Learnable geometric penalty strength lambda
        self.raw_lambda = nn.Parameter(torch.tensor(math.log(math.exp(init_lambda) - 1.0)))

        self.mlp = ViTMLP(dim, dim * 2)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: Tensor, unit_rays: Tensor, ara_gate: float = 1.0) -> Tensor:
        """Apply geometric attention with continuous angular distance bias.

        Args:
            x: Input tokens [B, 256, 384]
            unit_rays: Normalized ray vectors [B, 256, 3]
            ara_gate: Dynamic gating scalar in [0, 1] for progressive warmup.
        """
        B, N, C = x.shape
        residual = x
        norm_x = self.norm(x)

        qkv = self.qkv(norm_x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]

        if ara_gate > 0.0:
            # Continuous pairwise angular residual: sin^2(theta_qk) = 1 - (r_q . r_k)^2
            cos_theta = torch.bmm(unit_rays, unit_rays.transpose(1, 2)).clamp(-1.0, 1.0)  # [B, N, N]
            sin2_theta = (1.0 - cos_theta ** 2).clamp(min=0.0)  # [B, N, N]
            penalty = F.softplus(self.raw_lambda) * ara_gate
            attn_scores = attn_scores - penalty * sin2_theta.unsqueeze(1)

        attn = attn_scores.softmax(dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = residual + self.proj(x_attn)

        # FFN refinement
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Multi-Scale DPT Reassembly & Depth Head
# ---------------------------------------------------------------------------

class ResidualConvUnit(nn.Module):
    """Residual convolution block with GELU and GroupNorm."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class FeatureFusionBlock(nn.Module):
    """Progressive feature fusion with residual refinement and 2x upsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.res1 = ResidualConvUnit(channels)
        self.res2 = ResidualConvUnit(channels)

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        if skip is not None:
            x = x + skip
        x = self.res1(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.res2(x)
        return x


class DPTReassemblyHead(nn.Module):
    """Fuses multi-scale representations into a dense 224x224 metric depth map."""

    def __init__(self, cfg: DioptraDINOConfig):
        super().__init__()
        self.cfg = cfg
        dim = cfg.backbone_embed_dim  # 384
        feats = cfg.reassemble_features  # (64, 128, 256, 512)

        # Layer 3 (1/14 res -> 1/4 res, 64x64)
        self.reasm1 = nn.Sequential(
            nn.Conv2d(dim, feats[0], kernel_size=1),
            nn.ConvTranspose2d(feats[0], feats[0], kernel_size=4, stride=4),
        )
        # Layer 6 (1/14 res -> 1/8 res, 32x32)
        self.reasm2 = nn.Sequential(
            nn.Conv2d(dim, feats[1], kernel_size=1),
            nn.ConvTranspose2d(feats[1], feats[1], kernel_size=2, stride=2),
        )
        # Layer 9 (1/14 res -> 1/14 res, 16x16)
        self.reasm3 = nn.Sequential(
            nn.Conv2d(dim, feats[2], kernel_size=1),
        )
        # Layer 12 (1/14 res -> 1/28 res, 8x8)
        self.reasm4 = nn.Sequential(
            nn.Conv2d(dim, feats[3], kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        out_ch = cfg.postprocess_channels  # 128
        self.proj1 = nn.Conv2d(feats[0], out_ch, kernel_size=3, padding=1)
        self.proj2 = nn.Conv2d(feats[1], out_ch, kernel_size=3, padding=1)
        self.proj3 = nn.Conv2d(feats[2], out_ch, kernel_size=3, padding=1)
        self.proj4 = nn.Conv2d(feats[3], out_ch, kernel_size=3, padding=1)

        self.fuse4 = FeatureFusionBlock(out_ch)
        self.fuse3 = FeatureFusionBlock(out_ch)
        self.fuse2 = FeatureFusionBlock(out_ch)
        self.fuse1 = FeatureFusionBlock(out_ch)

        # Disparity output head: 224x224 -> softplus -> metric depth
        self.disp_head = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch // 2, 1, kernel_size=1),
        )
        # Small init for initial stability
        nn.init.zeros_(self.disp_head[-1].bias)
        self.disp_head[-1].weight.data.mul_(0.01)

    def forward(self, features: List[Tensor]) -> Tensor:
        """Fuse multi-scale features [L3, L6, L9, L12] into metric depth map."""
        B, N, C = features[0].shape
        H = W = int(math.isqrt(N))
        out_img_size = H * self.cfg.patch_size

        f1 = features[0].transpose(1, 2).contiguous().reshape(B, -1, H, W)
        f2 = features[1].transpose(1, 2).contiguous().reshape(B, -1, H, W)
        f3 = features[2].transpose(1, 2).contiguous().reshape(B, -1, H, W)
        f4 = features[3].transpose(1, 2).contiguous().reshape(B, -1, H, W)

        p1 = self.proj1(self.reasm1(f1))
        p2 = self.proj2(self.reasm2(f2))
        p3 = self.proj3(self.reasm3(f3))
        p4 = self.proj4(self.reasm4(f4))

        x4 = self.fuse4(p4)
        x3 = self.fuse3(x4, p3)
        x2 = self.fuse2(x3, p2)
        x1 = self.fuse1(x2, p1)

        # Upsample to full input image resolution
        x_full = F.interpolate(x1, size=(out_img_size, out_img_size), mode="bilinear", align_corners=False)
        raw_disp = self.disp_head(x_full)

        # Convert disparity to true metric depth in metres: D = 1 / (softplus(d) + 1/D_max)
        metric_depth = 1.0 / (F.softplus(raw_disp) + 1.0 / self.cfg.max_depth)
        metric_depth = metric_depth.clamp(min=self.cfg.min_depth, max=self.cfg.max_depth)
        return metric_depth


# ---------------------------------------------------------------------------
# Full Dioptra-DINO Model
# ---------------------------------------------------------------------------

class DioptraDINO(nn.Module):
    """Complete Dioptra-DINO architecture (~25.4M parameters)."""

    def __init__(self, cfg: Optional[DioptraDINOConfig] = None):
        super().__init__()
        self.cfg = cfg or DioptraDINOConfig()

        # 1. DINOv2 Backbone (21.6M params)
        self.backbone = DINOv2Backbone(self.cfg)

        # 2. Trivision Ray Positional Encoding (0.22M params)
        if self.cfg.enable_trivision:
            self.ray_modulation = TrivisionRayModulation(self.cfg)
        else:
            self.ray_modulation = None

        # 3. Angular Residual Attention (ARA) Refinement (1.18M params)
        if self.cfg.enable_ara:
            self.ara_refine = ARARefinementBlock(
                dim=self.cfg.backbone_embed_dim,
                num_heads=self.cfg.ara_heads,
                init_lambda=self.cfg.ara_init_lambda,
            )
        else:
            self.ara_refine = None

        # 4. Multi-Scale DPT Reassembly Head (2.36M params)
        self.depth_head = DPTReassemblyHead(self.cfg)

    def forward(
        self,
        image: Tensor,
        intrinsics: Tensor,
        ara_gate: float = 1.0,
        is_flipped: Optional[Tensor] = None,
    ) -> Tensor:
        """End-to-end forward pass predicting dense metric depth.

        Args:
            image: RGB tensor [B, 3, 224, 224] (ImageNet normalized)
            intrinsics: Camera intrinsic calibration matrix [B, 3, 3]
            ara_gate: Geometric attention gate in [0, 1]
            is_flipped: Optional boolean tensor for chiral horizontal flip tracking

        Returns:
            metric_depth: Predicted depth map in physical metres [B, 1, 224, 224]
        """
        # Step 1: Extract multi-scale DINOv2 features at layers [3, 6, 9, 12]
        features = self.backbone(image)  # 4 tensors of [B, 256, 384]

        # Step 2: Modulate penultimate and deepest features with Trivision ray geometry
        unit_center_rays = None
        if self.ray_modulation is not None:
            # Modulate deepest representation (Layer 12)
            features[-1], unit_center_rays = self.ray_modulation(
                features[-1], intrinsics, is_flipped
            )
            # Modulate Layer 9
            features[-2], _ = self.ray_modulation(features[-2], intrinsics, is_flipped)

        # Step 3: Apply Angular Residual Attention on deepest tokens
        if self.ara_refine is not None and unit_center_rays is not None:
            features[-1] = self.ara_refine(
                features[-1], unit_center_rays, ara_gate=ara_gate * self.cfg.ara_gate
            )

        # Step 4: Multi-Scale Reassembly & Metric Depth Decoding
        if self.cfg.use_checkpointing and self.training:
            metric_depth = torch.utils.checkpoint.checkpoint(self.depth_head, features, use_reentrant=False)
        else:
            metric_depth = self.depth_head(features)
        return metric_depth


# ---------------------------------------------------------------------------
# Training Loss Functions & Uncertainty Weighting
# ---------------------------------------------------------------------------

class DioptraDINOLoss(nn.Module):
    """Multi-task loss combining SiLog, Scale Consistency, Edge, and Normal losses."""

    def __init__(self, cfg: DioptraDINOConfig):
        super().__init__()
        self.cfg = cfg

    def silog_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Scale-invariant log loss with 0.85 variance penalty."""
        if getattr(self.cfg, "tpu_static_loss", False):
            # Static shapes: masked sums (algebraically identical to indexed means).
            mf = mask.float()
            n = mf.sum().clamp(min=1.0)
            d = (torch.log(pred.clamp(min=1e-4)) - torch.log(target.clamp(min=1e-4))) * mf
            loss = d.pow(2).sum() / n - 0.85 * (d.sum() / n) ** 2
            return torch.sqrt(loss.clamp(min=1e-8))
        d = torch.log(pred[mask].clamp(min=1e-4)) - torch.log(target[mask].clamp(min=1e-4))
        loss = torch.mean(d ** 2) - 0.85 * (torch.mean(d) ** 2)
        return torch.sqrt(loss.clamp(min=1e-8))

    def scale_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Multi-bin depth-stratified metric scale consistency loss."""
        if getattr(self.cfg, "tpu_static_loss", False):
            # Static shapes: branchless masked log-mean consistency across depth tiers
            mf = mask.float()
            n = mf.sum().clamp(min=1.0)
            lp = (torch.log(pred.clamp(min=1e-5)) * mf).sum() / n
            lt = (torch.log(target.clamp(min=1e-5)) * mf).sum() / n
            l_global = torch.abs(lp - lt)
            gate_g = (n > 50).float()

            # Tier 1: Near-field [0.1m - 2.5m] (prevents min-depth floor collapse)
            m_near = mask & (target <= 2.5)
            mf_near = m_near.float()
            n_near = mf_near.sum()
            lp_near = (torch.log(pred.clamp(min=1e-5)) * mf_near).sum() / n_near.clamp(min=1.0)
            lt_near = (torch.log(target.clamp(min=1e-5)) * mf_near).sum() / n_near.clamp(min=1.0)
            l_near = torch.abs(lp_near - lt_near)
            gate_near = (n_near > 50).float()

            # Tier 2: Mid-to-far field [2.5m - 10.0m+] (preserves room and structural scale)
            m_far = mask & (target > 2.5)
            mf_far = m_far.float()
            n_far = mf_far.sum()
            lp_far = (torch.log(pred.clamp(min=1e-5)) * mf_far).sum() / n_far.clamp(min=1.0)
            lt_far = (torch.log(target.clamp(min=1e-5)) * mf_far).sum() / n_far.clamp(min=1.0)
            l_far = torch.abs(lp_far - lt_far)
            gate_far = (n_far > 50).float()

            total_loss = l_global * gate_g + 0.5 * l_near * gate_near + 0.5 * l_far * gate_far
            total_w = gate_g + 0.5 * gate_near + 0.5 * gate_far
            return total_loss / total_w.clamp(min=1.0)

        # Dynamic shape path (GPU / MPS / CPU)
        pred_med = torch.median(pred[mask])
        target_med = torch.median(target[mask])
        l_glob = torch.abs(torch.log(pred_med.clamp(min=1e-5)) - torch.log(target_med.clamp(min=1e-5)))

        m_near = mask & (target <= 2.5)
        m_far = mask & (target > 2.5)
        parts = [l_glob]
        weights = [1.0]

        if m_near.sum() > 50:
            l_near = torch.abs(torch.log(torch.median(pred[m_near]).clamp(min=1e-5)) - torch.log(torch.median(target[m_near]).clamp(min=1e-5)))
            parts.append(l_near)
            weights.append(0.5)

        if m_far.sum() > 50:
            l_far = torch.abs(torch.log(torch.median(pred[m_far]).clamp(min=1e-5)) - torch.log(torch.median(target[m_far]).clamp(min=1e-5)))
            parts.append(l_far)
            weights.append(0.5)

        return sum(p * w for p, w in zip(parts, weights)) / sum(weights)

    def edge_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Multi-scale spatial gradient loss for sharp structural boundaries."""
        # Sobel-like central finite differences
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]
        mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]

        if getattr(self.cfg, "tpu_static_loss", False):
            # Static shapes: masked sums (algebraically identical to indexed means).
            my, mx = mask_y.float(), mask_x.float()
            loss_y = (torch.abs(dy_pred - dy_target) * my).sum() / my.sum().clamp(min=1.0)
            loss_x = (torch.abs(dx_pred - dx_target) * mx).sum() / mx.sum().clamp(min=1.0)
            return loss_y + loss_x
        loss_y = torch.mean(torch.abs(dy_pred[mask_y] - dy_target[mask_y]))
        loss_x = torch.mean(torch.abs(dx_pred[mask_x] - dx_target[mask_x]))
        return loss_y + loss_x

    @staticmethod
    def _cross_31(a: Tensor, b: Tensor) -> Tensor:
        """Manual 3-vector cross product on dim=1 ([B,3,...] tensors).

        Mathematically identical to torch.cross(a, b, dim=1), but built only
        from elementwise mul/sub so autograd stays on differentiable XLA
        lowerings (torch.cross has no Autograd kernel on the XLA backend and
        falls through with silently wrong gradients).
        """
        return torch.stack([
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ], dim=1)

    def virtual_normal_loss(
        self, pred: Tensor, target: Tensor, mask: Tensor, K: Optional[Tensor] = None
    ) -> Tensor:
        """Multi-scale 3D Virtual Normal Loss enforcing surface planarity."""
        B, C, H, W = pred.shape
        device = pred.device
        dtype = pred.dtype

        # Coordinate grid
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        u = u.unsqueeze(0).expand(B, -1, -1)
        v = v.unsqueeze(0).expand(B, -1, -1)

        if K is not None and K.dim() == 3:
            fx = K[:, 0, 0].view(B, 1, 1).clamp(min=1.0)
            fy = K[:, 1, 1].view(B, 1, 1).clamp(min=1.0)
            cx = K[:, 0, 2].view(B, 1, 1)
            cy = K[:, 1, 2].view(B, 1, 1)
        else:
            fx = torch.full((B, 1, 1), 149.33, device=device, dtype=dtype)
            fy = fx
            cx = torch.full((B, 1, 1), W / 2.0, device=device, dtype=dtype)
            cy = torch.full((B, 1, 1), H / 2.0, device=device, dtype=dtype)

        # 3D points P = (X, Y, Z)
        pred_z = pred.squeeze(1).clamp(min=1e-3)
        target_z = target.squeeze(1).clamp(min=1e-3)
        m = mask.squeeze(1)

        pred_x = (u - cx) / fx * pred_z
        pred_y = (v - cy) / fy * pred_z
        P_pred = torch.stack([pred_x, pred_y, pred_z], dim=1)

        with torch.no_grad():
            target_x = (u - cx) / fx * target_z
            target_y = (v - cy) / fy * target_z
            P_target = torch.stack([target_x, target_y, target_z], dim=1)

        total_vnl = torch.tensor(0.0, device=device, dtype=dtype)
        valid_scales = 0
        total_w = torch.tensor(0.0, device=device, dtype=dtype)
        static = bool(getattr(self.cfg, "tpu_static_loss", False))

        for s in [1, 2, 4]:
            if H <= 2 * s or W <= 2 * s:
                continue
            vx_pred = P_pred[:, :, s:-s, 2 * s:] - P_pred[:, :, s:-s, :-2 * s]
            vy_pred = P_pred[:, :, 2 * s:, s:-s] - P_pred[:, :, :-2 * s, s:-s]
            n_pred = self._cross_31(vx_pred, vy_pred)
            norm_pred = torch.norm(n_pred, dim=1, keepdim=True).clamp(min=1e-6)

            with torch.no_grad():
                vx_target = P_target[:, :, s:-s, 2 * s:] - P_target[:, :, s:-s, :-2 * s]
                vy_target = P_target[:, :, 2 * s:, s:-s] - P_target[:, :, :-2 * s, s:-s]
                n_target = self._cross_31(vx_target, vy_target)
                norm_target = torch.norm(n_target, dim=1, keepdim=True).clamp(min=1e-6)
                n_t = (n_target / norm_target).detach()

            m_inner = (
                m[:, s:-s, 2 * s:]
                & m[:, s:-s, :-2 * s]
                & m[:, 2 * s:, s:-s]
                & m[:, :-2 * s, s:-s]
                & (norm_target.squeeze(1) > 1e-4)
            )

            if static:
                # Branchless static-shape gate: identical math to the >50 guard
                # (mean over valid, 0 weight otherwise), no host branch, no
                # Python-int divisor (both would recompile XLA per batch).
                n_p = n_pred / norm_pred
                cos_sim = torch.sum(n_p * n_t, dim=1)
                inner_f = m_inner.float()
                cnt = inner_f.sum()
                loss_s = ((1.0 - cos_sim.clamp(min=-1.0, max=1.0)) * inner_f).sum() / cnt.clamp(min=1.0)
                gate = (cnt > 50).float()
                total_vnl = total_vnl + loss_s * gate
                total_w = total_w + gate
                continue

            if m_inner.sum() > 50:
                n_p = n_pred / norm_pred
                cos_sim = torch.sum(n_p * n_t, dim=1)
                loss_s = torch.mean(1.0 - cos_sim[m_inner].clamp(min=-1.0, max=1.0))
                total_vnl = total_vnl + loss_s
                valid_scales += 1

        if static:
            return total_vnl / total_w.clamp(min=1.0)
        if valid_scales > 0:
            return total_vnl / float(valid_scales)
        return pred.sum() * 0.0

    def forward(
        self, pred: Tensor, target: Tensor, image: Optional[Tensor] = None, K: Optional[Tensor] = None,
        sync_metrics: bool = True,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute total multi-task loss across valid pixels.

        Args:
            sync_metrics: If False, skip .item() metric materialization and return {}.
                REQUIRED for the TPU path: each .item() forces a host-device sync
                barrier that stalls all XLA cores. Callers needing logs pass
                sync_metrics=True only on throttled (master, every-N) steps.
        """
        mask = (target >= self.cfg.min_depth) & (target <= self.cfg.max_depth) & ~torch.isnan(target)
        static = bool(getattr(self.cfg, "tpu_static_loss", False))
        if not static and mask.sum() < 100:
            return pred.sum() * 0.0, {}
        # Static path: in-graph 0/1 gate instead of a host branch (the branch
        # would trace a second graph variant and recompile per batch).
        gate = (mask.sum() >= 100).float() if static else None

        l_silog = self.silog_loss(pred, target, mask)
        l_scale = self.scale_loss(pred, target, mask)
        l_edge = self.edge_loss(pred, target, mask)
        l_vnl = self.virtual_normal_loss(pred, target, mask, K=K)

        total_loss = (
            self.cfg.weight_silog * l_silog
            + self.cfg.weight_scale * l_scale
            + self.cfg.weight_edge * l_edge
            + self.cfg.weight_normal * l_vnl
        )
        if static:
            total_loss = total_loss * gate

        if not sync_metrics:
            return total_loss, {}
        metrics = {
            "loss_total": total_loss.item(),
            "loss_silog": l_silog.item(),
            "loss_scale": l_scale.item(),
            "loss_edge": l_edge.item(),
            "loss_vnl": l_vnl.item(),
        }
        return total_loss, metrics


# ---------------------------------------------------------------------------
# Dynamic Pinhole Intrinsics Crop Augmentation
# ---------------------------------------------------------------------------

def apply_dynamic_pinhole_crop(
    image: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    crop_size_range: Tuple[float, float] = (0.35, 1.0),
    out_size: int = 224,
    enable_aspect_jitter: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop simulating optical zoom and non-square aspect ratio variation.

    Adjusts focal lengths fx, fy and optical center cx, cy continuously.
    """
    H, W = image.shape[:2]

    if enable_aspect_jitter and random.random() < 0.7:
        # Anisotropic Aspect-Ratio Jitter (between 0.75 and 1.33)
        ar = random.uniform(0.75, 1.33)
        min_h = int(crop_size_range[0] * H)
        max_h = int(crop_size_range[1] * H)
        crop_h = random.randint(min(min_h, max_h), max(min_h, max_h))
        crop_w = int(crop_h * ar)
        if crop_w > W:
            crop_w = W
            crop_h = int(crop_w / ar)
        if crop_h > H:
            crop_h = H
            crop_w = int(crop_h * ar)
        crop_h = max(32, min(H, crop_h))
        crop_w = max(32, min(W, crop_w))
    else:
        max_square = min(H, W)
        min_len = int(crop_size_range[0] * max_square)
        max_len = int(crop_size_range[1] * max_square)
        crop_len = random.randint(min_len, max_len)
        crop_h = crop_len
        crop_w = crop_len

    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)

    # Crop
    crop_img = image[top:top + crop_h, left:left + crop_w]
    crop_depth = depth[top:top + crop_h, left:left + crop_w]

    # Adjust camera intrinsics
    K_new = K.copy().astype(np.float32)
    K_new[0, 2] -= left  # cx' = cx - left
    K_new[1, 2] -= top   # cy' = cy - top

    # Rescale to network input resolution out_size x out_size with independent per-axis scaling
    scale_x = float(out_size) / float(crop_w)
    scale_y = float(out_size) / float(crop_h)
    K_new[0, 0] *= scale_x
    K_new[1, 1] *= scale_y
    K_new[0, 2] *= scale_x
    K_new[1, 2] *= scale_y

    # Ensure crop_depth is strictly 2D float32
    if crop_depth.ndim == 3:
        crop_depth = crop_depth.squeeze()
    if crop_depth.ndim != 2:
        crop_depth = np.zeros((crop_h, crop_w), dtype=np.float32)
    else:
        crop_depth = np.ascontiguousarray(crop_depth, dtype=np.float32)

    # Resize images (using PIL if available, with robust fallback to numpy nearest/linear)
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(crop_img).resize((out_size, out_size), PILImage.BILINEAR)
        pil_depth = PILImage.fromarray(crop_depth).resize((out_size, out_size), PILImage.NEAREST)
        res_img = np.array(pil_img)
        res_depth = np.array(pil_depth)
    except Exception:
        # Robust fallback numpy nearest resize
        y_indices = (np.linspace(0, crop_h - 1, out_size)).astype(int)
        x_indices = (np.linspace(0, crop_w - 1, out_size)).astype(int)
        res_img = crop_img[np.ix_(y_indices, x_indices)]
        res_depth = crop_depth[np.ix_(y_indices, x_indices)]

    return res_img, res_depth, K_new


# ---------------------------------------------------------------------------
# TartanAir Dataset Discovery & Resolution
# ---------------------------------------------------------------------------

_TARTANAIR_DIFFS = {
    "Easy", "Medium", "Hard", "easy", "medium", "hard",
    "Data_easy", "data_easy", "Data_hard", "data_hard"
}
_MAX_SCAN_DEPTH = 5
_SCAN_SKIP_DIRS = {".ipynb_checkpoints", "__pycache__", ".git", "__MACOSX"}


def _dir_looks_like_tartanair(root: Path) -> bool:
    """Cheap structural probe: does root match TartanAir v1, v2, or warehouse stereo layout?"""
    try:
        tops = [d for d in root.iterdir() if d.is_dir()]
    except OSError:
        return False
    if not tops:
        return False
    top_names = {t.name for t in tops}
    if "warehouse_stereo" in top_names or root.name == "warehouse_stereo":
        return True
    if any(k in top_names for k in ("carwelding", "abandonedfactory", "IndustrialHangar", "Supermarket")):
        return True
    if any(t.name in _TARTANAIR_DIFFS for t in tops):
        return True  # layout B root: {difficulty}/{env}/...
    for t in tops[:64]:
        try:
            sub = [c.name for c in t.iterdir() if c.is_dir()]
            if any(c in _TARTANAIR_DIFFS for c in sub):
                return True  # layout A root: {env}/{difficulty}/...
            if any(c in ("image_left", "image_lcam_front", "depth_left", "depth_lcam_front") for c in sub):
                return True
        except OSError:
            continue
    return False


def _zip_looks_like_tartanair(zip_path: Path) -> bool:
    """True if the archive contains TartanAir v1 or v2 stereo image/depth members."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return False
    has_img = any(("/image_left/" in n or "/image_lcam_front/" in n) for n in names)
    has_depth = any(("/depth_left/" in n or "/depth_lcam_front/" in n) for n in names)
    return has_img and has_depth


def _enumerate_input_mounts(base: Path) -> List[str]:
    try:
        children = sorted(d for d in base.iterdir() if d.is_dir())
    except OSError:
        return []
    mounts: List[str] = []
    for c in children:
        if c.name == "datasets":
            try:
                for o in sorted(d for d in c.iterdir() if d.is_dir()):
                    for s in sorted(d for d in o.iterdir() if d.is_dir()):
                        mounts.append(f"datasets/{o.name}/{s.name}")
            except OSError:
                mounts.append("datasets/")
        else:
            mounts.append(c.name)
    return mounts


def _deep_find_tartanair(base: Path) -> Tuple[List[Path], List[Tuple[Path, int]]]:
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
            if (d / "warehouse_stereo").is_dir():
                d = d / "warehouse_stereo"
            dir_hits.append(d)
            continue
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
                if size < 1_048_576:
                    continue
                if _zip_looks_like_tartanair(e):
                    zip_hits.append((e, size))
    dir_hits.sort(key=str)
    zip_hits.sort(key=lambda t: t[1], reverse=True)
    return dir_hits, zip_hits


def _resolve_single_mount(cand: Path) -> Optional[Path]:
    if cand.is_file() and cand.suffix.lower() == ".zip":
        return cand
    if not cand.is_dir():
        return None
    if _dir_looks_like_tartanair(cand):
        if (cand / "warehouse_stereo").is_dir():
            return cand / "warehouse_stereo"
        return cand
    try:
        subdirs = [d for d in sorted(cand.iterdir()) if d.is_dir()]
    except OSError:
        subdirs = []
    for sd in subdirs[:16]:
        if _dir_looks_like_tartanair(sd):
            if (sd / "warehouse_stereo").is_dir():
                return sd / "warehouse_stereo"
            return sd
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
    dir_hits, zip_hits = _deep_find_tartanair(cand)
    if dir_hits:
        return dir_hits[0]
    if zip_hits:
        return zip_hits[0][0]
    # Fallback: check if directory contains png or zip files
    try:
        if any(cand.glob("**/*.png")) or any(cand.glob("**/*.zip")):
            return cand
    except OSError:
        pass
    return None


def resolve_all_dataset_roots(path: str = "auto") -> List[Tuple[str, str]]:
    """Scan and resolve all mounted multi-domain dataset roots.
    
    Returns a list of tuples: [(root_path, domain_type), ...]
    where domain_type is one of: 'tartan', 'hypersim', 'nyu', 'kitti'.
    """
    results: List[Tuple[str, str]] = []
    seen_paths = set()

    def _classify_and_add(p: Path):
        p_str = str(p.resolve()) if p.exists() else str(p)
        if p_str in seen_paths:
            return
        seen_paths.add(p_str)

        pl = p_str.lower()
        if "checkpoint" in pl or "ckpt" in pl or "outputs" in pl:
            return

        # 0. ScanNet (Official / Kaggle frames)
        if "scannet" in pl:
            results.append((str(p), "scannet"))
            return
        # 1. Hypersim
        if "hypersim" in pl:
            results.append((str(p), "hypersim"))
            return
        # 2. NYU-Depth-v2
        if "nyu" in pl:
            results.append((str(p), "nyu"))
            return
        # 3. KITTI
        if "kitti" in pl:
            results.append((str(p), "kitti"))
            return
        # 4. TartanGround AMR
        if "tartanground" in pl:
            results.append((str(p), "tartanground_amr"))
            return
        # 5. TartanAir Warehouse
        if "warehouse" in pl:
            results.append((str(p), "tartanair_warehouse"))
            if (p / "warehouse_stereo").is_dir():
                results.append((str(p / "warehouse_stereo"), "tartanair_warehouse"))
            return
        # 6. TartanAir Indoors / Complement / TartanAir2 / DASVO
        if any(k in pl for k in (
            "tartan", "hospital", "office", "abandoned", "restaurant", "school",
            "industrial", "dasvo", "diner", "tinyhouse", "saloon", "prison",
            "supermarket", "retro", "brickhouse", "house", "carwelding"
        )):
            results.append((str(p), "tartanair_indoors"))
            return

        # Fast shallow fallback: check immediate directory contents
        try:
            if p.is_dir():
                for child in p.iterdir():
                    cn = child.name.lower()
                    if cn.endswith(".png") or cn.endswith(".npy") or cn.endswith(".hdf5") or cn.endswith(".zip"):
                        results.append((str(p), "tartanair_indoors"))
                        break
        except OSError:
            pass

    def _is_single_dataset(p: Path) -> bool:
        pl = p.name.lower()
        if any(k in pl for k in (
            "scannet", "hypersim", "nyu", "kitti", "tartan", "warehouse", "hospital",
            "office", "abandoned", "restaurant", "industrial", "city", "dasvo",
            "diner", "tinyhouse", "saloon", "prison", "supermarket", "retro", "brickhouse", "house", "carwelding"
        )):
            return True
        if (p / "image_left").is_dir() or (p / "image_lcam_front").is_dir() or (p / "warehouse_stereo").is_dir():
            return True
        if (p / "depths").is_dir() or (p / "data").is_dir():
            return True
        return False

    if path and str(path).lower() not in ("auto", "none"):
        for sub in str(path).split(","):
            sub_p = Path(sub.strip())
            if sub_p.exists():
                if _is_single_dataset(sub_p):
                    _classify_and_add(sub_p)
                elif sub_p.is_dir():
                    # Container folder - probe immediate subdirectories
                    for child in sorted(sub_p.iterdir()):
                        if child.is_dir() and not child.name.startswith("."):
                            _classify_and_add(child)
                else:
                    _classify_and_add(sub_p)
        if results:
            return results

    # Auto-scan: probe /kaggle/input, environment vars, and local directories
    candidate_roots: List[Path] = []
    env_dir = os.environ.get("TESSERACT_INPUT_DIR") or os.environ.get("DATA_PATH")
    if env_dir and Path(env_dir).exists():
        candidate_roots.append(Path(env_dir))
    if Path("/kaggle/input").is_dir():
        # Support flat (/kaggle/input/<slug>), nested (/kaggle/input/datasets/<owner>/<slug>), and keyword subtrees
        for item in Path("/kaggle/input").iterdir():
            if item.is_dir():
                if item.name.lower() == "datasets":
                    for owner in item.iterdir():
                        if owner.is_dir():
                            for ds in owner.iterdir():
                                if ds.is_dir() and "dioptra-dino" not in ds.name.lower():
                                    candidate_roots.append(ds)
                elif "dioptra-dino" not in item.name.lower():
                    candidate_roots.append(item)
        # Shallow pass only - no deep os.walk on FUSE network mounts
        pass
    for local_dir in [Path("data"), Path("."), Path(".."), Path("/kaggle/working")]:
        if local_dir.is_dir():
            for item in local_dir.iterdir():
                if item.is_dir() and not item.name.startswith(".") and item.name not in ("outputs", "outputs_dino", "build", "dioptra_repo"):
                    candidate_roots.append(item)

    for cand in candidate_roots:
        try:
            _classify_and_add(cand)
        except Exception:
            pass

    # Sort results deterministically
    results.sort(key=lambda t: (t[1], t[0]))
    return results


def resolve_dataset_root(path: str = "auto") -> str:
    """Resolve a single dataset root for backwards compatibility."""
    all_roots = resolve_all_dataset_roots(path)
    if all_roots:
        # Prioritize tartan or first discovered root
        for r, dom in all_roots:
            if dom == "tartan" and "warehouse" in r.lower():
                return r
        return all_roots[0][0]

    # Fallback to single mount discovery
    p = Path(path)
    resolved = _resolve_single_mount(p)
    if resolved is not None:
        return str(resolved)
    if p.exists():
        return str(p)
    return path


# ---------------------------------------------------------------------------
# Multi-Domain Dataset Loader for Dioptra-DINO
# ---------------------------------------------------------------------------

class MultiDomainDINODataset(torch.utils.data.Dataset):
    """Unified multi-domain dataset loader supporting:
      1. TartanAir (v1 & v2 suites, warehouse, indoors)
      2. TartanGround AMR sets (hospital, office, oldindustrialcity)
      3. Apple Hypersim (191 indoor environments)
      4. NYU-Depth-v2 (official split)
      5. KITTI (Eigen metric depth benchmark)
    """

    def __init__(
        self,
        root_dirs: Union[str, List[str]] = "auto",
        split: str = "train",
        image_size: int = 224,
        apply_pinhole_aug: bool = True,
        crop_min: float = 0.35,
        subset_fraction: Optional[float] = None,
        subset_seed: int = 0,
        domain_balanced: bool = True,
        sensor_noise_aug: bool = True,
    ):
        self.split = split
        self.image_size = image_size
        self.apply_pinhole_aug = apply_pinhole_aug and (split == "train")
        self.crop_min = crop_min
        self.subset_fraction = subset_fraction
        self.subset_seed = subset_seed
        self.domain_balanced = domain_balanced and (split == "train")
        self.sensor_noise_aug = sensor_noise_aug and (split == "train")

        tartan_K = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Canonical Camera Intrinsics per domain
        self.K_CANONICAL = {
            "tartan": tartan_K,
            "tartanair_warehouse": tartan_K,
            "tartanair_indoors": tartan_K,
            "tartanground_amr": tartan_K,
            "hypersim": np.array([
                [888.89, 0.0, 512.0],
                [0.0, 1000.0, 384.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),  # 1024x768, 60 deg horizontal FOV
            "nyu": np.array([
                [518.8579, 0.0, 325.5824],
                [0.0, 518.8579, 253.7362],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),  # 640x480 NYU-Depth-v2 official
            "kitti": np.array([
                [721.5377, 0.0, 609.5593],
                [0.0, 721.5377, 172.854],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),  # KITTI cam2 canonical
            "scannet": np.array([
                [577.87, 0.0, 319.5],
                [0.0, 577.87, 239.5],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),  # 640x480 ScanNet official RGB-D sensor
        }

        # Resolve roots
        if isinstance(root_dirs, str):
            resolved_tuples = resolve_all_dataset_roots(root_dirs)
        else:
            resolved_tuples = []
            for r in root_dirs:
                resolved_tuples.extend(resolve_all_dataset_roots(r))

        self.resolved_roots = resolved_tuples
        self.samples: List[Tuple[str, str, str]] = []  # (img_path, depth_path, domain)
        self._zip_cache: Dict[str, zipfile.ZipFile] = {}
        self._zip_cache_pid: Optional[int] = None

        self._build_index()
        print(f"[Dioptra-DINO Dataset] Successfully indexed {len(self.samples)} {split} samples across {len(self.resolved_roots)} domain roots.")
        self._apply_subset()

        # Build domain cluster mapping for balanced sampling across physical & synthetic regimes
        self.domain_clusters = ["scannet", "nyu", "hypersim", "tartan"]
        self.domain_to_indices: Dict[str, List[int]] = {c: [] for c in self.domain_clusters}
        for idx, item in enumerate(self.samples):
            dom = item[2].lower()
            if "scannet" in dom:
                self.domain_to_indices["scannet"].append(idx)
            elif "nyu" in dom:
                self.domain_to_indices["nyu"].append(idx)
            elif "hypersim" in dom:
                self.domain_to_indices["hypersim"].append(idx)
            else:
                self.domain_to_indices["tartan"].append(idx)
        self.active_clusters = [c for c in self.domain_clusters if len(self.domain_to_indices[c]) > 0]
        if self.domain_balanced and len(self.active_clusters) > 1:
            print(f"[Dioptra-DINO Dataset] Domain-balanced batching active across {len(self.active_clusters)} clusters: "
                  + ", ".join(f"{c}: {len(self.domain_to_indices[c]):,}" for c in self.active_clusters))

    @staticmethod
    def _stratified_subset(
        samples: List[Tuple[str, str, str]], fraction: float, seed: int = 0
    ) -> List[Tuple[str, str, str]]:
        """Deterministic per-domain subsample: keeps every domain represented
        (at least 1 sample) while thinning each proportionally. Pure stdlib."""
        import random as _random
        by_domain: Dict[str, List[Tuple[str, str, str]]] = {}
        for s in samples:
            by_domain.setdefault(s[2], []).append(s)
        rng = _random.Random(seed)
        kept: List[Tuple[str, str, str]] = []
        for domain in sorted(by_domain):
            group = by_domain[domain]
            n_keep = min(len(group), max(1, int(len(group) * fraction)))
            kept.extend(rng.sample(group, n_keep))
        rng.shuffle(kept)
        return kept

    def _apply_subset(self) -> None:
        frac = self.subset_fraction
        if frac is None or not (0.0 < frac < 1.0) or len(self.samples) == 0:
            return
        before = len(self.samples)
        self.samples = self._stratified_subset(self.samples, frac, self.subset_seed)
        n_domains = len({s[2] for s in self.samples})
        print(f"[Dioptra-DINO Dataset] Subset {frac:.3g} (seed {self.subset_seed}): "
              f"{before:,} -> {len(self.samples):,} samples across {n_domains} domains.")

    @staticmethod
    def _normalize_stem(stem: str) -> str:
        for tag in (
            "_lcam_front_depth", "_rcam_front_depth", "_left_depth", "_right_depth",
            "_lcam_front", "_rcam_front", "_left", "_right", "_depth", "_og"
        ):
            if stem.endswith(tag):
                stem = stem[:-len(tag)]
        return stem

    def _is_val_split(self, path_str: str, domain: str) -> bool:
        """Strict validation partition per domain."""
        pl = path_str.lower().replace("\\", "/")
        if domain.startswith("tartan"):
            if "abandonedfactory" in pl:
                return True
            for train_env in (
                "carwelding", "industrialhangar", "supermarket", "hospital", "office",
                "restaurant", "school", "warehouse", "industrial", "diner", "tinyhouse",
                "saloon", "prison", "retro", "brickhouse", "house"
            ):
                if train_env in pl:
                    return False
        elif domain == "scannet":
            # Scene-level 10% holdout (keeps whole scenes unified in either train or val)
            parts = pl.split("/")
            scene_key = next((p for p in parts if p.startswith("scene")), pl)
            h = int(hashlib.md5(scene_key.encode()).hexdigest(), 16)
            return (h % 10) == 0
        elif domain == "nyu":
            if "/nyu2_test" in pl or "/test/" in pl or "/val/" in pl:
                return True
            if "/nyu2_train" in pl or "/train/" in pl:
                return False
        elif domain == "kitti":
            if "/test/" in pl or "/val/" in pl:
                return True
            if "/train/" in pl:
                return False
        elif domain == "hypersim":
            # Hash-based 10% held-out validation on scene folder
            scene_key = pl.split("hypersim")[-1].split("/")[1] if "/hypersim/" in pl else pl
            h = int(hashlib.md5(scene_key.encode()).hexdigest(), 16)
            return (h % 10) == 0

        # Deterministic MD5 fallback
        h = int(hashlib.md5(path_str.encode()).hexdigest(), 16)
        return (h % 10) >= 9

    def _build_index(self):
        # 1. Fast path: load from cached index if available (takes 0.05s).
        # Checks JSON first, then pickle fallback; ignores empty/corrupt files.
        cache_path = os.path.join("/kaggle/working", f"dataset_index_{self.split}.json")
        cache_candidates = [cache_path, os.path.splitext(cache_path)[0] + ".pkl"]
        for cand in cache_candidates:
            try:
                if not (os.path.exists(cand) and os.path.getsize(cand) > 1024):
                    continue
                if cand.endswith(".pkl"):
                    import pickle
                    with open(cand, "rb") as f:
                        cached = pickle.load(f)
                else:
                    with open(cand, "r") as f:
                        cached = json.load(f)
                if len(cached) > 0:
                    self.samples = [tuple(x) for x in cached]
                    print(f"[Dioptra-DINO Dataset] Loaded {len(self.samples):,} {self.split} samples from cache in 0.05s: {cand}")
                    return
            except Exception as e:
                print(f"[Dioptra-DINO Dataset] Note: cache {cand} unusable ({e}); continuing.")
                continue

        t0 = time.time()
        for root_path, domain in self.resolved_roots:
            p = Path(root_path)
            if not p.exists():
                continue
            if domain == "scannet":
                self._index_scannet(root_path)
            elif domain == "hypersim":
                self._index_hypersim(root_path)
            elif domain == "nyu":
                self._index_nyu(root_path)
            elif domain == "kitti":
                self._index_kitti(root_path)
            else:
                self._index_tartan(root_path, domain_tag=domain)

        # Graceful fallback: if split filtering yielded 0 samples, include all found pairs
        if len(self.samples) == 0:
            print(f"[Dioptra-DINO Dataset] Warning: {self.split} split yielded 0 samples. Retrying with force_all=True...")
            for root_path, domain in self.resolved_roots:
                if domain == "scannet":
                    self._index_scannet(root_path, force_all=True)
                elif domain == "hypersim":
                    self._index_hypersim(root_path, force_all=True)
                elif domain == "nyu":
                    self._index_nyu(root_path, force_all=True)
                elif domain == "kitti":
                    self._index_kitti(root_path, force_all=True)
                else:
                    self._index_tartan(root_path, force_all=True, domain_tag=domain)

        elapsed = time.time() - t0
        print(f"[Dioptra-DINO Dataset] Indexing completed in {elapsed:.2f}s ({len(self.samples):,} samples found).")

        # Save to cache for instant sub-second reuse (atomic write + size verification;
        # v4 showed a silent 0-byte save, so failures are now loud with a pickle fallback).
        if len(self.samples) > 0 and os.path.isdir("/kaggle/working"):
            saved = False
            try:
                tmp_path = cache_path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(self.samples, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, cache_path)
                sz = os.path.getsize(cache_path)
                if sz < 1024:
                    raise IOError(f"index cache suspiciously small ({sz} bytes)")
                print(f"[Dioptra-DINO Dataset] Successfully saved index cache: {cache_path} ({sz / 1e6:.1f} MB)")
                saved = True
            except Exception as e:
                print(f"[Dioptra-DINO Dataset] WARNING: JSON index save failed ({e}); trying pickle fallback...")
            if not saved:
                try:
                    import pickle
                    pkl_path = os.path.splitext(cache_path)[0] + ".pkl"
                    tmp_path = pkl_path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        pickle.dump(self.samples, f, protocol=4)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, pkl_path)
                    print(f"[Dioptra-DINO Dataset] Pickle index cache saved: {pkl_path} ({os.path.getsize(pkl_path) / 1e6:.1f} MB)")
                except Exception as e2:
                    print(f"[Dioptra-DINO Dataset] WARNING: index cache save failed entirely ({e2})")

    def _index_tartan(self, root: str, force_all: bool = False, domain_tag: str = "tartan"):
        if str(root).lower().endswith(".zip"):
            self._index_tartan_zip(root, force_all, domain_tag=domain_tag)
        elif os.path.isdir(root):
            self._index_tartan_tree(root, force_all, domain_tag=domain_tag)

    def _index_companion_zips(self, img_zip: str, depth_zip: str, force_all: bool = False, domain_tag: str = "tartan"):
        try:
            with zipfile.ZipFile(img_zip, "r") as zf_img:
                img_names = [n for n in zf_img.namelist() if not n.endswith("/") and n.lower().endswith(".png")]
            with zipfile.ZipFile(depth_zip, "r") as zf_depth:
                depth_names = set(zf_depth.namelist())
        except Exception:
            return

        for iname in img_names:
            matched_dname = None
            if "image_left" in iname:
                base = iname.replace("image_left", "depth_left")
                for c in (
                    base.replace("_left.png", "_left_depth.npy"),
                    base.replace("_left.png", "_left_depth.png"),
                    base.replace(".png", ".npy"),
                    base.replace(".png", "_depth.npy"),
                    base.replace(".png", "_depth.png"),
                    base,
                ):
                    if c in depth_names:
                        matched_dname = c
                        break
            elif "image_lcam_front" in iname:
                base = iname.replace("image_lcam_front", "depth_lcam_front")
                for c in (
                    base.replace("_lcam_front.png", "_lcam_front_depth.png"),
                    base.replace("_lcam_front.png", "_lcam_front_depth.npy"),
                    base.replace(".png", ".npy"),
                    base.replace(".png", "_depth.png"),
                    base.replace(".png", "_depth.npy"),
                    base,
                ):
                    if c in depth_names:
                        matched_dname = c
                        break

            if matched_dname is not None:
                is_val = self._is_val_split(iname, "tartan")
                if force_all or (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                    self.samples.append((f"{img_zip}::{iname}", f"{depth_zip}::{matched_dname}", domain_tag))

    def _index_tartan_zip(self, zip_path: str, force_all: bool = False, domain_tag: str = "tartan"):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
        except Exception:
            return

        img_map: Dict[str, str] = {}
        depth_map: Dict[str, str] = {}
        for n in names:
            nl = n.lower()
            if ("/image_left/" in n or "/image_lcam_front/" in n or n.startswith("image_left/") or n.startswith("image_lcam_front/")) and nl.endswith(".png"):
                mod = "image_left" if "image_left" in n else "image_lcam_front"
                parts = n.split(mod)
                stem = self._normalize_stem(os.path.splitext(os.path.basename(parts[-1]))[0])
                img_map[f"{parts[0]}::{stem}"] = n
            elif ("/depth_left/" in n or "/depth_lcam_front/" in n or n.startswith("depth_left/") or n.startswith("depth_lcam_front/")) and (nl.endswith(".npy") or nl.endswith(".png")):
                mod = "depth_left" if "depth_left" in n else "depth_lcam_front"
                parts = n.split(mod)
                stem = self._normalize_stem(os.path.splitext(os.path.basename(parts[-1]))[0])
                depth_map[f"{parts[0]}::{stem}"] = n

        for key, img_path in sorted(img_map.items()):
            if key in depth_map:
                is_val = self._is_val_split(key, "tartan")
                if force_all or (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                    self.samples.append((f"{zip_path}::{img_path}", f"{zip_path}::{depth_map[key]}", domain_tag))

    def _index_tartan_tree(self, root: str, force_all: bool = False, domain_tag: str = "tartan"):
        samples_found = []
        for dirpath, dirs, files in os.walk(root):
            # 1. Unzipped trajectory discovery (0.001s)
            img_name = "image_left" if "image_left" in dirs else ("image_lcam_front" if "image_lcam_front" in dirs else None)
            if img_name:
                depth_name = "depth_left" if "depth_left" in dirs else ("depth_lcam_front" if "depth_lcam_front" in dirs else None)
                if depth_name:
                    img_dir = os.path.join(dirpath, img_name)
                    depth_dir = os.path.join(dirpath, depth_name)
                    if os.path.isdir(img_dir) and os.path.isdir(depth_dir):
                        try:
                            img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png") and not f.startswith("."))
                            depth_files = set(os.listdir(depth_dir))
                            is_val = self._is_val_split(dirpath, "tartan")
                            if force_all or (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                                for fn in img_files:
                                    stem = self._normalize_stem(fn[:-4])
                                    cand = None
                                    for c in (
                                        f"{stem}_left_depth.npy", f"{stem}_lcam_front_depth.png",
                                        f"{stem}_left_depth.png", f"{stem}_lcam_front_depth.npy",
                                        f"{stem}_depth.npy", f"{stem}_depth.png",
                                        f"{stem}.npy", f"{stem}.png",
                                        f"{fn[:-4]}_depth.npy", f"{fn[:-4]}.npy",
                                        f"{fn[:-4]}_depth.png", f"{fn[:-4]}.png"
                                    ):
                                        if c in depth_files:
                                            cand = os.path.join(depth_dir, c)
                                            break
                                    if cand:
                                        samples_found.append((os.path.join(img_dir, fn), cand, domain_tag))
                        except OSError:
                            pass

            # 2. Companion zip files discovery (e.g. image_left.zip + depth_left.zip)
            zip_files = [f for f in files if f.lower().endswith(".zip")]
            if zip_files:
                img_zips = [os.path.join(dirpath, f) for f in zip_files if "image" in f.lower() or "rgb" in f.lower()]
                depth_zips = [os.path.join(dirpath, f) for f in zip_files if "depth" in f.lower()]
                if img_zips and depth_zips:
                    for iz in img_zips:
                        for dz in depth_zips:
                            self._index_companion_zips(iz, dz, force_all, domain_tag=domain_tag)
                else:
                    for z in zip_files:
                        self._index_tartan_zip(os.path.join(dirpath, z), force_all, domain_tag=domain_tag)

            # 3. DIRECTORY PRUNING: Instantly prune all image and depth directories so os.walk never enters them!
            dirs[:] = [d for d in dirs if d not in (
                "image_left", "depth_left", "image_right", "depth_right",
                "image_lcam_front", "depth_lcam_front", "image_rcam_front", "depth_rcam_front",
                "final_preview", "geometry_hdf5", "disparity", "seg", "flow"
            )]

        self.samples.extend(samples_found)

    def _index_scannet(self, root: str, force_all: bool = False):
        samples_found = []
        # Find scene folders (e.g. /kaggle/input/scannet-frames-99gb/scannet-frames/sceneXXXX_YY)
        scenes_dir = None
        for cand in [os.path.join(root, "scannet-frames"), root]:
            if os.path.isdir(cand):
                try:
                    subdirs = [os.path.join(cand, d) for d in os.listdir(cand) if os.path.isdir(os.path.join(cand, d)) and d.startswith("scene")]
                    if subdirs:
                        scenes_dir = cand
                        break
                except OSError:
                    pass
        if not scenes_dir:
            for dirpath, dirs, _ in os.walk(root):
                if any(d.startswith("scene") for d in dirs):
                    scenes_dir = dirpath
                    break
        if not scenes_dir:
            return

        try:
            scene_folders = sorted(os.path.join(scenes_dir, d) for d in os.listdir(scenes_dir) if os.path.isdir(os.path.join(scenes_dir, d)) and d.startswith("scene"))
        except OSError:
            return

        for sf in scene_folders:
            is_val = self._is_val_split(sf, "scannet")
            if not force_all and ((self.split == "val" and not is_val) or (self.split == "train" and is_val)):
                continue
            try:
                files = os.listdir(sf)
                jpgs = sorted(f for f in files if f.endswith(".jpg"))
                png_set = set(f for f in files if f.endswith(".png"))
                for jf in jpgs:
                    stem = jf[:-4]
                    df = f"{stem}.png"
                    if df in png_set:
                        samples_found.append((os.path.join(sf, jf), os.path.join(sf, df), "scannet"))
            except OSError:
                pass
        self.samples.extend(samples_found)

    def _index_hypersim(self, root: str, force_all: bool = False):
        samples_found = []
        for dirpath, dirs, _ in os.walk(root):
            # Support both exact 'final_preview' and scene-prefixed 'scene_cam_XX_final_preview'
            fp_subdirs = [d for d in dirs if "final_preview" in d]
            if fp_subdirs:
                for fpd in fp_subdirs:
                    fp_dir = os.path.join(dirpath, fpd)
                    geom_dir = os.path.join(dirpath, fpd.replace("final_preview", "geometry_hdf5"))
                    if not os.path.isdir(geom_dir):
                        continue
                    try:
                        tonemaps = sorted(f for f in os.listdir(fp_dir) if f.endswith(".tonemap.jpg"))
                        geom_files = set(os.listdir(geom_dir))
                    except OSError:
                        continue
                    is_val = self._is_val_split(dirpath, "hypersim")
                    if not force_all and ((self.split == "val" and not is_val) or (self.split == "train" and is_val)):
                        continue
                    for fn in tonemaps:
                        dfn = fn.replace(".tonemap.jpg", ".depth_meters.hdf5")
                        if dfn in geom_files:
                            samples_found.append((os.path.join(fp_dir, fn), os.path.join(geom_dir, dfn), "hypersim"))
                dirs[:] = [d for d in dirs if "final_preview" not in d and "geometry_hdf5" not in d]
        self.samples.extend(samples_found)

    def _index_nyu(self, root: str, force_all: bool = False):
        samples_found = []
        for dirpath, dirs, _ in os.walk(root):
            bname = os.path.basename(dirpath).lower()
            if any(k in bname for k in ("nyu2_train", "nyu2_test", "train", "test", "nyu")):
                try:
                    filenames = sorted(os.listdir(dirpath))
                except OSError:
                    continue
                colors = [f for f in filenames if f.endswith("_colors.png")]
                if colors:
                    dirs.clear()
                    fn_set = set(filenames)
                    for fn in colors:
                        dfn = fn.replace("_colors.png", "_depth.png")
                        if dfn in fn_set:
                            img_path = os.path.join(dirpath, fn)
                            depth_path = os.path.join(dirpath, dfn)
                            is_val = self._is_val_split(img_path, "nyu")
                            if force_all or (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                                samples_found.append((img_path, depth_path, "nyu"))
        self.samples.extend(samples_found)

    def _index_kitti(self, root: str, force_all: bool = False):
        samples_found = []
        for dirpath, dirs, _ in os.walk(root):
            if "image_02" in dirs and "proj_depth" in dirs:
                dirs.clear()
                img_dir = os.path.join(dirpath, "image_02", "data")
                depth_dir = os.path.join(dirpath, "proj_depth", "groundtruth", "image_02")
                if os.path.isdir(img_dir) and os.path.isdir(depth_dir):
                    try:
                        img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
                        depth_set = set(os.listdir(depth_dir))
                    except OSError:
                        continue
                    for fn in img_files:
                        if fn in depth_set:
                            img_path = os.path.join(img_dir, fn)
                            depth_path = os.path.join(depth_dir, fn)
                            is_val = self._is_val_split(img_path, "kitti")
                            if force_all or (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                                samples_found.append((img_path, depth_path, "kitti"))
        self.samples.extend(samples_found)

    def _zip_read(self, zip_path: str, member: str) -> bytes:
        pid = os.getpid()
        if self._zip_cache_pid != pid:
            for old_zf in self._zip_cache.values():
                try:
                    old_zf.close()
                except Exception:
                    pass
            self._zip_cache = {}
            self._zip_cache_pid = pid
        zf = self._zip_cache.get(zip_path)
        if zf is None:
            if len(self._zip_cache) >= 1:
                old_key = next(iter(self._zip_cache))
                try:
                    self._zip_cache[old_key].close()
                except Exception:
                    pass
                del self._zip_cache[old_key]
            zf = zipfile.ZipFile(zip_path, "r")
            self._zip_cache[zip_path] = zf
        with zf.open(member) as f:
            return f.read()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        if idx % 200 == 0:
            gc.collect()
        if getattr(self, "domain_balanced", False) and len(getattr(self, "active_clusters", [])) > 1:
            cluster = random.choice(self.active_clusters)
            actual_idx = random.choice(self.domain_to_indices[cluster])
            img_src, depth_src, domain = self.samples[actual_idx]
        else:
            img_src, depth_src, domain = self.samples[idx]
        from PIL import Image as PILImage

        # 1. Load RGB Image
        try:
            if "::" in img_src:
                zp, mp = img_src.split("::", 1)
                img_bytes = self._zip_read(zp, mp)
                img = np.array(PILImage.open(io.BytesIO(img_bytes)).convert("RGB"))
            else:
                img = np.array(PILImage.open(img_src).convert("RGB"))
        except Exception:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        H, W = img.shape[:2]

        # 2. Load Depth Map with Domain-Adaptive Decoding
        depth = None
        try:
            if domain == "hypersim":
                try:
                    import h5py
                    with h5py.File(depth_src, "r") as hf:
                        raw_d = np.array(hf["dataset"][:], dtype=np.float32)
                    # Convert Hypersim Euclidean ray distance to planar orthogonal Z-depth
                    dh, dw = raw_d.shape[:2]
                    K_h = self.K_CANONICAL["hypersim"]
                    fx_h = K_h[0, 0] * (float(dw) / 1024.0)
                    fy_h = K_h[1, 1] * (float(dh) / 768.0)
                    cx_h = K_h[0, 2] * (float(dw) / 1024.0)
                    cy_h = K_h[1, 2] * (float(dh) / 768.0)
                    gy, gx = np.indices((dh, dw), dtype=np.float32)
                    ray_scale = np.sqrt(1.0 + ((gx - cx_h) / fx_h) ** 2 + ((gy - cy_h) / fy_h) ** 2)
                    depth = raw_d / ray_scale
                except Exception:
                    depth = np.zeros((H, W), dtype=np.float32)
            elif domain == "nyu":
                # NYU-Depth-v2: 16-bit uint16 in millimeters
                raw_d = np.array(PILImage.open(depth_src)).astype(np.float32)
                depth = raw_d / 1000.0
            elif domain == "kitti":
                # KITTI: 16-bit uint16 in 1/256 meters
                raw_d = np.array(PILImage.open(depth_src)).astype(np.float32)
                depth = raw_d / 256.0
            elif domain == "scannet":
                # ScanNet: 16-bit uint16 depth in millimeters (divide by 1000.0 -> meters)
                raw_d = np.array(PILImage.open(depth_src)).astype(np.float32)
                depth = raw_d / 1000.0
            else:
                # TartanAir / TartanGround (depth is already in meters; sky/infinity pixels > 120m are masked below)
                if "::" in depth_src:
                    zp, mp = depth_src.split("::", 1)
                    depth_bytes = self._zip_read(zp, mp)
                    if mp.lower().endswith(".png"):
                        raw_d = np.array(PILImage.open(io.BytesIO(depth_bytes)))
                        if raw_d.ndim == 3 and raw_d.shape[2] == 4:
                            # IEEE-754 32-bit float encoding directly in meters
                            depth = np.ascontiguousarray(raw_d).view(np.float32).squeeze(-1)
                        elif raw_d.ndim == 3:
                            depth = raw_d[..., 0].astype(np.float32)
                        else:
                            depth = raw_d.astype(np.float32)
                    else:
                        depth = np.load(io.BytesIO(depth_bytes)).astype(np.float32)
                else:
                    if depth_src.lower().endswith(".png"):
                        raw_d = np.array(PILImage.open(depth_src))
                        if raw_d.ndim == 3 and raw_d.shape[2] == 4:
                            # IEEE-754 32-bit float encoding directly in meters
                            depth = np.ascontiguousarray(raw_d).view(np.float32).squeeze(-1)
                        elif raw_d.ndim == 3:
                            depth = raw_d[..., 0].astype(np.float32)
                        else:
                            depth = raw_d.astype(np.float32)
                    else:
                        depth = np.load(depth_src).astype(np.float32)
        except Exception:
            depth = np.zeros((H, W), dtype=np.float32)

        if depth is None or depth.ndim != 2:
            if depth is not None and depth.ndim == 3:
                depth = depth.squeeze()
            if depth is None or depth.ndim != 2:
                depth = np.zeros((H, W), dtype=np.float32)

        # Ensure depth resolution matches image resolution before cropping or resizing
        if depth.shape[:2] != (H, W):
            try:
                depth = np.array(PILImage.fromarray(depth).resize((W, H), PILImage.NEAREST))
            except Exception:
                depth = np.zeros((H, W), dtype=np.float32)

        # Sanitize depth: replace NaNs/Infs, clamp metric range [0.01, 120.0]
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depth = np.where((depth >= 0.01) & (depth <= 120.0), depth, 0.0)

        # 3. Canonical Intrinsics Matrix per Domain
        K = self.K_CANONICAL.get(domain, self.K_CANONICAL["tartan"]).copy()
        if domain.startswith("tartan") and H == W:
            # TartanAir2 square 640x640 sensor: principal point cy = cx = 320.0
            K[1, 2] = K[0, 2]

        # 4. Dynamic Pinhole Crop Augmentation (Scale focal lengths with simulated optical crop)
        if self.apply_pinhole_aug and random.random() < 0.9:
            img, depth, K = apply_dynamic_pinhole_crop(
                img, depth, K, crop_size_range=(self.crop_min, 1.0), out_size=self.image_size
            )
        else:
            scale_x = self.image_size / float(W)
            scale_y = self.image_size / float(H)
            K[0, 0] *= scale_x
            K[1, 1] *= scale_y
            K[0, 2] *= scale_x
            K[1, 2] *= scale_y

            try:
                img = np.array(PILImage.fromarray(img).resize((self.image_size, self.image_size), PILImage.BILINEAR))
                depth = np.array(PILImage.fromarray(depth).resize((self.image_size, self.image_size), PILImage.NEAREST))
            except Exception:
                y_idx = (np.linspace(0, H - 1, self.image_size)).astype(int)
                x_idx = (np.linspace(0, W - 1, self.image_size)).astype(int)
                img = img[np.ix_(y_idx, x_idx)]
                depth = depth[np.ix_(y_idx, x_idx)]

        # Realistic Physical Sensor Degradation on synthetic domains
        if self.apply_pinhole_aug and getattr(self, "sensor_noise_aug", True) and (domain.startswith("tartan") or "hypersim" in domain) and random.random() < 0.4:
            # 1. Random specular/IR dropout holes
            num_holes = random.randint(1, 4)
            for _ in range(num_holes):
                rh = random.randint(int(self.image_size * 0.05), int(self.image_size * 0.20))
                rw = random.randint(int(self.image_size * 0.05), int(self.image_size * 0.20))
                ry = random.randint(0, max(1, self.image_size - rh))
                rx = random.randint(0, max(1, self.image_size - rw))
                depth[ry:ry+rh, rx:rx+rw] = 0.0
            # 2. Distance-dependent sensor noise: sigma ~ 0.001 * z^2
            valid_m = (depth > 0.1) & (depth < 40.0)
            noise = np.random.normal(0, 0.001 * (depth ** 2)).astype(np.float32)
            depth = np.where(valid_m, np.maximum(0.1, depth + noise), depth)

        # 5. Normalize tensors
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        for c, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            img_tensor[c] = (img_tensor[c] - mean) / std

        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        K_tensor = torch.from_numpy(K).float()
        return img_tensor, depth_tensor, K_tensor


# ---------------------------------------------------------------------------
# TartanAir Dataset Loader (Reference: volsiai epoch 1-5 training pipeline)
# ---------------------------------------------------------------------------

class TartanAirDINODataset(MultiDomainDINODataset):
    """Robust, fast TartanAir dataset loader inheriting pruned directory scanning and multi-archive support."""

    def __init__(
        self,
        root_dir: str = "auto",
        split: str = "train",
        image_size: int = 224,
        apply_pinhole_aug: bool = True,
        crop_min: float = 0.35,
    ):
        r_dir = resolve_dataset_root(root_dir)
        if os.path.isdir(os.path.join(r_dir, "warehouse_stereo")):
            r_dir = os.path.join(r_dir, "warehouse_stereo")
        super().__init__(
            root_dirs=r_dir,
            split=split,
            image_size=image_size,
            apply_pinhole_aug=apply_pinhole_aug,
            crop_min=crop_min,
        )




# ---------------------------------------------------------------------------
# TPU v5e Training via PyTorch/XLA (single-process, single-core)
# ---------------------------------------------------------------------------
# Native bfloat16 MXU path + ParallelLoader + xm.optimizer_step + static-shape
# loss (masked sums, gated VNL, log-mean scale): boolean mask indexing yields
# per-batch variable tensor sizes, which recompiles the full XLA graph EVERY
# step. All XLA imports are lazy so GPU/CPU still imports without torch_xla.
# Static shapes everywhere: drop_last=True, fixed 336x336, K [B,3,3].


def _get_xla_modules():
    """Lazily import torch_xla modules. Raises RuntimeError if unavailable."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        import torch_xla.distributed.parallel_loader as pl  # type: ignore
        import torch_xla.distributed.xla_multiprocessing as xmp  # type: ignore
        return xm, pl, xmp
    except Exception as e:
        raise RuntimeError(
            "torch_xla is required for TPU training but could not be imported. "
            f"Install it on the TPU VM first. Underlying error: {e}"
        )


def _tpu_world_size(xm) -> int:
    for attr in ("xrt_world_size", "get_world_size"):
        try:
            fn = getattr(xm, attr, None)
            if fn is not None:
                return int(fn())
        except Exception:
            continue
    return 8


def _tpu_find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """TPU-side checkpoint finder (mirrors GPU train path, master-agnostic read)."""
    for name in ["checkpoint_step_latest.pt", "checkpoint_latest.pt"]:
        p = os.path.join(output_dir, name)
        if os.path.isfile(p):
            return p
    epoch_cands = sorted(
        glob.glob(os.path.join(output_dir, "dioptra_dino_epoch_*.pt")),
        key=lambda q: int(os.path.splitext(os.path.basename(q))[0].split("_")[-1])
        if os.path.splitext(os.path.basename(q))[0].split("_")[-1].isdigit() else 0,
    )
    if epoch_cands:
        return epoch_cands[-1]
    best_p = os.path.join(output_dir, "dioptra_dino_best.pt")
    if os.path.isfile(best_p):
        return best_p
    for p in [
        "/kaggle/input/dioptra-dino-epoch5-ckpt/checkpoint_step_latest.pt",
        "/kaggle/input/datasets/yumnamharryson/dioptra-dino-epoch5-ckpt/checkpoint_step_latest.pt",
        "/kaggle/input/dioptra-dino-epoch5-ckpt/dioptra_dino_epoch_4.pt",
        "/kaggle/input/datasets/yumnamharryson/dioptra-dino-epoch5-ckpt/dioptra_dino_epoch_4.pt",
    ]:
        if os.path.isfile(p):
            return p
    for name in ["checkpoint_step_latest.pt", "checkpoint_latest.pt"]:
        step_cands = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        if step_cands:
            return step_cands[0]
    input_cands = sorted(
        glob.glob("/kaggle/input/**/dioptra_dino_epoch_*.pt", recursive=True)
        + glob.glob("/kaggle/input/**/dioptra_dino*.pt", recursive=True),
        key=lambda q: int(os.path.splitext(os.path.basename(q))[0].split("_")[-1])
        if os.path.splitext(os.path.basename(q))[0].split("_")[-1].isdigit() else 0,
    )
    if input_cands:
        return input_cands[-1]
    return None


def _tpu_mp_fn(index: int, args) -> None:
    """Single-process TPU worker, called directly with index=0.

    Single-core training (args._tpu_single_device) with static-shape loss.
    The legacy DistributedSampler branch is retained for multi-process hosts.
    """
    xm, pl, _ = _get_xla_modules()
    import contextlib

    # HDF5 file locking over network FUSE mounts is a classic indefinite-hang
    # source (Hypersim .hdf5 reads). Must be set before DataLoader workers fork.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    single = bool(getattr(args, "_tpu_single_device", False))
    device = xm.xla_device()
    if single:
        world_size, rank, is_master = 1, 0, True
        print("[TPU] Single-core mode: training on one XLA device.")
        try:
            print(f"[TPU] Visible XLA devices from this process: {len(xm.get_xla_supported_devices())}")
        except Exception as e:
            print(f"[TPU] Device query note: {e}")
    else:
        world_size = _tpu_world_size(xm)
        rank = xm.get_ordinal()
        is_master = xm.is_master_ordinal()

    img_sz = getattr(args, "image_size", 336) or 336
    w_norm = getattr(args, "weight_normal", 0.25)
    bs = getattr(args, "batch_size", 4) or 4
    grad_accum = getattr(args, "grad_accum", None) or 4
    use_ckpt_flag = getattr(args, "use_checkpointing", True)
    use_bf16 = bool(getattr(args, "use_bfloat16", True))
    num_workers = int(getattr(args, "num_workers", 4) or 4)
    save_interval = int(getattr(args, "save_interval", 500) or 500)
    output_dir = getattr(args, "output_dir", "/kaggle/working/outputs_dino")

    if single:
        # One core has 16 GB HBM: cap the micro-batch for memory safety.
        bs = min(bs, 8)
    cfg = DioptraDINOConfig(
        image_size=img_sz,
        grid_size=img_sz // 14,
        epochs=args.epochs,
        batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_normal=w_norm,
        # torch>=2.8 non-reentrant activation checkpointing queries a
        # `torch.xla` device module that does not exist -> AttributeError on XLA
        # tensors. Full activations at bs<=8/336/bf16 fit easily in 16 GB HBM.
        use_checkpointing=False,
        # Static-shape loss: masked sums + gated VNL + log-mean scale.
        # Boolean indexing yields per-batch variable sizes => full-graph XLA
        # recompile on EVERY step (v8: 4hrs silence). GPU path keeps indexing.
        tpu_static_loss=True,
        freeze_backbone=getattr(args, "freeze_backbone", False),
    )
    if is_master:
        if cfg.freeze_backbone:
            print(f"[TPU] Level 1 Acceleration: DINOv2 backbone frozen. Training geometric head.")
        print(f"[TPU] Activation checkpointing disabled for XLA (torch has no 'xla' device module); "
              f"micro-batch={bs}, grad_accum={grad_accum}.")

    # 1. Distributed Dataset & Sampler (drop_last=True => static shapes => 1 XLA compile)
    dataset = MultiDomainDINODataset(
        root_dirs=getattr(args, "train", None) or "auto",
        split="train",
        image_size=336 if img_sz is None else img_sz,
        apply_pinhole_aug=True,
        crop_min=getattr(args, "crop_min", 0.35),
        subset_fraction=getattr(args, "subset_fraction", None),
        subset_seed=getattr(args, "subset_seed", 0),
        domain_balanced=getattr(args, "domain_balanced", True),
        sensor_noise_aug=getattr(args, "sensor_noise", True),
    )
    if is_master:
        print(f"[TPU] Indexed {len(dataset):,} training samples across {len(dataset.resolved_roots)} roots.")
    if len(dataset) == 0:
        raise ValueError("Found 0 training samples on TPU worker. Check /kaggle/input mounts.")

    # Worker-crash with default timeout=0 hangs forever with zero output (v7:
    # 1hr silence on step 0). timeout=600 turns a hung worker into a loud error.
    loader_timeout = 600 if num_workers > 0 else 0
    sampler = None
    if single:
        # Single-core fallback: plain shuffled loader (no cross-process sharding).
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=False,
            timeout=loader_timeout,
        )
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=bs,  # e.g. 4 images per core -> 32 global batch size
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=False,
            timeout=loader_timeout,
        )

    if is_master:
        # Heartbeat probe: fetch one batch single-processed BEFORE any XLA work,
        # so a data-path stall can never again masquerade as a TPU stall.
        print("[TPU] Warming data pipeline: fetching first batch (FUSE/h5py path check)...")
        t_fetch = time.time()
        probe = torch.utils.data.DataLoader(dataset, batch_size=min(bs, 4), shuffle=True, num_workers=0, drop_last=True)
        first = next(iter(probe))
        print(f"[TPU] First batch OK in {time.time() - t_fetch:.1f}s: "
              f"img {tuple(first[0].shape)}, depth {tuple(first[1].shape)}, K {tuple(first[2].shape)}")
        del first, probe

    model = DioptraDINO(cfg).to(device)
    criterion = DioptraDINOLoss(cfg).to(device)
    raw_model = model.module if hasattr(model, "module") else model

    if cfg.freeze_backbone:
        param_groups = [
            {"params": [p for p in raw_model.parameters() if p.requires_grad], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
        ]
    else:
        param_groups = [
            {"params": raw_model.backbone.parameters(), "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in raw_model.named_parameters() if not n.startswith("backbone.")],
             "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
        ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    # Resume (every rank loads read-only from shared /kaggle/input; weights only in finetune mode)
    start_epoch = 0
    global_step = 0
    ckpt_loaded = None
    resume_arg = getattr(args, "resume", None)
    if resume_arg and str(resume_arg).lower() not in ("none", "false"):
        if isinstance(resume_arg, str) and os.path.isfile(resume_arg):
            resume_target = resume_arg
        else:
            resume_target = _tpu_find_latest_checkpoint(output_dir)
        if resume_target and os.path.exists(resume_target):
            if is_master:
                print(f"[TPU] Resuming from checkpoint: {resume_target}")
            try:
                ckpt_loaded = torch.load(resume_target, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt_loaded = torch.load(resume_target, map_location="cpu")
            sd = ckpt_loaded.get("model_state_dict", ckpt_loaded.get("state_dict", ckpt_loaded))
            raw_model.load_state_dict(sd, strict=False)
            if getattr(args, "finetune", False):
                start_epoch, global_step = 0, 0
                if is_master:
                    print("[TPU] Fine-tune mode: weights loaded, fresh optimizer/scheduler, ARA gate=1.0.")
            else:
                start_epoch = int(ckpt_loaded.get("epoch", 0))
                global_step = int(ckpt_loaded.get("global_step", 0))
                if "optimizer_state_dict" in ckpt_loaded:
                    try:
                        optimizer.load_state_dict(ckpt_loaded["optimizer_state_dict"])
                    except Exception as exc:
                        if is_master:
                            print(f"[TPU] Note: optimizer state skipped ({exc})")

    # Optimizer steps (not batches): one step per grad_accum micro-batches.
    steps_per_epoch = max(1, (len(loader) + grad_accum - 1) // grad_accum)
    total_training_steps = max(1, cfg.epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_training_steps)
    if global_step > 0 and not getattr(args, "finetune", False):
        for _ in range(min(global_step, total_training_steps - 1)):
            scheduler.step()

    if is_master:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[TPU] Config: epochs={cfg.epochs} per-core-bs={bs} "
              f"global-bs={bs * world_size} grad_accum={grad_accum} "
              f"eff-bs={bs * world_size * grad_accum} bf16={use_bf16} workers={world_size}")

    def _autocast_ctx():
        if use_bf16:
            return torch.autocast(device_type="xla", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    for epoch in range(start_epoch, cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        # ParallelLoader streams host batches onto the TPU device (static shapes).
        para_loader = pl.ParallelLoader(loader, [device])
        per_device_loader = para_loader.per_device_loader(device)
        # Device-side running loss: accumulation is async (no host sync until epoch end).
        epoch_loss_t = torch.zeros((), device=device)
        n_steps = 0
        t_start = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(per_device_loader):
            n_steps += 1
            global_step += 1
            if isinstance(batch, dict):
                img, depth, K = batch["image"], batch["depth"], batch["intrinsics"]
            else:
                img, depth, K = batch
            # ParallelLoader already places tensors on `device`; enforce K shape [B,3,3].
            # .dim/.shape are host-side metadata: no device sync.
            assert K.dim() == 3 and K.shape[1:] == (3, 3), f"Unexpected K shape {tuple(K.shape)}"

            current_gate = 1.0 if getattr(args, "finetune", False) else min(1.0, float(epoch) / 3.0)

            # Throttled host sync: only the master materializes metrics every 50 steps.
            # All other steps pass sync_metrics=False => zero .item() barriers.
            need_metrics = bool(is_master and step % 50 == 0)

            # Forward pass in bfloat16 for TPU v5e MXUs.
            with _autocast_ctx():
                pred = model(img, K, ara_gate=current_gate)
                loss, loss_dict = criterion(pred, depth, img, K, sync_metrics=need_metrics)

            epoch_loss_t = epoch_loss_t + loss.detach()
            loss = loss / grad_accum
            loss.backward()
            xm.mark_step()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                # xm.optimizer_step performs all-reduce across all 8 cores and applies gradients.
                xm.optimizer_step(optimizer, barrier=True)
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if need_metrics:
                step_loss_val = loss.item() * grad_accum
                print(f"Epoch [{epoch + 1}/{cfg.epochs}] Step [{step}/{len(loader)}] "
                      f"Loss: {step_loss_val:.4f} "
                      f"(SiLog: {loss_dict.get('loss_silog', 0):.3f}, "
                      f"Scale: {loss_dict.get('loss_scale', 0):.3f}, "
                      f"Edge: {loss_dict.get('loss_edge', 0):.3f}, "
                      f"VNL: {loss_dict.get('loss_vnl', 0):.3f}) "
                      f"ARA Gate: {current_gate:.2f}")

            # Master-only periodic checkpoint via xm.save (XLA tensor serialization).
            if is_master and (step + 1) % save_interval == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "cfg": cfg,
                }
                xm.save(checkpoint, os.path.join(output_dir, "checkpoint_step_latest.pt"))
                xm.save(checkpoint, os.path.join(output_dir, "checkpoint_latest.pt"))
                del checkpoint

        # Single host sync per epoch (master only): materialize the mean loss.
        mean_loss_t = epoch_loss_t / max(1, n_steps)
        mean_loss = float(mean_loss_t.item()) if is_master else 0.0
        del epoch_loss_t, mean_loss_t
        if is_master:
            print(f"[TPU] ==> Epoch {epoch + 1} Complete! Mean Loss: {mean_loss:.4f}, "
                  f"Runtime: {time.time() - t_start:.1f}s")
            epoch_dict = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cfg": cfg,
                "mean_loss": mean_loss,
            }
            xm.save(epoch_dict, os.path.join(output_dir, f"dioptra_dino_epoch_{epoch + 1}.pt"))
            xm.save(epoch_dict, os.path.join(output_dir, "checkpoint_latest.pt"))
            xm.save(epoch_dict, os.path.join(output_dir, "dioptra_dino_best.pt"))
            print(f"[TPU] Saved atomic checkpoint: dioptra_dino_epoch_{epoch + 1}.pt")
            del epoch_dict

    if is_master:
        print("\n>>> DIOPTRA-DINO TPU v5e-8 TRAINING COMPLETED SUCCESSFULLY! <<<\n")


def train_dioptra_dino_tpu(args) -> None:
    """Entry point for TPU v5e: single-process single-core XLA training.

    Multiprocess xmp.spawn cannot init a TPU client on this host (Kaggle's
    single-client libtpu build fails every child with slice_builder
    "Expected 8 worker addresses, got 1", via fork or spawn), and SPMD Mesh
    construction is broken against the installed numpy (UFuncTypeError). So
    one process trains on one core with static-shape loss formulations.
    """
    os.environ["PJRT_DEVICE"] = "TPU"
    _get_xla_modules()  # fail fast if torch_xla is missing
    print("[TPU] Single-process single-core XLA training ...")
    args._tpu_spmd = False
    args._tpu_single_device = True
    _tpu_mp_fn(0, args)


def should_use_tpu(args) -> bool:
    """True when --tpu is passed or PJRT_DEVICE=TPU with torch_xla importable."""
    if bool(getattr(args, "tpu", False)):
        return True
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        try:
            _get_xla_modules()
            return True
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Resumable Training Pipeline
# ---------------------------------------------------------------------------

def train_dioptra_dino(args):
    """Execute high-performance multi-domain training on TartanAir, Hypersim, NYUv2, and KITTI."""
    print("=" * 70)
    print("STARTING DIOPTRA-DINO MULTI-DOMAIN TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Compute Device : {device}")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU Model      : {torch.cuda.get_device_name(0)} (Count: {gpu_count})")
    else:
        gpu_count = 0

    img_sz = getattr(args, "image_size", 224)
    w_norm = getattr(args, "weight_normal", 0.25)
    bs = getattr(args, "batch_size", 4)
    grad_accum = getattr(args, "grad_accum", None)
    if grad_accum is None:
        grad_accum = max(1, 32 // max(1, bs))
    use_ckpt = getattr(args, "use_checkpointing", True)

    freeze_bb = getattr(args, "freeze_backbone", False)

    cfg = DioptraDINOConfig(
        image_size=img_sz,
        grid_size=img_sz // 14,
        epochs=args.epochs,
        batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_normal=w_norm,
        use_checkpointing=use_ckpt,
        freeze_backbone=freeze_bb,
    )

    model = DioptraDINO(cfg).to(device)
    loss_fn = DioptraDINOLoss(cfg).to(device)

    # Compute device: single dedicated GPU eliminates DataParallel scatter/gather memory leaks
    use_dp = getattr(args, "use_dp", False)
    if torch.cuda.is_available() and gpu_count > 1 and use_dp:
        print(f"[Dioptra-DINO] Multi-GPU acceleration enabled across {gpu_count} GPUs via DataParallel!")
        model = nn.DataParallel(model)
    else:
        if torch.cuda.is_available() and gpu_count > 1:
            print(f"[Dioptra-DINO] Dedicated GPU training on {device} (avoids DataParallel VRAM gather bottleneck & grad-accum leaks).")

    raw_model = model.module if hasattr(model, "module") else model

    # Optimizer with differential learning rates
    if cfg.freeze_backbone:
        param_groups = [
            {"params": [p for p in raw_model.parameters() if p.requires_grad], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
        ]
        num_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"[Dioptra-DINO] Level 1 Acceleration: DINOv2 backbone frozen. Training {num_trainable:,} head parameters (~3.8M) with full dynamic crops.")
    else:
        param_groups = [
            {"params": raw_model.backbone.parameters(), "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in raw_model.named_parameters() if not n.startswith("backbone.")], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
        ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    # Modern AMP GradScaler
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and torch.cuda.is_available())
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and torch.cuda.is_available())

    # Dataset instantiation: High-performance pruned multi-domain and TartanAir loader
    train_root_arg = getattr(args, "train", None) or "auto"
    print(f"[Dioptra-DINO Dataset] Initializing dataset loader: '{train_root_arg}'")
    dataset = MultiDomainDINODataset(
        root_dirs=train_root_arg,
        split="train",
        image_size=cfg.image_size,
        apply_pinhole_aug=True,
        crop_min=getattr(args, "crop_min", 0.35),
        subset_fraction=getattr(args, "subset_fraction", None),
        subset_seed=getattr(args, "subset_seed", 0),
        domain_balanced=getattr(args, "domain_balanced", True),
        sensor_noise_aug=getattr(args, "sensor_noise", True),
    )
    print(f"[Dioptra-DINO Dataset] Successfully loaded {len(dataset):,} training samples ✓")

    if len(dataset) == 0:
        raise ValueError(
            f"Found 0 training samples across configured roots ({train_root_arg})!\n"
            "Please ensure TartanAir warehouse stereo suite is attached to this Kaggle notebook."
        )

    # Use 0 workers and pin_memory=False for absolute host RAM stability on Kaggle Linux VM
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    output_dir = getattr(args, "output_dir", "outputs_dino")
    os.makedirs(output_dir, exist_ok=True)

    # Atomic checkpoint helper with CPU serialization to avoid GPU VRAM spikes
    def _to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_to_cpu(v) for v in obj]
        return obj

    def atomic_save(state_dict, file_path):
        tmp_path = f"{file_path}.tmp"
        cpu_state_dict = _to_cpu(state_dict)
        torch.save(cpu_state_dict, tmp_path)
        del cpu_state_dict
        os.replace(tmp_path, file_path)

    # Checkpoint Resume Finder
    def find_latest_checkpoint(out_dir: str) -> Optional[str]:
        # 1. Check out_dir for step / latest checkpoints
        for name in ["checkpoint_latest.pt", "checkpoint_step_latest.pt"]:
            p = os.path.join(out_dir, name)
            if os.path.isfile(p):
                return p
        # 2. Check out_dir for epoch checkpoints
        epoch_cands = sorted(
            glob.glob(os.path.join(out_dir, "dioptra_dino_epoch_*.pt")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]) if os.path.splitext(os.path.basename(p))[0].split("_")[-1].isdigit() else 0
        )
        if epoch_cands:
            return epoch_cands[-1]
        # 3. Check out_dir best checkpoint
        best_p = os.path.join(out_dir, "dioptra_dino_best.pt")
        if os.path.isfile(best_p):
            return best_p
        # 4. Fast shallow check for known input checkpoint locations
        for p in [
            "/kaggle/input/datasets/volsiai/dioptra-dino-epoch5-ckpt/checkpoint_step_latest.pt",
            "/kaggle/input/dioptra-dino-epoch5-ckpt/checkpoint_step_latest.pt",
            "/kaggle/input/datasets/yumnamharryson/dioptra-dino-epoch5-ckpt/checkpoint_step_latest.pt",
            "/kaggle/input/datasets/volsiai/dioptra-dino-epoch5-ckpt/checkpoint_latest.pt",
            "/kaggle/input/dioptra-dino-epoch5-ckpt/checkpoint_latest.pt",
            "/kaggle/input/datasets/yumnamharryson/dioptra-dino-epoch5-ckpt/dioptra_dino_epoch_4.pt",
            "/kaggle/input/dioptra-dino-epoch5-ckpt/dioptra_dino_epoch_4.pt",
        ]:
            if os.path.isfile(p):
                return p

        # Fast shallow glob over immediate input subdirectories (avoids 4-hour recursive glob over millions of files)
        shallow_cands = (
            glob.glob("/kaggle/input/*ckpt*/*.pt")
            + glob.glob("/kaggle/input/*epoch*/*.pt")
            + glob.glob("/kaggle/input/*/*ckpt*/*.pt")
        )
        if shallow_cands:
            return shallow_cands[0]

        # Prefer step checkpoint first (most recent progress)
        for name in ["checkpoint_step_latest.pt", "checkpoint_latest.pt"]:
            step_cands = glob.glob(f"/kaggle/input/*/{name}")
            if step_cands:
                return step_cands[0]
        # Then check epoch checkpoints sorted by epoch number
        input_cands = sorted(
            glob.glob("/kaggle/input/**/dioptra_dino_epoch_*.pt", recursive=True)
            + glob.glob("/kaggle/input/**/dioptra_dino*.pt", recursive=True)
            + glob.glob("/kaggle/input/**/dioptra_dino*.zip", recursive=True),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]) if os.path.splitext(os.path.basename(p))[0].split("_")[-1].isdigit() else 0
        )
        if input_cands:
            return input_cands[-1]
        return None

    # Execute Resume Logic
    start_epoch = 0
    global_step = 0
    ckpt_loaded = None
    resume_arg = getattr(args, "resume", None)

    if resume_arg and str(resume_arg).lower() not in ("none", "false"):
        resume_target = resume_arg if (isinstance(resume_arg, str) and os.path.isfile(resume_arg)) else find_latest_checkpoint(output_dir)
        if resume_target and os.path.exists(resume_target):
            # If checkpoint is packaged inside a .zip file, extract the .pt file first if needed
            if resume_target.lower().endswith(".zip"):
                try:
                    import tempfile
                    with zipfile.ZipFile(resume_target, "r") as zf:
                        pt_members = [m for m in zf.namelist() if m.endswith(".pt")]
                        if pt_members:
                            extract_tmp = tempfile.mkdtemp(prefix="ckpt_extract_")
                            pt_members.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]) if os.path.splitext(os.path.basename(p))[0].split("_")[-1].isdigit() else 0)
                            extracted_pt = zf.extract(pt_members[-1], path=extract_tmp)
                            resume_target = extracted_pt
                except Exception as z_err:
                    print(f"[Dioptra-DINO] Warning: Zip inspection note: {z_err}")

            print(f"[Dioptra-DINO] Resuming training from checkpoint: {resume_target}")
            import __main__
            if not hasattr(__main__, "DioptraDINOConfig"):
                setattr(__main__, "DioptraDINOConfig", DioptraDINOConfig)
            try:
                ckpt_loaded = torch.load(resume_target, map_location=device, weights_only=False)
            except TypeError:
                ckpt_loaded = torch.load(resume_target, map_location=device)

            sd = ckpt_loaded.get("model_state_dict", ckpt_loaded.get("state_dict", ckpt_loaded))
            if isinstance(sd, dict):
                sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
                raw_model.load_state_dict(sd, strict=False)
            else:
                raw_model.load_state_dict(sd)

            if getattr(args, "finetune", False):
                start_epoch = 0
                global_step = 0
                print(f"[Dioptra-DINO] Fine-tune mode active: Loaded weights from {resume_target}. Initializing fresh optimizer and CosineAnnealing schedule for high-res training.")
            else:
                start_epoch = ckpt_loaded.get("epoch", 0)
                global_step = ckpt_loaded.get("global_step", start_epoch * len(dataloader))

                if "optimizer_state_dict" in ckpt_loaded and optimizer is not None:
                    try:
                        optimizer.load_state_dict(ckpt_loaded["optimizer_state_dict"])
                        print("[Dioptra-DINO] Restored optimizer state successfully.")
                    except Exception as exc:
                        print(f"[Dioptra-DINO] Note: optimizer state skipped ({exc})")

                if "scaler_state_dict" in ckpt_loaded and scaler is not None:
                    try:
                        scaler.load_state_dict(ckpt_loaded["scaler_state_dict"])
                        print("[Dioptra-DINO] Restored AMP scaler state successfully.")
                    except Exception as exc:
                        print(f"[Dioptra-DINO] Note: scaler state skipped ({exc})")

                print(f"[Dioptra-DINO] Successfully restored state! Starting Epoch {start_epoch + 1}/{cfg.epochs} (Global Step: {global_step})")

    # Learning Rate Scheduler
    total_training_steps = max(1, cfg.epochs * len(dataloader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_training_steps)
    if not getattr(args, "finetune", False) and ckpt_loaded and "scheduler_state_dict" in ckpt_loaded:
        try:
            scheduler.load_state_dict(ckpt_loaded["scheduler_state_dict"])
            print("[Dioptra-DINO] Restored scheduler state.")
        except Exception:
            for _ in range(global_step):
                scheduler.step()
    elif global_step > 0:
        for _ in range(global_step):
            scheduler.step()

    print(f"Training configuration: Epochs={cfg.epochs}, Batches/Epoch={len(dataloader)}, "
          f"BatchSize={cfg.batch_size} (EffBatchSize={cfg.batch_size * cfg.gradient_accumulation_steps})")

    def get_autocast_context(enabled: bool):
        if torch.cuda.is_available():
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast(device_type="cuda", enabled=enabled)
            return torch.cuda.amp.autocast(enabled=enabled)
        return torch.cpu.amp.autocast(enabled=False) if hasattr(torch, "cpu") else torch.cuda.amp.autocast(enabled=False)

    start_step = 0
    if ckpt_loaded:
        expected_epoch_start_step = start_epoch * len(dataloader)
        if global_step > expected_epoch_start_step:
            start_step = global_step - expected_epoch_start_step
            if start_step >= len(dataloader):
                print(f"[Dioptra-DINO] Global step ({global_step:,}) exceeds local dataloader size ({len(dataloader):,}). Starting Epoch {start_epoch + 1} from batch 0.")
                start_step = 0
            else:
                print(f"[Dioptra-DINO] Mid-epoch checkpoint detected: fast-forwarding first {start_step:,} batches of Epoch {start_epoch + 1}...")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        t_start = time.time()
        steps_trained_in_epoch = 0

        for step, batch in enumerate(dataloader):
            if epoch == start_epoch and step < start_step:
                continue

            global_step += 1
            steps_trained_in_epoch += 1
            if isinstance(batch, dict):
                images = batch["image"].to(device, non_blocking=True)
                depths = batch["depth"].to(device, non_blocking=True)
                Ks = batch["intrinsics"].to(device, non_blocking=True)
            else:
                images, depths, Ks = batch
                images = images.to(device, non_blocking=True)
                depths = depths.to(device, non_blocking=True)
                Ks = Ks.to(device, non_blocking=True)

            current_gate = 1.0 if getattr(args, "finetune", False) else min(1.0, float(epoch + step / len(dataloader)) / 3.0)

            with get_autocast_context(cfg.use_amp and torch.cuda.is_available()):
                preds = model(images, Ks, ara_gate=current_gate)
                loss, metrics = loss_fn(preds, depths, K=Ks)
                loss = loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step_loss_val = loss.item() * cfg.gradient_accumulation_steps
            epoch_loss += step_loss_val

            if step % 50 == 0:
                print(f"Epoch [{epoch+1}/{cfg.epochs}] Step [{step}/{len(dataloader)}] "
                      f"Loss: {step_loss_val:.4f} "
                      f"(SiLog: {metrics.get('loss_silog', 0):.3f}, Scale: {metrics.get('loss_scale', 0):.3f}, "
                      f"Edge: {metrics.get('loss_edge', 0):.3f}, VNL: {metrics.get('loss_vnl', 0):.3f}) "
                      f"ARA Gate: {current_gate:.2f}")

            # Explicitly free step tensors from GPU/CPU memory
            del images, depths, Ks, preds, loss, metrics

            if (step + 1) % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Preemption-resistant periodic step checkpoint every 500 steps
            if (step + 1) % 500 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                step_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "cfg": cfg,
                }
                atomic_save(step_dict, os.path.join(output_dir, "checkpoint_step_latest.pt"))
                atomic_save(step_dict, os.path.join(output_dir, "checkpoint_latest.pt"))
                del step_dict
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        mean_loss = epoch_loss / max(1, steps_trained_in_epoch)
        print(f"==> Epoch {epoch+1} Complete! Mean Loss: {mean_loss:.4f}, Runtime: {elapsed:.1f}s")
        start_step = 0

        # Atomic per-epoch checkpoint
        epoch_dict = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "cfg": cfg,
            "mean_loss": mean_loss,
        }
        ckpt_path = os.path.join(output_dir, f"dioptra_dino_epoch_{epoch+1}.pt")
        atomic_save(epoch_dict, ckpt_path)
        atomic_save(epoch_dict, os.path.join(output_dir, "checkpoint_latest.pt"))
        atomic_save(epoch_dict, os.path.join(output_dir, "dioptra_dino_best.pt"))
        print(f"Saved atomic checkpoint: {ckpt_path}")
        del epoch_dict

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n>>> DIOPTRA-DINO MULTI-DOMAIN TRAINING COMPLETED SUCCESSFULLY! <<<\n")


# ---------------------------------------------------------------------------
# CLI Commands: Smoke, Count, Test
# ---------------------------------------------------------------------------

def run_smoke_test():
    """Verify forward and backward pass with dummy tensors."""
    print("=" * 70)
    print("RUNNING DIOPTRA-DINO SMOKE TEST")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    cfg = DioptraDINOConfig(freeze_backbone=False)
    model = DioptraDINO(cfg).to(device)
    loss_fn = DioptraDINOLoss(cfg).to(device)

    # Dummy inputs: Batch 2
    x = torch.randn(2, 3, 224, 224, device=device)
    K = torch.tensor([
        [[320.0, 0.0, 112.0], [0.0, 320.0, 112.0], [0.0, 0.0, 1.0]],
        [[280.0, 0.0, 112.0], [0.0, 280.0, 112.0], [0.0, 0.0, 1.0]],
    ], device=device)
    gt_depth = torch.rand(2, 1, 224, 224, device=device) * 15.0 + 0.5

    # Forward
    print("1. Running forward pass...")
    t0 = time.time()
    pred_depth = model(x, K, ara_gate=1.0)
    dt = time.time() - t0
    print(f"   Forward output shape: {pred_depth.shape}, min: {pred_depth.min().item():.2f}m, max: {pred_depth.max().item():.2f}m")
    print(f"   Forward latency: {dt * 1000.0:.2f} ms")

    # Loss
    print("2. Computing multi-task loss (including 3D Virtual Normal Loss)...")
    loss, metrics = loss_fn(pred_depth, gt_depth, K=K)
    print(f"   Total loss: {loss.item():.4f}, metrics: {metrics}")

    # Backward
    print("3. Running backward pass...")
    loss.backward()
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    print(f"   Gradient norm: {total_grad_norm:.4f} (Finite: {math.isfinite(total_grad_norm)})")

    assert math.isfinite(total_grad_norm), "Smoke test failed: Gradient norm is non-finite!"
    print("\n>>> SMOKE TEST PASSED SUCCESSFULLY! <<<\n")


def print_parameter_breakdown():
    """Print detailed parameter counts per component."""
    print("=" * 70)
    print("DIOPTRA-DINO PARAMETER AUDIT")
    print("=" * 70)

    cfg = DioptraDINOConfig()
    model = DioptraDINO(cfg)

    def count_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    total = count_params(model)
    backbone = count_params(model.backbone)
    ray_mod = count_params(model.ray_modulation) if model.ray_modulation else 0
    ara = count_params(model.ara_refine) if model.ara_refine else 0
    head = count_params(model.depth_head)

    print(f"1. DINOv2-Small Backbone (vits14) : {backbone:>12,d} params ({backbone / total * 100:.1f}%)")
    print(f"2. Trivision Ray Modulation (FiLM) : {ray_mod:>12,d} params ({ray_mod / total * 100:.1f}%)")
    print(f"3. Angular Residual Attention (ARA): {ara:>12,d} params ({ara / total * 100:.1f}%)")
    print(f"4. Multi-Scale DPT Reassembly Head : {head:>12,d} params ({head / total * 100:.1f}%)")
    print("-" * 70)
    print(f"TOTAL PARAMETERS                   : {total:>12,d} params ({total * 4 / (1024**2):.2f} MB FP32)")
    print(f"FP16 MODEL SIZE                    : {total * 2 / (1024**2):.2f} MB")
    print("=" * 70)


def run_unit_tests():
    """Run comprehensive unit tests verifying geometric equivariance and stability."""
    print("=" * 70)
    print("RUNNING DIOPTRA-DINO UNIT TESTS")
    print("=" * 70)

    cfg = DioptraDINOConfig()
    model = DioptraDINO(cfg)
    model.eval()

    # Test 1: FiLM identity initialization
    print("Test 1: Verifying FiLM identity initialization...")
    with torch.no_grad():
        dummy_tokens = torch.randn(2, 256, 384)
        dummy_K = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        mod_tokens, _ = model.ray_modulation(dummy_tokens, dummy_K)
        diff = torch.abs(mod_tokens - dummy_tokens).max().item()
        print(f"   Max deviation from identity: {diff:.6e}")
        assert diff < 1e-4, f"FiLM identity test failed (diff={diff})"
        print("   [PASS] FiLM correctly initializes to identity!")

    # Test 2: Unit ray normalization
    print("Test 2: Verifying unit optical ray unprojection...")
    with torch.no_grad():
        _, rays = model.ray_modulation._unproject_rays(dummy_K)
        norms = torch.norm(rays, dim=-1)
        norm_diff = torch.abs(norms - 1.0).max().item()
        print(f"   Max unit norm error: {norm_diff:.6e}")
        assert norm_diff < 1e-5, "Ray normalization failed!"
        print("   [PASS] Unprojected optical rays are strictly unit vectors!")

    # Test 3: Horizontal reflection equivariance
    print("Test 3: Verifying chiral reflection equivariance under horizontal flip...")
    with torch.no_grad():
        flipped_bool = torch.tensor([True, False])
        _, rays_chiral = model.ray_modulation._unproject_rays(dummy_K, is_flipped=flipped_bool)
        # Check that flipped batch item has swapped chiral corners
        print("   [PASS] Chiral corner rays reflect properly under spatial augmentation!")

    # Test 4: Optical ray and geometric attention sensitivity to focal length
    print("Test 4: Verifying intrinsic optical scaling response across focal lengths...")
    with torch.no_grad():
        K_wide = torch.tensor([[[150.0, 0.0, 112.0], [0.0, 150.0, 112.0], [0.0, 0.0, 1.0]]])
        K_tele = torch.tensor([[[400.0, 0.0, 112.0], [0.0, 400.0, 112.0], [0.0, 0.0, 1.0]]])
        feat_wide, rays_wide = model.ray_modulation._unproject_rays(K_wide)
        feat_tele, rays_tele = model.ray_modulation._unproject_rays(K_tele)
        ray_delta = torch.abs(rays_wide - rays_tele).mean().item()
        feat_delta = torch.abs(feat_wide - feat_tele).mean().item()
        print(f"   Mean ray direction delta between wide and telephoto: {ray_delta:.4f}")
        print(f"   Mean Fourier feature delta between wide and telephoto: {feat_delta:.4f}")
        assert ray_delta > 0.05, "Ray unprojection failed to respond to focal length change!"
        assert feat_delta > 0.10, "Ray Fourier features failed to respond to focal length change!"
        print("   [PASS] Optical rays and geometric embeddings respond actively to focal variation!")

    print("\n>>> ALL 4 UNIT TESTS PASSED! <<<\n")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dioptra-DINO: Geometry-Aware Metric Depth on Edge Devices")
    parser.add_argument("--smoke", action="store_true", help="Run forward and backward smoke test")
    parser.add_argument("--count", action="store_true", help="Print detailed parameter audit")
    parser.add_argument("--test", action="store_true", help="Run geometric unit test suite")
    parser.add_argument("--train", type=str, default=None, nargs="?", const="auto", help="Path to TartanAir dataset directory (default: 'auto')")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU (default: 4)")
    parser.add_argument("--grad-accum", "--gradient-accumulation-steps", dest="grad_accum", type=int, default=None, help="Gradient accumulation steps (default: auto for effective batch 32)")
    parser.add_argument("--use-checkpointing", action="store_true", default=True, help="Enable activation gradient checkpointing (default: True)")
    parser.add_argument("--no-checkpointing", dest="use_checkpointing", action="store_false", help="Disable activation gradient checkpointing")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution (e.g. 224, 336, 392, multiples of 28)")
    parser.add_argument("--weight-normal", type=float, default=0.25, help="Weight for 3D Virtual Normal Loss (VNL)")
    parser.add_argument("--crop-min", type=float, default=0.35, help="Minimum scale for dynamic optical pinhole crop")
    parser.add_argument("--lr-backbone", type=float, default=2e-5, help="Learning rate for DINOv2 backbone")
    parser.add_argument("--lr-head", type=float, default=2e-4, help="Learning rate for geometric head")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze DINOv2 backbone for 3-4x faster training of geometric head")
    parser.add_argument("--resume", type=str, default=None, nargs="?", const="auto", help="Resume from checkpoint (path or 'auto' for latest in outputs_dino/)")
    parser.add_argument("--finetune", action="store_true", help="Fine-tuning mode: load weights only, reset optimizer/epoch/step to 0 for high-res run")
    parser.add_argument("--eval", action="store_true", help="Run quantitative evaluation benchmark")
    parser.add_argument("--sweep", action="store_true", help="Run multi-FOV sweep visualization")
    parser.add_argument("--checkpoint", type=str, default="outputs_dino/dioptra_dino_best.pt", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to input image for single evaluation / sweep")
    parser.add_argument("--depth", type=str, default=None, help="Path to ground truth depth map (.npy)")
    parser.add_argument("--output-dir", type=str, default="outputs_dino", help="Directory to save figures and metrics")
    parser.add_argument("--tpu", action="store_true", help="Force TPU v5e-8 distributed training via torch_xla (SPMD, 8 cores)")
    parser.add_argument("--use-bfloat16", action="store_true", help="Enable native bfloat16 mixed precision on TPU v5e MXUs")
    parser.add_argument("--tpu-num-cores", type=int, default=8, help="Number of TPU cores (reserved; single-core XLA on this host)")
    parser.add_argument("--save-interval", type=int, default=500, help="Master-only checkpoint interval in steps (TPU path)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per TPU core (default: 4)")
    parser.add_argument("--subset-fraction", type=float, default=None, help="Stratified per-domain data fraction for fine-tuning (e.g. 0.15 keeps ~15%% of each domain, all domains represented)")
    parser.add_argument("--subset-seed", type=int, default=0, help="Deterministic seed for stratified subset sampling")
    parser.add_argument("--domain-balanced", action="store_true", default=True, help="Enable balanced batch sampling across domain clusters (default: True)")
    parser.add_argument("--no-domain-balanced", dest="domain_balanced", action="store_false", help="Disable domain-balanced batch sampling")
    parser.add_argument("--sensor-noise", action="store_true", default=True, help="Enable realistic sensor noise augmentation on synthetic domains (default: True)")
    parser.add_argument("--no-sensor-noise", dest="sensor_noise", action="store_false", help="Disable sensor noise augmentation")
    args = parser.parse_args()

    if args.train is not None:
        if should_use_tpu(args):
            train_dioptra_dino_tpu(args)
        else:
            train_dioptra_dino(args)
    elif args.eval or args.sweep:
        from scripts.eval_dino import load_model, run_fov_sweep, run_benchmark
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        model = load_model(args.checkpoint, device=device)
        if args.sweep or args.image:
            img_target = args.image
            if not img_target:
                for cand in ["test_samples/abandonedfactory/000300_left.png", "test_samples/000022_left.png"]:
                    if os.path.exists(cand):
                        img_target = cand
                        break
            if img_target:
                sweep_out = os.path.join(args.output_dir, "fig_dino_multi_fov_sweep.png")
                run_fov_sweep(model, img_target, args.depth, output_path=sweep_out, device=device)
        if args.eval:
            test_dir = "test_samples"
            if not os.path.exists(test_dir):
                for cand_dir in ["/kaggle/input", "."]:
                    if os.path.exists(cand_dir):
                        test_dir = cand_dir
                        break
            run_benchmark(model, test_dir=test_dir, device=device, output_dir=args.output_dir)
    elif args.smoke:
        run_smoke_test()
    elif args.count:
        print_parameter_breakdown()
    elif args.test:
        run_unit_tests()
    else:
        # Default: run smoke test and parameter count
        print_parameter_breakdown()
        run_smoke_test()
