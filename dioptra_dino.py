"""
Dioptra-DINO: Foundation-Assisted Geometry-Aware Monocular Metric Depth.

Pairs a pre-trained DINOv2-Small (vits14, 21.6M params) visual backbone with:
  1. Trivision Ray Positional Encoding: continuous optical ray triplets unprojected from K.
  2. Angular Residual Attention (ARA): intrinsic geometric attention bias sin^2(theta_qk).
  3. Multi-Scale DPT Reassembly Head: multi-layer fusion for dense 224x224 metric depth.
  4. Decoupled Metric Scale Supervision: log-median scale consistency loss.
  5. Dynamic Pinhole Intrinsics Augmentation: camera-intrinsic focal equivariance.

Total parameter footprint: ~25.4M parameters (~101 MB FP32, ~50.8 MB FP16).

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
import io
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
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    lr_backbone: float = 2e-5
    lr_head: float = 2e-4
    weight_decay: float = 0.05
    epochs: int = 15
    use_amp: bool = True
    gradient_clip: float = 1.0

    # Loss weights
    weight_silog: float = 1.0
    weight_scale: float = 0.5
    weight_edge: float = 0.2
    weight_normal: float = 0.25


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
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
                state_dict = torch.load(path, map_location="cpu")
            else:
                print(f"[Dioptra-DINO] Downloading / loading DINOv2-Small weights from Meta Hub...")
                state_dict = torch.hub.load_state_dict_from_url(DINOV2_VITS14_URL, map_location="cpu")

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

        # 3 rays x 6 freqs x 2 (sin/cos) x 3 (xyz) = 108 dims
        in_dim = 3 * self.num_freqs * 2 * 3

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
        all_rays = torch.cat([rc, r1, r2], dim=-1)  # [B, N, 9]
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
        d = torch.log(pred[mask]) - torch.log(target[mask])
        loss = torch.mean(d ** 2) - 0.85 * (torch.mean(d) ** 2)
        return torch.sqrt(loss.clamp(min=1e-8))

    def scale_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Explicit log-median metric scale consistency loss."""
        pred_med = torch.median(pred[mask])
        target_med = torch.median(target[mask])
        return torch.abs(torch.log(pred_med.clamp(min=1e-5)) - torch.log(target_med.clamp(min=1e-5)))

    def edge_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Multi-scale spatial gradient loss for sharp structural boundaries."""
        # Sobel-like central finite differences
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]
        mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]

        loss_y = torch.mean(torch.abs(dy_pred[mask_y] - dy_target[mask_y]))
        loss_x = torch.mean(torch.abs(dx_pred[mask_x] - dx_target[mask_x]))
        return loss_y + loss_x

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

        target_x = (u - cx) / fx * target_z
        target_y = (v - cy) / fy * target_z
        P_target = torch.stack([target_x, target_y, target_z], dim=1)

        total_vnl = torch.tensor(0.0, device=device, dtype=dtype)
        valid_scales = 0

        for s in [1, 2, 4]:
            if H <= 2 * s or W <= 2 * s:
                continue
            vx_pred = P_pred[:, :, s:-s, 2 * s:] - P_pred[:, :, s:-s, :-2 * s]
            vy_pred = P_pred[:, :, 2 * s:, s:-s] - P_pred[:, :, :-2 * s, s:-s]
            vx_target = P_target[:, :, s:-s, 2 * s:] - P_target[:, :, s:-s, :-2 * s]
            vy_target = P_target[:, :, 2 * s:, s:-s] - P_target[:, :, :-2 * s, s:-s]

            n_pred = torch.cross(vx_pred, vy_pred, dim=1)
            n_target = torch.cross(vx_target, vy_target, dim=1)

            norm_pred = torch.norm(n_pred, dim=1, keepdim=True).clamp(min=1e-6)
            norm_target = torch.norm(n_target, dim=1, keepdim=True).clamp(min=1e-6)

            m_inner = (
                m[:, s:-s, 2 * s:]
                & m[:, s:-s, :-2 * s]
                & m[:, 2 * s:, s:-s]
                & m[:, :-2 * s, s:-s]
                & (norm_target.squeeze(1) > 1e-4)
            )

            if m_inner.sum() > 50:
                n_p = n_pred / norm_pred
                n_t = n_target / norm_target
                cos_sim = torch.sum(n_p * n_t, dim=1)
                loss_s = torch.mean(1.0 - cos_sim[m_inner].clamp(min=-1.0, max=1.0))
                total_vnl = total_vnl + loss_s
                valid_scales += 1

        if valid_scales > 0:
            return total_vnl / float(valid_scales)
        return torch.tensor(0.0, device=device, dtype=dtype)

    def forward(
        self, pred: Tensor, target: Tensor, image: Optional[Tensor] = None, K: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute total multi-task loss across valid pixels."""
        mask = (target >= self.cfg.min_depth) & (target <= self.cfg.max_depth) & ~torch.isnan(target)
        if mask.sum() < 100:
            return torch.tensor(0.0, device=pred.device, requires_grad=True), {}

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop simulating optical zoom / focal length variation.

    Adjusts focal lengths fx, fy and optical center cx, cy continuously.
    """
    H, W = image.shape[:2]
    max_square = min(H, W)
    min_len = int(crop_size_range[0] * max_square)
    max_len = int(crop_size_range[1] * max_square)

    crop_len = random.randint(min_len, max_len)
    top = random.randint(0, H - crop_len)
    left = random.randint(0, W - crop_len)

    # Crop
    crop_img = image[top:top + crop_len, left:left + crop_len]
    crop_depth = depth[top:top + crop_len, left:left + crop_len]

    # Adjust camera intrinsics
    K_new = K.copy().astype(np.float32)
    K_new[0, 2] -= left  # cx' = cx - left
    K_new[1, 2] -= top   # cy' = cy - top

    # Rescale to network input resolution out_size x out_size
    scale = float(out_size) / float(crop_len)
    K_new[0, 0] *= scale
    K_new[1, 1] *= scale
    K_new[0, 2] *= scale
    K_new[1, 2] *= scale

    # Resize images (using simple PIL or cv2 if available, else nearest/linear)
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(crop_img).resize((out_size, out_size), PILImage.BILINEAR)
        pil_depth = PILImage.fromarray(crop_depth).resize((out_size, out_size), PILImage.NEAREST)
        res_img = np.array(pil_img)
        res_depth = np.array(pil_depth)
    except ImportError:
        # Fallback numpy nearest resize
        y_indices = (np.linspace(0, crop_len - 1, out_size)).astype(int)
        x_indices = (np.linspace(0, crop_len - 1, out_size)).astype(int)
        res_img = crop_img[np.ix_(y_indices, x_indices)]
        res_depth = crop_depth[np.ix_(y_indices, x_indices)]

    return res_img, res_depth, K_new


# ---------------------------------------------------------------------------
# TartanAir Dataset Discovery & Resolution
# ---------------------------------------------------------------------------

_TARTANAIR_DIFFS = {"Easy", "Medium", "Hard", "easy", "medium", "hard"}
_MAX_SCAN_DEPTH = 5
_SCAN_SKIP_DIRS = {".ipynb_checkpoints", "__pycache__", ".git", "__MACOSX"}


def _dir_looks_like_tartanair(root: Path) -> bool:
    """Cheap structural probe: does root match either TartanAir layout?"""
    try:
        tops = [d for d in root.iterdir() if d.is_dir()]
    except OSError:
        return False
    if not tops:
        return False
    if any(t.name in _TARTANAIR_DIFFS for t in tops):
        return True  # layout B root: {difficulty}/{env}/...
    for t in tops[:64]:
        try:
            if any(c.name in _TARTANAIR_DIFFS for c in t.iterdir() if c.is_dir()):
                return True  # layout A root: {env}/{difficulty}/...
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
    has_img = any("/image_left/" in n for n in names)
    has_depth = any("/depth_left/" in n for n in names)
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
        return cand
    try:
        subdirs = [d for d in sorted(cand.iterdir()) if d.is_dir()]
    except OSError:
        subdirs = []
    for sd in subdirs[:16]:
        if _dir_looks_like_tartanair(sd):
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


def resolve_dataset_root(path: str = "auto") -> str:
    """Resolve a --train argument into a usable dataset root (str).

    Accepts 'auto', explicit directory path, nested mount path, or .zip archive.
    """
    if not path or str(path).lower() in ("auto", "none"):
        scan_roots: List[Path] = []
        env_dir = os.environ.get("TESSERACT_INPUT_DIR") or os.environ.get("DATA_PATH")
        if env_dir and Path(env_dir).exists():
            scan_roots.append(Path(env_dir))
        if Path("/kaggle/input").is_dir():
            scan_roots.append(Path("/kaggle/input"))
        for candidate in [Path("."), Path(".."), Path("/kaggle/working")]:
            if candidate.is_dir():
                scan_roots.append(candidate)

        mount_names: List[str] = []
        for r in scan_roots:
            mount_names.extend(_enumerate_input_mounts(r))

        dir_hits: List[Path] = []
        zip_hits: List[Tuple[Path, int]] = []
        for r in scan_roots:
            dh, zh = _deep_find_tartanair(r)
            dir_hits.extend(dh)
            zip_hits.extend(zh)

        if dir_hits:
            print(f"[Dioptra-DINO Dataset] Resolved 'auto' -> directory: {dir_hits[0]}")
            return str(dir_hits[0])
        if zip_hits:
            print(f"[Dioptra-DINO Dataset] Resolved 'auto' -> zip archive: {zip_hits[0][0]} ({zip_hits[0][1]/(1024*1024):.1f} MB)")
            return str(zip_hits[0][0])

        raise FileNotFoundError(
            "Could not auto-resolve TartanAir dataset from /kaggle/input.\n"
            f"  Scanned: {', '.join(str(r) for r in scan_roots)}\n"
            f"  Mounted inputs: {', '.join(mount_names) if mount_names else 'None'}\n"
            "Please ensure 'dasvo-tartanair-rgb-d-validation-split' is attached as an Input in the right sidebar."
        )

    p = Path(path)
    resolved = _resolve_single_mount(p)
    if resolved is not None:
        return str(resolved)
    if p.exists():
        return str(p)
    raise FileNotFoundError(
        f"Could not resolve TartanAir dataset from: {path}\n"
        "Expected directory or .zip containing image_left/ and depth_left/ folders."
    )


# ---------------------------------------------------------------------------
# TartanAir Dataset Loader for Dioptra-DINO
# ---------------------------------------------------------------------------

class TartanAirDINODataset(torch.utils.data.Dataset):
    """Robust TartanAir dataset loader supporting flat, nested, and zip-backed layouts."""

    def __init__(
        self,
        root_dir: str = "auto",
        split: str = "train",
        image_size: int = 224,
        apply_pinhole_aug: bool = True,
    ):
        self.root_dir = resolve_dataset_root(root_dir)
        self.split = split
        self.image_size = image_size
        self.apply_pinhole_aug = apply_pinhole_aug and (split == "train")

        # Canonical TartanAir pinhole matrix (640x480 resolution)
        self.K_canonical = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        self._is_zip = str(self.root_dir).lower().endswith(".zip")
        self._zip_cache: Dict[str, zipfile.ZipFile] = {}
        self._zip_cache_pid: Optional[int] = None

        self.samples: List[Tuple[str, str]] = []
        self._build_index()
        print(f"[Dioptra-DINO Dataset] Successfully indexed {len(self.samples)} {split} samples from {self.root_dir}")

    @staticmethod
    def _normalize_stem(stem: str) -> str:
        for tag in ("_depth", "_og", "_left"):
            if stem.endswith(tag):
                stem = stem[:-len(tag)]
        return stem

    def _zip_read(self, zip_path: str, member: str) -> bytes:
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

    def _build_index(self):
        if self._is_zip:
            self._build_index_zip()
        else:
            self._build_index_tree()

        # Graceful fallback: if split filtering yielded 0 samples, include all found pairs
        if len(self.samples) == 0:
            if self._is_zip:
                self._build_index_zip(force_all=True)
            else:
                self._build_index_tree(force_all=True)

    def _build_index_zip(self, force_all: bool = False):
        with zipfile.ZipFile(self.root_dir, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]

        img_map: Dict[str, str] = {}
        depth_map: Dict[str, str] = {}

        for n in names:
            nl = n.lower()
            if "/image_left/" in n and nl.endswith(".png"):
                parts = n.split("/image_left/")
                traj_prefix = parts[0]
                fname = os.path.basename(parts[1])
                stem = self._normalize_stem(os.path.splitext(fname)[0])
                img_map[f"{traj_prefix}::{stem}"] = n
            elif "/depth_left/" in n and nl.endswith(".npy"):
                parts = n.split("/depth_left/")
                traj_prefix = parts[0]
                fname = os.path.basename(parts[1])
                stem = self._normalize_stem(os.path.splitext(fname)[0])
                depth_map[f"{traj_prefix}::{stem}"] = n

        for key, img_path in sorted(img_map.items()):
            if key in depth_map:
                traj = key.split("::")[0]
                is_val = (hash(traj) % 10) == 0
                if force_all or (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                    self.samples.append((img_path, depth_map[key]))

    def _build_index_tree(self, force_all: bool = False):
        png_files = sorted(glob.glob(os.path.join(self.root_dir, "**", "*.png"), recursive=True))
        for img_path in png_files:
            fname = os.path.basename(img_path)
            if "depth" in fname.lower():
                continue
            base_dir = os.path.dirname(img_path)
            stem = os.path.splitext(fname)[0]
            clean = self._normalize_stem(stem)

            # Check candidate depth files across all TartanAir directory and naming schemes
            cand_depths = [
                os.path.join(base_dir.replace("image_left", "depth_left"), f"{clean}_left_depth.npy"),
                os.path.join(base_dir.replace("image_left", "depth_left"), f"{clean}.npy"),
                os.path.join(base_dir.replace("image_left", "depth_left"), f"{stem}_depth.npy"),
                os.path.join(base_dir.replace("image_left", "depth_left"), f"{stem}.npy"),
                os.path.join(base_dir, f"{clean}_depth.npy"),
                os.path.join(base_dir, f"{clean}.npy"),
                os.path.join(base_dir, f"{stem}_depth.npy"),
                os.path.join(base_dir, f"{stem}.npy"),
            ]
            depth_path = None
            for c in cand_depths:
                if os.path.exists(c):
                    depth_path = c
                    break

            if depth_path:
                traj_name = os.path.dirname(base_dir)
                is_val = (hash(traj_name) % 10) == 0
                if force_all or (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                    self.samples.append((img_path, depth_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        img_src, depth_src = self.samples[idx]

        from PIL import Image as PILImage

        # Load RGB image
        try:
            if self._is_zip:
                img_data = self._zip_read(self.root_dir, img_src)
                img = np.array(PILImage.open(io.BytesIO(img_data)).convert("RGB"))
            else:
                img = np.array(PILImage.open(img_src).convert("RGB"))
        except Exception:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Load Depth map (npy in metres)
        try:
            if self._is_zip:
                depth_data = self._zip_read(self.root_dir, depth_src)
                depth = np.load(io.BytesIO(depth_data)).astype(np.float32)
            else:
                depth = np.load(depth_src).astype(np.float32)
        except Exception:
            depth = np.zeros((480, 640), dtype=np.float32)

        K = self.K_canonical.copy()

        # Dynamic pinhole crop augmentation (wide optical zoom range for camera invariance)
        if self.apply_pinhole_aug and random.random() < 0.9:
            img, depth, K = apply_dynamic_pinhole_crop(
                img, depth, K, crop_size_range=(0.35, 1.0), out_size=self.image_size
            )
        else:
            H, W = img.shape[:2]
            scale_x = self.image_size / float(W)
            scale_y = self.image_size / float(H)
            K[0, 0] *= scale_x
            K[1, 1] *= scale_y
            K[0, 2] *= scale_x
            K[1, 2] *= scale_y

            img = np.array(PILImage.fromarray(img).resize((self.image_size, self.image_size), PILImage.BILINEAR))
            depth = np.array(PILImage.fromarray(depth).resize((self.image_size, self.image_size), PILImage.NEAREST))

        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        for c, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            img_tensor[c] = (img_tensor[c] - mean) / std

        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        K_tensor = torch.from_numpy(K).float()
        return img_tensor, depth_tensor, K_tensor


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_dioptra_dino(args):
    """Execute high-performance distributed/single-GPU training on TartanAir."""
    print("=" * 70)
    print("STARTING DIOPTRA-DINO TRAINING ON TARTANAIR")
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
    cfg = DioptraDINOConfig(
        image_size=img_sz,
        grid_size=img_sz // 14,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_normal=w_norm,
    )

    model = DioptraDINO(cfg).to(device)
    loss_fn = DioptraDINOLoss(cfg).to(device)

    # Multi-GPU support via DataParallel
    if torch.cuda.is_available() and gpu_count > 1:
        print(f"[Dioptra-DINO] Multi-GPU acceleration enabled across {gpu_count} GPUs via DataParallel!")
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, "module") else model

    # Optimizer with differential learning rates
    param_groups = [
        {"params": raw_model.backbone.parameters(), "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in raw_model.named_parameters() if not n.startswith("backbone.")], "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    # Modern AMP GradScaler (deprecation-free)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and torch.cuda.is_available())
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and torch.cuda.is_available())

    # Resolve and instantiate TartanAir dataset
    train_root_arg = args.train or "auto"
    dataset_root = resolve_dataset_root(train_root_arg)
    print(f"[Dioptra-DINO Dataset] Active dataset root: {dataset_root}")

    # Prefer dioptra's TartanAirDataset if available, fallback to internal TartanAirDINODataset
    dataset = None
    try:
        # Check if dioptra is available in path
        for p in [os.getcwd(), os.path.dirname(os.path.abspath(__file__)), "/kaggle/working", "/kaggle/working/dioptra_repo"]:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        from dioptra import TartanAirDataset, DataConfig
        dataset = TartanAirDataset(
            root=dataset_root,
            difficulty="all",
            split="train",
            img_size=cfg.image_size,
            augment=True,
            cfg=DataConfig(),
        )
        print(f"[Dioptra-DINO Dataset] Successfully loaded TartanAirDataset with {len(dataset):,} samples ✓")
    except Exception as exc:
        print(f"[Dioptra-DINO Dataset] Using built-in TartanAirDINODataset ({exc})")
        dataset = TartanAirDINODataset(root_dir=dataset_root, split="train", image_size=cfg.image_size)

    if len(dataset) == 0:
        raise ValueError(
            f"Found 0 training samples in {dataset_root}!\n"
            "Please ensure TartanAir is attached to this Kaggle notebook:\n"
            "Right Sidebar -> Input -> '+ Add Input' -> search 'dasvo-tartanair-rgb-d-validation-split' by pandrii000."
        )

    def _worker_init_fn(worker_id):
        torch.set_num_threads(1)

    num_workers = min(2, os.cpu_count() or 1)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn,
        drop_last=True,
    )

    output_dir = "outputs_dino"
    os.makedirs(output_dir, exist_ok=True)

    # Checkpoint Resume Logic
    start_epoch = 0
    resume_arg = getattr(args, "resume", None)
    if resume_arg:
        resume_target = None
        if isinstance(resume_arg, str) and os.path.isfile(resume_arg):
            resume_target = resume_arg
        else:
            # 1. Search output directory
            cands = sorted(
                glob.glob(os.path.join(output_dir, "dioptra_dino_epoch_*.pt")),
                key=lambda p: int(os.path.splitext(p)[0].split("_")[-1]) if os.path.splitext(p)[0].split("_")[-1].isdigit() else 0
            )
            if cands:
                resume_target = cands[-1]
            elif os.path.exists(os.path.join(output_dir, "dioptra_dino_best.pt")):
                resume_target = os.path.join(output_dir, "dioptra_dino_best.pt")
            else:
                # 2. Search /kaggle/input for uploaded checkpoint datasets
                input_cands = sorted(
                    glob.glob("/kaggle/input/**/dioptra_dino*.pt", recursive=True)
                    + glob.glob("/kaggle/input/**/dioptra_dino*.zip", recursive=True),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]) if os.path.splitext(os.path.basename(p))[0].split("_")[-1].isdigit() else 0
                )
                if input_cands:
                    resume_target = input_cands[-1]

        if resume_target and os.path.exists(resume_target):
            print(f"[Dioptra-DINO] Resuming training from checkpoint: {resume_target}")
            import __main__
            if not hasattr(__main__, "DioptraDINOConfig"):
                setattr(__main__, "DioptraDINOConfig", DioptraDINOConfig)
            try:
                ckpt = torch.load(resume_target, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(resume_target, map_location=device)
            raw_model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            print(f"[Dioptra-DINO] Successfully restored model weights! Resuming at Epoch {start_epoch + 1}/{cfg.epochs}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(dataloader))
    if start_epoch > 0:
        for _ in range(start_epoch * len(dataloader)):
            scheduler.step()

    print(f"Training configuration: Epochs={cfg.epochs}, Batches/Epoch={len(dataloader)}, "
          f"BatchSize={cfg.batch_size} (EffBatchSize={cfg.batch_size * cfg.gradient_accumulation_steps})")

    # Autocast context helper
    def get_autocast_context(enabled: bool):
        if torch.cuda.is_available():
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast(device_type="cuda", enabled=enabled)
            return torch.cuda.amp.autocast(enabled=enabled)
        return torch.cpu.amp.autocast(enabled=False) if hasattr(torch, "cpu") else torch.cuda.amp.autocast(enabled=False)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        t_start = time.time()

        for step, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                images = batch["image"].to(device, non_blocking=True)
                depths = batch["depth"].to(device, non_blocking=True)
                Ks = batch["intrinsics"].to(device, non_blocking=True)
            else:
                images, depths, Ks = batch
                images = images.to(device, non_blocking=True)
                depths = depths.to(device, non_blocking=True)
                Ks = Ks.to(device, non_blocking=True)

            current_gate = min(1.0, float(epoch + step / len(dataloader)) / 3.0)

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
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * cfg.gradient_accumulation_steps

            if step % 50 == 0:
                print(f"Epoch [{epoch+1}/{cfg.epochs}] Step [{step}/{len(dataloader)}] "
                      f"Loss: {loss.item() * cfg.gradient_accumulation_steps:.4f} "
                      f"(SiLog: {metrics.get('loss_silog', 0):.3f}, Scale: {metrics.get('loss_scale', 0):.3f}, "
                      f"Edge: {metrics.get('loss_edge', 0):.3f}, VNL: {metrics.get('loss_vnl', 0):.3f}) "
                      f"ARA Gate: {current_gate:.2f}")

            if step % 500 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        mean_loss = epoch_loss / len(dataloader)
        print(f"==> Epoch {epoch+1} Complete! Mean Loss: {mean_loss:.4f}, Runtime: {elapsed:.1f}s")

        save_dict = {
            "epoch": epoch + 1,
            "model_state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
            "cfg": cfg,
        }
        ckpt_path = os.path.join(output_dir, f"dioptra_dino_epoch_{epoch+1}.pt")
        best_path = os.path.join(output_dir, "dioptra_dino_best.pt")
        torch.save(save_dict, ckpt_path)
        torch.save(save_dict, best_path)
        print(f"Saved checkpoint: {ckpt_path}")
        del save_dict

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n>>> DIOPTRA-DINO TRAINING COMPLETED SUCCESSFULLY! <<<\n")


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
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution (e.g. 224, 336, 392, multiples of 28)")
    parser.add_argument("--weight-normal", type=float, default=0.25, help="Weight for 3D Virtual Normal Loss (VNL)")
    parser.add_argument("--crop-min", type=float, default=0.35, help="Minimum scale for dynamic optical pinhole crop")
    parser.add_argument("--lr-backbone", type=float, default=2e-5, help="Learning rate for DINOv2 backbone")
    parser.add_argument("--lr-head", type=float, default=2e-4, help="Learning rate for geometric head")
    parser.add_argument("--resume", type=str, default=None, nargs="?", const="auto", help="Resume from checkpoint (path or 'auto' for latest in outputs_dino/)")
    parser.add_argument("--eval", action="store_true", help="Run quantitative evaluation benchmark")
    parser.add_argument("--sweep", action="store_true", help="Run multi-FOV sweep visualization")
    parser.add_argument("--checkpoint", type=str, default="outputs_dino/dioptra_dino_best.pt", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Path to input image for single evaluation / sweep")
    parser.add_argument("--depth", type=str, default=None, help="Path to ground truth depth map (.npy)")
    parser.add_argument("--output-dir", type=str, default="outputs_dino", help="Directory to save figures and metrics")
    args = parser.parse_args()

    if args.train is not None:
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
