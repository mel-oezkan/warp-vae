"""RoMA-based region comparison utilities for latent consistency analysis."""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add RoMA2 to path
ROMA_PATH = Path(__file__).resolve().parents[2] / "RoMA2" / "src"
if str(ROMA_PATH) not in sys.path:
    sys.path.insert(0, str(ROMA_PATH))

from romav2 import RoMaV2


def load_roma_model(
    setting: str = "precise",
    device: str = "cuda",
    compile: bool = False
) -> RoMaV2:
    """Load RoMaV2 model with specified settings.

    Args:
        setting: "precise", "fast", "turbo", or "base"
        device: CUDA device string
        compile: Whether to use torch.compile (disable for older GPUs)

    Returns:
        Initialized RoMaV2 model in eval mode
    """
    cfg = RoMaV2.Cfg(compile=compile, setting=setting)
    model = RoMaV2(cfg=cfg).to(device)
    model.eval()
    print(f"Loaded RoMaV2 model with setting='{setting}'")
    return model


def warp_to_latent_warp(
    warp: torch.Tensor,
    image_resolution: int = 256,
    latent_resolution: int = 32
) -> torch.Tensor:
    """Downsample warp field from image resolution to latent resolution.

    The warp field contains normalized [-1, 1] coordinates which remain
    valid after downsampling since both spaces use the same normalization.

    Args:
        warp: Dense warp field (1, H, W, 2) in normalized coords
        image_resolution: Current warp resolution (typically matches image)
        latent_resolution: Target latent resolution (32 for 8x downsampling)

    Returns:
        Downsampled warp field (1, latent_H, latent_W, 2)
    """
    # warp is (1, H, W, 2), need (1, 2, H, W) for interpolate
    warp_bhwc = warp.permute(0, 3, 1, 2)  # (1, 2, H, W)
    warp_small = F.interpolate(
        warp_bhwc,
        size=(latent_resolution, latent_resolution),
        mode='bilinear',
        align_corners=False
    )
    return warp_small.permute(0, 2, 3, 1)  # (1, lat_H, lat_W, 2)


def confidence_to_latent_mask(
    confidence: torch.Tensor,
    threshold: float = 0.8,
    image_resolution: int = 256,
    latent_resolution: int = 32
) -> torch.Tensor:
    """Downsample confidence map to latent resolution using min-pooling.

    A latent cell is only marked valid if ALL corresponding image pixels
    are valid (confidence > threshold). This is conservative but ensures
    reliable region comparisons.

    Args:
        confidence: Confidence map (1, H, W, 1) or (1, H, W) in [0, 1]
        threshold: Minimum confidence for valid correspondences
        image_resolution: Current confidence resolution
        latent_resolution: Target latent resolution

    Returns:
        Boolean mask (1, lat_H, lat_W) where True = valid region
    """
    # Ensure shape is (1, H, W)
    if confidence.dim() == 4:
        confidence = confidence.squeeze(-1)

    # Binary threshold
    valid = (confidence > threshold).float()  # (1, H, W)

    # Min-pool using -max(-x) = min(x)
    pool_size = image_resolution // latent_resolution  # typically 8
    valid = valid.unsqueeze(1)  # (1, 1, H, W)
    valid_pooled = -F.max_pool2d(-valid, kernel_size=pool_size)
    valid_pooled = valid_pooled.squeeze(1)  # (1, lat_H, lat_W)

    return valid_pooled > 0.5  # Boolean mask


def compute_roma_correspondences(
    roma_model: RoMaV2,
    img_a: Union[Image.Image, torch.Tensor],
    img_b: Union[Image.Image, torch.Tensor],
    confidence_threshold: float = 0.8,
    latent_resolution: int = 32
) -> Dict[str, torch.Tensor]:
    """Compute RoMA correspondences between two images.

    Args:
        roma_model: Loaded RoMaV2 model
        img_a, img_b: Input images (PIL Images or tensors in [0,1])
        confidence_threshold: Minimum confidence for valid correspondences
        latent_resolution: Target latent resolution for downsampled outputs

    Returns:
        Dictionary with:
        - warp_ab: (1, H, W, 2) warp from A to B at original resolution
        - warp_ba: (1, H, W, 2) warp from B to A
        - overlap_ab: (1, H, W, 1) confidence A->B
        - overlap_ba: (1, H, W, 1) confidence B->A
        - warp_ab_latent: (1, lat_H, lat_W, 2) downsampled warp
        - valid_mask_ab: (1, lat_H, lat_W) boolean mask at latent resolution
        - valid_mask_ba: (1, lat_H, lat_W) boolean mask
        - valid_fraction_ab: fraction of latent cells with valid correspondences
        - valid_fraction_ba: fraction for reverse direction
    """
    with torch.no_grad():
        pred_ab = roma_model.match(img_a, img_b)
        pred_ba = roma_model.match(img_b, img_a)

    # Extract outputs
    warp_ab = pred_ab["warp_AB"]  # (1, H, W, 2)
    warp_ba = pred_ba["warp_AB"]  # (1, H, W, 2)

    # Get overlap confidence
    if pred_ab["overlap_AB"] is not None:
        overlap_ab = pred_ab["overlap_AB"]  # (1, H, W, 1)
    else:
        # Fallback: use mean of confidence channels
        overlap_ab = pred_ab["confidence_AB"].mean(dim=-1, keepdim=True)

    if pred_ba["overlap_AB"] is not None:
        overlap_ba = pred_ba["overlap_AB"]
    else:
        overlap_ba = pred_ba["confidence_AB"].mean(dim=-1, keepdim=True)

    # Get image resolution from warp
    image_resolution = warp_ab.shape[1]  # H dimension

    # Compute in-bounds mask (warp coordinates within [-1, 1])
    in_bounds_ab = (warp_ab.abs() <= 1.0).all(dim=-1, keepdim=True).float()
    in_bounds_ba = (warp_ba.abs() <= 1.0).all(dim=-1, keepdim=True).float()

    # Combined confidence: overlap AND in-bounds
    combined_conf_ab = overlap_ab * in_bounds_ab
    combined_conf_ba = overlap_ba * in_bounds_ba

    # Downsample warp to latent resolution
    warp_ab_latent = warp_to_latent_warp(warp_ab, image_resolution, latent_resolution)
    warp_ba_latent = warp_to_latent_warp(warp_ba, image_resolution, latent_resolution)

    # Get valid masks at latent resolution
    valid_mask_ab = confidence_to_latent_mask(
        combined_conf_ab, confidence_threshold, image_resolution, latent_resolution
    )
    valid_mask_ba = confidence_to_latent_mask(
        combined_conf_ba, confidence_threshold, image_resolution, latent_resolution
    )

    # Compute valid fractions
    valid_fraction_ab = valid_mask_ab.float().mean().item()
    valid_fraction_ba = valid_mask_ba.float().mean().item()

    return {
        "warp_ab": warp_ab,
        "warp_ba": warp_ba,
        "overlap_ab": overlap_ab,
        "overlap_ba": overlap_ba,
        "warp_ab_latent": warp_ab_latent,
        "warp_ba_latent": warp_ba_latent,
        "valid_mask_ab": valid_mask_ab,
        "valid_mask_ba": valid_mask_ba,
        "valid_fraction_ab": valid_fraction_ab,
        "valid_fraction_ba": valid_fraction_ba,
    }


def warp_latent(
    latent: torch.Tensor,
    warp: torch.Tensor
) -> torch.Tensor:
    """Warp latent representation using the correspondence field.

    Uses grid_sample to remap latent to a different coordinate frame.
    For example, warping latent_b with warp_ab gives latent_b in the
    coordinate frame of image_a.

    Args:
        latent: Latent tensor (1, C, H, W)
        warp: Warp field (1, H, W, 2) in normalized [-1, 1] coords

    Returns:
        Warped latent tensor (1, C, H, W)
    """
    # grid_sample expects grid in (1, H, W, 2) format with [-1, 1] coords
    warped = F.grid_sample(
        latent,
        warp,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    return warped


def compute_region_similarity(
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    warp_ab: torch.Tensor,
    valid_mask: torch.Tensor
) -> Dict[str, float]:
    """Compare latents only in regions with valid correspondences.

    Warps latent_b to the coordinate frame of latent_a using warp_ab,
    then compares only in regions where valid_mask is True.

    Args:
        latent_a: Latent tensor (1, C, H, W) from image A
        latent_b: Latent tensor (1, C, H, W) from image B
        warp_ab: Warp field (1, H, W, 2) mapping A coords to B coords
        valid_mask: Boolean mask (1, H, W) of valid regions

    Returns:
        Dictionary with:
        - region_mse: MSE over valid regions
        - region_mae: MAE over valid regions
        - region_cosine: Cosine similarity over valid regions
        - global_mse: MSE over full latent
        - global_mae: MAE over full latent
        - global_cosine: Cosine similarity over full latent
        - valid_fraction: Fraction of latent with valid correspondences
        - n_valid_cells: Number of valid latent cells
    """
    # Warp latent_b to A's coordinate frame
    latent_b_warped = warp_latent(latent_b, warp_ab)

    # Expand mask to match latent channels: (1, H, W) -> (1, C, H, W)
    C = latent_a.shape[1]
    mask_expanded = valid_mask.unsqueeze(1).expand(-1, C, -1, -1)

    # Extract valid regions
    valid_a = latent_a[mask_expanded]
    valid_b = latent_b_warped[mask_expanded]

    # Region metrics (only over valid areas)
    n_valid = valid_a.numel()
    if n_valid > 0:
        region_mse = F.mse_loss(valid_a, valid_b).item()
        region_mae = F.l1_loss(valid_a, valid_b).item()
        region_cosine = F.cosine_similarity(
            valid_a.unsqueeze(0), valid_b.unsqueeze(0)
        ).item()
    else:
        region_mse = float('nan')
        region_mae = float('nan')
        region_cosine = float('nan')

    # Global metrics (full latent comparison, no warping)
    flat_a = latent_a.flatten()
    flat_b = latent_b.flatten()
    global_mse = F.mse_loss(latent_a, latent_b).item()
    global_mae = F.l1_loss(latent_a, latent_b).item()
    global_cosine = F.cosine_similarity(
        flat_a.unsqueeze(0), flat_b.unsqueeze(0)
    ).item()

    # Valid fraction
    valid_fraction = valid_mask.float().mean().item()
    n_valid_cells = valid_mask.sum().item()

    return {
        "region_mse": region_mse,
        "region_mae": region_mae,
        "region_cosine": region_cosine,
        "global_mse": global_mse,
        "global_mae": global_mae,
        "global_cosine": global_cosine,
        "valid_fraction": valid_fraction,
        "n_valid_cells": int(n_valid_cells),
    }


def compute_bidirectional_region_similarity(
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    warp_ab: torch.Tensor,
    warp_ba: torch.Tensor,
    valid_mask_ab: torch.Tensor,
    valid_mask_ba: torch.Tensor
) -> Dict[str, float]:
    """Compute region similarity in both directions and average.

    This gives more robust estimates by using correspondences from both views.

    Args:
        latent_a, latent_b: Latent tensors (1, C, H, W)
        warp_ab: Warp from A to B coords
        warp_ba: Warp from B to A coords
        valid_mask_ab, valid_mask_ba: Boolean masks for each direction

    Returns:
        Dictionary with averaged metrics and per-direction details
    """
    # Forward direction: warp B to A's frame
    metrics_ab = compute_region_similarity(latent_a, latent_b, warp_ab, valid_mask_ab)

    # Reverse direction: warp A to B's frame
    metrics_ba = compute_region_similarity(latent_b, latent_a, warp_ba, valid_mask_ba)

    # Average the region metrics (skip if either is nan)
    def safe_mean(a, b):
        if np.isnan(a) and np.isnan(b):
            return float('nan')
        elif np.isnan(a):
            return b
        elif np.isnan(b):
            return a
        return (a + b) / 2

    return {
        # Averaged region metrics
        "region_mse": safe_mean(metrics_ab["region_mse"], metrics_ba["region_mse"]),
        "region_mae": safe_mean(metrics_ab["region_mae"], metrics_ba["region_mae"]),
        "region_cosine": safe_mean(metrics_ab["region_cosine"], metrics_ba["region_cosine"]),
        # Global metrics (same in both directions)
        "global_mse": metrics_ab["global_mse"],
        "global_mae": metrics_ab["global_mae"],
        "global_cosine": metrics_ab["global_cosine"],
        # Valid fractions
        "valid_fraction_ab": metrics_ab["valid_fraction"],
        "valid_fraction_ba": metrics_ba["valid_fraction"],
        "valid_fraction": safe_mean(metrics_ab["valid_fraction"], metrics_ba["valid_fraction"]),
        # Per-direction details
        "metrics_ab": metrics_ab,
        "metrics_ba": metrics_ba,
    }
