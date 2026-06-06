"""Shared helpers for the hydrant pair sweep scripts.

Used by:
  - scripts/warps/screen_hydrant_sequences.py  (pass 1)
  - scripts/warps/sweep_hydrant_pairs.py       (pass 2)
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.analysis.roma_metrics import load_roma_model
from warps.precompute_depth_warps import load_annotations

REPO = Path("/visinf/home/lab_mozkan/computer-vision-proj-lab")
ANNOT = REPO / "data/co3d_annotations/hydrant_train.jgz"
DATA_ROOT = Path("/data/lab_moezkan/co3d_full")

IMAGE_SIZE = 256
CONF_THRESHOLD = 0.8           # RoMA per-pixel confidence threshold
FRAC_CONF_GOOD = 0.8           # pair-level: frac_conf > this counts as "good"
ROMA_SETTING = "turbo"
BLACK_MAX = 5.0 / 255.0        # image considered black if .max() < this (in [0,1])


def load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return tfm(img)


def load_image_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def is_black_image(path: Path) -> bool:
    """Cheap black-image check via PIL — reads pixels but skips the torch pipeline."""
    try:
        img = Image.open(path).convert("RGB").resize((64, 64), Image.NEAREST)
        return float(np.asarray(img).max()) / 255.0 < BLACK_MAX
    except Exception:
        return True  # treat unreadable as invalid


@torch.no_grad()
def roma_warp(roma_model, pil_a, pil_b, device):
    pred = roma_model.match(pil_a, pil_b)
    warp = pred["warp_AB"]
    overlap = pred.get("overlap_AB")
    if overlap is None:
        overlap = pred["confidence_AB"].mean(dim=-1, keepdim=True)
    if warp.shape[1] != IMAGE_SIZE or warp.shape[2] != IMAGE_SIZE:
        warp = F.interpolate(warp.permute(0, 3, 1, 2), size=(IMAGE_SIZE, IMAGE_SIZE),
                             mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        overlap = F.interpolate(overlap.permute(0, 3, 1, 2), size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return warp.to(device), overlap.to(device)


def confidence_mask(conf_img: torch.Tensor, warp_img: torch.Tensor) -> torch.Tensor:
    in_bounds = (warp_img.abs() <= 1.0).all(dim=-1, keepdim=True).float()
    valid = (conf_img > CONF_THRESHOLD).float() * in_bounds
    return valid[0, ..., 0]  # (H, W)


@torch.no_grad()
def compute_frac_conf(roma_model, device, path_a: Path, path_b: Path) -> Tuple[float, Optional[torch.Tensor]]:
    """Returns (frac_conf, warp_tensor). warp is (1,H,W,2) on device."""
    pil_a = load_image_pil(path_a)
    pil_b = load_image_pil(path_b)
    warp_img, conf_img = roma_warp(roma_model, pil_a, pil_b, device)
    mask = confidence_mask(conf_img, warp_img)
    return float(mask.mean()), warp_img


def make_roma(device: torch.device):
    return load_roma_model(setting=ROMA_SETTING, device=str(device), compile=False)


__all__ = [
    "REPO", "ANNOT", "DATA_ROOT", "IMAGE_SIZE",
    "CONF_THRESHOLD", "FRAC_CONF_GOOD", "ROMA_SETTING", "BLACK_MAX",
    "load_image_tensor", "load_image_pil", "is_black_image",
    "roma_warp", "confidence_mask", "compute_frac_conf",
    "make_roma", "load_annotations",
]
