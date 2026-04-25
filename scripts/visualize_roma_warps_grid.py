"""
Visualize bidirectional RoMA warps in a 2x2 grid.

Layout:
    +---------------------+---------------------+
    |  Image A (original) |  Image B (original) |
    +---------------------+---------------------+
    |  B warped to A      |  A warped to B      |
    +---------------------+---------------------+

White regions indicate low-confidence / out-of-view areas.

Usage:
    python scripts/visualize_roma_warps_grid.py \
        --co3d_category hydrant --random \
        --output roma_warp_grid.png

    python scripts/visualize_roma_warps_grid.py \
        --img_a path/a.jpg --img_b path/b.jpg \
        --output roma_warp_grid.png
"""

import sys
import random
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.roma_metrics import load_roma_model


def load_co3d_pair(
    category: str,
    frame_step: int,
    co3d_root: str = "/data/lab_moezkan/co3d_full",
    sequence: str = None,
    randomize: bool = False,
):
    """Load a single image pair from a CO3D sequence."""
    category_dir = Path(co3d_root) / category
    if not category_dir.exists():
        raise FileNotFoundError(f"CO3D category not found: {category_dir}")

    seq_dirs = [
        d for d in category_dir.iterdir() if d.is_dir() and (d / "images").exists()
    ]
    if not seq_dirs:
        raise FileNotFoundError(f"No sequences found in {category_dir}")

    # iterate over all possible sequences
    valid_seqs = []
    for d in sorted(seq_dirs):
        n_frames = len(list((d / "images").glob("*.jpg")))
        if n_frames > frame_step:
            valid_seqs.append(d)

    if not valid_seqs:
        raise FileNotFoundError("No sequences with enough frames")

    if sequence:
        # check if provided sequence exists
        matches = [d for d in valid_seqs if d.name == sequence]
        if not matches:
            available = [d.name for d in valid_seqs[:20]]
            raise FileNotFoundError(
                f"Sequence '{sequence}' not found. Available[:20]: {available}"
            )
        seq_dir = matches[0]
    elif randomize:
        seq_dir = random.choice(valid_seqs)
    # default to the first sequence
    else:
        seq_dir = valid_seqs[0]

    image_files = sorted((seq_dir / "images").glob("*.jpg"))

    if randomize:
        idx_a = random.randint(0, len(image_files) - frame_step - 1)
    else:
        idx_a = 0
    idx_b = idx_a + frame_step

    img_a = Image.open(image_files[idx_a]).convert("RGB")
    img_b = Image.open(image_files[idx_b]).convert("RGB")
    print(
        f"Loaded pair from {seq_dir.name}: "
        f"{image_files[idx_a].name} <-> {image_files[idx_b].name}"
    )
    return img_a, img_b


def warp_image(img_tensor, warp, overlap, background=None):
    """Warp an image using RoMA warp field, blending low-confidence areas with a background.

    Args:
        img_tensor: (3, H, W) float tensor in [0, 1] — the source image to warp.
        warp: (H, W, 2) warp field in normalized [-1, 1] coords.
        overlap: (H, W, 1) confidence map in [0, 1].
        background: (3, H, W) float tensor in [0, 1] — fallback for low-confidence areas.
                    If None, uses white.

    Returns:
        (H, W, 3) numpy RGB image.
    """
    warped = F.grid_sample(
        img_tensor[None],
        warp[None],
        mode="bilinear",
        align_corners=False,
    )[0]  # (3, H, W)

    overlap_3 = overlap.permute(2, 0, 1)  # (1, H, W)
    if background is None:
        bg = torch.ones_like(warped)
    else:
        bg = background
    blended = overlap_3 * warped + (1 - overlap_3) * bg

    return blended.permute(1, 2, 0).cpu().numpy()


def main():
    parser = ArgumentParser(
        description="Visualize bidirectional RoMA warps in a 2x2 grid"
    )
    parser.add_argument("--img_a", type=str, default=None)
    parser.add_argument("--img_b", type=str, default=None)
    parser.add_argument("--co3d_category", type=str, default=None)
    parser.add_argument("--co3d_root", type=str, default="/data/lab_moezkan/co3d_full")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--frame_step", type=int, default=15)
    parser.add_argument(
        "--roma_setting",
        type=str,
        default="fast",
        choices=["fast", "precise", "turbo", "base"],
    )
    parser.add_argument("--output", type=str, default="roma_warp_grid.png")
    args = parser.parse_args()

    if args.img_a and args.img_b:
        img_a = Image.open(args.img_a).convert("RGB")
        img_b = Image.open(args.img_b).convert("RGB")
    elif args.co3d_category:
        img_a, img_b = load_co3d_pair(
            args.co3d_category,
            args.frame_step,
            args.co3d_root,
            args.sequence,
            args.random,
        )
    else:
        parser.error("Provide --img_a and --img_b, or --co3d_category")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading RoMA ({args.roma_setting})...")
    roma = load_roma_model(setting=args.roma_setting, device=device, compile=False)

    # Get the resolution RoMA will use
    H = roma.H_hr if roma.H_hr is not None else roma.H_lr
    W = roma.W_hr if roma.W_hr is not None else roma.W_lr

    # Resize images to RoMA resolution for pixel-accurate warping
    img_a_resized = img_a.resize((W, H))
    img_b_resized = img_b.resize((W, H))

    print("Computing correspondences (A->B and B->A)...")
    with torch.no_grad():
        preds_ab = roma.match(img_a, img_b)
        preds_ba = roma.match(img_b, img_a)

    warp_ab = preds_ab["warp_AB"][0].to(device)  # (H, W, 2): maps A coords -> B coords
    overlap_ab = preds_ab["overlap_AB"][0].to(device)  # (H, W, 1)
    warp_ba = preds_ba["warp_AB"][0].to(device)  # (H, W, 2): maps B coords -> A coords
    overlap_ba = preds_ba["overlap_AB"][0].to(device)  # (H, W, 1)

    # Convert images to tensors (3, H, W) in [0, 1]
    img_a_t = (
        (torch.tensor(np.array(img_a_resized), dtype=torch.float32) / 255)
        .permute(2, 0, 1)
        .to(device)
    )
    img_b_t = (
        (torch.tensor(np.array(img_b_resized), dtype=torch.float32) / 255)
        .permute(2, 0, 1)
        .to(device)
    )

    # Warp: grid_sample(B, warp_AB) = B warped into A's viewpoint
    print("Warping images...")
    b_warped_to_a_white = warp_image(img_b_t, warp_ab, overlap_ab)
    a_warped_to_b_white = warp_image(img_a_t, warp_ba, overlap_ba)

    # Third row: use target image as background
    b_warped_to_a_bg = warp_image(img_b_t, warp_ab, overlap_ab, background=img_a_t)
    a_warped_to_b_bg = warp_image(img_a_t, warp_ba, overlap_ba, background=img_b_t)

    mean_conf_ab = overlap_ab.mean().item()
    mean_conf_ba = overlap_ba.mean().item()

    # Plot 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    axes[0, 0].imshow(np.array(img_a_resized))
    axes[0, 0].set_title("Image A (original)", fontsize=14, fontweight="bold")

    axes[0, 1].imshow(np.array(img_b_resized))
    axes[0, 1].set_title("Image B (original)", fontsize=14, fontweight="bold")

    axes[1, 0].imshow(b_warped_to_a_white)
    axes[1, 0].set_title(
        f"B warped to A — white bg (conf: {mean_conf_ab:.1%})", fontsize=14, fontweight="bold"
    )

    axes[1, 1].imshow(a_warped_to_b_white)
    axes[1, 1].set_title(
        f"A warped to B — white bg (conf: {mean_conf_ba:.1%})", fontsize=14, fontweight="bold"
    )

    axes[2, 0].imshow(b_warped_to_a_bg)
    axes[2, 0].set_title(
        "B warped to A — target bg", fontsize=14, fontweight="bold"
    )

    axes[2, 1].imshow(a_warped_to_b_bg)
    axes[2, 1].set_title(
        "A warped to B — target bg", fontsize=14, fontweight="bold"
    )

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("RoMA Bidirectional Warps", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
