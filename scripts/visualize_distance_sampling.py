"""Evaluate warp quality for CO3D image pairs at given camera distances.

For each sampled pair, computes RoMA warps, warps images in pixel space,
measures MSE, and saves a visualization with source, target, warped images,
and L2 error heatmaps.

Usage:
    python scripts/visualize_distance_sampling.py
    python scripts/visualize_distance_sampling.py --distance_min 1.5 --distance_max 4.0
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.camera_utils import (
    compute_camera_distance_matrix,
    load_co3d_annotations,
)
from src.analysis.roma_metrics import (
    compute_roma_correspondences,
    load_roma_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate warp quality for CO3D image pairs at given camera distances"
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="/visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz",
    )
    parser.add_argument(
        "--co3d_root",
        type=str,
        default="/visinf/projects_students/dlcv2025_groupZ/co3d_full",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs/warp_quality",
    )
    parser.add_argument("--distance_min", type=float, default=2.0)
    parser.add_argument("--distance_max", type=float, default=5.0)
    parser.add_argument("--num_sequences", type=int, default=10)
    parser.add_argument("--pairs_per_sequence", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--roma_setting",
        type=str,
        default="fast",
        choices=["precise", "fast", "turbo", "base"],
    )
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper functions (local copies to avoid heavy imports from visualize_warp_dataset)
# ---------------------------------------------------------------------------

def warp_image(image: torch.Tensor, warp_field: torch.Tensor) -> torch.Tensor:
    """Warp an image using a warp field.

    Args:
        image: (C, H, W) image tensor
        warp_field: (H, W, 2) warp field in [-1, 1] normalized coordinates

    Returns:
        Warped image tensor (C, H, W)
    """
    image = image.unsqueeze(0)
    warp_field = warp_field.unsqueeze(0)
    warped = F.grid_sample(
        image, warp_field,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )
    return warped.squeeze(0)


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor [-1, 1] or [0, 1] to displayable numpy array."""
    img = img_tensor.cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.min() < 0:
        img = (img + 1) / 2
    return np.clip(img, 0, 1)


def compute_pixel_mse(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    """Compute MSE between two image tensors of shape (C, H, W)."""
    return F.mse_loss(img_a, img_b).item()


def compute_l2_error_map(img_a: torch.Tensor, img_b: torch.Tensor) -> np.ndarray:
    """Per-pixel L2 (Euclidean) distance between two (C, H, W) tensors.

    Returns:
        Error map (H, W) as numpy array.
    """
    diff = (img_a - img_b) ** 2
    l2_map = diff.sum(dim=0).sqrt()
    return l2_map.cpu().numpy()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_pair(
    source_img: np.ndarray,
    target_img: np.ndarray,
    warped_src_to_tgt: np.ndarray,
    warped_tgt_to_src: np.ndarray,
    l2_error_fwd: np.ndarray,
    l2_error_bwd: np.ndarray,
    mse_fwd: float,
    mse_bwd: float,
    camera_distance: float,
    seq_name: str,
    src_idx: int,
    tgt_idx: int,
    save_path: Path,
) -> None:
    """Save a 2x3 visualization grid for one image pair."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 0: Source, Warped Src->Tgt, L2 Error Forward
    axes[0, 0].imshow(source_img)
    axes[0, 0].set_title("Source", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(warped_src_to_tgt)
    axes[0, 1].set_title("Warped Src \u2192 Tgt", fontsize=12)
    axes[0, 1].axis("off")

    im_fwd = axes[0, 2].imshow(l2_error_fwd, cmap="Blues", vmin=0)
    axes[0, 2].set_title(f"L2 Error (fwd)\nMSE={mse_fwd:.4f}", fontsize=11)
    axes[0, 2].axis("off")
    plt.colorbar(im_fwd, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 1: Target, Warped Tgt->Src, L2 Error Backward
    axes[1, 0].imshow(target_img)
    axes[1, 0].set_title("Target", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(warped_tgt_to_src)
    axes[1, 1].set_title("Warped Tgt \u2192 Src", fontsize=12)
    axes[1, 1].axis("off")

    im_bwd = axes[1, 2].imshow(l2_error_bwd, cmap="hot", vmin=0)
    axes[1, 2].set_title(f"L2 Error (bwd)\nMSE={mse_bwd:.4f}", fontsize=11)
    axes[1, 2].axis("off")
    plt.colorbar(im_bwd, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{seq_name}  |  f{src_idx} \u2194 f{tgt_idx}  |  cam dist: {camera_distance:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir) / f"dist_{args.distance_min:.0f}_{args.distance_max:.0f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    co3d_root = Path(args.co3d_root)
    device = args.device

    img_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Load annotations
    print(f"Loading annotations from {args.annotation_path}")
    annotations = load_co3d_annotations(args.annotation_path)
    all_seq_names = list(annotations.keys())
    seq_names = random.sample(all_seq_names, min(args.num_sequences, len(all_seq_names)))
    print(f"Selected {len(seq_names)} sequences")

    # Load RoMA model
    roma_model = load_roma_model(setting=args.roma_setting, device=device, compile=False)

    all_mse_fwd = []
    all_mse_bwd = []
    pair_count = 0

    for seq_name in seq_names:
        frames = annotations[seq_name]
        if len(frames) < 2:
            continue

        dist_mat = compute_camera_distance_matrix(frames)

        source_indices = random.sample(
            range(len(frames)), min(args.pairs_per_sequence, len(frames))
        )

        for src_idx in source_indices:
            dists = dist_mat[src_idx]
            valid_indices = np.where(
                (dists >= args.distance_min) & (dists <= args.distance_max)
            )[0]
            if len(valid_indices) == 0:
                continue

            tgt_idx = int(np.random.choice(valid_indices))
            camera_distance = float(dists[tgt_idx])

            # Load images
            path_a = co3d_root / frames[src_idx]["filepath"]
            path_b = co3d_root / frames[tgt_idx]["filepath"]
            if not path_a.exists() or not path_b.exists():
                continue

            img_a_pil = Image.open(path_a).convert("RGB")
            img_b_pil = Image.open(path_b).convert("RGB")

            # Resized PIL images for RoMA
            img_a_pil_resized = img_a_pil.resize(
                (args.image_size, args.image_size), Image.LANCZOS
            )
            img_b_pil_resized = img_b_pil.resize(
                (args.image_size, args.image_size), Image.LANCZOS
            )

            # Tensors for warping ([-1, 1] normalized)
            img_a_tensor = img_transform(img_a_pil).to(device)
            img_b_tensor = img_transform(img_b_pil).to(device)

            # Compute RoMA correspondences
            roma_out = compute_roma_correspondences(
                roma_model,
                img_a_pil_resized,
                img_b_pil_resized,
                confidence_threshold=args.confidence_threshold,
                latent_resolution=32,
            )

            # Resize warp fields to match image tensor resolution
            # RoMA may output warps at a different resolution than image_size
            warp_ab = roma_out["warp_ab"].to(device)  # (1, H_roma, W_roma, 2)
            warp_ba = roma_out["warp_ba"].to(device)
            roma_h = warp_ab.shape[1]
            if roma_h != args.image_size:
                # (1, H, W, 2) -> (1, 2, H, W) for interpolate -> back
                warp_ab = F.interpolate(
                    warp_ab.permute(0, 3, 1, 2),
                    size=(args.image_size, args.image_size),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                warp_ba = F.interpolate(
                    warp_ba.permute(0, 3, 1, 2),
                    size=(args.image_size, args.image_size),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
            warp_ab = warp_ab.squeeze(0)  # (H, W, 2)
            warp_ba = warp_ba.squeeze(0)

            # Warp images in pixel space
            # warp_image(img_a, warp_ab) -> A warped into B's frame, compare with B
            warped_a_to_b = warp_image(img_a_tensor, warp_ab)
            warped_b_to_a = warp_image(img_b_tensor, warp_ba)

            # Compute metrics
            mse_fwd = compute_pixel_mse(warped_a_to_b, img_b_tensor)
            mse_bwd = compute_pixel_mse(warped_b_to_a, img_a_tensor)
            all_mse_fwd.append(mse_fwd)
            all_mse_bwd.append(mse_bwd)

            # Compute L2 error maps
            l2_error_fwd = compute_l2_error_map(warped_a_to_b, img_b_tensor)
            l2_error_bwd = compute_l2_error_map(warped_b_to_a, img_a_tensor)

            # Visualize and save
            save_path = output_dir / f"{seq_name}_f{src_idx}_f{tgt_idx}.png"
            visualize_pair(
                source_img=denormalize_image(img_a_tensor),
                target_img=denormalize_image(img_b_tensor),
                warped_src_to_tgt=denormalize_image(warped_a_to_b),
                warped_tgt_to_src=denormalize_image(warped_b_to_a),
                l2_error_fwd=l2_error_fwd,
                l2_error_bwd=l2_error_bwd,
                mse_fwd=mse_fwd,
                mse_bwd=mse_bwd,
                camera_distance=camera_distance,
                seq_name=seq_name,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                save_path=save_path,
            )
            pair_count += 1
            print(
                f"  [{pair_count}] {seq_name} f{src_idx}<->f{tgt_idx} "
                f"dist={camera_distance:.2f} MSE_fwd={mse_fwd:.4f} MSE_bwd={mse_bwd:.4f}"
            )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total pairs evaluated: {pair_count}")
    print(f"Distance range: [{args.distance_min}, {args.distance_max}]")

    if all_mse_fwd:
        all_mse = all_mse_fwd + all_mse_bwd
        print("\nForward MSE (warped_src vs target):")
        print(f"  Mean: {np.mean(all_mse_fwd):.4f}")
        print(f"  Std:  {np.std(all_mse_fwd):.4f}")
        print("\nBackward MSE (warped_tgt vs source):")
        print(f"  Mean: {np.mean(all_mse_bwd):.4f}")
        print(f"  Std:  {np.std(all_mse_bwd):.4f}")
        print("\nOverall MSE (both directions):")
        print(f"  Mean: {np.mean(all_mse):.4f}")
        print(f"  Std:  {np.std(all_mse):.4f}")
    else:
        print("No valid pairs found in the specified distance range.")

    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
