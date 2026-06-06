"""
Visualization script for WarpCO3DDataset outputs.

Generates visualizations of:
- Source and target images
- Warp maps (A->B and B->A)
- Confidence scores
- Warped images
- Warp errors
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.warp_dataset import WarpCO3DDataset


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor [-1, 1] or [0, 1] to displayable numpy array."""
    img = img_tensor.cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    # Handle normalization (assume [-1, 1] if min < 0)
    if img.min() < 0:
        img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    return img


def warp_image(image: torch.Tensor, warp_field: torch.Tensor) -> torch.Tensor:
    """
    Warp an image using a warp field.

    Args:
        image: (C, H, W) image tensor
        warp_field: (H, W, 2) warp field in [-1, 1] normalized coordinates

    Returns:
        Warped image tensor (C, H, W)
    """
    # Add batch dimension
    image = image.unsqueeze(0)  # (1, C, H, W)
    warp_field = warp_field.unsqueeze(0)  # (1, H, W, 2)

    # grid_sample expects (N, H, W, 2) grid
    warped = F.grid_sample(
        image,
        warp_field,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    return warped.squeeze(0)


def compute_warp_error(
    source_img: torch.Tensor,
    target_img: torch.Tensor,
    warp_field: torch.Tensor,
    confidence: torch.Tensor
) -> np.ndarray:
    """
    Compute the warp error between warped source and target.

    Returns:
        Error map as numpy array (H, W)
    """
    warped = warp_image(source_img, warp_field)

    # Compute per-pixel L1 error
    error = torch.abs(warped - target_img).mean(dim=0)  # (H, W)

    return error.cpu().numpy()


def visualize_warp_field(warp: torch.Tensor) -> np.ndarray:
    """
    Visualize a warp field as an RGB image using HSV color coding.

    Args:
        warp: (H, W, 2) warp field in [-1, 1] coordinates

    Returns:
        RGB visualization (H, W, 3)
    """
    warp_np = warp.cpu().numpy()

    # Compute flow magnitude and angle
    u = warp_np[..., 0]
    v = warp_np[..., 1]

    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)

    # Normalize magnitude for visualization
    mag_normalized = magnitude / (magnitude.max() + 1e-8)

    # Convert to HSV
    hsv = np.zeros((warp_np.shape[0], warp_np.shape[1], 3))
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)  # Hue: angle
    hsv[..., 1] = 1.0  # Saturation
    hsv[..., 2] = mag_normalized  # Value: magnitude

    # Convert HSV to RGB
    import matplotlib.colors as mcolors
    rgb = mcolors.hsv_to_rgb(hsv)

    return rgb


def save_figure(fig, output_path: str):
    """Save figure and close it."""
    fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {output_path}")


def visualize_sample(
    sample: dict,
    output_dir: str,
    sample_idx: int
):
    """
    Visualize all components of a warp dataset sample.

    Creates (2, 1) subplot figures for each visualization type.
    """
    # Extract data
    img_a = sample['image']
    img_b = sample['image_target']
    warp_ab = sample['warp_ab']
    warp_ba = sample['warp_ba']
    conf_ab = sample['confidence_ab']
    conf_ba = sample['confidence_ba']

    # 1. Source and Target Images
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    axes[0].imshow(denormalize_image(img_a))
    axes[0].axis('off')
    axes[1].imshow(denormalize_image(img_b))
    axes[1].axis('off')
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, f"sample_{sample_idx:04d}_inputs.png"))

    # 2. Warp Maps (A->B and B->A)
    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    axes[0].imshow(visualize_warp_field(warp_ab))
    axes[0].axis('off')
    axes[1].imshow(visualize_warp_field(warp_ba))
    axes[1].axis('off')
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, f"sample_{sample_idx:04d}_warp_maps.png"))

    # 3. Confidence Scores
    conf_ab_np = conf_ab.cpu().numpy()
    conf_ba_np = conf_ba.cpu().numpy()
    mean_conf_ab = conf_ab_np.mean()
    mean_conf_ba = conf_ba_np.mean()

    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    im0 = axes[0].imshow(conf_ab_np, cmap='viridis', vmin=0, vmax=1)
    axes[0].axis('off')
    axes[0].text(
        0.5, 0.02, f"mean: {mean_conf_ab:.3f}",
        transform=axes[0].transAxes, ha='center', va='bottom',
        fontsize=12, color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
    )
    im1 = axes[1].imshow(conf_ba_np, cmap='viridis', vmin=0, vmax=1)
    axes[1].axis('off')
    axes[1].text(
        0.5, 0.02, f"mean: {mean_conf_ba:.3f}",
        transform=axes[1].transAxes, ha='center', va='bottom',
        fontsize=12, color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
    )
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, f"sample_{sample_idx:04d}_confidence.png"))

    # 4. Warped Images (A warped to B, B warped to A)
    warped_a_to_b = warp_image(img_a, warp_ab)
    warped_b_to_a = warp_image(img_b, warp_ba)

    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    axes[0].imshow(denormalize_image(warped_a_to_b))
    axes[0].axis('off')
    axes[1].imshow(denormalize_image(warped_b_to_a))
    axes[1].axis('off')
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, f"sample_{sample_idx:04d}_warped.png"))

    # 5. Warp Errors
    error_ab = compute_warp_error(img_a, img_b, warp_ab, conf_ab)
    error_ba = compute_warp_error(img_b, img_a, warp_ba, conf_ba)
    mean_error_ab = np.abs(error_ab).mean()
    mean_error_ba = np.abs(error_ba).mean()

    fig, axes = plt.subplots(2, 1, figsize=(6, 12))
    im0 = axes[0].imshow(error_ab, cmap='hot', vmin=0, vmax=1)
    axes[0].axis('off')
    axes[0].text(
        0.5, 0.02, f"mean: {mean_error_ab:.3f}",
        transform=axes[0].transAxes, ha='center', va='bottom',
        fontsize=12, color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
    )
    im1 = axes[1].imshow(error_ba, cmap='hot', vmin=0, vmax=1)
    axes[1].axis('off')
    axes[1].text(
        0.5, 0.02, f"mean: {mean_error_ba:.3f}",
        transform=axes[1].transAxes, ha='center', va='bottom',
        fontsize=12, color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
    )
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, f"sample_{sample_idx:04d}_warp_error.png"))


def main():
    parser = argparse.ArgumentParser(description="Visualize WarpCO3DDataset outputs")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/data/lab_moezkan/co3d_full",
        help="Path to CO3D dataset root"
    )
    parser.add_argument(
        "--bb_file",
        type=str,
        default="/data/lab_moezkan/co3d_bboxes/toybus_test.jgz",
        help="Path to bounding box file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs/visualize_repa",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size"
    )
    parser.add_argument(
        "--romav2_setting",
        type=str,
        default="turbo",
        choices=["turbo", "fast", "base", "precise"],
        help="RoMaV2 setting"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample indices to visualize (overrides --num_samples)"
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dataset
    print("Loading WarpCO3DDataset...")
    dataset = WarpCO3DDataset(
        root_dir=args.root_dir,
        bb_file=args.bb_file,
        image_size=args.image_size,
        romav2_setting=args.romav2_setting,
        pair_sampling="random",
        max_pair_distance=20,
        warp_resolution=args.image_size,
    )

    print(f"Dataset loaded with {len(dataset)} samples")

    # Determine which samples to visualize
    if args.sample_indices is not None:
        indices = args.sample_indices
    else:
        # Select random samples
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
        indices = sorted(indices.tolist())

    print(f"Visualizing samples: {indices}")

    # Visualize each sample
    for i, idx in enumerate(indices):
        print(f"\nProcessing sample {i+1}/{len(indices)} (index={idx})...")
        sample = dataset[idx]
        visualize_sample(sample, args.output_dir, idx)

    print(f"\nVisualization complete! Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
