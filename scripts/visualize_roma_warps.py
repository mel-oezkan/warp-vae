"""
RoMA Warp and Latent Visualization Script.

Visualizes RoMA correspondences, confidence maps, and VAE latent embeddings
for image pairs from CO3D (toytruck) and OmniObject3D datasets.

Usage:
    python scripts/visualize_roma_warps.py

Output:
    eval_outputs/roma_visualization/
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import gzip
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.roma_metrics import (
    load_roma_model,
    compute_roma_correspondences,
    warp_latent,
)
from src.analysis.model_utils import load_model, encode_images, denormalize


# ============================================================================
# Data Loading Functions
# ============================================================================


def load_co3d_pairs(
    co3d_root: str = "/data/lab_moezkan/co3d_full",
    categories: List[str] = None,
    image_size: int = 256,
    max_pairs: int = 5,
    min_frame_distance: int = 10,
) -> List[Dict]:
    """
    Load image pairs from CO3D categories by directly reading image directories.

    Returns list of dicts with:
        - img_a, img_b: PIL Images
        - name: pair identifier
        - source: 'co3d_{category}'
    """
    if categories is None:
        categories = ["toytruck", "apple", "ball", "bench"]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    pairs = []

    for category in categories:
        category_dir = Path(co3d_root) / category
        if not category_dir.exists():
            print(f"Warning: CO3D category not found at {category_dir}")
            continue

        # Find all sequence directories
        seq_dirs = [
            d for d in category_dir.iterdir() if d.is_dir() and (d / "images").exists()
        ]

        for seq_dir in seq_dirs[:3]:  # Limit sequences per category
            images_dir = seq_dir / "images"
            image_files = sorted(images_dir.glob("*.jpg"))

            if len(image_files) < 10:
                continue

            # Select pairs with larger spacing for visible changes in presentation
            # Use frame steps that are at least min_frame_distance apart
            frame_steps = [s for s in [10, 15, 20, 25] if s >= min_frame_distance]
            if not frame_steps:
                frame_steps = [min_frame_distance]

            for frame_step in frame_steps:
                for i in range(0, min(len(image_files) - frame_step, 30), frame_step):
                    j = i + frame_step

                    if j >= len(image_files):
                        continue

                    try:
                        img_a = Image.open(image_files[i]).convert("RGB")
                        img_b = Image.open(image_files[j]).convert("RGB")
                    except Exception as e:
                        print(f"Error loading images: {e}")
                        continue

                    frame_i = image_files[i].stem.replace("frame", "")
                    frame_j = image_files[j].stem.replace("frame", "")

                    pairs.append(
                        {
                            "img_a": img_a,
                            "img_b": img_b,
                            "img_a_tensor": transform(img_a),
                            "img_b_tensor": transform(img_b),
                            "name": f"{seq_dir.name}_f{frame_i}-{frame_j}",
                            "source": f"co3d_{category}",
                        }
                    )

                    if len(pairs) >= max_pairs:
                        break

                if len(pairs) >= max_pairs:
                    break

            if len(pairs) >= max_pairs:
                break

        if len(pairs) >= max_pairs:
            break

    print(f"Loaded {len(pairs)} pairs from CO3D")
    return pairs


def load_omniobject_pairs(
    omni_root: str = "/data/lab_moezkan/omni_obj/blender_renders_24_views",
    image_size: int = 256,
    max_pairs: int = 5,
    objects_to_use: Optional[List[str]] = None,
    min_view_distance: int = 3,
) -> List[Dict]:
    """
    Load image pairs from OmniObject3D dataset.

    Note: OmniObject images have transparent backgrounds (RGBA).
    We composite them onto a white background for better RoMA matching.

    Returns list of dicts with:
        - img_a, img_b: PIL Images
        - name: pair identifier
        - source: 'omniobject'
    """
    img_dir = Path(omni_root) / "img"
    if not img_dir.exists():
        print(f"Warning: OmniObject img dir not found at {img_dir}")
        return []

    # Get all object directories
    obj_dirs = sorted([d for d in img_dir.iterdir() if d.is_dir()])

    if objects_to_use:
        obj_dirs = [d for d in obj_dirs if d.name in objects_to_use]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    def load_with_white_bg(path: Path) -> Image.Image:
        """Load RGBA image and composite onto white background."""
        img = Image.open(path)
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            # Paste image using alpha channel as mask
            background.paste(img, mask=img.split()[3])
            return background
        return img.convert("RGB")

    pairs = []
    for obj_dir in obj_dirs:
        # Get all view images (000.png to 023.png)
        view_files = sorted(obj_dir.glob("*.png"))
        if len(view_files) < 2:
            continue

        # Select pairs with larger view separation for visible changes in presentation
        # OmniObject has 24 views, so steps of 3-6 give good viewpoint change
        view_steps = [s for s in [3, 4, 5, 6] if s >= min_view_distance]
        if not view_steps:
            view_steps = [min_view_distance]

        for view_step in view_steps:
            for i in range(0, min(len(view_files), 24 - view_step)):
                j = i + view_step
                if j >= len(view_files):
                    continue

                try:
                    img_a = load_with_white_bg(view_files[i])
                    img_b = load_with_white_bg(view_files[j])
                except Exception as e:
                    print(f"Error loading images: {e}")
                    continue

                pairs.append(
                    {
                        "img_a": img_a,
                        "img_b": img_b,
                        "img_a_tensor": transform(img_a),
                        "img_b_tensor": transform(img_b),
                        "name": f"{obj_dir.name}_v{i:02d}-{j:02d}",
                        "source": "omniobject",
                    }
                )

                if len(pairs) >= max_pairs:
                    break

            if len(pairs) >= max_pairs:
                break

        if len(pairs) >= max_pairs:
            break

    print(f"Loaded {len(pairs)} pairs from OmniObject3D")
    return pairs


# ============================================================================
# Visualization Functions
# ============================================================================


def compute_warp_visualization(
    warp: torch.Tensor,
    image_size: int = 256,
) -> np.ndarray:
    """
    Convert warp field to RGB visualization.

    Warp is (1, H, W, 2) with normalized coords in [-1, 1].
    Maps x,y coordinates to R,G channels.
    """
    warp_np = warp[0].cpu().numpy()  # (H, W, 2)

    # Normalize from [-1, 1] to [0, 1]
    warp_norm = (warp_np + 1) / 2

    # Create RGB: R=x, G=y, B=0.5
    rgb = np.zeros((warp_np.shape[0], warp_np.shape[1], 3))
    rgb[..., 0] = warp_norm[..., 0]  # x -> R
    rgb[..., 1] = warp_norm[..., 1]  # y -> G
    rgb[..., 2] = 0.5  # constant blue for visibility

    return np.clip(rgb, 0, 1)


def latent_to_pca_rgb(
    latent: torch.Tensor,
    pca_model: Optional[PCA] = None,
) -> Tuple[np.ndarray, PCA]:
    """Convert latent to RGB using PCA on channels."""
    if latent.dim() == 4:
        latent = latent[0]

    C, H, W = latent.shape
    lat_flat = latent.cpu().numpy().reshape(C, -1).T  # (H*W, C)

    if pca_model is None:
        pca_model = PCA(n_components=3)
        pca_model.fit(lat_flat)

    lat_pca = pca_model.transform(lat_flat)  # (H*W, 3)
    lat_rgb = lat_pca.reshape(H, W, 3)

    # Percentile normalization
    for c in range(3):
        channel = lat_rgb[..., c]
        vmin, vmax = np.percentile(channel, [2, 98])
        lat_rgb[..., c] = np.clip((channel - vmin) / (vmax - vmin + 1e-8), 0, 1)

    return lat_rgb, pca_model


def save_individual_plot(
    data: np.ndarray,
    save_path: Path,
    title: str = None,
    cmap: str = None,
    vmin: float = None,
    vmax: float = None,
    add_colorbar: bool = False,
    dpi: int = 150,
):
    """Save a single plot as an individual image file."""
    fig, ax = plt.subplots(figsize=(6, 6))

    if cmap:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        if add_colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(data)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_roma_pair(
    pair_data: Dict,
    roma_model,
    vae_model,
    vae_type: str,
    device: str = "cuda",
    confidence_threshold: float = 0.7,
    save_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict], bool]:
    """
    Create comprehensive visualization for a single image pair.

    Saves both combined figure and individual subplot images.

    Returns:
        - metrics dict with mean confidence values
        - bool indicating if pair meets confidence threshold
    """
    img_a = pair_data["img_a"]
    img_b = pair_data["img_b"]
    img_a_tensor = pair_data["img_a_tensor"].unsqueeze(0).to(device)
    img_b_tensor = pair_data["img_b_tensor"].unsqueeze(0).to(device)
    name = pair_data["name"]
    source = pair_data["source"]

    # Compute RoMA correspondences
    print(f"  Computing RoMA correspondences for {name}...")
    roma_outputs = compute_roma_correspondences(
        roma_model,
        img_a,
        img_b,
        confidence_threshold=confidence_threshold,
        latent_resolution=32,
    )

    # Get confidence values
    overlap_ab = roma_outputs["overlap_ab"]  # (1, H, W, 1)
    overlap_ba = roma_outputs["overlap_ba"]

    mean_conf_ab = overlap_ab.mean().item()
    mean_conf_ba = overlap_ba.mean().item()
    mean_conf = (mean_conf_ab + mean_conf_ba) / 2

    print(f"    Mean confidence A->B: {mean_conf_ab:.1%}, B->A: {mean_conf_ba:.1%}")

    # Check confidence threshold (but still generate visualization for review)
    meets_threshold = mean_conf >= 0.70
    if not meets_threshold:
        print(
            f"    Low confidence pair (mean {mean_conf:.1%} < 70%) - will still visualize"
        )

    # Encode images with VAE
    print(f"  Encoding with VAE...")
    latent_a = encode_images(vae_model, img_a_tensor, device, vae_type)
    latent_b = encode_images(vae_model, img_b_tensor, device, vae_type)

    # Warp latent B to A's coordinate frame using latent-resolution warp
    warp_ab_latent = roma_outputs["warp_ab_latent"].to(device)
    latent_b_warped = warp_latent(latent_b, warp_ab_latent)

    # Prepare data for visualization
    img_a_display = np.array(img_a.resize((256, 256)))
    img_b_display = np.array(img_b.resize((256, 256)))

    warp_ab_vis = compute_warp_visualization(roma_outputs["warp_ab"])
    warp_ba_vis = compute_warp_visualization(roma_outputs["warp_ba"])

    conf_ab = overlap_ab[0, :, :, 0].cpu().numpy()
    conf_ba = overlap_ba[0, :, :, 0].cpu().numpy()

    latent_a_rgb, pca_model = latent_to_pca_rgb(latent_a)
    latent_b_rgb, _ = latent_to_pca_rgb(latent_b, pca_model)
    latent_b_warped_rgb, _ = latent_to_pca_rgb(latent_b_warped, pca_model)

    valid_mask = roma_outputs["valid_mask_ab"][0].cpu().numpy()
    latent_diff = (latent_a - latent_b_warped).abs().mean(dim=1)[0].cpu().numpy()
    latent_diff_masked = latent_diff.copy()
    latent_diff_masked[~valid_mask] = np.nan
    valid_frac = roma_outputs["valid_fraction_ab"]

    # Create folder for this pair and save individual images
    if output_dir:
        pair_folder = output_dir / f"{source}_{name}"
        pair_folder.mkdir(parents=True, exist_ok=True)

        print(f"    Saving individual plots to {pair_folder}/")

        # Save input images
        save_individual_plot(img_a_display, pair_folder / "01_image_a.png", "Image A")
        save_individual_plot(img_b_display, pair_folder / "02_image_b.png", "Image B")

        # Save warp visualizations
        save_individual_plot(
            warp_ab_vis, pair_folder / "03_warp_a_to_b.png", "Warp A→B"
        )
        save_individual_plot(
            warp_ba_vis, pair_folder / "04_warp_b_to_a.png", "Warp B→A"
        )

        # Save confidence maps
        save_individual_plot(
            conf_ab,
            pair_folder / "05_confidence_a_to_b.png",
            f"Confidence A→B (mean: {mean_conf_ab:.1%})",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            add_colorbar=True,
        )
        save_individual_plot(
            conf_ba,
            pair_folder / "06_confidence_b_to_a.png",
            f"Confidence B→A (mean: {mean_conf_ba:.1%})",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            add_colorbar=True,
        )

        # Save latent embeddings (PCA)
        save_individual_plot(
            latent_a_rgb, pair_folder / "07_latent_a_pca.png", "Latent A (PCA)"
        )
        save_individual_plot(
            latent_b_rgb, pair_folder / "08_latent_b_pca.png", "Latent B (PCA)"
        )
        save_individual_plot(
            latent_b_warped_rgb,
            pair_folder / "09_latent_b_warped_pca.png",
            "Latent B Warped to A (PCA)",
        )

        # Save difference maps
        save_individual_plot(
            latent_diff,
            pair_folder / "10_latent_difference.png",
            "Latent Difference |A - B_warped|",
            cmap="hot",
            add_colorbar=True,
        )
        save_individual_plot(
            valid_mask.astype(float),
            pair_folder / "11_valid_mask.png",
            f"Valid Region Mask (coverage: {valid_frac:.1%})",
            cmap="Greens",
            vmin=0,
            vmax=1,
        )
        save_individual_plot(
            latent_diff_masked,
            pair_folder / "12_latent_diff_masked.png",
            "Latent Difference (Valid Regions Only)",
            cmap="hot",
            add_colorbar=True,
        )

        # Save metadata
        with open(pair_folder / "metadata.txt", "w") as f:
            f.write(f"Pair: {name}\n")
            f.write(f"Source: {source}\n")
            f.write(f"Mean Confidence A→B: {mean_conf_ab:.2%}\n")
            f.write(f"Mean Confidence B→A: {mean_conf_ba:.2%}\n")
            f.write(f"Mean Confidence: {mean_conf:.2%}\n")
            f.write(f"Valid Fraction A→B: {roma_outputs['valid_fraction_ab']:.2%}\n")
            f.write(f"Valid Fraction B→A: {roma_outputs['valid_fraction_ba']:.2%}\n")
            f.write(f"Meets 70% threshold: {meets_threshold}\n")

    # Create combined visualization figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.2)

    # Row 1: Input images
    ax_img_a = fig.add_subplot(gs[0, 0])
    ax_img_b = fig.add_subplot(gs[0, 1])

    ax_img_a.imshow(img_a_display)
    ax_img_a.set_title("Image A", fontsize=12, fontweight="bold")
    ax_img_a.axis("off")

    ax_img_b.imshow(img_b_display)
    ax_img_b.set_title("Image B", fontsize=12, fontweight="bold")
    ax_img_b.axis("off")

    # Row 1: Warp visualizations
    ax_warp_ab = fig.add_subplot(gs[0, 2])
    ax_warp_ba = fig.add_subplot(gs[0, 3])

    ax_warp_ab.imshow(warp_ab_vis)
    ax_warp_ab.set_title("Warp A→B", fontsize=12)
    ax_warp_ab.axis("off")

    ax_warp_ba.imshow(warp_ba_vis)
    ax_warp_ba.set_title("Warp B→A", fontsize=12)
    ax_warp_ba.axis("off")

    # Row 1: Confidence maps
    ax_conf_ab = fig.add_subplot(gs[0, 4])
    ax_conf_ba = fig.add_subplot(gs[0, 5])

    im_ab = ax_conf_ab.imshow(conf_ab, cmap="RdYlGn", vmin=0, vmax=1)
    ax_conf_ab.set_title(f"Confidence A→B\n(mean: {mean_conf_ab:.1%})", fontsize=11)
    ax_conf_ab.axis("off")
    plt.colorbar(im_ab, ax=ax_conf_ab, fraction=0.046, pad=0.04)

    im_ba = ax_conf_ba.imshow(conf_ba, cmap="RdYlGn", vmin=0, vmax=1)
    ax_conf_ba.set_title(f"Confidence B→A\n(mean: {mean_conf_ba:.1%})", fontsize=11)
    ax_conf_ba.axis("off")
    plt.colorbar(im_ba, ax=ax_conf_ba, fraction=0.046, pad=0.04)

    # Row 2: Latent embeddings (PCA)
    ax_lat_a = fig.add_subplot(gs[1, 0:2])
    ax_lat_b = fig.add_subplot(gs[1, 2:4])
    ax_lat_b_warped = fig.add_subplot(gs[1, 4:6])

    ax_lat_a.imshow(latent_a_rgb)
    ax_lat_a.set_title("Latent A (PCA)", fontsize=12, fontweight="bold")
    ax_lat_a.axis("off")

    ax_lat_b.imshow(latent_b_rgb)
    ax_lat_b.set_title("Latent B (PCA)", fontsize=12, fontweight="bold")
    ax_lat_b.axis("off")

    ax_lat_b_warped.imshow(latent_b_warped_rgb)
    ax_lat_b_warped.set_title(
        "Latent B Warped to A (PCA)", fontsize=12, fontweight="bold"
    )
    ax_lat_b_warped.axis("off")

    # Row 3: Difference maps and validity mask
    ax_diff = fig.add_subplot(gs[2, 0:2])
    im_diff = ax_diff.imshow(latent_diff, cmap="hot")
    ax_diff.set_title("Latent Difference |A - B_warped|", fontsize=12)
    ax_diff.axis("off")
    plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

    # Valid mask
    ax_mask = fig.add_subplot(gs[2, 2:4])
    ax_mask.imshow(valid_mask, cmap="Greens", vmin=0, vmax=1)
    ax_mask.set_title(f"Valid Region Mask\n(coverage: {valid_frac:.1%})", fontsize=12)
    ax_mask.axis("off")

    # Masked latent difference
    ax_diff_masked = fig.add_subplot(gs[2, 4:6])
    im_diff_m = ax_diff_masked.imshow(latent_diff_masked, cmap="hot")
    ax_diff_masked.set_title("Latent Difference (Valid Regions Only)", fontsize=12)
    ax_diff_masked.axis("off")
    plt.colorbar(im_diff_m, ax=ax_diff_masked, fraction=0.046, pad=0.04)

    # Main title
    fig.suptitle(
        f"{name}\n{pair_data['source']} | Mean Confidence: {mean_conf:.1%}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Save combined figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"    Saved combined: {save_path}")

    plt.close(fig)

    return {
        "mean_conf_ab": mean_conf_ab,
        "mean_conf_ba": mean_conf_ba,
        "valid_fraction_ab": roma_outputs["valid_fraction_ab"],
        "valid_fraction_ba": roma_outputs["valid_fraction_ba"],
    }, meets_threshold


def create_summary_figure(
    successful_pairs: List[Dict],
    all_pairs_with_metrics: List[Dict],
    output_dir: Path,
):
    """Create a summary grid of all successful pairs for presentation."""
    # Sort by confidence (highest first)
    sorted_pairs = sorted(
        all_pairs_with_metrics,
        key=lambda x: (x["metrics"]["mean_conf_ab"] + x["metrics"]["mean_conf_ba"]) / 2,
        reverse=True,
    )

    # Take top pairs
    pairs_to_show = sorted_pairs[: min(12, len(sorted_pairs))]

    if not pairs_to_show:
        print("No pairs for summary figure")
        return

    n_pairs = len(pairs_to_show)
    n_cols = min(4, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols * 2 > 1:
        axes = axes.reshape(1, -1)
    elif n_rows == 1 and n_cols * 2 == 1:
        axes = np.array([[axes]])

    for idx, pair_info in enumerate(pairs_to_show):
        row = idx // n_cols
        col = idx % n_cols

        ax_a = axes[row, col * 2]
        ax_b = axes[row, col * 2 + 1]

        img_a = pair_info["img_a"].resize((256, 256))
        img_b = pair_info["img_b"].resize((256, 256))

        conf = (
            pair_info["metrics"]["mean_conf_ab"] + pair_info["metrics"]["mean_conf_ba"]
        ) / 2
        is_high_conf = conf >= 0.70

        ax_a.imshow(img_a)
        title_color = "green" if is_high_conf else "orange"
        ax_a.set_title(
            f"{pair_info['source']}\n{pair_info['name']}", fontsize=9, color=title_color
        )
        ax_a.axis("off")

        ax_b.imshow(img_b)
        ax_b.set_title(
            f"Conf: {conf:.1%}", fontsize=10, fontweight="bold", color=title_color
        )
        ax_b.axis("off")

        # Add border for high-confidence pairs
        if is_high_conf:
            for ax in [ax_a, ax_b]:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("green")
                    spine.set_linewidth(3)

    # Hide unused axes
    for idx in range(n_pairs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if row < axes.shape[0] and col * 2 < axes.shape[1]:
            axes[row, col * 2].axis("off")
            axes[row, col * 2 + 1].axis("off")

    high_conf_count = len(
        [
            p
            for p in pairs_to_show
            if (p["metrics"]["mean_conf_ab"] + p["metrics"]["mean_conf_ba"]) / 2 >= 0.70
        ]
    )
    plt.suptitle(
        f"RoMA Visualization: Top {n_pairs} Pairs by Confidence\n"
        f"(Green: >=70% conf [{high_conf_count}], Orange: <70%)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path = output_dir / "summary_pairs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved summary figure: {save_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    roma_setting = "fast"  # fast setting as requested
    confidence_threshold = (
        0.5  # 50% mean confidence threshold (lowered for larger viewpoint changes)
    )
    image_size = 256
    min_frame_distance = 10  # Minimum frame distance for CO3D (for visible changes)
    min_view_distance = (
        3  # Minimum view distance for OmniObject (3-6 views = ~45-90 degrees)
    )

    # Paths
    vae_checkpoint = (
        PROJECT_ROOT / "checkpoints" / "eq-vae" / "diffusion_pytorch_model.safetensors"
    )
    vae_config = PROJECT_ROOT / "checkpoints" / "eq-vae" / "config.json"
    output_dir = PROJECT_ROOT / "eval_outputs" / "roma_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RoMA Warp and Latent Visualization")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"RoMA setting: {roma_setting}")
    print(f"Confidence threshold: {confidence_threshold:.0%}")
    print(f"Output directory: {output_dir}")
    print()

    # Load models
    print("Loading RoMA model...")
    roma_model = load_roma_model(setting=roma_setting, device=device, compile=False)

    print("Loading VAE model...")
    vae_model, vae_type = load_model(str(vae_checkpoint), str(vae_config))
    vae_model = vae_model.to(device)
    vae_model.eval()
    print(f"  VAE type: {vae_type}")
    print()

    # Load image pairs
    print("Loading image pairs...")

    # CO3D pairs - use categories that exist in the full dataset
    co3d_pairs = load_co3d_pairs(
        categories=["book", "hydrant", "toybus"],
        max_pairs=10,
        image_size=image_size,
        min_frame_distance=min_frame_distance,
    )

    # OmniObject pairs - use textured objects for better RoMA matching
    # Avoid textureless objects like apples/balls which give low confidence
    omni_objects = [
        "book_001",
        "book_002",
        "book_003",
        "book_004",  # books have good texture
        "box_001",
        "box_002",
        "box_003",  # boxes with patterns
        "antique_004",
        "antique_005",  # antique items with texture
    ]
    omni_pairs = load_omniobject_pairs(
        max_pairs=10,
        image_size=image_size,
        objects_to_use=omni_objects,
        min_view_distance=min_view_distance,
    )

    all_pairs = co3d_pairs + omni_pairs
    print(f"\nTotal pairs to process: {len(all_pairs)}")
    print()

    # Process pairs
    successful_pairs = []
    all_pairs_with_metrics = []

    for idx, pair_data in enumerate(all_pairs):
        print(f"\nProcessing pair {idx + 1}/{len(all_pairs)}: {pair_data['name']}")

        save_path = output_dir / f"{pair_data['source']}_{pair_data['name']}.png"

        try:
            metrics, meets_threshold = visualize_roma_pair(
                pair_data,
                roma_model,
                vae_model,
                vae_type,
                device=device,
                confidence_threshold=confidence_threshold,
                save_path=save_path,
                output_dir=output_dir,  # Pass output_dir for individual plots
            )

            pair_info = {
                "name": pair_data["name"],
                "source": pair_data["source"],
                "img_a": pair_data["img_a"],
                "img_b": pair_data["img_b"],
                "metrics": metrics,
            }
            all_pairs_with_metrics.append(pair_info)

            if meets_threshold:
                successful_pairs.append(pair_info)

        except Exception as e:
            print(f"  Error processing pair: {e}")
            import traceback

            traceback.print_exc()

    # Create summary figure
    print("\n" + "=" * 60)
    print("Creating summary figure...")
    create_summary_figure(successful_pairs, all_pairs_with_metrics, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total pairs processed: {len(all_pairs_with_metrics)}")
    print(f"High confidence (>= 70%): {len(successful_pairs)}")
    print(
        f"Low confidence (< 70%): {len(all_pairs_with_metrics) - len(successful_pairs)}"
    )

    # Sort all by confidence
    sorted_all = sorted(
        all_pairs_with_metrics,
        key=lambda x: (x["metrics"]["mean_conf_ab"] + x["metrics"]["mean_conf_ba"]) / 2,
        reverse=True,
    )

    print("\nAll pairs ranked by confidence:")
    for p in sorted_all:
        conf = (p["metrics"]["mean_conf_ab"] + p["metrics"]["mean_conf_ba"]) / 2
        marker = "[HIGH]" if conf >= 0.70 else "[low]"
        print(f"  {marker} {p['source']}/{p['name']}: {conf:.1%}")

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
