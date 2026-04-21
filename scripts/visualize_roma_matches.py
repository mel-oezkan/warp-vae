"""
Visualize RoMA feature matches between image pairs with connecting lines.

Places two images side-by-side and draws lines between matched keypoints,
colored by confidence. Works with CO3D image pairs.

Usage:
    python scripts/visualize_roma_matches.py \
        --img_a path/to/image_a.jpg \
        --img_b path/to/image_b.jpg \
        --num_matches 100 \
        --output roma_matches.png

    # Or use a CO3D sequence (picks a pair automatically):
    python scripts/visualize_roma_matches.py \
        --co3d_category hydrant \
        --frame_step 15 \
        --num_matches 100 \
        --output roma_matches.png
"""

import sys
from pathlib import Path
from argparse import ArgumentParser

import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.roma_metrics import load_roma_model


def load_co3d_pair(category: str, frame_step: int, image_size: int,
                   co3d_root: str = "/data/lab_moezkan/co3d_full",
                   sequence: str = None, randomize: bool = False):
    """Load a single image pair from a CO3D sequence.

    Args:
        sequence: Specific sequence name to use. If None, picks first (or random).
        randomize: If True, pick a random sequence and random starting frame.
    """
    category_dir = Path(co3d_root) / category
    if not category_dir.exists():
        raise FileNotFoundError(f"CO3D category not found: {category_dir}")

    seq_dirs = [d for d in category_dir.iterdir()
                if d.is_dir() and (d / "images").exists()]
    if not seq_dirs:
        raise FileNotFoundError(f"No sequences found in {category_dir}")

    # Filter to only sequences with enough frames
    valid_seqs = []
    for d in sorted(seq_dirs):
        n_frames = len(list((d / "images").glob("*.jpg")))
        if n_frames > frame_step:
            valid_seqs.append(d)

    if not valid_seqs:
        raise FileNotFoundError("No sequences with enough frames")

    # Select sequence
    if sequence:
        matches = [d for d in valid_seqs if d.name == sequence]
        if not matches:
            available = [d.name for d in valid_seqs[:20]]
            raise FileNotFoundError(
                f"Sequence '{sequence}' not found. Available: {available}")
        seq_dir = matches[0]
    elif randomize:
        seq_dir = random.choice(valid_seqs)
    else:
        seq_dir = valid_seqs[0]

    image_files = sorted((seq_dir / "images").glob("*.jpg"))

    # Select frame indices
    if randomize:
        idx_a = random.randint(0, len(image_files) - frame_step - 1)
    else:
        idx_a = 0
    idx_b = idx_a + frame_step

    img_a = Image.open(image_files[idx_a]).convert("RGB")
    img_b = Image.open(image_files[idx_b]).convert("RGB")
    print(f"Loaded pair from {seq_dir.name}: "
          f"{image_files[idx_a].name} <-> {image_files[idx_b].name}")

    # Load foreground masks if available
    mask_a, mask_b = None, None
    masks_dir = seq_dir / "masks"
    if masks_dir.exists():
        mask_files = sorted(masks_dir.glob("*.png"))
        if idx_a < len(mask_files) and idx_b < len(mask_files):
            mask_a = Image.open(mask_files[idx_a]).convert("L")
            mask_b = Image.open(mask_files[idx_b]).convert("L")

    return img_a, img_b, mask_a, mask_b


def filter_matches_by_mask(matches: np.ndarray, confidence: np.ndarray,
                           mask_a: Image.Image, mask_b: Image.Image,
                           threshold: float = 128) -> tuple:
    """Keep only matches where both endpoints land on the foreground object.

    Args:
        matches: (N, 4) in normalized [-1, 1] coords.
        confidence: (N,) scores.
        mask_a, mask_b: Grayscale PIL masks (white = foreground).
        threshold: Pixel value threshold for foreground (0-255).

    Returns:
        Filtered (matches, confidence).
    """
    mask_a_np = np.array(mask_a.resize((512, 512)))  # match RoMA internal res
    mask_b_np = np.array(mask_b.resize((512, 512)))
    h, w = mask_a_np.shape

    # Normalized [-1, 1] -> pixel coords
    px_a = ((matches[:, 0] + 1) / 2 * w).astype(int).clip(0, w - 1)
    py_a = ((matches[:, 1] + 1) / 2 * h).astype(int).clip(0, h - 1)
    px_b = ((matches[:, 2] + 1) / 2 * w).astype(int).clip(0, w - 1)
    py_b = ((matches[:, 3] + 1) / 2 * h).astype(int).clip(0, h - 1)

    fg_a = mask_a_np[py_a, px_a] > threshold
    fg_b = mask_b_np[py_b, px_b] > threshold
    keep = fg_a & fg_b

    print(f"  Mask filter: {keep.sum()}/{len(keep)} matches on foreground object")
    return matches[keep], confidence[keep]


def visualize_matches(img_a: Image.Image, img_b: Image.Image,
                      matches: np.ndarray, confidence: np.ndarray,
                      output_path: str, image_size: int = 512,
                      line_alpha: float = 0.6, point_size: float = 12):
    """
    Draw matched keypoints with connecting lines on a side-by-side canvas.

    Args:
        img_a, img_b: Input images.
        matches: (N, 4) array of [x_A, y_A, x_B, y_B] in normalized [-1, 1].
        confidence: (N,) confidence scores.
        output_path: Where to save the figure.
        image_size: Display size for each image.
        line_alpha: Line transparency.
        point_size: Keypoint marker size.
    """
    img_a = img_a.resize((image_size, image_size))
    img_b = img_b.resize((image_size, image_size))

    # Convert normalized [-1, 1] coords to pixel coords
    # Image A sits at x in [0, image_size), Image B at [image_size, 2*image_size)
    def norm_to_pixel(nx, ny):
        px = (nx + 1) / 2 * image_size
        py = (ny + 1) / 2 * image_size
        return px, py

    pts_a = np.stack(norm_to_pixel(matches[:, 0], matches[:, 1]), axis=1)
    pts_b = np.stack(norm_to_pixel(matches[:, 2], matches[:, 3]), axis=1)
    pts_b[:, 0] += image_size  # shift x for the right image

    # Build side-by-side canvas
    canvas = np.concatenate([np.array(img_a), np.array(img_b)], axis=1)

    # Color by confidence
    conf_norm = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)
    cmap = plt.cm.RdYlGn  # red = low, green = high

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(canvas)

    # Draw lines and points
    for i in range(len(matches)):
        color = cmap(conf_norm[i])
        line = mlines.Line2D(
            [pts_a[i, 0], pts_b[i, 0]],
            [pts_a[i, 1], pts_b[i, 1]],
            color=color, alpha=line_alpha, linewidth=0.8,
        )
        ax.add_line(line)

    ax.scatter(pts_a[:, 0], pts_a[:, 1], c=conf_norm, cmap='RdYlGn',
               s=point_size, edgecolors='k', linewidths=0.3, zorder=5)
    ax.scatter(pts_b[:, 0], pts_b[:, 1], c=conf_norm, cmap='RdYlGn',
               s=point_size, edgecolors='k', linewidths=0.3, zorder=5)

    # Separator line between images
    ax.axvline(x=image_size, color='white', linewidth=2, linestyle='--', alpha=0.5)

    ax.set_xlim(0, 2 * image_size)
    ax.set_ylim(image_size, 0)
    ax.axis('off')
    ax.set_title(f"RoMA Feature Matches ({len(matches)} correspondences)", fontsize=14)

    # Colorbar for confidence
    sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                                norm=plt.Normalize(confidence.min(), confidence.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('Confidence', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved to {output_path}")


def main():
    parser = ArgumentParser(description="Visualize RoMA feature matches between two images")
    parser.add_argument("--img_a", type=str, default=None, help="Path to image A")
    parser.add_argument("--img_b", type=str, default=None, help="Path to image B")
    parser.add_argument("--co3d_category", type=str, default=None,
                        help="CO3D category to auto-load a pair from (e.g. hydrant)")
    parser.add_argument("--co3d_root", type=str, default="/data/lab_moezkan/co3d_full")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Specific CO3D sequence name to use")
    parser.add_argument("--random", action="store_true",
                        help="Randomly pick sequence and starting frame")
    parser.add_argument("--frame_step", type=int, default=15,
                        help="Frame distance for CO3D pairs")
    parser.add_argument("--num_matches", type=int, default=200,
                        help="Number of keypoint matches to sample")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Display size per image")
    parser.add_argument("--roma_setting", type=str, default="fast",
                        choices=["fast", "precise", "turbo", "base"])
    parser.add_argument("--foreground_only", action="store_true",
                        help="Filter matches to foreground object using CO3D masks")
    parser.add_argument("--output", type=str, default="roma_matches.png",
                        help="Output image path")
    args = parser.parse_args()

    # Load images
    mask_a, mask_b = None, None
    if args.img_a and args.img_b:
        img_a = Image.open(args.img_a).convert("RGB")
        img_b = Image.open(args.img_b).convert("RGB")
    elif args.co3d_category:
        img_a, img_b, mask_a, mask_b = load_co3d_pair(
            args.co3d_category, args.frame_step, args.image_size, args.co3d_root,
            sequence=args.sequence, randomize=args.random)
    else:
        parser.error("Provide --img_a and --img_b, or --co3d_category")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load RoMA and compute matches
    print(f"Loading RoMA ({args.roma_setting})...")
    roma = load_roma_model(setting=args.roma_setting, device=device, compile=False)

    print("Computing correspondences...")
    with torch.no_grad():
        preds = roma.match(img_a, img_b)

    # Sample sparse keypoint matches
    matches, confidence, *_ = roma.sample(preds, num_corresp=args.num_matches)
    matches = matches.cpu().numpy()      # (N, 4): x_A, y_A, x_B, y_B
    confidence = confidence.cpu().numpy() # (N,)

    print(f"Sampled {len(matches)} matches, "
          f"confidence range [{confidence.min():.3f}, {confidence.max():.3f}]")

    # Filter to foreground object if requested
    if args.foreground_only and mask_a is not None and mask_b is not None:
        matches, confidence = filter_matches_by_mask(matches, confidence, mask_a, mask_b)
        if len(matches) == 0:
            print("No matches on foreground — try increasing --num_matches")
            return
    elif args.foreground_only:
        print("Warning: --foreground_only requires CO3D masks, skipping filter")

    # Visualize
    visualize_matches(img_a, img_b, matches, confidence,
                      args.output, args.image_size)


if __name__ == "__main__":
    main()
