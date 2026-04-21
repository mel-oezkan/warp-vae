"""
Visualize depth-based warp generation on CO3D data.

Shows for a few example pairs:
  - Image A (cropped), Image B (cropped)
  - Depth map A, Depth map B
  - Warp confidence map (A->B)
  - Image B warped to A (using warp_ab)
  - Checkerboard overlay of A and warped B

Usage:
    python scripts/visualize_depth_warps.py \
        --annotation_file data/co3d_annotations/hydrant_train_50seq_depth.jgz \
        --num_pairs 4 --crop_images
"""

import argparse
import gzip
import json
import sys
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_process.co3d_dataset import square_bbox
from precompute_depth_warps import (
    load_annotations,
    build_flat_samples,
    load_co3d_depth,
    load_co3d_depth_mask,
    build_intrinsic_matrix,
    compute_depth_warp,
    compute_sequence_distance_matrix,
    get_crop_bbox,
    get_image_size,
)


def load_and_crop_image(filepath, crop_bbox, resolution):
    """Load image, optionally crop to bbox, resize to resolution."""
    img = Image.open(filepath).convert("RGB")
    if crop_bbox is not None:
        x1, y1, x2, y2 = crop_bbox
        img = img.crop((x1, y1, x2, y2))
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def apply_warp(image_tensor, warp):
    """Warp image_tensor (C,H,W) using warp (H_w,W_w,2) in [-1,1]."""
    img = image_tensor.unsqueeze(0).float()  # (1,C,H,W)
    grid = warp.unsqueeze(0).float()  # (1,H,W,2)
    # Resize image to warp resolution if needed
    if img.shape[2] != grid.shape[1] or img.shape[3] != grid.shape[2]:
        img = F.interpolate(img, size=(grid.shape[1], grid.shape[2]), mode="bilinear", align_corners=False)
    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return warped.squeeze(0)


def checkerboard_blend(img_a, img_b, block_size=16):
    """Create checkerboard overlay of two images."""
    H, W = img_a.shape[1], img_a.shape[2]
    mask = torch.zeros(1, H, W)
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                mask[:, i:i+block_size, j:j+block_size] = 1.0
    return img_a * mask + img_b * (1 - mask)


def select_example_pairs(sequence_to_indices, samples, num_pairs, max_cam_dist=2.0, min_cam_dist=0.3):
    """Select pairs with noticeable viewpoint difference for visualization."""
    pairs = []
    seq_names = list(sequence_to_indices.keys())
    random.shuffle(seq_names)

    for seq_name in seq_names:
        if len(pairs) >= num_pairs:
            break
        indices = sequence_to_indices[seq_name]
        if len(indices) < 2:
            continue
        frames = [samples[i] for i in indices]
        # Check depth availability
        has_depth = [i for i, f in zip(indices, frames) if "depth_path" in f]
        if len(has_depth) < 2:
            continue

        dist_matrix = compute_sequence_distance_matrix(frames)
        # Pick a pair with moderate-to-large distance for visible difference
        best_pair = None
        best_dist = 0
        for _ in range(50):
            i, j = random.sample(range(len(indices)), 2)
            d = dist_matrix[i, j]
            if min_cam_dist <= d <= max_cam_dist and d > best_dist:
                best_pair = (indices[i], indices[j])
                best_dist = d
        if best_pair is not None:
            pairs.append(best_pair)

    return pairs[:num_pairs]


def main():
    parser = argparse.ArgumentParser(description="Visualize depth-based warps on CO3D")
    parser.add_argument("--annotation_file", type=str,
                        default="data/co3d_annotations/hydrant_train_50seq_depth.jgz")
    parser.add_argument("--root_dir", type=str,
                        default="/visinf/projects_students/dlcv2025_groupZ/co3d_full")
    parser.add_argument("--num_pairs", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--crop_images", action="store_true")
    parser.add_argument("--output", type=str, default="depth_warp_visualization.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    root_dir = Path(args.root_dir)

    print(f"Loading annotations from {args.annotation_file}...")
    annotations = load_annotations(args.annotation_file)
    samples, sequence_to_indices = build_flat_samples(annotations)
    print(f"Loaded {len(samples)} samples from {len(sequence_to_indices)} sequences")

    pairs = select_example_pairs(sequence_to_indices, samples, args.num_pairs)
    print(f"Selected {len(pairs)} pairs for visualization")

    if not pairs:
        print("No valid pairs found! Check annotation file has depth_path fields.")
        return

    # 8 columns: img_a, img_b, depth_a, depth_b, confidence, warped_b, |A - warped B|, checkerboard
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 8, figsize=(32, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes[None, :]

    col_titles = ["Image A (cropped)", "Image B (cropped)", "Depth A", "Depth B",
                  "Confidence (A→B)", "B warped to A", "|A - warped B|", "Checkerboard"]

    for row, (idx_a, idx_b) in enumerate(pairs):
        sample_a = samples[idx_a]
        sample_b = samples[idx_b]

        # Crop bboxes
        crop_a = get_crop_bbox(sample_a, args.crop_images)
        crop_b = get_crop_bbox(sample_b, args.crop_images)

        # Load images
        img_a_pil = load_and_crop_image(str(root_dir / sample_a["filepath"]), crop_a, args.resolution)
        img_b_pil = load_and_crop_image(str(root_dir / sample_b["filepath"]), crop_b, args.resolution)
        img_a_np = np.array(img_a_pil) / 255.0
        img_b_np = np.array(img_b_pil) / 255.0
        img_a_t = torch.from_numpy(img_a_np).permute(2, 0, 1).float()
        img_b_t = torch.from_numpy(img_b_np).permute(2, 0, 1).float()

        # Load depth
        depth_a = load_co3d_depth(str(root_dir / sample_a["depth_path"]),
                                  sample_a.get("depth_scale_adjustment", 1.0))
        depth_b = load_co3d_depth(str(root_dir / sample_b["depth_path"]),
                                  sample_b.get("depth_scale_adjustment", 1.0))
        depth_mask_a = (depth_a > 0) & np.isfinite(depth_a)
        depth_mask_b = (depth_b > 0) & np.isfinite(depth_b)

        # Image sizes and intrinsics
        image_size_a = get_image_size(sample_a, root_dir)
        image_size_b = get_image_size(sample_b, root_dir)
        K_a = build_intrinsic_matrix(np.array(sample_a["focal_length"]),
                                     np.array(sample_a["principal_point"]), image_size_a)
        K_b = build_intrinsic_matrix(np.array(sample_b["focal_length"]),
                                     np.array(sample_b["principal_point"]), image_size_b)
        R_a, T_a = np.array(sample_a["R"]), np.array(sample_a["T"])
        R_b, T_b = np.array(sample_b["R"]), np.array(sample_b["T"])

        # Compute warp A->B
        warp_ab, conf_ab = compute_depth_warp(
            depth_a, depth_mask_a, R_a, T_a, K_a,
            depth_b, depth_mask_b, R_b, T_b, K_b,
            warp_resolution=args.resolution,
            image_size_a=image_size_a, image_size_b=image_size_b,
            crop_bbox_a=crop_a, crop_bbox_b=crop_b,
        )

        # Warp image B to A's viewpoint using warp_ab
        # warp_ab maps each pixel in A to its corresponding location in B
        # So grid_sample(B, warp_ab) gives "B content at A's pixel locations"
        warped_b = apply_warp(img_b_t, warp_ab)
        warped_b_np = warped_b.permute(1, 2, 0).numpy().clip(0, 1)

        # Mask warped image by confidence
        conf_mask = conf_ab.numpy()
        warped_b_masked = warped_b_np * conf_mask[..., None]

        # Checkerboard
        checker = checkerboard_blend(img_a_t, warped_b, block_size=16)
        checker_np = checker.permute(1, 2, 0).numpy().clip(0, 1)

        # Prepare depth visualization (crop to match)
        if crop_a is not None:
            x1, y1, x2, y2 = crop_a
            H_d, W_d = depth_a.shape
            H_orig, W_orig = image_size_a[1], image_size_a[0]
            # Scale crop to depth map coords
            dx1 = int(x1 / W_orig * W_d)
            dy1 = int(y1 / H_orig * H_d)
            dx2 = int(x2 / W_orig * W_d)
            dy2 = int(y2 / H_orig * H_d)
            depth_a_vis = depth_a[max(0,dy1):min(H_d,dy2), max(0,dx1):min(W_d,dx2)]
        else:
            depth_a_vis = depth_a

        if crop_b is not None:
            x1, y1, x2, y2 = crop_b
            H_d, W_d = depth_b.shape
            H_orig, W_orig = image_size_b[1], image_size_b[0]
            dx1 = int(x1 / W_orig * W_d)
            dy1 = int(y1 / H_orig * H_d)
            dx2 = int(x2 / W_orig * W_d)
            dy2 = int(y2 / H_orig * H_d)
            depth_b_vis = depth_b[max(0,dy1):min(H_d,dy2), max(0,dx1):min(W_d,dx2)]
        else:
            depth_b_vis = depth_b

        # Compute camera distance for title
        pos_a = -R_a.T @ T_a
        pos_b = -R_b.T @ T_b
        cam_dist = np.linalg.norm(pos_a - pos_b)
        valid_pct = conf_ab.mean().item() * 100

        # Plot row
        axes[row, 0].imshow(img_a_np)
        axes[row, 0].set_ylabel(f"Pair {row+1}\ncam_dist={cam_dist:.2f}\nconf={valid_pct:.1f}%",
                                fontsize=9)

        axes[row, 1].imshow(img_b_np)

        d_a_valid = depth_a_vis[depth_a_vis > 0]
        if len(d_a_valid) > 0:
            axes[row, 2].imshow(depth_a_vis, cmap="turbo",
                                vmin=np.percentile(d_a_valid, 2), vmax=np.percentile(d_a_valid, 98))
        else:
            axes[row, 2].imshow(depth_a_vis, cmap="turbo")

        d_b_valid = depth_b_vis[depth_b_vis > 0]
        if len(d_b_valid) > 0:
            axes[row, 3].imshow(depth_b_vis, cmap="turbo",
                                vmin=np.percentile(d_b_valid, 2), vmax=np.percentile(d_b_valid, 98))
        else:
            axes[row, 3].imshow(depth_b_vis, cmap="turbo")

        axes[row, 4].imshow(conf_mask, cmap="gray", vmin=0, vmax=1)

        axes[row, 5].imshow(warped_b_masked)

        # Difference image: |A - warped B| amplified, only where confident
        diff = np.abs(img_a_np - warped_b_np) * conf_mask[..., None]
        # Amplify for visibility
        diff_vis = (diff * 3.0).clip(0, 1)
        axes[row, 6].imshow(diff_vis)

        axes[row, 7].imshow(checker_np)

        for c in range(8):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
            if row == 0:
                axes[row, c].set_title(col_titles[c], fontsize=10)

    plt.suptitle("CO3D Depth-Based Warp Visualization (with cropping)" if args.crop_images
                 else "CO3D Depth-Based Warp Visualization", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
