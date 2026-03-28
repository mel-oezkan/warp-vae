"""
Precompute depth-based warp fields for multi-view consistency training.

Instead of using RoMaV2 (which produces noisy correspondences), this script
computes geometrically exact flow maps from CO3D ground-truth depth maps and
camera poses via depth unprojection + reprojection.

For each image pair (A, B):
  1. Load depth map for A, unproject each pixel to 3D using camera A intrinsics/extrinsics
  2. Reproject the 3D points into camera B to get the flow field A->B
  3. Do the same in reverse for B->A
  4. Build confidence maps from: valid depth, reprojected depth consistency,
     and in-bounds checks

Supports filtering pairs by camera distance to prefer nearby viewpoints.

Output format is identical to precompute_warps.py (warp_XXXXX_YYYYY.pt files)
so PrecomputedWarpDataset can load them without modification.

Usage:
    python precompute_depth_warps.py \\
        --annotation_file data/co3d_annotations/hydrant_train_50seq.jgz \\
        --output_dir /path/to/precomputed_warps/hydrant_depth \\
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \\
        --max_camera_distance 1.0 \\
        --num_pairs_per_sample 3 \\
        --warp_resolution 256
"""

import argparse
import gzip
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from data_process.co3d_dataset import square_bbox


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotations(annotation_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load CO3D annotations from preprocessed .jgz file."""
    with gzip.open(annotation_file, "r") as f:
        data = json.loads(f.read())
    return data


def build_flat_samples(
    annotations: Dict[str, List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Flatten per-sequence annotations into a flat sample list."""
    samples: List[Dict[str, Any]] = []
    sequence_to_indices: Dict[str, List[int]] = {}

    for seq_name, frames in annotations.items():
        sequence_to_indices[seq_name] = []
        for frame in frames:
            frame = dict(frame)
            frame["sequence_key"] = seq_name
            sequence_to_indices[seq_name].append(len(samples))
            samples.append(frame)

    return samples, sequence_to_indices


# ---------------------------------------------------------------------------
# Depth loading (CO3D format)
# ---------------------------------------------------------------------------

def load_co3d_depth(depth_path: str, scale_adjustment: float = 1.0) -> np.ndarray:
    """Load CO3D depth map from PNG file.

    CO3D stores depth as uint16 PNG encoding float16 values.
    Actual depth = png_value_as_float16 * scale_adjustment.

    Returns:
        depth: (H, W) float32 array in camera-space units.
    """
    depth_pil = Image.open(depth_path)
    depth = (
        np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
        .astype(np.float32)
        .reshape((depth_pil.size[1], depth_pil.size[0]))
    )
    return depth * scale_adjustment


def load_co3d_depth_mask(mask_path: str) -> np.ndarray:
    """Load CO3D depth mask from PNG file.

    Returns:
        mask: (H, W) boolean array. True = valid depth.
    """
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------

def _extract_camera_position(frame: Dict[str, Any]) -> np.ndarray:
    """Compute camera world position from CO3D W2C convention: pos = -R^T @ T."""
    R = np.array(frame["R"])
    T = np.array(frame["T"])
    return -R.T @ T


def compute_sequence_distance_matrix(
    frames: List[Dict[str, Any]]
) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between camera positions."""
    positions = np.stack([_extract_camera_position(f) for f in frames])
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def build_intrinsic_matrix(
    focal_length: np.ndarray,
    principal_point: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Build 3x3 intrinsic matrix from CO3D NDC parameters.

    CO3D / PyTorch3D uses isotropic NDC where both focal_length and
    principal_point are normalized by half_min_image_size:

        x_ndc = -fx * X_cam / Z_cam + px
        y_ndc = -fy * Y_cam / Z_cam + py

    NDC to screen conversion:
        u = -x_ndc * half_min + W/2
        v = -y_ndc * half_min + H/2

    Combining:
        u = fx * half_min * X/Z - px * half_min + W/2
        v = fy * half_min * Y/Z - py * half_min + H/2

    Args:
        focal_length: [fx, fy] in NDC units
        principal_point: [px, py] in NDC units
        image_size: (W, H) of the image

    Returns:
        K: 3x3 intrinsic matrix in pixel coordinates
    """
    W, H = image_size
    half_min = min(W, H) / 2.0

    fx_px = focal_length[0] * half_min
    fy_px = focal_length[1] * half_min
    cx_px = -principal_point[0] * half_min + W / 2.0
    cy_px = -principal_point[1] * half_min + H / 2.0

    K = np.array([
        [fx_px, 0,     cx_px],
        [0,     fy_px, cy_px],
        [0,     0,     1    ],
    ], dtype=np.float64)
    return K


def compute_depth_warp(
    depth_a: np.ndarray,
    depth_mask_a: np.ndarray,
    R_a: np.ndarray,
    T_a: np.ndarray,
    K_a: np.ndarray,
    depth_b: np.ndarray,
    depth_mask_b: np.ndarray,
    R_b: np.ndarray,
    T_b: np.ndarray,
    K_b: np.ndarray,
    warp_resolution: int,
    image_size_a: Tuple[int, int],
    image_size_b: Tuple[int, int],
    depth_consistency_threshold: float = 0.1,
    crop_bbox_a: Optional[np.ndarray] = None,
    crop_bbox_b: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute depth-based warp from image A to image B.

    Steps:
        1. For each pixel in A (at warp_resolution), find its pixel coord in
           the original (uncropped) image.
        2. Look up depth_a at that pixel, unproject to 3D in camera A frame.
        3. Transform 3D point to world frame, then to camera B frame.
        4. Project into image B pixel coordinates.
        5. Convert to normalized [-1, 1] coordinates relative to the
           (possibly cropped) output image.
        6. Build confidence: valid depth, in-bounds in B, depth consistency.

    Args:
        depth_a: (H_orig, W_orig) depth map for image A
        depth_mask_a: (H_orig, W_orig) valid depth mask for A
        R_a, T_a: Camera A extrinsics (CO3D W2C: X_cam = X_world @ R + T)
        K_a: 3x3 intrinsic matrix for A (in original pixel coords)
        depth_b: (H_orig, W_orig) depth map for image B
        depth_mask_b: (H_orig, W_orig) valid depth mask for B
        R_b, T_b: Camera B extrinsics
        K_b: 3x3 intrinsic matrix for B (in original pixel coords)
        warp_resolution: Output warp field resolution
        image_size_a: (W, H) original image size for A
        image_size_b: (W, H) original image size for B
        depth_consistency_threshold: Relative depth error threshold for confidence
        crop_bbox_a: Optional [x1, y1, x2, y2] crop box applied to image A
        crop_bbox_b: Optional [x1, y1, x2, y2] crop box applied to image B

    Returns:
        warp_ab: (warp_resolution, warp_resolution, 2) in normalized [-1, 1]
        confidence_ab: (warp_resolution, warp_resolution) in [0, 1]
    """
    H_a = warp_resolution
    W_a = warp_resolution

    # Build pixel grid for the output (cropped, resized) image A
    # These are pixel coordinates in [0, warp_resolution-1]
    ys = torch.linspace(0.5, warp_resolution - 0.5, H_a)
    xs = torch.linspace(0.5, warp_resolution - 0.5, W_a)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    # (H, W)

    # Map from cropped/resized coords back to original image pixel coords
    if crop_bbox_a is not None:
        x1, y1, x2, y2 = crop_bbox_a
        crop_w = x2 - x1
        crop_h = y2 - y1
        orig_x = grid_x.numpy() / warp_resolution * crop_w + x1
        orig_y = grid_y.numpy() / warp_resolution * crop_h + y1
    else:
        W_orig_a, H_orig_a = image_size_a
        orig_x = grid_x.numpy() / warp_resolution * W_orig_a
        orig_y = grid_y.numpy() / warp_resolution * H_orig_a

    # Sample depth at original pixel locations (nearest neighbor)
    H_depth, W_depth = depth_a.shape
    # Scale to depth map coordinates (depth might differ from image size)
    W_orig_a, H_orig_a = image_size_a
    depth_x = np.clip((orig_x / W_orig_a * W_depth).astype(int), 0, W_depth - 1)
    depth_y = np.clip((orig_y / H_orig_a * H_depth).astype(int), 0, H_depth - 1)
    z_a = depth_a[depth_y, depth_x]  # (H, W)

    # Filter out invalid/zero depth
    # Note: CO3D depth masks are foreground-only (~7% of pixels), but depth
    # values are valid for the entire scene (~96%). We use z > 0 as the
    # primary validity check since we want warps for the full image.
    valid_depth = (z_a > 0) & np.isfinite(z_a)

    # Unproject to 3D in camera A frame
    # x_cam = K_inv @ [u, v, 1]^T * z
    K_a_inv = np.linalg.inv(K_a)
    ones = np.ones_like(orig_x)
    pixel_coords = np.stack([orig_x, orig_y, ones], axis=-1)  # (H, W, 3)
    # Reshape for matrix multiply
    pixel_flat = pixel_coords.reshape(-1, 3)  # (N, 3)
    rays_cam_a = (K_a_inv @ pixel_flat.T).T  # (N, 3)
    z_flat = z_a.reshape(-1)
    points_cam_a = rays_cam_a * z_flat[:, None]  # (N, 3)

    # Transform to world frame: X_cam = X_world @ R + T => X_world = (X_cam - T) @ R^{-1}
    R_a_f = R_a.astype(np.float64)
    T_a_f = T_a.astype(np.float64)
    R_a_inv = np.linalg.inv(R_a_f)
    points_world = (points_cam_a - T_a_f[None, :]) @ R_a_inv  # (N, 3)

    # Transform to camera B frame: X_cam_b = X_world @ R_b + T_b
    R_b_f = R_b.astype(np.float64)
    T_b_f = T_b.astype(np.float64)
    points_cam_b = points_world @ R_b_f + T_b_f[None, :]  # (N, 3)

    # Project into image B pixel coordinates
    z_b_reprojected = points_cam_b[:, 2]  # depth in camera B
    # Avoid division by zero
    z_b_safe = np.where(np.abs(z_b_reprojected) > 1e-6, z_b_reprojected, 1e-6)
    points_cam_b_normalized = points_cam_b / z_b_safe[:, None]
    pixels_b = (K_b @ points_cam_b_normalized.T).T  # (N, 3)
    u_b = pixels_b[:, 0].reshape(H_a, W_a)
    v_b = pixels_b[:, 1].reshape(H_a, W_a)
    z_b_reproj = z_b_reprojected.reshape(H_a, W_a)

    # Convert pixel coords in B to normalized [-1, 1] relative to the
    # (possibly cropped) output image B
    if crop_bbox_b is not None:
        x1_b, y1_b, x2_b, y2_b = crop_bbox_b
        crop_w_b = x2_b - x1_b
        crop_h_b = y2_b - y1_b
        # Pixel coords in crop space
        u_b_crop = (u_b - x1_b) / crop_w_b
        v_b_crop = (v_b - y1_b) / crop_h_b
        # To normalized [-1, 1]
        norm_x = u_b_crop * 2.0 - 1.0
        norm_y = v_b_crop * 2.0 - 1.0
        # In-bounds check relative to crop
        in_bounds = (u_b >= x1_b) & (u_b < x2_b) & (v_b >= y1_b) & (v_b < y2_b)
    else:
        W_orig_b, H_orig_b = image_size_b
        norm_x = u_b / W_orig_b * 2.0 - 1.0
        norm_y = v_b / H_orig_b * 2.0 - 1.0
        in_bounds = (u_b >= 0) & (u_b < W_orig_b) & (v_b >= 0) & (v_b < H_orig_b)

    # Depth consistency: compare reprojected depth with actual depth in B
    W_orig_b, H_orig_b = image_size_b
    H_depth_b, W_depth_b = depth_b.shape
    u_b_depth = np.clip((u_b / W_orig_b * W_depth_b).astype(int), 0, W_depth_b - 1)
    v_b_depth = np.clip((v_b / H_orig_b * H_depth_b).astype(int), 0, H_depth_b - 1)
    z_b_actual = depth_b[v_b_depth, u_b_depth]

    # Relative depth error (only where target depth is valid)
    z_b_valid = (z_b_actual > 0) & np.isfinite(z_b_actual)
    z_b_safe_check = np.where(z_b_actual > 0, z_b_actual, 1.0)
    depth_error = np.abs(z_b_reproj - z_b_actual) / z_b_safe_check
    depth_consistent = depth_error < depth_consistency_threshold

    # Points must be in front of camera B
    in_front = z_b_reproj > 0

    # Build confidence
    confidence = (
        valid_depth &
        in_bounds &
        in_front &
        z_b_valid &
        depth_consistent
    ).astype(np.float32)

    # Build warp tensor
    warp_ab = torch.tensor(
        np.stack([norm_x, norm_y], axis=-1),
        dtype=torch.float32
    )  # (H, W, 2)

    confidence_ab = torch.tensor(confidence, dtype=torch.float32)  # (H, W)

    return warp_ab, confidence_ab


# ---------------------------------------------------------------------------
# Image / depth loading helpers
# ---------------------------------------------------------------------------

def get_crop_bbox(sample: Dict[str, Any], crop_images: bool) -> Optional[np.ndarray]:
    """Get the square crop bbox for a sample, or None if not cropping."""
    if not crop_images or "bbox" not in sample:
        return None
    bbox = square_bbox(np.array(sample["bbox"]))
    return np.around(bbox).astype(int)


def get_image_size(sample: Dict[str, Any], root_dir: Path) -> Tuple[int, int]:
    """Get (W, H) of the original image."""
    if "image_size" in sample:
        return tuple(sample["image_size"])  # [W, H] from preprocessing
    # Fallback: read from file
    img_path = root_dir / sample["filepath"]
    with Image.open(img_path) as img:
        return img.size  # (W, H)


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def select_pairs(
    sequence_to_indices: Dict[str, List[int]],
    samples: List[Dict[str, Any]],
    max_camera_distance: float,
    min_camera_distance: float,
    num_pairs_per_sample: int,
) -> List[Tuple[int, int]]:
    """Select pairs within a camera distance range.

    For each frame, sample up to num_pairs_per_sample partners whose camera
    position is within [min_camera_distance, max_camera_distance].
    """
    all_pairs: set = set()

    for seq_name, seq_indices in sequence_to_indices.items():
        if len(seq_indices) < 2:
            continue

        frames = [samples[i] for i in seq_indices]
        dist_matrix = compute_sequence_distance_matrix(frames)

        for seq_pos, global_idx in enumerate(seq_indices):
            # Find candidates within distance range
            candidates = []
            for j in range(len(seq_indices)):
                if j == seq_pos:
                    continue
                d = dist_matrix[seq_pos, j]
                if min_camera_distance <= d <= max_camera_distance:
                    candidates.append(seq_indices[j])

            if not candidates:
                continue

            k = min(num_pairs_per_sample, len(candidates))
            chosen = random.sample(candidates, k)
            for partner in chosen:
                pair = (min(global_idx, partner), max(global_idx, partner))
                all_pairs.add(pair)

    return sorted(all_pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute depth-based warp fields for multi-view consistency training"
    )
    parser.add_argument(
        "--annotation_file", type=str, required=True,
        help="Path to preprocessed CO3D annotation .jgz file (must include depth_path fields; "
             "re-run preprocess_co3d.py if missing)"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--root_dir", type=str,
        default="/visinf/projects_students/dlcv2025_groupZ/co3d_full",
    )
    parser.add_argument("--warp_resolution", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=256,
                        help="Training image size (for metadata)")
    parser.add_argument(
        "--max_camera_distance", type=float, default=1.0,
        help="Maximum camera distance for pair selection (default: 1.0, "
             "smaller = closer views = less occlusion)"
    )
    parser.add_argument(
        "--min_camera_distance", type=float, default=0.05,
        help="Minimum camera distance (skip near-identical views)"
    )
    parser.add_argument(
        "--num_pairs_per_sample", type=int, default=3,
        help="Number of pairs to sample per frame"
    )
    parser.add_argument(
        "--depth_consistency_threshold", type=float, default=0.1,
        help="Relative depth error threshold for confidence (default: 0.1 = 10%%)"
    )
    parser.add_argument("--crop_images", action="store_true",
                        help="Crop images to square bbox (must match training config)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already computed pairs")
    args = parser.parse_args()

    random.seed(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root_dir = Path(args.root_dir)

    # Load annotations
    print(f"Loading annotations from {args.annotation_file}...")
    annotations = load_annotations(args.annotation_file)
    samples, sequence_to_indices = build_flat_samples(annotations)
    print(f"Loaded {len(samples)} samples from {len(sequence_to_indices)} sequences")

    # Check that depth info is present
    has_depth = sum(1 for s in samples if "depth_path" in s)
    if has_depth == 0:
        print("ERROR: No depth_path fields found in annotations!")
        print("Re-run preprocess_co3d.py to include depth information.")
        return
    print(f"Samples with depth info: {has_depth}/{len(samples)}")

    # Select pairs based on camera distance
    print(f"Selecting pairs with camera distance in [{args.min_camera_distance}, {args.max_camera_distance}]...")
    all_pairs = select_pairs(
        sequence_to_indices, samples,
        max_camera_distance=args.max_camera_distance,
        min_camera_distance=args.min_camera_distance,
        num_pairs_per_sample=args.num_pairs_per_sample,
    )
    print(f"Total unique pairs: {len(all_pairs)}")

    if not all_pairs:
        print("No pairs found! Try increasing --max_camera_distance.")
        return

    # Filter already computed if resuming
    if args.resume:
        existing = set()
        for f in output_dir.glob("warp_*.pt"):
            parts = f.stem.split("_")
            if len(parts) == 3:
                try:
                    existing.add((int(parts[1]), int(parts[2])))
                except ValueError:
                    pass
        all_pairs = [p for p in all_pairs if p not in existing]
        print(f"Resuming: {len(existing)} already computed, {len(all_pairs)} remaining")

    if not all_pairs:
        print("All pairs already computed!")
        return

    # Compute warps
    n_success = 0
    n_skip_no_depth = 0
    n_errors = 0

    for idx_a, idx_b in tqdm(all_pairs, desc="Computing depth warps"):
        output_file = output_dir / f"warp_{idx_a:05d}_{idx_b:05d}.pt"
        if output_file.exists():
            continue

        sample_a = samples[idx_a]
        sample_b = samples[idx_b]

        # Check depth availability
        if "depth_path" not in sample_a or "depth_path" not in sample_b:
            n_skip_no_depth += 1
            continue

        try:
            # Load depth maps
            depth_a = load_co3d_depth(
                str(root_dir / sample_a["depth_path"]),
                sample_a.get("depth_scale_adjustment", 1.0)
            )
            depth_b = load_co3d_depth(
                str(root_dir / sample_b["depth_path"]),
                sample_b.get("depth_scale_adjustment", 1.0)
            )

            # Load depth masks
            if "depth_mask_path" in sample_a:
                depth_mask_a = load_co3d_depth_mask(str(root_dir / sample_a["depth_mask_path"]))
            else:
                depth_mask_a = (depth_a > 0) & np.isfinite(depth_a)

            if "depth_mask_path" in sample_b:
                depth_mask_b = load_co3d_depth_mask(str(root_dir / sample_b["depth_mask_path"]))
            else:
                depth_mask_b = (depth_b > 0) & np.isfinite(depth_b)

            # Get image sizes
            image_size_a = get_image_size(sample_a, root_dir)
            image_size_b = get_image_size(sample_b, root_dir)

            # Build intrinsic matrices
            K_a = build_intrinsic_matrix(
                np.array(sample_a["focal_length"]),
                np.array(sample_a["principal_point"]),
                image_size_a,
            )
            K_b = build_intrinsic_matrix(
                np.array(sample_b["focal_length"]),
                np.array(sample_b["principal_point"]),
                image_size_b,
            )

            # Camera extrinsics
            R_a = np.array(sample_a["R"])
            T_a = np.array(sample_a["T"])
            R_b = np.array(sample_b["R"])
            T_b = np.array(sample_b["T"])

            # Crop bboxes
            crop_bbox_a = get_crop_bbox(sample_a, args.crop_images)
            crop_bbox_b = get_crop_bbox(sample_b, args.crop_images)

            # Compute A -> B warp
            warp_ab, confidence_ab = compute_depth_warp(
                depth_a, depth_mask_a, R_a, T_a, K_a,
                depth_b, depth_mask_b, R_b, T_b, K_b,
                warp_resolution=args.warp_resolution,
                image_size_a=image_size_a,
                image_size_b=image_size_b,
                depth_consistency_threshold=args.depth_consistency_threshold,
                crop_bbox_a=crop_bbox_a,
                crop_bbox_b=crop_bbox_b,
            )

            # Compute B -> A warp
            warp_ba, confidence_ba = compute_depth_warp(
                depth_b, depth_mask_b, R_b, T_b, K_b,
                depth_a, depth_mask_a, R_a, T_a, K_a,
                warp_resolution=args.warp_resolution,
                image_size_a=image_size_b,
                image_size_b=image_size_a,
                depth_consistency_threshold=args.depth_consistency_threshold,
                crop_bbox_a=crop_bbox_b,
                crop_bbox_b=crop_bbox_a,
            )

            # Save in same format as precompute_warps.py
            warp_data = {
                "warp_ab": warp_ab,
                "confidence_ab": confidence_ab,
                "warp_ba": warp_ba,
                "confidence_ba": confidence_ba,
            }
            torch.save(warp_data, output_file)
            n_success += 1

        except Exception as e:
            n_errors += 1
            print(f"Error for pair ({idx_a}, {idx_b}): {e}")

    print(f"\nDone! {n_success} pairs computed, {n_skip_no_depth} skipped (no depth), {n_errors} errors")

    # Save metadata
    metadata = {
        "annotation_file": args.annotation_file,
        "root_dir": args.root_dir,
        "warp_method": "depth_reprojection",
        "image_size": args.image_size,
        "warp_resolution": args.warp_resolution,
        "max_camera_distance": args.max_camera_distance,
        "min_camera_distance": args.min_camera_distance,
        "num_pairs_per_sample": args.num_pairs_per_sample,
        "depth_consistency_threshold": args.depth_consistency_threshold,
        "crop_images": args.crop_images,
        "num_samples": len(samples),
        "num_pairs_computed": n_success,
        "num_pairs_skipped_no_depth": n_skip_no_depth,
        "num_errors": n_errors,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved warp files to {output_dir}")


if __name__ == "__main__":
    main()
