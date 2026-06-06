"""
Depth + pose reprojection on real CO3D hydrant frames.

Picks one hydrant sequence, selects two frames whose cameras are a moderate
distance apart, and runs the depth-unproject -> world -> reproject pipeline
that `precompute_depth_warps.py` uses in training.  We visualize:

    Frame A | Frame B | Depth(A) | W(A) -> B | Confidence | |W(A) - B|

The point of the demo: even though the inter-frame warp is *not* a group
element, every failure has a geometric explanation that the confidence mask
captures:
    - occlusions: pixels in A that land behind a closer surface in B
      (depth-consistency check fails)
    - disocclusions / out-of-frame: invalid depth or out-of-bounds
    - depth holes (typical in CO3D non-foreground regions): pre-masked

Run:
    conda activate cv
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/demos/depth_warp_hydrant_demo.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

# Use the project's own implementations so the demo and training agree.
sys.path.insert(0, "/visinf/home/lab_mozkan/computer-vision-proj-lab/scripts")
from warps.precompute_depth_warps import (
    build_intrinsic_matrix,
    compute_depth_warp,
    compute_sequence_distance_matrix,
    load_co3d_depth,
    load_co3d_depth_mask,
)


ANNOT = "/visinf/home/lab_mozkan/computer-vision-proj-lab/data/co3d_annotations/hydrant_train_50seq_depth.jgz"
ROOT = Path("/visinf/projects_students/dlcv2025_groupZ/co3d_full")
OUT = "/visinf/home/lab_mozkan/computer-vision-proj-lab/outputs/scripts/depth_warp_hydrant_demo.png"
WARP_RES = 256
TARGET_DIST = 0.4   # camera-position distance we'd like between A and B


def load_image(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def pick_pair(frames: list[dict]) -> tuple[int, int]:
    """Pick two frames whose camera positions are closest to TARGET_DIST apart."""
    D = compute_sequence_distance_matrix(frames)
    n = D.shape[0]
    best, best_score = (0, 1), float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            score = abs(D[i, j] - TARGET_DIST)
            if score < best_score:
                best, best_score = (i, j), score
    return best


def main():
    with gzip.open(ANNOT) as f:
        annots = json.loads(f.read())

    # Pick the first sequence that exists on disk.
    chosen_seq, chosen_frames = None, None
    for seq, frames in annots.items():
        if (ROOT / frames[0]["filepath"]).exists():
            chosen_seq, chosen_frames = seq, frames
            break
    if chosen_seq is None:
        raise SystemExit("No accessible hydrant sequence found under ROOT.")
    print(f"sequence: {chosen_seq}  ({len(chosen_frames)} frames)")

    ia, ib = pick_pair(chosen_frames)
    A, B = chosen_frames[ia], chosen_frames[ib]
    print(f"frames: A=#{ia}  B=#{ib}")

    # Load raw depth + RGB at full resolution; resize RGB to WARP_RES.
    # NOTE: CO3D depth_mask is foreground-only (~7% of pixels) while the depth
    # values themselves are valid across the full scene (~96%).  We follow the
    # training-time code (precompute_depth_warps.py) and use z > 0 as validity.
    depth_a = load_co3d_depth(str(ROOT / A["depth_path"]), A.get("depth_scale_adjustment", 1.0))
    depth_b = load_co3d_depth(str(ROOT / B["depth_path"]), B.get("depth_scale_adjustment", 1.0))
    mask_a = (depth_a > 0) & np.isfinite(depth_a)
    mask_b = (depth_b > 0) & np.isfinite(depth_b)

    img_a = load_image(ROOT / A["filepath"], WARP_RES)
    img_b = load_image(ROOT / B["filepath"], WARP_RES)

    size_a = tuple(A["image_size"])  # (W, H)
    size_b = tuple(B["image_size"])

    K_a = build_intrinsic_matrix(np.array(A["focal_length"]), np.array(A["principal_point"]), size_a)
    K_b = build_intrinsic_matrix(np.array(B["focal_length"]), np.array(B["principal_point"]), size_b)
    R_a, T_a = np.array(A["R"]), np.array(A["T"])
    R_b, T_b = np.array(B["R"]), np.array(B["T"])

    # Inverse warp B<-A: for each pixel in B, where does it come from in A?
    # `compute_depth_warp(A_args, B_args)` returns the *forward* A->B flow
    # sampled on A's grid.  Calling it with arguments swapped gives the flow
    # we want: sampled on B's grid, pointing into A.
    warp_ba, conf_ba = compute_depth_warp(
        depth_b, mask_b, R_b, T_b, K_b,
        depth_a, mask_a, R_a, T_a, K_a,
        warp_resolution=WARP_RES,
        image_size_a=size_b, image_size_b=size_a,
        depth_consistency_threshold=0.1,
    )

    # Use grid_sample to pull A into B's grid.
    img_a_t = torch.from_numpy(img_a).permute(2, 0, 1)[None]           # (1,3,H,W)
    grid = warp_ba[None]                                                # (1,H,W,2)
    w_a_in_b = F.grid_sample(img_a_t, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)[0]
    w_a_in_b = w_a_in_b.permute(1, 2, 0).numpy()                        # (H,W,3)

    conf = conf_ba.numpy()
    mask = conf > 0.5
    residual = np.abs(w_a_in_b - img_b).mean(-1) * mask
    mean_err = residual[mask].mean() if mask.any() else float("nan")
    cov = mask.mean()

    # Build figure.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    axes[0, 0].imshow(img_a); axes[0, 0].set_title(f"Frame A (#{ia})"); axes[0, 0].axis("off")
    axes[0, 1].imshow(img_b); axes[0, 1].set_title(f"Frame B (#{ib})"); axes[0, 1].axis("off")
    # Show depth_a clipped to a sensible range for visibility.
    da_view = depth_a.copy()
    da_view[da_view <= 0] = np.nan
    axes[0, 2].imshow(da_view, cmap="magma"); axes[0, 2].set_title("Depth(A)"); axes[0, 2].axis("off")

    axes[1, 0].imshow(w_a_in_b); axes[1, 0].set_title("W(A) splatted into B"); axes[1, 0].axis("off")
    axes[1, 1].imshow(conf, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Confidence (cov={cov:.0%})"); axes[1, 1].axis("off")
    axes[1, 2].imshow(residual, cmap="inferno", vmin=0, vmax=0.5)
    axes[1, 2].set_title(f"|W(A)-B|  mean={mean_err:.3f}"); axes[1, 2].axis("off")

    fig.suptitle(
        f"Depth + pose reprojection on CO3D hydrant — seq {chosen_seq}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"saved: {OUT}")
    print(f"mean |W(A)-B| inside confidence mask = {mean_err:.4f}   coverage = {cov:.1%}")


if __name__ == "__main__":
    main()
