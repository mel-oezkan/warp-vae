"""
Phase 3 (test): Compute warp fields between CO3D image pairs using DepthAnything-V3
predicted depth (instead of CO3D ground-truth depth or RoMA correspondences).

Pipeline per pair (A, B):
  1. Predict DA3 depth for A and B (single-image inference).
  2. Resize DA3 depth to the original image resolution.
  3. Affinely align DA3 depth to CO3D GT depth on each frame
     (least-squares fit on valid pixels) — DA3 is scale-ambiguous, but the
     warp geometry needs depth in the same metric units as CO3D camera poses.
  4. Feed the aligned depth into the existing `compute_depth_warp` from
     scripts/warps/precompute_depth_warps.py (geometry is unchanged — only the
     depth source is swapped).
  5. Save .pt warp files in the same format as precompute_depth_warps.py and
     write a side-by-side visualisation (A, B, B->A using GT warp, B->A using
     DA3 warp) for each pair.

Usage:
    python scripts/warps/precompute_da3_warps.py \\
        --annotation_file data/co3d_annotations/hydrant_train_50seq_depth.jgz \\
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \\
        --output_dir eval_outputs/da3_warps_phase1 \\
        --num_sequences 5 --frames_per_sequence 4
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from depth_anything_3.api import DepthAnything3

# Reuse all the CO3D-depth pipeline plumbing — only the depth source changes.
sys.path.insert(0, str(Path(__file__).parent))
from warps.precompute_depth_warps import (  # noqa: E402
    build_intrinsic_matrix,
    compute_depth_warp,
    compute_sequence_distance_matrix,
    load_annotations,
    load_co3d_depth,
)
from preprocessing.predict_depth_anything import align_scale  # noqa: E402

from data_process.co3d_dataset import square_bbox  # noqa: E402


def select_close_frames(annotations, num_sequences, frames_per_sequence):
    """For pair-warping we need camera-near frames (cap is ~1.0 units).
    Pick `frames_per_sequence` consecutive frame indices starting from
    the middle of each sequence so adjacent pairs fall inside the range."""
    selected = []
    for seq_name in list(annotations.keys())[:num_sequences]:
        frames = annotations[seq_name]
        if len(frames) < frames_per_sequence:
            continue
        start = max(0, len(frames) // 2 - frames_per_sequence // 2)
        for i in range(start, start + frames_per_sequence):
            f = dict(frames[i])
            f["sequence_key"] = seq_name
            f["frame_index"] = i
            selected.append(f)
    return selected


# ---------------------------------------------------------------------------
# DA3 helpers
# ---------------------------------------------------------------------------


def predict_da3_depth(model, rgb_np, process_res=504):
    pred = model.inference(
        [rgb_np],
        process_res=process_res,
        process_res_method="upper_bound_resize",
        export_dir=None,
    )
    return pred.depth[0].astype(np.float32)


def resize_to(arr, target_hw):
    """Bilinear resize a (H, W) float array to target (H, W)."""
    H, W = target_hw
    return np.array(Image.fromarray(arr).resize((W, H), Image.BILINEAR)).astype(
        np.float32
    )


def aligned_da3_depth(da3_depth, gt_depth):
    """Resize DA3 depth to GT shape, then affinely align to GT scale."""
    da3_at_gt = resize_to(da3_depth, gt_depth.shape)
    valid = (gt_depth > 0) & np.isfinite(gt_depth) & np.isfinite(da3_at_gt)
    aligned, s, t = align_scale(da3_at_gt, gt_depth, valid)
    # The fit can produce negatives in the background; clamp to a tiny
    # positive value so the unprojection stays well-defined.
    aligned = np.where(aligned > 1e-3, aligned, 0.0).astype(np.float32)
    return aligned, float(s), float(t)


# ---------------------------------------------------------------------------
# Pair selection (same approach as visualize_depth_warps.py: pairs within
# the small batch we already chose)
# ---------------------------------------------------------------------------


def build_pairs(frames, pairs_per_sequence=2, min_cam_dist=0.05, max_cam_dist=1.0):
    """Within each sequence, pick `pairs_per_sequence` pairs whose camera
    distance falls in [min_cam_dist, max_cam_dist] — same range used by
    `precompute_depth_warps.py` (see Warp_VAE_Training.md). Pairs that are
    too far apart give ~0 overlap once cropped to the foreground bbox."""
    by_seq = {}
    for i, f in enumerate(frames):
        by_seq.setdefault(f["sequence_key"], []).append(i)

    pairs = []
    for seq, idxs in by_seq.items():
        if len(idxs) < 2:
            continue
        seq_frames = [frames[i] for i in idxs]
        D = compute_sequence_distance_matrix(seq_frames)
        cand = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                d = D[a, b]
                if min_cam_dist <= d <= max_cam_dist:
                    cand.append((d, idxs[a], idxs[b]))
        # Prefer larger distance within the valid range (more visible parallax)
        cand.sort(reverse=True)
        for _, ia, ib in cand[:pairs_per_sequence]:
            pairs.append((ia, ib))
    return pairs


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def apply_warp(image_t, warp_t):
    """Warp (C, H, W) image with grid (H, W, 2) in normalised [-1, 1]."""
    img = image_t.unsqueeze(0).float()
    grid = warp_t.unsqueeze(0).float()

    # check if image is in the correct shape
    if img.shape[2:] != grid.shape[1:3]:
        # sclae im to match grid
        img = F.interpolate(
            img, size=tuple(grid.shape[1:3]), mode="bilinear", align_corners=False
        )

    # warp the image using the grid
    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(0)


def crop_resize(rgb_np, crop_bbox, resolution):
    """Simple crop+resize to match the same input that was fed to the warp computation."""
    img = Image.fromarray(rgb_np)
    if crop_bbox is not None:
        x1, y1, x2, y2 = crop_bbox
        img = img.crop((int(x1), int(y1), int(x2), int(y2)))
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return np.array(img)


def to_tensor(rgb_np):
    return torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0


def visualise_pair(
    img_a_np,
    img_b_np,
    warp_gt_ab,
    conf_gt_ab,
    warp_da3_ab,
    conf_da3_ab,
    out_path,
    title,
):
    img_b_t = to_tensor(img_b_np)
    warped_gt = apply_warp(img_b_t, warp_gt_ab).permute(1, 2, 0).numpy()
    warped_da3 = apply_warp(img_b_t, warp_da3_ab).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.7))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(img_a_np)
    axes[0].set_title("A", fontsize=9)

    axes[1].imshow(img_b_np)
    axes[1].set_title("B", fontsize=9)

    axes[2].imshow(np.clip(warped_gt, 0, 1))
    axes[2].set_title(
        f"B→A (GT depth)\nconf={float(conf_gt_ab.mean()):.2f}", fontsize=8
    )

    axes[3].imshow(np.clip(warped_da3, 0, 1))
    axes[3].set_title(
        f"B→A (DA3 depth)\nconf={float(conf_da3_ab.mean()):.2f}", fontsize=8
    )

    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_file", required=True)
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_sequences", type=int, default=5)
    ap.add_argument("--frames_per_sequence", type=int, default=4)
    ap.add_argument("--pairs_per_sequence", type=int, default=2)
    ap.add_argument("--warp_resolution", type=int, default=256)
    ap.add_argument("--crop_images", action="store_true", default=True)
    ap.add_argument("--depth_consistency_threshold", type=float, default=0.1)
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    ap.add_argument("--process_res", type=int, default=504)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "warps").mkdir(parents=True, exist_ok=True)
    (out_dir / "vis").mkdir(parents=True, exist_ok=True)
    root = Path(args.root_dir)

    print(f"Loading annotations from {args.annotation_file}")
    annotations = load_annotations(args.annotation_file)
    frames = select_close_frames(
        annotations, args.num_sequences, args.frames_per_sequence
    )
    print(f"Selected {len(frames)} frames; building pairs...")
    pairs = build_pairs(frames, pairs_per_sequence=args.pairs_per_sequence)
    print(f"Built {len(pairs)} pairs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} onto {device}")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    # Cache depth & intrinsics per frame index so we don't redo work.
    cache = {}

    def get_frame_data(idx):
        if idx in cache:
            return cache[idx]
        f = frames[idx]
        img_path = root / f["filepath"]
        depth_path = root / f["depth_path"]
        rgb = np.array(Image.open(img_path).convert("RGB"))
        gt_depth = load_co3d_depth(
            str(depth_path), f.get("depth_scale_adjustment", 1.0)
        )
        da3 = predict_da3_depth(model, rgb, process_res=args.process_res)
        da3_aligned, s, t = aligned_da3_depth(da3, gt_depth)
        image_size = tuple(f["image_size"])  # (W, H)
        K = build_intrinsic_matrix(
            np.array(f["focal_length"]),
            np.array(f["principal_point"]),
            image_size,
        )
        bbox = None
        if args.crop_images and "bbox" in f:
            bbox = np.around(square_bbox(np.array(f["bbox"]))).astype(int)
        data = dict(
            f=f,
            rgb=rgb,
            gt=gt_depth,
            da3_aligned=da3_aligned,
            K=K,
            R=np.array(f["R"]),
            T=np.array(f["T"]),
            image_size=image_size,
            bbox=bbox,
            scale=s,
            shift=t,
        )
        cache[idx] = data
        return data

    metrics = []
    for ia, ib in pairs:
        A = get_frame_data(ia)
        B = get_frame_data(ib)
        seq = A["f"]["sequence_key"]
        tag = f"{seq}_{ia:05d}_{ib:05d}"

        valid_mask_a = (A["gt"] > 0) & np.isfinite(A["gt"])
        valid_mask_b = (B["gt"] > 0) & np.isfinite(B["gt"])

        # Two warps: GT-depth reference and DA3-depth candidate
        warp_gt_ab, conf_gt_ab = compute_depth_warp(
            A["gt"],
            valid_mask_a,
            A["R"],
            A["T"],
            A["K"],
            B["gt"],
            valid_mask_b,
            B["R"],
            B["T"],
            B["K"],
            warp_resolution=args.warp_resolution,
            image_size_a=A["image_size"],
            image_size_b=B["image_size"],
            depth_consistency_threshold=args.depth_consistency_threshold,
            crop_bbox_a=A["bbox"],
            crop_bbox_b=B["bbox"],
        )
        warp_gt_ba, conf_gt_ba = compute_depth_warp(
            B["gt"],
            valid_mask_b,
            B["R"],
            B["T"],
            B["K"],
            A["gt"],
            valid_mask_a,
            A["R"],
            A["T"],
            A["K"],
            warp_resolution=args.warp_resolution,
            image_size_a=B["image_size"],
            image_size_b=A["image_size"],
            depth_consistency_threshold=args.depth_consistency_threshold,
            crop_bbox_a=B["bbox"],
            crop_bbox_b=A["bbox"],
        )

        valid_da3_a = (A["da3_aligned"] > 0) & np.isfinite(A["da3_aligned"])
        valid_da3_b = (B["da3_aligned"] > 0) & np.isfinite(B["da3_aligned"])
        warp_da3_ab, conf_da3_ab = compute_depth_warp(
            A["da3_aligned"],
            valid_da3_a,
            A["R"],
            A["T"],
            A["K"],
            B["da3_aligned"],
            valid_da3_b,
            B["R"],
            B["T"],
            B["K"],
            warp_resolution=args.warp_resolution,
            image_size_a=A["image_size"],
            image_size_b=B["image_size"],
            depth_consistency_threshold=args.depth_consistency_threshold,
            crop_bbox_a=A["bbox"],
            crop_bbox_b=B["bbox"],
        )
        warp_da3_ba, conf_da3_ba = compute_depth_warp(
            B["da3_aligned"],
            valid_da3_b,
            B["R"],
            B["T"],
            B["K"],
            A["da3_aligned"],
            valid_da3_a,
            A["R"],
            A["T"],
            A["K"],
            warp_resolution=args.warp_resolution,
            image_size_a=B["image_size"],
            image_size_b=A["image_size"],
            depth_consistency_threshold=args.depth_consistency_threshold,
            crop_bbox_a=B["bbox"],
            crop_bbox_b=A["bbox"],
        )

        torch.save(
            {
                "warp_ab": warp_da3_ab,
                "confidence_ab": conf_da3_ab,
                "warp_ba": warp_da3_ba,
                "confidence_ba": conf_da3_ba,
            },
            out_dir / "warps" / f"warp_{ia:05d}_{ib:05d}.pt",
        )

        # Visualise: crop+resize the RGBs the same way warps were computed
        img_a_vis = crop_resize(A["rgb"], A["bbox"], args.warp_resolution)
        img_b_vis = crop_resize(B["rgb"], B["bbox"], args.warp_resolution)
        visualise_pair(
            img_a_vis,
            img_b_vis,
            warp_gt_ab,
            conf_gt_ab,
            warp_da3_ab,
            conf_da3_ab,
            out_dir / "vis" / f"{tag}.png",
            title=f"{seq}  pair ({ia},{ib})  scaleA={A['scale']:.2f} scaleB={B['scale']:.2f}",
        )

        # Compare warps where both are confident
        both = (conf_gt_ab.numpy() > 0) & (conf_da3_ab.numpy() > 0)
        if both.any():
            diff = (warp_gt_ab.numpy() - warp_da3_ab.numpy())[both]
            warp_l1 = float(np.mean(np.abs(diff)))
        else:
            warp_l1 = float("nan")
        metrics.append(
            {
                "pair": (ia, ib),
                "tag": tag,
                "conf_gt": float(conf_gt_ab.mean()),
                "conf_da3": float(conf_da3_ab.mean()),
                "warp_l1_vs_gt": warp_l1,
            }
        )
        print(
            f"  {tag}: conf GT={conf_gt_ab.mean():.2f}  DA3={conf_da3_ab.mean():.2f}  "
            f"warp L1 vs GT={warp_l1:.4f}"
        )

    # Summary
    if metrics:
        mean_conf_gt = np.mean([m["conf_gt"] for m in metrics])
        mean_conf_da3 = np.mean([m["conf_da3"] for m in metrics])
        mean_l1 = np.nanmean([m["warp_l1_vs_gt"] for m in metrics])
        print(f"\nSummary over {len(metrics)} pairs:")
        print(f"  mean confidence  GT={mean_conf_gt:.3f}  DA3={mean_conf_da3:.3f}")
        print(f"  mean warp L1 (DA3 vs GT, normalised coords): {mean_l1:.4f}")
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
