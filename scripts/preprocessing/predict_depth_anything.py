"""
Phase 1: Generate Depth Anything V3 predictions for a small batch of CO3D frames
and visualise them next to the ground-truth CO3D depth maps for sanity checking.

The goal here is purely diagnostic — we want to eyeball whether predicted depth
is consistent enough (after scale alignment) to substitute for GT depth in the
downstream warp pipeline.

Usage:
    python scripts/preprocessing/predict_depth_anything.py \\
        --annotation_file data/co3d_annotations/hydrant_train_50seq_depth.jgz \\
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \\
        --output_dir eval_outputs/depth_anything_phase1 \\
        --num_sequences 5 --frames_per_sequence 4 \\
        --model depth-anything/DA3-BASE
"""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


def load_annotations(path: str):
    with gzip.open(path, "r") as f:
        return json.loads(f.read())


def load_co3d_depth(depth_path: str, scale_adjustment: float = 1.0) -> np.ndarray:
    depth_pil = Image.open(depth_path)
    depth = (
        np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
        .astype(np.float32)
        .reshape((depth_pil.size[1], depth_pil.size[0]))
    )
    return depth * scale_adjustment


def select_frames(annotations, num_sequences: int, frames_per_sequence: int):
    """Pick a small batch: first N sequences, evenly-spaced frames inside each."""
    selected = []
    for seq_name in list(annotations.keys())[:num_sequences]:
        frames = annotations[seq_name]
        if len(frames) == 0:
            continue
        idx = np.linspace(0, len(frames) - 1, frames_per_sequence).round().astype(int)
        for i in idx:
            f = dict(frames[int(i)])
            f["sequence_key"] = seq_name
            f["frame_index"] = int(i)
            selected.append(f)
    return selected


def align_scale(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    """Solve for s, t minimising ||s*pred + t - gt||_2 on valid pixels.

    DA3 returns depth in an unknown affine-ambiguous scale, so we align it
    to the GT scale before comparing pixel-wise. Returns (aligned_pred, s, t).
    """
    p = pred[valid]
    g = gt[valid]
    if p.size < 100:
        return pred.copy(), 1.0, 0.0

    # Least squares: [p, 1] @ [s, t]^T = g
    A = np.stack([p, np.ones_like(p)], axis=1)
    sol, *_ = np.linalg.lstsq(A, g, rcond=None)
    s, t = float(sol[0]), float(sol[1])
    return s * pred + t, s, t


def visualise_pair(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    rgb: np.ndarray,
    out_path: Path,
    title: str,
):
    """Save a 1x3 panel: RGB | predicted depth | GT depth (compact)."""
    gt_resized = np.array(
        Image.fromarray(gt_depth).resize(
            (pred_depth.shape[1], pred_depth.shape[0]), Image.NEAREST
        )
    )
    valid = (gt_resized > 0) & np.isfinite(gt_resized) & np.isfinite(pred_depth)

    aligned, s, t = align_scale(pred_depth, gt_resized, valid)
    rel_err = np.where(
        valid & (gt_resized > 0), np.abs(aligned - gt_resized) / gt_resized, 0.0
    )
    median_rel = float(np.median(rel_err[valid])) if valid.any() else float("nan")

    rgb_resized = np.array(
        Image.fromarray(rgb).resize(
            (pred_depth.shape[1], pred_depth.shape[0]), Image.BILINEAR
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6))
    axes[0].imshow(rgb_resized)
    axes[0].set_title("RGB", fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    pim = axes[1].imshow(pred_depth, cmap="turbo")
    axes[1].set_title("DA3", fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(pim, ax=axes[1], fraction=0.046)

    gim = axes[2].imshow(np.where(valid, gt_resized, np.nan), cmap="turbo")
    axes[2].set_title("CO3D GT", fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(gim, ax=axes[2], fraction=0.046)

    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return median_rel, s, t, int(valid.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_file", required=True)
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_sequences", type=int, default=5)
    ap.add_argument("--frames_per_sequence", type=int, default=4)
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    ap.add_argument(
        "--process_res",
        type=int,
        default=504,
        help="DA3 internal processing resolution",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "depth_npy").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth_only").mkdir(parents=True, exist_ok=True)
    root = Path(args.root_dir)

    print(f"Loading annotations from {args.annotation_file}")
    annotations = load_annotations(args.annotation_file)
    frames = select_frames(annotations, args.num_sequences, args.frames_per_sequence)
    print(f"Selected {len(frames)} frames from {args.num_sequences} sequences")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} onto {device}")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)


    for f in frames:
        img_path = root / f["filepath"]
        depth_path = root / f["depth_path"]
        seq = f["sequence_key"]
        fi = f["frame_index"]
        tag = f"{seq}_f{fi:04d}"

        rgb_np = np.array(Image.open(img_path).convert("RGB"))

        # Single-image inference. DA3 returns depth at its internal resolution.
        pred = model.inference(
            [rgb_np],
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )
        pred_depth = pred.depth[0].astype(np.float32)  # (H, W)

        gt = load_co3d_depth(str(depth_path), f.get("depth_scale_adjustment", 1.0))

        np.save(out_dir / "depth_npy" / f"{tag}.npy", pred_depth)

        med_rel, s, t, n_valid = visualise_pair(
            pred_depth,
            gt,
            rgb_np,
            out_dir / "depth_only" / f"{tag}.png",
            title=f"{seq} f{fi}",
        )
        print(
            f"  {tag}: median rel err = {med_rel:.3f}  (s={s:.3f}, t={t:.3f}, "
            f"valid px = {n_valid})"
        )
    
if __name__ == "__main__":
    main()
