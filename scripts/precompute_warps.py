"""
Precompute RoMaV2 warp fields for faster training.

This script precomputes all warp fields between image pairs and saves them
to disk, eliminating the RoMaV2 computation from the training loop.

Pair selection uses **stratified camera distance sampling**: the distance
range is divided into bins, and one pair is drawn at random from each bin
per source frame. This prevents the bias toward nearby views that arises
from uniform random sampling (nearby views are more numerous and produce
higher-confidence warps, so they would dominate).

After computing a warp, **cycle-consistency filtering** discards pairs where
the forward-backward warp error is too large (the warp is geometrically
unreliable). This is a purely geometric quality gate, unlike confidence
filtering which conflates overlap area with quality.

Supports multi-GPU acceleration for 1.8-2.0x speedup with 2 GPUs.

Input annotation format: the preprocessed .jgz files produced by
preprocess_co3d.py, organised as {seq_name: [{filepath, R, T, ...}, ...]}.

Usage:
    # Uniform random pairs (like toybus), no cropping, no cycle filter
    CUDA_VISIBLE_DEVICES=0,1 python precompute_warps.py \\
        --annotation_file data/co3d_annotations/hydrant_train_50seq.jgz \\
        --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant_50seq_nocrop \\
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \\
        --pair_mode uniform --num_pairs_per_sample 3 \\
        --num_workers 2 --gpu_ids 0 1 \\
        --cycle_consistency_threshold 0

    # Stratified sampling with cropping + cycle-consistency filter
    CUDA_VISIBLE_DEVICES=0,1 python precompute_warps.py \\
        --annotation_file /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \\
        --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant_cropped \\
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \\
        --pair_mode stratified \\
        --num_workers 2 --gpu_ids 0 1 \\
        --crop_images \\
        --distance_bins 0.5 1.5 2.5 3.0 \\
        --num_pairs_per_bin 1 \\
        --cycle_consistency_threshold 0.1
"""

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_process.co3d_dataset import square_bbox


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotations(annotation_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load CO3D annotations from preprocessed .jgz file.

    Returns:
        Dict mapping sequence_name -> list of frame dicts
        (each with keys: filepath, R, T, focal_length, principal_point, bbox)
    """
    with gzip.open(annotation_file, "r") as f:
        data = json.loads(f.read())
    return data


def build_flat_samples(
    annotations: Dict[str, List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Flatten per-sequence annotations into a flat sample list.

    Returns:
        samples: flat list of frame dicts (augmented with sequence_key)
        sequence_to_indices: seq_name -> list of indices into `samples`
    """
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
# Distance-based pair selection
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


def get_stratified_pair_candidates(
    seq_pos: int,
    seq_indices: List[int],
    dist_matrix: np.ndarray,
    distance_bins: List[float],
    num_pairs_per_bin: int,
) -> List[int]:
    """Return pairs sampled uniformly across distance bins.

    The distance range is split into (len(distance_bins)-1) bins defined by
    the bin-edge values. For each bin, up to `num_pairs_per_bin` targets are
    drawn at random from frames whose camera distance falls in that bin.

    This avoids the bias toward nearby views that arises from picking a fixed
    total number of random pairs from the full range: nearby views are more
    numerous, so they dominate without stratification.

    Args:
        seq_pos: Position of the source frame within the sequence.
        seq_indices: Global flat indices for all frames in the sequence.
        dist_matrix: (N_seq, N_seq) pairwise distance matrix for the sequence.
        distance_bins: Sorted bin edges, e.g. [0.5, 1.5, 2.5, 3.0].
        num_pairs_per_bin: How many pairs to sample per bin.

    Returns:
        List of global flat indices of selected pair partners.
    """
    dists = dist_matrix[seq_pos]
    selected: List[int] = []

    for bin_lo, bin_hi in zip(distance_bins[:-1], distance_bins[1:]):
        candidates_in_bin = [
            j for j in range(len(seq_indices))
            if j != seq_pos and bin_lo <= dists[j] < bin_hi
        ]
        if not candidates_in_bin:
            continue
        k = min(num_pairs_per_bin, len(candidates_in_bin))
        chosen = random.sample(candidates_in_bin, k)
        selected.extend(seq_indices[j] for j in chosen)

    return selected


# ---------------------------------------------------------------------------
# Warp computation helpers
# ---------------------------------------------------------------------------

def compute_warp(
    model,
    img_a: Image.Image,
    img_b: Image.Image,
    warp_resolution: int = 256
) -> Dict[str, torch.Tensor]:
    """Compute RoMaV2 warp field between two images."""
    with torch.no_grad():
        pred_ab = model.match(img_a, img_b)
        pred_ba = model.match(img_b, img_a)

    # Extract warp fields
    warp_ab = pred_ab["warp_AB"].squeeze(0).cpu()
    if pred_ab["overlap_AB"] is not None:
        confidence_ab = pred_ab["overlap_AB"].squeeze(0).squeeze(-1).cpu()
    else:
        confidence_ab = pred_ab["confidence_AB"].squeeze(0).mean(dim=-1).cpu()

    warp_ba = pred_ba["warp_AB"].squeeze(0).cpu()
    if pred_ba["overlap_AB"] is not None:
        confidence_ba = pred_ba["overlap_AB"].squeeze(0).squeeze(-1).cpu()
    else:
        confidence_ba = pred_ba["confidence_AB"].squeeze(0).mean(dim=-1).cpu()

    # Resize to target resolution if needed
    if warp_ab.shape[0] != warp_resolution:
        warp_ab = resize_warp(warp_ab, warp_resolution)
        confidence_ab = F.interpolate(
            confidence_ab.unsqueeze(0).unsqueeze(0),
            size=(warp_resolution, warp_resolution),
            mode="bilinear",
            align_corners=False
        ).squeeze()

        warp_ba = resize_warp(warp_ba, warp_resolution)
        confidence_ba = F.interpolate(
            confidence_ba.unsqueeze(0).unsqueeze(0),
            size=(warp_resolution, warp_resolution),
            mode="bilinear",
            align_corners=False
        ).squeeze()

    return {
        "warp_ab": warp_ab,
        "confidence_ab": confidence_ab,
        "warp_ba": warp_ba,
        "confidence_ba": confidence_ba,
    }


def resize_warp(warp: torch.Tensor, target_size: int) -> torch.Tensor:
    """Resize warp field while maintaining normalized coordinates."""
    warp = warp.permute(2, 0, 1).unsqueeze(0)
    warp = F.interpolate(
        warp,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False
    )
    return warp.squeeze(0).permute(1, 2, 0)


def compute_cycle_consistency_error(
    warp_ab: torch.Tensor,
    warp_ba: torch.Tensor,
) -> float:
    """Compute mean forward-backward cycle-consistency error.

    Samples warp_ba at the positions given by warp_ab; the result should be
    close to the identity grid if the warps are geometrically consistent.
    Error is measured in normalized [-1, 1] coordinates (so 0.1 ≈ 6% of the
    image width).

    Args:
        warp_ab: (H, W, 2) warp from A to B, values in [-1, 1].
        warp_ba: (H, W, 2) warp from B to A, values in [-1, 1].

    Returns:
        Mean cycle error (scalar, in [-1, 1] units).
    """
    H, W = warp_ab.shape[:2]

    # grid_sample expects (N, C, H_in, W_in) input and (N, H_out, W_out, 2) grid
    # warp_ba is the "image" to sample; warp_ab provides the sampling grid
    warp_ba_chw = warp_ba.permute(2, 0, 1).unsqueeze(0)   # (1, 2, H, W)
    grid = warp_ab.unsqueeze(0)                              # (1, H, W, 2)

    # Sample warp_ba at positions defined by warp_ab → should recover identity
    cycle = F.grid_sample(warp_ba_chw, grid, mode="bilinear", align_corners=True)
    cycle = cycle.squeeze(0).permute(1, 2, 0)               # (H, W, 2)

    # Build identity grid in [-1, 1]
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    identity = torch.stack([grid_x, grid_y], dim=-1)        # (H, W, 2)

    error = (cycle - identity).norm(dim=-1).mean().item()
    return error


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_and_crop_image(
    img_path: Path,
    sample: Dict[str, Any],
    image_size: int,
    crop_images: bool,
) -> Image.Image:
    """Load a CO3D image, optionally crop to square bbox, then resize."""
    pil = Image.open(img_path).convert("RGB")
    if crop_images and "bbox" in sample:
        bbox = square_bbox(np.array(sample["bbox"]))
        bbox = np.around(bbox).astype(int)
        # PIL crop: (left, upper, right, lower)
        pil = pil.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    pil = pil.resize((image_size, image_size), Image.LANCZOS)
    return pil


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------

def worker_process_pairs(
    pair_batch: List[Tuple[int, int]],
    samples: List[Dict[str, Any]],
    gpu_id: int,
    root_dir: str,
    output_dir: str,
    image_size: int,
    warp_resolution: int,
    romav2_setting: str,
    crop_images: bool = False,
    cycle_consistency_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Worker function to process a batch of pairs on a specific GPU."""
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    romav2_src = os.path.join(os.path.dirname(__file__), "RoMA2", "src")
    if romav2_src not in sys.path:
        sys.path.insert(0, romav2_src)

    from romav2 import RoMaV2

    cfg = RoMaV2.Cfg(compile=False, setting=romav2_setting)
    model = RoMaV2(cfg=cfg).to(device)
    model.eval()

    stats = {"gpu_id": gpu_id, "processed": 0, "errors": 0, "skipped": 0, "filtered": 0}
    output_dir_path = Path(output_dir)
    root_dir_path = Path(root_dir)

    for idx_a, idx_b in pair_batch:
        output_file = output_dir_path / f"warp_{idx_a:05d}_{idx_b:05d}.pt"

        if output_file.exists():
            stats["skipped"] += 1
            continue

        try:
            img_a = load_and_crop_image(
                root_dir_path / samples[idx_a]["filepath"], samples[idx_a], image_size, crop_images
            )
            img_b = load_and_crop_image(
                root_dir_path / samples[idx_b]["filepath"], samples[idx_b], image_size, crop_images
            )

            warp_data = compute_warp(model, img_a, img_b, warp_resolution)

            if cycle_consistency_threshold is not None:
                cycle_err = compute_cycle_consistency_error(
                    warp_data["warp_ab"], warp_data["warp_ba"]
                )
                if cycle_err > cycle_consistency_threshold:
                    stats["filtered"] += 1
                    continue

            torch.save(warp_data, output_file)
            stats["processed"] += 1

        except Exception as e:
            stats["errors"] += 1
            print(f"[GPU {gpu_id}] Error computing warp for pair ({idx_a}, {idx_b}): {e}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Precompute RoMaV2 warp fields")
    parser.add_argument(
        "--annotation_file", type=str, required=True,
        help="Path to preprocessed CO3D annotation .jgz file (with R/T pose data)"
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for precomputed warps")
    parser.add_argument("--root_dir", type=str,
                        default="/visinf/projects_students/dlcv2025_groupZ/co3d_full",
                        help="CO3D dataset root directory")
    parser.add_argument("--romav2_setting", type=str, default="turbo",
                        choices=["turbo", "fast", "base", "precise"],
                        help="RoMaV2 model setting")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for warp computation")
    parser.add_argument("--warp_resolution", type=int, default=256,
                        help="Output warp field resolution")
    parser.add_argument(
        "--distance_bins", type=float, nargs="+", default=[0.5, 1.5, 2.5, 3.0],
        help="Sorted bin edges for stratified distance sampling, e.g. 0.5 1.5 2.5 3.0 "
             "produces bins [0.5,1.5), [1.5,2.5), [2.5,3.0)"
    )
    parser.add_argument(
        "--num_pairs_per_bin", type=int, default=1,
        help="Number of pairs to sample per distance bin per source frame"
    )
    parser.add_argument(
        "--cycle_consistency_threshold", type=float, default=0.1,
        help="Discard pairs whose mean forward-backward cycle error exceeds this "
             "value (in normalized [-1,1] coords). 0.1 ≈ 6%% of image width. "
             "Set to 0 to disable filtering."
    )
    parser.add_argument(
        "--pair_mode", type=str, default="stratified",
        choices=["stratified", "uniform"],
        help="Pair selection mode: 'stratified' uses camera distance bins, "
             "'uniform' samples random pairs per frame within each sequence "
             "(like the original toybus setup)"
    )
    parser.add_argument(
        "--num_pairs_per_sample", type=int, default=3,
        help="(uniform mode only) Number of random pairs to sample per frame"
    )
    parser.add_argument("--crop_images", action="store_true",
                        help="Crop images to square bbox before computing warps "
                             "(matches crop_images=true in dataset config)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (single-worker mode only)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already computed pairs")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (1 = single GPU)")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="GPU IDs to use (e.g., 0 1). Defaults to first num_workers GPUs")
    args = parser.parse_args()

    random.seed(0)

    # Validate mode-specific args
    if args.pair_mode == "stratified":
        if len(args.distance_bins) < 2:
            parser.error("--distance_bins must have at least 2 values (one bin)")
        if args.distance_bins != sorted(args.distance_bins):
            parser.error("--distance_bins must be sorted in ascending order")

    cycle_threshold: Optional[float] = args.cycle_consistency_threshold if args.cycle_consistency_threshold > 0 else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations (new format with R/T)
    print(f"Loading annotations from {args.annotation_file}...")
    annotations = load_annotations(args.annotation_file)
    samples, sequence_to_indices = build_flat_samples(annotations)
    print(f"Loaded {len(samples)} samples from {len(sequence_to_indices)} sequences")

    # Generate pairs
    all_pairs: set = set()

    if args.pair_mode == "uniform":
        print(f"Generating uniform random pairs: {args.num_pairs_per_sample} pair(s) per frame...")
        for seq_indices in sequence_to_indices.values():
            if len(seq_indices) < 2:
                continue
            for i, global_idx in enumerate(seq_indices):
                others = [j for j in range(len(seq_indices)) if j != i]
                k = min(args.num_pairs_per_sample, len(others))
                chosen = random.sample(others, k)
                for j in chosen:
                    partner = seq_indices[j]
                    pair = (min(global_idx, partner), max(global_idx, partner))
                    all_pairs.add(pair)
    else:
        bins_str = " → ".join(
            f"[{lo},{hi})" for lo, hi in zip(args.distance_bins[:-1], args.distance_bins[1:])
        )
        print(f"Generating stratified pairs: {bins_str}, {args.num_pairs_per_bin} pair(s) per bin per frame...")

        for seq_indices in sequence_to_indices.values():
            frames_in_seq = [samples[i] for i in seq_indices]
            if len(frames_in_seq) < 2:
                continue

            dist_matrix = compute_sequence_distance_matrix(frames_in_seq)

            for seq_pos, global_idx in enumerate(seq_indices):
                candidates = get_stratified_pair_candidates(
                    seq_pos=seq_pos,
                    seq_indices=seq_indices,
                    dist_matrix=dist_matrix,
                    distance_bins=args.distance_bins,
                    num_pairs_per_bin=args.num_pairs_per_bin,
                )
                for partner_idx in candidates:
                    pair = (min(global_idx, partner_idx), max(global_idx, partner_idx))
                    all_pairs.add(pair)

    all_pairs_list = sorted(all_pairs)
    print(f"Total unique pairs to compute: {len(all_pairs_list)}")
    if cycle_threshold is not None:
        print(f"Cycle-consistency filter enabled (threshold={cycle_threshold})")
    else:
        print("Cycle-consistency filter disabled")

    # Filter already computed if resuming
    if args.resume:
        existing: set = set()
        for f in output_dir.glob("warp_*.pt"):
            parts = f.stem.split("_")
            if len(parts) == 3:
                try:
                    existing.add((int(parts[1]), int(parts[2])))
                except ValueError:
                    pass
        all_pairs_list = [p for p in all_pairs_list if p not in existing]
        print(f"Resuming: {len(existing)} already computed, {len(all_pairs_list)} remaining")

    if not all_pairs_list:
        print("No pairs to compute!")
        return

    # Determine GPU IDs
    if args.gpu_ids is None:
        gpu_ids = list(range(args.num_workers))
    else:
        gpu_ids = args.gpu_ids
        if len(gpu_ids) < args.num_workers:
            print(f"Warning: Only {len(gpu_ids)} GPU IDs provided, adjusting num_workers")
            args.num_workers = len(gpu_ids)

    # Single-worker mode
    if args.num_workers == 1:
        print(f"\n=== Single-GPU Mode (GPU {gpu_ids[0]}) ===")
        print(f"Loading RoMaV2 with setting={args.romav2_setting}...")

        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.disable()

        romav2_src = os.path.join(os.path.dirname(__file__), "RoMA2", "src")
        if romav2_src not in sys.path:
            sys.path.insert(0, romav2_src)

        from romav2 import RoMaV2

        device = torch.device(f"cuda:{gpu_ids[0]}")
        cfg = RoMaV2.Cfg(compile=False, setting=args.romav2_setting)
        model = RoMaV2(cfg=cfg).to(device)
        model.eval()
        print(f"RoMaV2 loaded on {device}")

        n_filtered = 0
        print("Computing warps...")
        for idx_a, idx_b in tqdm(all_pairs_list, desc="Precomputing warps"):
            output_file = output_dir / f"warp_{idx_a:05d}_{idx_b:05d}.pt"
            if output_file.exists():
                continue

            img_a = load_and_crop_image(
                Path(args.root_dir) / samples[idx_a]["filepath"], samples[idx_a], args.image_size, args.crop_images
            )
            img_b = load_and_crop_image(
                Path(args.root_dir) / samples[idx_b]["filepath"], samples[idx_b], args.image_size, args.crop_images
            )

            try:
                warp_data = compute_warp(model, img_a, img_b, args.warp_resolution)
            except Exception as e:
                print(f"Error computing warp for pair ({idx_a}, {idx_b}): {e}")
                continue

            if cycle_threshold is not None:
                cycle_err = compute_cycle_consistency_error(
                    warp_data["warp_ab"], warp_data["warp_ba"]
                )
                if cycle_err > cycle_threshold:
                    n_filtered += 1
                    continue

            torch.save(warp_data, output_file)

        if cycle_threshold is not None:
            print(f"Cycle-consistency filter discarded {n_filtered} pairs")

    # Multi-worker mode
    else:
        print(f"\n=== Multi-GPU Mode ({args.num_workers} workers) ===")
        print(f"Using GPUs: {gpu_ids[:args.num_workers]}")

        pairs_per_worker = len(all_pairs_list) // args.num_workers
        pair_batches = []
        for i in range(args.num_workers):
            start_idx = i * pairs_per_worker
            end_idx = len(all_pairs_list) if i == args.num_workers - 1 else (i + 1) * pairs_per_worker
            pair_batches.append(all_pairs_list[start_idx:end_idx])

        print(f"Distributing {len(all_pairs_list)} pairs across {args.num_workers} workers")
        for i, batch in enumerate(pair_batches):
            print(f"  Worker {i} (GPU {gpu_ids[i]}): {len(batch)} pairs")

        worker_fn = partial(
            worker_process_pairs,
            samples=samples,
            root_dir=args.root_dir,
            output_dir=str(output_dir),
            image_size=args.image_size,
            warp_resolution=args.warp_resolution,
            romav2_setting=args.romav2_setting,
            crop_images=args.crop_images,
            cycle_consistency_threshold=cycle_threshold,
        )

        print("\nProcessing pairs...")
        with tqdm(total=len(all_pairs_list), desc="Overall progress") as pbar:
            tasks = [(pair_batches[i], gpu_ids[i]) for i in range(args.num_workers)]

            with Pool(processes=args.num_workers) as pool:
                results = []
                for batch, gpu_id in tasks:
                    result = pool.apply_async(worker_fn, kwds={"pair_batch": batch, "gpu_id": gpu_id})
                    results.append(result)

                all_stats = []
                for result in results:
                    worker_stats = result.get()
                    all_stats.append(worker_stats)
                    pbar.update(worker_stats["processed"])

        print("\n=== Processing Complete ===")
        total_processed = sum(s["processed"] for s in all_stats)
        total_errors = sum(s["errors"] for s in all_stats)
        total_skipped = sum(s["skipped"] for s in all_stats)
        total_filtered = sum(s["filtered"] for s in all_stats)

        for stats in all_stats:
            print(f"GPU {stats['gpu_id']}: {stats['processed']} processed, "
                  f"{stats['filtered']} filtered, "
                  f"{stats['errors']} errors, {stats['skipped']} skipped")

        print(f"Total: {total_processed} processed, {total_filtered} filtered, "
              f"{total_errors} errors, {total_skipped} skipped")

    # Save metadata
    metadata = {
        "annotation_file": args.annotation_file,
        "root_dir": args.root_dir,
        "romav2_setting": args.romav2_setting,
        "image_size": args.image_size,
        "warp_resolution": args.warp_resolution,
        "pair_mode": args.pair_mode,
        "crop_images": args.crop_images,
        "num_samples": len(samples),
        "num_pairs_candidate": len(all_pairs_list),
        "num_workers": args.num_workers,
        "gpu_ids": gpu_ids[:args.num_workers],
    }
    if args.pair_mode == "stratified":
        metadata["distance_bins"] = args.distance_bins
        metadata["num_pairs_per_bin"] = args.num_pairs_per_bin
    else:
        metadata["num_pairs_per_sample"] = args.num_pairs_per_sample
    if cycle_threshold is not None:
        metadata["cycle_consistency_threshold"] = args.cycle_consistency_threshold
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! Saved warp files to {output_dir}")


if __name__ == "__main__":
    main()
