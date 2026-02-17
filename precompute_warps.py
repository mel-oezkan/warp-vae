"""
Precompute RoMaV2 warp fields for faster training.

This script precomputes all warp fields between image pairs and saves them
to disk, eliminating the RoMaV2 computation from the training loop.

Pair selection uses camera Euclidean distance (computed from R/T poses) rather
than frame-index proximity, so pairs are geometrically meaningful regardless
of capture order.

Supports multi-GPU acceleration for 1.8-2.0x speedup with 2 GPUs.

Input annotation format: the preprocessed .jgz files produced by
preprocess_co3d.py, organised as {seq_name: [{filepath, R, T, ...}, ...]}.

Usage:
    # Single GPU
    python precompute_warps.py \
        --annotation_file /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \
        --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant \
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \
        --romav2_setting turbo \
        --distance_min 0.5 \
        --distance_max 3.0 \
        --num_pairs_per_sample 3

    # Dual GPU (2x speedup)
    python precompute_warps.py \
        --annotation_file /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \
        --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant \
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \
        --num_workers 2 \
        --gpu_ids 0 1 \
        --romav2_setting turbo
"""

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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


def get_distance_pair_candidates(
    seq_pos: int,
    seq_indices: List[int],
    dist_matrix: np.ndarray,
    distance_min: float,
    distance_max: float,
    num_pairs: int,
) -> List[int]:
    """Return up to `num_pairs` sequence positions (within [distance_min, distance_max])
    for a given source frame at `seq_pos` within its sequence.

    Args:
        seq_pos: Position of the source frame within the sequence
        seq_indices: Global flat indices for all frames in the sequence
        dist_matrix: (N_seq, N_seq) pairwise distance matrix for the sequence
        distance_min: Minimum camera distance to accept a pair
        distance_max: Maximum camera distance to accept a pair
        num_pairs: How many pairs to return (random sample if more available)

    Returns:
        List of global flat indices of selected pair partners.
    """
    dists = dist_matrix[seq_pos]
    valid_seq_positions = [
        j for j in range(len(seq_indices))
        if j != seq_pos and distance_min <= dists[j] <= distance_max
    ]
    if not valid_seq_positions:
        return []
    if len(valid_seq_positions) <= num_pairs:
        selected = valid_seq_positions
    else:
        selected = random.sample(valid_seq_positions, num_pairs)
    return [seq_indices[j] for j in selected]


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

    stats = {"gpu_id": gpu_id, "processed": 0, "errors": 0, "skipped": 0}
    output_dir_path = Path(output_dir)
    root_dir_path = Path(root_dir)

    for idx_a, idx_b in pair_batch:
        output_file = output_dir_path / f"warp_{idx_a:05d}_{idx_b:05d}.pt"

        if output_file.exists():
            stats["skipped"] += 1
            continue

        try:
            img_a = Image.open(root_dir_path / samples[idx_a]["filepath"]).convert("RGB")
            img_b = Image.open(root_dir_path / samples[idx_b]["filepath"]).convert("RGB")

            img_a = img_a.resize((image_size, image_size), Image.LANCZOS)
            img_b = img_b.resize((image_size, image_size), Image.LANCZOS)

            warp_data = compute_warp(model, img_a, img_b, warp_resolution)
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
        "--distance_min", type=float, default=0.5,
        help="Minimum camera Euclidean distance for pair selection"
    )
    parser.add_argument(
        "--distance_max", type=float, default=3.0,
        help="Maximum camera Euclidean distance for pair selection"
    )
    parser.add_argument("--num_pairs_per_sample", type=int, default=3,
                        help="Number of pairs to generate per sample")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations (new format with R/T)
    print(f"Loading annotations from {args.annotation_file}...")
    annotations = load_annotations(args.annotation_file)
    samples, sequence_to_indices = build_flat_samples(annotations)
    print(f"Loaded {len(samples)} samples from {len(sequence_to_indices)} sequences")

    # Generate all pairs using camera distance metric
    print("Generating distance-based pair list...")
    all_pairs: set = set()

    for seq_indices in sequence_to_indices.values():
        frames_in_seq = [samples[i] for i in seq_indices]
        if len(frames_in_seq) < 2:
            continue

        dist_matrix = compute_sequence_distance_matrix(frames_in_seq)

        for seq_pos, global_idx in enumerate(seq_indices):
            candidates = get_distance_pair_candidates(
                seq_pos=seq_pos,
                seq_indices=seq_indices,
                dist_matrix=dist_matrix,
                distance_min=args.distance_min,
                distance_max=args.distance_max,
                num_pairs=args.num_pairs_per_sample,
            )
            for partner_idx in candidates:
                pair = (min(global_idx, partner_idx), max(global_idx, partner_idx))
                all_pairs.add(pair)

    all_pairs_list = sorted(all_pairs)
    print(f"Total unique pairs to compute: {len(all_pairs_list)}")

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

        print("Computing warps...")
        for idx_a, idx_b in tqdm(all_pairs_list, desc="Precomputing warps"):
            output_file = output_dir / f"warp_{idx_a:05d}_{idx_b:05d}.pt"
            if output_file.exists():
                continue

            img_a = Image.open(Path(args.root_dir) / samples[idx_a]["filepath"]).convert("RGB")
            img_b = Image.open(Path(args.root_dir) / samples[idx_b]["filepath"]).convert("RGB")
            img_a = img_a.resize((args.image_size, args.image_size), Image.LANCZOS)
            img_b = img_b.resize((args.image_size, args.image_size), Image.LANCZOS)

            try:
                warp_data = compute_warp(model, img_a, img_b, args.warp_resolution)
            except Exception as e:
                print(f"Error computing warp for pair ({idx_a}, {idx_b}): {e}")
                continue

            torch.save(warp_data, output_file)

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

        for stats in all_stats:
            print(f"GPU {stats['gpu_id']}: {stats['processed']} processed, "
                  f"{stats['errors']} errors, {stats['skipped']} skipped")

        print(f"Total: {total_processed} processed, {total_errors} errors, {total_skipped} skipped")

    # Save metadata
    metadata = {
        "annotation_file": args.annotation_file,
        "root_dir": args.root_dir,
        "romav2_setting": args.romav2_setting,
        "image_size": args.image_size,
        "warp_resolution": args.warp_resolution,
        "distance_min": args.distance_min,
        "distance_max": args.distance_max,
        "num_pairs_per_sample": args.num_pairs_per_sample,
        "num_samples": len(samples),
        "num_pairs": len(all_pairs_list),
        "num_workers": args.num_workers,
        "gpu_ids": gpu_ids[:args.num_workers],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! Saved warp files to {output_dir}")


if __name__ == "__main__":
    main()
