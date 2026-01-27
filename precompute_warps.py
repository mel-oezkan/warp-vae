"""
Precompute RoMaV2 warp fields for faster training.

This script precomputes all warp fields between image pairs and saves them
to disk, eliminating the RoMaV2 computation from the training loop.

Usage:
    python precompute_warps.py \
        --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz \
        --output_dir /data/lab_moezkan/precomputed_warps/toybus \
        --root_dir /data/lab_moezkan/co3d_full \
        --romav2_setting turbo \
        --max_pair_distance 20 \
        --num_pairs_per_sample 3
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import IO, Dict, List, Any, Tuple, cast
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image


def load_samples(bb_file: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Load samples from bounding box file and organize by sequence."""
    samples = []
    sequence_to_indices: Dict[str, List[int]] = {}

    with gzip.GzipFile(bb_file, "rb") as f:
        obj_dict = json.loads(cast(IO, f).read().decode("utf8"))

    idx = 0
    for seq_key, subdir in obj_dict.items():
        sequence_to_indices[seq_key] = []
        for sample in subdir:
            sample["sequence_key"] = seq_key
            samples.append(sample)
            sequence_to_indices[seq_key].append(idx)
            idx += 1

    return samples, sequence_to_indices


def get_pair_candidates(
    idx: int,
    samples: List[Dict[str, Any]],
    sequence_to_indices: Dict[str, List[int]],
    max_pair_distance: int,
    num_pairs: int
) -> List[int]:
    """Get candidate pair indices for a given sample."""
    import random

    sample_data = samples[idx]
    seq_key = sample_data["sequence_key"]
    seq_indices = sequence_to_indices[seq_key]

    if len(seq_indices) < 2:
        return []

    pos = seq_indices.index(idx)
    min_idx = max(0, pos - max_pair_distance)
    max_idx = min(len(seq_indices) - 1, pos + max_pair_distance)

    candidates = [seq_indices[i] for i in range(min_idx, max_idx + 1) if seq_indices[i] != idx]

    if not candidates:
        return []

    # Sample up to num_pairs candidates
    if len(candidates) <= num_pairs:
        return candidates
    else:
        return random.sample(candidates, num_pairs)


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


def main():
    parser = argparse.ArgumentParser(description="Precompute RoMaV2 warp fields")
    parser.add_argument("--bb_file", type=str, required=True,
                        help="Path to bounding box .jgz file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for precomputed warps")
    parser.add_argument("--root_dir", type=str, default="/data/lab_moezkan/co3d_full",
                        help="CO3D dataset root directory")
    parser.add_argument("--romav2_setting", type=str, default="turbo",
                        choices=["turbo", "outdoor", "indoor"],
                        help="RoMaV2 model setting")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for warp computation")
    parser.add_argument("--warp_resolution", type=int, default=256,
                        help="Output warp field resolution")
    parser.add_argument("--max_pair_distance", type=int, default=20,
                        help="Maximum frame distance for pairs")
    parser.add_argument("--num_pairs_per_sample", type=int, default=3,
                        help="Number of pairs to generate per sample")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for RoMaV2")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing files (skip already computed)")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load samples
    print(f"Loading samples from {args.bb_file}...")
    samples, sequence_to_indices = load_samples(args.bb_file)
    print(f"Loaded {len(samples)} samples from {len(sequence_to_indices)} sequences")

    # Initialize RoMaV2
    print(f"Loading RoMaV2 with setting={args.romav2_setting}...")

    # Disable torch.compile for older GPUs
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()

    # Add RoMaV2 to path
    romav2_src = os.path.join(os.path.dirname(__file__), "RoMA2", "src")
    if romav2_src not in sys.path:
        sys.path.insert(0, romav2_src)

    from romav2 import RoMaV2

    device = torch.device(args.device)
    cfg = RoMaV2.Cfg(compile=False, setting=args.romav2_setting)
    model = RoMaV2(cfg=cfg).to(device)
    model.eval()
    print(f"RoMaV2 loaded on {device}")

    # Generate all pairs
    print("Generating pair list...")
    all_pairs = set()
    for idx in range(len(samples)):
        candidates = get_pair_candidates(
            idx, samples, sequence_to_indices,
            args.max_pair_distance, args.num_pairs_per_sample
        )
        for target_idx in candidates:
            # Store as ordered tuple to avoid duplicates
            pair = (min(idx, target_idx), max(idx, target_idx))
            all_pairs.add(pair)

    all_pairs = sorted(list(all_pairs))
    print(f"Total unique pairs to compute: {len(all_pairs)}")

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

    # Compute warps
    print("Computing warps...")
    for idx_a, idx_b in tqdm(all_pairs, desc="Precomputing warps"):
        # Load images
        img_a_path = Path(args.root_dir) / samples[idx_a]["filepath"]
        img_b_path = Path(args.root_dir) / samples[idx_b]["filepath"]

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        # Resize
        img_a = img_a.resize((args.image_size, args.image_size), Image.LANCZOS)
        img_b = img_b.resize((args.image_size, args.image_size), Image.LANCZOS)

        # Compute warp
        try:
            warp_data = compute_warp(model, img_a, img_b, args.warp_resolution)
        except Exception as e:
            print(f"Error computing warp for pair ({idx_a}, {idx_b}): {e}")
            continue

        # Save
        output_file = output_dir / f"warp_{idx_a:05d}_{idx_b:05d}.pt"
        torch.save(warp_data, output_file)

    # Save metadata
    metadata = {
        "bb_file": args.bb_file,
        "root_dir": args.root_dir,
        "romav2_setting": args.romav2_setting,
        "image_size": args.image_size,
        "warp_resolution": args.warp_resolution,
        "max_pair_distance": args.max_pair_distance,
        "num_pairs_per_sample": args.num_pairs_per_sample,
        "num_samples": len(samples),
        "num_pairs": len(all_pairs),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! Saved {len(all_pairs)} warp files to {output_dir}")


if __name__ == "__main__":
    main()
