"""
Sweep DA3-warp quality vs camera distance, per sequence.

For each of N sequences, sample up to M pairs spread across camera-distance
bins, compute (a) GT-depth warp confidence, (b) DA3-depth warp confidence,
(c) DA3 vs GT warp L1 in normalised grid coords, and aggregate per (sequence,
distance bin). Goal: find the distance range where DA3 warps are most useful,
and check whether the optimum is the same across sequences.

Outputs:
  <output_dir>/per_pair.csv       — every pair: seq, dist, conf_gt, conf_da3, L1
  <output_dir>/per_sequence.csv   — mean per (sequence, bin)
  <output_dir>/per_bin.csv        — mean across all sequences per bin
  <output_dir>/sweep.png          — conf + L1 vs distance, one line per sequence

Usage:
    python scripts/sweeps/sweep_da3_distance.py \\
        --annotation_file data/co3d_annotations/hydrant_train_50seq_depth.jgz \\
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \\
        --output_dir eval_outputs/da3_distance_sweep \\
        --num_sequences 15 --pairs_per_sequence 50
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3

sys.path.insert(0, str(Path(__file__).parent))
from warps.precompute_depth_warps import (  # noqa: E402
    build_intrinsic_matrix,
    compute_depth_warp,
    compute_sequence_distance_matrix,
    load_annotations,
    load_co3d_depth,
)
from warps.precompute_da3_warps import (  # noqa: E402
    aligned_da3_depth,
    predict_da3_depth,
)
from data_process.co3d_dataset import square_bbox  # noqa: E402


def stratified_pairs(seq_frames, n_pairs, bin_edges, rng):
    """Return up to n_pairs pairs sampled across distance bins (round-robin
    so under-populated bins still contribute when possible)."""
    D = compute_sequence_distance_matrix(seq_frames)
    n = len(seq_frames)
    by_bin = {b: [] for b in range(len(bin_edges) - 1)}
    for a in range(n):
        for b in range(a + 1, n):
            d = float(D[a, b])
            for bi in range(len(bin_edges) - 1):
                if bin_edges[bi] <= d < bin_edges[bi + 1]:
                    by_bin[bi].append((a, b, d))
                    break
    for v in by_bin.values():
        rng.shuffle(v)

    chosen = []
    cursors = {b: 0 for b in by_bin}
    while len(chosen) < n_pairs:
        progressed = False
        for bi in by_bin:
            if cursors[bi] < len(by_bin[bi]):
                chosen.append((bi, *by_bin[bi][cursors[bi]]))
                cursors[bi] += 1
                progressed = True
                if len(chosen) >= n_pairs:
                    break
        if not progressed:
            break
    return chosen


def get_frame_data(model, frame, root, process_res):
    img_path = root / frame["filepath"]
    depth_path = root / frame["depth_path"]
    rgb = np.array(Image.open(img_path).convert("RGB"))
    gt = load_co3d_depth(str(depth_path), frame.get("depth_scale_adjustment", 1.0))
    da3 = predict_da3_depth(model, rgb, process_res=process_res)
    da3_aligned, _, _ = aligned_da3_depth(da3, gt)
    image_size = tuple(frame["image_size"])
    K = build_intrinsic_matrix(
        np.array(frame["focal_length"]),
        np.array(frame["principal_point"]),
        image_size,
    )
    bbox = None
    if "bbox" in frame:
        bbox = np.around(square_bbox(np.array(frame["bbox"]))).astype(int)
    return dict(
        gt=gt, da3_aligned=da3_aligned, K=K,
        R=np.array(frame["R"]), T=np.array(frame["T"]),
        image_size=image_size, bbox=bbox,
    )


def warp_pair(A, B, warp_resolution, depth_consistency_threshold):
    valid_a_gt = (A["gt"] > 0) & np.isfinite(A["gt"])
    valid_b_gt = (B["gt"] > 0) & np.isfinite(B["gt"])
    warp_gt, conf_gt = compute_depth_warp(
        A["gt"], valid_a_gt, A["R"], A["T"], A["K"],
        B["gt"], valid_b_gt, B["R"], B["T"], B["K"],
        warp_resolution=warp_resolution,
        image_size_a=A["image_size"], image_size_b=B["image_size"],
        depth_consistency_threshold=depth_consistency_threshold,
        crop_bbox_a=A["bbox"], crop_bbox_b=B["bbox"],
    )
    valid_a_da3 = (A["da3_aligned"] > 0) & np.isfinite(A["da3_aligned"])
    valid_b_da3 = (B["da3_aligned"] > 0) & np.isfinite(B["da3_aligned"])
    warp_da3, conf_da3 = compute_depth_warp(
        A["da3_aligned"], valid_a_da3, A["R"], A["T"], A["K"],
        B["da3_aligned"], valid_b_da3, B["R"], B["T"], B["K"],
        warp_resolution=warp_resolution,
        image_size_a=A["image_size"], image_size_b=B["image_size"],
        depth_consistency_threshold=depth_consistency_threshold,
        crop_bbox_a=A["bbox"], crop_bbox_b=B["bbox"],
    )
    return warp_gt, conf_gt, warp_da3, conf_da3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_file", required=True)
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_sequences", type=int, default=15)
    ap.add_argument("--pairs_per_sequence", type=int, default=50)
    ap.add_argument("--warp_resolution", type=int, default=256)
    ap.add_argument("--depth_consistency_threshold", type=float, default=0.1)
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument(
        "--bin_edges", type=float, nargs="+",
        default=[0.05, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root_dir)
    bin_edges = list(args.bin_edges)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    rng = np.random.default_rng(args.seed)

    print(f"Loading annotations from {args.annotation_file}")
    annotations = load_annotations(args.annotation_file)
    seq_names = list(annotations.keys())[:args.num_sequences]
    print(f"Sweeping {len(seq_names)} sequences, up to "
          f"{args.pairs_per_sequence} pairs each, bins={bin_edges}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} onto {device}")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    rows = []  # per-pair records
    for sidx, seq in enumerate(seq_names):
        frames = annotations[seq]
        if len(frames) < 2:
            continue
        chosen = stratified_pairs(frames, args.pairs_per_sequence, bin_edges, rng)
        if not chosen:
            print(f"  [{sidx+1}/{len(seq_names)}] {seq}: no pairs in bins")
            continue
        # All unique frame indices we need
        frame_ids = sorted({a for _, a, _, _ in chosen} | {b for _, _, b, _ in chosen})
        # Predict + cache once per frame
        cache = {}
        for fid in frame_ids:
            cache[fid] = get_frame_data(model, frames[fid], root, args.process_res)

        for bin_idx, ia, ib, dist in chosen:
            A, B = cache[ia], cache[ib]
            try:
                warp_gt, conf_gt, warp_da3, conf_da3 = warp_pair(
                    A, B, args.warp_resolution, args.depth_consistency_threshold
                )
            except Exception as e:
                print(f"    pair ({ia},{ib}) failed: {e}")
                continue
            both = (conf_gt.numpy() > 0) & (conf_da3.numpy() > 0)
            if both.any():
                l1 = float(np.mean(np.abs(
                    warp_gt.numpy() - warp_da3.numpy()
                )[both]))
            else:
                l1 = float("nan")
            rows.append({
                "sequence": seq,
                "frame_a": ia, "frame_b": ib,
                "distance": dist,
                "bin": bin_idx,
                "bin_lo": bin_edges[bin_idx],
                "bin_hi": bin_edges[bin_idx + 1],
                "conf_gt": float(conf_gt.mean()),
                "conf_da3": float(conf_da3.mean()),
                "warp_l1_da3_vs_gt": l1,
            })
        print(f"  [{sidx+1}/{len(seq_names)}] {seq}: {len(chosen)} pairs done "
              f"(unique frames: {len(frame_ids)})")

    # ---- Write CSVs ----
    per_pair_csv = out_dir / "per_pair.csv"
    with open(per_pair_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    by_seq_bin = {}
    for r in rows:
        by_seq_bin.setdefault((r["sequence"], r["bin"]), []).append(r)

    per_seq_csv = out_dir / "per_sequence.csv"
    with open(per_seq_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "bin", "bin_lo", "bin_hi", "n",
                    "mean_conf_gt", "mean_conf_da3", "mean_warp_l1"])
        for (seq, bi), group in sorted(by_seq_bin.items()):
            w.writerow([
                seq, bi, bin_edges[bi], bin_edges[bi+1], len(group),
                float(np.mean([g["conf_gt"] for g in group])),
                float(np.mean([g["conf_da3"] for g in group])),
                float(np.nanmean([g["warp_l1_da3_vs_gt"] for g in group])),
            ])

    by_bin = {}
    for r in rows:
        by_bin.setdefault(r["bin"], []).append(r)
    per_bin_csv = out_dir / "per_bin.csv"
    with open(per_bin_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "bin_lo", "bin_hi", "n",
                    "mean_conf_gt", "mean_conf_da3", "mean_warp_l1"])
        for bi in sorted(by_bin):
            group = by_bin[bi]
            w.writerow([
                bi, bin_edges[bi], bin_edges[bi+1], len(group),
                float(np.mean([g["conf_gt"] for g in group])),
                float(np.mean([g["conf_da3"] for g in group])),
                float(np.nanmean([g["warp_l1_da3_vs_gt"] for g in group])),
            ])

    # ---- Plot: confidence and L1 per sequence vs bin centre ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seq in seq_names:
        xs, conf_da3, l1 = [], [], []
        for bi in range(len(bin_edges) - 1):
            group = by_seq_bin.get((seq, bi))
            if not group:
                continue
            xs.append(bin_centers[bi])
            conf_da3.append(np.mean([g["conf_da3"] for g in group]))
            l1.append(np.nanmean([g["warp_l1_da3_vs_gt"] for g in group]))
        if xs:
            axes[0].plot(xs, conf_da3, marker="o", alpha=0.6, label=seq[:14])
            axes[1].plot(xs, l1, marker="o", alpha=0.6, label=seq[:14])

    # Aggregate line on top
    agg_x, agg_conf, agg_l1 = [], [], []
    for bi in sorted(by_bin):
        agg_x.append(bin_centers[bi])
        agg_conf.append(np.mean([g["conf_da3"] for g in by_bin[bi]]))
        agg_l1.append(np.nanmean([g["warp_l1_da3_vs_gt"] for g in by_bin[bi]]))
    axes[0].plot(agg_x, agg_conf, "k-", linewidth=2.5, label="mean")
    axes[1].plot(agg_x, agg_l1, "k-", linewidth=2.5, label="mean")

    axes[0].set_xlabel("camera distance (units)")
    axes[0].set_ylabel("DA3 warp confidence")
    axes[0].set_title("Confidence vs distance (per sequence)")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("camera distance (units)")
    axes[1].set_ylabel("warp L1 (DA3 vs GT, normalised)")
    axes[1].set_title("DA3-vs-GT warp L1 vs distance")
    axes[1].grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Console summary ----
    print(f"\nWrote {len(rows)} pairs to {per_pair_csv}")
    print(f"Per-sequence aggregate: {per_seq_csv}")
    print(f"Per-bin aggregate: {per_bin_csv}")
    print(f"Plot: {out_dir / 'sweep.png'}")
    print("\nMean across all sequences, per bin:")
    print(f"  {'bin':>20s}  {'n':>4s}  {'conf_gt':>8s}  {'conf_da3':>8s}  {'L1':>8s}")
    for bi in sorted(by_bin):
        group = by_bin[bi]
        print(f"  [{bin_edges[bi]:>5.2f},{bin_edges[bi+1]:>5.2f}]  "
              f"{len(group):>4d}  "
              f"{np.mean([g['conf_gt'] for g in group]):>8.3f}  "
              f"{np.mean([g['conf_da3'] for g in group]):>8.3f}  "
              f"{np.nanmean([g['warp_l1_da3_vs_gt'] for g in group]):>8.4f}")

    # Per-sequence optimum (max conf_da3)
    print("\nPer-sequence optimal bin (max DA3 confidence):")
    for seq in seq_names:
        bins_for_seq = [(bi, by_seq_bin.get((seq, bi))) for bi in range(len(bin_edges)-1)]
        bins_for_seq = [(bi, g) for bi, g in bins_for_seq if g]
        if not bins_for_seq:
            continue
        best_bi, best_g = max(
            bins_for_seq,
            key=lambda x: float(np.mean([g["conf_da3"] for g in x[1]])),
        )
        print(f"  {seq:>30s}: bin [{bin_edges[best_bi]:.2f},{bin_edges[best_bi+1]:.2f}]"
              f"  conf={np.mean([g['conf_da3'] for g in best_g]):.3f}"
              f"  L1={np.nanmean([g['warp_l1_da3_vs_gt'] for g in best_g]):.4f}"
              f"  (n={len(best_g)})")


if __name__ == "__main__":
    main()
