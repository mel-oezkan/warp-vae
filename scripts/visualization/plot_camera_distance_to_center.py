"""Plot per-frame camera distance-to-centroid for many CO3D sequences in one figure.

For each sequence, the "center" is the centroid of its camera positions. For each
frame we compute the Euclidean distance from that camera to the centroid, then
plot all sequences as line plots in a single image. Sequences with the highest/
lowest mean distance are highlighted as outliers, plus a few near the median.

Usage:
    python scripts/visualization/plot_camera_distance_to_center.py
    python scripts/visualization/plot_camera_distance_to_center.py --annotation_path ... --num_sequences 50
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.camera_utils import (
    extract_co3d_camera_positions,
    load_co3d_annotations,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot camera distance-to-center for CO3D sequences")
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="/visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs/camera/hydrant",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=50,
        help="Number of sequences to include (use -1 for all)",
    )
    parser.add_argument(
        "--n_highlight",
        type=int,
        default=3,
        help="How many sequences to highlight at each end (highest/lowest/median).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_co3d_annotations(args.annotation_path)
    seq_names = list(annotations.keys())
    if args.num_sequences > 0:
        seq_names = seq_names[: args.num_sequences]
    print(f"Processing {len(seq_names)} sequences")

    records = []
    for name in seq_names:
        positions = extract_co3d_camera_positions(annotations[name])
        if len(positions) < 2:
            continue
        centroid = positions.mean(axis=0)
        dists = np.linalg.norm(positions - centroid, axis=1)
        frame_idx = np.arange(len(dists)) / max(len(dists) - 1, 1)  # normalized 0..1
        records.append({
            "name": name,
            "positions": positions,
            "centroid": centroid,
            "dists": dists,
            "frame_idx_norm": frame_idx,
            "mean": float(dists.mean()),
            "min": float(dists.min()),
            "max": float(dists.max()),
            "std": float(dists.std()),
            "n_frames": len(dists),
        })

    means = np.array([r["mean"] for r in records])
    order = np.argsort(means)
    k = args.n_highlight
    low_idx = order[:k].tolist()
    high_idx = order[-k:].tolist()
    mid_start = max(0, len(order) // 2 - k // 2)
    median_idx = order[mid_start : mid_start + k].tolist()
    highlight = {i: tag for i, tag in
                 [(i, "low") for i in low_idx]
                 + [(i, "high") for i in high_idx]
                 + [(i, "median") for i in median_idx]}

    colors = {"low": "tab:blue", "high": "tab:red", "median": "tab:green"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2.2, 1]})

    # Left: per-frame distance lines, normalized frame index on x
    for i, r in enumerate(records):
        if i in highlight:
            continue
        ax1.plot(r["frame_idx_norm"], r["dists"], color="lightgray", linewidth=0.7, alpha=0.6, zorder=1)

    for i, tag in highlight.items():
        r = records[i]
        ax1.plot(
            r["frame_idx_norm"], r["dists"],
            color=colors[tag], linewidth=1.8, alpha=0.95, zorder=3,
            label=f"{tag}: {r['name']} (mean={r['mean']:.2f})",
        )

    ax1.set_xlabel("Normalized frame index (0=first, 1=last)")
    ax1.set_ylabel("Distance to sequence centroid")
    ax1.set_title(f"Per-frame camera distance to centroid ({len(records)} sequences)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=7, ncol=1)

    # Right: per-sequence stats summary (sorted by mean)
    sorted_records = [records[i] for i in order]
    xs = np.arange(len(sorted_records))
    means_s = np.array([r["mean"] for r in sorted_records])
    mins_s = np.array([r["min"] for r in sorted_records])
    maxs_s = np.array([r["max"] for r in sorted_records])

    ax2.fill_between(xs, mins_s, maxs_s, color="lightgray", alpha=0.6, label="min..max")
    ax2.plot(xs, means_s, color="black", linewidth=1.2, label="mean")

    for i, tag in highlight.items():
        rank = order.tolist().index(i)
        ax2.scatter(rank, records[i]["mean"], color=colors[tag], s=40, zorder=5,
                    edgecolors="k", linewidths=0.4)

    ax2.set_xlabel("Sequence rank (sorted by mean distance)")
    ax2.set_ylabel("Distance to centroid")
    ax2.set_title("Per-sequence min / mean / max")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("CO3D camera distance-to-centroid summary", fontsize=13)
    fig.tight_layout()

    out_path = output_dir / "distance_to_center_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")

    # --- Trajectory plot (top-down XZ) for all sequences, highlights on top ---
    fig2, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax_all, ax_hi = axes

    all_x = np.concatenate([r["positions"][:, 0] for r in records])
    all_z = np.concatenate([r["positions"][:, 2] for r in records])
    pad_x = (all_x.max() - all_x.min()) * 0.05
    pad_z = (all_z.max() - all_z.min()) * 0.05
    xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    zlim = (all_z.min() - pad_z, all_z.max() + pad_z)

    # Left: all trajectories centered at origin (subtract per-seq centroid) to compare shapes
    for i, r in enumerate(records):
        if i in highlight:
            continue
        p = r["positions"] - r["centroid"]
        ax_all.plot(p[:, 0], p[:, 2], color="lightgray", linewidth=0.6, alpha=0.6, zorder=1)
    for i, tag in highlight.items():
        r = records[i]
        p = r["positions"] - r["centroid"]
        ax_all.plot(p[:, 0], p[:, 2], color=colors[tag], linewidth=1.6, alpha=0.95, zorder=3,
                    label=f"{tag}: {r['name']}")
        ax_all.scatter(p[0, 0], p[0, 2], color=colors[tag], marker="^", s=40,
                       edgecolors="k", linewidths=0.4, zorder=4)
    ax_all.set_xlabel("X - centroid_x")
    ax_all.set_ylabel("Z - centroid_z")
    ax_all.set_title(f"All trajectories (centered, top-down XZ, n={len(records)})")
    ax_all.set_aspect("equal")
    ax_all.grid(True, alpha=0.3)
    ax_all.legend(loc="best", fontsize=7)

    # Right: highlights only, in world coords with start/end markers
    for i, tag in highlight.items():
        r = records[i]
        p = r["positions"]
        ax_hi.plot(p[:, 0], p[:, 2], color=colors[tag], linewidth=1.4, alpha=0.9,
                   label=f"{tag}: {r['name']} (mean={r['mean']:.2f})")
        ax_hi.scatter(p[0, 0], p[0, 2], color=colors[tag], marker="^", s=60,
                      edgecolors="k", linewidths=0.4, zorder=5)
        ax_hi.scatter(p[-1, 0], p[-1, 2], color=colors[tag], marker="s", s=60,
                      edgecolors="k", linewidths=0.4, zorder=5)
        ax_hi.scatter(r["centroid"][0], r["centroid"][2], color=colors[tag], marker="x", s=60,
                      linewidths=1.5, zorder=5)
    ax_hi.set_xlim(xlim)
    ax_hi.set_ylim(zlim)
    ax_hi.set_xlabel("X (world)")
    ax_hi.set_ylabel("Z (world)")
    ax_hi.set_title("Highlighted sequences (world coords) — ▲ start, ■ end, × centroid")
    ax_hi.set_aspect("equal")
    ax_hi.grid(True, alpha=0.3)
    ax_hi.legend(loc="best", fontsize=7)

    fig2.suptitle("CO3D camera trajectories (top-down XZ)", fontsize=13)
    fig2.tight_layout()
    out_path2 = output_dir / "trajectories_summary.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved {out_path2}")

    # Also print a short summary table
    print("\nTop high-mean outliers:")
    for i in reversed(high_idx):
        r = records[i]
        print(f"  {r['name']:30s} mean={r['mean']:.3f}  min={r['min']:.3f}  max={r['max']:.3f}  n={r['n_frames']}")
    print("Low-mean outliers:")
    for i in low_idx:
        r = records[i]
        print(f"  {r['name']:30s} mean={r['mean']:.3f}  min={r['min']:.3f}  max={r['max']:.3f}  n={r['n_frames']}")
    print("Near-median sequences:")
    for i in median_idx:
        r = records[i]
        print(f"  {r['name']:30s} mean={r['mean']:.3f}  min={r['min']:.3f}  max={r['max']:.3f}  n={r['n_frames']}")


if __name__ == "__main__":
    main()
