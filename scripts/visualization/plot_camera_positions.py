"""Plot 2D top-down (XZ) camera position maps for CO3D sequences.

Generates one plot per sequence with consistent axis scales across all plots,
so camera scatter patterns can be visually compared.

Usage:
    python scripts/visualization/plot_camera_positions.py
    python scripts/visualization/plot_camera_positions.py --annotation_path /path/to/hydrant_train.jgz --num_sequences 10
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial.distance import pdist

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.camera_utils import (
    extract_co3d_camera_positions,
    load_co3d_annotations,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot 2D camera positions for CO3D sequences")
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="/visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz",
        help="Path to preprocessed CO3D annotation .jgz file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs/camera/hydrant",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=6,
        help="Number of sequences to plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    print(f"Loading annotations from {args.annotation_path}")
    annotations = load_co3d_annotations(args.annotation_path)
    seq_names = list(annotations.keys())[: args.num_sequences]
    print(f"Plotting {len(seq_names)} sequences")

    # Extract positions for all sequences and compute global axis limits
    seq_data = {}
    global_x_min, global_x_max = np.inf, -np.inf
    global_z_min, global_z_max = np.inf, -np.inf

    for name in seq_names:
        frames = annotations[name]
        positions = extract_co3d_camera_positions(frames)  # (N, 3)
        x, z = positions[:, 0], positions[:, 2]

        # Spatial extent: max pairwise Euclidean distance (full 3D)
        extent = pdist(positions).max() if len(positions) > 1 else 0.0

        seq_data[name] = {"positions": positions, "x": x, "z": z, "extent": extent}

        global_x_min = min(global_x_min, x.min())
        global_x_max = max(global_x_max, x.max())
        global_z_min = min(global_z_min, z.min())
        global_z_max = max(global_z_max, z.max())

    # Add padding to axis limits
    pad_x = (global_x_max - global_x_min) * 0.1
    pad_z = (global_z_max - global_z_min) * 0.1
    xlim = (global_x_min - pad_x, global_x_max + pad_x)
    zlim = (global_z_min - pad_z, global_z_max + pad_z)

    # Plot each sequence
    for name, data in seq_data.items():
        x, z = data["x"], data["z"]
        n_frames = len(x)
        extent = data["extent"]

        fig, ax = plt.subplots(figsize=(7, 7))

        # Trajectory line colored by frame order
        points = np.column_stack([x, z]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, n_frames - 1)
        lc = LineCollection(segments, cmap="viridis", norm=norm, linewidths=1.0, alpha=0.5)
        lc.set_array(np.arange(n_frames - 1))
        ax.add_collection(lc)

        # Scatter points colored by frame index
        sc = ax.scatter(x, z, c=np.arange(n_frames), cmap="viridis", s=30, zorder=5, edgecolors="k", linewidths=0.3)
        cbar = fig.colorbar(sc, ax=ax, label="Frame index")

        # Mark start and end
        ax.scatter(x[0], z[0], c="green", s=100, marker="^", zorder=10, edgecolors="k", label="Start")
        ax.scatter(x[-1], z[-1], c="red", s=100, marker="s", zorder=10, edgecolors="k", label="End")

        ax.set_xlim(xlim)
        ax.set_ylim(zlim)
        ax.set_xlabel("X (world)")
        ax.set_ylabel("Z (world)")
        ax.set_title(f"{name}  |  {n_frames} frames  |  extent: {extent:.2f}")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        out_path = output_dir / f"{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"Done. {len(seq_data)} plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
