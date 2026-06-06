"""Sample random MVImgNet2 sequences and plot their COLMAP camera poses in a single 3D plot.

Each MVImgNet2 sequence stores a COLMAP sparse reconstruction under
``<seq>/sparse/0/{cameras,images,points3D}.bin``. We read the per-image
``cam_from_world`` poses with pycolmap, recover the camera centers and
viewing directions in world space, and scatter them (one color per
sequence) so the capture trajectories can be compared at a glance.

Usage:
    python scripts/visualization/plot_mvi2_cameras.py \
        --root /visinf/projects_students/dlcv2025_groupZ/mvimgnet2/mvi2_00 \
        --num-sequences 8 --seed 0 --out scripts/mvi2_cameras.png
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pycolmap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def find_sequences(root: Path):
    """Return all sequence dirs containing a COLMAP sparse/0 reconstruction."""
    seqs = []
    for sparse_dir in root.glob("*/*/sparse/0"):
        if (sparse_dir / "images.bin").exists():
            seqs.append(sparse_dir.parent.parent)
    return sorted(seqs)


def camera_poses(sparse0: Path):
    """Read a reconstruction; return (centers [N,3], forward dirs [N,3])."""
    rec = pycolmap.Reconstruction(str(sparse0))
    centers, forwards = [], []
    for img in rec.images.values():
        world_from_cam = img.cam_from_world().inverse()
        R = world_from_cam.rotation.matrix()  # world_from_cam rotation
        centers.append(world_from_cam.translation)
        # camera looks down +z in its own frame -> world direction is R @ [0,0,1]
        forwards.append(R @ np.array([0.0, 0.0, 1.0]))
    return np.asarray(centers), np.asarray(forwards)


def normalize(centers: np.ndarray):
    """Center at centroid and scale to unit median radius for comparability."""
    c = centers - centers.mean(axis=0, keepdims=True)
    radius = np.median(np.linalg.norm(c, axis=1))
    if radius > 1e-8:
        c = c / radius
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/visinf/projects_students/dlcv2025_groupZ/mvimgnet2/mvi2_00"))
    ap.add_argument("--num-sequences", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/scripts/mvi2_cameras.png"))
    ap.add_argument("--no-normalize", action="store_true",
                    help="Plot raw COLMAP coordinates instead of normalizing each sequence.")
    ap.add_argument("--arrows", action="store_true",
                    help="Draw per-camera viewing-direction arrows (cluttered for many seqs).")
    args = ap.parse_args()

    all_seqs = find_sequences(args.root)
    if not all_seqs:
        raise SystemExit(f"No COLMAP sequences found under {args.root}")
    print(f"Found {len(all_seqs)} sequences with COLMAP reconstructions.")

    rng = random.Random(args.seed)
    n = min(args.num_sequences, len(all_seqs))
    sampled = rng.sample(all_seqs, n)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")

    for i, seq in enumerate(sampled):
        centers, forwards = camera_poses(seq / "sparse" / "0")
        if not args.no_normalize:
            raw = centers.copy()
            centers = normalize(centers)
            # rescale forward arrows to match the normalized coordinate scale
            scale = (np.median(np.linalg.norm(raw - raw.mean(0), axis=1)) or 1.0)
            forwards = forwards / scale
        color = cmap(i % cmap.N)
        label = f"{seq.parent.name}/{seq.name} (n={len(centers)})"
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   color=color, s=18, label=label)
        # draw the capture path in order
        ax.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                color=color, alpha=0.4, linewidth=1)
        # mark the first frame so the capture start is identifiable
        ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2],
                   color=color, s=70, marker="*", edgecolors="k", linewidths=0.4)
        # optional short viewing-direction arrows
        if args.arrows:
            alen = 0.15 if not args.no_normalize else None
            ax.quiver(centers[:, 0], centers[:, 1], centers[:, 2],
                      forwards[:, 0], forwards[:, 1], forwards[:, 2],
                      color=color, length=alen, normalize=bool(alen),
                      alpha=0.5, linewidth=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = f"MVImgNet2 camera poses — {n} random sequences (seed={args.seed})"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
