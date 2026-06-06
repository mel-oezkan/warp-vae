"""Compute pairwise view co-visibility / overlap for an MVImgNet object.

Each MVImgNet object folder ships a COLMAP sparse reconstruction under
``sparse/0/{cameras,images,points3D}.bin``. Every 3D point stores a *track*:
the list of ``(image_id, keypoint_idx)`` pairs naming the images that observed
it. Two images "overlap" to the extent they observe the same 3D points, so we
can read overlap straight out of the tracks -- no depth or re-association
needed.

For each ordered/unordered image pair (A, B) we define:

    shared(A, B)   = | points(A) ∩ points(B) |          (co-visible 3D points)
    iou(A, B)      = shared / | points(A) ∪ points(B) |  (symmetric, IoU-style)
    frac_A(A, B)   = shared / | points(A) |              (asymmetric coverage)

The IoU matrix is symmetric; the fraction matrix is not. This is *sparse*
(feature-track-level) overlap -- a co-visibility signal for picking view pairs,
not dense pixel coverage.

Example
-------
    python scripts/preprocessing/mvimgnet_covisibility.py \
        --object_dir /visinf/projects_students/dlcv2025_groupZ/mvimgnet2/mvi2_00/626/550a4dfb \
        --out_dir /tmp/covis_550a4dfb --plot --top_k 10
"""

import argparse
import os
import struct
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# Minimal COLMAP binary reader (no pycolmap dependency).
# Format reference: colmap/scripts/python/read_write_model.py
# --------------------------------------------------------------------------- #
def _read(fid, num_bytes, fmt, endian="<"):
    return struct.unpack(endian + fmt, fid.read(num_bytes))


def read_images_point3d_ids(path):
    """Read ``images.bin`` -> {image_id: (name, set_of_point3D_ids)}.

    Only keypoints that were triangulated into a 3D point are kept (COLMAP
    stores ``-1`` for keypoints with no 3D point).
    """
    images = {}
    with open(path, "rb") as fid:
        num_reg = _read(fid, 8, "Q")[0]
        for _ in range(num_reg):
            image_id = _read(fid, 4, "i")[0]
            _ = _read(fid, 32, "dddd")  # qvec (w, x, y, z)
            _ = _read(fid, 24, "ddd")  # tvec
            _ = _read(fid, 4, "i")[0]  # camera_id
            name = b""
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name += c
            num_pts2d = _read(fid, 8, "Q")[0]
            data = _read(fid, 24 * num_pts2d, "ddq" * num_pts2d)
            p3d_ids = data[2::3]  # (x, y, point3D_id) triples
            valid = {pid for pid in p3d_ids if pid != -1}
            images[image_id] = (name.decode("utf-8", "replace"), valid)
    return images


# --------------------------------------------------------------------------- #
# Overlap computation
# --------------------------------------------------------------------------- #
def compute_overlap(images):
    """Given {image_id: (name, point_set)}, return (ids, names, iou, frac, shared).

    ``iou[i, j]``    = |P_i ∩ P_j| / |P_i ∪ P_j|     (symmetric)
    ``frac[i, j]``   = |P_i ∩ P_j| / |P_i|           (asymmetric: how much of i is covered by j)
    ``shared[i, j]`` = |P_i ∩ P_j|                   (raw co-visible point count)
    Diagonal is set to 1.0 (iou/frac) / |P_i| (shared).
    """
    image_ids = sorted(images.keys())
    names = [images[i][0] for i in image_ids]
    point_sets = [images[i][1] for i in image_ids]
    sizes = np.array([len(s) for s in point_sets], dtype=np.int64)
    n = len(image_ids)

    shared = np.zeros((n, n), dtype=np.int64)
    for i in tqdm(range(n), desc="pairs", leave=False):
        si = point_sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            c = len(si & point_sets[j])
            shared[i, j] = c
            shared[j, i] = c
    np.fill_diagonal(shared, sizes)

    union = sizes[:, None] + sizes[None, :] - shared
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, shared / union, 0.0)
        frac = np.where(sizes[:, None] > 0, shared / sizes[:, None], 0.0)
    np.fill_diagonal(iou, 1.0)
    np.fill_diagonal(frac, 1.0)
    return image_ids, names, iou, frac, shared


def top_pairs(names, iou, shared, k):
    """Return the k highest-IoU unordered pairs as (name_i, name_j, iou, shared)."""
    n = len(names)
    triu = [(iou[i, j], shared[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
    triu.sort(reverse=True)
    return [(names[i], names[j], v, int(s)) for v, s, i, j in triu[:k]]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def find_sparse_model(object_dir):
    """Locate the COLMAP model dir (one containing images.bin) under object_dir."""
    candidates = [
        os.path.join(object_dir, "sparse", "0"),
        os.path.join(object_dir, "sparse"),
        object_dir,
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "images.bin")):
            return c
    # Fall back to a recursive search.
    for root, _, files in os.walk(object_dir):
        if "images.bin" in files:
            return root
    raise FileNotFoundError(f"No images.bin found under {object_dir}")


def save_outputs(out_dir, image_ids, names, iou, frac, shared, plot):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "covisibility.npz"),
        image_ids=np.array(image_ids),
        names=np.array(names),
        iou=iou,
        frac=frac,
        shared=shared,
    )

    # Human-readable IoU matrix as CSV with image-name header.
    header = "," + ",".join(names)
    rows = [header]
    for i, name in enumerate(names):
        rows.append(name + "," + ",".join(f"{v:.4f}" for v in iou[i]))
    with open(os.path.join(out_dir, "iou_matrix.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, mat, title in (
            (axes[0], iou, "IoU (symmetric)"),
            (axes[1], frac, "frac of row covered by col"),
        ):
            im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("image idx")
            ax.set_ylabel("image idx")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "covisibility.png"), dpi=150)
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute pairwise view co-visibility/overlap for an MVImgNet object "
        "from its COLMAP sparse reconstruction."
    )
    p.add_argument(
        "--object_dir",
        type=str,
        required=True,
        help="MVImgNet object folder (contains sparse/0/{images,points3D}.bin).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to write covisibility.npz / iou_matrix.csv / covisibility.png. "
        "Default: <object_dir>/covisibility.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Print the K highest-IoU image pairs (0 to disable).",
    )
    p.add_argument("--plot", action="store_true", help="Save a heatmap PNG.")
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = find_sparse_model(args.object_dir)
    out_dir = args.out_dir or os.path.join(args.object_dir, "covisibility")

    images = read_images_point3d_ids(os.path.join(model_dir, "images.bin"))
    image_ids, names, iou, frac, shared = compute_overlap(images)

    n = len(image_ids)
    sizes = [len(images[i][1]) for i in image_ids]
    print(f"model: {model_dir}")
    print(f"registered images: {n}")
    print(
        f"points per image: min {min(sizes)}  max {max(sizes)}  mean {np.mean(sizes):.1f}"
    )
    # Mean IoU over off-diagonal entries.
    off = iou[~np.eye(n, dtype=bool)]
    print(f"mean pairwise IoU: {off.mean():.4f}  (max {off.max():.4f})")

    if args.top_k > 0:
        print(f"\nTop {args.top_k} most-overlapping pairs (IoU):")
        for a, b, v, s in top_pairs(names, iou, shared, args.top_k):
            print(f"  {a:>16s}  <->  {b:<16s}  IoU={v:.3f}  shared={s}")

    save_outputs(out_dir, image_ids, names, iou, frac, shared, args.plot)
    print(f"\nwrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
