"""Sequence overlap decay measured with dense RoMA correspondences.

This is the RoMA counterpart of plot_sequence_overlap_decay.py. The COLMAP
version derived overlap from sparse SfM tracks, which are too few and noisy on
these textureless MVImgNet objects to tell a clean story. Here we instead run
RoMA dense matching between an anchor frame and each successor, and define

    overlap kept = fraction of the anchor's object-mask pixels that have a
                   confident, in-bounds correspondence landing on the object in
                   the successor frame.

Grid layout (unchanged): 3 rows, one per anchor frame spread across the
sequence (early / middle / late). Each row has 6 cells:

    [ anchor ] [ +1 ] [ +2 ] [ +3 ] [ +4 ] [ +5 ]

In every successor cell, the anchor's object pixels are warped into that frame
and the ones that survive (confident + in-bounds + land on the object) are shown
in green over the frame; the title reads out the kept fraction. Walking left to
right, the green mass shrinks as the camera moves away from the anchor.

Example
-------
    python scripts/visualization/plot_sequence_overlap_decay_roma.py \
        --object_dir /visinf/projects_students/dlcv2025_groupZ/mvimgnet2/mvi2_00/11/5708d390

Outputs default to outputs/scripts/mvi2_overlap/.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.roma_metrics import load_roma_model  # noqa: E402

# Reuse the anchor picker and output dir from the COLMAP sibling.
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "visualization"))
from plot_sequence_overlap_decay import pick_anchors  # noqa: E402
from plot_mvi2_overlap import DEFAULT_OUT_DIR, ONLY_A  # noqa: E402

SHARED = "#23c552"  # green (kept overlap)


def list_frames(object_dir):
    """Return sorted list of image file names under images/ (basename only)."""
    img_dir = os.path.join(object_dir, "images")
    names = [f for f in os.listdir(img_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return sorted(names)


def load_rgb(object_dir, name, size):
    return Image.open(os.path.join(object_dir, "images", name)).convert("RGB").resize(
        (size, size), Image.BILINEAR
    )


def load_mask(object_dir, name, size, thresh=127):
    """Foreground bool mask (size, size) for a frame, or None if absent."""
    base = os.path.basename(name)
    for cand in (base, os.path.splitext(base)[0] + ".png",
                 os.path.splitext(base)[0] + ".jpg"):
        path = os.path.join(object_dir, "masks", cand)
        if os.path.isfile(path):
            m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
            return np.asarray(m) > thresh
    return None


def roma_overlap(model, img_a, img_b, mask_a, mask_b, conf_thresh,
                 cycle_tol=0.02):
    """Overlap of anchor A into successor B via dense RoMA.

    Returns (keep_frac, covis_frac, dst) where

    - keep_frac : fraction of A's object pixels whose RoMA match is confident,
                  in-bounds, and lands on B's object mask. This is optimistic:
                  RoMA confidently *extrapolates* a smooth warp over occluded
                  parts, and "lands on B's object" accepts any object pixel, so
                  occluded anchor pixels get folded onto the visible part instead
                  of dropped. It measures "what RoMA will place on the object",
                  not strict visibility.
    - covis_frac: the stricter subset that is also cycle-consistent (A->B->A
                  returns within cycle_tol of the origin, in normalized coords).
                  Folds and hallucinations fail the round trip, so this tracks
                  true co-visibility and falls on heavily occluded frames.
    - dst       : (H, W) bool overlay of the *kept* pixels placed in B's grid.
    """
    with torch.no_grad():
        pred = model.match(img_a, img_b)
        pred_rev = model.match(img_b, img_a)  # for the cycle check
    warp = pred["warp_AB"][0]            # (H, W, 2) normalized coords into B
    overlap = pred["overlap_AB"]
    if overlap is None:
        overlap = pred["confidence_AB"].mean(dim=-1, keepdim=True)
    overlap = overlap[0, ..., 0].float().cpu().numpy()  # (H, W) in [0, 1]
    warp = warp.float().cpu().numpy()
    H, W = overlap.shape

    # Resize the anchor mask to RoMA's working resolution.
    ma = _resize_bool(mask_a, H, W)
    confident = overlap > conf_thresh
    in_bounds = np.all(np.abs(warp) <= 1.0, axis=-1)
    kept = ma & confident & in_bounds

    # Require the match to land on the object in B as well.
    if mask_b is not None:
        mb = _resize_bool(mask_b, H, W)
        bx = np.clip(((warp[..., 0] + 1) / 2 * W).astype(int), 0, W - 1)
        by = np.clip(((warp[..., 1] + 1) / 2 * H).astype(int), 0, H - 1)
        kept &= mb[by, bx]

    denom = int(ma.sum())
    keep_frac = float(kept.sum()) / denom if denom else 0.0

    # Cycle consistency: warp A->B then sample the B->A field at those B coords;
    # the round trip should land back near each A pixel's own normalized position.
    cycle_ok = _cycle_consistent(warp, pred_rev["warp_AB"][0].float().cpu().numpy(),
                                 cycle_tol)
    covis = kept & cycle_ok
    covis_frac = float(covis.sum()) / denom if denom else 0.0

    # Warped destination pixels in B's grid, so the overlay lands on the object
    # where it actually moved (kept anchor pixel -> its RoMA match in B).
    ky, kx = np.where(kept)
    dst_x = np.clip(((warp[ky, kx, 0] + 1) / 2 * W).astype(int), 0, W - 1)
    dst_y = np.clip(((warp[ky, kx, 1] + 1) / 2 * H).astype(int), 0, H - 1)
    dst = np.zeros((H, W), dtype=bool)
    dst[dst_y, dst_x] = True
    return keep_frac, covis_frac, dst


def _cycle_consistent(warp_ab, warp_ba, tol):
    """Bool (H, W): True where A->B->A returns within tol (normalized coords)."""
    H, W, _ = warp_ab.shape
    wba = torch.tensor(warp_ba).permute(2, 0, 1)[None]      # (1, 2, H, W)
    grid = torch.tensor(warp_ab)[None].float()             # sample at B coords
    back = F.grid_sample(wba, grid, align_corners=False)[0].permute(1, 2, 0).numpy()
    ys, xs = np.mgrid[0:H, 0:W]
    ox = xs / (W - 1) * 2 - 1
    oy = ys / (H - 1) * 2 - 1
    err = np.sqrt((back[..., 0] - ox) ** 2 + (back[..., 1] - oy) ** 2)
    return err < tol


def _resize_bool(mask, H, W):
    if mask is None:
        return np.ones((H, W), dtype=bool)
    if mask.shape == (H, W):
        return mask
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    return np.asarray(m) > 127


def _draw_anchor(ax, img_a, mask_a, name):
    pic = np.asarray(img_a)
    ax.imshow(pic)
    ax.imshow(np.zeros(pic.shape[:2]), cmap="gray", alpha=0.45, vmin=0, vmax=1)
    if mask_a is not None:
        H, W = pic.shape[:2]
        ma = _resize_bool(mask_a, H, W)
        ys, xs = np.where(ma)
        ax.scatter(xs, ys, s=0.5, c=SHARED, linewidths=0, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(ONLY_A); spine.set_linewidth(3)
    ax.set_title(f"ANCHOR  {name}", fontsize=10, color=ONLY_A, fontweight="bold")


def _draw_step(ax, img_b, kept_dst, keep_frac, covis_frac, name):
    """Show successor frame B with the surviving anchor pixels in B's frame.

    kept_dst already lives in B's pixel grid (each kept anchor pixel placed at
    its RoMA-matched location in B), so the green overlay tracks the object as it
    moves between frames.
    """
    pic = np.asarray(img_b)
    ax.imshow(pic)
    ax.imshow(np.zeros(pic.shape[:2]), cmap="gray", alpha=0.55, vmin=0, vmax=1)
    H, W = pic.shape[:2]
    km = _resize_bool(kept_dst, H, W)
    ys, xs = np.where(km)
    if len(xs):
        ax.scatter(xs, ys, s=0.5, c=SHARED, linewidths=0, alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}\nkeeps {keep_frac:.0%}  |  covisible {covis_frac:.0%}",
                 fontsize=10)


def make_figure(object_dir, model, out_path, n_anchors=3, n_steps=5,
                size=512, conf_thresh=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = list_frames(object_dir)
    anchor_idxs = pick_anchors(names, n_anchors, n_steps)
    ncols = n_steps + 1

    # Cache images / masks lazily.
    imgs, masks = {}, {}

    def get(name):
        if name not in imgs:
            imgs[name] = load_rgb(object_dir, name, size)
            masks[name] = load_mask(object_dir, name, size)
        return imgs[name], masks[name]

    fig, axes = plt.subplots(
        n_anchors, ncols, figsize=(3.0 * ncols, 3.4 * n_anchors), squeeze=False
    )
    for r, ai in enumerate(anchor_idxs):
        a_name = names[ai]
        a_img, a_mask = get(a_name)
        _draw_anchor(axes[r][0], a_img, a_mask, a_name)
        row_names = names[ai + 1:ai + ncols]
        for c in range(1, ncols):
            ax = axes[r][c]
            if c - 1 >= len(row_names):
                ax.axis("off"); continue
            b_name = row_names[c - 1]
            b_img, b_mask = get(b_name)
            keep_frac, covis_frac, kept = roma_overlap(
                model, a_img, b_img, a_mask, b_mask, conf_thresh)
            _draw_step(ax, b_img, kept, keep_frac, covis_frac, b_name)
            print(f"  anchor {a_name} -> {b_name}: "
                  f"keeps {keep_frac:.1%}  covisible {covis_frac:.1%}")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SHARED,
               markersize=9, label="anchor object pixels still co-visible (RoMA)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=1, fontsize=11,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(
        f"RoMA sequence overlap decay  —  "
        f"{os.path.basename(os.path.normpath(object_dir))}\n"
        f"each row = an anchor (red border) + its next {n_steps} frames;  "
        "green = anchor object pixels with a confident RoMA match;  "
        "keeps = lands on object  |  covisible = also cycle-consistent (stricter)",
        fontsize=13, y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--object_dir", type=str, required=True,
                   help="MVImgNet object folder (with images/ and masks/).")
    p.add_argument("--anchors", type=int, default=3, help="Anchor rows. Default 3.")
    p.add_argument("--steps", type=int, default=5,
                   help="Successor frames per anchor. Default 5.")
    p.add_argument("--setting", type=str, default="precise",
                   choices=("precise", "fast", "turbo", "base"),
                   help="RoMaV2 setting. Default: precise.")
    p.add_argument("--conf-thresh", type=float, default=0.5,
                   help="Min RoMA overlap confidence for a kept match. Default 0.5.")
    p.add_argument("--size", type=int, default=512,
                   help="Working image resolution. Default 512.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default=None,
                   help="Output PNG. Default: "
                        "outputs/scripts/mvi2_overlap/seq_decay_roma_<object>.png")
    return p.parse_args()


def main():
    args = parse_args()
    model = load_roma_model(setting=args.setting, device=args.device)

    if args.out:
        out = args.out
    else:
        obj_name = os.path.basename(os.path.normpath(args.object_dir))
        out = os.path.join(DEFAULT_OUT_DIR, f"seq_decay_roma_{obj_name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    make_figure(args.object_dir, model, out, args.anchors, args.steps,
                args.size, args.conf_thresh)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
