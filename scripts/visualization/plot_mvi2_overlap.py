"""Visualise view co-visibility / overlap for an MVImgNet object.

Demonstrates that the MVImgNet2 COLMAP reconstructions carry cross-image
attribution: every 3D point stores a *track* of ``(image_id, keypoint)`` pairs,
so for any two frames A and B we can split their observed 3D points into

    only-A  (seen in A, not B)        -> red
    only-B  (seen in B, not A)        -> blue
    shared  (seen in both = overlap)  -> green / yellow

The figure overlays these colour-coded keypoints on the actual pixels of each
frame and draws the matched (shared) correspondences as connecting lines, plus a
proportion bar that reads out the overlap ratio (Jaccard IoU).

Two modes
---------
  --mode pair     (default) one rich figure for two frames of a single object.
  --mode classes  one figure per class, each showing the overlap gradient
                  (high -> mid -> low baseline). Delegates to
                  plot_mvi2_overlap_classes.py.

Examples
--------
    # one object, explicit (or auto-picked) frame pair
    python scripts/visualization/plot_mvi2_overlap.py --mode pair \
        --object_dir /visinf/projects_students/dlcv2025_groupZ/mvimgnet2/mvi2_00/626/550a4dfb \
        --image_a images/002.jpg --image_b images/030.jpg

    # one figure per class
    python scripts/visualization/plot_mvi2_overlap.py --mode classes --num_classes 6

Outputs default to outputs/scripts/mvi2_overlap/ (override with --out / --out_dir).
If --image_a/--image_b are omitted in pair mode, an informative partial-overlap
pair is chosen automatically.
"""

import argparse
import os
import struct

import numpy as np

# Repo root is three levels up from scripts/visualization/<this file>.
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "outputs", "scripts", "mvi2_overlap")


# --------------------------------------------------------------------------- #
# Minimal COLMAP images.bin reader -> per-image (name, keypoints, point3D ids).
# Format reference: colmap/scripts/python/read_write_model.py
# --------------------------------------------------------------------------- #
def _read(fid, num_bytes, fmt):
    return struct.unpack("<" + fmt, fid.read(num_bytes))


def read_images_binary(path):
    """Return {name: {"xys": (N,2) float, "p3d": (N,) int64 (-1 if none)}}."""
    images = {}
    with open(path, "rb") as fid:
        num_reg = _read(fid, 8, "Q")[0]
        for _ in range(num_reg):
            _ = _read(fid, 4, "i")[0]  # image_id
            _ = _read(fid, 32, "dddd")  # qvec
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
            arr = np.array(data, dtype=np.float64).reshape(-1, 3)
            images[name.decode("utf-8", "replace")] = {
                "xys": arr[:, :2],
                "p3d": arr[:, 2].astype(np.int64),
            }
    return images


# --------------------------------------------------------------------------- #
# Mask filtering -> keep only 3D points that land on the object foreground.
#
# A 3D point is observed in many frames, so we give it a single object/background
# label by majority vote over its track: it is "object" if its projection falls
# inside the foreground mask in more than half of the frames that observe it.
# This keeps point_set / IoU consistent across views (the same id is never object
# in one frame and background in another).
# --------------------------------------------------------------------------- #
def _mask_dir(object_dir):
    """Return the masks/ folder for an object dir, or None if absent."""
    d = os.path.join(object_dir, "masks")
    return d if os.path.isdir(d) else None


def _load_mask(mask_dir, image_name, thresh=127):
    """Load a foreground mask for a COLMAP image name (e.g. 'images/001.jpg').

    MVImgNet2 masks live in masks/<basename> and are binary 0/255 stored as JPEG,
    so we threshold (the in-between values are just compression fringing).
    Returns a (H, W) bool array, or None if no matching mask file exists.
    """
    from PIL import Image

    base = os.path.basename(image_name)
    path = os.path.join(mask_dir, base)
    if not os.path.isfile(path):  # masks are usually .jpg; try .png as a fallback
        alt = os.path.join(mask_dir, os.path.splitext(base)[0] + ".png")
        if not os.path.isfile(alt):
            return None
        path = alt
    return np.asarray(Image.open(path).convert("L")) > thresh


def object_point_ids(images, object_dir):
    """Set of 3D point ids that are on the object (majority vote over track).

    Each keypoint votes for its 3D id based on whether it lands inside that
    frame's mask. A 3D id is kept if >50% of its observations vote "object".
    Frames without a mask file are skipped (they cast no vote). If no masks are
    found at all, returns None (caller should leave points unfiltered).
    """
    mask_dir = _mask_dir(object_dir)
    if mask_dir is None:
        return None
    inside = {}  # pid -> count of observations inside a mask
    total = {}   # pid -> count of observations with a mask available
    found_any = False
    for name, img in images.items():
        mask = _load_mask(mask_dir, name)
        if mask is None:
            continue
        found_any = True
        h, w = mask.shape
        xys, p3d = img["xys"], img["p3d"]
        xi = np.clip(xys[:, 0].astype(int), 0, w - 1)
        yi = np.clip(xys[:, 1].astype(int), 0, h - 1)
        hit = mask[yi, xi]
        for pid, on in zip(p3d, hit):
            pid = int(pid)
            if pid == -1:
                continue
            total[pid] = total.get(pid, 0) + 1
            if on:
                inside[pid] = inside.get(pid, 0) + 1
    if not found_any:
        return None
    return {pid for pid, t in total.items() if inside.get(pid, 0) > t / 2}


def filter_to_object(images, object_dir, verbose=True):
    """Rewrite p3d in place so only object 3D points survive (rest -> -1).

    All downstream consumers (point_set, split_keypoints, pair pickers) then
    operate on object-only points without further changes. No-op (returns False)
    when no masks are available.
    """
    keep = object_point_ids(images, object_dir)
    if keep is None:
        return False
    before = after = 0
    for img in images.values():
        p3d = img["p3d"]
        valid = p3d != -1
        before += int(valid.sum())
        drop = valid & ~np.isin(p3d, list(keep))
        p3d[drop] = -1
        after += int((p3d != -1).sum())
    if verbose:
        print(f"mask filter: kept {after}/{before} keypoint observations "
              f"({len(keep)} object 3D points)")
    return True


# --------------------------------------------------------------------------- #
# Pair selection
# --------------------------------------------------------------------------- #
def point_set(img):
    return set(int(p) for p in img["p3d"] if p != -1)


def auto_pick_pair(images, target_iou=None):
    """Pick a frame pair automatically.

    target_iou is None -> the most informative partial-overlap pair (IoU in the
    0.25-0.4 band with balanced unique parts). Otherwise, the pair whose IoU is
    closest to target_iou (e.g. 0.8 for a high-overlap example).
    """
    names = sorted(images)
    sets = {n: point_set(images[n]) for n in names}

    if target_iou is not None:
        best = None  # (abs_dist, name_a, name_b)
        for i, a in enumerate(names):
            sa = sets[a]
            if not sa:
                continue
            for b in names[i + 1 :]:
                sb = sets[b]
                if not sb:
                    continue
                iou = len(sa & sb) / len(sa | sb)
                dist = abs(iou - target_iou)
                if best is None or dist < best[0]:
                    best = (dist, a, b)
        if best is None:
            raise SystemExit("no frame pair with shared 3D points found")
        return best[1], best[2]

    best = None
    for i, a in enumerate(names):
        sa = sets[a]
        if not sa:
            continue
        for b in names[i + 1 :]:
            sb = sets[b]
            if not sb:
                continue
            inter = len(sa & sb)
            union = len(sa | sb)
            iou = inter / union
            if 0.25 < iou < 0.4 and inter > 150:
                score = min(len(sa - sb), len(sb - sa))
                if best is None or score > best[0]:
                    best = (score, a, b)
    if best is None:  # fall back to highest-IoU pair
        for i, a in enumerate(names):
            sa = sets[a]
            for b in names[i + 1 :]:
                sb = sets[b]
                if not sa or not sb:
                    continue
                iou = len(sa & sb) / len(sa | sb)
                if best is None or iou > best[0]:
                    best = (iou, a, b)
    return best[1], best[2]


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
ONLY_A = "#e8453c"  # red
ONLY_B = "#3b6fd4"  # blue
SHARED = "#23c552"  # green
LINK = "#f5d000"  # yellow links


def split_keypoints(img, shared_ids):
    """Return (shared_xy ordered by id, only_xy) for one image."""
    p3d = img["p3d"]
    xys = img["xys"]
    shared_xy = {}
    only_xy = []
    for (x, y), pid in zip(xys, p3d):
        if pid == -1:
            continue
        if pid in shared_ids:
            shared_xy[pid] = (x, y)
        else:
            only_xy.append((x, y))
    only_xy = np.array(only_xy) if only_xy else np.zeros((0, 2))
    return shared_xy, only_xy


def load_image(object_dir, name):
    from PIL import Image

    path = os.path.join(object_dir, name)
    return np.asarray(Image.open(path).convert("RGB"))


def make_figure(object_dir, name_a, name_b, img_a, img_b, out_path, dim=0.45):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch

    set_a, set_b = point_set(img_a), point_set(img_b)
    shared_ids = set_a & set_b
    only_a_n, only_b_n, shared_n = len(set_a - set_b), len(set_b - set_a), len(shared_ids)
    union_n = len(set_a | set_b)
    iou = shared_n / union_n
    frac_a = shared_n / len(set_a)
    frac_b = shared_n / len(set_b)

    pic_a = load_image(object_dir, name_a)
    pic_b = load_image(object_dir, name_b)

    shared_a, only_a = split_keypoints(img_a, shared_ids)
    shared_b, only_b = split_keypoints(img_b, shared_ids)

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[6, 1.0], width_ratios=[1, 1], hspace=0.04, wspace=0.04
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    for ax, pic, title, only_xy, shared_xy, n_only, frac in (
        (ax_a, pic_a, name_a, only_a, shared_a, only_a_n, frac_a),
        (ax_b, pic_b, name_b, only_b, shared_b, only_b_n, frac_b),
    ):
        ax.imshow(pic)
        # dim the background so points pop
        ax.imshow(np.zeros_like(pic[:, :, 0]), cmap="gray", alpha=dim, vmin=0, vmax=1)
        only_color = ONLY_A if ax is ax_a else ONLY_B
        if len(only_xy):
            ax.scatter(
                only_xy[:, 0], only_xy[:, 1], s=14, c=only_color,
                edgecolors="white", linewidths=0.3, label=f"only here ({n_only})",
            )
        if shared_xy:
            sx = np.array(list(shared_xy.values()))
            ax.scatter(
                sx[:, 0], sx[:, 1], s=18, c=SHARED, edgecolors="white",
                linewidths=0.4, label=f"shared ({shared_n})", zorder=3,
            )
        ax.set_title(
            f"{os.path.basename(title)}\n{frac:.0%} of its points are shared",
            fontsize=13,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    # connecting lines between shared correspondences (subsample for clarity)
    common = sorted(shared_ids)
    rng = np.random.default_rng(0)
    if len(common) > 120:
        common = list(rng.choice(common, size=120, replace=False))
    for pid in common:
        xa, ya = shared_a[pid]
        xb, yb = shared_b[pid]
        con = ConnectionPatch(
            xyA=(xa, ya), coordsA=ax_a.transData,
            xyB=(xb, yb), coordsB=ax_b.transData,
            color=LINK, alpha=0.35, linewidth=0.6, zorder=1,
        )
        fig.add_artist(con)

    # ---- proportion bar (the overlap read-out) ----
    ax_bar.barh(0, only_a_n, color=ONLY_A, edgecolor="white")
    ax_bar.barh(0, shared_n, left=only_a_n, color=SHARED, edgecolor="white")
    ax_bar.barh(0, only_b_n, left=only_a_n + shared_n, color=ONLY_B, edgecolor="white")
    ax_bar.set_xlim(0, union_n)
    ax_bar.set_ylim(-0.6, 0.6)
    ax_bar.set_yticks([])
    for label, x in (
        (f"only A\n{only_a_n}", only_a_n / 2),
        (f"shared (overlap)\n{shared_n}", only_a_n + shared_n / 2),
        (f"only B\n{only_b_n}", only_a_n + shared_n + only_b_n / 2),
    ):
        ax_bar.text(x, 0, label, ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold")
    ax_bar.set_xlabel(
        f"union of 3D points = {union_n}        "
        f"overlap (IoU) = shared / union = {iou:.2f}",
        fontsize=12,
    )
    ax_bar.set_title("Co-visible 3D points between the two views", fontsize=12)

    fig.suptitle(
        f"MVImgNet2 view overlap  —  {os.path.basename(os.path.normpath(object_dir))}\n"
        f"yellow lines = matched 3D points seen in both frames (cross-image attribution from COLMAP tracks)",
        fontsize=15,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dict(iou=iou, shared=shared_n, only_a=only_a_n, only_b=only_b_n)


# --------------------------------------------------------------------------- #
def find_sparse_model(object_dir):
    for c in (
        os.path.join(object_dir, "sparse", "0"),
        os.path.join(object_dir, "sparse"),
        object_dir,
    ):
        if os.path.isfile(os.path.join(c, "images.bin")):
            return c
    for root, _, files in os.walk(object_dir):
        if "images.bin" in files:
            return root
    raise FileNotFoundError(f"No images.bin found under {object_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("pair", "classes"), default="pair",
                   help="'pair': one figure for two frames of one object. "
                        "'classes': one figure per class (overlap gradient).")
    p.add_argument("--mask", dest="mask", action="store_true", default=None,
                   help="Keep only 3D points on the object foreground "
                        "(majority vote over track using masks/). Default: on "
                        "when a masks/ folder exists.")
    p.add_argument("--no-mask", dest="mask", action="store_false",
                   help="Disable mask filtering (plot all triangulated points).")
    # --- pair mode ---
    p.add_argument("--object_dir", type=str, default=None,
                   help="[pair] MVImgNet object folder (with sparse/0 and images/).")
    p.add_argument("--image_a", type=str, default=None,
                   help="[pair] Name of frame A in COLMAP (e.g. images/002.jpg).")
    p.add_argument("--image_b", type=str, default=None,
                   help="[pair] Name of frame B (e.g. images/030.jpg).")
    p.add_argument("--target-iou", dest="target_iou", type=float, default=None,
                   help="[pair] Auto-pick the pair whose overlap (IoU) is closest "
                        "to this value (e.g. 0.8). Ignored if --image_a/--image_b "
                        "are given. Default: most informative partial-overlap pair.")
    p.add_argument("--out", type=str, default=None,
                   help="[pair] Output PNG. Default: "
                        "outputs/scripts/mvi2_overlap/overlap_<A>_<B>.png")
    # --- classes mode ---
    p.add_argument("--root", type=str,
                   default="/visinf/projects_students/dlcv2025_groupZ/mvimgnet2/mvi2_00",
                   help="[classes] Dir containing <category>/<object>/ folders.")
    p.add_argument("--num_classes", type=int, default=6,
                   help="[classes] How many classes to render (one figure each).")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                   help="[classes] Directory to write one PNG per class into. "
                        "Default: outputs/scripts/mvi2_overlap/")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Classes mode: one figure per class, columns = high -> mid -> low overlap.
# --------------------------------------------------------------------------- #
def _find_model(object_dir):
    for c in (os.path.join(object_dir, "sparse", "0"),
              os.path.join(object_dir, "sparse"), object_dir):
        if os.path.isfile(os.path.join(c, "images.bin")):
            return c
    return None


def discover_objects(root, num_classes, min_images=20):
    """Return up to num_classes (cat, object_dir, images_dict) with valid models."""
    out = []
    for cat in sorted(os.listdir(root), key=lambda s: (len(s), s)):
        catdir = os.path.join(root, cat)
        if not os.path.isdir(catdir):
            continue
        for obj in sorted(os.listdir(catdir)):
            objdir = os.path.join(catdir, obj)
            model = _find_model(objdir)
            if model is None or not os.path.isdir(os.path.join(objdir, "images")):
                continue
            images = read_images_binary(os.path.join(model, "images.bin"))
            if len(images) >= min_images:
                out.append((cat, objdir, images))
                break
        if len(out) >= num_classes:
            break
    return out


def pick_pairs_by_overlap(images, levels=("high", "mid", "low")):
    """Pick one frame pair per overlap level, spread over this object.

    Targets are per-object IoU quantiles (so "high/mid/low" adapt to whatever
    range the object actually spans), and chosen pairs are de-duplicated.
    Returns list of (level, name_a, name_b, iou).
    """
    names = sorted(images)
    sets = {n: point_set(images[n]) for n in names}
    pairs = []
    for i, a in enumerate(names):
        sa = sets[a]
        if not sa:
            continue
        for b in names[i + 1:]:
            sb = sets[b]
            if not sb:
                continue
            u = len(sa | sb)
            if u == 0:
                continue
            pairs.append((len(sa & sb) / u, a, b))
    if not pairs:
        return []
    pairs.sort(reverse=True)
    iou_vals = np.array([p[0] for p in pairs])
    q = {"high": 0.97, "mid": 0.55, "low": 0.06}
    chosen, used = [], set()
    for lvl in levels:
        target = float(np.quantile(iou_vals, q[lvl]))
        order = sorted(pairs, key=lambda p: abs(p[0] - target))
        pick = next((p for p in order if (p[1], p[2]) not in used), order[0])
        used.add((pick[1], pick[2]))
        chosen.append((lvl, pick[1], pick[2], pick[0]))
    return chosen


def _split_xy(img, shared_ids):
    """Like split_keypoints but returns (shared_xy_array, only_xy_array)."""
    p3d, xys = img["p3d"], img["xys"]
    shared_xy, only_xy = [], []
    for (x, y), pid in zip(xys, p3d):
        if pid == -1:
            continue
        (shared_xy if pid in shared_ids else only_xy).append((x, y))
    sh = np.array(shared_xy) if shared_xy else np.zeros((0, 2))
    on = np.array(only_xy) if only_xy else np.zeros((0, 2))
    return sh, on


def _draw_cell(ax, object_dir, img_a, img_b, name_a, name_b, level, iou):
    """Stack frames A (top) and B (bottom) vertically inside one cell axis."""
    set_a, set_b = point_set(img_a), point_set(img_b)
    shared_ids = set_a & set_b
    pic_a = load_image(object_dir, name_a)
    pic_b = load_image(object_dir, name_b)
    H, W = pic_a.shape[:2]

    gap = int(0.03 * H)
    canvas = np.full((2 * H + gap, W, 3), 255, dtype=np.uint8)
    canvas[:H] = pic_a
    canvas[H + gap:] = pic_b
    ax.imshow(canvas)
    ax.imshow(np.zeros((canvas.shape[0], canvas.shape[1])), cmap="gray",
              alpha=0.4, vmin=0, vmax=1)

    for img, only_c, y_off in ((img_a, ONLY_A, 0), (img_b, ONLY_B, H + gap)):
        sh, on = _split_xy(img, shared_ids)
        if len(on):
            ax.scatter(on[:, 0], on[:, 1] + y_off, s=3, c=only_c, linewidths=0)
        if len(sh):
            ax.scatter(sh[:, 0], sh[:, 1] + y_off, s=4, c=SHARED, linewidths=0,
                       zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{level}  IoU={iou:.2f}", fontsize=10)


def make_class_figure(cat, objdir, images, out_path, levels=("high", "mid", "low")):
    """One figure for a single class: high -> mid -> low overlap columns."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    chosen = pick_pairs_by_overlap(images, levels)
    ncols = len(chosen)
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 6.2), squeeze=False)
    for c, (level, na, nb, iou) in enumerate(chosen):
        _draw_cell(axes[0][c], objdir, images[na], images[nb], na, nb, level, iou)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SHARED,
               markersize=9, label="shared (overlap)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ONLY_A,
               markersize=9, label="only top frame"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ONLY_B,
               markersize=9, label="only bottom frame"),
    ]
    fig.suptitle(
        f"MVImgNet2 view overlap  —  class {cat}\n"
        "columns = wider camera baseline;  "
        "green = 3D points co-visible in both frames (overlap from COLMAP tracks)",
        fontsize=13, y=1.06,
    )
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_classes(args):
    """Render one figure per class (overlap gradient)."""
    objects = discover_objects(args.root, args.num_classes)
    if not objects:
        raise SystemExit(f"No valid COLMAP objects found under {args.root}")
    os.makedirs(args.out_dir, exist_ok=True)
    for cat, objdir, images in objects:
        if args.mask is not False:  # None (auto) or True -> filter when masks exist
            filter_to_object(images, objdir)
        out_path = os.path.join(args.out_dir, f"overlap_class_{cat}.png")
        make_class_figure(cat, objdir, images, out_path)
        print(f"class {cat}: {objdir}  [{len(images)} imgs]  ->  {out_path}")
    print(f"wrote {len(objects)} figures to {args.out_dir}")


def main():
    args = parse_args()
    if args.mode == "classes":
        run_classes(args)
        return

    if not args.object_dir:
        raise SystemExit("--object_dir is required for --mode pair")
    model_dir = find_sparse_model(args.object_dir)
    images = read_images_binary(os.path.join(model_dir, "images.bin"))

    if args.mask is not False:  # None (auto) or True -> filter when masks exist
        if not filter_to_object(images, args.object_dir) and args.mask:
            print("warning: --mask requested but no masks/ folder found")

    if args.image_a and args.image_b:
        name_a, name_b = args.image_a, args.image_b
    else:
        name_a, name_b = auto_pick_pair(images, args.target_iou)
        tgt = f" (target IoU={args.target_iou})" if args.target_iou is not None else ""
        print(f"auto-selected pair{tgt}: {name_a}  <->  {name_b}")

    for n in (name_a, name_b):
        if n not in images:
            raise KeyError(
                f"{n!r} not in COLMAP model. Available e.g.: {sorted(images)[:5]}"
            )

    if args.out:
        out = args.out
    else:
        obj_name = os.path.basename(os.path.normpath(args.object_dir))
        out = os.path.join(
            DEFAULT_OUT_DIR,
            f"overlap_{obj_name}_{os.path.basename(name_a).split('.')[0]}_"
            f"{os.path.basename(name_b).split('.')[0]}.png",
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stats = make_figure(args.object_dir, name_a, name_b,
                        images[name_a], images[name_b], out)
    print(
        f"IoU={stats['iou']:.3f}  shared={stats['shared']}  "
        f"only_A={stats['only_a']}  only_B={stats['only_b']}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
