"""
Per-(bin, sequence) good vs bad confidence pair grids.

For each requested distance bin, walks every sequence with pairs in that bin
and renders one PNG per sequence containing up to 5 high-confidence pairs
(conf_da3 > good_thresh) and up to 5 low-confidence pairs (conf_da3 <
bad_thresh) in a 6-column layout: imA, imB, B->A, A->B, DA3 depth A, DA3
depth B.

Output:
    <sweep_dir>/good_bad_grids/bin_<lo>-<hi>/<seq>.png

Usage:
    python scripts/visualization/plot_da3_sweep_good_bad.py \\
        --sweep_dir eval_outputs/da3_distance_sweep \\
        --annotation_file data/co3d_annotations/hydrant_train_50seq_depth.jgz \\
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full
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
from warps.precompute_depth_warps import compute_depth_warp, load_annotations  # noqa: E402
from warps.precompute_da3_warps import apply_warp, crop_resize  # noqa: E402
from sweeps.sweep_da3_distance import get_frame_data  # noqa: E402


def to_tensor(rgb):
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def crop_depth(depth, bbox, resolution):
    h, w = depth.shape
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        depth = depth[y1c:y2c, x1c:x2c]
    return np.array(Image.fromarray(depth).resize(
        (resolution, resolution), Image.NEAREST
    ))


def warp_pair_full(A, B, warp_resolution, depth_consistency_threshold):
    """Compute DA3 warp in full-image coords (no bbox crop). Returns
    (warp_da3, conf_da3) at warp_resolution × warp_resolution covering the
    full image domain — caller is responsible for cropping to bbox."""
    valid_a = (A["da3_aligned"] > 0) & np.isfinite(A["da3_aligned"])
    valid_b = (B["da3_aligned"] > 0) & np.isfinite(B["da3_aligned"])
    return compute_depth_warp(
        A["da3_aligned"], valid_a, A["R"], A["T"], A["K"],
        B["da3_aligned"], valid_b, B["R"], B["T"], B["K"],
        warp_resolution=warp_resolution,
        image_size_a=A["image_size"], image_size_b=B["image_size"],
        depth_consistency_threshold=depth_consistency_threshold,
        crop_bbox_a=None, crop_bbox_b=None,
    )


def crop_full_to_bbox(arr_hw_or_hwc, bbox, image_size_hw, display_res):
    """Crop a full-image array (H_warp, W_warp[, C]) to A's bbox and resize
    to display_res. The warp grid covers the full original image, so we map
    bbox pixel coords into the warp grid via image_size_hw."""
    H_orig, W_orig = image_size_hw
    if arr_hw_or_hwc.ndim == 3:
        H_w, W_w, _ = arr_hw_or_hwc.shape
    else:
        H_w, W_w = arr_hw_or_hwc.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    sx, sy = W_w / W_orig, H_w / H_orig
    x1c = max(0, int(round(x1 * sx)))
    y1c = max(0, int(round(y1 * sy)))
    x2c = min(W_w, int(round(x2 * sx)))
    y2c = min(H_w, int(round(y2 * sy)))
    crop = arr_hw_or_hwc[y1c:y2c, x1c:x2c]
    if crop.ndim == 3:
        pil = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8))
        return np.array(pil.resize((display_res, display_res), Image.BILINEAR)) / 255.0
    pil = Image.fromarray(crop.astype(np.float32))
    return np.array(pil.resize((display_res, display_res), Image.BILINEAR))


def render_seq_bin(seq, bin_lo, bin_hi, good, bad, model, annotations,
                   root, warp_resolution, depth_thresh, process_res, out_path,
                   display_res=256):
    NCOLS = 6
    chosen = [("good", r) for r in good] + [("bad", r) for r in bad]
    if not chosen:
        return False

    fig, axes = plt.subplots(len(chosen), NCOLS,
                             figsize=(15.5, 2.5 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]
    cache = {}
    for row_i, (tag, r) in enumerate(chosen):
        ia, ib = int(r["frame_a"]), int(r["frame_b"])
        for fid in (ia, ib):
            if fid not in cache:
                cache[fid] = get_frame_data(
                    model, annotations[seq][fid], root, process_res
                )
        A, B = cache[ia], cache[ib]
        try:
            # Full-image warps at warp_resolution × warp_resolution covering
            # the entire image; we crop the warped output to A's bbox after.
            warp_ab, conf_ab = warp_pair_full(A, B, warp_resolution, depth_thresh)
            warp_ba, conf_ba = warp_pair_full(B, A, warp_resolution, depth_thresh)
        except Exception as e:
            print(f"    {seq} ({ia},{ib}) failed: {e}")
            continue

        # Load full images (no crop) and resize to the warp grid resolution
        full_a = np.array(Image.open(root / annotations[seq][ia]["filepath"]).convert("RGB"))
        full_b = np.array(Image.open(root / annotations[seq][ib]["filepath"]).convert("RGB"))
        full_a_w = np.array(Image.fromarray(full_a).resize(
            (warp_resolution, warp_resolution), Image.BILINEAR))
        full_b_w = np.array(Image.fromarray(full_b).resize(
            (warp_resolution, warp_resolution), Image.BILINEAR))

        # grid_sample full B with warp_ab -> full-image B-at-A's-locations
        warped_b_full = apply_warp(to_tensor(full_b_w), warp_ab).permute(1, 2, 0).numpy()
        warped_a_full = apply_warp(to_tensor(full_a_w), warp_ba).permute(1, 2, 0).numpy()

        # Display: crop everything to A's / B's bbox at display_res
        img_a = crop_resize(full_a, A["bbox"], display_res)
        img_b = crop_resize(full_b, B["bbox"], display_res)
        warped_b_to_a = crop_full_to_bbox(
            warped_b_full, A["bbox"], A["image_size"], display_res
        )
        warped_a_to_b = crop_full_to_bbox(
            warped_a_full, B["bbox"], B["image_size"], display_res
        )
        depth_a = crop_depth(A["da3_aligned"], A["bbox"], display_res)
        depth_b = crop_depth(B["da3_aligned"], B["bbox"], display_res)

        for ax in axes[row_i]:
            ax.set_xticks([]); ax.set_yticks([])
        color = "tab:green" if tag == "good" else "tab:red"
        axes[row_i, 0].imshow(img_a)
        axes[row_i, 0].set_title(
            f"[{tag}] A f{ia}  conf={r['conf_da3']:.2f}  d={r['distance']:.2f}",
            fontsize=8, color=color,
        )
        axes[row_i, 1].imshow(img_b)
        l1 = r["warp_l1_da3_vs_gt"]
        l1_str = "nan" if np.isnan(l1) else f"{l1:.4f}"
        axes[row_i, 1].set_title(f"B f{ib}  L1={l1_str}", fontsize=8, color=color)
        # Confidence mean restricted to the displayed bbox region
        conf_ab_crop = crop_full_to_bbox(
            conf_ab.numpy(), A["bbox"], A["image_size"], display_res
        )
        conf_ba_crop = crop_full_to_bbox(
            conf_ba.numpy(), B["bbox"], B["image_size"], display_res
        )
        axes[row_i, 2].imshow(np.clip(warped_b_to_a, 0, 1))
        axes[row_i, 2].set_title(
            f"warp_ab (B→A)\nconf_bbox={float(conf_ab_crop.mean()):.2f}", fontsize=7,
        )
        axes[row_i, 3].imshow(np.clip(warped_a_to_b, 0, 1))
        axes[row_i, 3].set_title(
            f"warp_ba (A→B)\nconf_bbox={float(conf_ba_crop.mean()):.2f}", fontsize=7,
        )
        im_da = axes[row_i, 4].imshow(
            np.where(depth_a > 0, depth_a, np.nan), cmap="turbo"
        )
        axes[row_i, 4].set_title("DA3 depth A", fontsize=8)
        plt.colorbar(im_da, ax=axes[row_i, 4], fraction=0.046)
        im_db = axes[row_i, 5].imshow(
            np.where(depth_b > 0, depth_b, np.nan), cmap="turbo"
        )
        axes[row_i, 5].set_title("DA3 depth B", fontsize=8)
        plt.colorbar(im_db, ax=axes[row_i, 5], fraction=0.046)

    fig.suptitle(
        f"{seq}   bin d ∈ [{bin_lo:.2f}, {bin_hi:.2f}]   "
        f"good (conf>{0.8}) / bad (conf<{0.5})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--annotation_file", required=True)
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--bin_ranges", type=float, nargs="+",
                    default=[0.7, 1.0, 1.0, 1.5, 1.5, 2.0],
                    help="Flat list of (lo hi) pairs to render. "
                         "Default = (0.7 1.0) (1.0 1.5) (1.5 2.0).")
    ap.add_argument("--good_thresh", type=float, default=0.8)
    ap.add_argument("--bad_thresh", type=float, default=0.5)
    ap.add_argument("--n_per_class", type=int, default=5)
    ap.add_argument("--warp_resolution", type=int, default=720,
                    help="Full-image warp grid size; warped output is cropped to bbox.")
    ap.add_argument("--depth_consistency_threshold", type=float, default=0.1)
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if len(args.bin_ranges) % 2 != 0:
        raise SystemExit("--bin_ranges must be flat lo/hi pairs")
    target_ranges = list(zip(args.bin_ranges[::2], args.bin_ranges[1::2]))

    sweep_dir = Path(args.sweep_dir)
    rows = list(csv.DictReader(open(sweep_dir / "per_pair.csv")))
    for r in rows:
        r["frame_a"] = int(r["frame_a"])
        r["frame_b"] = int(r["frame_b"])
        r["distance"] = float(r["distance"])
        r["bin_lo"] = float(r["bin_lo"])
        r["bin_hi"] = float(r["bin_hi"])
        r["conf_da3"] = float(r["conf_da3"])
        r["warp_l1_da3_vs_gt"] = (
            float("nan") if r["warp_l1_da3_vs_gt"] in ("", "nan")
            else float(r["warp_l1_da3_vs_gt"])
        )

    rng = np.random.default_rng(args.seed)
    print(f"Loading annotations from {args.annotation_file}")
    annotations = load_annotations(args.annotation_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)
    root = Path(args.root_dir)

    out_root = sweep_dir / "good_bad_grids"
    out_root.mkdir(exist_ok=True)

    for lo, hi in target_ranges:
        bin_rows = [r for r in rows
                    if abs(r["bin_lo"] - lo) < 1e-6 and abs(r["bin_hi"] - hi) < 1e-6]
        if not bin_rows:
            print(f"  bin [{lo:.2f}, {hi:.2f}]: no rows match, skipping")
            continue

        bin_dir = out_root / f"bin_{lo:.2f}-{hi:.2f}"
        bin_dir.mkdir(exist_ok=True)

        by_seq = {}
        for r in bin_rows:
            by_seq.setdefault(r["sequence"], []).append(r)

        print(f"\nBin [{lo:.2f}, {hi:.2f}]: {len(by_seq)} sequences, "
              f"{len(bin_rows)} pairs")
        n_written = 0
        for seq in sorted(by_seq.keys()):
            seq_rows = by_seq[seq]
            good_pool = [r for r in seq_rows if r["conf_da3"] > args.good_thresh]
            bad_pool = [r for r in seq_rows if r["conf_da3"] < args.bad_thresh]
            if not good_pool and not bad_pool:
                continue

            def sample(pool, n):
                if len(pool) <= n:
                    return pool
                idx = rng.choice(len(pool), size=n, replace=False)
                return [pool[int(i)] for i in idx]

            good = sorted(sample(good_pool, args.n_per_class),
                          key=lambda r: -r["conf_da3"])
            bad = sorted(sample(bad_pool, args.n_per_class),
                         key=lambda r: r["conf_da3"])

            out_path = bin_dir / f"{seq}.png"
            ok = render_seq_bin(
                seq, lo, hi, good, bad, model, annotations, root,
                args.warp_resolution, args.depth_consistency_threshold,
                args.process_res, out_path,
            )
            if ok:
                print(f"  wrote {out_path}  (good={len(good)}, bad={len(bad)})")
                n_written += 1
        print(f"  -> {n_written} figures in {bin_dir}")


if __name__ == "__main__":
    main()
