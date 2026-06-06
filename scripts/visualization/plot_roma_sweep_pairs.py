"""
Render per-sequence pair grids with RoMA2 warps.

This mirrors the DA3 pair-grid workflow, but instead of plotting depth maps,
it plots RoMA2 warp-field maps. Warped images fill low-confidence/missing
regions with a translucent target-image background instead of pure white.

Usage:
    python scripts/visualization/plot_roma_sweep_pairs.py \
        --sweep_dir eval_outputs/da3_distance_sweep \
        --annotation_file data/co3d_annotations/hydrant_train_50seq_depth.jgz \
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full

Notes:
- Requires a per-pair CSV at <sweep_dir>/per_pair.csv with at least:
  sequence, frame_a, frame_b, distance.
- Pair ranking defaults to auto-detecting a score column; override with
  --score_key and --score_mode.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.roma_metrics import compute_roma_correspondences, load_roma_model  # noqa: E402
from warps.precompute_depth_warps import load_annotations  # noqa: E402
from data_process.co3d_dataset import square_bbox  # noqa: E402


def to_tensor(rgb):
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def pick_highlighted_sequences(rows, score_key, lower_is_better):
    seq_scores = {}
    for r in rows:
        if score_key not in r:
            continue
        v = r[score_key]
        if np.isnan(v):
            continue
        seq_scores.setdefault(r["sequence"], []).append(v)

    if not seq_scores:
        return sorted({r["sequence"] for r in rows})[:9]

    means = {s: float(np.mean(vs)) for s, vs in seq_scores.items()}
    sorted_s = sorted(means.items(), key=lambda kv: kv[1], reverse=not lower_is_better)
    overall = float(np.mean(list(means.values())))
    best3 = [s for s, _ in sorted_s[:3]]
    worst3 = [s for s, _ in sorted_s[-3:]]
    mid_pool = [(s, v) for s, v in means.items() if s not in best3 and s not in worst3]
    mid3 = [s for s, _ in sorted(mid_pool, key=lambda kv: abs(kv[1] - overall))[:3]]
    return best3 + mid3 + worst3


def auto_score_key(rows):
    if not rows:
        return None
    candidates = [
        "warp_l1_roma_vs_gt",
        "warp_l1_roma",
        "warp_l1",
        "warp_l1_da3_vs_gt",
        "conf_roma",
        "valid_fraction_ab",
        "conf_da3",
    ]
    for k in candidates:
        if k in rows[0]:
            return k
    return None


def auto_score_mode(score_key):
    if score_key is None:
        return "none"
    if any(tag in score_key.lower() for tag in ["conf", "fraction", "iou", "sim", "psnr"]):
        return "high"
    return "low"


def parse_rows(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["frame_a"] = int(r["frame_a"])
        r["frame_b"] = int(r["frame_b"])
        r["distance"] = float(r["distance"])
        for k, v in list(r.items()):
            if k in {"sequence", "frame_a", "frame_b"}:
                continue
            if isinstance(v, str):
                vv = v.strip().lower()
                if vv in {"", "nan", "none"}:
                    r[k] = float("nan")
                    continue
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def load_frame_rgb(frame, root, image_size, use_bbox_crop):
    img_path = root / frame["filepath"]
    rgb = np.array(Image.open(img_path).convert("RGB"))

    bbox = None
    if use_bbox_crop and "bbox" in frame:
        bbox = np.around(square_bbox(np.array(frame["bbox"]))).astype(int)

    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = rgb.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 > x1 and y2 > y1:
            rgb = rgb[y1:y2, x1:x2]

    rgb = np.array(Image.fromarray(rgb).resize((image_size, image_size), Image.LANCZOS))
    return rgb


def resize_roma_outputs(warp, overlap, image_size):
    # warp: (1, H, W, 2), overlap: (1, H, W, 1)
    if warp.shape[1] != image_size or warp.shape[2] != image_size:
        warp = F.interpolate(
            warp.permute(0, 3, 1, 2),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        overlap = F.interpolate(
            overlap.permute(0, 3, 1, 2),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
    return warp[0], overlap[0]


def warp_field_to_rgb(warp_hw2):
    # Normalized coords [-1,1] -> RGB visualization
    warp_norm = (warp_hw2 + 1.0) / 2.0
    rgb = np.zeros((warp_hw2.shape[0], warp_hw2.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = warp_norm[..., 0]
    rgb[..., 1] = warp_norm[..., 1]
    rgb[..., 2] = 0.5
    return np.clip(rgb, 0.0, 1.0)


def warp_with_translucent_target_bg(src_img, target_img, warp, overlap, bg_alpha):
    """Warp source image and fill low-confidence regions with translucent target.

    Args:
        src_img: (3, H, W) in [0,1]
        target_img: (3, H, W) in [0,1]
        warp: (H, W, 2)
        overlap: (H, W, 1)
        bg_alpha: amount of target image in fallback background
    """
    warped = F.grid_sample(
        src_img[None],
        warp[None],
        mode="bilinear",
        align_corners=False,
    )[0]

    in_bounds = (warp.abs() <= 1.0).all(dim=-1, keepdim=True).float()
    conf = torch.clamp(overlap * in_bounds, 0.0, 1.0).permute(2, 0, 1)

    # Slight transparency effect: mix target against white before composing.
    translucent_target = bg_alpha * target_img + (1.0 - bg_alpha) * torch.ones_like(target_img)
    blended = conf * warped + (1.0 - conf) * translucent_target
    return blended.permute(1, 2, 0).cpu().numpy(), float(conf.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--annotation_file", required=True)
    ap.add_argument("--root_dir", required=True)
    ap.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Override sequence names. Default picks best/mid/worst by score.",
    )
    ap.add_argument(
        "--roma_setting",
        type=str,
        default="fast",
        choices=["fast", "precise", "turbo", "base"],
    )
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--confidence_threshold", type=float, default=0.3)
    ap.add_argument("--score_key", type=str, default="auto")
    ap.add_argument(
        "--score_mode",
        type=str,
        default="auto",
        choices=["auto", "low", "high", "none"],
        help="How to rank pairs: low=smaller better, high=larger better, none=random.",
    )
    ap.add_argument("--bg_alpha", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_subdir", default="pair_grids_roma")
    ap.add_argument("--use_bbox_crop", action="store_true")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    rows = parse_rows(sweep_dir / "per_pair.csv")

    score_key = auto_score_key(rows) if args.score_key == "auto" else args.score_key
    score_mode = auto_score_mode(score_key) if args.score_mode == "auto" else args.score_mode
    lower_is_better = score_mode == "low"

    seqs = args.sequences or pick_highlighted_sequences(rows, score_key, lower_is_better)
    print(f"Rendering {len(seqs)} sequences: {seqs}")
    print(f"Score key: {score_key} | score mode: {score_mode}")

    out_dir = sweep_dir / args.out_subdir
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading annotations from {args.annotation_file}")
    annotations = load_annotations(args.annotation_file)
    root = Path(args.root_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading RoMA2 ({args.roma_setting}) on {device}")
    roma = load_roma_model(setting=args.roma_setting, device=device, compile=False)

    rows_by_seq = {}
    for r in rows:
        if r["sequence"] not in seqs:
            continue
        rows_by_seq.setdefault(r["sequence"], []).append(r)

    NCOLS = 6
    for seq in seqs:
        seq_rows = rows_by_seq.get(seq) or []
        if not seq_rows:
            print(f"  {seq}: no rows, skipping")
            continue

        if score_mode == "none" or score_key is None:
            chosen_rows = seq_rows if len(seq_rows) <= 4 else [
                seq_rows[int(i)] for i in rng.choice(len(seq_rows), size=4, replace=False)
            ]
            chosen = [(f"rand{i + 1}", r) for i, r in enumerate(chosen_rows)]
        else:
            usable = [r for r in seq_rows if score_key in r and not np.isnan(r[score_key])]
            if not usable:
                print(f"  {seq}: no rows with score '{score_key}', skipping")
                continue
            srt = sorted(usable, key=lambda r: r[score_key], reverse=not lower_is_better)
            best, worst = srt[0], srt[-1]
            pool = [r for r in srt if r is not best and r is not worst]
            n_rand = min(2, len(pool))
            rand = [pool[int(i)] for i in rng.choice(len(pool), size=n_rand, replace=False)] if n_rand else []
            chosen = [("best", best), ("worst", worst)] + [(f"rand{i + 1}", r) for i, r in enumerate(rand)]

        fig, axes = plt.subplots(len(chosen), NCOLS, figsize=(16, 2.6 * len(chosen)))
        if len(chosen) == 1:
            axes = axes[None, :]

        frame_cache = {}
        for row_i, (tag, r) in enumerate(chosen):
            ia, ib = r["frame_a"], r["frame_b"]

            for fid in (ia, ib):
                if fid not in frame_cache:
                    frame_cache[fid] = load_frame_rgb(
                        annotations[seq][fid],
                        root,
                        args.image_size,
                        args.use_bbox_crop,
                    )

            img_a = frame_cache[ia]
            img_b = frame_cache[ib]
            img_a_pil = Image.fromarray(img_a)
            img_b_pil = Image.fromarray(img_b)

            try:
                roma_out = compute_roma_correspondences(
                    roma,
                    img_a_pil,
                    img_b_pil,
                    confidence_threshold=args.confidence_threshold,
                    latent_resolution=32,
                )
            except Exception as e:
                print(f"    {seq} ({ia},{ib}) failed: {e}")
                continue

            warp_ab, overlap_ab = resize_roma_outputs(
                roma_out["warp_ab"].to(device),
                roma_out["overlap_ab"].to(device),
                args.image_size,
            )
            warp_ba, overlap_ba = resize_roma_outputs(
                roma_out["warp_ba"].to(device),
                roma_out["overlap_ba"].to(device),
                args.image_size,
            )

            img_a_t = to_tensor(img_a).to(device)
            img_b_t = to_tensor(img_b).to(device)

            # A->B warp maps A-coordinates to B sampling positions.
            # So B warped with warp_ab lands in A's frame.
            warped_b_to_a, conf_ab = warp_with_translucent_target_bg(
                img_b_t, img_a_t, warp_ab, overlap_ab, args.bg_alpha
            )
            # B->A warp maps B-coordinates to A sampling positions.
            # So A warped with warp_ba lands in B's frame.
            warped_a_to_b, conf_ba = warp_with_translucent_target_bg(
                img_a_t, img_b_t, warp_ba, overlap_ba, args.bg_alpha
            )

            warp_ab_vis = warp_field_to_rgb(warp_ab.detach().cpu().numpy())
            warp_ba_vis = warp_field_to_rgb(warp_ba.detach().cpu().numpy())

            for ax in axes[row_i]:
                ax.set_xticks([])
                ax.set_yticks([])

            score_txt = ""
            if score_key in r and isinstance(r[score_key], float) and not np.isnan(r[score_key]):
                score_txt = f"  {score_key}={r[score_key]:.4f}"

            axes[row_i, 0].imshow(img_a)
            axes[row_i, 0].set_title(f"[{tag}] A f{ia}  d={r['distance']:.2f}{score_txt}", fontsize=8)
            axes[row_i, 1].imshow(img_b)
            axes[row_i, 1].set_title(f"B f{ib}", fontsize=8)
            axes[row_i, 2].imshow(np.clip(warped_b_to_a, 0, 1))
            axes[row_i, 2].set_title(f"B->A (translucent bg)\nconf={conf_ab:.2f}", fontsize=7)
            axes[row_i, 3].imshow(np.clip(warped_a_to_b, 0, 1))
            axes[row_i, 3].set_title(f"A->B (translucent bg)\nconf={conf_ba:.2f}", fontsize=7)
            axes[row_i, 4].imshow(warp_ab_vis)
            axes[row_i, 4].set_title("warp map A->B", fontsize=8)
            axes[row_i, 5].imshow(warp_ba_vis)
            axes[row_i, 5].set_title("warp map B->A", fontsize=8)

        fig.suptitle(f"{seq}  - RoMA2 warps (best / worst / random)", fontsize=10)
        fig.tight_layout()
        out = out_dir / f"{seq}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
