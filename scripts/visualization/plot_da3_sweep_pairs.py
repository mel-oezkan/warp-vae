"""
Render per-sequence pair grids for the DA3 distance sweep.

For each requested sequence, picks 4 pairs from per_pair.csv (best L1, worst
L1, 2 random) and renders a 6-column grid: imA, imB, B->A warp, A->B warp,
DA3 depth A, DA3 depth B. Recomputes warps + depth on the fly (per-pair.csv
only stores summary stats).

Usage:
    python scripts/visualization/plot_da3_sweep_pairs.py \\
        --sweep_dir eval_outputs/da3_distance_sweep \\
        --annotation_file data/co3d_annotations/hydrant_train_50seq_depth.jgz \\
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full

Sequence selection (default: best/mid/worst 3 by mean L1, same as
plot_da3_sweep.py). Override with --sequences seq1 seq2 ...
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
from warps.precompute_depth_warps import load_annotations  # noqa: E402
from warps.precompute_da3_warps import apply_warp, crop_resize  # noqa: E402
from sweeps.sweep_da3_distance import get_frame_data, warp_pair  # noqa: E402


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


def pick_highlighted_sequences(rows):
    seq_l1 = {}
    for r in rows:
        v = r["warp_l1_da3_vs_gt"]
        if np.isnan(v):
            continue
        seq_l1.setdefault(r["sequence"], []).append(v)
    means = {s: float(np.mean(vs)) for s, vs in seq_l1.items()}
    sorted_s = sorted(means.items(), key=lambda kv: kv[1])
    overall = float(np.mean(list(means.values())))
    best3 = [s for s, _ in sorted_s[:3]]
    worst3 = [s for s, _ in sorted_s[-3:]]
    mid_pool = [(s, v) for s, v in means.items() if s not in best3 and s not in worst3]
    mid3 = [s for s, _ in sorted(mid_pool, key=lambda kv: abs(kv[1] - overall))[:3]]
    return best3 + mid3 + worst3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--annotation_file", required=True)
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--sequences", nargs="+", default=None,
                    help="Override: list of sequence names. Default = best+mid+worst 3.")
    ap.add_argument("--warp_resolution", type=int, default=256)
    ap.add_argument("--depth_consistency_threshold", type=float, default=0.1)
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    ap.add_argument("--process_res", type=int, default=504)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_subdir", default="pair_grids")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    rows = list(csv.DictReader(open(sweep_dir / "per_pair.csv")))
    for r in rows:
        r["frame_a"] = int(r["frame_a"])
        r["frame_b"] = int(r["frame_b"])
        r["distance"] = float(r["distance"])
        r["warp_l1_da3_vs_gt"] = (
            float("nan") if r["warp_l1_da3_vs_gt"] in ("", "nan")
            else float(r["warp_l1_da3_vs_gt"])
        )

    seqs = args.sequences or pick_highlighted_sequences(rows)
    print(f"Rendering {len(seqs)} sequences: {seqs}")

    out_dir = sweep_dir / args.out_subdir
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading annotations from {args.annotation_file}")
    annotations = load_annotations(args.annotation_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device}")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)
    root = Path(args.root_dir)

    rows_by_seq = {}
    for r in rows:
        if r["sequence"] not in seqs:
            continue
        if np.isnan(r["warp_l1_da3_vs_gt"]):
            continue
        rows_by_seq.setdefault(r["sequence"], []).append(r)

    NCOLS = 6
    for seq in seqs:
        seq_rows = rows_by_seq.get(seq) or []
        if not seq_rows:
            print(f"  {seq}: no usable pairs, skipping")
            continue
        srt = sorted(seq_rows, key=lambda r: r["warp_l1_da3_vs_gt"])
        best, worst = srt[0], srt[-1]
        pool = [r for r in srt if r is not best and r is not worst]
        n_rand = min(2, len(pool))
        rand = [pool[int(i)] for i in
                rng.choice(len(pool), size=n_rand, replace=False)] if n_rand else []
        chosen = [("best", best), ("worst", worst)] + [
            (f"rand{i+1}", r) for i, r in enumerate(rand)
        ]

        fig, axes = plt.subplots(len(chosen), NCOLS,
                                 figsize=(15.5, 2.5 * len(chosen)))
        if len(chosen) == 1:
            axes = axes[None, :]
        cache = {}
        for row_i, (tag, r) in enumerate(chosen):
            ia, ib = r["frame_a"], r["frame_b"]
            for fid in (ia, ib):
                if fid not in cache:
                    cache[fid] = get_frame_data(
                        model, annotations[seq][fid], root, args.process_res
                    )
            A, B = cache[ia], cache[ib]
            try:
                _, _, warp_ab, conf_ab = warp_pair(
                    A, B, args.warp_resolution, args.depth_consistency_threshold
                )
                _, _, warp_ba, conf_ba = warp_pair(
                    B, A, args.warp_resolution, args.depth_consistency_threshold
                )
            except Exception as e:
                print(f"    {seq} ({ia},{ib}) failed: {e}")
                continue

            img_a = crop_resize(
                np.array(Image.open(root / annotations[seq][ia]["filepath"]).convert("RGB")),
                A["bbox"], args.warp_resolution,
            )
            img_b = crop_resize(
                np.array(Image.open(root / annotations[seq][ib]["filepath"]).convert("RGB")),
                B["bbox"], args.warp_resolution,
            )
            warped_b_to_a = apply_warp(to_tensor(img_b), warp_ab).permute(1, 2, 0).numpy()
            warped_a_to_b = apply_warp(to_tensor(img_a), warp_ba).permute(1, 2, 0).numpy()
            depth_a = crop_depth(A["da3_aligned"], A["bbox"], args.warp_resolution)
            depth_b = crop_depth(B["da3_aligned"], B["bbox"], args.warp_resolution)

            for ax in axes[row_i]:
                ax.set_xticks([]); ax.set_yticks([])
            axes[row_i, 0].imshow(img_a)
            axes[row_i, 0].set_title(
                f"[{tag}] A f{ia}  d={r['distance']:.2f}", fontsize=8
            )
            axes[row_i, 1].imshow(img_b)
            axes[row_i, 1].set_title(
                f"B f{ib}  L1={r['warp_l1_da3_vs_gt']:.4f}", fontsize=8
            )
            axes[row_i, 2].imshow(np.clip(warped_b_to_a, 0, 1))
            axes[row_i, 2].set_title(
                f"warp_ab (B→A)\nconf={float(conf_ab.mean()):.2f}", fontsize=7
            )
            axes[row_i, 3].imshow(np.clip(warped_a_to_b, 0, 1))
            axes[row_i, 3].set_title(
                f"warp_ba (A→B)\nconf={float(conf_ba.mean()):.2f}", fontsize=7
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

        fig.suptitle(f"{seq}  — best / worst / random pairs (DA3)", fontsize=10)
        fig.tight_layout()
        out = out_dir / f"{seq}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
