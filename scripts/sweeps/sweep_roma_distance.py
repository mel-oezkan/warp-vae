"""
Sweep RoMA-warp quality vs camera distance, per sequence.

For each of N sequences, sample up to M frame pairs spread across
camera-distance bins, compute:
  (a) GT-depth warp confidence
  (b) RoMA overlap confidence
  (c) RoMA vs GT warp L1 in normalized grid coords

Outputs:
  <output_dir>/per_pair.csv       - every pair: seq, dist, conf_gt, conf_roma, L1
  <output_dir>/per_sequence.csv   - mean per (sequence, bin)
  <output_dir>/per_bin.csv        - mean across all sequences per bin
  <output_dir>/sweep.png          - conf + L1 vs distance, one line per sequence

Usage:
    python scripts/sweeps/sweep_roma_distance.py \
        --annotation_file data/co3d_annotations/hydrant_train_50seq_depth.jgz \
        --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \
        --output_dir eval_outputs/roma_distance_sweep \
        --num_sequences 15 --pairs_per_sequence 50
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

from src.analysis.roma_metrics import load_roma_model  # noqa: E402
from warps.precompute_depth_warps import (  # noqa: E402
    build_intrinsic_matrix,
    compute_depth_warp,
    compute_sequence_distance_matrix,
    load_annotations,
    load_co3d_depth,
)
from data_process.co3d_dataset import square_bbox  # noqa: E402


def stratified_pairs(seq_frames, n_pairs, bin_edges, rng):
    """Sample up to n_pairs across distance bins in round-robin order."""
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


def load_frame_geom(frame, root):
    img_path = root / frame["filepath"]
    depth_path = root / frame["depth_path"]
    gt = load_co3d_depth(str(depth_path), frame.get("depth_scale_adjustment", 1.0))

    image_size = tuple(frame["image_size"])
    K = build_intrinsic_matrix(
        np.array(frame["focal_length"]),
        np.array(frame["principal_point"]),
        image_size,
    )

    bbox = None
    if "bbox" in frame:
        bbox = np.around(square_bbox(np.array(frame["bbox"]))).astype(int)

    return {
        "img_path": img_path,
        "gt": gt,
        "K": K,
        "R": np.array(frame["R"]),
        "T": np.array(frame["T"]),
        "image_size": image_size,
        "bbox": bbox,
    }


def crop_resize_rgb(img_path, bbox, resolution):
    rgb = np.array(Image.open(img_path).convert("RGB"))
    pil = Image.fromarray(rgb)
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pil = pil.crop((x1, y1, x2, y2))
    pil = pil.resize((resolution, resolution), Image.LANCZOS)
    return pil


def compute_gt_warp(A, B, warp_resolution, depth_consistency_threshold):
    valid_a_gt = (A["gt"] > 0) & np.isfinite(A["gt"])
    valid_b_gt = (B["gt"] > 0) & np.isfinite(B["gt"])

    warp_gt, conf_gt = compute_depth_warp(
        A["gt"],
        valid_a_gt,
        A["R"],
        A["T"],
        A["K"],
        B["gt"],
        valid_b_gt,
        B["R"],
        B["T"],
        B["K"],
        warp_resolution=warp_resolution,
        image_size_a=A["image_size"],
        image_size_b=B["image_size"],
        depth_consistency_threshold=depth_consistency_threshold,
        crop_bbox_a=A["bbox"],
        crop_bbox_b=B["bbox"],
    )
    return warp_gt, conf_gt


def compute_roma_warp_and_conf(roma_model, img_a_pil, img_b_pil, warp_resolution, device):
    with torch.no_grad():
        pred = roma_model.match(img_a_pil, img_b_pil)

    warp = pred["warp_AB"]  # (1, H, W, 2)
    overlap = pred.get("overlap_AB", None)
    if overlap is None:
        overlap = pred["confidence_AB"].mean(dim=-1, keepdim=True)

    if warp.shape[1] != warp_resolution or warp.shape[2] != warp_resolution:
        warp = F.interpolate(
            warp.permute(0, 3, 1, 2),
            size=(warp_resolution, warp_resolution),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        overlap = F.interpolate(
            overlap.permute(0, 3, 1, 2),
            size=(warp_resolution, warp_resolution),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

    warp = warp[0].to(device=device)
    overlap = overlap[0].to(device=device)

    # Confidence should also respect in-bounds warp coordinates.
    in_bounds = (warp.abs() <= 1.0).all(dim=-1, keepdim=True).float()
    conf = torch.clamp(overlap * in_bounds, 0.0, 1.0)[..., 0]  # (H, W)

    return warp.cpu(), conf.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation_file", required=True)
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_sequences", type=int, default=15)
    ap.add_argument("--pairs_per_sequence", type=int, default=50)
    ap.add_argument("--warp_resolution", type=int, default=256)
    ap.add_argument("--depth_consistency_threshold", type=float, default=0.1)
    ap.add_argument("--roma_setting", default="fast", choices=["fast", "precise", "turbo", "base"])
    ap.add_argument(
        "--bin_edges",
        type=float,
        nargs="+",
        default=[0.05, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root_dir)
    bin_edges = list(args.bin_edges)
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    rng = np.random.default_rng(args.seed)

    print(f"Loading annotations from {args.annotation_file}")
    annotations = load_annotations(args.annotation_file)
    seq_names = list(annotations.keys())[: args.num_sequences]
    print(
        f"Sweeping {len(seq_names)} sequences, up to {args.pairs_per_sequence} pairs each, "
        f"bins={bin_edges}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading RoMA2 ({args.roma_setting}) onto {device}")
    roma = load_roma_model(setting=args.roma_setting, device=str(device), compile=False)

    rows = []
    for sidx, seq in enumerate(seq_names):
        frames = annotations[seq]
        if len(frames) < 2:
            continue

        chosen = stratified_pairs(frames, args.pairs_per_sequence, bin_edges, rng)
        if not chosen:
            print(f"  [{sidx + 1}/{len(seq_names)}] {seq}: no pairs in bins")
            continue

        frame_ids = sorted({a for _, a, _, _ in chosen} | {b for _, _, b, _ in chosen})
        cache = {fid: load_frame_geom(frames[fid], root) for fid in frame_ids}

        for bin_idx, ia, ib, dist in chosen:
            A, B = cache[ia], cache[ib]
            try:
                warp_gt, conf_gt = compute_gt_warp(
                    A,
                    B,
                    args.warp_resolution,
                    args.depth_consistency_threshold,
                )

                img_a = crop_resize_rgb(A["img_path"], A["bbox"], args.warp_resolution)
                img_b = crop_resize_rgb(B["img_path"], B["bbox"], args.warp_resolution)
                warp_roma, conf_roma = compute_roma_warp_and_conf(
                    roma,
                    img_a,
                    img_b,
                    args.warp_resolution,
                    device,
                )
            except Exception as e:
                print(f"    pair ({ia},{ib}) failed: {e}")
                continue

            warp_gt_np = warp_gt.numpy()
            conf_gt_np = conf_gt.numpy()
            warp_roma_np = warp_roma.numpy()
            conf_roma_np = conf_roma.numpy()

            both = (conf_gt_np > 0) & (conf_roma_np > 0)
            if both.any():
                l1 = float(np.mean(np.abs(warp_gt_np - warp_roma_np)[both]))
            else:
                l1 = float("nan")

            rows.append(
                {
                    "sequence": seq,
                    "frame_a": ia,
                    "frame_b": ib,
                    "distance": dist,
                    "bin": bin_idx,
                    "bin_lo": bin_edges[bin_idx],
                    "bin_hi": bin_edges[bin_idx + 1],
                    "conf_gt": float(conf_gt.mean()),
                    "conf_roma": float(conf_roma.mean()),
                    "warp_l1_roma_vs_gt": l1,
                }
            )

        print(
            f"  [{sidx + 1}/{len(seq_names)}] {seq}: {len(chosen)} pairs done "
            f"(unique frames: {len(frame_ids)})"
        )

    # Write per-pair CSV
    per_pair_csv = out_dir / "per_pair.csv"
    with open(per_pair_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Aggregate per-sequence/bin and per-bin
    by_seq_bin = {}
    by_bin = {}
    for r in rows:
        by_seq_bin.setdefault((r["sequence"], r["bin"]), []).append(r)
        by_bin.setdefault(r["bin"], []).append(r)

    per_seq_csv = out_dir / "per_sequence.csv"
    with open(per_seq_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sequence",
            "bin",
            "bin_lo",
            "bin_hi",
            "n",
            "mean_conf_gt",
            "mean_conf_roma",
            "mean_warp_l1",
        ])
        for (seq, bi), group in sorted(by_seq_bin.items()):
            w.writerow(
                [
                    seq,
                    bi,
                    bin_edges[bi],
                    bin_edges[bi + 1],
                    len(group),
                    float(np.mean([g["conf_gt"] for g in group])),
                    float(np.mean([g["conf_roma"] for g in group])),
                    float(np.nanmean([g["warp_l1_roma_vs_gt"] for g in group])),
                ]
            )

    per_bin_csv = out_dir / "per_bin.csv"
    with open(per_bin_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "bin",
            "bin_lo",
            "bin_hi",
            "n",
            "mean_conf_gt",
            "mean_conf_roma",
            "mean_warp_l1",
        ])
        for bi in sorted(by_bin):
            group = by_bin[bi]
            w.writerow(
                [
                    bi,
                    bin_edges[bi],
                    bin_edges[bi + 1],
                    len(group),
                    float(np.mean([g["conf_gt"] for g in group])),
                    float(np.mean([g["conf_roma"] for g in group])),
                    float(np.nanmean([g["warp_l1_roma_vs_gt"] for g in group])),
                ]
            )

    # Plot curves per sequence + aggregate
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seq in seq_names:
        xs, conf_roma_vals, l1_vals = [], [], []
        for bi in range(len(bin_edges) - 1):
            group = by_seq_bin.get((seq, bi))
            if not group:
                continue
            xs.append(bin_centers[bi])
            conf_roma_vals.append(np.mean([g["conf_roma"] for g in group]))
            l1_vals.append(np.nanmean([g["warp_l1_roma_vs_gt"] for g in group]))
        if xs:
            axes[0].plot(xs, conf_roma_vals, marker="o", alpha=0.6, label=seq[:14])
            axes[1].plot(xs, l1_vals, marker="o", alpha=0.6, label=seq[:14])

    agg_x, agg_conf, agg_l1 = [], [], []
    for bi in sorted(by_bin):
        agg_x.append(bin_centers[bi])
        agg_conf.append(np.mean([g["conf_roma"] for g in by_bin[bi]]))
        agg_l1.append(np.nanmean([g["warp_l1_roma_vs_gt"] for g in by_bin[bi]]))

    axes[0].plot(agg_x, agg_conf, "k-", linewidth=2.5, label="mean")
    axes[1].plot(agg_x, agg_l1, "k-", linewidth=2.5, label="mean")

    axes[0].set_xlabel("camera distance (units)")
    axes[0].set_ylabel("RoMA warp confidence")
    axes[0].set_title("Confidence vs distance (per sequence)")
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("camera distance (units)")
    axes[1].set_ylabel("warp L1 (RoMA vs GT, normalized)")
    axes[1].set_title("RoMA-vs-GT warp L1 vs distance")
    axes[1].grid(alpha=0.3)

    axes[0].legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {len(rows)} pairs to {per_pair_csv}")
    print(f"Per-sequence aggregate: {per_seq_csv}")
    print(f"Per-bin aggregate: {per_bin_csv}")
    print(f"Plot: {out_dir / 'sweep.png'}")


if __name__ == "__main__":
    main()
