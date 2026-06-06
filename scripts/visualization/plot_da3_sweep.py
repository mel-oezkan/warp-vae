"""
Re-plot the DA3 distance sweep from per_pair.csv (no recomputation).

Highlight 3 best, 3 worst, and 3 closest-to-mean sequences (ranked by mean
warp L1 across all of that sequence's bins). Other sequences are drawn faint
and unlabelled. The cross-sequence mean is plotted on top.

Usage:
    python scripts/visualization/plot_da3_sweep.py --sweep_dir eval_outputs/da3_distance_sweep
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True,
                    help="Dir containing per_pair.csv (output of sweep_da3_distance.py)")
    ap.add_argument("--out_name", default="sweep_highlighted.png")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    rows = list(csv.DictReader(open(sweep_dir / "per_pair.csv")))
    for r in rows:
        r["distance"] = float(r["distance"])
        r["bin"] = int(r["bin"])
        r["bin_lo"] = float(r["bin_lo"])
        r["bin_hi"] = float(r["bin_hi"])
        r["conf_da3"] = float(r["conf_da3"])
        r["warp_l1_da3_vs_gt"] = (
            float("nan") if r["warp_l1_da3_vs_gt"] in ("", "nan")
            else float(r["warp_l1_da3_vs_gt"])
        )

    seq_names = sorted({r["sequence"] for r in rows})
    bins = sorted({r["bin"] for r in rows})
    bin_center = {bi: (next(r["bin_lo"] for r in rows if r["bin"] == bi)
                       + next(r["bin_hi"] for r in rows if r["bin"] == bi)) / 2
                  for bi in bins}

    by_seq_bin = {}
    for r in rows:
        by_seq_bin.setdefault((r["sequence"], r["bin"]), []).append(r)
    by_bin = {}
    for r in rows:
        by_bin.setdefault(r["bin"], []).append(r)

    # Rank sequences by mean L1 across all their pairs
    seq_mean_l1 = {}
    for seq in seq_names:
        vals = [r["warp_l1_da3_vs_gt"] for r in rows
                if r["sequence"] == seq and not np.isnan(r["warp_l1_da3_vs_gt"])]
        if vals:
            seq_mean_l1[seq] = float(np.mean(vals))
    sorted_seqs = sorted(seq_mean_l1.items(), key=lambda kv: kv[1])
    overall_mean = float(np.mean(list(seq_mean_l1.values()))) if seq_mean_l1 else 0.0
    best3 = [s for s, _ in sorted_seqs[:3]]
    worst3 = [s for s, _ in sorted_seqs[-3:]]
    mid_pool = [(s, v) for s, v in seq_mean_l1.items()
                if s not in best3 and s not in worst3]
    mid3 = [s for s, _ in sorted(mid_pool, key=lambda kv: abs(kv[1] - overall_mean))[:3]]
    highlight = {**{s: ("best", "tab:green") for s in best3},
                 **{s: ("worst", "tab:red") for s in worst3},
                 **{s: ("mid", "tab:blue") for s in mid3}}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for seq in seq_names:
        xs, conf, l1 = [], [], []
        for bi in bins:
            group = by_seq_bin.get((seq, bi))
            if not group:
                continue
            xs.append(bin_center[bi])
            conf.append(np.mean([g["conf_da3"] for g in group]))
            l1.append(np.nanmean([g["warp_l1_da3_vs_gt"] for g in group]))
        if not xs:
            continue
        if seq in highlight:
            tag, color = highlight[seq]
            label = f"{tag}: {seq[:18]}"
            axes[0].plot(xs, conf, marker="o", alpha=0.9, color=color,
                         label=label, linewidth=1.6)
            axes[1].plot(xs, l1, marker="o", alpha=0.9, color=color,
                         label=label, linewidth=1.6)
        else:
            axes[0].plot(xs, conf, marker=".", alpha=0.15, color="gray",
                         linewidth=0.8)
            axes[1].plot(xs, l1, marker=".", alpha=0.15, color="gray",
                         linewidth=0.8)

    agg_x, agg_conf, agg_l1 = [], [], []
    for bi in bins:
        agg_x.append(bin_center[bi])
        agg_conf.append(np.mean([g["conf_da3"] for g in by_bin[bi]]))
        agg_l1.append(np.nanmean([g["warp_l1_da3_vs_gt"] for g in by_bin[bi]]))
    axes[0].plot(agg_x, agg_conf, "k-", linewidth=2.5, label="mean")
    axes[1].plot(agg_x, agg_l1, "k-", linewidth=2.5, label="mean")

    axes[0].set_xlabel("camera distance (units)")
    axes[0].set_ylabel("DA3 warp confidence")
    axes[0].set_title("Confidence vs distance")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    axes[1].set_xlabel("camera distance (units)")
    axes[1].set_ylabel("warp L1 (DA3 vs GT)")
    axes[1].set_title("DA3-vs-GT warp L1 vs distance")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = sweep_dir / args.out_name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"highlighted: best={best3}  mid={mid3}  worst={worst3}")


if __name__ == "__main__":
    main()
