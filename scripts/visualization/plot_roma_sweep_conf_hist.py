"""
Per-bin histogram of RoMA warp confidence from per_pair.csv.

Small multiples: one histogram per distance bin, shared x-axis (confidence
in [0, 1]). Mean + median overlaid as vertical lines.

Usage:
    python scripts/visualization/plot_roma_sweep_conf_hist.py --sweep_dir eval_outputs/roma_distance_sweep
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--out_name", default="conf_hist_per_bin.png")
    ap.add_argument("--bins", type=int, default=30, help="Histogram bins")
    ap.add_argument("--ncols", type=int, default=4)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    rows = list(csv.DictReader(open(sweep_dir / "per_pair.csv")))
    by_bin = {}
    for r in rows:
        bi = int(r["bin"])
        by_bin.setdefault(bi, []).append({
            "conf_roma": float(r["conf_roma"]),
            "lo": float(r["bin_lo"]),
            "hi": float(r["bin_hi"]),
        })

    bins_sorted = sorted(by_bin.keys())
    n = len(bins_sorted)
    ncols = min(args.ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows),
                             sharex=True, sharey=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, bi in enumerate(bins_sorted):
        ax = axes[idx // ncols, idx % ncols]
        vals = np.array([g["conf_roma"] for g in by_bin[bi]])
        lo, hi = by_bin[bi][0]["lo"], by_bin[bi][0]["hi"]
        ax.hist(vals, bins=args.bins, range=(0, 1),
                color="tab:blue", alpha=0.75, edgecolor="white", linewidth=0.4)
        mean, median = float(vals.mean()), float(np.median(vals))
        ax.axvline(mean, color="black", linewidth=1.4, label=f"mean={mean:.2f}")
        ax.axvline(median, color="tab:red", linewidth=1.0, linestyle="--",
                   label=f"med={median:.2f}")
        ax.set_title(f"d ∈ [{lo:.2f}, {hi:.2f}]   n={len(vals)}", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("RoMA warp confidence")
        if idx % ncols == 0:
            ax.set_ylabel("pairs")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle("RoMA warp confidence distribution per distance bin", fontsize=11)
    fig.tight_layout()
    out = sweep_dir / args.out_name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"bins: {n}  total pairs: {sum(len(v) for v in by_bin.values())}")


if __name__ == "__main__":
    main()
