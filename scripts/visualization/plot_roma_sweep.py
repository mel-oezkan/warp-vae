"""
Re-plot a distance sweep for RoMA results from per_pair.csv.

This is analogous to plot_da3_sweep.py, but with auto-detection of RoMA metric
columns so it works across slightly different CSV schemas.

Priority for confidence-like metric:
  conf_roma, valid_fraction_ab, conf_da3

Priority for error-like metric:
  warp_l1_roma_vs_gt, warp_l1_roma, warp_l1, warp_l1_da3_vs_gt

Usage:
    python scripts/visualization/plot_roma_sweep.py --sweep_dir eval_outputs/da3_distance_sweep
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pick_metric_key(header, keys):
    for k in keys:
        if k in header:
            return k
    return None


def parse_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if s in {"", "nan", "none"}:
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True,
                    help="Dir containing per_pair.csv")
    ap.add_argument("--out_name", default="sweep_roma_highlighted.png")
    ap.add_argument("--conf_key", default="auto",
                    help="Confidence column name or 'auto'")
    ap.add_argument("--err_key", default="auto",
                    help="Error column name or 'auto'")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    rows = list(csv.DictReader(open(sweep_dir / "per_pair.csv")))
    if not rows:
        raise SystemExit("per_pair.csv is empty")

    header = rows[0].keys()
    conf_key = args.conf_key
    if conf_key == "auto":
        conf_key = pick_metric_key(
            header,
            ["conf_roma", "valid_fraction_ab", "conf_da3"],
        )
    err_key = args.err_key
    if err_key == "auto":
        err_key = pick_metric_key(
            header,
            ["warp_l1_roma_vs_gt", "warp_l1_roma", "warp_l1", "warp_l1_da3_vs_gt"],
        )

    if conf_key is None:
        raise SystemExit("No confidence metric column found. Pass --conf_key explicitly.")
    if err_key is None:
        raise SystemExit("No error metric column found. Pass --err_key explicitly.")

    for r in rows:
        r["distance"] = parse_float(r.get("distance", "nan"))
        r["bin"] = int(r["bin"])
        r["bin_lo"] = parse_float(r["bin_lo"])
        r["bin_hi"] = parse_float(r["bin_hi"])
        r[conf_key] = parse_float(r.get(conf_key, "nan"))
        r[err_key] = parse_float(r.get(err_key, "nan"))

    rows = [r for r in rows if np.isfinite(r["distance"])]
    if not rows:
        raise SystemExit("No valid rows with numeric distance")

    seq_names = sorted({r["sequence"] for r in rows})
    bins = sorted({r["bin"] for r in rows})

    bin_center = {}
    for bi in bins:
        match = next(r for r in rows if r["bin"] == bi)
        bin_center[bi] = (match["bin_lo"] + match["bin_hi"]) / 2

    by_seq_bin = {}
    by_bin = {}
    for r in rows:
        by_seq_bin.setdefault((r["sequence"], r["bin"]), []).append(r)
        by_bin.setdefault(r["bin"], []).append(r)

    # Rank sequences by mean error (lower is better)
    seq_mean_err = {}
    for seq in seq_names:
        vals = [r[err_key] for r in rows if r["sequence"] == seq and not np.isnan(r[err_key])]
        if vals:
            seq_mean_err[seq] = float(np.mean(vals))

    sorted_seqs = sorted(seq_mean_err.items(), key=lambda kv: kv[1])
    overall_mean = float(np.mean(list(seq_mean_err.values()))) if seq_mean_err else 0.0
    best3 = [s for s, _ in sorted_seqs[:3]]
    worst3 = [s for s, _ in sorted_seqs[-3:]]
    mid_pool = [(s, v) for s, v in seq_mean_err.items() if s not in best3 and s not in worst3]
    mid3 = [s for s, _ in sorted(mid_pool, key=lambda kv: abs(kv[1] - overall_mean))[:3]]

    highlight = {
        **{s: ("best", "tab:green") for s in best3},
        **{s: ("worst", "tab:red") for s in worst3},
        **{s: ("mid", "tab:blue") for s in mid3},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for seq in seq_names:
        xs, conf_vals, err_vals = [], [], []
        for bi in bins:
            group = by_seq_bin.get((seq, bi))
            if not group:
                continue
            conf_group = [g[conf_key] for g in group if not np.isnan(g[conf_key])]
            err_group = [g[err_key] for g in group if not np.isnan(g[err_key])]
            c = float(np.mean(conf_group)) if conf_group else float("nan")
            e = float(np.mean(err_group)) if err_group else float("nan")
            if np.isnan(c) and np.isnan(e):
                continue
            xs.append(bin_center[bi])
            conf_vals.append(c)
            err_vals.append(e)

        if not xs:
            continue

        if seq in highlight:
            tag, color = highlight[seq]
            label = f"{tag}: {seq[:18]}"
            axes[0].plot(xs, conf_vals, marker="o", alpha=0.9, color=color,
                         label=label, linewidth=1.6)
            axes[1].plot(xs, err_vals, marker="o", alpha=0.9, color=color,
                         label=label, linewidth=1.6)
        else:
            axes[0].plot(xs, conf_vals, marker=".", alpha=0.15, color="gray", linewidth=0.8)
            axes[1].plot(xs, err_vals, marker=".", alpha=0.15, color="gray", linewidth=0.8)

    agg_x, agg_conf, agg_err = [], [], []
    for bi in bins:
        group = by_bin[bi]
        agg_x.append(bin_center[bi])
        agg_conf.append(np.nanmean([g[conf_key] for g in group]))
        agg_err.append(np.nanmean([g[err_key] for g in group]))

    axes[0].plot(agg_x, agg_conf, "k-", linewidth=2.5, label="mean")
    axes[1].plot(agg_x, agg_err, "k-", linewidth=2.5, label="mean")

    axes[0].set_xlabel("camera distance (units)")
    axes[0].set_ylabel(conf_key)
    axes[0].set_title("RoMA confidence-like metric vs distance")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")

    axes[1].set_xlabel("camera distance (units)")
    axes[1].set_ylabel(err_key)
    axes[1].set_title("RoMA error-like metric vs distance")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")

    fig.tight_layout()
    out = sweep_dir / args.out_name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    print(f"used conf_key={conf_key}, err_key={err_key}")
    print(f"highlighted: best={best3} mid={mid3} worst={worst3}")


if __name__ == "__main__":
    main()
