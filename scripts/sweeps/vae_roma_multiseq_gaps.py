"""How often does RoMA produce a trainable warp at small frame gaps (2-5)?

Survey: 10 random CO3D hydrant sequences x 3 random anchors x gaps {2,3,4,5}.
For each (anchor, gap) we compute the RoMA warp from anchor frame x_A to target
x_B = anchor + gap, then plot:
  row 1: target x_B
  row 2: warp(x_A) in pixel space
  row 3: x_A with red overlay where |warp(x_A) - x_B| is large
  row 4: RoMA confidence mask (binary, threshold 0.8)
One figure per anchor (4 rows x 4 gap-columns). Plus a single summary figure
aggregating the fraction-of-confident-pixels across all (anchor, gap) pairs.

Run:
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/sweeps/vae_roma_multiseq_gaps.py
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.analysis.roma_metrics import load_roma_model
from warps.precompute_depth_warps import load_annotations

REPO = Path("/visinf/home/lab_mozkan/computer-vision-proj-lab")
ANNOT = REPO / "data/co3d_annotations/hydrant_train_50seq.jgz"
DATA_ROOT = Path("/data/lab_moezkan/co3d_full")
OUT_DIR = REPO / "outputs/scripts/vae_roma_sweep/multiseq_gap2to5"

IMAGE_SIZE = 256
GAPS = [2, 3, 4, 5]
CONF_THRESHOLD = 0.8
ROMA_SETTING = "fast"
N_SEQUENCES = 20
ANCHORS_PER_SEQUENCE = 3
SEED = 0
MIN_FRAMES_REQUIRED = max(GAPS) + 10  # need room around the anchor


# ---------- I/O ----------

def load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return tfm(img)


def load_image_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def to_display(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1, 1)
    return ((x + 1) / 2).permute(1, 2, 0).numpy()


def red_diff_overlay(src: torch.Tensor, warped: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor, strength: float = 2.0) -> np.ndarray:
    """Grayscale x_A with red where |warp(x_A) - x_B| is large.
    Outside the confidence mask we paint white so it doesn't compete with the signal."""
    src01 = ((src.detach().cpu().clamp(-1, 1) + 1) / 2)
    w01 = ((warped.detach().cpu().clamp(-1, 1) + 1) / 2)
    t01 = ((target.detach().cpu().clamp(-1, 1) + 1) / 2)
    m = mask.detach().cpu()                                         # (H, W)
    diff = (w01 - t01).abs().mean(dim=0)
    alpha = (diff * strength).clamp(0, 1) * m                       # zero outside mask
    gray = src01.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    red = torch.zeros_like(gray); red[0] = 1.0
    fg = gray * (1 - alpha) + red * alpha                           # (3, H, W)
    white = torch.ones_like(fg)
    out = fg * m.unsqueeze(0) + white * (1 - m.unsqueeze(0))
    return out.permute(1, 2, 0).numpy()


def warped_with_white_bg(warped: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Warped image with low-confidence pixels replaced by white."""
    w01 = ((warped.detach().cpu().clamp(-1, 1) + 1) / 2)            # (3, H, W)
    m = mask.detach().cpu().unsqueeze(0)                            # (1, H, W)
    out = w01 * m + torch.ones_like(w01) * (1 - m)
    return out.permute(1, 2, 0).numpy()


# ---------- RoMA ----------

@torch.no_grad()
def roma_warp(roma_model, pil_a, pil_b, device):
    pred = roma_model.match(pil_a, pil_b)
    warp = pred["warp_AB"]
    overlap = pred.get("overlap_AB")
    if overlap is None:
        overlap = pred["confidence_AB"].mean(dim=-1, keepdim=True)
    if warp.shape[1] != IMAGE_SIZE or warp.shape[2] != IMAGE_SIZE:
        warp = F.interpolate(warp.permute(0, 3, 1, 2), size=(IMAGE_SIZE, IMAGE_SIZE),
                             mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        overlap = F.interpolate(overlap.permute(0, 3, 1, 2), size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return warp.to(device), overlap.to(device)


def confidence_mask(conf_img: torch.Tensor, warp_img: torch.Tensor) -> torch.Tensor:
    in_bounds = (warp_img.abs() <= 1.0).all(dim=-1, keepdim=True).float()
    valid = (conf_img > CONF_THRESHOLD).float() * in_bounds
    return valid[0, ..., 0]  # (H,W)


# ---------- sweep ----------

def pick_anchors(seq_frames, n_anchors, rng) -> List[int]:
    n = len(seq_frames)
    lo = max(GAPS)
    hi = n - max(GAPS) - 1
    if hi <= lo:
        return []
    candidates = list(range(lo, hi + 1))
    rng.shuffle(candidates)
    return sorted(candidates[:n_anchors])


@torch.no_grad()
def process_anchor(roma_model, device, seq_frames, anchor_idx, seq_name):
    """Returns dict per gap: imgs/conf_mean. Also a list of (gap, frac_conf) for summary."""
    anchor_frame = seq_frames[anchor_idx]
    x_a_path = DATA_ROOT / anchor_frame["filepath"]
    x_a = load_image_tensor(x_a_path).to(device)
    pil_a = load_image_pil(x_a_path)

    per_gap = {}
    for g in GAPS:
        tgt_frame = seq_frames[anchor_idx + g]
        x_b_path = DATA_ROOT / tgt_frame["filepath"]
        x_b = load_image_tensor(x_b_path).to(device)
        pil_b = load_image_pil(x_b_path)

        warp_img, conf_img = roma_warp(roma_model, pil_a, pil_b, device)
        mask = confidence_mask(conf_img, warp_img)
        warped_pixels = F.grid_sample(x_a.unsqueeze(0), warp_img,
                                      mode="bilinear", padding_mode="zeros",
                                      align_corners=False)[0]

        # Masked photometric L1 in [0,1] space (convert from [-1,1] first).
        a01 = (x_a.clamp(-1, 1) + 1) / 2
        b01 = (x_b.clamp(-1, 1) + 1) / 2
        w01 = (warped_pixels.clamp(-1, 1) + 1) / 2
        denom = mask.sum().clamp_min(1.0) * a01.shape[0]
        baseline_l1 = float(((a01 - b01).abs() * mask).sum() / denom)
        warped_l1 = float(((w01 - b01).abs() * mask).sum() / denom)
        improvement = 1.0 - warped_l1 / max(baseline_l1, 1e-6)

        per_gap[g] = {
            "target": to_display(x_b),
            "warped": warped_with_white_bg(warped_pixels, mask),
            "diff": red_diff_overlay(x_a, warped_pixels, x_b, mask),
            "mask": mask.detach().cpu().numpy(),
            "frac_conf": float(mask.mean()),
            "baseline_l1": baseline_l1,
            "warped_l1": warped_l1,
            "improvement": improvement,
        }
    return per_gap


def save_anchor_grid(per_gap, seq_name: str, anchor_idx: int, out_path: Path):
    n = len(GAPS)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8.5))
    row_titles = ["target x_B",
                  "warp(x_A) [pixel-space]",
                  "x_A + red |warp(x_A) - x_B|",
                  f"RoMA conf mask (>{CONF_THRESHOLD})"]
    rows_key = ["target", "warped", "diff", "mask"]
    for r, key in enumerate(rows_key):
        for c, g in enumerate(GAPS):
            ax = axes[r, c]
            im = per_gap[g][key]
            if key == "mask":
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                d = per_gap[g]
                ax.set_title(
                    f"gap=+{g}  frac_conf={d['frac_conf']:.2f}\n"
                    f"baseline L1={d['baseline_l1']:.3f}  warped L1={d['warped_l1']:.3f}\n"
                    f"improvement={d['improvement']:+.2f}",
                    fontsize=8)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=10)
    short_seq = seq_name if len(seq_name) <= 30 else seq_name[:27] + "..."
    fig.suptitle(f"{short_seq}   anchor=f{anchor_idx:03d}   "
                 f"(RoMA={ROMA_SETTING}, thr={CONF_THRESHOLD})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_summary(records: List[dict], out_path: Path):
    """One row of plots per metric, columns per gap; plus a baseline-vs-warped scatter."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- Top-left: coverage per gap (jittered scatter + mean/median lines)
    by_gap_conf = {g: [r["frac_conf"] for r in records if r["gap"] == g] for g in GAPS}
    rng = np.random.default_rng(0)
    for g in GAPS:
        ys = by_gap_conf[g]
        xs = g + rng.uniform(-0.15, 0.15, size=len(ys))
        axes[0, 0].scatter(xs, ys, alpha=0.55, s=30)
    axes[0, 0].plot(GAPS, [np.mean(by_gap_conf[g]) for g in GAPS], "k-",
                    linewidth=2.2, label="mean")
    axes[0, 0].plot(GAPS, [np.median(by_gap_conf[g]) for g in GAPS], "k--",
                    linewidth=1.5, label="median")
    axes[0, 0].set_xlabel("frame gap")
    axes[0, 0].set_ylabel(f"frac pixels with RoMA conf > {CONF_THRESHOLD}")
    axes[0, 0].set_title("RoMA coverage per pair")
    axes[0, 0].set_xticks(GAPS)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    # --- Top-right: improvement ratio per gap
    by_gap_impr = {g: [r["improvement"] for r in records if r["gap"] == g] for g in GAPS}
    for g in GAPS:
        ys = by_gap_impr[g]
        xs = g + rng.uniform(-0.15, 0.15, size=len(ys))
        axes[0, 1].scatter(xs, ys, alpha=0.55, s=30)
    axes[0, 1].plot(GAPS, [np.mean(by_gap_impr[g]) for g in GAPS], "k-",
                    linewidth=2.2, label="mean")
    axes[0, 1].plot(GAPS, [np.median(by_gap_impr[g]) for g in GAPS], "k--",
                    linewidth=1.5, label="median")
    axes[0, 1].axhline(0.0, color="gray", linestyle="-", alpha=0.4)
    axes[0, 1].axhline(0.5, color="green", linestyle=":", alpha=0.6,
                       label="improvement >= 0.5")
    axes[0, 1].set_xlabel("frame gap")
    axes[0, 1].set_ylabel("improvement = 1 - warped_L1 / baseline_L1")
    axes[0, 1].set_title("How much does the warp beat identity?")
    axes[0, 1].set_xticks(GAPS)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    # --- Bottom-left: baseline vs warped scatter (y=x is identity; below the line = warp helps)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(GAPS)))
    for g, color in zip(GAPS, colors):
        rs = [r for r in records if r["gap"] == g]
        bx = [r["baseline_l1"] for r in rs]
        wy = [r["warped_l1"] for r in rs]
        axes[1, 0].scatter(bx, wy, alpha=0.65, s=35, color=color, label=f"gap=+{g}")
    lo = 0.0
    hi = max(max(r["baseline_l1"] for r in records),
             max(r["warped_l1"] for r in records)) * 1.05
    axes[1, 0].plot([lo, hi], [lo, hi], "k-", alpha=0.5, label="y=x (no improvement)")
    axes[1, 0].plot([lo, hi], [lo / 2, hi / 2], "k:", alpha=0.4,
                    label="y=x/2 (improvement=0.5)")
    axes[1, 0].set_xlabel("baseline_L1 = masked L1(x_A, x_B)")
    axes[1, 0].set_ylabel("warped_L1 = masked L1(warp(x_A), x_B)")
    axes[1, 0].set_title("Warped vs baseline L1 (below diagonal = warp helps)")
    axes[1, 0].set_xlim(lo, hi)
    axes[1, 0].set_ylim(lo, hi)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    # --- Bottom-right: fraction of pairs clearing selection thresholds vs gap
    impr_thresholds = [0.3, 0.5, 0.7]
    for thr in impr_thresholds:
        ys = [np.mean(np.array(by_gap_impr[g]) >= thr) for g in GAPS]
        axes[1, 1].plot(GAPS, ys, "o-", label=f"improvement >= {thr}")
        for g, y in zip(GAPS, ys):
            axes[1, 1].annotate(f"{y:.0%}", (g, y), textcoords="offset points",
                                xytext=(0, 6), ha="center", fontsize=7)
    axes[1, 1].set_xlabel("frame gap")
    axes[1, 1].set_ylabel("frac of pairs passing")
    axes[1, 1].set_title("Selectable pairs at various improvement bars")
    axes[1, 1].set_xticks(GAPS)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"RoMA warp quality on small frame gaps  "
                 f"({N_SEQUENCES} hydrant seqs x {ANCHORS_PER_SEQUENCE} anchors, "
                 f"RoMA={ROMA_SETTING}, thr={CONF_THRESHOLD})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    print("\n--- summary ---")
    for g in GAPS:
        rs = [r for r in records if r["gap"] == g]
        conf = np.array([r["frac_conf"] for r in rs])
        impr = np.array([r["improvement"] for r in rs])
        base = np.array([r["baseline_l1"] for r in rs])
        warp = np.array([r["warped_l1"] for r in rs])
        print(f"  gap=+{g}: conf mean={conf.mean():.2f}  "
              f"baseline_L1 mean={base.mean():.3f}  warped_L1 mean={warp.mean():.3f}  "
              f"improvement mean={impr.mean():+.2f}  median={np.median(impr):+.2f}  "
              f">=0.5: {(impr >= 0.5).mean():.0%}  >=0.3: {(impr >= 0.3).mean():.0%}")


BUCKET_EDGES = [-1.0, -0.30, -0.15, -0.05, 0.05, 1.0]
BUCKET_LABELS = ["impr < -0.30 (broken)",
                 "-0.30 <= impr < -0.15",
                 "-0.15 <= impr < -0.05",
                 "-0.05 <= impr < 0.05 (cliff)",
                 "impr >= 0.05 (warp helps)"]
N_PER_BUCKET = 4


def save_cliff_figure(records, out_path: Path):
    """Pick N_PER_BUCKET representative pairs per improvement bucket and lay them out
    so the user can scroll from broken -> good and find the cliff visually."""
    rng = np.random.default_rng(0)
    buckets = []
    for lo, hi, label in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:], BUCKET_LABELS):
        in_bucket = [r for r in records if lo <= r["improvement"] < hi]
        in_bucket.sort(key=lambda r: r["improvement"])  # worst-first within bucket
        if not in_bucket:
            buckets.append((label, []))
            continue
        if len(in_bucket) <= N_PER_BUCKET:
            picked = in_bucket
        else:
            # Spread picks evenly across the sorted bucket
            idxs = np.linspace(0, len(in_bucket) - 1, N_PER_BUCKET).round().astype(int)
            picked = [in_bucket[i] for i in idxs]
        buckets.append((label, picked))

    n_cols = 4  # target / warped+fill / diff / mask
    n_rows = sum(max(1, len(p)) for _, p in buckets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ["target x_B", "warp(x_A) (white = low conf)",
                  "x_A + red diff", f"conf mask (>{CONF_THRESHOLD})"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10)

    row = 0
    for label, picked in buckets:
        if not picked:
            # blank row for the bucket header
            for c in range(n_cols):
                axes[row, c].axis("off")
            axes[row, 0].text(0.0, 0.5, f"{label}  (no pairs)",
                              transform=axes[row, 0].transAxes,
                              fontsize=10, color="gray", va="center")
            row += 1
            continue
        for i, r in enumerate(picked):
            axes[row, 0].imshow(r["target"])
            axes[row, 1].imshow(r["warped"])
            axes[row, 2].imshow(r["diff"])
            axes[row, 3].imshow(r["mask"], cmap="gray", vmin=0, vmax=1)
            for c in range(n_cols):
                axes[row, c].set_xticks([])
                axes[row, c].set_yticks([])
            short = r["seq"] if len(r["seq"]) <= 22 else r["seq"][:19] + "..."
            tag = (f"{label.split(' (')[0] if i == 0 else ''}\n"
                   f"{short}  f{r['anchor']:03d}+{r['gap']}\n"
                   f"impr={r['improvement']:+.2f}  conf={r['frac_conf']:.2f}\n"
                   f"base={r['baseline_l1']:.3f} warp={r['warped_l1']:.3f}")
            axes[row, 0].set_ylabel(tag, fontsize=8, rotation=0, ha="right", va="center",
                                    labelpad=70)
            row += 1
    fig.suptitle("Sorted by improvement bucket - inspect where the cliff sits",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Loading annotations from {ANNOT}")
    annotations = load_annotations(str(ANNOT))

    # Filter to sequences with enough frames and existing on-disk images.
    candidates = []
    for seq_name, frames in annotations.items():
        if len(frames) < MIN_FRAMES_REQUIRED:
            continue
        sample_path = DATA_ROOT / frames[0]["filepath"]
        if sample_path.exists():
            candidates.append(seq_name)
    print(f"{len(candidates)} sequences with >= {MIN_FRAMES_REQUIRED} frames and images on disk")

    rng = np.random.default_rng(SEED)
    rng.shuffle(candidates)
    chosen_seqs = candidates[:N_SEQUENCES]
    print(f"Sampling {len(chosen_seqs)} sequences x {ANCHORS_PER_SEQUENCE} anchors each")

    print(f"Loading RoMaV2 ({ROMA_SETTING})")
    roma_model = load_roma_model(setting=ROMA_SETTING, device=str(device), compile=False)

    records: List[Tuple[str, int, int, float]] = []
    for sidx, seq_name in enumerate(chosen_seqs):
        frames = annotations[seq_name]
        anchors = pick_anchors(frames, ANCHORS_PER_SEQUENCE, rng)
        print(f"\n[{sidx+1}/{len(chosen_seqs)}] {seq_name} ({len(frames)} frames)  "
              f"anchors={anchors}")
        for aidx in anchors:
            try:
                per_gap = process_anchor(roma_model, device, frames, aidx, seq_name)
            except Exception as e:
                print(f"  anchor f{aidx}: FAILED ({e})")
                continue
            seq_tag = seq_name.replace("/", "_")
            out_path = OUT_DIR / f"{sidx:02d}_{seq_tag}_anchor{aidx:04d}.png"
            save_anchor_grid(per_gap, seq_name, aidx, out_path)
            for g in GAPS:
                d = per_gap[g]
                records.append({
                    "seq": seq_name, "anchor": aidx, "gap": g,
                    "frac_conf": d["frac_conf"],
                    "baseline_l1": d["baseline_l1"],
                    "warped_l1": d["warped_l1"],
                    "improvement": d["improvement"],
                    "target": d["target"],
                    "warped": d["warped"],
                    "diff": d["diff"],
                    "mask": d["mask"],
                })
                print(f"  anchor f{aidx} gap=+{g}: frac_conf={d['frac_conf']:.2f}  "
                      f"base={d['baseline_l1']:.3f}  warped={d['warped_l1']:.3f}  "
                      f"impr={d['improvement']:+.2f}")

    save_summary(records, OUT_DIR / "_summary.png")
    save_cliff_figure(records, OUT_DIR / "_cliff_buckets.png")


if __name__ == "__main__":
    main()
