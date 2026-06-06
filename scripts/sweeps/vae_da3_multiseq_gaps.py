"""How often does a DA3-depth warp produce a trainable signal at small frame gaps (2-5)?

DA3 analogue of scripts/sweeps/vae_roma_multiseq_gaps.py. The only thing that changes
is the warp source: instead of RoMA's learned warp_AB, we predict DepthAnything-V3
depth on each frame, affinely align it to the CO3D GT depth, and feed it through
the existing compute_depth_warp geometric pipeline (so the warp comes from the
known camera pose + predicted depth, exactly like precompute_da3_warps.py).

Layout matches the RoMA script: 10 (default 20) random hydrant sequences x 3
random anchors x gaps {2,3,4,5}; one 4-row figure per anchor:
  row 1: target x_B
  row 2: warp(x_A) in pixel space
  row 3: x_A + red overlay where |warp(x_A) - x_B| is large
  row 4: confidence mask
Plus a _summary figure (coverage / improvement / scatter / threshold curves) and
a _cliff_buckets figure (broken -> good).

No cropping: frames are resized to 256x256 directly. The depth warp is computed
in the full-image coordinate frame (crop_bbox=None), matching what the RoMA script
showed.

Run:
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/sweeps/vae_da3_multiseq_gaps.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from depth_anything_3.api import DepthAnything3

REPO = Path("/visinf/home/lab_mozkan/computer-vision-proj-lab")
sys.path.insert(0, str(REPO / "scripts"))

from warps.precompute_depth_warps import (  # noqa: E402
    build_intrinsic_matrix,
    compute_depth_warp,
    load_annotations,
    load_co3d_depth,
)
from warps.precompute_da3_warps import aligned_da3_depth, predict_da3_depth  # noqa: E402

ANNOT = REPO / "data/co3d_annotations/hydrant_train_50seq_depth.jgz"
DATA_ROOT = Path("/visinf/projects_students/dlcv2025_groupZ/co3d_full")
OUT_DIR = REPO / "outputs/scripts/vae_da3_sweep/multiseq_gap2to5"

IMAGE_SIZE = 256
GAPS = [2, 3, 4, 5]
DEPTH_CONSISTENCY_THRESHOLD = 0.1
DA3_MODEL = "depth-anything/DA3-BASE"
PROCESS_RES = 504
N_SEQUENCES = 20
ANCHORS_PER_SEQUENCE = 3
SEED = 0
MIN_FRAMES_REQUIRED = max(GAPS) + 10


# ---------- I/O ----------

def load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return tfm(img)


def to_display(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1, 1)
    return ((x + 1) / 2).permute(1, 2, 0).numpy()


def red_diff_overlay(src: torch.Tensor, warped: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor, strength: float = 2.0) -> np.ndarray:
    src01 = ((src.detach().cpu().clamp(-1, 1) + 1) / 2)
    w01 = ((warped.detach().cpu().clamp(-1, 1) + 1) / 2)
    t01 = ((target.detach().cpu().clamp(-1, 1) + 1) / 2)
    m = mask.detach().cpu()
    diff = (w01 - t01).abs().mean(dim=0)
    alpha = (diff * strength).clamp(0, 1) * m
    gray = src01.mean(dim=0, keepdim=True).repeat(3, 1, 1)
    red = torch.zeros_like(gray); red[0] = 1.0
    fg = gray * (1 - alpha) + red * alpha
    white = torch.ones_like(fg)
    out = fg * m.unsqueeze(0) + white * (1 - m.unsqueeze(0))
    return out.permute(1, 2, 0).numpy()


def warped_with_white_bg(warped: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    w01 = ((warped.detach().cpu().clamp(-1, 1) + 1) / 2)
    m = mask.detach().cpu().unsqueeze(0)
    out = w01 * m + torch.ones_like(w01) * (1 - m)
    return out.permute(1, 2, 0).numpy()


# ---------- DA3-warp ----------

def load_frame_geom(frame, root):
    """Predict DA3 depth, align to GT, build intrinsics. No bbox crop here."""
    img_path = root / frame["filepath"]
    depth_path = root / frame["depth_path"]
    rgb = np.array(Image.open(img_path).convert("RGB"))
    gt = load_co3d_depth(str(depth_path), frame.get("depth_scale_adjustment", 1.0))
    return dict(
        img_path=img_path,
        rgb=rgb,
        gt=gt,
        image_size=tuple(frame["image_size"]),
        K=build_intrinsic_matrix(
            np.array(frame["focal_length"]),
            np.array(frame["principal_point"]),
            tuple(frame["image_size"]),
        ),
        R=np.array(frame["R"]),
        T=np.array(frame["T"]),
    )


def da3_align(model, frame_geom):
    da3 = predict_da3_depth(model, frame_geom["rgb"], process_res=PROCESS_RES)
    aligned, _, _ = aligned_da3_depth(da3, frame_geom["gt"])
    return aligned


def da3_warp_pair(A, B, A_aligned, B_aligned):
    """Compute the geometric warp from A->B coords using DA3-aligned depth.

    Returns (warp_BA, conf_BA): a grid such that sampling x_A with warp_BA
    produces an image registered to x_B (i.e. grid_sample(x_A, warp_BA) ~= x_B).
    """
    valid_a = (A_aligned > 0) & np.isfinite(A_aligned)
    valid_b = (B_aligned > 0) & np.isfinite(B_aligned)
    warp_ba, conf_ba = compute_depth_warp(
        B_aligned, valid_b, B["R"], B["T"], B["K"],
        A_aligned, valid_a, A["R"], A["T"], A["K"],
        warp_resolution=IMAGE_SIZE,
        image_size_a=B["image_size"], image_size_b=A["image_size"],
        depth_consistency_threshold=DEPTH_CONSISTENCY_THRESHOLD,
        crop_bbox_a=None, crop_bbox_b=None,
    )
    return warp_ba, conf_ba


def confidence_mask(conf_hw: torch.Tensor, warp_hw2: torch.Tensor) -> torch.Tensor:
    """compute_depth_warp already returns a ~binary depth-consistency mask;
    here we just AND with in-bounds (warp coords in [-1, 1])."""
    in_bounds = (warp_hw2.abs() <= 1.0).all(dim=-1).float()
    return ((conf_hw > 0).float() * in_bounds)


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
def process_anchor(model, device, seq_frames, anchor_idx, seq_name):
    anchor_frame = seq_frames[anchor_idx]
    A_geom = load_frame_geom(anchor_frame, DATA_ROOT)
    A_aligned = da3_align(model, A_geom)
    x_a = load_image_tensor(A_geom["img_path"]).to(device)

    per_gap = {}
    for g in GAPS:
        tgt_frame = seq_frames[anchor_idx + g]
        B_geom = load_frame_geom(tgt_frame, DATA_ROOT)
        B_aligned = da3_align(model, B_geom)
        x_b = load_image_tensor(B_geom["img_path"]).to(device)

        warp_ba, conf_ba = da3_warp_pair(A_geom, B_geom, A_aligned, B_aligned)
        warp_ba = warp_ba.to(device).float()
        conf_ba = conf_ba.to(device).float()
        mask = confidence_mask(conf_ba, warp_ba)

        # grid_sample expects (N, H, W, 2); compute_depth_warp returns (H, W, 2)
        warped_pixels = F.grid_sample(
            x_a.unsqueeze(0),
            warp_ba.unsqueeze(0),
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )[0]

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
                  "DA3 depth-consistency mask"]
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
                 f"(DA3={DA3_MODEL}, depth_thr={DEPTH_CONSISTENCY_THRESHOLD})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_summary(records: List[dict], out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

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
    axes[0, 0].set_ylabel("frac pixels with valid DA3 warp")
    axes[0, 0].set_title("DA3 coverage per pair")
    axes[0, 0].set_xticks(GAPS)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

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

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(GAPS)))
    for g, color in zip(GAPS, colors):
        rs = [r for r in records if r["gap"] == g]
        bx = [r["baseline_l1"] for r in rs]
        wy = [r["warped_l1"] for r in rs]
        axes[1, 0].scatter(bx, wy, alpha=0.65, s=35, color=color, label=f"gap=+{g}")
    lo = 0.0
    hi = max(max(r["baseline_l1"] for r in records),
             max(r["warped_l1"] for r in records)) * 1.05
    axes[1, 0].plot([lo, hi], [lo, hi], "k-", alpha=0.5, label="y=x")
    axes[1, 0].plot([lo, hi], [lo / 2, hi / 2], "k:", alpha=0.4,
                    label="y=x/2 (impr=0.5)")
    axes[1, 0].set_xlabel("baseline_L1 = masked L1(x_A, x_B)")
    axes[1, 0].set_ylabel("warped_L1 = masked L1(warp(x_A), x_B)")
    axes[1, 0].set_title("Warped vs baseline L1 (below diagonal = warp helps)")
    axes[1, 0].set_xlim(lo, hi)
    axes[1, 0].set_ylim(lo, hi)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

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

    fig.suptitle(f"DA3 warp quality on small frame gaps  "
                 f"({N_SEQUENCES} hydrant seqs x {ANCHORS_PER_SEQUENCE} anchors, "
                 f"depth_thr={DEPTH_CONSISTENCY_THRESHOLD})", fontsize=12)
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
    buckets = []
    for lo, hi, label in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:], BUCKET_LABELS):
        in_bucket = [r for r in records if lo <= r["improvement"] < hi]
        in_bucket.sort(key=lambda r: r["improvement"])
        if not in_bucket:
            buckets.append((label, []))
            continue
        if len(in_bucket) <= N_PER_BUCKET:
            picked = in_bucket
        else:
            idxs = np.linspace(0, len(in_bucket) - 1, N_PER_BUCKET).round().astype(int)
            picked = [in_bucket[i] for i in idxs]
        buckets.append((label, picked))

    n_cols = 4
    n_rows = sum(max(1, len(p)) for _, p in buckets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = ["target x_B", "warp(x_A) (white = low conf)",
                  "x_A + red diff", "DA3 mask"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10)

    row = 0
    for label, picked in buckets:
        if not picked:
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

    print(f"Loading {DA3_MODEL} onto {device}")
    model = DepthAnything3.from_pretrained(DA3_MODEL).to(device=device)

    records: List[Tuple[str, int, int, float]] = []
    for sidx, seq_name in enumerate(chosen_seqs):
        frames = annotations[seq_name]
        anchors = pick_anchors(frames, ANCHORS_PER_SEQUENCE, rng)
        print(f"\n[{sidx+1}/{len(chosen_seqs)}] {seq_name} ({len(frames)} frames)  "
              f"anchors={anchors}")
        for aidx in anchors:
            try:
                per_gap = process_anchor(model, device, frames, aidx, seq_name)
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
