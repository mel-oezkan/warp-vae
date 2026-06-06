"""How often does RoMA produce a trainable warp on OmniObject3D at various
camera-angle offsets?

Omni analogue of scripts/sweeps/vae_roma_multiseq_gaps.py. OmniObject3D's 24 views are
independent samples on the sphere (no temporal ordering), so the "gap" axis
becomes the great-circle angle between the anchor camera direction and the
target's. For each (anchor) and each TARGET_ANGLES bin, we pick the unused
view whose realized angle is closest to the target.

Layout matches the RoMA script:
  N_INSTANCES random instances x ANCHORS_PER_INSTANCE random anchors x
  4 angle bins; one 4-row figure per anchor:
    row 1: target x_B
    row 2: warp(x_A) in pixel space
    row 3: x_A + red overlay where |warp(x_A) - x_B| is large
    row 4: RoMA confidence mask (binary, threshold CONF_THRESHOLD)
Plus a _summary figure (coverage / improvement / scatter / threshold curves) and
a _cliff_buckets figure (broken -> good).

Run:
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/sweeps/vae_roma_omni_anglebins.py
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.analysis.roma_metrics import load_roma_model

REPO = Path("/visinf/home/lab_mozkan/computer-vision-proj-lab")
DATA_ROOT = Path("/data/lab_moezkan/omni_obj/blender_renders_24_views")
CAM_ROOT = DATA_ROOT / "camera"
IMG_ROOT = DATA_ROOT / "img"
OUT_DIR = REPO / "outputs/scripts/vae_roma_omni_sweep/multiinst_closest"

IMAGE_SIZE = 256
# Omni's 24 views are sparse on the sphere (median nearest-neighbor ~10°), so
# instead of fixed target angles we take the k closest views to each anchor.
# NEAREST_RANKS = [1, 2, 3, 4] means "closest, 2nd closest, ..." -- the truest
# analogue of "one or two frames apart" for an unordered view set.
NEAREST_RANKS = [1, 2, 3, 4]
CONF_THRESHOLD = 0.8
ROMA_SETTING = "fast"
N_INSTANCES = 20
ANCHORS_PER_INSTANCE = 3
SEED = 0


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
    return valid[0, ..., 0]


# ---------- omni camera helpers ----------

def list_all_instances(cam_root: Path) -> List[Tuple[str, str]]:
    insts = []
    for cat in sorted(os.listdir(cam_root)):
        cat_dir = cam_root / cat
        if not cat_dir.is_dir():
            continue
        for inst in sorted(os.listdir(cat_dir)):
            if (cat_dir / inst / "elevation.npy").exists():
                insts.append((cat, inst))
    return insts


def camera_dirs(elev_deg: np.ndarray, azim_deg: np.ndarray) -> np.ndarray:
    e = np.deg2rad(elev_deg)
    a = np.deg2rad(azim_deg)
    return np.stack(
        [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)],
        axis=1,
    )  # (N, 3)


def angles_to_anchor_deg(dirs: np.ndarray, anchor_idx: int) -> np.ndarray:
    cos = np.clip(dirs @ dirs[anchor_idx], -1.0, 1.0)
    return np.rad2deg(np.arccos(cos))


def pick_anchors(n_views: int, n_anchors: int, rng) -> List[int]:
    return sorted(rng.choice(n_views, size=min(n_anchors, n_views), replace=False).tolist())


def pick_neighbors_for_anchor(dirs: np.ndarray, anchor_idx: int,
                              ranks: List[int]) -> Dict[int, Tuple[int, float]]:
    """Pick the k-th nearest views to the anchor (k in `ranks`, 1-indexed).

    Returns rank -> (view_idx, realized_deg). This keeps angle changes as small
    as the (sparse) view set allows -- the closest neighbors to each anchor.
    """
    d = angles_to_anchor_deg(dirs, anchor_idx)
    order = [int(i) for i in np.argsort(d) if int(i) != anchor_idx]
    chosen: Dict[int, Tuple[int, float]] = {}
    for r in ranks:
        if r - 1 < len(order):
            nbr = order[r - 1]
            chosen[r] = (nbr, float(d[nbr]))
    return chosen


# ---------- sweep ----------

@torch.no_grad()
def process_anchor(roma_model, device, inst: str, dirs: np.ndarray,
                   anchor_idx: int) -> Dict[int, dict]:
    anchor_path = IMG_ROOT / inst / f"{anchor_idx:03d}.png"
    x_a = load_image_tensor(anchor_path).to(device)
    pil_a = load_image_pil(anchor_path)
    picks = pick_neighbors_for_anchor(dirs, anchor_idx, NEAREST_RANKS)

    per_target: Dict[int, dict] = {}
    for t in NEAREST_RANKS:
        if t not in picks:
            continue
        nbr_idx, realized = picks[t]
        target_path = IMG_ROOT / inst / f"{nbr_idx:03d}.png"
        x_b = load_image_tensor(target_path).to(device)
        pil_b = load_image_pil(target_path)

        warp_img, conf_img = roma_warp(roma_model, pil_a, pil_b, device)
        mask = confidence_mask(conf_img, warp_img)
        warped_pixels = F.grid_sample(x_a.unsqueeze(0), warp_img,
                                      mode="bilinear", padding_mode="zeros",
                                      align_corners=False)[0]

        a01 = (x_a.clamp(-1, 1) + 1) / 2
        b01 = (x_b.clamp(-1, 1) + 1) / 2
        w01 = (warped_pixels.clamp(-1, 1) + 1) / 2
        denom = mask.sum().clamp_min(1.0) * a01.shape[0]
        baseline_l1 = float(((a01 - b01).abs() * mask).sum() / denom)
        warped_l1 = float(((w01 - b01).abs() * mask).sum() / denom)
        improvement = 1.0 - warped_l1 / max(baseline_l1, 1e-6)

        per_target[t] = {
            "neighbor_idx": nbr_idx,
            "realized_deg": realized,
            "target": to_display(x_b),
            "warped": warped_with_white_bg(warped_pixels, mask),
            "diff": red_diff_overlay(x_a, warped_pixels, x_b, mask),
            "mask": mask.detach().cpu().numpy(),
            "frac_conf": float(mask.mean()),
            "baseline_l1": baseline_l1,
            "warped_l1": warped_l1,
            "improvement": improvement,
        }
    return per_target


def save_anchor_grid(per_target: Dict[int, dict], inst: str, anchor_idx: int,
                     out_path: Path):
    n = len(NEAREST_RANKS)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8.5))
    row_titles = ["target x_B",
                  "warp(x_A) [pixel-space]",
                  "x_A + red |warp(x_A) - x_B|",
                  f"RoMA conf mask (>{CONF_THRESHOLD})"]
    rows_key = ["target", "warped", "diff", "mask"]
    for r, key in enumerate(rows_key):
        for c, t in enumerate(NEAREST_RANKS):
            ax = axes[r, c]
            if t not in per_target:
                ax.axis("off"); continue
            im = per_target[t][key]
            if key == "mask":
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                d = per_target[t]
                ax.set_title(
                    f"#{t} nearest  Δ={d['realized_deg']:.1f}°\n"
                    f"frac_conf={d['frac_conf']:.2f}\n"
                    f"base L1={d['baseline_l1']:.3f}  warp L1={d['warped_l1']:.3f}\n"
                    f"impr={d['improvement']:+.2f}",
                    fontsize=8)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=10)
    short = inst if len(inst) <= 30 else inst[:27] + "..."
    fig.suptitle(f"{short}   anchor=v{anchor_idx:02d}   "
                 f"(RoMA={ROMA_SETTING}, thr={CONF_THRESHOLD})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_summary(records: List[dict], out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    rng = np.random.default_rng(0)

    by_t_conf = {t: [r["frac_conf"] for r in records if r["rank"] == t] for t in NEAREST_RANKS}
    by_t_impr = {t: [r["improvement"] for r in records if r["rank"] == t] for t in NEAREST_RANKS}
    by_t_real = {t: [r["realized_deg"] for r in records if r["rank"] == t] for t in NEAREST_RANKS}
    # x position for each rank = mean realized angle, so the axis reads in degrees
    x_of = {t: float(np.mean(by_t_real[t])) if by_t_real[t] else t for t in NEAREST_RANKS}
    xs_ranks = [x_of[t] for t in NEAREST_RANKS]
    jitter = 0.6

    for t in NEAREST_RANKS:
        ys = by_t_conf[t]
        xs = x_of[t] + rng.uniform(-jitter, jitter, size=len(ys))
        axes[0, 0].scatter(xs, ys, alpha=0.55, s=30, label=f"#{t} nearest")
    axes[0, 0].plot(xs_ranks, [np.mean(by_t_conf[t]) for t in NEAREST_RANKS], "k-",
                    linewidth=2.2, label="mean")
    axes[0, 0].plot(xs_ranks, [np.median(by_t_conf[t]) for t in NEAREST_RANKS], "k--",
                    linewidth=1.5, label="median")
    axes[0, 0].set_xlabel("realized camera angle Δ (deg)  [#1..#4 nearest view]")
    axes[0, 0].set_ylabel(f"frac pixels with RoMA conf > {CONF_THRESHOLD}")
    axes[0, 0].set_title("RoMA coverage per pair")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    for t in NEAREST_RANKS:
        ys = by_t_impr[t]
        xs = x_of[t] + rng.uniform(-jitter, jitter, size=len(ys))
        axes[0, 1].scatter(xs, ys, alpha=0.55, s=30)
    axes[0, 1].plot(xs_ranks, [np.mean(by_t_impr[t]) for t in NEAREST_RANKS], "k-",
                    linewidth=2.2, label="mean")
    axes[0, 1].plot(xs_ranks, [np.median(by_t_impr[t]) for t in NEAREST_RANKS], "k--",
                    linewidth=1.5, label="median")
    axes[0, 1].axhline(0.0, color="gray", linestyle="-", alpha=0.4)
    axes[0, 1].axhline(0.5, color="green", linestyle=":", alpha=0.6,
                       label="improvement >= 0.5")
    axes[0, 1].set_xlabel("realized camera angle Δ (deg)  [#1..#4 nearest view]")
    axes[0, 1].set_ylabel("improvement = 1 - warped_L1 / baseline_L1")
    axes[0, 1].set_title("How much does the warp beat identity?")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(NEAREST_RANKS)))
    for t, color in zip(NEAREST_RANKS, colors):
        rs = [r for r in records if r["rank"] == t]
        bx = [r["baseline_l1"] for r in rs]
        wy = [r["warped_l1"] for r in rs]
        axes[1, 0].scatter(bx, wy, alpha=0.65, s=35, color=color,
                           label=f"#{t} (~{x_of[t]:.0f}°)")
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

    impr_thresholds = [0.3, 0.5, 0.7]
    for thr in impr_thresholds:
        ys = [np.mean(np.array(by_t_impr[t]) >= thr) for t in NEAREST_RANKS]
        axes[1, 1].plot(xs_ranks, ys, "o-", label=f"improvement >= {thr}")
        for x, y in zip(xs_ranks, ys):
            axes[1, 1].annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                                xytext=(0, 6), ha="center", fontsize=7)
    axes[1, 1].set_xlabel("realized camera angle Δ (deg)  [#1..#4 nearest view]")
    axes[1, 1].set_ylabel("frac of pairs passing")
    axes[1, 1].set_title("Selectable pairs at various improvement bars")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"RoMA warp quality on OmniObject3D, closest views  "
                 f"({N_INSTANCES} instances x {ANCHORS_PER_INSTANCE} anchors, "
                 f"RoMA={ROMA_SETTING}, thr={CONF_THRESHOLD})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    print("\n--- summary ---")
    for t in NEAREST_RANKS:
        rs = [r for r in records if r["rank"] == t]
        if not rs:
            print(f"  #{t} nearest: no pairs"); continue
        conf = np.array([r["frac_conf"] for r in rs])
        impr = np.array([r["improvement"] for r in rs])
        base = np.array([r["baseline_l1"] for r in rs])
        warp = np.array([r["warped_l1"] for r in rs])
        real = np.array([r["realized_deg"] for r in rs])
        print(f"  #{t} nearest  (Δ {real.mean():.1f}±{real.std():.1f}°): "
              f"conf mean={conf.mean():.2f}  "
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
                  "x_A + red diff", f"conf mask (>{CONF_THRESHOLD})"]
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
            short = r["inst"] if len(r["inst"]) <= 22 else r["inst"][:19] + "..."
            tag = (f"{label.split(' (')[0] if i == 0 else ''}\n"
                   f"{short}  v{r['anchor']:02d}->v{r['neighbor_idx']:02d}\n"
                   f"#{r['rank']} nearest  Δ={r['realized_deg']:.1f}°\n"
                   f"impr={r['improvement']:+.2f}  conf={r['frac_conf']:.2f}\n"
                   f"base={r['baseline_l1']:.3f} warp={r['warped_l1']:.3f}")
            axes[row, 0].set_ylabel(tag, fontsize=8, rotation=0, ha="right", va="center",
                                    labelpad=80)
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

    insts_all = list_all_instances(CAM_ROOT)
    print(f"{len(insts_all)} omni instances available")
    rng = np.random.default_rng(SEED)
    chosen_idx = rng.choice(len(insts_all), size=min(N_INSTANCES, len(insts_all)),
                            replace=False)
    chosen = [insts_all[i] for i in chosen_idx]
    print(f"Sampling {len(chosen)} instances x {ANCHORS_PER_INSTANCE} anchors each, "
          f"nearest-view ranks={NEAREST_RANKS}")

    print(f"Loading RoMaV2 ({ROMA_SETTING})")
    roma_model = load_roma_model(setting=ROMA_SETTING, device=str(device), compile=False)

    records: List[dict] = []
    for sidx, (cat, inst) in enumerate(chosen):
        elev = np.load(CAM_ROOT / cat / inst / "elevation.npy")
        azim = np.load(CAM_ROOT / cat / inst / "rotation.npy")
        dirs = camera_dirs(elev, azim)
        anchors = pick_anchors(len(elev), ANCHORS_PER_INSTANCE, rng)
        print(f"\n[{sidx+1}/{len(chosen)}] {cat}/{inst}  anchors={anchors}")
        for aidx in anchors:
            try:
                per_target = process_anchor(roma_model, device, inst, dirs, aidx)
            except Exception as e:
                print(f"  anchor v{aidx:02d}: FAILED ({e})")
                continue
            out_path = OUT_DIR / f"{sidx:02d}_{inst}_anchor{aidx:02d}.png"
            save_anchor_grid(per_target, inst, aidx, out_path)
            for t, d in per_target.items():
                records.append({
                    "inst": inst, "anchor": aidx,
                    "rank": t,
                    "realized_deg": d["realized_deg"],
                    "neighbor_idx": d["neighbor_idx"],
                    "frac_conf": d["frac_conf"],
                    "baseline_l1": d["baseline_l1"],
                    "warped_l1": d["warped_l1"],
                    "improvement": d["improvement"],
                    "target": d["target"],
                    "warped": d["warped"],
                    "diff": d["diff"],
                    "mask": d["mask"],
                })
                print(f"  v{aidx:02d}->v{d['neighbor_idx']:02d} #{t} nearest "
                      f"Δ={d['realized_deg']:.1f}°  "
                      f"frac_conf={d['frac_conf']:.2f}  "
                      f"base={d['baseline_l1']:.3f}  warped={d['warped_l1']:.3f}  "
                      f"impr={d['improvement']:+.2f}")

    save_summary(records, OUT_DIR / "_summary.png")
    save_cliff_figure(records, OUT_DIR / "_cliff_buckets.png")


if __name__ == "__main__":
    main()
