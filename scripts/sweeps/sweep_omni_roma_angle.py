"""RoMA confidence vs camera-angle change on OmniObject3D.

OmniObject3D's 24 views are not an ordered trajectory; they're independent
samples on a sphere. For each sampled instance we:
  1. pick 2 random anchor views,
  2. compute great-circle angular distance from each anchor to every other view,
  3. for each target offset in TARGET_OFFSETS_DEG, pick the neighbor whose
     realized gap is closest to that target,
  4. run RoMA on (anchor, neighbor) and record mean overlap confidence
     together with the realized angular gap.

Output: scatter + binned mean(+std) of RoMA mean overlap confidence vs
angular distance, saved next to this script.
"""

import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path("/visinf/home/lab_mozkan/computer-vision-proj-lab")
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.roma_metrics import compute_roma_correspondences, load_roma_model

DATA_ROOT = Path("/data/lab_moezkan/omni_obj/blender_renders_24_views")
CAM_ROOT = DATA_ROOT / "camera"
IMG_ROOT = DATA_ROOT / "img"
OUT_PNG = REPO_ROOT / "outputs" / "scripts" / "omni_roma_vs_angle.png"
OUT_CSV = REPO_ROOT / "outputs" / "scripts" / "omni_roma_vs_angle.csv"

N_INSTANCES = 20
N_ANCHORS_PER_INSTANCE = 2
TARGET_OFFSETS_DEG = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 90]
SEED = 0


def list_all_instances(cam_root: Path):
    insts = []
    for cat in sorted(os.listdir(cam_root)):
        cat_dir = cam_root / cat
        if not cat_dir.is_dir():
            continue
        for inst in sorted(os.listdir(cat_dir)):
            if (cat_dir / inst / "elevation.npy").exists():
                insts.append((cat, inst))
    return insts


def cam_dirs(elev_deg: np.ndarray, azim_deg: np.ndarray) -> np.ndarray:
    e = np.deg2rad(elev_deg)
    a = np.deg2rad(azim_deg)
    x = np.cos(e) * np.cos(a)
    y = np.cos(e) * np.sin(a)
    z = np.sin(e)
    return np.stack([x, y, z], axis=1)  # (N,3) unit vectors


def angular_distances_deg(anchor_dir: np.ndarray, all_dirs: np.ndarray) -> np.ndarray:
    cos = np.clip(all_dirs @ anchor_dir, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos))


def load_image(inst: str, view_idx: int) -> Image.Image:
    return Image.open(IMG_ROOT / inst / f"{view_idx:03d}.png").convert("RGB")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    insts_all = list_all_instances(CAM_ROOT)
    print(f"total instances available: {len(insts_all)}")
    insts = random.sample(insts_all, N_INSTANCES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading RoMA on {device}...")
    roma = load_roma_model(setting="fast", device=device, compile=False)

    rows = []  # (instance, anchor_idx, target_deg, realized_deg, mean_conf, n_pairs_per_anchor)
    for ii, (cat, inst) in enumerate(insts):
        elev = np.load(CAM_ROOT / cat / inst / "elevation.npy")
        azim = np.load(CAM_ROOT / cat / inst / "rotation.npy")
        dirs = cam_dirs(elev, azim)
        n_views = len(elev)

        anchor_indices = random.sample(range(n_views), N_ANCHORS_PER_INSTANCE)
        for a_idx in anchor_indices:
            d = angular_distances_deg(dirs[a_idx], dirs)
            used = {a_idx}
            for target in TARGET_OFFSETS_DEG:
                order = np.argsort(np.abs(d - target))
                neighbor = None
                for cand in order:
                    if cand not in used:
                        neighbor = int(cand)
                        break
                if neighbor is None:
                    continue
                used.add(neighbor)
                realized = float(d[neighbor])

                img_a = load_image(inst, a_idx)
                img_b = load_image(inst, neighbor)
                out = compute_roma_correspondences(roma, img_a, img_b)
                # mean of both directions, averaged across pixels
                conf_ab = out["overlap_ab"].mean().item()
                conf_ba = out["overlap_ba"].mean().item()
                mean_conf = 0.5 * (conf_ab + conf_ba)

                rows.append((inst, a_idx, target, realized, mean_conf))
                print(
                    f"[{ii+1:02d}/{N_INSTANCES}] {inst} anchor={a_idx:2d} "
                    f"target={target:>3d}° realized={realized:5.1f}° conf={mean_conf:.3f}"
                )

    # save CSV
    with open(OUT_CSV, "w") as f:
        f.write("instance,anchor,target_deg,realized_deg,mean_conf\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.6f}\n")
    print(f"wrote {OUT_CSV}  ({len(rows)} pairs)")

    realized = np.array([r[3] for r in rows])
    conf = np.array([r[4] for r in rows])

    # bin by target offset (groups pairs with same intended gap)
    targets = np.array([r[2] for r in rows])
    uniq = sorted(set(targets.tolist()))
    means, stds, centers, ns = [], [], [], []
    for t in uniq:
        sel = targets == t
        means.append(conf[sel].mean())
        stds.append(conf[sel].std())
        centers.append(realized[sel].mean())
        ns.append(int(sel.sum()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(realized, conf, s=12, alpha=0.4, c="steelblue", label="per pair")
    ax.errorbar(
        centers, means, yerr=stds, fmt="o-", color="crimson",
        capsize=3, lw=1.5, ms=6, label="binned mean ± std",
    )
    for x, y, n in zip(centers, means, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="crimson")
    ax.set_xlabel("angular distance between camera directions (deg)")
    ax.set_ylabel("RoMA mean overlap confidence")
    ax.set_title(
        f"OmniObject3D — RoMA confidence vs camera-angle change\n"
        f"{N_INSTANCES} instances × {N_ANCHORS_PER_INSTANCE} anchors × "
        f"{len(TARGET_OFFSETS_DEG)} offsets = {len(rows)} pairs"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
