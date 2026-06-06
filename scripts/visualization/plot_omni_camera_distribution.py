"""Plot the distribution of all OmniObject3D camera viewpoints in a single figure.

Each instance stores 24 cameras as (elevation, azimuth) in degrees on a sphere
of unknown but assumed-constant radius. We convert to unit-sphere positions and
show two complementary 2D views that share a coordinate space across all
instances:
  (a) equirectangular (azimuth vs elevation) — every camera as one dot,
  (b) top-down (X vs Y on the unit sphere) — colored by elevation.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CAM_ROOT = Path("/data/lab_moezkan/omni_obj/blender_renders_24_views/camera")
OUT_PATH = Path("/visinf/home/lab_mozkan/computer-vision-proj-lab/outputs/scripts/omni_camera_distribution.png")


def collect_cameras(root: Path):
    elev_all, azim_all = [], []
    for cat in sorted(os.listdir(root)):
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue
        for inst in sorted(os.listdir(cat_dir)):
            inst_dir = cat_dir / inst
            e = inst_dir / "elevation.npy"
            r = inst_dir / "rotation.npy"
            if not (e.exists() and r.exists()):
                continue
            elev_all.append(np.load(e))
            azim_all.append(np.load(r))
    return np.concatenate(elev_all), np.concatenate(azim_all)


def main():
    elev_deg, azim_deg = collect_cameras(CAM_ROOT)
    print(f"loaded {elev_deg.size} cameras from {CAM_ROOT}")

    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)

    x = np.cos(elev) * np.cos(azim)
    y = np.cos(elev) * np.sin(azim)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(azim_deg, elev_deg, s=1, alpha=0.05, c="steelblue", rasterized=True)
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("elevation (deg)")
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_title(f"Equirectangular  (N={elev_deg.size})")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(x, y, s=1, alpha=0.05, c=elev_deg, cmap="viridis", rasterized=True)
    ax.set_xlabel("X (unit sphere)")
    ax.set_ylabel("Y (unit sphere)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title("Top-down view (color = elevation)")
    circle = plt.Circle((0, 0), 1.0, fill=False, color="k", lw=0.5)
    ax.add_patch(circle)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("elevation (deg)")

    fig.suptitle("OmniObject3D 24-view camera distribution (all instances)")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
