"""
Multi-sequence version of the downsample-vs-warp comparison, with a small
2-frame camera movement.

For several CO3D hydrant sequences we take a pair (frame N, frame N+2) -- a
small baseline -- compute the RoMaV2 'fast' warp at full 256x256 resolution,
and compare two ways of getting a 32x32 (latent-sized) warped image:

  (A) warp @256 -> area-downsample to 32     (the correct reference)
  (B) downsample image + warp to 32 -> warp   (the latent-grid shortcut)

We show, per sequence: Frame A, Frame B target, Path A result, Path B result,
and the |A - B| residual at 32x32 with its MAE. A final line reports the mean
MAE across all sequences.

Output: outputs/scripts/downsample_warp_multiseq_demo.png

Run:
    conda activate cv
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/demos/downsample_warp_multiseq_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

ROMA_PATH = Path(__file__).resolve().parents[2] / "third_party" / "RoMA2" / "src"
if not ROMA_PATH.exists():
    ROMA_PATH = Path(__file__).resolve().parents[2] / "RoMA2" / "src"
if str(ROMA_PATH) not in sys.path:
    sys.path.insert(0, str(ROMA_PATH))

from romav2 import RoMaV2


IMAGE_RES = 256       # Warp-VAE training resolution
LATENT_RES = 32       # 256 / 8 -> SD-VAE latent grid
ROMA_SETTING = "fast"
FRAME_GAP = 2         # "movement of 2 frames": pair frame N with frame N+2
START_FRAME = 1       # 1-indexed frameNNNNNN.jpg

CATEGORY_ROOT = "/visinf/projects_students/dlcv2025_groupZ/co3d_full/hydrant"
N_SEQUENCES = 6
OUT_PATH = str(Path(__file__).resolve().parents[2] / "outputs/scripts/downsample_warp_multiseq_demo.png")


def list_sequences(root: str, n: int) -> list[Path]:
    seqs = sorted(p for p in Path(root).iterdir() if (p / "images").is_dir())
    return seqs[:n]


def frame_path(seq: Path, idx: int) -> Path:
    return seq / "images" / f"frame{idx:06d}.jpg"


def load_image(path: Path, res: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((res, res), Image.LANCZOS)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def warp_image(img: torch.Tensor, warp: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(img, warp, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


def downsample_image(img: torch.Tensor, res: int) -> torch.Tensor:
    return F.interpolate(img, size=(res, res), mode="area")


def downsample_warp(warp: torch.Tensor, res: int) -> torch.Tensor:
    w = warp.permute(0, 3, 1, 2)
    w = F.interpolate(w, size=(res, res), mode="bilinear", align_corners=False)
    return w.permute(0, 2, 3, 1)


def to_disp(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def run_pair(roma, img_a_pil, img_b_pil, device):
    """Return (warped_hi_to_latent, warped_small, resid_map, mae)."""
    with torch.no_grad():
        pred = roma.match(img_a_pil, img_b_pil)
    warp_ab = pred["warp_AB"]
    if warp_ab.shape[1] != IMAGE_RES:
        warp_ab = downsample_warp(warp_ab, IMAGE_RES)
    warp_ab = warp_ab.to(device)

    img_a = pil_to_tensor(img_a_pil).to(device)

    # Path A: warp hi-res, then downsample.
    warped_hi_to_latent = downsample_image(warp_image(img_a, warp_ab), LATENT_RES)

    # Path B: downsample image + warp first, then warp.
    img_a_small = downsample_image(img_a, LATENT_RES)
    warp_small = downsample_warp(warp_ab, LATENT_RES)
    warped_small = warp_image(img_a_small, warp_small)

    resid = (warped_hi_to_latent - warped_small).abs()
    resid_map = resid.mean(dim=1).squeeze(0).cpu().numpy()
    mae = resid.mean().item()
    return warped_hi_to_latent, warped_small, resid_map, mae


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = list_sequences(CATEGORY_ROOT, N_SEQUENCES)
    print(f"Using {len(seqs)} sequences, frame gap = {FRAME_GAP}")

    print(f"Loading RoMaV2 (setting={ROMA_SETTING}) on {device} ...")
    cfg = RoMaV2.Cfg(compile=False, setting=ROMA_SETTING)
    roma = RoMaV2(cfg=cfg).to(device).eval()

    up = lambda t: F.interpolate(t, size=(IMAGE_RES, IMAGE_RES), mode="nearest")

    n = len(seqs)
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.1 * n))
    if n == 1:
        axes = axes[None, :]

    maes = []
    for r, seq in enumerate(seqs):
        pa = frame_path(seq, START_FRAME)
        pb = frame_path(seq, START_FRAME + FRAME_GAP)
        if not pa.exists() or not pb.exists():
            print(f"  [skip] {seq.name}: missing frames")
            for c in range(5):
                axes[r, c].axis("off")
            continue

        img_a_pil = load_image(pa, IMAGE_RES)
        img_b_pil = load_image(pb, IMAGE_RES)
        warped_hi_lat, warped_small, resid_map, mae = run_pair(
            roma, img_a_pil, img_b_pil, device)
        maes.append(mae)
        print(f"  {seq.name}: MAE @ {LATENT_RES} = {mae:.4f}")

        axes[r, 0].imshow(np.asarray(img_a_pil))
        axes[r, 0].set_ylabel(seq.name, fontsize=9)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title(f"Frame N ({IMAGE_RES})")

        axes[r, 1].imshow(np.asarray(img_b_pil)); axes[r, 1].axis("off")
        if r == 0:
            axes[r, 1].set_title(f"Frame N+{FRAME_GAP} target")

        axes[r, 2].imshow(to_disp(up(warped_hi_lat))); axes[r, 2].axis("off")
        if r == 0:
            axes[r, 2].set_title(f"A: warp@256 -> down {LATENT_RES}")

        axes[r, 3].imshow(to_disp(up(warped_small))); axes[r, 3].axis("off")
        if r == 0:
            axes[r, 3].set_title(f"B: down {LATENT_RES} -> warp")

        vmax = max(0.05, float(resid_map.max()))
        im = axes[r, 4].imshow(resid_map, cmap="inferno", vmin=0, vmax=vmax)
        axes[r, 4].axis("off")
        axes[r, 4].set_title(f"|A - B|  MAE={mae:.3f}")
        fig.colorbar(im, ax=axes[r, 4], fraction=0.046)

    mean_mae = float(np.mean(maes)) if maes else float("nan")
    print(f"\nMean MAE across {len(maes)} sequences: {mean_mae:.4f}")
    fig.suptitle(
        f"Downsample-then-warp vs warp-then-downsample over {len(maes)} sequences  "
        f"(image {IMAGE_RES}, latent {LATENT_RES}, RoMA '{ROMA_SETTING}', "
        f"frame gap {FRAME_GAP})  |  mean MAE = {mean_mae:.3f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_PATH, dpi=110, bbox_inches="tight")
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
