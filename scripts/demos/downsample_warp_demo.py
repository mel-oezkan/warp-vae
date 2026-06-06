"""
Demo: what happens if we downsample the image AND the warp map first, then warp
the downsampled image — compared with warping at full resolution and only then
downsampling to the latent grid?

Setup
  - Image resolution: 256x256 (the resolution the Warp-VAE configs train at).
  - Latent resolution: 32x32 (256 / 8, the SD-VAE downsampling factor).
  - Warp source: RoMaV2 with setting="fast".

We compare two ways of getting a 32x32 warped image:

  (A) HI-RES then downsample (the "correct" reference)
        warp 256x256 image with the 256x256 warp via grid_sample,
        then area-downsample the 256x256 result to 32x32.

  (B) DOWNSAMPLE then warp (the naive shortcut)
        downsample image -> 32x32, downsample warp map -> 32x32,
        then grid_sample the 32x32 image with the 32x32 warp.

Path (B) is what you'd do if you tried to operate purely on a latent-sized grid.
The residual |A - B| shows the error this shortcut introduces — chiefly from
aliasing the high-frequency warp field down to a 32x32 grid before resampling.

Output: outputs/scripts/downsample_warp_demo.png

Run:
    conda activate cv
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/demos/downsample_warp_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# RoMA V2 lives in third_party/RoMA2/src (same path the analysis utils use).
ROMA_PATH = Path(__file__).resolve().parents[2] / "third_party" / "RoMA2" / "src"
if not ROMA_PATH.exists():
    ROMA_PATH = Path(__file__).resolve().parents[2] / "RoMA2" / "src"
if str(ROMA_PATH) not in sys.path:
    sys.path.insert(0, str(ROMA_PATH))

from romav2 import RoMaV2


IMAGE_RES = 256       # what the Warp-VAE configs train at
LATENT_RES = 32       # 256 / 8 -> the SD-VAE latent grid
ROMA_SETTING = "fast"

SEQ = "/visinf/projects_students/dlcv2025_groupZ/co3d_full/hydrant/106_12648_23157/images"
FRAME_A = "frame000001.jpg"
FRAME_B = "frame000020.jpg"
OUT_PATH = str(Path(__file__).resolve().parents[2] / "outputs/scripts/downsample_warp_demo.png")


def load_image(path: str, res: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((res, res), Image.LANCZOS)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> (1, 3, H, W) float in [0, 1]."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def warp_image(img: torch.Tensor, warp: torch.Tensor) -> torch.Tensor:
    """Backward-warp img (1,3,H,W) with warp (1,H,W,2) in normalized [-1,1]."""
    return F.grid_sample(
        img, warp, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def downsample_image(img: torch.Tensor, res: int) -> torch.Tensor:
    """Anti-aliased area downsampling for the image (1,3,H,W)."""
    return F.interpolate(img, size=(res, res), mode="area")


def downsample_warp(warp: torch.Tensor, res: int) -> torch.Tensor:
    """Downsample warp field (1,H,W,2) -> (1,res,res,2).

    Warp coords are normalized to [-1, 1], so they stay valid at any grid size;
    we just resample the field bilinearly. This mirrors warp_to_latent_warp() in
    src/analysis/roma_metrics.py.
    """
    w = warp.permute(0, 3, 1, 2)  # (1,2,H,W)
    w = F.interpolate(w, size=(res, res), mode="bilinear", align_corners=False)
    return w.permute(0, 2, 3, 1)  # (1,res,res,2)


def to_disp(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) in [0,1] -> (H,W,3) numpy for imshow."""
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_a_pil = load_image(f"{SEQ}/{FRAME_A}", IMAGE_RES)
    img_b_pil = load_image(f"{SEQ}/{FRAME_B}", IMAGE_RES)

    print(f"Loading RoMaV2 (setting={ROMA_SETTING}) on {device} ...")
    cfg = RoMaV2.Cfg(compile=False, setting=ROMA_SETTING)
    roma = RoMaV2(cfg=cfg).to(device).eval()

    with torch.no_grad():
        pred = roma.match(img_a_pil, img_b_pil)
    warp_ab = pred["warp_AB"]                       # (1, H, W, 2), maps B-grid -> A
    if warp_ab.shape[1] != IMAGE_RES:               # resize warp to image res if needed
        warp_ab = downsample_warp(warp_ab, IMAGE_RES)
    warp_ab = warp_ab.to(device)
    print(f"warp_AB shape: {tuple(warp_ab.shape)}")

    img_a = pil_to_tensor(img_a_pil).to(device)     # we warp A into B's frame

    # --- Path A: warp at hi-res, THEN downsample to latent grid -------------
    warped_hi = warp_image(img_a, warp_ab)                   # 256x256
    warped_hi_to_latent = downsample_image(warped_hi, LATENT_RES)  # -> 32x32

    # --- Path B: downsample image + warp FIRST, then warp -------------------
    img_a_small = downsample_image(img_a, LATENT_RES)        # 32x32
    warp_small = downsample_warp(warp_ab, LATENT_RES)        # 32x32
    warped_small = warp_image(img_a_small, warp_small)       # 32x32 directly

    # --- Residual between the two 32x32 results -----------------------------
    resid = (warped_hi_to_latent - warped_small).abs()
    resid_map = resid.mean(dim=1).squeeze(0).cpu().numpy()   # (32,32)
    mae = resid.mean().item()
    print(f"Mean abs diff (Path A vs Path B) at {LATENT_RES}x{LATENT_RES}: {mae:.4f}")

    # Upsample both 32x32 results back to 256 just so the difference is visible.
    up = lambda t: F.interpolate(t, size=(IMAGE_RES, IMAGE_RES), mode="nearest")

    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(to_disp(img_a)); axes[0, 0].set_title(f"Frame A ({IMAGE_RES})")
    axes[0, 1].imshow(to_disp(pil_to_tensor(img_b_pil))); axes[0, 1].set_title(f"Frame B target ({IMAGE_RES})")
    axes[0, 2].imshow(to_disp(warped_hi)); axes[0, 2].set_title("Warp(A) @256  (hi-res)")
    axes[0, 3].imshow(to_disp(up(warped_hi_to_latent)))
    axes[0, 3].set_title(f"A: warp@256 -> down to {LATENT_RES}")

    axes[1, 0].imshow(to_disp(up(img_a_small))); axes[1, 0].set_title(f"A down to {LATENT_RES}")
    axes[1, 1].imshow(to_disp(up(warped_small)))
    axes[1, 1].set_title(f"B: down to {LATENT_RES} -> warp")
    im = axes[1, 2].imshow(resid_map, cmap="inferno", vmin=0, vmax=max(0.05, resid_map.max()))
    axes[1, 2].set_title(f"|A - B| @ {LATENT_RES}  (MAE={mae:.3f})")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)
    # warp magnitude at latent res, to show what got aliased
    flow = warp_small.squeeze(0).cpu().numpy()
    axes[1, 3].imshow(np.linalg.norm(flow, axis=-1), cmap="viridis")
    axes[1, 3].set_title(f"|warp| @ {LATENT_RES}")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        f"Downsample-then-warp vs warp-then-downsample  "
        f"(image {IMAGE_RES}, latent {LATENT_RES}, RoMA '{ROMA_SETTING}')",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PATH, dpi=110, bbox_inches="tight")
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
