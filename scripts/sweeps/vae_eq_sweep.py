"""Equivariance sweep for the f8 SD-VAE (no training, just probing).

EQ-VAE's claim: Dec(T(Enc(x))) ~= T(x) for transforms T in {rotate, scale}. The stock
SD-VAE is not trained for this, so applying T to the latent grid and decoding should
diverge from T(x) much faster than applying T in image space then round-tripping.

For each transform we sweep an intensity parameter, then for every step:
  - image branch: T(x) -> encode -> decode  (round-trip on the transformed image)
  - latent branch: encode(x) -> T(z) -> decode  (transform applied in latent space)
both compared against the ground-truth target T(x).

Outputs mirror vae_sp_sweep.py: per-transform grids + overlaid metric curves.

Run:
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/sweeps/vae_eq_sweep.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)
from torchvision import transforms

from src.analysis.model_utils import load_model

REPO = Path("/visinf/home/lab_mozkan/computer-vision-proj-lab")
CKPT = REPO / "weights/f8/model.ckpt"
CFG = REPO / "config/baseVAE.yaml"
OUT_DIR = REPO / "outputs/scripts/vae_eq_sweep"

IMG_PATH = "/data/lab_moezkan/co3d_full/hydrant/415_57151_110224/images/frame000070.jpg"
IMAGE_SIZE = 256

# Sweep grids
ROT_ANGLES = [0, 10, 30, 60, 90, 135, 180]      # degrees
SCALE_FACTORS = [1.0, 0.85, 0.70, 0.50, 1.20, 1.50, 2.00]


# ---------- I/O helpers (shared shape with vae_sp_sweep) ----------

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return tfm(img)


def to_display(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    return x.permute(1, 2, 0).numpy()


def latent_to_display(z: torch.Tensor) -> np.ndarray:
    v = z[:3].detach().cpu()
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return v.permute(1, 2, 0).numpy()


def _to_01(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clamp(-1, 1)
    return ((x + 1) / 2).unsqueeze(0)


def _sobel(img01: torch.Tensor) -> torch.Tensor:
    C = img01.shape[1]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img01.dtype,
                      device=img01.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img01.dtype,
                      device=img01.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    gx = F.conv2d(img01, kx, padding=1, groups=C)
    gy = F.conv2d(img01, ky, padding=1, groups=C)
    return torch.cat([gx, gy], dim=1)


def _highpass(img01: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    k = max(3, int(sigma * 6) | 1)
    coords = torch.arange(k, device=img01.device, dtype=img01.dtype) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    C = img01.shape[1]
    g_h = g.repeat(C, 1, 1, 1)
    g_v = g.transpose(-1, -2).repeat(C, 1, 1, 1)
    blur = F.conv2d(img01, g_h, padding=(0, k // 2), groups=C)
    blur = F.conv2d(blur, g_v, padding=(k // 2, 0), groups=C)
    return img01 - blur


class MetricBank:
    def __init__(self, device):
        self.device = device
        self.lpips = lpips.LPIPS(net="vgg").to(device).eval()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    @torch.no_grad()
    def compute(self, recon: torch.Tensor, source: torch.Tensor) -> Dict[str, float]:
        a = _to_01(recon).to(self.device)
        b = _to_01(source).to(self.device)
        a_pm = a * 2 - 1
        b_pm = b * 2 - 1
        return {
            "L1": (a - b).abs().mean().item(),
            "LPIPS": self.lpips(a_pm, b_pm).item(),
            "SSIM": 1.0 - self.ssim(a, b).item(),
            "MS-SSIM": 1.0 - self.msssim(a, b).item(),
            "grad-L1": (_sobel(a) - _sobel(b)).abs().mean().item(),
            "highfreq-L1": (_highpass(a) - _highpass(b)).abs().mean().item(),
        }


# ---------- transforms ----------
# A transform is a callable t(tensor_CHW) -> tensor_CHW that works on both images
# (3,H,W) and latents (4,h,w). Rotation uses bilinear interpolation; scale uses
# center-crop-then-resize for s>1 and resize-then-pad for s<1, so the output shape
# always matches the input.

Transform = Callable[[torch.Tensor], torch.Tensor]


def make_rotate(angle_deg: float) -> Transform:
    def t(x: torch.Tensor) -> torch.Tensor:
        # TF.rotate keeps spatial dims; bilinear avoids aliasing at off-axis angles.
        return TF.rotate(x, angle=float(angle_deg),
                         interpolation=TF.InterpolationMode.BILINEAR,
                         expand=False, fill=0.0)
    return t


def make_scale(factor: float) -> Transform:
    """Isotropic zoom that preserves spatial dims.

    factor > 1: zoom in (center-crop a 1/factor window, resize back up).
    factor < 1: zoom out (resize down, then pad to original size with zeros).
    """
    def t(x: torch.Tensor) -> torch.Tensor:
        C, H, W = x.shape
        if abs(factor - 1.0) < 1e-6:
            return x.clone()
        if factor > 1.0:
            crop_h = max(1, int(round(H / factor)))
            crop_w = max(1, int(round(W / factor)))
            top = (H - crop_h) // 2
            left = (W - crop_w) // 2
            cropped = x[:, top:top + crop_h, left:left + crop_w]
            return F.interpolate(cropped.unsqueeze(0), size=(H, W),
                                 mode="bilinear", align_corners=False)[0]
        # factor < 1
        new_h = max(1, int(round(H * factor)))
        new_w = max(1, int(round(W * factor)))
        small = F.interpolate(x.unsqueeze(0), size=(new_h, new_w),
                              mode="bilinear", align_corners=False)[0]
        out = torch.zeros_like(x)
        top = (H - new_h) // 2
        left = (W - new_w) // 2
        out[:, top:top + new_h, left:left + new_w] = small
        return out
    return t


# ---------- sweep ----------

METRIC_NAMES = ["L1", "LPIPS", "SSIM", "MS-SSIM", "grad-L1", "highfreq-L1"]


@dataclass
class SweepResult:
    name: str                       # e.g. "rotate" or "scale"
    param_label: str                # e.g. "angle (deg)"
    params: list                    # the sweep values
    targets: list = field(default_factory=list)        # T(x) for display
    recons_img: list = field(default_factory=list)     # decode(encode(T(x)))
    latents_T: list = field(default_factory=list)      # T(z) for display
    recons_lat: list = field(default_factory=list)     # decode(T(z))
    metrics: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


@torch.no_grad()
def run_sweep(name: str, param_label: str, model, metrics: MetricBank,
              x: torch.Tensor, z0: torch.Tensor,
              params, make_transform: Callable[[float], Transform]) -> SweepResult:
    res = SweepResult(name=name, param_label=param_label, params=list(params))
    res.metrics = {m: {"image": [], "latent": []} for m in METRIC_NAMES}
    for p in params:
        T = make_transform(p)
        target = T(x)

        # Image branch: T applied in pixel space, then VAE round-trip.
        x_t = target
        rec_img = model.decode(model.encode(x_t.unsqueeze(0)).mode())[0]
        res.targets.append(to_display(target))
        res.recons_img.append(to_display(rec_img))
        m_img = metrics.compute(rec_img, target)

        # Latent branch: T applied to the latent grid, decode.
        z_t = T(z0)
        rec_lat = model.decode(z_t.unsqueeze(0))[0]
        res.latents_T.append(latent_to_display(z_t))
        res.recons_lat.append(to_display(rec_lat))
        m_lat = metrics.compute(rec_lat, target)

        for k in METRIC_NAMES:
            res.metrics[k]["image"].append(m_img[k])
            res.metrics[k]["latent"].append(m_lat[k])

        print(f"  [{name}] {param_label}={p}  "
              + "  ".join(f"{k}: img={m_img[k]:.3f}/lat={m_lat[k]:.3f}" for k in METRIC_NAMES))
    return res


def save_grid(res: SweepResult, out_path: Path):
    n = len(res.params)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8.5))
    row_titles = ["target T(x)", "image branch: decode(encode(T(x)))",
                  "T(z)  [z[:3]]", "latent branch: decode(T(z))"]
    rows = [res.targets, res.recons_img, res.latents_T, res.recons_lat]
    l1_img = res.metrics["L1"]["image"]
    l1_lat = res.metrics["L1"]["latent"]
    lp_img = res.metrics["LPIPS"]["image"]
    lp_lat = res.metrics["LPIPS"]["latent"]
    score_rows = [None, list(zip(l1_img, lp_img)), None, list(zip(l1_lat, lp_lat))]
    for r, (row_imgs, scores) in enumerate(zip(rows, score_rows)):
        for c, im in enumerate(row_imgs):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{res.param_label}={res.params[c]}", fontsize=10)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=10)
            if scores is not None:
                l1, lp = scores[c]
                ax.set_xlabel(f"L1={l1:.3f}  LPIPS={lp:.3f}", fontsize=8)
    fig.suptitle(f"{res.name} equivariance sweep — f8 SD-VAE (no EQ training)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_metric_curves(res: SweepResult, out_path: Path):
    n_m = len(METRIC_NAMES)
    cols = 3
    rows = (n_m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)
    for i, m in enumerate(METRIC_NAMES):
        ax = axes[i]
        ax.plot(res.params, res.metrics[m]["image"], "o-", label="image branch")
        ax.plot(res.params, res.metrics[m]["latent"], "s--", label="latent branch")
        ax.set_title(m)
        ax.set_xlabel(res.param_label)
        ax.set_ylabel(f"{m}(recon, T(x))" + (" [1-score]" if m in ("SSIM", "MS-SSIM") else ""))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(n_m, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Equivariance gap under {res.name}: image vs latent transform", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    print(f"Loading VAE from {CKPT}")
    model, _ = load_model(str(CKPT), str(CFG), model_type="ldm")
    model = model.to(device).eval()

    x = load_image(IMG_PATH).to(device)
    z0 = model.encode(x.unsqueeze(0)).mode()[0]
    print(f"Latent shape: {tuple(z0.shape)}, range [{z0.min():.2f}, {z0.max():.2f}]")

    metrics = MetricBank(device)

    print("Sweep: rotate")
    res_rot = run_sweep("rotate", "angle (deg)", model, metrics, x, z0,
                        ROT_ANGLES, make_rotate)
    save_grid(res_rot, OUT_DIR / "rotate_sweep_grid.png")
    save_metric_curves(res_rot, OUT_DIR / "rotate_metrics.png")

    print("Sweep: scale")
    res_sc = run_sweep("scale", "factor", model, metrics, x, z0,
                       SCALE_FACTORS, make_scale)
    save_grid(res_sc, OUT_DIR / "scale_sweep_grid.png")
    save_metric_curves(res_sc, OUT_DIR / "scale_metrics.png")


if __name__ == "__main__":
    main()
