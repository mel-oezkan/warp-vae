"""Sweep latent/image corruption intensity for the f8 SD-VAE.

A "corruption op" picks a random subset of cells (Bernoulli(p) per cell) and replaces
their values with something. We compare two ops:

  - salt_pepper: replace with +/- extreme values
  - nbr_mean:    replace with the mean of the 8 spatial neighbors (3x3 blur at those cells)

For each op we sweep p, apply it in image space and in latent space, decode, and report
L1 against the clean source. Outputs: per-op grids + overlaid L1 curves.

Run:
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/sweeps/vae_sp_sweep.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
OUT_DIR = REPO / "outputs/scripts/vae_sp_sweep"

IMG_PATH = "/data/lab_moezkan/co3d_full/hydrant/415_57151_110224/images/frame000070.jpg"
IMAGE_SIZE = 256
PROBS = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50]
SEED = 0


# ---------- I/O helpers ----------

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
    """(C,H,W) in [-1,1] -> (1,C,H,W) in [0,1]."""
    x = x.detach().clamp(-1, 1)
    return ((x + 1) / 2).unsqueeze(0)


def _sobel(img01: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B,2C,H,W) Sobel magnitude per channel (gx,gy stacked)."""
    C = img01.shape[1]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img01.dtype,
                      device=img01.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img01.dtype,
                      device=img01.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    gx = F.conv2d(img01, kx, padding=1, groups=C)
    gy = F.conv2d(img01, ky, padding=1, groups=C)
    return torch.cat([gx, gy], dim=1)


def _highpass(img01: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Image minus its Gaussian blur (residual high frequencies)."""
    # Build separable Gaussian kernel
    k = max(3, int(sigma * 6) | 1)  # odd
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
    """Computes a battery of similarity metrics between (recon, source) tensors in [-1,1]."""

    def __init__(self, device):
        self.device = device
        self.lpips = lpips.LPIPS(net="vgg").to(device).eval()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    @torch.no_grad()
    def compute(self, recon: torch.Tensor, source: torch.Tensor) -> Dict[str, float]:
        a = _to_01(recon).to(self.device)
        b = _to_01(source).to(self.device)
        # LPIPS expects [-1,1]
        a_pm = a * 2 - 1
        b_pm = b * 2 - 1
        out = {
            "L1": (a - b).abs().mean().item(),
            "LPIPS": self.lpips(a_pm, b_pm).item(),
            "SSIM": 1.0 - self.ssim(a, b).item(),         # convert to distance
            "MS-SSIM": 1.0 - self.msssim(a, b).item(),
            "grad-L1": (_sobel(a) - _sobel(b)).abs().mean().item(),
            "highfreq-L1": (_highpass(a) - _highpass(b)).abs().mean().item(),
        }
        return out


# ---------- fill ops ----------
# Signature: (x, mask) -> tensor where x has shape (C, H, W) and mask is bool (C, H, W).
# Each op produces replacement values for the positions where mask is True.

FillOp = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def make_salt_pepper(lo: float, hi: float, rng: np.random.Generator) -> FillOp:
    def op(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        salt = torch.from_numpy(rng.random(out.shape) < 0.5).to(out.device)
        out[mask & salt] = hi
        out[mask & ~salt] = lo
        return out
    return op


def nbr_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked cells with the mean of their 8 spatial neighbors (per channel).

    Implemented as a 3x3 box convolution that excludes the center, applied per channel.
    Edges: divide by the actual neighbor count (zero-padded edges contribute 0 weight).
    """
    out = x.clone()
    C, H, W = x.shape
    kernel = torch.tensor(
        [[1.0, 1.0, 1.0],
         [1.0, 0.0, 1.0],
         [1.0, 1.0, 1.0]],
        device=x.device, dtype=x.dtype,
    ).view(1, 1, 3, 3)
    # Per-channel conv: reshape (C, H, W) -> (C, 1, H, W), conv with groups=1 per-channel.
    x_4d = x.unsqueeze(1)  # (C, 1, H, W)
    nbr_sum = F.conv2d(x_4d, kernel, padding=1)  # (C, 1, H, W)
    ones = torch.ones_like(x_4d)
    nbr_count = F.conv2d(ones, kernel, padding=1).clamp_min(1.0)
    nbr_avg = (nbr_sum / nbr_count).squeeze(1)  # (C, H, W)
    out[mask] = nbr_avg[mask]
    return out


def apply_corruption(x: torch.Tensor, prob: float, op: FillOp,
                     rng: np.random.Generator) -> torch.Tensor:
    if prob <= 0:
        return x.clone()
    mask = torch.from_numpy(rng.random(x.shape) < prob).to(x.device)
    return op(x, mask)


# ---------- sweep ----------

METRIC_NAMES = ["L1", "LPIPS", "SSIM", "MS-SSIM", "grad-L1", "highfreq-L1"]


@dataclass
class SweepResult:
    name: str
    probs: list
    inputs_img: list = field(default_factory=list)
    recons_img: list = field(default_factory=list)
    inputs_lat: list = field(default_factory=list)
    recons_lat: list = field(default_factory=list)
    # metrics[metric_name] = {"image": [...], "latent": [...]}
    metrics: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


@torch.no_grad()
def run_sweep(name: str, model, metrics: MetricBank, x: torch.Tensor, z0: torch.Tensor,
              probs, op_image_factory, op_latent_factory) -> SweepResult:
    res = SweepResult(name=name, probs=list(probs))
    res.metrics = {m: {"image": [], "latent": []} for m in METRIC_NAMES}
    rng_img_mask = np.random.default_rng(SEED)
    rng_lat_mask = np.random.default_rng(SEED + 1)
    for prob in probs:
        op_img = op_image_factory()
        op_lat = op_latent_factory()

        x_corr = apply_corruption(x, prob, op_img, rng_img_mask)
        rec_img = model.decode(model.encode(x_corr.unsqueeze(0)).mode())[0]
        res.inputs_img.append(to_display(x_corr))
        res.recons_img.append(to_display(rec_img))
        m_img = metrics.compute(rec_img, x)

        z_corr = apply_corruption(z0, prob, op_lat, rng_lat_mask)
        rec_lat = model.decode(z_corr.unsqueeze(0))[0]
        res.inputs_lat.append(latent_to_display(z_corr))
        res.recons_lat.append(to_display(rec_lat))
        m_lat = metrics.compute(rec_lat, x)

        for k in METRIC_NAMES:
            res.metrics[k]["image"].append(m_img[k])
            res.metrics[k]["latent"].append(m_lat[k])

        print(f"  [{name}] p={prob:0.3f}  "
              + "  ".join(f"{k}: img={m_img[k]:.3f}/lat={m_lat[k]:.3f}" for k in METRIC_NAMES))
    return res


def save_grid(res: SweepResult, out_path: Path):
    n = len(res.probs)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8.5))
    row_titles = ["image input (corrupted)", "image-space recon",
                  "latent input (z[:3])", "latent-space recon"]
    rows = [res.inputs_img, res.recons_img, res.inputs_lat, res.recons_lat]
    # Label rows 1 and 3 (the recons) with L1 + LPIPS underneath each panel.
    l1_img = res.metrics["L1"]["image"]
    l1_lat = res.metrics["L1"]["latent"]
    lpips_img = res.metrics["LPIPS"]["image"]
    lpips_lat = res.metrics["LPIPS"]["latent"]
    score_rows = [None, list(zip(l1_img, lpips_img)), None, list(zip(l1_lat, lpips_lat))]
    for r, (row_imgs, scores) in enumerate(zip(rows, score_rows)):
        for c, im in enumerate(row_imgs):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"p={res.probs[c]:.3f}", fontsize=10)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=10)
            if scores is not None:
                l1, lp = scores[c]
                ax.set_xlabel(f"L1={l1:.3f}  LPIPS={lp:.3f}", fontsize=8)
    fig.suptitle(f"{res.name} sweep — f8 SD-VAE on a CO3D hydrant", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_metric_curves(results: List[SweepResult], out_path: Path):
    n_m = len(METRIC_NAMES)
    cols = 3
    rows = (n_m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)
    style = {"salt_pepper": ("o-", "s-"), "nbr_mean": ("^--", "D--")}
    for i, m in enumerate(METRIC_NAMES):
        ax = axes[i]
        for res in results:
            m_img, m_lat = style.get(res.name, ("o-", "s-"))
            ax.plot(res.probs, res.metrics[m]["image"], m_img, label=f"{res.name} (image)")
            ax.plot(res.probs, res.metrics[m]["latent"], m_lat, label=f"{res.name} (latent)")
        ax.set_title(m)
        ax.set_xlabel("corruption probability p")
        ax.set_ylabel(f"{m}(recon, source)" + (" [1-score]" if m in ("SSIM", "MS-SSIM") else ""))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(n_m, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Perceptual / structural metrics under sparse corruption", fontsize=13)
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
    z = model.encode(x.unsqueeze(0)).mode()
    z0 = z[0]
    z_lo, z_hi = z0.min().item(), z0.max().item()
    z_amp = max(abs(z_lo), abs(z_hi)) * 1.5
    print(f"Latent range: [{z_lo:.2f}, {z_hi:.2f}], S&P latent extremes: +/-{z_amp:.2f}")

    # Each factory returns a fresh op; S&P needs its own value-rng to keep salt/pepper
    # choices deterministic per sweep step.
    sp_value_rng_img = np.random.default_rng(SEED + 100)
    sp_value_rng_lat = np.random.default_rng(SEED + 101)

    metrics = MetricBank(device)

    results = []

    print("Sweep: salt_pepper")
    results.append(run_sweep(
        "salt_pepper", model, metrics, x, z0, PROBS,
        op_image_factory=lambda: make_salt_pepper(-1.0, 1.0, sp_value_rng_img),
        op_latent_factory=lambda: make_salt_pepper(-z_amp, z_amp, sp_value_rng_lat),
    ))
    save_grid(results[-1], OUT_DIR / "sp_sweep_grid.png")

    print("Sweep: nbr_mean")
    results.append(run_sweep(
        "nbr_mean", model, metrics, x, z0, PROBS,
        op_image_factory=lambda: nbr_mean,
        op_latent_factory=lambda: nbr_mean,
    ))
    save_grid(results[-1], OUT_DIR / "nbrmean_sweep_grid.png")

    save_metric_curves(results, OUT_DIR / "sweep_metrics.png")


if __name__ == "__main__":
    main()
