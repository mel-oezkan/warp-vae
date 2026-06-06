"""RoMA-warp equivariance sweep for the f8 SD-VAE (no training, just probing).

Like vae_eq_sweep.py, but the transform T is a *learned* dense warp from RoMA
mapping a reference frame x_A onto a target frame x_B. Sweep axis = frame gap
inside one CO3D sequence (small gap -> near-identity warp; large gap -> big warp).

For each frame gap g we pick target x_B = sequence[ref + g] and compare:
  - image branch:  warp(x_A, W_{A->B})            -> encode -> decode
  - latent branch: encode(x_A) -> warp(z_A, W_{A->B}_downsampled) -> decode
both against x_B as the ground-truth target. Metrics masked to RoMA
high-confidence regions (occlusion / out-of-frame pixels excluded).

Run:
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/sweeps/vae_roma_sweep.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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
from src.analysis.roma_metrics import load_roma_model, warp_to_latent_warp

REPO = Path("/visinf/home/lab_mozkan/computer-vision-proj-lab")
CKPT = REPO / "weights/f8/model.ckpt"
CFG = REPO / "config/baseVAE.yaml"
OUT_DIR = REPO / "outputs/scripts/vae_roma_sweep"

SEQ_DIR = Path("/data/lab_moezkan/co3d_full/hydrant/415_57151_110224/images")
REF_FRAME_IDX = 70                        # 1-based filename index -> frame000070.jpg
FRAME_GAPS = [0, 2, 5, 10, 20, 40, 80]    # offset from reference
IMAGE_SIZE = 256
LATENT_SIZE = 32
CONF_THRESHOLDS = [0.8]
ROMA_SETTINGS = ["turbo", "fast", "base"]  # "precise" OOMs at 1280x1280 on 11GB GPUs


# ---------- I/O helpers ----------

def frame_path(idx: int) -> Path:
    return SEQ_DIR / f"frame{idx:06d}.jpg"


def load_image_tensor(path: Path) -> torch.Tensor:
    """Load image as (3,H,W) tensor in [-1,1]."""
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return tfm(img)


def load_image_pil(path: Path) -> Image.Image:
    """Load image at IMAGE_SIZE for RoMA (RoMA wants PIL)."""
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return img


def to_display(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    return x.permute(1, 2, 0).numpy()


def conf_to_display(c: torch.Tensor) -> np.ndarray:
    return c.detach().cpu().numpy()


def red_diff_overlay(src: torch.Tensor, warped: torch.Tensor, target: torch.Tensor,
                     strength: float = 2.0) -> np.ndarray:
    """Render src image in grayscale with a red overlay proportional to |warped - target|.

    All inputs are (3,H,W) in [-1,1]. Returns (H,W,3) uint8-ish float in [0,1].
    strength: multiplier on the per-pixel L1 (in [0,1] units) before clamping.
    """
    src01 = ((src.detach().cpu().clamp(-1, 1) + 1) / 2)         # (3,H,W) in [0,1]
    w01 = ((warped.detach().cpu().clamp(-1, 1) + 1) / 2)
    t01 = ((target.detach().cpu().clamp(-1, 1) + 1) / 2)
    diff = (w01 - t01).abs().mean(dim=0)                        # (H,W) in [0,1]
    alpha = (diff * strength).clamp(0, 1)                       # (H,W)
    gray = src01.mean(dim=0, keepdim=True).repeat(3, 1, 1)      # (3,H,W) desaturated bg
    red = torch.zeros_like(gray); red[0] = 1.0
    out = gray * (1 - alpha) + red * alpha
    return out.permute(1, 2, 0).numpy()


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
    """Same metrics as vae_eq_sweep; computes a confidence-masked variant too."""

    def __init__(self, device):
        self.device = device
        self.lpips = lpips.LPIPS(net="vgg").to(device).eval()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    @torch.no_grad()
    def compute(self, recon: torch.Tensor, source: torch.Tensor,
                mask: torch.Tensor) -> Dict[str, float]:
        """mask: (H,W) float in [0,1]. L1 / grad / highfreq use it; SSIM/LPIPS are global
        (no native masking) but operate on masked-zeroed images for fairness."""
        a = _to_01(recon).to(self.device)
        b = _to_01(source).to(self.device)
        m = mask.to(self.device).view(1, 1, *mask.shape)
        denom = m.sum().clamp_min(1.0)

        a_m = a * m
        b_m = b * m
        a_pm = a_m * 2 - 1
        b_pm = b_m * 2 - 1

        l1 = ((a - b).abs() * m).sum() / (denom * a.shape[1])
        grad_l1 = (((_sobel(a) - _sobel(b)).abs() * m).sum()
                   / (denom * a.shape[1] * 2))
        hf_l1 = (((_highpass(a) - _highpass(b)).abs() * m).sum()
                 / (denom * a.shape[1]))
        return {
            "L1": l1.item(),
            "LPIPS": self.lpips(a_pm, b_pm).item(),
            "SSIM": 1.0 - self.ssim(a_m, b_m).item(),
            "MS-SSIM": 1.0 - self.msssim(a_m, b_m).item(),
            "grad-L1": grad_l1.item(),
            "highfreq-L1": hf_l1.item(),
        }


# ---------- RoMA helpers ----------

@torch.no_grad()
def roma_warp(roma_model, pil_a: Image.Image, pil_b: Image.Image,
              device) -> tuple:
    """Returns (warp_image (1,H,W,2), conf_image (1,H,W,1)) at IMAGE_SIZE."""
    pred = roma_model.match(pil_a, pil_b)
    warp = pred["warp_AB"]
    overlap = pred.get("overlap_AB")
    if overlap is None:
        overlap = pred["confidence_AB"].mean(dim=-1, keepdim=True)

    if warp.shape[1] != IMAGE_SIZE or warp.shape[2] != IMAGE_SIZE:
        warp = F.interpolate(warp.permute(0, 3, 1, 2),
                             size=(IMAGE_SIZE, IMAGE_SIZE),
                             mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        overlap = F.interpolate(overlap.permute(0, 3, 1, 2),
                                size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return warp.to(device), overlap.to(device)


def confidence_mask(conf_img: torch.Tensor, warp_img: torch.Tensor,
                    size: int, threshold: float) -> torch.Tensor:
    """conf_img: (1,H,W,1), warp_img: (1,H,W,2). Returns (size,size) float mask in {0,1}.
    A pixel is valid iff RoMA confidence > threshold AND warp lands in-bounds."""
    in_bounds = (warp_img.abs() <= 1.0).all(dim=-1, keepdim=True).float()  # (1,H,W,1)
    valid = (conf_img > threshold).float() * in_bounds                     # (1,H,W,1)
    valid = valid.permute(0, 3, 1, 2)                                      # (1,1,H,W)
    if valid.shape[-1] != size:
        valid = F.interpolate(valid, size=(size, size), mode="bilinear",
                              align_corners=False)
        valid = (valid > 0.5).float()
    return valid[0, 0]


# ---------- sweep ----------

METRIC_NAMES = ["L1", "LPIPS", "SSIM", "MS-SSIM", "grad-L1", "highfreq-L1"]


@dataclass
class SweepResult:
    gaps: list
    target_idx: list = field(default_factory=list)        # absolute frame indices
    targets: list = field(default_factory=list)           # x_B
    masks: list = field(default_factory=list)             # conf masks (image res)
    img_warped: list = field(default_factory=list)        # warp(x_A) in pixel space
    src_diff: list = field(default_factory=list)          # x_A with red diff overlay
    img_recon: list = field(default_factory=list)         # decode(encode(warp(x_A)))
    lat_recon: list = field(default_factory=list)         # decode(warp(z_A))
    metrics: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


@torch.no_grad()
def run_sweep(model, roma_model, metrics: MetricBank, device,
              ref_idx: int, gaps, thresholds) -> Dict[float, SweepResult]:
    """Returns one SweepResult per confidence threshold. RoMA + VAE work is shared."""
    results = {t: SweepResult(gaps=list(gaps)) for t in thresholds}
    for t in thresholds:
        results[t].metrics = {m: {"image": [], "latent": []} for m in METRIC_NAMES}

    x_a = load_image_tensor(frame_path(ref_idx)).to(device)
    pil_a = load_image_pil(frame_path(ref_idx))
    z_a = model.encode(x_a.unsqueeze(0)).mode()[0]

    for g in gaps:
        tgt_idx = ref_idx + g
        tgt_path = frame_path(tgt_idx)
        if not tgt_path.exists():
            print(f"  gap={g}: {tgt_path} missing, skipping")
            continue
        x_b = load_image_tensor(tgt_path).to(device)
        pil_b = load_image_pil(tgt_path)

        if g == 0:
            warp_img = torch.stack(torch.meshgrid(
                torch.linspace(-1, 1, IMAGE_SIZE, device=device),
                torch.linspace(-1, 1, IMAGE_SIZE, device=device),
                indexing="xy"), dim=-1).unsqueeze(0)
            conf_img = torch.ones(1, IMAGE_SIZE, IMAGE_SIZE, 1, device=device)
        else:
            warp_img, conf_img = roma_warp(roma_model, pil_a, pil_b, device)

        warp_lat = warp_to_latent_warp(warp_img, IMAGE_SIZE, LATENT_SIZE)

        # image branch: warp pixels then VAE round-trip
        warped_pixels = F.grid_sample(x_a.unsqueeze(0), warp_img,
                                      mode="bilinear", padding_mode="zeros",
                                      align_corners=False)[0]
        rec_img = model.decode(model.encode(warped_pixels.unsqueeze(0)).mode())[0]

        # latent branch: warp the latent grid then decode
        warped_lat = F.grid_sample(z_a.unsqueeze(0), warp_lat,
                                   mode="bilinear", padding_mode="zeros",
                                   align_corners=False)[0]
        rec_lat = model.decode(warped_lat.unsqueeze(0))[0]

        for t in thresholds:
            mask_img = confidence_mask(conf_img, warp_img, IMAGE_SIZE, t)
            m_img = metrics.compute(rec_img, x_b, mask_img)
            m_lat = metrics.compute(rec_lat, x_b, mask_img)

            res = results[t]
            res.target_idx.append(tgt_idx)
            res.targets.append(to_display(x_b))
            res.masks.append(conf_to_display(mask_img))
            res.img_warped.append(to_display(warped_pixels))
            res.src_diff.append(red_diff_overlay(x_a, warped_pixels, x_b))
            res.img_recon.append(to_display(rec_img))
            res.lat_recon.append(to_display(rec_lat))
            for k in METRIC_NAMES:
                res.metrics[k]["image"].append(m_img[k])
                res.metrics[k]["latent"].append(m_lat[k])

            print(f"  gap={g:3d} thr={t:.2f}  conf_mean={float(mask_img.mean()):.2f}  "
                  + "  ".join(f"{k}: img={m_img[k]:.3f}/lat={m_lat[k]:.3f}" for k in METRIC_NAMES))

    return results


def save_grid(res: SweepResult, ref_idx: int, out_path: Path, threshold: float):
    n = len(res.gaps)
    fig, axes = plt.subplots(6, n, figsize=(2.0 * n, 12.5))
    row_titles = ["target x_B",
                  "warp(x_A) [pixel-space]",
                  "x_A + red |warp(x_A) - x_B|",
                  "image branch recon",
                  "latent branch recon",
                  "RoMA confidence mask"]
    rows = [res.targets, res.img_warped, res.src_diff,
            res.img_recon, res.lat_recon, res.masks]
    l1_img = res.metrics["L1"]["image"]; l1_lat = res.metrics["L1"]["latent"]
    lp_img = res.metrics["LPIPS"]["image"]; lp_lat = res.metrics["LPIPS"]["latent"]
    score_rows = [None, None, None,
                  list(zip(l1_img, lp_img)), list(zip(l1_lat, lp_lat)), None]
    mask_row = len(rows) - 1
    for r, (row_imgs, scores) in enumerate(zip(rows, score_rows)):
        for c, im in enumerate(row_imgs):
            ax = axes[r, c]
            if r == mask_row:
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"gap=+{res.gaps[c]}  (f{res.target_idx[c]:03d})", fontsize=9)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=10)
            if scores is not None:
                l1, lp = scores[c]
                ax.set_xlabel(f"L1={l1:.3f}  LPIPS={lp:.3f}", fontsize=8)
    fig.suptitle(f"RoMA-warp equivariance sweep — ref frame f{ref_idx:03d}, "
                 f"f8 SD-VAE  (conf threshold={threshold:.2f})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_metric_curves(res: SweepResult, out_path: Path, threshold: float):
    n_m = len(METRIC_NAMES)
    cols = 3
    rows = (n_m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)
    for i, m in enumerate(METRIC_NAMES):
        ax = axes[i]
        ax.plot(res.gaps, res.metrics[m]["image"], "o-", label="image branch")
        ax.plot(res.gaps, res.metrics[m]["latent"], "s--", label="latent branch")
        ax.set_title(m)
        ax.set_xlabel("frame gap (+frames from ref)")
        ax.set_ylabel(f"{m}(recon, x_B)  [conf-masked]"
                      + (" [1-score]" if m in ("SSIM", "MS-SSIM") else ""))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(n_m, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Equivariance gap under RoMA warp: image vs latent  "
                 f"(conf threshold={threshold:.2f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_setting_comparison(per_setting: Dict[str, Dict[float, SweepResult]],
                            threshold: float, out_path: Path):
    """One panel per metric; one line per RoMA setting; both branches shown."""
    cols = 3
    rows = (len(METRIC_NAMES) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(per_setting)))
    for i, m in enumerate(METRIC_NAMES):
        ax = axes[i]
        for (setting, results), color in zip(per_setting.items(), colors):
            res = results[threshold]
            ax.plot(res.gaps, res.metrics[m]["image"], "o-", color=color, alpha=0.5,
                    label=f"{setting} (image)")
            ax.plot(res.gaps, res.metrics[m]["latent"], "s--", color=color,
                    label=f"{setting} (latent)")
        ax.set_title(m)
        ax.set_xlabel("frame gap")
        ax.set_ylabel(f"{m}(recon, x_B)" + (" [1-score]" if m in ("SSIM", "MS-SSIM") else ""))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, ncol=2)
    for j in range(len(METRIC_NAMES), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"RoMA setting comparison (conf threshold={threshold:.2f})", fontsize=13)
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

    metrics = MetricBank(device)

    print(f"Sweep over frame gaps {FRAME_GAPS} from reference f{REF_FRAME_IDX:03d}, "
          f"thresholds={CONF_THRESHOLDS}, settings={ROMA_SETTINGS}")

    # per_setting[setting][threshold] = SweepResult
    per_setting: Dict[str, Dict[float, SweepResult]] = {}
    for setting in ROMA_SETTINGS:
        print(f"\n=== RoMA setting: {setting} ===")
        roma_model = load_roma_model(setting=setting, device=str(device), compile=False)
        results = run_sweep(model, roma_model, metrics, device,
                            REF_FRAME_IDX, FRAME_GAPS, CONF_THRESHOLDS)
        per_setting[setting] = results
        for t, res in results.items():
            tag = f"{setting}_conf{int(round(t * 100)):02d}"
            save_grid(res, REF_FRAME_IDX, OUT_DIR / f"roma_sweep_grid_{tag}.png", t)
            save_metric_curves(res, OUT_DIR / f"roma_sweep_metrics_{tag}.png", t)
        del roma_model
        torch.cuda.empty_cache()

    save_setting_comparison(per_setting, CONF_THRESHOLDS[0],
                            OUT_DIR / "roma_setting_comparison.png")


if __name__ == "__main__":
    main()
