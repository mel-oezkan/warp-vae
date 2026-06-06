"""VAE robustness demo: black-box occlusion + salt&pepper noise in image and latent space.

Loads the f8 SD-VAE baseline, picks a few CO3D hydrant images, and produces a grid
showing how reconstructions degrade under four corruption types:
  - image-space block-out
  - image-space salt & pepper
  - latent-space block-out (zero a square region of z)
  - latent-space salt & pepper (replace random latent cells with +/- extreme values)

Run:
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/demos/vae_corruption_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.analysis.model_utils import load_model

REPO = Path("/visinf/home/lab_mozkan/computer-vision-proj-lab")
CKPT = REPO / "weights/f8/model.ckpt"
CFG = REPO / "config/baseVAE.yaml"
OUT_DIR = REPO / "outputs/scripts/vae_corruption_demo"

IMG_PATHS = [
    "/data/lab_moezkan/co3d_full/hydrant/415_57151_110224/images/frame000070.jpg",
    "/data/lab_moezkan/co3d_full/hydrant/415_57151_110224/images/frame000138.jpg",
    "/data/lab_moezkan/co3d_full/hydrant/415_57151_110224/images/frame000193.jpg",
]

IMAGE_SIZE = 256
BLOCK_FRAC = 0.35      # side length of black square, as fraction of dim
SP_PROB = 0.10         # fraction of pixels/cells to corrupt
SEED = 0


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
    ])
    return tfm(img)


def to_display(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1) / 2
    return x.permute(1, 2, 0).numpy()


def block_out(x: torch.Tensor, frac: float, rng: np.random.Generator, fill: float = -1.0) -> torch.Tensor:
    """Zero a random square region. Works in image (-1..1) or latent space."""
    out = x.clone()
    _, h, w = out.shape
    bh = max(1, int(round(h * frac)))
    bw = max(1, int(round(w * frac)))
    top = int(rng.integers(0, h - bh + 1))
    left = int(rng.integers(0, w - bw + 1))
    out[:, top:top + bh, left:left + bw] = fill
    return out


def salt_pepper(x: torch.Tensor, prob: float, rng: np.random.Generator,
                lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    """Replace random cells with lo/hi values."""
    out = x.clone()
    mask = torch.from_numpy(rng.random(out.shape) < prob)
    salt = torch.from_numpy(rng.random(out.shape) < 0.5)
    out[mask & salt] = hi
    out[mask & ~salt] = lo
    return out


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading VAE from {CKPT}")
    model, _ = load_model(str(CKPT), str(CFG), model_type="ldm")
    model = model.to(device).eval()

    rng = np.random.default_rng(SEED)

    # Pre-encode one image to learn latent stats for salt/pepper extremes.
    probe = load_image(IMG_PATHS[0]).unsqueeze(0).to(device)
    probe_z = model.encode(probe).mode()
    z_lo, z_hi = probe_z.min().item(), probe_z.max().item()
    z_amp = max(abs(z_lo), abs(z_hi)) * 1.5  # push beyond the normal range
    print(f"Latent range observed: [{z_lo:.2f}, {z_hi:.2f}]  using +/-{z_amp:.2f} for S&P")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def lat_vis(zt: torch.Tensor) -> np.ndarray:
        v = zt[:3].detach().cpu()
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        return v.permute(1, 2, 0).numpy()

    def l1_vs_source(recon: torch.Tensor, source: torch.Tensor) -> float:
        # Compare in [0,1] image space against the clean source image.
        a = (recon.detach().cpu().clamp(-1, 1) + 1) / 2
        b = (source.detach().cpu().clamp(-1, 1) + 1) / 2
        return (a - b).abs().mean().item()

    def save_panels(panels, recon_for_l1, source, fname):
        # panels: list of (image_np, title). recon_for_l1: tensor scored vs source.
        l1 = l1_vs_source(recon_for_l1, source)
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.8))
        if n == 1:
            axes = [axes]
        for ax, (im, title) in zip(axes, panels):
            ax.imshow(im); ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"L1 vs source = {l1:.3f}", fontsize=11)
        fig.tight_layout()
        out = OUT_DIR / fname
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out.name}  (L1={l1:.3f})")

    for r, p in enumerate(IMG_PATHS):
        x = load_image(p).to(device)
        xb = x.unsqueeze(0)
        src_disp = to_display(x)

        # Clean encode/decode
        z = model.encode(xb).mode()
        recon = model.decode(z)[0]

        # Image-space corruptions
        x_block = block_out(x, BLOCK_FRAC, rng, fill=-1.0)
        x_sp = salt_pepper(x, SP_PROB, rng, lo=-1.0, hi=1.0)
        recon_block_img = model.decode(model.encode(x_block.unsqueeze(0)).mode())[0]
        recon_sp_img = model.decode(model.encode(x_sp.unsqueeze(0)).mode())[0]

        # Latent-space corruptions (corrupt z, then decode)
        z0 = z[0]
        z_block = block_out(z0, BLOCK_FRAC, rng, fill=0.0)
        z_sp = salt_pepper(z0, SP_PROB, rng, lo=-z_amp, hi=z_amp)
        recon_block_lat = model.decode(z_block.unsqueeze(0))[0]
        recon_sp_lat = model.decode(z_sp.unsqueeze(0))[0]

        print(f"[image {r}] {Path(p).name}")
        save_panels(
            [(src_disp, "source"), (to_display(recon), "recon (clean)")],
            recon, x, f"img{r}_0_clean.png")
        save_panels(
            [(to_display(x_block), "input: img block"), (to_display(recon_block_img), "recon")],
            recon_block_img, x, f"img{r}_1_img_block.png")
        save_panels(
            [(to_display(x_sp), "input: img S&P"), (to_display(recon_sp_img), "recon")],
            recon_sp_img, x, f"img{r}_2_img_sp.png")
        save_panels(
            [(src_disp, "source"),
             (lat_vis(z_block), "latent block (z[:3])"),
             (to_display(recon_block_lat), "recon")],
            recon_block_lat, x, f"img{r}_3_latent_block.png")
        save_panels(
            [(src_disp, "source"),
             (lat_vis(z_sp), "latent S&P (z[:3])"),
             (to_display(recon_sp_lat), "recon")],
            recon_sp_lat, x, f"img{r}_4_latent_sp.png")


if __name__ == "__main__":
    main()
