"""
Test latent-space inpainting strategies for warped latents.

Idea: when warping latent_A -> view B via RoMA correspondences, the warp
field has invalid / low-confidence regions (occlusion, out-of-bounds, low
texture). Decoding these holes directly produces black blobs in the
reconstruction. We test several strategies for filling those holes using
the target view's own latent (latent_B), which is the cheapest possible
"inpainter".

Strategies tested (per pair):
  (1) baseline_warp     : decode( warp(z_A) )                     -- holes -> 0
  (2) hard_copy         : holes filled with z_B (hard binary mask)
  (3) soft_blend        : c * warp(z_A) + (1 - c) * z_B           (confidence blend)
  (4) feathered_copy    : hard mask with Gaussian-feathered seam
  (5) blurred_seam      : hard copy, then 3x3 Gaussian blur on seam latents

For each variant we save:
  - the decoded image
  - L1 in latent space vs z_B (the "oracle")
  - L1 in pixel space vs target image

Usage:
    conda activate cv
    PYTHONPATH=/visinf/home/lab_mozkan/computer-vision-proj-lab \
        python /visinf/home/lab_mozkan/computer-vision-proj-lab/scripts/demos/test_latent_inpaint_warp.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.model_utils import load_model, denormalize
from src.data.warp_dataset import PrecomputedWarpDataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT_PATH = PROJECT_ROOT / "weights" / "f8" / "model.ckpt"
CONFIG_PATH = PROJECT_ROOT / "config" / "warp_vae_hydrant.yaml"

CO3D_ROOT = "/visinf/projects_students/dlcv2025_groupZ/co3d_full"
BB_FILE = "/visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz"
WARP_DIR = "/visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant_cropped"

OUT_DIR = PROJECT_ROOT / "eval_outputs" / "latent_inpaint_warp"
IMAGE_SIZE = 256
NUM_PAIRS = 4
CONFIDENCE_THRESHOLD = 0.25  # used for hard-mask strategies
SEED = 0


# ---------------------------------------------------------------------------
# Latent-space inpainting strategies
# ---------------------------------------------------------------------------
def gaussian_kernel(channels: int, ksize: int = 3, sigma: float = 1.0,
                    device="cpu", dtype=torch.float32) -> torch.Tensor:
    coords = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    return kernel_2d.expand(channels, 1, ksize, ksize).contiguous()


def warp_latent(z: torch.Tensor, warp_img: torch.Tensor) -> torch.Tensor:
    """Warp a latent (B,C,h,w) using an image-resolution warp field (B,H,W,2)."""
    h = z.shape[2]
    warp_lat = F.interpolate(
        warp_img.permute(0, 3, 1, 2), size=(h, h),
        mode="bilinear", align_corners=False,
    ).permute(0, 2, 3, 1)
    return F.grid_sample(z, warp_lat, mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def latent_confidence(conf_img: torch.Tensor, h: int) -> torch.Tensor:
    """Resize image-resolution confidence (B,H,W) -> latent (B,1,h,h) in [0,1]."""
    c = F.interpolate(conf_img.unsqueeze(1), size=(h, h),
                      mode="bilinear", align_corners=False)
    return c.clamp(0, 1)


def warp_validity(warp_img: torch.Tensor, h: int) -> torch.Tensor:
    """In-bounds mask at latent resolution (B,1,h,h)."""
    warp_lat = F.interpolate(
        warp_img.permute(0, 3, 1, 2), size=(h, h),
        mode="bilinear", align_corners=False,
    )  # (B, 2, h, h)
    in_bounds = ((warp_lat[:, 0] >= -1) & (warp_lat[:, 0] <= 1) &
                 (warp_lat[:, 1] >= -1) & (warp_lat[:, 1] <= 1))
    return in_bounds.unsqueeze(1).float()


def inpaint_variants(z_warped: torch.Tensor,
                     z_target: torch.Tensor,
                     conf_latent: torch.Tensor,
                     valid_latent: torch.Tensor,
                     hard_threshold: float = 0.25) -> dict:
    """Return dict of latent inpainting variants keyed by name."""
    C = z_warped.shape[1]
    device, dtype = z_warped.device, z_warped.dtype

    # combined trust mask in [0, 1]: confidence AND in-bounds
    trust = conf_latent * valid_latent  # (B,1,h,h)

    # 1) baseline (no inpainting): pure warp, zeros in holes
    z_baseline = z_warped * valid_latent  # zero out OOB so it matches what decoder sees

    # 2) hard copy
    hard_mask = (trust > hard_threshold).float()
    z_hard = hard_mask * z_warped + (1 - hard_mask) * z_target

    # 3) soft confidence blend
    z_soft = trust * z_warped + (1 - trust) * z_target

    # 4) feathered hard copy (Gaussian-blur the hard mask -> soft alpha at the seam)
    kernel = gaussian_kernel(1, ksize=5, sigma=1.5, device=device, dtype=dtype)
    feather = F.conv2d(hard_mask, kernel, padding=2).clamp(0, 1)
    z_feather = feather * z_warped + (1 - feather) * z_target

    # 5) blurred seam: take hard copy, then blur per-channel near the seam
    seam = (feather > 0.05) & (feather < 0.95)
    seam = seam.float()
    kC = gaussian_kernel(C, ksize=3, sigma=0.8, device=device, dtype=dtype)
    z_blur_all = F.conv2d(z_hard, kC, padding=1, groups=C)
    z_seam_blur = seam * z_blur_all + (1 - seam) * z_hard

    return {
        "baseline_warp": z_baseline,
        "hard_copy": z_hard,
        "soft_blend": z_soft,
        "feathered_copy": z_feather,
        "blurred_seam": z_seam_blur,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def to_img(t: torch.Tensor) -> np.ndarray:
    return denormalize(t).clamp(0, 1).cpu().permute(1, 2, 0).numpy()


def plot_pair(pair_idx, img_a, img_b, recon_a, recon_b,
              decoded_variants, conf_latent, valid_latent, metrics, out_path):
    variant_names = list(decoded_variants.keys())
    n_cols = 4 + len(variant_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols, 3.2))

    axes[0].imshow(to_img(img_a)); axes[0].set_title("source A")
    axes[1].imshow(to_img(img_b)); axes[1].set_title("target B")
    axes[2].imshow(to_img(recon_a)); axes[2].set_title("recon(z_A)")
    axes[3].imshow(to_img(recon_b)); axes[3].set_title("recon(z_B)\noracle")

    trust = (conf_latent * valid_latent)[0, 0].cpu().numpy()
    for i, name in enumerate(variant_names):
        ax = axes[4 + i]
        ax.imshow(to_img(decoded_variants[name][0]))
        m = metrics[name]
        ax.set_title(f"{name}\nL1_lat={m['l1_lat']:.3f}\nL1_pix={m['l1_pix']:.3f}")

    # overlay trust mask on the source for context
    axes[0].imshow(np.kron(trust, np.ones((IMAGE_SIZE // trust.shape[0],) * 2)),
                   cmap="gray", alpha=0.35)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Pair #{pair_idx}: latent inpainting via target-view latent", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, _ = load_model(str(CKPT_PATH), str(CONFIG_PATH), model_type="ldm")
    model = model.to(device).eval()

    print("Loading dataset...")
    dataset = PrecomputedWarpDataset(
        root_dir=CO3D_ROOT,
        bb_file=BB_FILE,
        warp_dir=WARP_DIR,
        image_size=IMAGE_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        crop_images=True,
    )
    print(f"  {len(dataset)} precomputed pairs")

    # Spread picks across the dataset for variety
    pick = np.linspace(0, len(dataset) - 1, NUM_PAIRS, dtype=int)

    summary = {name: {"l1_lat": [], "l1_pix": []} for name in
               ["baseline_warp", "hard_copy", "soft_blend", "feathered_copy", "blurred_seam"]}

    for n, idx in enumerate(pick):
        sample = dataset[int(idx)]
        img_a = sample["image"].unsqueeze(0).to(device)
        img_b = sample["image_target"].unsqueeze(0).to(device)
        warp_ab = sample["warp_ab"].unsqueeze(0).to(device)
        conf_ab = sample["confidence_ab"].unsqueeze(0).to(device)

        with torch.no_grad():
            post_a = model.encode(img_a); z_a = post_a.mode()
            post_b = model.encode(img_b); z_b = post_b.mode()

            h = z_a.shape[2]
            z_warp = warp_latent(z_a, warp_ab)
            conf_lat = latent_confidence(conf_ab, h)
            valid_lat = warp_validity(warp_ab, h)

            variants = inpaint_variants(z_warp, z_b, conf_lat, valid_lat,
                                        hard_threshold=CONFIDENCE_THRESHOLD)
            decoded = {k: model.decode(v) for k, v in variants.items()}
            recon_a = model.decode(z_a)
            recon_b = model.decode(z_b)

        # metrics: vs oracle latent z_b, and vs target image
        metrics = {}
        for name, z in variants.items():
            l1_lat = (z - z_b).abs().mean().item()
            l1_pix = (denormalize(decoded[name]) - denormalize(img_b)).abs().mean().item()
            metrics[name] = {"l1_lat": l1_lat, "l1_pix": l1_pix}
            summary[name]["l1_lat"].append(l1_lat)
            summary[name]["l1_pix"].append(l1_pix)

        out_path = OUT_DIR / f"pair_{n:02d}_idx{int(idx)}.png"
        plot_pair(n, img_a[0], img_b[0], recon_a[0], recon_b[0],
                  {k: v for k, v in decoded.items()},
                  conf_lat, valid_lat, metrics, out_path)
        print(f"[{n+1}/{NUM_PAIRS}] saved {out_path.name}")
        for name, m in metrics.items():
            print(f"    {name:16s}  L1_lat={m['l1_lat']:.4f}  L1_pix={m['l1_pix']:.4f}")

    # aggregate
    print("\n=== Averages across pairs ===")
    print(f"{'variant':<18s} {'L1_latent':>10s} {'L1_pixel':>10s}")
    for name, m in summary.items():
        print(f"{name:<18s} {np.mean(m['l1_lat']):>10.4f} {np.mean(m['l1_pix']):>10.4f}")

    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
