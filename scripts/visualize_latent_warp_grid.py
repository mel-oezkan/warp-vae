"""
Visualize latent-space warping with an EQ-VAE model.

For a pair of images (A, B) with precomputed RoMA warps, shows:
  Row 0: Original images A and B
  Row 1: VAE reconstructions of A and B
  Row 2: Warped latent decoded — white background (low-confidence → zero latent)
  Row 3: Warped latent decoded — source background (low-confidence → source latent)
  Row 4: Warped latent decoded — target background (low-confidence → target latent)

Usage:
    python scripts/visualize_latent_warp_grid.py \
        --checkpoint "checkpoints/natural-illegal-..." \
        --config config/eqvae_hydrant_cropped.yaml \
        --pair_idx 42 --output latent_warp_grid.png
"""

import sys
import random
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.model_utils import load_model
from src.data.warp_dataset import PrecomputedWarpDataset


def warp_latent(latent_src, warp, confidence, background, conf_threshold=0.2):
    """Warp a latent tensor using a RoMA warp field.

    Args:
        latent_src: (1, C, Hl, Wl) source latent to sample from
        warp: (H, W, 2) warp field in [-1, 1] at image resolution
        confidence: (H, W) confidence map at image resolution
        background: (1, C, Hl, Wl) latent to use for low-confidence regions
        conf_threshold: confidence below this is treated as zero

    Returns:
        (1, C, Hl, Wl) blended warped latent
    """
    _, C, Hl, Wl = latent_src.shape

    # Downsample warp and confidence to latent resolution
    warp_lr = (
        F.interpolate(
            warp.permute(2, 0, 1).unsqueeze(0),  # (1, 2, H, W)
            size=(Hl, Wl),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )  # (Hl, Wl, 2)

    conf_lr = F.interpolate(
        confidence.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        size=(Hl, Wl),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, Hl, Wl)

    # Apply confidence threshold
    conf_lr = torch.clamp(conf_lr - conf_threshold, min=0) / (1 - conf_threshold + 1e-8)

    # Warp source latent
    warped = F.grid_sample(
        latent_src,
        warp_lr.unsqueeze(0),  # (1, Hl, Wl, 2)
        mode="bilinear",
        align_corners=False,
    )  # (1, C, Hl, Wl)

    # Blend with background
    blended = conf_lr * warped + (1 - conf_lr) * background
    return blended


def main():
    parser = ArgumentParser(description="Visualize latent-space warping with VAE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--pair_idx",
        type=int,
        default=None,
        help="Index into precomputed warp pairs (random if not set)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Number of pair samples to visualize"
    )
    parser.add_argument("--output", type=str, default="latent_warp_grid.png")
    parser.add_argument("--conf_threshold", type=float, default=0.2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("Loading model...")
    model, model_type = load_model(args.checkpoint, args.config)
    model = model.to(device).eval()

    # Load dataset from warp config to get pairs
    from omegaconf import OmegaConf

    warp_config = OmegaConf.load(
        str(PROJECT_ROOT / "config" / "warp_vae_hydrant_recon_crop.yaml")
    )
    OmegaConf.resolve(warp_config)
    ds_params = warp_config.data.params.dataset_config.params

    dataset = PrecomputedWarpDataset(
        root_dir=ds_params.root_dir,
        bb_file=ds_params.bb_file,
        warp_dir=ds_params.warp_dir,
        image_size=int(warp_config.training.image_size),
        confidence_threshold=0.0,  # we handle thresholding ourselves
        crop_images=ds_params.get("crop_images", False),
    )
    print(f"Dataset has {len(dataset)} pairs")

    # Pick pair(s)
    if args.pair_idx is not None:
        indices = [args.pair_idx]
    else:
        indices = random.sample(range(len(dataset)), min(args.n_samples, len(dataset)))

    for sample_i, pair_idx in enumerate(indices):
        sample = dataset[pair_idx]
        img_a = sample["image"].unsqueeze(0).to(device)  # (1, 3, H, W) in [-1, 1]
        img_b = sample["image_target"].unsqueeze(0).to(device)
        warp_ab = sample["warp_ab"].to(device)  # (H, W, 2)
        conf_ab = sample["confidence_ab"].to(device)  # (H, W)
        warp_ba = sample["warp_ba"].to(device)
        conf_ba = sample["confidence_ba"].to(device)

        with torch.no_grad():
            # Encode
            post_a = model.encode(img_a)
            post_b = model.encode(img_b)
            z_a = post_a.mode()  # (1, 4, 32, 32)
            z_b = post_b.mode()

            # Reconstruct
            recon_a = model.decode(z_a)
            recon_b = model.decode(z_b)

            # Zero latent for white-ish background
            z_zero = torch.zeros_like(z_a)

            # Warp A's latent into B's viewpoint (using warp_ba: maps B coords → A coords)
            # grid_sample(z_a, warp_ba) = z_a sampled at B's grid positions
            z_a_to_b_white = warp_latent(
                z_a, warp_ba, conf_ba, z_zero, args.conf_threshold
            )
            z_a_to_b_src = warp_latent(z_a, warp_ba, conf_ba, z_a, args.conf_threshold)
            z_a_to_b_tgt = warp_latent(z_a, warp_ba, conf_ba, z_b, args.conf_threshold)

            # Warp B's latent into A's viewpoint (using warp_ab: maps A coords → B coords)
            z_b_to_a_white = warp_latent(
                z_b, warp_ab, conf_ab, z_zero, args.conf_threshold
            )
            z_b_to_a_src = warp_latent(z_b, warp_ab, conf_ab, z_b, args.conf_threshold)
            z_b_to_a_tgt = warp_latent(z_b, warp_ab, conf_ab, z_a, args.conf_threshold)

            # Decode all warped latents
            dec_a_to_b_white = model.decode(z_a_to_b_white)
            dec_a_to_b_src = model.decode(z_a_to_b_src)
            dec_a_to_b_tgt = model.decode(z_a_to_b_tgt)

            dec_b_to_a_white = model.decode(z_b_to_a_white)
            dec_b_to_a_src = model.decode(z_b_to_a_src)
            dec_b_to_a_tgt = model.decode(z_b_to_a_tgt)

            # Pixel-space warps (image warped directly, not through latent)
            img_a_01 = img_a * 0.5 + 0.5  # (1, 3, H, W) in [0, 1]
            img_b_01 = img_b * 0.5 + 0.5

            def warp_pixels(src, warp, conf, bg):
                """Warp src pixels using warp field, blend with bg where low confidence."""
                warped = F.grid_sample(src, warp.unsqueeze(0), mode="bilinear", align_corners=False)
                c = conf.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                c = torch.clamp(c - args.conf_threshold, min=0) / (1 - args.conf_threshold + 1e-8)
                return c * warped + (1 - c) * bg

            pix_b_to_a_white = warp_pixels(img_b_01, warp_ab, conf_ab, torch.ones_like(img_a_01))
            pix_a_to_b_white = warp_pixels(img_a_01, warp_ba, conf_ba, torch.ones_like(img_b_01))
            pix_b_to_a_tgt   = warp_pixels(img_b_01, warp_ab, conf_ab, img_a_01)
            pix_a_to_b_tgt   = warp_pixels(img_a_01, warp_ba, conf_ba, img_b_01)

        def to_np(t):
            """Convert (1, 3, H, W) tensor in [-1, 1] to (H, W, 3) numpy in [0, 1]."""
            return t[0].permute(1, 2, 0).cpu().clamp(-1, 1).mul(0.5).add(0.5).numpy()

        def to_np_01(t):
            """Convert (1, 3, H, W) tensor in [0, 1] to (H, W, 3) numpy."""
            return t[0].permute(1, 2, 0).cpu().clamp(0, 1).numpy()

        def latent_to_np(z):
            """Convert (1, C, Hl, Wl) latent to (Hl, Wl, 3) RGB visualization.
            Uses first 3 channels, normalized per-image to [0, 1]."""
            vis = z[0, :3].cpu()  # (3, Hl, Wl)
            lo, hi = vis.min(), vis.max()
            if hi - lo > 1e-6:
                vis = (vis - lo) / (hi - lo)
            else:
                vis = vis * 0
            return vis.permute(1, 2, 0).numpy()

        def save_grid(rows, title, path):
            fig, axes = plt.subplots(len(rows), 2, figsize=(10, 5 * len(rows)))
            if len(rows) == 1:
                axes = axes[np.newaxis, :]
            for r, (left_img, left_title, right_img, right_title) in enumerate(rows):
                axes[r, 0].imshow(left_img)
                axes[r, 0].set_title(left_title, fontsize=13, fontweight="bold")
                axes[r, 1].imshow(right_img)
                axes[r, 1].set_title(right_title, fontsize=13, fontweight="bold")
            for ax in axes.flat:
                ax.axis("off")
            plt.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  Saved {path}")

        conf_ab_mean = conf_ab.mean().item()
        conf_ba_mean = conf_ba.mean().item()
        conf_str = f"conf A→B: {conf_ab_mean:.1%}, B→A: {conf_ba_mean:.1%}"

        # Derive sequence id from the source sample's filepath
        # (e.g. "hydrant/106_12687_26288/images/frame000001.jpg" → "106_12687_26288")
        idx_a = sample["index"]
        src_filepath = dataset.samples[idx_a]["filepath"]
        seq_id = Path(src_filepath).parts[1]

        # Output goes into a per-sequence subdirectory
        base = Path(args.output)
        out_dir = base.parent / seq_id
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = base.stem
        suffix = base.suffix

        # 1) Reconstruction figure: originals + reconstructions + latents
        save_grid(
            [
                (to_np(img_a), "Image A", to_np(img_b), "Image B"),
                (to_np(recon_a), "Recon A", to_np(recon_b), "Recon B"),
                (latent_to_np(z_a), "Latent A", latent_to_np(z_b), "Latent B"),
            ],
            f"Reconstruction — pair {pair_idx}  ({conf_str})",
            out_dir / f"{stem}_recon{suffix}",
        )

        # 2) Pixel-space warps figure
        save_grid(
            [
                (to_np(img_a), "Image A", to_np(img_b), "Image B"),
                (to_np_01(pix_b_to_a_white), "B→A pixel warp (white bg)", to_np_01(pix_a_to_b_white), "A→B pixel warp (white bg)"),
                (to_np_01(pix_b_to_a_tgt), "B→A pixel warp (target bg)", to_np_01(pix_a_to_b_tgt), "A→B pixel warp (target bg)"),
            ],
            f"Pixel Warps — pair {pair_idx}  ({conf_str})",
            out_dir / f"{stem}_pixwarp{suffix}",
        )

        # 3) Latent warp figures — one per background variant
        for bg_name, dec_b2a, dec_a2b, z_b2a, z_a2b in [
            ("zero",   dec_b_to_a_white, dec_a_to_b_white, z_b_to_a_white, z_a_to_b_white),
            ("source", dec_b_to_a_src,   dec_a_to_b_src,   z_b_to_a_src,   z_a_to_b_src),
            ("target", dec_b_to_a_tgt,   dec_a_to_b_tgt,   z_b_to_a_tgt,   z_a_to_b_tgt),
        ]:
            save_grid(
                [
                    (to_np(img_a), "Image A", to_np(img_b), "Image B"),
                    (latent_to_np(z_b2a), f"Latent B→A ({bg_name} bg)", latent_to_np(z_a2b), f"Latent A→B ({bg_name} bg)"),
                    (to_np(dec_b2a), f"Decoded B→A ({bg_name} bg)", to_np(dec_a2b), f"Decoded A→B ({bg_name} bg)"),
                ],
                f"Latent Warp ({bg_name} bg) — pair {pair_idx}  ({conf_str})",
                out_dir / f"{stem}_latwarp_{bg_name}{suffix}",
            )


if __name__ == "__main__":
    main()
