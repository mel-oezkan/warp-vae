#!/usr/bin/env python
"""Plot reconstruction grids for warp VAE models across different CO3D objects.

Creates a figure with rows = objects, columns showing original + reconstructions
from each model, to illustrate reconstruction artifacts/patterns.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/plot_warp_reconstructions.py
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.model_utils import load_model, encode_images, decode_latents, denormalize

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


# --- Config ---
DEVICE = "cuda"
IMAGE_SIZE = 256

MODELS = {
    "Warp-VAE": {
        "checkpoint": "checkpoints/stereotyped-tireless-starfish-of-fame_hydrant 50seq cropped, from scratch, warp_w=0.02, warp_recon_w=0.02, disc_w=0.5, l1 warp consistency/last.ckpt",
        "config": "config/warp_vae_hydrant_recon_crop_l1.yaml",
    },
}

CO3D_DIR = Path("/visinf/projects_students/dlcv2025_groupZ/co3d_data")

# 3 diverse categories, 1 image each (columns = objects)
CATEGORIES = ["apple", "hydrant", "toaster"]
IMAGES_PER_CATEGORY = 1


def find_images(category_dir, n=2):
    """Find n random images from a CO3D category."""
    all_imgs = []
    for seq_dir in sorted(category_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        img_dir = seq_dir / "images"
        if not img_dir.exists():
            continue
        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        all_imgs.extend(imgs)

    if len(all_imgs) == 0:
        return []

    # Filter out black/near-black images
    filtered = []
    for p in all_imgs:
        img = Image.open(p).convert("RGB")
        if np.array(img).mean() > 10:
            filtered.append(p)
    all_imgs = filtered

    if len(all_imgs) == 0:
        return []

    rng = np.random.RandomState(123)
    indices = rng.choice(len(all_imgs), min(n, len(all_imgs)), replace=False)
    return [all_imgs[i] for i in indices]


def load_and_preprocess(image_paths):
    """Load images and return tensor in [-1, 1]."""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tensors = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img))
    return torch.stack(tensors)


@torch.no_grad()
def reconstruct(model, model_type, images):
    """Encode and decode images."""
    latents = encode_images(model, images, DEVICE, model_type)
    recons = decode_latents(model, latents, DEVICE, model_type)
    return recons.cpu()


def main():
    # Collect images per category
    print("Collecting images...")
    category_images = {}
    valid_categories = []
    for cat in CATEGORIES:
        cat_dir = CO3D_DIR / cat
        if not cat_dir.exists():
            print(f"  Skipping {cat} (not found)")
            continue
        paths = find_images(cat_dir, IMAGES_PER_CATEGORY)
        if len(paths) == 0:
            print(f"  Skipping {cat} (no images)")
            continue
        category_images[cat] = paths
        valid_categories.append(cat)
        print(f"  {cat}: {len(paths)} images")

    if not valid_categories:
        print("No valid categories found!")
        return

    # Load all images
    all_paths = []
    for cat in valid_categories:
        all_paths.extend(category_images[cat])
    all_images = load_and_preprocess(all_paths)

    # Load models and reconstruct
    model_recons = {}
    for name, cfg in MODELS.items():
        print(f"\nLoading {name}...")
        model, mtype = load_model(cfg["checkpoint"], cfg["config"])
        model = model.to(DEVICE).eval()
        recons = reconstruct(model, mtype, all_images)
        model_recons[name] = recons
        del model
        torch.cuda.empty_cache()

    # Plot grid: rows = [Source, Recon1, Recon2], cols = objects (categories)
    n_objects = len(valid_categories)
    model_names = list(MODELS.keys())
    n_rows = 1 + len(model_names)  # Source + one row per model
    row_labels = ["Source"] + model_names

    fig, axes = plt.subplots(n_rows, n_objects, figsize=(3.5 * n_objects, 3.5 * n_rows))
    if n_objects == 1:
        axes = axes[:, np.newaxis]

    for col in range(n_objects):
        # Source row
        orig = denormalize(all_images[col]).permute(1, 2, 0).numpy()
        axes[0, col].imshow(orig)
        axes[0, col].set_title(valid_categories[col], fontsize=13, fontweight='bold')

        # Reconstruction rows
        for row_idx, name in enumerate(model_names, 1):
            recon = denormalize(model_recons[name][col]).permute(1, 2, 0).numpy()
            axes[row_idx, col].imshow(recon)

    # Row labels
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, fontweight='bold', rotation=90, labelpad=10)

    # Clean up axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = "/visinf/projects_students/dlcv2025_groupZ/warp_vae_reconstruction_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
