#!/usr/bin/env python
"""Plot source vs reconstruction examples from the OmniObject3D Warp-VAE run.

Creates two figures:
1. Single sequence: multiple views of one object (source vs reconstruction)
2. Multi-object: one view per object across diverse categories (source vs reconstruction)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/plot_omniobject_reconstructions.py
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

CHECKPOINT = "checkpoints/outgoing-amiable-bird-of-novelty_omniobject3d, RoMA warps, cosine consistency, no background/last.ckpt"
CONFIG = "config/warp_vae_omniobject.yaml"

OMNI_IMG_DIR = Path("/data/lab_moezkan/omni_obj/blender_renders_24_views/img")

# For single-sequence plot: pick a visually interesting object
SINGLE_SEQUENCE = "guitar_001"
SINGLE_SEQUENCE_VIEWS = [0, 3, 6, 9, 12, 15, 18, 21]  # 8 evenly spaced views

# For multi-object plot: diverse categories
MULTI_OBJECTS = [
    "apple_001",
    "backpack_001",
    "chair_001",
    "dinosaur_001",
    "helmet_001",
    "shoe_001",
    "teapot_001",
    "vase_001",
]

OUTPUT_DIR = Path("/visinf/projects_students/dlcv2025_groupZ")


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
def reconstruct(model, model_type, images, batch_size=2):
    """Encode and decode images in small batches to avoid OOM."""
    all_recons = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        latents = encode_images(model, batch, DEVICE, model_type)
        recons = decode_latents(model, latents, DEVICE, model_type)
        all_recons.append(recons.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_recons, dim=0)


def get_sequence_paths(obj_name, view_indices):
    """Get image paths for specific views of an object."""
    obj_dir = OMNI_IMG_DIR / obj_name
    paths = []
    for idx in view_indices:
        p = obj_dir / f"{idx:03d}.png"
        if p.exists():
            paths.append(p)
    return paths


def get_single_view(obj_name, view_idx=0):
    """Get a single view of an object."""
    p = OMNI_IMG_DIR / obj_name / f"{view_idx:03d}.png"
    return p if p.exists() else None


def plot_single_sequence(model, model_type):
    """Plot: one object, multiple views, source vs reconstruction."""
    paths = get_sequence_paths(SINGLE_SEQUENCE, SINGLE_SEQUENCE_VIEWS)
    if not paths:
        print(f"No images found for {SINGLE_SEQUENCE}")
        return

    images = load_and_preprocess(paths)
    recons = reconstruct(model, model_type, images)

    n_views = len(paths)
    fig, axes = plt.subplots(2, n_views, figsize=(2.5 * n_views, 5.5))

    for col in range(n_views):
        # Source
        orig = denormalize(images[col]).permute(1, 2, 0).numpy()
        axes[0, col].imshow(orig)
        axes[0, col].set_title(f"View {SINGLE_SEQUENCE_VIEWS[col]}", fontsize=10)

        # Reconstruction
        rec = denormalize(recons[col]).permute(1, 2, 0).numpy()
        axes[1, col].imshow(rec)

    axes[0, 0].set_ylabel("Source", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Reconstruction", fontsize=13, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    obj_label = SINGLE_SEQUENCE.replace("_", " ").title()
    fig.suptitle(f"Warp-VAE Reconstruction — {obj_label} (multiple views)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / "omniobject_single_sequence_recon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved single-sequence plot to {out}")


def plot_multi_object(model, model_type):
    """Plot: multiple objects, one view each, source vs reconstruction."""
    paths = []
    valid_names = []
    for obj_name in MULTI_OBJECTS:
        p = get_single_view(obj_name, view_idx=0)
        if p is not None:
            paths.append(p)
            valid_names.append(obj_name)
        else:
            print(f"  Skipping {obj_name} (not found)")

    if not paths:
        print("No valid objects found!")
        return

    images = load_and_preprocess(paths)
    recons = reconstruct(model, model_type, images)

    n_obj = len(paths)
    fig, axes = plt.subplots(2, n_obj, figsize=(2.5 * n_obj, 5.5))
    if n_obj == 1:
        axes = axes[:, np.newaxis]

    for col in range(n_obj):
        orig = denormalize(images[col]).permute(1, 2, 0).numpy()
        axes[0, col].imshow(orig)
        label = valid_names[col].rsplit("_", 1)[0].replace("_", " ").title()
        axes[0, col].set_title(label, fontsize=10, fontweight="bold")

        rec = denormalize(recons[col]).permute(1, 2, 0).numpy()
        axes[1, col].imshow(rec)

    axes[0, 0].set_ylabel("Source", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Reconstruction", fontsize=13, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Warp-VAE Reconstruction — Multiple Objects", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = OUTPUT_DIR / "omniobject_multi_object_recon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-object plot to {out}")


def main():
    # Verify chosen objects exist
    print("Checking object directories...")
    for obj in [SINGLE_SEQUENCE] + MULTI_OBJECTS:
        d = OMNI_IMG_DIR / obj
        if not d.exists():
            print(f"  WARNING: {obj} not found at {d}")

    # Load model
    print(f"\nLoading model from {CHECKPOINT}...")
    model, model_type = load_model(CHECKPOINT, CONFIG)
    model = model.to(DEVICE).eval()
    print(f"  Model type: {model_type}")

    # Generate plots
    print("\n--- Single Sequence Plot ---")
    plot_single_sequence(model, model_type)

    print("\n--- Multi-Object Plot ---")
    plot_multi_object(model, model_type)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
