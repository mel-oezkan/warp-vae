"""Visualization utilities for latent analysis."""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .model_utils import denormalize


def latent_to_pca_rgb(
    latent: torch.Tensor,
    pca_model: Optional[PCA] = None
) -> Tuple[np.ndarray, PCA]:
    """Convert latent tensor to RGB image using PCA.

    Args:
        latent: Tensor of shape (1, C, H, W) or (C, H, W)
        pca_model: Optional fitted PCA model. If None, fits a new one.

    Returns:
        RGB image as numpy array (H, W, 3), normalized to [0, 1]
        Fitted PCA model
    """
    if latent.dim() == 4:
        latent = latent[0]

    C, H, W = latent.shape
    lat_flat = latent.cpu().numpy().reshape(C, -1).T  # (H*W, C)

    if pca_model is None:
        pca_model = PCA(n_components=3)
        pca_model.fit(lat_flat)

    lat_pca = pca_model.transform(lat_flat)  # (H*W, 3)
    lat_rgb = lat_pca.reshape(H, W, 3)

    # Normalize each channel independently using percentiles to avoid outlier dominance
    for c in range(3):
        channel = lat_rgb[..., c]
        vmin, vmax = np.percentile(channel, [2, 98])
        lat_rgb[..., c] = np.clip((channel - vmin) / (vmax - vmin + 1e-8), 0, 1)

    return lat_rgb, pca_model


def visualize_reconstructions(data: dict, save_path: str, n_samples: int = 10) -> None:
    """Create reconstruction grid: original vs reconstructed.

    Args:
        data: Dictionary with 'images' and 'reconstructions' tensors
        save_path: Path to save the figure
        n_samples: Number of samples to display
    """
    images = data['images']
    recons = data['reconstructions']

    n_available = min(n_samples, images.shape[0])

    fig, axes = plt.subplots(2, n_available, figsize=(n_available * 2.5, 5))

    if n_available == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_available):
        # Original
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)

        # Reconstruction
        rec = denormalize(recons[i]).permute(1, 2, 0).numpy()
        rec = np.clip(rec, 0, 1)
        axes[1, i].imshow(rec)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Recon', fontsize=12)

    # Compute MSE for title
    mse = F.mse_loss(recons[:n_available], images[:n_available]).item()

    plt.suptitle(f'Reconstructions (MSE: {mse:.4f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def visualize_latent_channels(data: dict, save_path: str, n_samples: int = 5) -> None:
    """Visualize individual latent channels for random samples.

    Creates as many rows as there are channels in the latent representation.

    Args:
        data: Dictionary with 'images' and 'latents' tensors
        save_path: Path to save the figure
        n_samples: Number of samples to display
    """
    images = data['images']
    latents = data['latents'].numpy()

    n_channels = latents.shape[1]
    n_available = min(n_samples, images.shape[0])

    n_rows = 1 + n_channels

    fig, axes = plt.subplots(n_rows, n_available, figsize=(n_available * 2.5, n_rows * 2))

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_available == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_available):
        # Row 0: Input image
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Sample {i}', fontsize=10)
        axes[0, i].axis('off')

        # Rows 1 to n_channels: Individual latent channels
        for c in range(n_channels):
            lat_channel = latents[i, c]
            vmin, vmax = np.percentile(lat_channel, [2, 98])
            lat_norm = np.clip((lat_channel - vmin) / (vmax - vmin + 1e-8), 0, 1)
            axes[c + 1, i].imshow(lat_norm, cmap='viridis')
            axes[c + 1, i].axis('off')

    # Add y-labels
    axes[0, 0].set_ylabel('Input', fontsize=10)
    for c in range(n_channels):
        axes[c + 1, 0].set_ylabel(f'Ch {c}', fontsize=10)

    for row in range(n_rows):
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].yaxis.label.set_visible(True)

    plt.suptitle(f'Latent Channels (C={n_channels})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def visualize_latent_pca(data: dict, save_path: str, n_samples: int = 5) -> None:
    """Visualize PCA of latents for random samples.

    Args:
        data: Dictionary with 'images' and 'latents' tensors
        save_path: Path to save the figure
        n_samples: Number of samples to display
    """
    images = data['images']
    latents = data['latents'].numpy()

    n_channels = latents.shape[1]
    n_available = min(n_samples, images.shape[0])
    _, _, h, w = latents.shape

    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_available, figsize=(n_available * 2.5, n_rows * 2.5))

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_available == 1:
        axes = axes.reshape(-1, 1)

    explained_variances = []

    for i in range(n_available):
        # Row 0: Input image
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Sample {i}', fontsize=10)
        axes[0, i].axis('off')

        # Row 1: Raw latent (first 3 channels as RGB)
        lat_raw = latents[i][:min(3, n_channels)]
        if lat_raw.shape[0] < 3:
            lat_raw = np.concatenate([lat_raw, np.zeros((3 - lat_raw.shape[0], h, w))], axis=0)
        lat_raw = np.transpose(lat_raw, (1, 2, 0))
        lat_raw = (lat_raw - lat_raw.min()) / (lat_raw.max() - lat_raw.min() + 1e-8)
        axes[1, i].imshow(lat_raw)
        axes[1, i].axis('off')

        # Row 2: PCA latent
        lat_single = latents[i]
        lat_flat = lat_single.transpose(1, 2, 0).reshape(-1, n_channels)

        pca = PCA(n_components=min(3, n_channels))
        lat_pca_flat = pca.fit_transform(lat_flat)
        explained_variances.append(pca.explained_variance_ratio_)

        if lat_pca_flat.shape[1] < 3:
            lat_pca_flat = np.concatenate([
                lat_pca_flat,
                np.zeros((lat_pca_flat.shape[0], 3 - lat_pca_flat.shape[1]))
            ], axis=1)

        lat_pca = lat_pca_flat.reshape(h, w, 3)
        lat_pca_norm = np.zeros_like(lat_pca)
        for c in range(3):
            channel = lat_pca[..., c]
            vmin, vmax = np.percentile(channel, [2, 98])
            lat_pca_norm[..., c] = np.clip((channel - vmin) / (vmax - vmin + 1e-8), 0, 1)

        axes[2, i].imshow(lat_pca_norm)
        axes[2, i].axis('off')

    # Add y-labels
    axes[0, 0].set_ylabel('Input', fontsize=10)
    axes[1, 0].set_ylabel('Latent', fontsize=10)
    axes[2, 0].set_ylabel('PCA', fontsize=10)

    for row in range(n_rows):
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].yaxis.label.set_visible(True)

    # Compute average explained variance
    if explained_variances:
        avg_var = np.mean(explained_variances, axis=0)
        n_pcs = len(avg_var)
        var_parts = [f"PC{i+1}={avg_var[i]:.1%}" for i in range(min(3, n_pcs))]
        var_text = f"Avg PCA: {', '.join(var_parts)}"
    else:
        var_text = ""

    plt.suptitle(f'Latent PCA\n{var_text}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")
