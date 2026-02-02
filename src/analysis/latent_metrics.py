"""Latent space similarity and consistency metrics."""

from typing import Dict, List
import torch
import torch.nn.functional as F
import numpy as np


def compute_latent_similarity(latent1: torch.Tensor, latent2: torch.Tensor) -> Dict[str, float]:
    """Compute various similarity metrics between two latent representations.

    Args:
        latent1, latent2: Latent tensors of shape (1, C, H, W)

    Returns:
        Dictionary of similarity metrics
    """
    flat1 = latent1.flatten()
    flat2 = latent2.flatten()

    assert latent1.shape == latent2.shape, "Latent shapes must match otherwise distance does not make sense."

    # MSE and MAE
    mse = F.mse_loss(latent1, latent2).item()
    mae = F.l1_loss(latent1, latent2).item()

    # Cosine similarity (global)
    cos_sim = F.cosine_similarity(
        flat1.unsqueeze(0), 
        flat2.unsqueeze(0)
    ).item()

    # Per-channel cosine similarity
    C = latent1.shape[1]
    channel_cos_sims = []
    for c in range(C):
        ch1 = latent1[0, c].flatten()
        ch2 = latent2[0, c].flatten()
        ch_cos = F.cosine_similarity(ch1.unsqueeze(0), ch2.unsqueeze(0)).item()
        channel_cos_sims.append(ch_cos)

    # Correlation coefficient
    mean1 = flat1.mean()
    mean2 = flat2.mean()
    std1 = flat1.std()
    std2 = flat2.std()
    correlation = ((flat1 - mean1) * (flat2 - mean2)).mean() / (std1 * std2 + 1e-8)
    correlation = correlation.item()

    # PSNR (treating latents as signals)
    max_val = max(latent1.abs().max().item(), latent2.abs().max().item())
    psnr = 10 * np.log10((max_val ** 2) / (mse + 1e-8))

    return {
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cos_sim,
        "correlation": correlation,
        "psnr": psnr,
        "channel_cosine_sims": channel_cos_sims,
    }


def compute_sequence_consistency(latents: List[torch.Tensor]) -> Dict[str, float]:
    """Compute consistency metrics for a sequence of latents.

    Args:
        latents: List of latent tensors, each (1, C, H, W)

    Returns:
        Dictionary with sequence consistency metrics
    """
    n = len(latents)
    if n < 2:
        return {}

    # Pairwise similarities (across all pairs)
    pairwise_cos = []
    pairwise_mse = []
    for i in range(n):
        for j in range(i + 1, n):
            flat_i = latents[i].flatten()
            flat_j = latents[j].flatten()
            cos = F.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0)).item()
            mse = F.mse_loss(latents[i], latents[j]).item()
            pairwise_cos.append(cos)
            pairwise_mse.append(mse)

    # Consecutive similarities (for sequence order)
    consecutive_cos = []
    consecutive_mse = []
    for i in range(n - 1):
        flat_i = latents[i].flatten()
        flat_j = latents[i + 1].flatten()
        cos = F.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0)).item()
        mse = F.mse_loss(latents[i], latents[i + 1]).item()
        consecutive_cos.append(cos)
        consecutive_mse.append(mse)

    # Compute mean latent and variance
    stacked = torch.cat(latents, dim=0)  # (N, C, H, W)
    mean_latent = stacked.mean(dim=0, keepdim=True)  # (1, C, H, W)
    variance = ((stacked - mean_latent) ** 2).mean().item()

    return {
        "num_views": n,
        "pairwise_cos_mean": np.mean(pairwise_cos),
        "pairwise_cos_std": np.std(pairwise_cos),
        "pairwise_cos_min": min(pairwise_cos),
        "pairwise_mse_mean": np.mean(pairwise_mse),
        "consecutive_cos_mean": np.mean(consecutive_cos),
        "consecutive_mse_mean": np.mean(consecutive_mse),
        "latent_variance": variance,
    }


def compute_latent_stats(latents: torch.Tensor, name: str) -> Dict:
    """Compute statistics for latent representations.

    Args:
        latents: Tensor of shape (N, C, H, W)
        name: Name identifier for the statistics

    Returns:
        Dictionary of statistics
    """
    return {
        "name": name,
        "shape": list(latents.shape),
        "mean": latents.mean().item(),
        "std": latents.std().item(),
        "min": latents.min().item(),
        "max": latents.max().item(),
        "channel_means": latents.mean(dim=(0, 2, 3)).tolist(),
        "channel_stds": latents.std(dim=(0, 2, 3)).tolist(),
    }


def print_latent_stats(stats: Dict) -> None:
    """Pretty print latent statistics."""
    print(f"\n  {stats['name']}:")
    print(f"    Shape: {stats['shape']}")
    print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    print(f"    Channel means: {[f'{x:.3f}' for x in stats['channel_means']]}")
    print(f"    Channel stds: {[f'{x:.3f}' for x in stats['channel_stds']]}")


def save_stats_to_file(stats_list: List[Dict], save_path: str) -> None:
    """Save latent statistics to a text file."""
    with open(save_path, 'w') as f:
        for stats in stats_list:
            f.write(f"{stats['name']}:\n")
            f.write(f"  Shape: {stats['shape']}\n")
            f.write(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}\n")
            f.write(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}\n")
            f.write(f"  Channel means: {[f'{x:.3f}' for x in stats['channel_means']]}\n")
            f.write(f"  Channel stds: {[f'{x:.3f}' for x in stats['channel_stds']]}\n")
            f.write("\n")
    print(f"  Saved {save_path}")
