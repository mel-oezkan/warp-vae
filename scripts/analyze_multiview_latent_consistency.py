#!/usr/bin/env python
"""
Analyze latent consistency for overlapping multi-view images in OmniObject3D.

This script:
1. Loads a VAE model
2. For each object, identifies view pairs with high overlap (nearby camera positions)
3. Encodes both views to latent space
4. Computes similarity metrics between latent representations
5. Visualizes the relationship between camera overlap and latent similarity

Usage:
    python scripts/analyze_multiview_latent_consistency.py \
        --checkpoint outputs/my_model/checkpoints/last.ckpt \
        --config config/my_config.yaml \
        --output_name multiview_analysis
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Import model loading utilities from compare_latents
from compare_latents import load_model, encode_images, denormalize

# Default paths for f8 baseline VAE
F8_BASELINE_CHECKPOINT = "weights/f8/model.ckpt"
F8_BASELINE_CONFIG = "config/baseVAE.yaml"


def load_f8_baseline_vae(device="cuda"):
    """Load the f8 SD-VAE baseline model.

    This loads the Stable Diffusion VAE with f=8 downsampling factor,
    which matches the architecture used in this project's models.
    Uses ch_mult=[1,2,4,4] giving 3 downsampling stages (256 -> 32).
    """
    print(f"Loading f8 baseline VAE from {F8_BASELINE_CHECKPOINT}")
    model, model_type = load_model(
        checkpoint_path=F8_BASELINE_CHECKPOINT,
        config_path=F8_BASELINE_CONFIG,
        model_type="ldm"
    )
    model = model.to(device)
    model.eval()
    print("Loaded f8 baseline VAE (SD 2.x compatible)")
    return model, model_type


def load_camera_data(transforms_path):
    """Load camera transforms from JSON file."""
    with open(transforms_path) as f:
        data = json.load(f)
    return data


def extract_camera_positions(camera_data):
    """Extract camera positions from transform matrices."""
    positions = []
    for frame in camera_data["frames"]:
        transform = np.array(frame["transform_matrix"])
        position = transform[:3, 3]
        positions.append(position)
    return np.array(positions)


def compute_angular_separation(positions):
    """Compute angular separation matrix between all camera positions."""
    n_views = len(positions)
    angular_sep = np.zeros((n_views, n_views))

    for i in range(n_views):
        for j in range(n_views):
            if i == j:
                angular_sep[i, j] = 0
            else:
                dot = np.dot(positions[i], positions[j])
                norm_prod = np.linalg.norm(positions[i]) * np.linalg.norm(positions[j])
                cos_angle = np.clip(dot / (norm_prod + 1e-8), -1, 1)
                angular_sep[i, j] = np.arccos(cos_angle) * 180 / np.pi

    return angular_sep


def find_overlapping_pairs(angular_sep, max_angle=30, min_angle=5):
    """Find view pairs with angular separation in specified range.

    Args:
        angular_sep: (N, N) angular separation matrix in degrees
        max_angle: Maximum angular separation to consider as "overlapping"
        min_angle: Minimum angular separation (to avoid nearly identical views)

    Returns:
        List of (i, j, angle) tuples
    """
    n = angular_sep.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            angle = angular_sep[i, j]
            if min_angle <= angle <= max_angle:
                pairs.append((i, j, angle))
    return sorted(pairs, key=lambda x: x[2])


def find_view_sequences(positions, angular_sep, seq_length=3, max_pairwise_angle=30):
    """Find sequences of views that form a coherent sweep around the object.

    Args:
        positions: (N, 3) camera positions
        angular_sep: (N, N) angular separation matrix in degrees
        seq_length: Number of views in each sequence
        max_pairwise_angle: Maximum angle between consecutive views in sequence

    Returns:
        List of tuples (view_indices, total_span_angle, avg_step_angle)
    """
    n = len(positions)
    sequences = []

    # For each starting view, try to build a sequence
    for start_idx in range(n):
        # Sort other views by angular distance from start
        distances = [(i, angular_sep[start_idx, i]) for i in range(n) if i != start_idx]
        distances.sort(key=lambda x: x[1])

        # Try to build sequence greedily
        sequence = [start_idx]
        current_idx = start_idx

        for _ in range(seq_length - 1):
            # Find nearest view not yet in sequence
            best_next = None
            best_angle = float('inf')

            for idx, _ in distances:
                if idx not in sequence:
                    angle = angular_sep[current_idx, idx]
                    if angle <= max_pairwise_angle and angle < best_angle:
                        best_next = idx
                        best_angle = angle

            if best_next is not None:
                sequence.append(best_next)
                current_idx = best_next
            else:
                break

        if len(sequence) == seq_length:
            # Compute total span (angle between first and last)
            total_span = angular_sep[sequence[0], sequence[-1]]

            # Compute average step angle
            step_angles = [angular_sep[sequence[i], sequence[i+1]]
                          for i in range(len(sequence)-1)]
            avg_step = np.mean(step_angles)

            sequences.append((tuple(sequence), total_span, avg_step))

    # Remove duplicate sequences (same views, different order)
    unique_sequences = []
    seen_sets = set()
    for seq, span, avg in sequences:
        seq_set = frozenset(seq)
        if seq_set not in seen_sets:
            seen_sets.add(seq_set)
            unique_sequences.append((seq, span, avg))

    return unique_sequences


def compute_sequence_consistency(latents):
    """Compute consistency metrics for a sequence of latents.

    Args:
        latents: List of latent tensors, each (1, C, H, W)

    Returns:
        Dictionary with sequence consistency metrics
    """
    n = len(latents)
    if n < 2:
        return {}

    # Pairwise similarities
    pairwise_cos = []
    pairwise_mse = []
    for i in range(n):
        for j in range(i+1, n):
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
        flat_j = latents[i+1].flatten()
        cos = F.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0)).item()
        mse = F.mse_loss(latents[i], latents[i+1]).item()
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


def load_view_image(obj_dir, view_idx, transform, device):
    """Load and preprocess a single view image."""
    img_path = obj_dir / f"{view_idx:03d}.png"
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def compute_latent_similarity(latent1, latent2):
    """Compute various similarity metrics between two latent representations.

    Args:
        latent1, latent2: Latent tensors of shape (1, C, H, W)

    Returns:
        Dictionary of similarity metrics
    """
    # Flatten for vector comparisons
    flat1 = latent1.flatten()
    flat2 = latent2.flatten()

    # MSE
    mse = F.mse_loss(latent1, latent2).item()

    # MAE
    mae = F.l1_loss(latent1, latent2).item()

    # Cosine similarity (global)
    cos_sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()

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


@torch.no_grad()
def analyze_object(model, obj_dir, transform, device, model_type,
                   max_angle=45, min_angle=2, max_pairs=50):
    """Analyze latent consistency for a single object.

    Args:
        model: VAE model
        obj_dir: Path to object directory
        transform: Image transform
        device: Torch device
        model_type: Model type string
        max_angle: Maximum angular separation to analyze
        min_angle: Minimum angular separation
        max_pairs: Maximum number of pairs to analyze

    Returns:
        List of dictionaries with pair info and similarity metrics
    """
    transforms_path = obj_dir / "transforms.json"
    if not transforms_path.exists():
        return []

    # Load camera data
    camera_data = load_camera_data(transforms_path)
    positions = extract_camera_positions(camera_data)

    # Compute angular separations
    angular_sep = compute_angular_separation(positions)

    # Find overlapping pairs
    pairs = find_overlapping_pairs(angular_sep, max_angle=max_angle, min_angle=min_angle)

    if len(pairs) > max_pairs:
        # Sample pairs across the angle range
        pairs = pairs[::len(pairs)//max_pairs][:max_pairs]

    results = []
    for view1_idx, view2_idx, angle in pairs:
        # Load images
        img1 = load_view_image(obj_dir, view1_idx, transform, device)
        img2 = load_view_image(obj_dir, view2_idx, transform, device)

        # Encode to latent space
        latent1 = encode_images(model, img1, device, model_type)
        latent2 = encode_images(model, img2, device, model_type)

        # Compute similarity
        similarity = compute_latent_similarity(latent1, latent2)

        results.append({
            "view1_idx": view1_idx,
            "view2_idx": view2_idx,
            "angular_separation": angle,
            **similarity
        })

    return results


@torch.no_grad()
def analyze_object_sequences(model, obj_dir, transform, device, model_type,
                             seq_lengths=[3, 4, 5], max_pairwise_angle=25,
                             max_sequences_per_length=10):
    """Analyze latent consistency for view sequences of varying lengths.

    Args:
        model: VAE model
        obj_dir: Path to object directory
        transform: Image transform
        device: Torch device
        model_type: Model type string
        seq_lengths: List of sequence lengths to analyze
        max_pairwise_angle: Max angle between consecutive views
        max_sequences_per_length: Max sequences to analyze per length

    Returns:
        Dictionary with sequence analysis results
    """
    transforms_path = obj_dir / "transforms.json"
    if not transforms_path.exists():
        return {}

    camera_data = load_camera_data(transforms_path)
    positions = extract_camera_positions(camera_data)
    angular_sep = compute_angular_separation(positions)

    results = {"object_name": obj_dir.name, "sequences": {}}

    for seq_len in seq_lengths:
        sequences = find_view_sequences(
            positions, angular_sep,
            seq_length=seq_len,
            max_pairwise_angle=max_pairwise_angle
        )

        if len(sequences) > max_sequences_per_length:
            # Sample diverse sequences
            sequences = sequences[::len(sequences)//max_sequences_per_length][:max_sequences_per_length]

        seq_results = []
        for view_indices, total_span, avg_step in sequences:
            # Load and encode all views in sequence
            latents = []
            for view_idx in view_indices:
                img = load_view_image(obj_dir, view_idx, transform, device)
                latent = encode_images(model, img, device, model_type)
                latents.append(latent)

            # Compute sequence consistency
            consistency = compute_sequence_consistency(latents)
            consistency["view_indices"] = view_indices
            consistency["total_span_angle"] = total_span
            consistency["avg_step_angle"] = avg_step

            seq_results.append(consistency)

        results["sequences"][seq_len] = seq_results

    return results


@torch.no_grad()
def analyze_full_object(model, obj_dir, transform, device, model_type):
    """Analyze all 24 views of an object and return detailed per-object results.

    Returns latents for all views and pairwise similarity matrix.
    """
    transforms_path = obj_dir / "transforms.json"
    if not transforms_path.exists():
        return None

    camera_data = load_camera_data(transforms_path)
    positions = extract_camera_positions(camera_data)
    angular_sep = compute_angular_separation(positions)
    n_views = len(positions)

    # Encode all views
    latents = []
    images = []
    for view_idx in range(n_views):
        img = load_view_image(obj_dir, view_idx, transform, device)
        latent = encode_images(model, img, device, model_type)
        latents.append(latent)
        images.append(img)

    # Compute pairwise similarity matrix
    cos_sim_matrix = np.zeros((n_views, n_views))
    mse_matrix = np.zeros((n_views, n_views))

    for i in range(n_views):
        for j in range(n_views):
            if i == j:
                cos_sim_matrix[i, j] = 1.0
                mse_matrix[i, j] = 0.0
            else:
                flat_i = latents[i].flatten()
                flat_j = latents[j].flatten()
                cos_sim_matrix[i, j] = F.cosine_similarity(
                    flat_i.unsqueeze(0), flat_j.unsqueeze(0)
                ).item()
                mse_matrix[i, j] = F.mse_loss(latents[i], latents[j]).item()

    return {
        "object_name": obj_dir.name,
        "n_views": n_views,
        "latents": latents,
        "images": images,
        "positions": positions,
        "angular_sep": angular_sep,
        "cos_sim_matrix": cos_sim_matrix,
        "mse_matrix": mse_matrix,
    }


def visualize_results(all_results, output_dir, model_name):
    """Create visualizations of the analysis results."""
    if not all_results:
        print("No results to visualize")
        return

    # Extract data for plotting
    angles = [r["angular_separation"] for r in all_results]
    mse_values = [r["mse"] for r in all_results]
    cos_values = [r["cosine_similarity"] for r in all_results]
    corr_values = [r["correlation"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: MSE vs Angular Separation
    ax1 = axes[0, 0]
    ax1.scatter(angles, mse_values, alpha=0.5, s=20)
    ax1.set_xlabel("Angular Separation (degrees)")
    ax1.set_ylabel("Latent MSE")
    ax1.set_title("Latent MSE vs Camera Angular Separation")
    ax1.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(angles, mse_values, 2)
    p = np.poly1d(z)
    x_line = np.linspace(min(angles), max(angles), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label='Quadratic fit')
    ax1.legend()

    # Plot 2: Cosine Similarity vs Angular Separation
    ax2 = axes[0, 1]
    ax2.scatter(angles, cos_values, alpha=0.5, s=20, color='green')
    ax2.set_xlabel("Angular Separation (degrees)")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Latent Cosine Similarity vs Camera Angular Separation")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Add trend line
    z = np.polyfit(angles, cos_values, 2)
    p = np.poly1d(z)
    ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label='Quadratic fit')
    ax2.legend()

    # Plot 3: Correlation vs Angular Separation
    ax3 = axes[1, 0]
    ax3.scatter(angles, corr_values, alpha=0.5, s=20, color='orange')
    ax3.set_xlabel("Angular Separation (degrees)")
    ax3.set_ylabel("Correlation Coefficient")
    ax3.set_title("Latent Correlation vs Camera Angular Separation")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Histogram of similarities by angle bins
    ax4 = axes[1, 1]

    # Bin by angle
    angle_bins = [(0, 10), (10, 20), (20, 30), (30, 45)]
    bin_cos_means = []
    bin_cos_stds = []
    bin_labels = []

    for low, high in angle_bins:
        bin_cos = [r["cosine_similarity"] for r in all_results
                   if low <= r["angular_separation"] < high]
        if bin_cos:
            bin_cos_means.append(np.mean(bin_cos))
            bin_cos_stds.append(np.std(bin_cos))
            bin_labels.append(f"{low}-{high}°")

    if bin_cos_means:
        x_pos = np.arange(len(bin_labels))
        ax4.bar(x_pos, bin_cos_means, yerr=bin_cos_stds, capsize=5, alpha=0.7)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(bin_labels)
        ax4.set_xlabel("Angular Separation Range")
        ax4.set_ylabel("Mean Cosine Similarity")
        ax4.set_title("Latent Similarity by Angular Separation Bin")
        ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Multi-View Latent Consistency Analysis\nModel: {model_name}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "multiview_latent_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_dir / 'multiview_latent_analysis.png'}")


def visualize_example_pairs(model, obj_dir, transform, device, model_type,
                            output_dir, n_examples=4):
    """Visualize example view pairs with their latents."""
    transforms_path = obj_dir / "transforms.json"
    if not transforms_path.exists():
        return

    camera_data = load_camera_data(transforms_path)
    positions = extract_camera_positions(camera_data)
    angular_sep = compute_angular_separation(positions)

    # Get pairs at different angular separations
    example_pairs = []
    for target_angle in [5, 15, 30, 60]:
        best_pair = None
        best_diff = float('inf')
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                diff = abs(angular_sep[i, j] - target_angle)
                if diff < best_diff:
                    best_diff = diff
                    best_pair = (i, j, angular_sep[i, j])
        if best_pair:
            example_pairs.append(best_pair)

    n_pairs = len(example_pairs)
    fig, axes = plt.subplots(n_pairs, 5, figsize=(15, 3*n_pairs))

    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for row, (view1_idx, view2_idx, angle) in enumerate(example_pairs):
        # Load images
        img1 = load_view_image(obj_dir, view1_idx, transform, device)
        img2 = load_view_image(obj_dir, view2_idx, transform, device)

        # Encode
        latent1 = encode_images(model, img1, device, model_type)
        latent2 = encode_images(model, img2, device, model_type)

        # Compute similarity
        similarity = compute_latent_similarity(latent1, latent2)

        # Plot image 1
        img1_np = denormalize(img1[0]).permute(1, 2, 0).cpu().numpy()
        axes[row, 0].imshow(np.clip(img1_np, 0, 1))
        axes[row, 0].set_title(f"View {view1_idx}")
        axes[row, 0].axis('off')

        # Plot image 2
        img2_np = denormalize(img2[0]).permute(1, 2, 0).cpu().numpy()
        axes[row, 1].imshow(np.clip(img2_np, 0, 1))
        axes[row, 1].set_title(f"View {view2_idx}")
        axes[row, 1].axis('off')

        # Plot latent 1 (first 3 channels as RGB)
        lat1_vis = latent1[0, :3].cpu().numpy()
        lat1_vis = np.transpose(lat1_vis, (1, 2, 0))
        lat1_vis = (lat1_vis - lat1_vis.min()) / (lat1_vis.max() - lat1_vis.min() + 1e-8)
        axes[row, 2].imshow(lat1_vis)
        axes[row, 2].set_title("Latent 1")
        axes[row, 2].axis('off')

        # Plot latent 2
        lat2_vis = latent2[0, :3].cpu().numpy()
        lat2_vis = np.transpose(lat2_vis, (1, 2, 0))
        lat2_vis = (lat2_vis - lat2_vis.min()) / (lat2_vis.max() - lat2_vis.min() + 1e-8)
        axes[row, 3].imshow(lat2_vis)
        axes[row, 3].set_title("Latent 2")
        axes[row, 3].axis('off')

        # Plot latent difference
        lat_diff = (latent1 - latent2).abs()[0, :3].cpu().numpy()
        lat_diff = np.transpose(lat_diff, (1, 2, 0))
        lat_diff = (lat_diff - lat_diff.min()) / (lat_diff.max() - lat_diff.min() + 1e-8)
        axes[row, 4].imshow(lat_diff)
        axes[row, 4].set_title(f"|L1-L2|\nAngle: {angle:.1f}°\nCos: {similarity['cosine_similarity']:.3f}")
        axes[row, 4].axis('off')

    plt.suptitle("Example View Pairs with Latent Representations", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "example_view_pairs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved example pairs to {output_dir / 'example_view_pairs.png'}")


def visualize_sequence_analysis(seq_results_list, output_dir, model_name):
    """Visualize sequence consistency results."""
    if not seq_results_list:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Collect data by sequence length
    data_by_length = {}
    for obj_results in seq_results_list:
        for seq_len, sequences in obj_results.get("sequences", {}).items():
            if seq_len not in data_by_length:
                data_by_length[seq_len] = []
            data_by_length[seq_len].extend(sequences)

    seq_lengths = sorted(data_by_length.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(seq_lengths)))

    # Plot 1: Pairwise cosine similarity by sequence length
    ax1 = axes[0, 0]
    for idx, seq_len in enumerate(seq_lengths):
        cos_means = [s["pairwise_cos_mean"] for s in data_by_length[seq_len] if "pairwise_cos_mean" in s]
        spans = [s["total_span_angle"] for s in data_by_length[seq_len] if "total_span_angle" in s]
        if cos_means and spans:
            ax1.scatter(spans, cos_means, alpha=0.6, s=30, color=colors[idx],
                       label=f"{seq_len} views")
    ax1.set_xlabel("Total Span Angle (degrees)")
    ax1.set_ylabel("Mean Pairwise Cosine Similarity")
    ax1.set_title("Sequence Consistency vs Angular Span")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plot by sequence length
    ax2 = axes[0, 1]
    box_data = []
    box_labels = []
    for seq_len in seq_lengths:
        cos_means = [s["pairwise_cos_mean"] for s in data_by_length[seq_len] if "pairwise_cos_mean" in s]
        if cos_means:
            box_data.append(cos_means)
            box_labels.append(f"{seq_len} views")
    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Pairwise Cosine Similarity")
    ax2.set_title("Consistency Distribution by Sequence Length")
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Latent variance by sequence length
    ax3 = axes[1, 0]
    for idx, seq_len in enumerate(seq_lengths):
        variances = [s["latent_variance"] for s in data_by_length[seq_len] if "latent_variance" in s]
        spans = [s["total_span_angle"] for s in data_by_length[seq_len] if "total_span_angle" in s]
        if variances and spans:
            ax3.scatter(spans, variances, alpha=0.6, s=30, color=colors[idx],
                       label=f"{seq_len} views")
    ax3.set_xlabel("Total Span Angle (degrees)")
    ax3.set_ylabel("Latent Variance")
    ax3.set_title("Latent Variance vs Angular Span")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    headers = ["Seq Len", "N", "Cos Sim (mean±std)", "Variance (mean)"]
    for seq_len in seq_lengths:
        seqs = data_by_length[seq_len]
        n = len(seqs)
        cos_vals = [s["pairwise_cos_mean"] for s in seqs if "pairwise_cos_mean" in s]
        var_vals = [s["latent_variance"] for s in seqs if "latent_variance" in s]
        if cos_vals:
            table_data.append([
                f"{seq_len}",
                f"{n}",
                f"{np.mean(cos_vals):.3f}±{np.std(cos_vals):.3f}",
                f"{np.mean(var_vals):.3f}" if var_vals else "N/A"
            ])

    if table_data:
        table = ax4.table(cellText=table_data, colLabels=headers,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title("Summary Statistics by Sequence Length", pad=20)

    plt.suptitle(f"Multi-View Sequence Analysis\nModel: {model_name}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "sequence_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sequence analysis to {output_dir / 'sequence_analysis.png'}")


def visualize_per_object_results(object_results_list, output_dir, model_name, n_objects=5):
    """Visualize detailed results for individual objects."""
    if not object_results_list or len(object_results_list) < n_objects:
        n_objects = len(object_results_list)

    if n_objects == 0:
        return

    # Select n_objects evenly spaced through the list
    indices = np.linspace(0, len(object_results_list)-1, n_objects, dtype=int)
    selected_objects = [object_results_list[i] for i in indices]

    fig, axes = plt.subplots(n_objects, 4, figsize=(20, 4*n_objects))

    if n_objects == 1:
        axes = axes.reshape(1, -1)

    for row, obj_data in enumerate(selected_objects):
        obj_name = obj_data["object_name"]
        cos_matrix = obj_data["cos_sim_matrix"]
        mse_matrix = obj_data["mse_matrix"]
        angular_sep = obj_data["angular_sep"]
        n_views = obj_data["n_views"]

        # Plot 1: Angular separation matrix
        im1 = axes[row, 0].imshow(angular_sep, cmap='viridis', aspect='equal')
        axes[row, 0].set_title(f"{obj_name}\nAngular Separation")
        axes[row, 0].set_xlabel("View Index")
        axes[row, 0].set_ylabel("View Index")
        plt.colorbar(im1, ax=axes[row, 0], label='Degrees')

        # Plot 2: Cosine similarity matrix
        im2 = axes[row, 1].imshow(cos_matrix, cmap='RdYlGn', aspect='equal',
                                   vmin=0.5, vmax=1.0)
        axes[row, 1].set_title("Latent Cosine Similarity")
        axes[row, 1].set_xlabel("View Index")
        axes[row, 1].set_ylabel("View Index")
        plt.colorbar(im2, ax=axes[row, 1], label='Cosine Sim')

        # Plot 3: MSE matrix
        im3 = axes[row, 2].imshow(mse_matrix, cmap='hot', aspect='equal')
        axes[row, 2].set_title("Latent MSE")
        axes[row, 2].set_xlabel("View Index")
        axes[row, 2].set_ylabel("View Index")
        plt.colorbar(im3, ax=axes[row, 2], label='MSE')

        # Plot 4: Scatter of angular sep vs cosine sim
        # Extract upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(n_views, k=1)
        angles_flat = angular_sep[triu_idx]
        cos_flat = cos_matrix[triu_idx]

        axes[row, 3].scatter(angles_flat, cos_flat, alpha=0.5, s=20)
        axes[row, 3].set_xlabel("Angular Separation (°)")
        axes[row, 3].set_ylabel("Cosine Similarity")
        axes[row, 3].set_title("Angle vs Similarity")
        axes[row, 3].grid(True, alpha=0.3)

        # Add correlation
        corr = np.corrcoef(angles_flat, cos_flat)[0, 1]
        axes[row, 3].text(0.05, 0.95, f"r = {corr:.3f}",
                         transform=axes[row, 3].transAxes,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f"Per-Object Latent Consistency Analysis\nModel: {model_name}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "per_object_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-object analysis to {output_dir / 'per_object_analysis.png'}")


def visualize_object_sequence_examples(obj_data, output_dir, model_name, seq_length=5):
    """Visualize example sequences for a single object with images and latents."""
    obj_name = obj_data["object_name"]
    latents = obj_data["latents"]
    images = obj_data["images"]
    angular_sep = obj_data["angular_sep"]
    positions = obj_data["positions"]
    n_views = obj_data["n_views"]

    # Find a good sequence of seq_length views
    sequences = find_view_sequences(positions, angular_sep, seq_length=seq_length, max_pairwise_angle=30)

    if not sequences:
        # Fall back to consecutive views
        sequences = [(tuple(range(seq_length)), angular_sep[0, seq_length-1], 15.0)]

    # Take the first sequence
    view_indices, total_span, avg_step = sequences[0]

    fig, axes = plt.subplots(3, seq_length, figsize=(4*seq_length, 12))

    for col, view_idx in enumerate(view_indices):
        # Row 1: Images
        img_np = denormalize(images[view_idx][0]).permute(1, 2, 0).cpu().numpy()
        axes[0, col].imshow(np.clip(img_np, 0, 1))
        axes[0, col].set_title(f"View {view_idx}")
        axes[0, col].axis('off')

        # Row 2: Latents (first 3 channels as RGB)
        lat_vis = latents[view_idx][0, :3].cpu().numpy()
        lat_vis = np.transpose(lat_vis, (1, 2, 0))
        lat_vis = (lat_vis - lat_vis.min()) / (lat_vis.max() - lat_vis.min() + 1e-8)
        axes[1, col].imshow(lat_vis)
        axes[1, col].set_title("Latent")
        axes[1, col].axis('off')

        # Row 3: Difference from first view
        if col == 0:
            axes[2, col].text(0.5, 0.5, "Reference", ha='center', va='center',
                             fontsize=12, transform=axes[2, col].transAxes)
            axes[2, col].axis('off')
        else:
            lat_diff = (latents[view_idx] - latents[view_indices[0]]).abs()[0, :3].cpu().numpy()
            lat_diff = np.transpose(lat_diff, (1, 2, 0))
            lat_diff = (lat_diff - lat_diff.min()) / (lat_diff.max() - lat_diff.min() + 1e-8)
            axes[2, col].imshow(lat_diff)

            # Compute similarity to first view
            flat_ref = latents[view_indices[0]].flatten()
            flat_cur = latents[view_idx].flatten()
            cos_sim = F.cosine_similarity(flat_ref.unsqueeze(0), flat_cur.unsqueeze(0)).item()
            angle = angular_sep[view_indices[0], view_idx]
            axes[2, col].set_title(f"|Δ| cos={cos_sim:.2f}\nΔang={angle:.1f}°")
            axes[2, col].axis('off')

    # Add row labels
    axes[0, 0].set_ylabel("Images", fontsize=12)
    axes[1, 0].set_ylabel("Latents", fontsize=12)
    axes[2, 0].set_ylabel("Diff from v0", fontsize=12)

    plt.suptitle(f"Sequence Example: {obj_name}\nSpan: {total_span:.1f}°, Avg Step: {avg_step:.1f}°\nModel: {model_name}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"sequence_example_{obj_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sequence example to {output_dir / f'sequence_example_{obj_name}.png'}")


def save_results_to_file(all_results, output_path):
    """Save analysis results to a text file."""
    with open(output_path, 'w') as f:
        f.write("Multi-View Latent Consistency Analysis Results\n")
        f.write("=" * 60 + "\n\n")

        # Summary statistics
        angles = [r["angular_separation"] for r in all_results]
        cos_values = [r["cosine_similarity"] for r in all_results]
        mse_values = [r["mse"] for r in all_results]

        f.write("Summary Statistics:\n")
        f.write(f"  Total pairs analyzed: {len(all_results)}\n")
        f.write(f"  Angular separation range: {min(angles):.1f}° - {max(angles):.1f}°\n")
        f.write(f"\n  Cosine Similarity:\n")
        f.write(f"    Mean: {np.mean(cos_values):.4f}\n")
        f.write(f"    Std:  {np.std(cos_values):.4f}\n")
        f.write(f"    Min:  {min(cos_values):.4f}\n")
        f.write(f"    Max:  {max(cos_values):.4f}\n")
        f.write(f"\n  MSE:\n")
        f.write(f"    Mean: {np.mean(mse_values):.4f}\n")
        f.write(f"    Std:  {np.std(mse_values):.4f}\n")

        # Binned statistics
        f.write("\n\nBinned Statistics:\n")
        f.write("-" * 40 + "\n")

        for low, high in [(0, 10), (10, 20), (20, 30), (30, 45), (45, 90)]:
            bin_results = [r for r in all_results if low <= r["angular_separation"] < high]
            if bin_results:
                bin_cos = [r["cosine_similarity"] for r in bin_results]
                bin_mse = [r["mse"] for r in bin_results]
                f.write(f"\n  {low}° - {high}° ({len(bin_results)} pairs):\n")
                f.write(f"    Cosine Sim: {np.mean(bin_cos):.4f} ± {np.std(bin_cos):.4f}\n")
                f.write(f"    MSE: {np.mean(bin_mse):.4f} ± {np.std(bin_mse):.4f}\n")

        # Correlation between angle and similarity
        f.write("\n\nCorrelation Analysis:\n")
        f.write("-" * 40 + "\n")
        corr_angle_cos = np.corrcoef(angles, cos_values)[0, 1]
        corr_angle_mse = np.corrcoef(angles, mse_values)[0, 1]
        f.write(f"  Correlation(angle, cosine_sim): {corr_angle_cos:.4f}\n")
        f.write(f"  Correlation(angle, mse): {corr_angle_mse:.4f}\n")

    print(f"Saved results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze latent consistency for multi-view images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "ldm", "eqvae", "diffusers"],
                        help="Model type")
    parser.add_argument("--output_name", type=str, required=True,
                        help="Output subfolder name")

    parser.add_argument("--data_dir", type=str,
                        default="/data/lab_moezkan/omni_obj/blender_renders_24_views",
                        help="OmniObject3D dataset directory")
    parser.add_argument("--num_objects", type=int, default=50,
                        help="Number of objects to analyze")
    parser.add_argument("--max_angle", type=float, default=60,
                        help="Maximum angular separation to analyze")
    parser.add_argument("--min_angle", type=float, default=2,
                        help="Minimum angular separation")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for encoding")

    parser.add_argument("--compare_baseline", action="store_true",
                        help="Also compare with f8 baseline VAE (weights/f8/model.ckpt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Sequence analysis options
    parser.add_argument("--analyze_sequences", action="store_true",
                        help="Analyze longer sequences (3, 4, 5 views)")
    parser.add_argument("--seq_lengths", type=int, nargs='+', default=[3, 4, 5],
                        help="Sequence lengths to analyze")
    parser.add_argument("--per_object_plots", action="store_true",
                        help="Generate per-object detailed plots")
    parser.add_argument("--n_detailed_objects", type=int, default=5,
                        help="Number of objects for detailed per-object analysis")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    output_dir = Path("eval_outputs") / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, model_type = load_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        model_type=args.model_type,
    )
    model = model.to(device)
    model.eval()

    # Find object directories
    data_dir = Path(args.data_dir) / "img"
    object_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if len(object_dirs) > args.num_objects:
        np.random.shuffle(object_dirs)
        object_dirs = object_dirs[:args.num_objects]

    print(f"Analyzing {len(object_dirs)} objects...")

    # Analyze all objects (pairs)
    all_results = []
    for obj_dir in tqdm(object_dirs, desc="Processing objects (pairs)"):
        results = analyze_object(
            model, obj_dir, transform, device, model_type,
            max_angle=args.max_angle, min_angle=args.min_angle
        )
        all_results.extend(results)

    print(f"\nTotal view pairs analyzed: {len(all_results)}")

    # Save and visualize results
    model_name = Path(args.checkpoint).parent.parent.name
    save_results_to_file(all_results, output_dir / "latent_consistency_stats.txt")
    visualize_results(all_results, output_dir, model_name)

    # Create example visualization
    if object_dirs:
        visualize_example_pairs(
            model, object_dirs[0], transform, device, model_type, output_dir
        )

    # Sequence analysis (3+ views)
    if args.analyze_sequences:
        print("\n" + "=" * 60)
        print(f"Analyzing sequences of lengths: {args.seq_lengths}")
        print("=" * 60)

        seq_results_list = []
        for obj_dir in tqdm(object_dirs, desc="Processing sequences"):
            seq_results = analyze_object_sequences(
                model, obj_dir, transform, device, model_type,
                seq_lengths=args.seq_lengths,
                max_pairwise_angle=args.max_angle // 2
            )
            if seq_results:
                seq_results_list.append(seq_results)

        visualize_sequence_analysis(seq_results_list, output_dir, model_name)

        # Save sequence stats
        with open(output_dir / "sequence_stats.txt", 'w') as f:
            f.write("Sequence Analysis Results\n")
            f.write("=" * 60 + "\n\n")
            for seq_len in args.seq_lengths:
                all_seqs = []
                for obj_res in seq_results_list:
                    all_seqs.extend(obj_res.get("sequences", {}).get(seq_len, []))
                if all_seqs:
                    cos_vals = [s["pairwise_cos_mean"] for s in all_seqs if "pairwise_cos_mean" in s]
                    var_vals = [s["latent_variance"] for s in all_seqs if "latent_variance" in s]
                    f.write(f"\nSequence Length {seq_len}:\n")
                    f.write(f"  Total sequences: {len(all_seqs)}\n")
                    f.write(f"  Pairwise Cosine Sim: {np.mean(cos_vals):.4f} ± {np.std(cos_vals):.4f}\n")
                    f.write(f"  Latent Variance: {np.mean(var_vals):.4f} ± {np.std(var_vals):.4f}\n")
        print(f"Saved sequence stats to {output_dir / 'sequence_stats.txt'}")

    # Per-object detailed analysis
    if args.per_object_plots:
        print("\n" + "=" * 60)
        print(f"Generating detailed per-object analysis for {args.n_detailed_objects} objects...")
        print("=" * 60)

        # Select objects for detailed analysis
        detailed_objects = object_dirs[:args.n_detailed_objects]

        object_results_list = []
        for obj_dir in tqdm(detailed_objects, desc="Full object analysis"):
            obj_data = analyze_full_object(model, obj_dir, transform, device, model_type)
            if obj_data:
                object_results_list.append(obj_data)

        # Visualize per-object matrices
        visualize_per_object_results(object_results_list, output_dir, model_name, args.n_detailed_objects)

        # Visualize sequence examples for each object
        for obj_data in object_results_list:
            visualize_object_sequence_examples(obj_data, output_dir, model_name, seq_length=5)

    # Compare with f8 baseline VAE if requested
    if args.compare_baseline:
        print("\n" + "=" * 60)
        print("Comparing with f8 baseline VAE...")
        print("=" * 60)

        baseline_vae, baseline_type = load_f8_baseline_vae(device)
        baseline_output_dir = output_dir / "f8_baseline_comparison"
        baseline_output_dir.mkdir(exist_ok=True)

        baseline_results = []
        for obj_dir in tqdm(object_dirs, desc="Processing with f8 baseline"):
            results = analyze_object(
                baseline_vae, obj_dir, transform, device, baseline_type,
                max_angle=args.max_angle, min_angle=args.min_angle
            )
            baseline_results.extend(results)

        save_results_to_file(baseline_results, baseline_output_dir / "latent_consistency_stats.txt")
        visualize_results(baseline_results, baseline_output_dir, "f8 Baseline VAE (SD 2.x)")

        if object_dirs:
            visualize_example_pairs(
                baseline_vae, object_dirs[0], transform, device, baseline_type, baseline_output_dir
            )

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
