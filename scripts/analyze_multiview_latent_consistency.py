#!/usr/bin/env python
"""
Compare latent consistency across multiple VAE models on multi-view images.

This script compares one or more models on OmniObject3D multi-view data:
1. Loads multiple VAE models (custom checkpoints + optional baseline)
2. Encodes views from multiple objects with all models
3. Computes similarity metrics (cosine sim, MSE) vs angular separation
4. Generates comparative visualizations

Usage:
    # Compare single model with baseline
    python scripts/analyze_multiview_latent_consistency.py \
        --checkpoints outputs/my_model/checkpoints/last.ckpt \
        --configs config/my_config.yaml \
        --model_names "My Model" \
        --compare_baseline \
        --output_name comparison_test

    # Compare multiple models
    python scripts/analyze_multiview_latent_consistency.py \
        --checkpoints model1.ckpt model2.ckpt model3.ckpt \
        --configs config1.yaml config2.yaml config3.yaml \
        --model_names "Model A" "Model B" "Model C" \
        --output_name multi_model_comparison
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.decomposition import PCA

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis import (
    load_model,
    encode_images,
    denormalize,
    compute_latent_similarity,
    compute_pairwise_similarity_matrices,
    load_camera_data,
    extract_camera_positions,
    compute_angular_separation,
    find_overlapping_pairs,
    find_view_sequences,
    latent_to_pca_rgb,
)

# Default paths for f8 baseline VAE
F8_BASELINE_CHECKPOINT = "weights/f8/model.ckpt"
F8_BASELINE_CONFIG = "config/baseVAE.yaml"

# Color palette for models
MODEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def load_f8_baseline_vae(device="cuda"):
    """Load the f8 SD-VAE baseline model."""
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


def load_view_image(obj_dir: Path, view_idx: int, transform, device: str) -> torch.Tensor:
    """Load and preprocess a single view image."""
    img_path = obj_dir / f"{view_idx:03d}.png"
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


@torch.no_grad()
def analyze_object_with_models(
    models: List[Tuple],  # List of (model, model_type, model_name)
    obj_dir: Path,
    transform,
    device: str,
    max_angle: float = 60,
    min_angle: float = 2,
    max_pairs: int = 50
) -> Dict[str, List[Dict]]:
    """Analyze latent consistency for a single object across all models.

    Returns:
        Dictionary mapping model_name -> list of pair results
    """
    transforms_path = obj_dir / "transforms.json"
    if not transforms_path.exists():
        return {}

    camera_data = load_camera_data(transforms_path)
    positions = extract_camera_positions(camera_data)
    angular_sep = compute_angular_separation(positions)
    pairs = find_overlapping_pairs(angular_sep, max_angle=max_angle, min_angle=min_angle)

    if len(pairs) > max_pairs:
        pairs = pairs[::len(pairs)//max_pairs][:max_pairs]

    results_by_model = {}

    for model, model_type, model_name in models:
        results = []
        for view1_idx, view2_idx, angle in pairs:
            img1 = load_view_image(obj_dir, view1_idx, transform, device)
            img2 = load_view_image(obj_dir, view2_idx, transform, device)

            latent1 = encode_images(model, img1, device, model_type)
            latent2 = encode_images(model, img2, device, model_type)

            similarity = compute_latent_similarity(latent1, latent2)

            results.append({
                "view1_idx": view1_idx,
                "view2_idx": view2_idx,
                "angular_separation": angle,
                **similarity
            })

        results_by_model[model_name] = results

    return results_by_model


@torch.no_grad()
def encode_object_views(
    models: List[Tuple],
    obj_dir: Path,
    transform,
    device: str,
    max_views: int = 24
) -> Dict[str, Dict]:
    """Encode all views of an object with all models.

    Returns:
        Dictionary mapping model_name -> {latents, images, positions, angular_sep, ...}
    """
    transforms_path = obj_dir / "transforms.json"
    if not transforms_path.exists():
        return {}

    camera_data = load_camera_data(transforms_path)
    positions = extract_camera_positions(camera_data)
    angular_sep = compute_angular_separation(positions)
    n_views = min(len(positions), max_views)

    # Load images once
    images = []
    for view_idx in range(n_views):
        img = load_view_image(obj_dir, view_idx, transform, device)
        images.append(img)

    results = {}
    for model, model_type, model_name in models:
        latents = []
        for img in images:
            latent = encode_images(model, img, device, model_type)
            latents.append(latent)

        # Compute pairwise similarity matrices
        matrices = compute_pairwise_similarity_matrices(latents)

        results[model_name] = {
            "latents": latents,
            **matrices,
        }

    # Add shared data (same for all models)
    shared = {
        "images": images,
        "positions": positions,
        "angular_sep": angular_sep,
        "n_views": n_views,
        "object_name": obj_dir.name,
    }

    return {"models": results, "shared": shared}


def visualize_model_comparison(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
    model_colors: Dict[str, str]
):
    """Create main comparison visualization across all models."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    model_names = list(all_results.keys())

    # Plot 1: Cosine Similarity vs Angular Separation (scatter + trend)
    ax1 = axes[0, 0]
    for model_name in model_names:
        results = all_results[model_name]
        angles = [r["angular_separation"] for r in results]
        cos_values = [r["cosine_similarity"] for r in results]
        color = model_colors[model_name]

        ax1.scatter(angles, cos_values, alpha=0.3, s=10, color=color, label=model_name)

        # Trend line
        if len(angles) > 3:
            z = np.polyfit(angles, cos_values, 2)
            x_line = np.linspace(min(angles), max(angles), 100)
            ax1.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax1.set_xlabel("Angular Separation (degrees)")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Latent Cosine Similarity vs Camera Angle")
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.4, 1.0])

    # Plot 2: MSE vs Angular Separation (scatter + trend)
    ax2 = axes[0, 1]
    for model_name in model_names:
        results = all_results[model_name]
        angles = [r["angular_separation"] for r in results]
        mse_values = [r["mse"] for r in results]
        color = model_colors[model_name]

        ax2.scatter(angles, mse_values, alpha=0.3, s=10, color=color, label=model_name)

        if len(angles) > 3:
            z = np.polyfit(angles, mse_values, 2)
            x_line = np.linspace(min(angles), max(angles), 100)
            ax2.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax2.set_xlabel("Angular Separation (degrees)")
    ax2.set_ylabel("Latent MSE")
    ax2.set_title("Latent MSE vs Camera Angle")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: MAE vs Angular Separation (scatter + trend)
    ax3 = axes[0, 2]
    for model_name in model_names:
        results = all_results[model_name]
        angles = [r["angular_separation"] for r in results]
        mae_values = [r["mae"] for r in results]
        color = model_colors[model_name]

        ax3.scatter(angles, mae_values, alpha=0.3, s=10, color=color, label=model_name)

        if len(angles) > 3:
            z = np.polyfit(angles, mae_values, 2)
            x_line = np.linspace(min(angles), max(angles), 100)
            ax3.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax3.set_xlabel("Angular Separation (degrees)")
    ax3.set_ylabel("Latent MAE")
    ax3.set_title("Latent MAE vs Camera Angle")
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Box plot comparison (Cosine Similarity)
    ax4 = axes[1, 0]
    box_data = [
        [r["cosine_similarity"] for r in all_results[name]]
        for name in model_names
    ]
    bp = ax4.boxplot(box_data, labels=model_names, patch_artist=True)
    for patch, name in zip(bp['boxes'], model_names):
        patch.set_facecolor(to_rgba(model_colors[name], 0.6))
    ax4.set_ylabel("Cosine Similarity")
    ax4.set_title("Cosine Similarity Distribution")
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=20, ha='right')

    # Plot 5: Box plot comparison (MAE)
    ax5 = axes[1, 1]
    box_data_mae = [
        [r["mae"] for r in all_results[name]]
        for name in model_names
    ]
    bp_mae = ax5.boxplot(box_data_mae, labels=model_names, patch_artist=True)
    for patch, name in zip(bp_mae['boxes'], model_names):
        patch.set_facecolor(to_rgba(model_colors[name], 0.6))
    ax5.set_ylabel("MAE")
    ax5.set_title("MAE Distribution")
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=20, ha='right')

    # Plot 6: Binned cosine similarity comparison
    ax6 = axes[1, 2]
    angle_bins = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 90)]
    bin_labels = [f"{low}-{high}°" for low, high in angle_bins]
    x = np.arange(len(bin_labels))
    width = 0.8 / len(model_names)

    for idx, model_name in enumerate(model_names):
        results = all_results[model_name]
        means = []
        stds = []
        for low, high in angle_bins:
            bin_vals = [r["cosine_similarity"] for r in results if low <= r["angular_separation"] < high]
            means.append(np.mean(bin_vals) if bin_vals else 0)
            stds.append(np.std(bin_vals) if bin_vals else 0)

        offset = (idx - len(model_names)/2 + 0.5) * width
        ax6.bar(x + offset, means, width, yerr=stds, label=model_name,
                color=model_colors[model_name], alpha=0.7, capsize=2)

    ax6.set_xticks(x)
    ax6.set_xticklabels(bin_labels)
    ax6.set_xlabel("Angular Separation Range")
    ax6.set_ylabel("Mean Cosine Similarity")
    ax6.set_title("Cosine Similarity by Angle Bin")
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # Plot 7: Binned MSE comparison
    ax7 = axes[2, 0]
    for idx, model_name in enumerate(model_names):
        results = all_results[model_name]
        means = []
        stds = []
        for low, high in angle_bins:
            bin_vals = [r["mse"] for r in results if low <= r["angular_separation"] < high]
            means.append(np.mean(bin_vals) if bin_vals else 0)
            stds.append(np.std(bin_vals) if bin_vals else 0)

        offset = (idx - len(model_names)/2 + 0.5) * width
        ax7.bar(x + offset, means, width, yerr=stds, label=model_name,
                color=model_colors[model_name], alpha=0.7, capsize=2)

    ax7.set_xticks(x)
    ax7.set_xticklabels(bin_labels)
    ax7.set_xlabel("Angular Separation Range")
    ax7.set_ylabel("Mean MSE")
    ax7.set_title("Latent MSE by Angle Bin")
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # Plot 8: Binned MAE comparison
    ax8 = axes[2, 1]
    for idx, model_name in enumerate(model_names):
        results = all_results[model_name]
        means = []
        stds = []
        for low, high in angle_bins:
            bin_vals = [r["mae"] for r in results if low <= r["angular_separation"] < high]
            means.append(np.mean(bin_vals) if bin_vals else 0)
            stds.append(np.std(bin_vals) if bin_vals else 0)

        offset = (idx - len(model_names)/2 + 0.5) * width
        ax8.bar(x + offset, means, width, yerr=stds, label=model_name,
                color=model_colors[model_name], alpha=0.7, capsize=2)

    ax8.set_xticks(x)
    ax8.set_xticklabels(bin_labels)
    ax8.set_xlabel("Angular Separation Range")
    ax8.set_ylabel("Mean MAE")
    ax8.set_title("Latent MAE by Angle Bin")
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # Plot 9: Summary statistics table
    ax9 = axes[2, 2]
    ax9.axis('off')

    table_data = []
    headers = ["Metric"] + [name[:12] for name in model_names]

    # Compute stats for each model
    stats = {}
    for model_name in model_names:
        results = all_results[model_name]
        angles = [r["angular_separation"] for r in results]
        cos_vals = [r["cosine_similarity"] for r in results]
        mse_vals = [r["mse"] for r in results]
        mae_vals = [r["mae"] for r in results]

        stats[model_name] = {
            "n_pairs": len(results),
            "cos_mean": np.mean(cos_vals),
            "cos_std": np.std(cos_vals),
            "mse_mean": np.mean(mse_vals),
            "mae_mean": np.mean(mae_vals),
            "mae_std": np.std(mae_vals),
            "angle_cos_corr": np.corrcoef(angles, cos_vals)[0, 1] if len(angles) > 1 else 0,
        }

    table_data.append(["N pairs"] + [f"{stats[n]['n_pairs']}" for n in model_names])
    table_data.append(["Cos Sim (mean)"] + [f"{stats[n]['cos_mean']:.4f}" for n in model_names])
    table_data.append(["Cos Sim (std)"] + [f"{stats[n]['cos_std']:.4f}" for n in model_names])
    table_data.append(["MSE (mean)"] + [f"{stats[n]['mse_mean']:.4f}" for n in model_names])
    table_data.append(["MAE (mean)"] + [f"{stats[n]['mae_mean']:.4f}" for n in model_names])
    table_data.append(["MAE (std)"] + [f"{stats[n]['mae_std']:.4f}" for n in model_names])
    table_data.append(["Angle-Cos Corr"] + [f"{stats[n]['angle_cos_corr']:.4f}" for n in model_names])

    table = ax9.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    ax9.set_title("Summary Statistics", pad=20)

    plt.suptitle("Multi-View Latent Consistency: Model Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison to {output_dir / 'model_comparison.png'}")


def visualize_sequence_comparison(
    object_data: Dict,
    output_dir: Path,
    model_colors: Dict[str, str],
    seq_length: int = 5
):
    """Visualize a sequence of views with latent PCA for all models."""
    shared = object_data["shared"]
    models_data = object_data["models"]

    obj_name = shared["object_name"]
    images = shared["images"]
    positions = shared["positions"]
    angular_sep = shared["angular_sep"]
    n_views = shared["n_views"]

    model_names = list(models_data.keys())
    n_models = len(model_names)

    # Find a good sequence
    sequences = find_view_sequences(positions, angular_sep, seq_length=seq_length, max_pairwise_angle=30)

    if not sequences:
        if n_views >= seq_length:
            sequences = [(tuple(range(seq_length)), angular_sep[0, seq_length-1], 15.0)]
        else:
            print(f"Not enough views for sequence of length {seq_length}")
            return

    view_indices, total_span, avg_step = sequences[0]

    # Fit PCA on combined latents from all models for fair visualization
    all_lat_flat = []
    for model_name in model_names:
        latents = models_data[model_name]["latents"]
        for view_idx in view_indices:
            lat = latents[view_idx][0].cpu().numpy()
            C, H, W = lat.shape
            all_lat_flat.append(lat.reshape(C, -1).T)
    all_lat_flat = np.vstack(all_lat_flat)

    pca_model = PCA(n_components=3)
    pca_model.fit(all_lat_flat)

    # Create figure: (1 + n_models) rows x seq_length columns
    n_rows = 1 + n_models
    fig, axes = plt.subplots(n_rows, seq_length, figsize=(4*seq_length, int(2.75*n_rows)))
    
    print(model_names)
    for col, view_idx in enumerate(view_indices):
        # Row 0: Original images
        img_np = denormalize(images[view_idx][0]).permute(1, 2, 0).cpu().numpy()
        axes[0, col].imshow(np.clip(img_np, 0, 1))
        axes[0, col].set_title(f"View {view_idx}")
        axes[0, col].axis('off')

        # Rows 1 to n_models: Latent PCA for each model
        for row, model_name in enumerate(model_names, start=1):
            latents = models_data[model_name]["latents"]
            lat_rgb, _ = latent_to_pca_rgb(latents[view_idx], pca_model)
            axes[row, col].imshow(lat_rgb)
            if col == 0:
                axes[row, col].set_ylabel(model_name, fontsize=11, fontweight='bold')

            axes[row, col].axis('off')

    axes[0, 0].set_ylabel("Images", fontsize=11, fontweight='bold')

    plt.suptitle(f"Latent Comparison: {obj_name}\nSpan: {total_span:.1f}°, Avg Step: {avg_step:.1f}°",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"sequence_{obj_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sequence comparison to {output_dir / f'sequence_{obj_name}.png'}")


def visualize_similarity_matrices(
    object_data: Dict,
    output_dir: Path,
    model_colors: Dict[str, str]
):
    """Visualize similarity matrices for all models side by side."""
    shared = object_data["shared"]
    models_data = object_data["models"]

    obj_name = shared["object_name"]
    angular_sep = shared["angular_sep"]

    model_names = list(models_data.keys())
    n_models = len(model_names)

    # Create figure: 3 rows (cos_sim, mse, mae) x (1 + n_models) columns
    fig, axes = plt.subplots(3, 1 + n_models, figsize=(4*(1+n_models), 11))

    # Column 0: Angular separation matrix
    im0 = axes[0, 0].imshow(angular_sep, cmap='viridis', aspect='equal')
    axes[0, 0].set_title("Angular Sep (°)")
    axes[0, 0].set_xlabel("View")
    axes[0, 0].set_ylabel("View")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    axes[1, 0].axis('off')  # Empty cell
    axes[2, 0].axis('off')  # Empty cell

    # Columns 1 to n_models: Similarity matrices for each model
    for col, model_name in enumerate(model_names, start=1):
        cos_matrix = models_data[model_name]["cos_sim_matrix"]
        mse_matrix = models_data[model_name]["mse_matrix"]
        mae_matrix = models_data[model_name]["mae_matrix"]

        # Cosine similarity
        im1 = axes[0, col].imshow(cos_matrix, cmap='RdYlGn', aspect='equal', vmin=0.5, vmax=1.0)
        axes[0, col].set_title(f"{model_name}\nCosine Sim")
        axes[0, col].set_xlabel("View")
        plt.colorbar(im1, ax=axes[0, col], shrink=0.8)

        # MSE
        im2 = axes[1, col].imshow(mse_matrix, cmap='hot', aspect='equal')
        axes[1, col].set_title(f"{model_name}\nMSE")
        axes[1, col].set_xlabel("View")
        plt.colorbar(im2, ax=axes[1, col], shrink=0.8)

        # MAE
        im3 = axes[2, col].imshow(mae_matrix, cmap='hot', aspect='equal')
        axes[2, col].set_title(f"{model_name}\nMAE")
        axes[2, col].set_xlabel("View")
        plt.colorbar(im3, ax=axes[2, col], shrink=0.8)

    plt.suptitle(f"Similarity Matrices: {obj_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"matrices_{obj_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity matrices to {output_dir / f'matrices_{obj_name}.png'}")


def visualize_angle_vs_similarity_per_object(
    object_data: Dict,
    output_dir: Path,
    model_colors: Dict[str, str]
):
    """Scatter plot of angle vs similarity for a single object, all models."""
    shared = object_data["shared"]
    models_data = object_data["models"]

    obj_name = shared["object_name"]
    angular_sep = shared["angular_sep"]
    n_views = shared["n_views"]

    model_names = list(models_data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Extract upper triangle indices
    triu_idx = np.triu_indices(n_views, k=1)
    angles_flat = angular_sep[triu_idx]

    # Plot cosine similarity
    ax1 = axes[0]
    for model_name in model_names:
        cos_matrix = models_data[model_name]["cos_sim_matrix"]
        cos_flat = cos_matrix[triu_idx]
        color = model_colors[model_name]

        ax1.scatter(angles_flat, cos_flat, alpha=0.5, s=30, color=color, label=model_name)

        # Trend line
        if len(angles_flat) > 3:
            z = np.polyfit(angles_flat, cos_flat, 2)
            x_line = np.linspace(min(angles_flat), max(angles_flat), 100)
            ax1.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax1.set_xlabel("Angular Separation (°)")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Cosine Similarity vs Angle")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot MSE
    ax2 = axes[1]
    for model_name in model_names:
        mse_matrix = models_data[model_name]["mse_matrix"]
        mse_flat = mse_matrix[triu_idx]
        color = model_colors[model_name]

        ax2.scatter(angles_flat, mse_flat, alpha=0.5, s=30, color=color, label=model_name)

        if len(angles_flat) > 3:
            z = np.polyfit(angles_flat, mse_flat, 2)
            x_line = np.linspace(min(angles_flat), max(angles_flat), 100)
            ax2.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax2.set_xlabel("Angular Separation (°)")
    ax2.set_ylabel("MSE")
    ax2.set_title("MSE vs Angle")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot MAE
    ax3 = axes[2]
    for model_name in model_names:
        mae_matrix = models_data[model_name]["mae_matrix"]
        mae_flat = mae_matrix[triu_idx]
        color = model_colors[model_name]

        ax3.scatter(angles_flat, mae_flat, alpha=0.5, s=30, color=color, label=model_name)

        if len(angles_flat) > 3:
            z = np.polyfit(angles_flat, mae_flat, 2)
            x_line = np.linspace(min(angles_flat), max(angles_flat), 100)
            ax3.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax3.set_xlabel("Angular Separation (°)")
    ax3.set_ylabel("MAE")
    ax3.set_title("MAE vs Angle")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f"Angle vs Similarity: {obj_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"angle_vs_sim_{obj_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved angle vs similarity to {output_dir / f'angle_vs_sim_{obj_name}.png'}")


def save_comparison_stats(
    all_results: Dict[str, List[Dict]],
    output_path: Path
):
    """Save comparison statistics to a text file."""
    model_names = list(all_results.keys())

    with open(output_path, 'w') as f:
        f.write("Multi-View Latent Consistency: Model Comparison\n")
        f.write("=" * 70 + "\n\n")

        for model_name in model_names:
            results = all_results[model_name]
            angles = [r["angular_separation"] for r in results]
            cos_values = [r["cosine_similarity"] for r in results]
            mse_values = [r["mse"] for r in results]
            mae_values = [r["mae"] for r in results]

            f.write(f"Model: {model_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Total pairs analyzed: {len(results)}\n")
            f.write(f"  Angular separation range: {min(angles):.1f}° - {max(angles):.1f}°\n")
            f.write("\n  Cosine Similarity:\n")
            f.write(f"    Mean: {np.mean(cos_values):.4f}\n")
            f.write(f"    Std:  {np.std(cos_values):.4f}\n")
            f.write(f"    Min:  {min(cos_values):.4f}\n")
            f.write(f"    Max:  {max(cos_values):.4f}\n")
            f.write("\n  MSE:\n")
            f.write(f"    Mean: {np.mean(mse_values):.4f}\n")
            f.write(f"    Std:  {np.std(mse_values):.4f}\n")
            f.write("\n  MAE:\n")
            f.write(f"    Mean: {np.mean(mae_values):.4f}\n")
            f.write(f"    Std:  {np.std(mae_values):.4f}\n")
            f.write(f"    Min:  {min(mae_values):.4f}\n")
            f.write(f"    Max:  {max(mae_values):.4f}\n")

            corr = np.corrcoef(angles, cos_values)[0, 1] if len(angles) > 1 else 0
            f.write(f"\n  Correlation(angle, cos_sim): {corr:.4f}\n")
            f.write("\n")

        # Binned comparison (Cosine Similarity)
        f.write("\nBinned Comparison (Cosine Similarity)\n")
        f.write("=" * 70 + "\n")

        angle_bins = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 90)]

        header = f"{'Bin':<12}" + "".join([f"{name[:12]:<14}" for name in model_names])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for low, high in angle_bins:
            row = f"{low}-{high}°".ljust(12)
            for model_name in model_names:
                results = all_results[model_name]
                bin_vals = [r["cosine_similarity"] for r in results if low <= r["angular_separation"] < high]
                if bin_vals:
                    row += f"{np.mean(bin_vals):.3f}±{np.std(bin_vals):.3f}".ljust(14)
                else:
                    row += "N/A".ljust(14)
            f.write(row + "\n")

        # Binned comparison (MAE)
        f.write("\n\nBinned Comparison (MAE)\n")
        f.write("=" * 70 + "\n")

        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for low, high in angle_bins:
            row = f"{low}-{high}°".ljust(12)
            for model_name in model_names:
                results = all_results[model_name]
                bin_vals = [r["mae"] for r in results if low <= r["angular_separation"] < high]
                if bin_vals:
                    row += f"{np.mean(bin_vals):.3f}±{np.std(bin_vals):.3f}".ljust(14)
                else:
                    row += "N/A".ljust(14)
            f.write(row + "\n")

    print(f"Saved statistics to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare latent consistency across multiple VAE models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model inputs (can specify multiple)
    parser.add_argument(
        "--checkpoints", type=str, nargs='+', required=True,
        help="Paths to model checkpoints (one or more)"
    )
    parser.add_argument(
        "--configs", type=str, nargs='+', required=True,
        help="Paths to config files (one per checkpoint)"
    )
    parser.add_argument(
        "--model_names", type=str, nargs='+', default=None,
        help="Names for each model (defaults to checkpoint names)"
    )
    parser.add_argument(
        "--model_types", type=str, nargs='+', default=None,
        choices=["auto", "ldm", "eqvae", "diffusers"],
        help="Model types (one per checkpoint, defaults to 'auto')"
    )

    # Baseline comparison
    parser.add_argument(
        "--compare_baseline", action="store_true",
        help="Include f8 baseline VAE in comparison"
    )

    # Output
    parser.add_argument(
        "--output_name", type=str, required=True,
        help="Output subfolder name under eval_outputs/"
    )

    # Data options
    parser.add_argument(
        "--data_dir", type=str,
        default="/data/lab_moezkan/omni_obj/blender_renders_24_views",
        help="OmniObject3D dataset directory"
    )
    parser.add_argument(
        "--num_objects", type=int, default=50,
        help="Number of objects to analyze for aggregate statistics"
    )
    parser.add_argument(
        "--num_detailed_objects", type=int, default=5,
        help="Number of objects for detailed per-object visualizations"
    )
    parser.add_argument(
        "--max_angle", type=float, default=30,
        help="Maximum angular separation to analyze"
    )
    parser.add_argument(
        "--min_angle", type=float, default=2,
        help="Minimum angular separation"
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Image size for encoding"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    n_models = len(args.checkpoints)
    if len(args.configs) != n_models:
        raise ValueError(f"Number of configs ({len(args.configs)}) must match checkpoints ({n_models})")

    if args.model_names is None:
        args.model_names = [Path(ckpt).parent.parent.name for ckpt in args.checkpoints]
    elif len(args.model_names) != n_models:
        raise ValueError(f"Number of model names ({len(args.model_names)}) must match checkpoints ({n_models})")

    if args.model_types is None:
        args.model_types = ["auto"] * n_models
    elif len(args.model_types) != n_models:
        raise ValueError(f"Number of model types ({len(args.model_types)}) must match checkpoints ({n_models})")

    # Setup
    output_dir = Path("eval_outputs") / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load all models
    models = []
    print("\nLoading models...")
    for ckpt, cfg, name, mtype in zip(args.checkpoints, args.configs, args.model_names, args.model_types):
        print(f"\n  Loading {name} from {ckpt}")
        model, model_type = load_model(checkpoint_path=ckpt, config_path=cfg, model_type=mtype)
        model = model.to(device)
        model.eval()
        models.append((model, model_type, name))

    # Add baseline if requested
    if args.compare_baseline:
        baseline_model, baseline_type = load_f8_baseline_vae(device)
        models.append((baseline_model, baseline_type, "f8 Baseline"))

    # Assign colors
    model_colors = {name: MODEL_COLORS[i % len(MODEL_COLORS)]
                   for i, (_, _, name) in enumerate(models)}

    # Find object directories
    data_dir = Path(args.data_dir) / "img"
    object_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if len(object_dirs) > args.num_objects:
        np.random.shuffle(object_dirs)
        object_dirs = object_dirs[:args.num_objects]

    print(f"\nAnalyzing {len(object_dirs)} objects with {len(models)} models...")

    # Aggregate analysis across all objects
    all_results = {name: [] for _, _, name in models}
    for obj_dir in tqdm(object_dirs, desc="Processing objects"):
        results_by_model = analyze_object_with_models(
            models, obj_dir, transform, device,
            max_angle=args.max_angle, min_angle=args.min_angle
        )
        for model_name, results in results_by_model.items():
            all_results[model_name].extend(results)

    # Generate aggregate visualizations
    print("\nGenerating aggregate visualizations...")
    visualize_model_comparison(all_results, output_dir, model_colors)
    save_comparison_stats(all_results, output_dir / "comparison_stats.txt")

    # Detailed per-object visualizations
    print(f"\nGenerating detailed visualizations for {args.num_detailed_objects} objects...")
    detailed_dirs = object_dirs[:args.num_detailed_objects]

    for obj_dir in tqdm(detailed_dirs, desc="Detailed analysis"):
        object_data = encode_object_views(models, obj_dir, transform, device)
        if object_data:
            visualize_sequence_comparison(object_data, output_dir, model_colors)
            visualize_similarity_matrices(object_data, output_dir, model_colors)
            visualize_angle_vs_similarity_per_object(object_data, output_dir, model_colors)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
