#!/usr/bin/env python
"""
Evaluation script for Warp VAE training.

Visualizes:
1. Dataset samples with warp fields
2. Warp quality (warped images vs targets)
3. Model reconstructions and latent consistency
4. Training metrics summary

Usage:
    python eval_warp_vae.py [--checkpoint PATH] [--num_samples N]
"""

import os
import argparse

# Set GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Disable torch.compile for older GPUs
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


def load_dataset(config_path: str = None):
    """Load the WarpCO3DDataset."""
    from src.data.warp_dataset import WarpCO3DDataset

    dataset = WarpCO3DDataset(
        root_dir="/data/lab_moezkan/co3d_full",
        bb_file="/data/lab_moezkan/co3d_bboxes/toybus_test.jgz",
        image_size=128,
        romav2_setting="turbo",
        pair_sampling="random",
        max_pair_distance=10,
        warp_resolution=128,
        confidence_threshold=0.5,
    )
    return dataset


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] range."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def tensor_to_numpy(tensor):
    """Convert tensor to numpy for visualization."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Remove batch dim
    return denormalize(tensor).permute(1, 2, 0).cpu().numpy()


def visualize_dataset_samples(
    dataset, num_samples=4, save_path="eval_dataset_samples.png"
):
    """Visualize dataset samples with warps."""
    print(f"\n[1] Visualizing {num_samples} dataset samples...")

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))

    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        sample = dataset[idx]

        # Get images
        img_a = tensor_to_numpy(sample["image"])
        img_b = tensor_to_numpy(sample["image_target"])

        # Warp image A to B
        img_a_tensor = sample["image"].unsqueeze(0)
        warp_ab = sample["warp_ab"].unsqueeze(0)

        warped_a = F.grid_sample(
            img_a_tensor,
            warp_ab,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        warped_a_np = tensor_to_numpy(warped_a)

        # Confidence map
        conf = sample["confidence_ab"].numpy()

        # Compute difference
        diff = np.abs(warped_a_np - img_b)

        # Plot
        axes[i, 0].imshow(img_a)
        axes[i, 0].set_title(f"Source (idx={idx})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_b)
        axes[i, 1].set_title("Target")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(warped_a_np)
        axes[i, 2].set_title("Warped Source")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(conf, cmap="hot", vmin=0, vmax=1)
        axes[i, 3].set_title(f"Confidence (mean={conf.mean():.2f})")
        axes[i, 3].axis("off")

        axes[i, 4].imshow(diff)
        axes[i, 4].set_title(f"Warp Error (mean={diff.mean():.3f})")
        axes[i, 4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to {save_path}")
    plt.close()


def visualize_warp_flow(dataset, num_samples=2, save_path="eval_warp_flow.png"):
    """Visualize warp flow fields."""
    print("\n[2] Visualizing warp flow fields...")

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        sample = dataset[idx]

        img_a = tensor_to_numpy(sample["image"])
        img_b = tensor_to_numpy(sample["image_target"])

        warp_ab = sample["warp_ab"].numpy()  # (H, W, 2)
        warp_ba = sample["warp_ba"].numpy()

        # Flow visualization (convert from [-1,1] grid coords to pixel displacement)
        H, W = warp_ab.shape[:2]

        # Create coordinate grid
        y, x = np.mgrid[0:H, 0:W]
        x_norm = 2 * x / (W - 1) - 1  # Convert to [-1, 1]
        y_norm = 2 * y / (H - 1) - 1

        # Compute displacement
        flow_x = (warp_ab[..., 0] - x_norm) * W / 2
        flow_y = (warp_ab[..., 1] - y_norm) * H / 2

        # Flow magnitude
        flow_mag = np.sqrt(flow_x**2 + flow_y**2)

        axes[i, 0].imshow(img_a)
        axes[i, 0].set_title("Source")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_b)
        axes[i, 1].set_title("Target")
        axes[i, 1].axis("off")

        # Flow magnitude
        im = axes[i, 2].imshow(flow_mag, cmap="jet")
        axes[i, 2].set_title(f"Flow Magnitude (max={flow_mag.max():.1f}px)")
        axes[i, 2].axis("off")
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

        # Flow quiver (subsampled)
        step = 8
        axes[i, 3].imshow(img_a, alpha=0.5)
        axes[i, 3].quiver(
            x[::step, ::step],
            y[::step, ::step],
            flow_x[::step, ::step],
            -flow_y[::step, ::step],  # Negative y for display
            color="red",
            scale=100,
            width=0.003,
        )
        axes[i, 3].set_title("Flow Vectors")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to {save_path}")
    plt.close()


def visualize_model_outputs(
    model, dataset, device, num_samples=4, save_path="eval_model_outputs.png"
):
    """Visualize model reconstructions and warped latents."""
    print("\n[3] Visualizing model outputs...")

    model.eval()
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(len(dataset))
            sample = dataset[idx]

            # Prepare batch
            img_a = sample["image"].unsqueeze(0).to(device)
            img_b = sample["image_target"].unsqueeze(0).to(device)
            warp_ab = sample["warp_ab"].unsqueeze(0).to(device)

            # Encode and reconstruct
            recon_a, posterior_a = model(img_a, sample_posterior=True)
            recon_b, posterior_b = model(img_b, sample_posterior=True)

            latent_a = posterior_a.sample()
            latent_b = posterior_b.sample()

            # Warp source image to target view
            warped_img_a = F.grid_sample(
                img_a,
                warp_ab,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

            # Warp reconstruction to target view
            warped_recon_a = F.grid_sample(
                recon_a,
                warp_ab,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

            # Resize warp for latent space
            latent_h, latent_w = latent_a.shape[2:]
            warp_latent = F.interpolate(
                warp_ab.permute(0, 3, 1, 2),
                size=(latent_h, latent_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

            # Warp latent A to B
            warped_latent_a = F.grid_sample(
                latent_a,
                warp_latent,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

            # Compute latent difference
            latent_diff = (warped_latent_a - latent_b).abs().mean(dim=1, keepdim=True)

            # Convert to numpy
            img_a_np = tensor_to_numpy(img_a)
            img_b_np = tensor_to_numpy(img_b)
            recon_a_np = tensor_to_numpy(recon_a)
            warped_img_np = tensor_to_numpy(warped_img_a)
            warped_recon_np = tensor_to_numpy(warped_recon_a)
            latent_diff_np = latent_diff[0, 0].cpu().numpy()

            # Plot
            axes[i, 0].imshow(img_a_np)
            axes[i, 0].set_title("Source")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(recon_a_np)
            axes[i, 1].set_title("Reconstruction")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(img_b_np)
            axes[i, 2].set_title("Target")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(warped_img_np)
            axes[i, 3].set_title("Warped Source")
            axes[i, 3].axis("off")

            axes[i, 4].imshow(warped_recon_np)
            axes[i, 4].set_title("Warped Recon")
            axes[i, 4].axis("off")

            im = axes[i, 5].imshow(latent_diff_np, cmap="hot")
            axes[i, 5].set_title(f"Latent Diff (mean={latent_diff_np.mean():.3f})")
            axes[i, 5].axis("off")
            plt.colorbar(im, ax=axes[i, 5], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to {save_path}")
    plt.close()


def visualize_latent_consistency(
    model, dataset, device, num_samples=100, save_path="eval_latent_consistency.png"
):
    """Compute and visualize latent consistency statistics."""
    print(f"\n[4] Computing latent consistency over {num_samples} samples...")

    model.eval()

    consistency_scores = []
    recon_errors = []
    warp_errors = []

    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(len(dataset))
            sample = dataset[idx]

            img_a = sample["image"].unsqueeze(0).to(device)
            img_b = sample["image_target"].unsqueeze(0).to(device)
            warp_ab = sample["warp_ab"].unsqueeze(0).to(device)
            conf_ab = sample["confidence_ab"].unsqueeze(0).to(device)

            # Encode
            posterior_a = model.encode(img_a)
            posterior_b = model.encode(img_b)
            latent_a = posterior_a.sample()
            latent_b = posterior_b.sample()

            # Reconstruct
            recon_a = model.decode(latent_a)

            # Resize warp for latent
            latent_h, latent_w = latent_a.shape[2:]
            warp_latent = F.interpolate(
                warp_ab.permute(0, 3, 1, 2),
                size=(latent_h, latent_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

            conf_latent = F.interpolate(
                conf_ab.unsqueeze(1),
                size=(latent_h, latent_w),
                mode="bilinear",
                align_corners=False,
            )

            # Warp latent A to B
            warped_latent_a = F.grid_sample(
                latent_a,
                warp_latent,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

            # Compute weighted consistency
            diff = (warped_latent_a - latent_b).abs()
            weighted_diff = (diff * conf_latent).sum() / (conf_latent.sum() + 1e-8)
            consistency_scores.append(weighted_diff.item())

            # Reconstruction error
            recon_err = F.l1_loss(recon_a, img_a).item()
            recon_errors.append(recon_err)

            # Warp error in image space
            warped_img = F.grid_sample(
                img_a,
                warp_ab,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            warp_err = F.l1_loss(warped_img, img_b).item()
            warp_errors.append(warp_err)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{num_samples}")

    # Plot statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(consistency_scores, bins=30, edgecolor="black", alpha=0.7)
    axes[0].axvline(
        np.mean(consistency_scores),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(consistency_scores):.3f}",
    )
    axes[0].set_xlabel("Latent Consistency Loss")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Latent Consistency Distribution")
    axes[0].legend()

    axes[1].hist(recon_errors, bins=30, edgecolor="black", alpha=0.7, color="green")
    axes[1].axvline(
        np.mean(recon_errors),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(recon_errors):.3f}",
    )
    axes[1].set_xlabel("Reconstruction L1 Error")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Reconstruction Error Distribution")
    axes[1].legend()

    axes[2].hist(warp_errors, bins=30, edgecolor="black", alpha=0.7, color="orange")
    axes[2].axvline(
        np.mean(warp_errors),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(warp_errors):.3f}",
    )
    axes[2].set_xlabel("Image Warp L1 Error")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Warp Quality Distribution")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved to {save_path}")
    plt.close()

    # Print summary
    print("\n  Summary Statistics:")
    print(
        f"    Latent Consistency: {np.mean(consistency_scores):.4f} +/- {np.std(consistency_scores):.4f}"
    )
    print(
        f"    Reconstruction L1:  {np.mean(recon_errors):.4f} +/- {np.std(recon_errors):.4f}"
    )
    print(
        f"    Warp L1 Error:      {np.mean(warp_errors):.4f} +/- {np.std(warp_errors):.4f}"
    )

    return {
        "consistency": np.mean(consistency_scores),
        "recon_error": np.mean(recon_errors),
        "warp_error": np.mean(warp_errors),
    }


def visualize_loss_explanation(
    model, dataset, device, output_dir="./eval_outputs", sample_idx=None
):
    """
    Create visual explanations of all loss components used in Warp VAE training.
    Saves separate plots for each loss type in a dedicated subdirectory.

    Args:
        model: The VAE model
        dataset: The dataset to sample from
        device: torch device
        output_dir: Base output directory
        sample_idx: Specific sample index to use (random if None)
    """
    # Create subdirectory for loss explanations
    loss_dir = os.path.join(output_dir, "loss_explanation")
    os.makedirs(loss_dir, exist_ok=True)

    # Select sample
    if sample_idx is None:
        idx = np.random.randint(len(dataset))
    else:
        idx = sample_idx % len(dataset)

    print(f"\n[5] Creating loss explanation visualizations (sample idx={idx})...")
    print(f"    Output directory: {loss_dir}/")

    model.eval()
    sample = dataset[idx]

    with torch.no_grad():
        # Prepare inputs
        img_a = sample["image"].unsqueeze(0).to(device)
        img_b = sample["image_target"].unsqueeze(0).to(device)
        warp_ab = sample["warp_ab"].unsqueeze(0).to(device)
        warp_ba = sample["warp_ba"].unsqueeze(0).to(device)
        conf_ab = sample["confidence_ab"].unsqueeze(0).to(device)
        conf_ba = sample["confidence_ba"].unsqueeze(0).to(device)

        # Forward pass
        recon_a, posterior_a = model(img_a, sample_posterior=True)
        recon_b, posterior_b = model(img_b, sample_posterior=True)
        latent_a = posterior_a.sample()
        latent_b = posterior_b.sample()

        # Resize warps to latent resolution
        latent_h, latent_w = latent_a.shape[2:]
        warp_ab_latent = F.interpolate(
            warp_ab.permute(0, 3, 1, 2), size=(latent_h, latent_w),
            mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        warp_ba_latent = F.interpolate(
            warp_ba.permute(0, 3, 1, 2), size=(latent_h, latent_w),
            mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

        conf_ab_latent = F.interpolate(
            conf_ab.unsqueeze(1), size=(latent_h, latent_w),
            mode="bilinear", align_corners=False
        )
        conf_ba_latent = F.interpolate(
            conf_ba.unsqueeze(1), size=(latent_h, latent_w),
            mode="bilinear", align_corners=False
        )

        # Warp latents
        warped_latent_a = F.grid_sample(
            latent_a, warp_ab_latent, mode="bilinear",
            padding_mode="border", align_corners=False
        )
        warped_latent_b = F.grid_sample(
            latent_b, warp_ba_latent, mode="bilinear",
            padding_mode="border", align_corners=False
        )

        # Compute individual losses
        # 1. Reconstruction Loss
        recon_diff = (recon_a - img_a).abs()
        recon_loss = recon_diff.mean().item()

        # 2. KL Divergence (per-pixel visualization)
        kl_map = 0.5 * (posterior_a.mean ** 2 + posterior_a.var - 1 - posterior_a.logvar)
        kl_loss = kl_map.mean().item()

        # 3. Warp Consistency Loss (A->B)
        warp_diff_ab = (warped_latent_a - latent_b).abs()
        warp_diff_ab_weighted = warp_diff_ab * conf_ab_latent
        warp_loss_ab = warp_diff_ab_weighted.sum() / (conf_ab_latent.sum() + 1e-8)

        # 4. Warp Consistency Loss (B->A)
        warp_diff_ba = (warped_latent_b - latent_a).abs()
        warp_diff_ba_weighted = warp_diff_ba * conf_ba_latent
        warp_loss_ba = warp_diff_ba_weighted.sum() / (conf_ba_latent.sum() + 1e-8)

        # Bidirectional warp loss
        warp_loss_total = (warp_loss_ab + warp_loss_ba) / 2

    # === Plot 1: Input Images ===
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(tensor_to_numpy(img_a))
    axes[0].set_title("Image A (Source)", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(tensor_to_numpy(img_b))
    axes[1].set_title("Image B (Target)", fontsize=12)
    axes[1].axis("off")
    conf_np = conf_ab[0].cpu().numpy()
    im = axes[2].imshow(conf_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Confidence A->B (mean={conf_np.mean():.2f})", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(f"Input Image Pair (Sample idx={idx})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, "1_input_images.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved 1_input_images.png")

    # === Plot 2: Reconstruction Loss ===
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(tensor_to_numpy(img_a))
    axes[0].set_title("Input Image A", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(tensor_to_numpy(recon_a))
    axes[1].set_title("Reconstruction", fontsize=11)
    axes[1].axis("off")
    recon_diff_np = recon_diff[0].mean(dim=0).cpu().numpy()
    im = axes[2].imshow(recon_diff_np, cmap="hot", vmin=0)
    axes[2].set_title("|Recon - Input|", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].axis("off")
    axes[3].text(0.5, 0.5,
        f"RECONSTRUCTION LOSS\n\n"
        f"L_rec = mean(|x - x_hat|)\n\n"
        f"= {recon_loss:.4f}\n\n"
        f"Measures how well the VAE\n"
        f"reconstructs the input image.",
        ha="center", va="center", fontsize=12, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
    fig.suptitle("Reconstruction Loss", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, "2_reconstruction_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved 2_reconstruction_loss.png")

    # === Plot 3: KL Divergence Loss ===
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    latent_vis = latent_a[0, :3].permute(1, 2, 0).cpu().numpy()
    latent_vis = (latent_vis - latent_vis.min()) / (latent_vis.max() - latent_vis.min() + 1e-8)
    axes[0].imshow(latent_vis)
    axes[0].set_title("Latent z (ch 0-2)", fontsize=11)
    axes[0].axis("off")
    mean_vis = posterior_a.mean[0, :3].permute(1, 2, 0).cpu().numpy()
    mean_vis = (mean_vis - mean_vis.min()) / (mean_vis.max() - mean_vis.min() + 1e-8)
    axes[1].imshow(mean_vis)
    axes[1].set_title("Posterior mean", fontsize=11)
    axes[1].axis("off")
    var_vis = posterior_a.var[0].mean(dim=0).cpu().numpy()
    im = axes[2].imshow(var_vis, cmap="viridis")
    axes[2].set_title("Posterior variance", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].axis("off")
    axes[3].text(0.5, 0.5,
        f"KL DIVERGENCE LOSS\n\n"
        f"L_kl = 0.5 * (mu^2 + var\n"
        f"       - 1 - log(var))\n\n"
        f"= {kl_loss:.4f}\n\n"
        f"Weight in training: 1e-6\n"
        f"Regularizes latent to N(0,1)",
        ha="center", va="center", fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
    fig.suptitle("KL Divergence Loss", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, "3_kl_divergence_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved 3_kl_divergence_loss.png")

    # === Plot 4: Warp Consistency Loss (A->B) ===
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    latent_a_vis = latent_a[0, :3].permute(1, 2, 0).cpu().numpy()
    latent_a_vis = (latent_a_vis - latent_a_vis.min()) / (latent_a_vis.max() - latent_a_vis.min() + 1e-8)
    axes[0].imshow(latent_a_vis)
    axes[0].set_title("Latent A", fontsize=11)
    axes[0].axis("off")
    warped_a_vis = warped_latent_a[0, :3].permute(1, 2, 0).cpu().numpy()
    warped_a_vis = (warped_a_vis - warped_a_vis.min()) / (warped_a_vis.max() - warped_a_vis.min() + 1e-8)
    axes[1].imshow(warped_a_vis)
    axes[1].set_title("Warp(Latent A) -> B", fontsize=11)
    axes[1].axis("off")
    latent_b_vis = latent_b[0, :3].permute(1, 2, 0).cpu().numpy()
    latent_b_vis = (latent_b_vis - latent_b_vis.min()) / (latent_b_vis.max() - latent_b_vis.min() + 1e-8)
    axes[2].imshow(latent_b_vis)
    axes[2].set_title("Latent B (target)", fontsize=11)
    axes[2].axis("off")
    warp_diff_vis = warp_diff_ab[0].mean(dim=0).cpu().numpy()
    im = axes[3].imshow(warp_diff_vis, cmap="hot")
    axes[3].set_title("|Warp(z_A) - z_B|", fontsize=11)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    axes[4].axis("off")
    axes[4].text(0.5, 0.5,
        f"WARP CONSISTENCY (A->B)\n\n"
        f"L = sum(|warp(z_A) - z_B|\n"
        f"        * confidence)\n"
        f"    / sum(confidence)\n\n"
        f"= {warp_loss_ab.item():.4f}\n\n"
        f"Enforces 3D-aware latents",
        ha="center", va="center", fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightsalmon", alpha=0.8))
    fig.suptitle("Warp Consistency Loss (A -> B)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, "4_warp_consistency_ab.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved 4_warp_consistency_ab.png")

    # === Plot 5: Warp Consistency Loss (B->A) ===
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    latent_b_vis = latent_b[0, :3].permute(1, 2, 0).cpu().numpy()
    latent_b_vis = (latent_b_vis - latent_b_vis.min()) / (latent_b_vis.max() - latent_b_vis.min() + 1e-8)
    axes[0].imshow(latent_b_vis)
    axes[0].set_title("Latent B", fontsize=11)
    axes[0].axis("off")
    warped_b_vis = warped_latent_b[0, :3].permute(1, 2, 0).cpu().numpy()
    warped_b_vis = (warped_b_vis - warped_b_vis.min()) / (warped_b_vis.max() - warped_b_vis.min() + 1e-8)
    axes[1].imshow(warped_b_vis)
    axes[1].set_title("Warp(Latent B) -> A", fontsize=11)
    axes[1].axis("off")
    latent_a_vis = latent_a[0, :3].permute(1, 2, 0).cpu().numpy()
    latent_a_vis = (latent_a_vis - latent_a_vis.min()) / (latent_a_vis.max() - latent_a_vis.min() + 1e-8)
    axes[2].imshow(latent_a_vis)
    axes[2].set_title("Latent A (target)", fontsize=11)
    axes[2].axis("off")

    warp_diff_ba_vis = warp_diff_ba[0].mean(dim=0).cpu().numpy()
    im = axes[3].imshow(warp_diff_ba_vis, cmap="hot")
    axes[3].set_title("|Warp(z_B) - z_A|", fontsize=11)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    
    axes[4].axis("off")
    axes[4].text(0.5, 0.5,
        f"WARP CONSISTENCY (B->A)\n\n"
        f"L = sum(|warp(z_B) - z_A|\n"
        f"        * confidence)\n"
        f"    / sum(confidence)\n\n"
        f"= {warp_loss_ba.item():.4f}\n\n"
        f"Bidirectional for robustness",
        ha="center", va="center", fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightsalmon", alpha=0.8))
    fig.suptitle("Warp Consistency Loss (B -> A)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, "5_warp_consistency_ba.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved 5_warp_consistency_ba.png")

    # === Plot 6: Confidence Weighting ===
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    conf_ab_vis = conf_ab_latent[0, 0].cpu().numpy()
    im0 = axes[0].imshow(conf_ab_vis, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Confidence A->B (latent res)", fontsize=11)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    weighted_diff_ab = warp_diff_ab_weighted[0].mean(dim=0).cpu().numpy()
    im1 = axes[1].imshow(weighted_diff_ab, cmap="hot")
    axes[1].set_title("Weighted |Warp(A) - B|", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    conf_ba_vis = conf_ba_latent[0, 0].cpu().numpy()
    im2 = axes[2].imshow(conf_ba_vis, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Confidence B->A (latent res)", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    weighted_diff_ba = warp_diff_ba_weighted[0].mean(dim=0).cpu().numpy()
    im3 = axes[3].imshow(weighted_diff_ba, cmap="hot")
    axes[3].set_title("Weighted |Warp(B) - A|", fontsize=11)
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    fig.suptitle("Confidence Weighting (masks out uncertain regions)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, "6_confidence_weighting.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved 6_confidence_weighting.png")

    # === Plot 7: Total Loss Summary ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis("off")
    total_loss = recon_loss + 1e-6 * kl_loss + 0.5 * warp_loss_total.item()
    summary_text = f"""
WARP VAE TOTAL LOSS SUMMARY
{'=' * 40}

Sample Index: {idx}

LOSS COMPONENTS:
{'-' * 40}
1. Reconstruction Loss (L_rec):     {recon_loss:.4f}
   Formula: mean(|x - x_hat|)

2. KL Divergence Loss (L_kl):       {kl_loss:.4f}
   Formula: 0.5 * (mu^2 + var - 1 - log(var))
   Weight: 1e-6

3. Warp Consistency Loss (L_warp):  {warp_loss_total.item():.4f}
   - A->B: {warp_loss_ab.item():.4f}
   - B->A: {warp_loss_ba.item():.4f}
   Formula: (L_ab + L_ba) / 2
   Weight: 0.5

TOTAL LOSS:
{'-' * 40}
L_total = L_rec + 1e-6 * L_kl + 0.5 * L_warp
        = {recon_loss:.4f} + 1e-6 * {kl_loss:.4f} + 0.5 * {warp_loss_total.item():.4f}
        = {total_loss:.4f}
"""
    ax.text(0.5, 0.5, summary_text, ha="center", va="center", fontsize=12,
            family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))
    fig.suptitle("Total Loss Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, "7_total_loss_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved 7_total_loss_summary.png")

    print(f"    All loss explanation plots saved to: {loss_dir}/")

    return {
        "sample_idx": idx,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "warp_loss_ab": warp_loss_ab.item(),
        "warp_loss_ba": warp_loss_ba.item(),
        "warp_loss_total": warp_loss_total.item(),
    }


class HFVAEWrapper(torch.nn.Module):
    """Wrapper to make HuggingFace diffusers VAE compatible with our eval code."""

    def __init__(self, hf_vae):
        super().__init__()
        self.vae = hf_vae
        # Get scaling factor from the VAE config (default 0.18215 for SD)
        self.scale_factor = getattr(hf_vae.config, 'scaling_factor', 0.18215)

    def encode(self, x):
        """Encode image to latent distribution.

        Note: diffusers VAE expects input in [-1, 1] range (same as our code).
        The latent_dist is NOT scaled - we handle scaling in encode/decode.
        """
        posterior = self.vae.encode(x).latent_dist
        return posterior

    def decode(self, z):
        """Decode latent to image.

        Expects unscaled latents (raw from posterior.sample()).
        We apply the scaling factor here for proper decoding.
        """
        # Scale latents for decoding (divide by scale_factor = multiply by ~5.5)
        z_scaled = z / self.scale_factor
        decoded = self.vae.decode(z_scaled).sample
        return decoded

    def forward(self, x, sample_posterior=True):
        """Full forward pass: encode then decode."""
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        # decode() handles the scaling
        dec = self.decode(z)
        return dec, posterior


def load_model(checkpoint_path=None, use_vanilla_sd=False, hf_model_id=None):
    """Load the VAE model, optionally from checkpoint or HuggingFace.

    Args:
        checkpoint_path: Path to checkpoint file
        use_vanilla_sd: If True, load vanilla SD-VAE 2.1 architecture (ch=128, ch_mult=[1,2,4,4])
        hf_model_id: HuggingFace model ID to load VAE from (e.g., "stabilityai/stable-diffusion-2-1-base")
    """
    # If loading from HuggingFace, use diffusers VAE directly
    if hf_model_id:
        print(f"\nLoading VAE from HuggingFace: {hf_model_id}")
        from diffusers import AutoencoderKL as DiffusersAutoencoderKL

        # Load the VAE from HuggingFace
        hf_vae = DiffusersAutoencoderKL.from_pretrained(hf_model_id, subfolder="vae")
        print("  HuggingFace VAE loaded successfully")

        # Wrap it in a compatibility class
        return HFVAEWrapper(hf_vae)

    from ldm.models.autoencoder import AutoencoderKL

    if use_vanilla_sd:
        # Full Stable Diffusion VAE architecture (SD 2.1)
        print("\nUsing vanilla SD-VAE 2.1 architecture (ch=128, ch_mult=[1,2,4,4])")
        ddconfig = {
            "double_z": True,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }
    else:
        # Model config matching warp_vae_co3d_small
        ddconfig = {
            "double_z": True,
            "z_channels": 4,
            "resolution": 128,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 64,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }

    lossconfig = {
        "target": "ldm.modules.losses.LPIPSWithDiscriminator",
        "params": {
            "disc_start": 50001,
            "kl_weight": 0.000001,
            "disc_weight": 0.5,
            "perceptual_weight": 1.0,
            "disc_in_channels": 3,
            "disc_num_layers": 2 if not use_vanilla_sd else 3,
            "use_actnorm": False,
        },
    }

    model = AutoencoderKL(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        embed_dim=4,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            # Check for first_stage_model prefix (SD checkpoint format)
            if any("first_stage_model" in k for k in state_dict.keys()):
                print("  Detected SD checkpoint format (first_stage_model prefix)")
                state_dict = {
                    k.replace("first_stage_model.", ""): v
                    for k, v in state_dict.items()
                    if "first_stage_model" in k
                }
            # Check for model. prefix (our training format)
            elif any(k.startswith("model.") for k in state_dict.keys()):
                print("  Detected training checkpoint format (model. prefix)")
                state_dict = {
                    k.replace("model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("model.")
                }
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print("  Checkpoint loaded successfully")
    else:
        print("\nUsing randomly initialized model (no checkpoint)")

    return model


def find_latest_checkpoint(output_dir="./outputs/warp_vae_co3d_small"):
    """Find the latest checkpoint in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    # Look for checkpoint files
    checkpoints = list(output_path.glob("**/checkpoints/*.ckpt"))
    checkpoints += list(output_path.glob("**/*.ckpt"))

    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(checkpoints[0])


def main():
    parser = argparse.ArgumentParser(description="Evaluate Warp VAE training")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-detects if not specified)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--skip_model", action="store_true", help="Skip model-dependent visualizations"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Specific sample index for loss explanation (random if not specified)",
    )
    parser.add_argument(
        "--vanilla_sd",
        action="store_true",
        help="Use vanilla SD-VAE 2.1 architecture (ch=128, ch_mult=[1,2,4,4]). "
             "If no checkpoint specified, loads from sd_model/v1-5-pruned.ckpt",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default=None,
        help="HuggingFace model ID to load VAE from (e.g., 'stabilityai/stable-diffusion-2-1-base')",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Warp VAE Evaluation")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset()
    print(f"  Dataset size: {len(dataset)}")

    # Visualize dataset samples
    visualize_dataset_samples(
        dataset,
        num_samples=args.num_samples,
        save_path=os.path.join(args.output_dir, "dataset_samples.png"),
    )

    # Visualize warp flow
    visualize_warp_flow(
        dataset,
        num_samples=min(args.num_samples, 3),
        save_path=os.path.join(args.output_dir, "warp_flow.png"),
    )

    if not args.skip_model:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")

        if args.hf_model:
            # Load from HuggingFace
            print(f"Loading VAE from HuggingFace: {args.hf_model}")
            model = load_model(hf_model_id=args.hf_model)
        else:
            # Find or use specified checkpoint
            checkpoint = args.checkpoint
            if checkpoint is None:
                if args.vanilla_sd:
                    # Default to SD 1.5 checkpoint for vanilla SD architecture
                    checkpoint = "./sd_model/v1-5-pruned.ckpt"
                    print(f"\nUsing default SD checkpoint: {checkpoint}")
                else:
                    checkpoint = find_latest_checkpoint()
                    if checkpoint:
                        print(f"\nFound checkpoint: {checkpoint}")

            print("Using Model from Checkpoint:", checkpoint)
            model = load_model(checkpoint, use_vanilla_sd=args.vanilla_sd)

        model = model.to(device)
        model.eval()

        # Visualize model outputs
        visualize_model_outputs(
            model,
            dataset,
            device,
            num_samples=args.num_samples,
            save_path=os.path.join(args.output_dir, "model_outputs.png"),
        )

        # Compute latent consistency statistics
        stats = visualize_latent_consistency(
            model,
            dataset,
            device,
            num_samples=100,
            save_path=os.path.join(args.output_dir, "latent_consistency.png"),
        )

        # Visualize loss components explanation
        loss_stats = visualize_loss_explanation(
            model,
            dataset,
            device,
            output_dir=args.output_dir,
            sample_idx=args.sample_idx,
        )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Visualizations saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
