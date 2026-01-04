"""
Validation script for EQ-VAE implementation.

Tests equivariance property: decode(T(z)) ≈ T(decode(z))

Where T is a transformation (scaling or rotation), this script verifies that:
1. Transforming the latent code then decoding
2. Decoding then transforming the output

Produce similar results, demonstrating the model has learned equivariant representations.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from ldm.models.autoencoder import EQVAEAutoencoder


def validate_equivariance(model, x, transform_params):
    """
    Validate equivariance property.

    Compare two paths:
    1. encode(x) -> sample z -> transform z -> decode
    2. encode(x) -> sample z -> decode -> transform output

    They should be similar (not exact due to interpolation artifacts).

    Args:
        model: EQVAEAutoencoder model
        x: Input image tensor [B, C, H, W]
        transform_params: Dict with 'scale' and 'rotation' keys

    Returns:
        Dict with 'mse', 'path1', 'path2'
    """
    model.eval()

    with torch.no_grad():
        # Path 1: Transform latent then decode
        posterior = model.encode(x)
        z = posterior.mode()  # Use mode for deterministic behavior
        z_transformed = model._transform_latent(z, transform_params)
        x_recon_from_transformed_z = model.decode(z_transformed)

        # Path 2: Decode then transform output
        x_recon = model.decode(z)
        x_recon_transformed = model._transform_image(x_recon, transform_params)

    # Compute similarity (MSE)
    mse = torch.nn.functional.mse_loss(
        x_recon_from_transformed_z, x_recon_transformed
    )

    return {
        'mse': mse.item(),
        'path1': x_recon_from_transformed_z,
        'path2': x_recon_transformed,
        'original': x,
        'reconstruction': x_recon
    }


def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1] for visualization.

    Args:
        tensor: Image tensor

    Returns:
        Denormalized tensor
    """
    return (tensor + 1.0) / 2.0


def visualize_equivariance(config_path='config/eqvae_omniobject.yaml',
                          checkpoint_path=None,
                          save_dir='outputs/eqvae_validation'):
    """
    Create visualization comparing both equivariance paths.

    Args:
        config_path: Path to EQ-VAE config file
        checkpoint_path: Optional path to checkpoint (if None, uses untrained model)
        save_dir: Directory to save visualization
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Load configuration
    print(f"Loading config from {config_path}...")
    config = OmegaConf.load(config_path)

    # Instantiate model
    print("Instantiating EQVAEAutoencoder...")
    model = EQVAEAutoencoder(
        ddconfig=config.model.params.ddconfig,
        lossconfig=config.model.params.lossconfig,  # Pass config dict, not instantiated object
        embed_dim=config.model.params.embed_dim,
        p_prior=config.model.params.get('p_prior', 0.9),
        scale_range=config.model.params.get('scale_range', [0.25, 1.0]),
        use_rotation=config.model.params.get('use_rotation', True)
    )

    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print("Checkpoint loaded successfully")
    else:
        print("Warning: No checkpoint provided, using untrained model")

    model.eval()

    # Load sample image from OmniObject dataset
    print("Loading sample from OmniObject dataset...")
    try:
        from data_process.omniobject_dataset import OmniObjectDataset
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        dataset = OmniObjectDataset(
            data_dir=config.data.data_dir,
            transform=transform,
            patch_num=None,
            image_size=256,
            sample_mode='pairs'
        )

        sample = dataset[0]
        x = sample['image'].unsqueeze(0)  # [1, 3, 256, 256]
        print(f"Loaded sample: {sample.get('object_name', 'unknown')}")

    except Exception as e:
        print(f"Could not load from dataset: {e}")
        print("Using random tensor instead")
        x = torch.randn(1, 3, 256, 256) * 0.5  # Random image

    # Test different transformations
    test_cases = [
        {'scale': 0.5, 'rotation': 0, 'name': 'Scale 0.5x'},
        {'scale': 0.75, 'rotation': 0, 'name': 'Scale 0.75x'},
        {'scale': 1.0, 'rotation': 1, 'name': 'Rotate 90°'},
        {'scale': 0.5, 'rotation': 2, 'name': 'Scale 0.5x + Rotate 180°'},
    ]

    print(f"\nValidating equivariance for {len(test_cases)} transformations...")

    # Create figure
    fig, axes = plt.subplots(len(test_cases), 5, figsize=(20, 4*len(test_cases)))
    if len(test_cases) == 1:
        axes = axes[np.newaxis, :]

    for i, params in enumerate(test_cases):
        print(f"\nTest case {i+1}: {params['name']}")
        result = validate_equivariance(model, x, params)

        print(f"  MSE between paths: {result['mse']:.6f}")

        # Denormalize for visualization
        original = denormalize(result['original'][0]).cpu().permute(1, 2, 0).numpy()
        reconstruction = denormalize(result['reconstruction'][0]).cpu().permute(1, 2, 0).numpy()
        path1 = denormalize(result['path1'][0]).cpu().permute(1, 2, 0).numpy()
        path2 = denormalize(result['path2'][0]).cpu().permute(1, 2, 0).numpy()

        # Compute absolute difference
        diff = torch.abs(result['path1'] - result['path2'])[0].cpu()
        diff_normalized = (diff / diff.max()).permute(1, 2, 0).numpy()

        # Plot original
        axes[i, 0].imshow(original.clip(0, 1))
        axes[i, 0].set_title(f"Original Input", fontsize=10)
        axes[i, 0].axis('off')

        # Plot reconstruction (no transform)
        axes[i, 1].imshow(reconstruction.clip(0, 1))
        axes[i, 1].set_title(f"Reconstruction", fontsize=10)
        axes[i, 1].axis('off')

        # Plot path 1: decode(T(z))
        axes[i, 2].imshow(path1.clip(0, 1))
        axes[i, 2].set_title(f"Path 1: decode(T(z))\n{params['name']}", fontsize=10)
        axes[i, 2].axis('off')

        # Plot path 2: T(decode(z))
        axes[i, 3].imshow(path2.clip(0, 1))
        axes[i, 3].set_title(f"Path 2: T(decode(z))\n{params['name']}", fontsize=10)
        axes[i, 3].axis('off')

        # Plot difference
        axes[i, 4].imshow(diff_normalized, cmap='hot')
        axes[i, 4].set_title(f"Absolute Difference\nMSE: {result['mse']:.6f}", fontsize=10)
        axes[i, 4].axis('off')

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, 'eqvae_equivariance_validation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved validation plot to {save_path}")

    # Also save as PDF
    pdf_path = os.path.join(save_dir, 'eqvae_equivariance_validation.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved validation plot to {pdf_path}")

    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("Equivariance Validation Summary")
    print("="*60)
    for i, params in enumerate(test_cases):
        result = validate_equivariance(model, x, params)
        print(f"{params['name']:30s} MSE: {result['mse']:.6f}")
    print("="*60)

    if checkpoint_path is None:
        print("\nNote: This is an untrained model. MSE will be high.")
        print("After training, MSE should be < 0.01 for good equivariance.")
    else:
        print("\nInterpretation:")
        print("- MSE < 0.01: Excellent equivariance")
        print("- MSE < 0.1:  Good equivariance")
        print("- MSE > 0.1:  Poor equivariance (needs more training)")

    return save_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate EQ-VAE equivariance property')
    parser.add_argument('--config', type=str, default='config/eqvae_omniobject.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (optional)')
    parser.add_argument('--output-dir', type=str, default='outputs/eqvae_validation',
                       help='Directory to save validation outputs')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("EQ-VAE Equivariance Validation")
    print("="*60 + "\n")

    visualize_equivariance(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        save_dir=args.output_dir
    )

    print("\n✅ Validation complete!")
