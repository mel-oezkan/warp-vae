"""
Equivariance property metrics.

Tests whether transformations are preserved:
  T(decode(z)) ≈ decode(T(z))
"""

import torch
import torch.nn.functional as F
from typing import Dict
from tqdm import tqdm
import numpy as np


class EquivarianceMetrics:
    """Calculate equivariance consistency metrics."""

    def __init__(self, model, device='cuda'):
        """
        Initialize equivariance metrics calculator.

        Args:
            model: EQVAE model
            device: Device to run on
        """
        self.model = model
        self.device = device

    def _transform_image(self, images, scale=1.0, rotation=0):
        """
        Apply transformation to images.

        Args:
            images: [B, C, H, W] tensor
            scale: Scale factor
            rotation: Rotation in multiples of 90 degrees (0, 1, 2, 3)

        Returns:
            Transformed images
        """
        # Scale
        if scale != 1.0:
            h, w = images.shape[2:]
            new_h, new_w = int(h * scale), int(w * scale)
            images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)

            # Pad or crop to original size
            if scale < 1.0:
                # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                images = F.pad(images, (pad_w, pad_w, pad_h, pad_h))
                # Handle odd sizes
                if images.shape[2] > h:
                    images = images[:, :, :h, :]
                if images.shape[3] > w:
                    images = images[:, :, :, :w]
            else:
                # Crop from center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                images = images[:, :, start_h:start_h+h, start_w:start_w+w]

        # Rotate
        if rotation > 0:
            images = torch.rot90(images, k=rotation, dims=[2, 3])

        return images

    def _transform_latent(self, latents, scale=1.0, rotation=0):
        """
        Apply transformation to latents (same as model's _transform_latent).

        Args:
            latents: [B, C, H, W] tensor
            scale: Scale factor
            rotation: Rotation in multiples of 90 degrees

        Returns:
            Transformed latents
        """
        # Scale
        if scale != 1.0:
            h, w = latents.shape[2:]
            new_h, new_w = int(h * scale), int(w * scale)
            latents = F.interpolate(latents, size=(new_h, new_w), mode='bilinear', align_corners=False)

            # Pad or crop to original size
            if scale < 1.0:
                # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                latents = F.pad(latents, (pad_w, pad_w, pad_h, pad_h))
                # Handle odd sizes
                if latents.shape[2] > h:
                    latents = latents[:, :, :h, :]
                if latents.shape[3] > w:
                    latents = latents[:, :, :, :w]
            else:
                # Crop from center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                latents = latents[:, :, start_h:start_h+h, start_w:start_w+w]

        # Rotate
        if rotation > 0:
            latents = torch.rot90(latents, k=rotation, dims=[2, 3])

        return latents

    def test_transformation_equivariance(self, images, scale=1.0, rotation=0):
        """
        Test equivariance for a specific transformation.

        Args:
            images: [B, C, H, W] input images
            scale: Scale factor
            rotation: Rotation in multiples of 90 degrees

        Returns:
            L2 error between the two paths
        """
        with torch.no_grad():
            # Encode original images
            if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                with self.model.ema_scope():
                    posterior = self.model.encode(images)
                    z = posterior.sample()

                    # Path A: Transform image then encode-decode
                    images_transformed = self._transform_image(images, scale, rotation)
                    recon_path_a, _ = self.model(images_transformed)

                    # Path B: Transform latent then decode
                    z_transformed = self._transform_latent(z, scale, rotation)
                    recon_path_b = self.model.decode(z_transformed)
            else:
                posterior = self.model.encode(images)
                z = posterior.sample()

                # Path A: Transform image then encode-decode
                images_transformed = self._transform_image(images, scale, rotation)
                recon_path_a, _ = self.model(images_transformed)

                # Path B: Transform latent then decode
                z_transformed = self._transform_latent(z, scale, rotation)
                recon_path_b = self.model.decode(z_transformed)

            # Compute L2 error
            error = F.mse_loss(recon_path_a, recon_path_b)

        return error.item()

    def compute(self, dataloader, num_samples=500) -> Dict:
        """
        Compute equivariance metrics over dataloader.

        Args:
            dataloader: DataLoader with validation data
            num_samples: Number of samples to test

        Returns:
            Dictionary with equivariance errors
        """
        self.model.eval()

        # Test different scales
        scales = [0.25, 0.5, 0.75, 1.0]
        scale_errors = {f"{s:.2f}": [] for s in scales}

        # Test different rotations
        rotations = [0, 1, 2, 3]  # 0°, 90°, 180°, 270°
        rotation_errors = {f"{r*90}": [] for r in rotations}

        samples_processed = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="  Testing equivariance"):
                if samples_processed >= num_samples:
                    break

                images = batch['image'].to(self.device)
                batch_size = images.size(0)

                # Test scales
                for scale in scales:
                    error = self.test_transformation_equivariance(images, scale=scale, rotation=0)
                    scale_errors[f"{scale:.2f}"].append(error)

                # Test rotations
                for rotation in rotations:
                    error = self.test_transformation_equivariance(images, scale=1.0, rotation=rotation)
                    rotation_errors[f"{rotation*90}"].append(error)

                samples_processed += batch_size

        # Aggregate results
        scale_results = {k: float(np.mean(v)) for k, v in scale_errors.items()}
        rotation_results = {k: float(np.mean(v)) for k, v in rotation_errors.items()}

        # Combined statistics
        all_errors = []
        for v in scale_errors.values():
            all_errors.extend(v)
        for v in rotation_errors.values():
            all_errors.extend(v)

        metrics = {
            'scale': scale_results,
            'rotation': rotation_results,
            'combined_mean': float(np.mean(all_errors)),
            'combined_std': float(np.std(all_errors)),
            'combined_max': float(np.max(all_errors)),
        }

        print(f"  ✓ Scale equivariance errors: {scale_results}")
        print(f"  ✓ Rotation equivariance errors: {rotation_results}")
        print(f"  ✓ Combined mean error: {metrics['combined_mean']:.6f}")

        return metrics
