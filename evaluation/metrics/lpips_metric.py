"""
LPIPS perceptual similarity metric.
"""

import torch
from typing import Dict
from tqdm import tqdm


class LPIPSCalculator:
    """Calculate LPIPS perceptual similarity."""

    def __init__(self, model):
        """
        Initialize LPIPS calculator.

        Args:
            model: EQVAE model (contains LPIPS in loss module)
        """
        self.model = model
        # Extract LPIPS from model's loss module
        if hasattr(model, 'loss') and hasattr(model.loss, 'perceptual_loss'):
            self.lpips = model.loss.perceptual_loss
        else:
            raise ValueError("Model does not have LPIPS perceptual loss")

    def compute(self, dataloader, num_samples=None) -> Dict[str, float]:
        """
        Compute LPIPS over dataloader.

        Args:
            dataloader: DataLoader with validation data
            num_samples: Maximum number of samples (None for all)

        Returns:
            Dictionary with mean and std LPIPS
        """
        lpips_values = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="  Computing LPIPS")):
                if num_samples and batch_idx * dataloader.batch_size >= num_samples:
                    break

                images = batch['image'].to(next(self.model.parameters()).device)

                # Get reconstructions
                if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                    with self.model.ema_scope():
                        reconstructions, _ = self.model(images)
                else:
                    reconstructions, _ = self.model(images)

                # Compute LPIPS
                # LPIPS expects inputs in range [-1, 1]
                lpips_val = self.lpips(images, reconstructions)

                lpips_values.append(lpips_val.mean().item())

        # Aggregate
        import numpy as np
        lpips_mean = np.mean(lpips_values)
        lpips_std = np.std(lpips_values)

        metrics = {
            'mean': float(lpips_mean),
            'std': float(lpips_std),
        }

        print(f"  ✓ LPIPS: {lpips_mean:.4f} ± {lpips_std:.4f}")

        return metrics
