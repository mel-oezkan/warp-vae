"""
LPIPS perceptual similarity metric.

Works with any VAE variant via a reconstruct_fn callback.
"""

import torch
import numpy as np
from typing import Dict, Optional, Callable
from tqdm import tqdm

try:
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class LPIPSCalculator:
    """Calculate LPIPS perceptual similarity between inputs and reconstructions."""

    def __init__(
        self,
        device: str = 'cuda',
        reconstruct_fn: Optional[Callable] = None,
        net: str = 'vgg',
    ):
        """
        Args:
            device: Device to run on
            reconstruct_fn: Callable (model, images) -> reconstructions.
                           If None, uses model(images)[0].
            net: LPIPS backbone ('vgg' or 'alex')
        """
        self.device = device
        self.reconstruct_fn = reconstruct_fn

        if not LPIPS_AVAILABLE:
            raise ImportError("lpips package required. Install with: pip install lpips")
        self.lpips_fn = lpips_lib.LPIPS(net=net).to(device).eval()

    def _get_reconstructions(self, model, images):
        if self.reconstruct_fn is not None:
            return self.reconstruct_fn(model, images)
        reconstructions, *_ = model(images)
        return reconstructions

    def compute(self, dataloader, model, num_samples=None) -> Dict[str, float]:
        """Compute LPIPS over dataloader.

        Args:
            dataloader: DataLoader with validation data
            model: VAE model
            num_samples: Maximum number of samples (None for all)

        Returns:
            Dictionary with mean and std LPIPS
        """
        lpips_values = []

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="  Computing LPIPS")):
                if num_samples and batch_idx * dataloader.batch_size >= num_samples:
                    break

                images = batch['image'].to(self.device)
                reconstructions = self._get_reconstructions(model, images)

                # LPIPS expects [-1, 1]; clamp reconstructions
                reconstructions = reconstructions.clamp(-1, 1)

                lpips_val = self.lpips_fn(images, reconstructions)
                lpips_values.append(lpips_val.mean().item())

        lpips_mean = np.mean(lpips_values)
        lpips_std = np.std(lpips_values)

        metrics = {
            'mean': float(lpips_mean),
            'std': float(lpips_std),
        }

        print(f"  LPIPS: {lpips_mean:.4f} +/- {lpips_std:.4f}")
        return metrics
