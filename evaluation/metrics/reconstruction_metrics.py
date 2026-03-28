"""
Reconstruction quality metrics: MSE, PSNR, SSIM

Works with any VAE variant via a reconstruct_fn callback.
"""

import torch
import numpy as np
from typing import Dict, Optional, Callable
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


class ReconstructionMetrics:
    """Calculate reconstruction quality metrics."""

    def __init__(self, device='cuda', reconstruct_fn: Optional[Callable] = None):
        """
        Args:
            device: Device to run on
            reconstruct_fn: Callable (model, images) -> reconstructions.
                           If None, uses model(images)[0].
        """
        self.device = device
        self.reconstruct_fn = reconstruct_fn

    def _get_reconstructions(self, model, images):
        if self.reconstruct_fn is not None:
            return self.reconstruct_fn(model, images)
        reconstructions, *_ = model(images)
        return reconstructions

    def compute(self, dataloader, model, num_samples=None) -> Dict[str, float]:
        """Compute reconstruction metrics over dataloader.

        Args:
            dataloader: DataLoader with validation data
            model: VAE model
            num_samples: Maximum number of samples (None for all)

        Returns:
            Dictionary with MSE, PSNR, SSIM metrics
        """
        mse_values = []
        ssim_values = []

        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="  Computing reconstruction metrics")):
                if num_samples and batch_idx * dataloader.batch_size >= num_samples:
                    break

                images = batch['image'].to(self.device)
                reconstructions = self._get_reconstructions(model, images)

                images_np = images.cpu().numpy()
                recons_np = reconstructions.clamp(-1, 1).cpu().numpy()

                for i in range(images.size(0)):
                    img = images_np[i]
                    rec = recons_np[i]

                    mse = np.mean((img - rec) ** 2)
                    mse_values.append(mse)

                    ssim_per_channel = []
                    for c in range(img.shape[0]):
                        ssim_val = ssim(
                            img[c], rec[c],
                            data_range=img[c].max() - img[c].min()
                        )
                        ssim_per_channel.append(ssim_val)
                    ssim_values.append(np.mean(ssim_per_channel))

        mse_mean = np.mean(mse_values)
        ssim_mean = np.mean(ssim_values)
        psnr = 10 * np.log10(1.0 / (mse_mean + 1e-10))

        metrics = {
            'mse': float(mse_mean),
            'psnr': float(psnr),
            'ssim': float(ssim_mean),
        }

        print(f"  MSE: {mse_mean:.6f} | PSNR: {psnr:.2f} dB | SSIM: {ssim_mean:.4f}")
        return metrics
