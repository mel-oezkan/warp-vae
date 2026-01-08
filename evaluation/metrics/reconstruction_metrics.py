"""
Reconstruction quality metrics: MSE, PSNR, SSIM
"""

import torch
import torch.nn.functional as F
from typing import Dict
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import numpy as np


class ReconstructionMetrics:
    """Calculate reconstruction quality metrics."""

    def __init__(self, model, device='cuda'):
        """
        Initialize reconstruction metrics calculator.

        Args:
            model: EQVAE model
            device: Device to run on
        """
        self.model = model
        self.device = device

    def compute(self, dataloader, num_samples=None) -> Dict[str, float]:
        """
        Compute reconstruction metrics over dataloader.

        Args:
            dataloader: DataLoader with validation data
            num_samples: Maximum number of samples (None for all)

        Returns:
            Dictionary with MSE, PSNR, SSIM metrics
        """
        mse_values = []
        ssim_values = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="  Computing reconstruction metrics")):
                if num_samples and batch_idx * dataloader.batch_size >= num_samples:
                    break

                images = batch['image'].to(self.device)

                # Get reconstructions
                if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                    with self.model.ema_scope():
                        reconstructions, _ = self.model(images)
                else:
                    reconstructions, _ = self.model(images)

                # Move to CPU for metric computation
                images_np = images.cpu().numpy()
                recons_np = reconstructions.cpu().numpy()

                # Compute metrics for each sample in batch
                for i in range(images.size(0)):
                    img = images_np[i]
                    rec = recons_np[i]

                    # MSE
                    mse = np.mean((img - rec) ** 2)
                    mse_values.append(mse)

                    # SSIM (per channel, then average)
                    ssim_per_channel = []
                    for c in range(img.shape[0]):
                        ssim_val = ssim(
                            img[c],
                            rec[c],
                            data_range=img[c].max() - img[c].min()
                        )
                        ssim_per_channel.append(ssim_val)
                    ssim_values.append(np.mean(ssim_per_channel))

        # Aggregate results
        mse_mean = np.mean(mse_values)
        ssim_mean = np.mean(ssim_values)

        # PSNR from MSE
        psnr = 10 * np.log10(1.0 / (mse_mean + 1e-10))

        metrics = {
            'mse': float(mse_mean),
            'psnr': float(psnr),
            'ssim': float(ssim_mean),
        }

        print(f"  ✓ MSE: {mse_mean:.6f}")
        print(f"  ✓ PSNR: {psnr:.2f} dB")
        print(f"  ✓ SSIM: {ssim_mean:.4f}")

        return metrics
