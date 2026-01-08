"""
Reconstruction visualization: side-by-side grids of original vs reconstructed images.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class ReconstructionVisualizer:
    """Visualize reconstruction quality."""

    def __init__(self, model, device='cuda'):
        """
        Initialize reconstruction visualizer.

        Args:
            model: EQVAE model
            device: Device to run on
        """
        self.model = model
        self.device = device

    def create_reconstruction_grid(
        self,
        dataloader,
        num_samples=16,
        save_path=None
    ):
        """
        Create grid showing original | reconstruction | difference.

        Args:
            dataloader: DataLoader with images
            num_samples: Number of samples to visualize (should be perfect square)
            save_path: Path to save figure (without extension)
        """
        # Get samples
        images_list = []
        recons_list = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)

                # Get reconstructions
                if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                    with self.model.ema_scope():
                        reconstructions, _ = self.model(images)
                else:
                    reconstructions, _ = self.model(images)

                images_list.append(images.cpu())
                recons_list.append(reconstructions.cpu())

                if len(images_list) * images.size(0) >= num_samples:
                    break

        # Concatenate
        images = torch.cat(images_list, dim=0)[:num_samples]
        recons = torch.cat(recons_list, dim=0)[:num_samples]

        # Compute differences
        diffs = torch.abs(images - recons)

        # Create figure
        n = int(np.sqrt(num_samples))
        fig, axes = plt.subplots(n, n * 3, figsize=(n * 6, n * 2))

        for i in range(n):
            for j in range(n):
                idx = i * n + j

                # Original
                ax_orig = axes[i, j * 3]
                img_orig = self._to_image(images[idx])
                ax_orig.imshow(img_orig)
                ax_orig.axis('off')
                if i == 0:
                    ax_orig.set_title('Original', fontsize=10)

                # Reconstruction
                ax_recon = axes[i, j * 3 + 1]
                img_recon = self._to_image(recons[idx])
                ax_recon.imshow(img_recon)
                ax_recon.axis('off')
                if i == 0:
                    ax_recon.set_title('Reconstruction', fontsize=10)

                # Difference
                ax_diff = axes[i, j * 3 + 2]
                img_diff = self._to_image(diffs[idx])
                ax_diff.imshow(img_diff, cmap='hot')
                ax_diff.axis('off')
                if i == 0:
                    ax_diff.set_title('Difference', fontsize=10)

        plt.tight_layout()

        # Save
        if save_path:
            save_path = Path(save_path)
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight', dpi=150)
            plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=150)
            print(f"  ✓ Saved to {save_path}.pdf/.png")

        plt.close()

    def _to_image(self, tensor):
        """Convert tensor to numpy image for visualization."""
        # tensor: [C, H, W] in range [-1, 1] or [0, 1]
        img = tensor.numpy()

        # Convert to [H, W, C]
        img = np.transpose(img, (1, 2, 0))

        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Handle grayscale
        if img.shape[2] == 1:
            img = img.squeeze(2)

        return img
