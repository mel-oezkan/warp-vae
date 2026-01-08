"""
Equivariance visualization: show transformation preservation.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class EquivarianceVisualizer:
    """Visualize equivariance properties."""

    def __init__(self, model, device='cuda'):
        """
        Initialize equivariance visualizer.

        Args:
            model: EQVAE model
            device: Device to run on
        """
        self.model = model
        self.device = device

    def _transform_image(self, images, scale=1.0, rotation=0):
        """Apply transformation to images."""
        # Scale
        if scale != 1.0:
            h, w = images.shape[2:]
            new_h, new_w = int(h * scale), int(w * scale)
            images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)

            if scale < 1.0:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                images = F.pad(images, (pad_w, pad_w, pad_h, pad_h))
                if images.shape[2] > h:
                    images = images[:, :, :h, :]
                if images.shape[3] > w:
                    images = images[:, :, :, :w]
            else:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                images = images[:, :, start_h:start_h+h, start_w:start_w+w]

        # Rotate
        if rotation > 0:
            images = torch.rot90(images, k=rotation, dims=[2, 3])

        return images

    def _transform_latent(self, latents, scale=1.0, rotation=0):
        """Apply transformation to latents."""
        # Scale
        if scale != 1.0:
            h, w = latents.shape[2:]
            new_h, new_w = int(h * scale), int(w * scale)
            latents = F.interpolate(latents, size=(new_h, new_w), mode='bilinear', align_corners=False)

            if scale < 1.0:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                latents = F.pad(latents, (pad_w, pad_w, pad_h, pad_h))
                if latents.shape[2] > h:
                    latents = latents[:, :, :h, :]
                if latents.shape[3] > w:
                    latents = latents[:, :, :, :w]
            else:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                latents = latents[:, :, start_h:start_h+h, start_w:start_w+w]

        # Rotate
        if rotation > 0:
            latents = torch.rot90(latents, k=rotation, dims=[2, 3])

        return latents

    def visualize_transformation_tests(
        self,
        dataloader,
        num_samples=6,
        save_path=None
    ):
        """
        Create grid showing equivariance tests.

        Rows: Different test images
        Columns: Original | PathA (transform image) | PathB (transform latent) | Difference

        Args:
            dataloader: DataLoader
            num_samples: Number of test images
            save_path: Path to save figure
        """
        # Get test images
        batch = next(iter(dataloader))
        images = batch['image'].to(self.device)[:num_samples]

        # Test transformations
        transformations = [
            ('Scale 0.5', {'scale': 0.5, 'rotation': 0}),
            ('Rotate 90°', {'scale': 1.0, 'rotation': 1}),
            ('Rotate 180°', {'scale': 1.0, 'rotation': 2}),
        ]

        num_transforms = len(transformations)

        # Create figure
        fig, axes = plt.subplots(
            num_samples,
            1 + num_transforms * 3,
            figsize=(4 + num_transforms * 6, num_samples * 2)
        )

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        self.model.eval()
        with torch.no_grad():
            for sample_idx in range(num_samples):
                img = images[sample_idx:sample_idx+1]

                # Show original
                axes[sample_idx, 0].imshow(self._to_image(img[0].cpu()))
                axes[sample_idx, 0].axis('off')
                if sample_idx == 0:
                    axes[sample_idx, 0].set_title('Original', fontsize=10)

                # Encode
                if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                    with self.model.ema_scope():
                        posterior = self.model.encode(img)
                        z = posterior.sample()
                else:
                    posterior = self.model.encode(img)
                    z = posterior.sample()

                for transform_idx, (name, params) in enumerate(transformations):
                    col_offset = 1 + transform_idx * 3

                    # Path A: Transform image then encode-decode
                    img_transformed = self._transform_image(img, **params)
                    if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                        with self.model.ema_scope():
                            recon_path_a, _ = self.model(img_transformed)
                    else:
                        recon_path_a, _ = self.model(img_transformed)

                    # Path B: Transform latent then decode
                    z_transformed = self._transform_latent(z, **params)
                    if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                        with self.model.ema_scope():
                            recon_path_b = self.model.decode(z_transformed)
                    else:
                        recon_path_b = self.model.decode(z_transformed)

                    # Compute difference
                    diff = torch.abs(recon_path_a - recon_path_b)

                    # Show Path A
                    axes[sample_idx, col_offset].imshow(self._to_image(recon_path_a[0].cpu()))
                    axes[sample_idx, col_offset].axis('off')
                    if sample_idx == 0:
                        axes[sample_idx, col_offset].set_title(f'{name}\nPath A', fontsize=9)

                    # Show Path B
                    axes[sample_idx, col_offset + 1].imshow(self._to_image(recon_path_b[0].cpu()))
                    axes[sample_idx, col_offset + 1].axis('off')
                    if sample_idx == 0:
                        axes[sample_idx, col_offset + 1].set_title(f'{name}\nPath B', fontsize=9)

                    # Show difference
                    axes[sample_idx, col_offset + 2].imshow(self._to_image(diff[0].cpu()), cmap='hot')
                    axes[sample_idx, col_offset + 2].axis('off')
                    if sample_idx == 0:
                        axes[sample_idx, col_offset + 2].set_title(f'{name}\nError', fontsize=9)

                    # Add error metric
                    error_val = F.mse_loss(recon_path_a, recon_path_b).item()
                    axes[sample_idx, col_offset + 2].text(
                        0.5, -0.1,
                        f'MSE: {error_val:.4f}',
                        transform=axes[sample_idx, col_offset + 2].transAxes,
                        ha='center',
                        fontsize=8
                    )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight', dpi=150)
            plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=150)
            print(f"  ✓ Saved to {save_path}.pdf/.png")

        plt.close()

    def _to_image(self, tensor):
        """Convert tensor to numpy image."""
        img = tensor.numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        if img.shape[2] == 1:
            img = img.squeeze(2)
        return img
