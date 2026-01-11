"""
Latent space visualizations: t-SNE, distributions, interpolations.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm


class LatentVisualizer:
    """Visualize latent space properties."""

    def __init__(self, model, device='cuda'):
        """
        Initialize latent visualizer.

        Args:
            model: EQVAE model
            device: Device to run on
        """
        self.model = model
        self.device = device

    def extract_latents(self, dataloader, num_samples=2000):
        """Extract latents from dataloader."""
        latents = []

        self.model.eval()
        samples_processed = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="  Extracting latents"):
                if samples_processed >= num_samples:
                    break

                images = batch['image'].to(self.device)

                # Encode
                if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                    with self.model.ema_scope():
                        posterior = self.model.encode(images)
                else:
                    posterior = self.model.encode(images)

                z = posterior.sample()

                # Flatten spatial dimensions
                z_flat = z.view(z.size(0), z.size(1), -1).mean(dim=2)  # [B, C]
                latents.append(z_flat.cpu())

                samples_processed += images.size(0)

        latents = torch.cat(latents, dim=0)
        return latents.numpy()

    def visualize_latent_tsne(self, dataloader, num_samples=2000, save_path=None):
        """
        Create t-SNE visualization of latent space.

        Args:
            dataloader: DataLoader
            num_samples: Number of samples
            save_path: Path to save figure
        """
        from sklearn.manifold import TSNE

        # Extract latents
        latents = self.extract_latents(dataloader, num_samples)

        # Compute t-SNE
        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latents_2d = tsne.fit_transform(latents)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            latents_2d[:, 0],
            latents_2d[:, 1],
            c=np.arange(len(latents_2d)),
            cmap='viridis',
            alpha=0.6,
            s=5
        )
        ax.set_title('t-SNE Projection of Latent Space', fontsize=14)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, label='Sample index')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=150)
            print(f"  ✓ Saved to {save_path}.pdf/.png")

        plt.close()

    def visualize_latent_distributions(self, dataloader, num_samples=2000, save_path=None):
        """
        Visualize latent distributions per channel.

        Args:
            dataloader: DataLoader
            num_samples: Number of samples
            save_path: Path to save figure
        """
        # Extract latents
        latents = self.extract_latents(dataloader, num_samples)

        num_channels = latents.shape[1]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Per-channel distributions
        ax = axes[0, 0]
        for i in range(min(num_channels, 4)):
            ax.hist(latents[:, i], bins=50, alpha=0.5, label=f'Channel {i}')
        ax.set_xlabel('Latent value')
        ax.set_ylabel('Frequency')
        ax.set_title('Per-Channel Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Mean and std per channel
        ax = axes[0, 1]
        means = latents.mean(axis=0)
        stds = latents.std(axis=0)
        ax.bar(range(num_channels), means, alpha=0.6, label='Mean')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Mean value')
        ax.set_title('Per-Channel Means')
        ax.grid(True, alpha=0.3)

        ax2 = axes[1, 0]
        ax2.bar(range(num_channels), stds, alpha=0.6, color='orange', label='Std')
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='N(0,1)')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Std value')
        ax2.set_title('Per-Channel Standard Deviations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 2D histogram (channel 0 vs channel 1)
        ax = axes[1, 1]
        if num_channels >= 2:
            ax.hist2d(latents[:, 0], latents[:, 1], bins=50, cmap='viridis')
            ax.set_xlabel('Channel 0')
            ax.set_ylabel('Channel 1')
            ax.set_title('Joint Distribution (Ch0 vs Ch1)')

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=150)
            print(f"  ✓ Saved to {save_path}.pdf/.png")

        plt.close()

    def visualize_interpolations(
        self,
        dataloader,
        num_pairs=4,
        num_steps=10,
        save_path=None
    ):
        """
        Visualize linear interpolations in latent space.

        Args:
            dataloader: DataLoader
            num_pairs: Number of image pairs to interpolate
            num_steps: Number of interpolation steps
            save_path: Path to save figure
        """
        # Get images
        batch = next(iter(dataloader))
        images = batch['image'].to(self.device)[:num_pairs * 2]

        self.model.eval()
        with torch.no_grad():
            # Encode
            if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                with self.model.ema_scope():
                    posterior = self.model.encode(images)
                    latents = posterior.sample()
            else:
                posterior = self.model.encode(images)
                latents = posterior.sample()

        # Create interpolations
        fig, axes = plt.subplots(num_pairs, num_steps + 2, figsize=(num_steps + 2, num_pairs))

        for pair_idx in range(num_pairs):
            z1 = latents[pair_idx * 2]
            z2 = latents[pair_idx * 2 + 1]

            # Show start image
            axes[pair_idx, 0].imshow(self._to_image(images[pair_idx * 2].cpu()))
            axes[pair_idx, 0].axis('off')
            if pair_idx == 0:
                axes[pair_idx, 0].set_title('Start', fontsize=8)

            # Interpolate
            alphas = np.linspace(0, 1, num_steps)
            for step_idx, alpha in enumerate(alphas):
                z_interp = (1 - alpha) * z1 + alpha * z2

                # Decode
                with torch.no_grad():
                    if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                        with self.model.ema_scope():
                            img_interp = self.model.decode(z_interp.unsqueeze(0))
                    else:
                        img_interp = self.model.decode(z_interp.unsqueeze(0))

                axes[pair_idx, step_idx + 1].imshow(self._to_image(img_interp[0].cpu()))
                axes[pair_idx, step_idx + 1].axis('off')
                if pair_idx == 0:
                    axes[pair_idx, step_idx + 1].set_title(f'{alpha:.1f}', fontsize=8)

            # Show end image
            axes[pair_idx, -1].imshow(self._to_image(images[pair_idx * 2 + 1].cpu()))
            axes[pair_idx, -1].axis('off')
            if pair_idx == 0:
                axes[pair_idx, -1].set_title('End', fontsize=8)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
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
