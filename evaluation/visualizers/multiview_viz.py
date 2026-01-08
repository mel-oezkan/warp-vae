"""
Multi-view consistency visualization for OmniObject's 24 views.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class MultiViewVisualizer:
    """Visualize multi-view consistency."""

    def __init__(self, model, device='cuda'):
        """
        Initialize multi-view visualizer.

        Args:
            model: EQVAE model
            device: Device to run on
        """
        self.model = model
        self.device = device

    def visualize_24_view_consistency(self, dataloader, save_path=None):
        """
        Visualize consistency across 24 views of an object.

        Note: This is a simplified version since dataloader provides single views.
        For full 24-view analysis, would need to modify dataset to load all views.

        Args:
            dataloader: DataLoader
            save_path: Path to save figure
        """
        # Get a batch of images
        batch = next(iter(dataloader))
        images = batch['image'].to(self.device)[:min(24, len(batch['image']))]
        num_views = images.size(0)

        # Extract latents
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'ema_scope') and hasattr(self.model, 'model_ema'):
                with self.model.ema_scope():
                    posterior = self.model.encode(images)
                    latents = posterior.sample()
            else:
                posterior = self.model.encode(images)
                latents = posterior.sample()

        # Flatten spatial dimensions
        latents_flat = latents.view(latents.size(0), latents.size(1), -1).mean(dim=2)  # [N, C]
        latents_np = latents_flat.cpu().numpy()

        # Create figure with 3 subplots
        fig = plt.figure(figsize=(18, 5))

        # 1. Cosine similarity heatmap
        ax1 = plt.subplot(1, 3, 1)
        similarity_matrix = cosine_similarity(latents_np)
        im = ax1.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        ax1.set_title('Latent Cosine Similarity Matrix', fontsize=12)
        ax1.set_xlabel('View index')
        ax1.set_ylabel('View index')
        plt.colorbar(im, ax=ax1, label='Cosine similarity')

        # 2. PCA trajectory
        ax2 = plt.subplot(1, 3, 2)
        if latents_np.shape[0] > 2:
            pca = PCA(n_components=2)
            latents_2d = pca.fit_transform(latents_np)

            # Plot trajectory
            ax2.plot(latents_2d[:, 0], latents_2d[:, 1], 'o-', alpha=0.6, linewidth=2)
            ax2.scatter(latents_2d[0, 0], latents_2d[0, 1], c='green', s=100, label='Start', zorder=5)
            ax2.scatter(latents_2d[-1, 0], latents_2d[-1, 1], c='red', s=100, label='End', zorder=5)

            # Annotate some points
            for i in [0, len(latents_2d)//2, len(latents_2d)-1]:
                ax2.annotate(f'{i}', (latents_2d[i, 0], latents_2d[i, 1]),
                           fontsize=8, ha='center')

            ax2.set_title(f'Latent Trajectory (PCA)\nVariance explained: {pca.explained_variance_ratio_.sum():.2%}',
                         fontsize=12)
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Pairwise distances
        ax3 = plt.subplot(1, 3, 3)
        distances = []
        for i in range(len(latents_np) - 1):
            dist = np.linalg.norm(latents_np[i] - latents_np[i+1])
            distances.append(dist)

        ax3.plot(distances, 'o-')
        ax3.set_title('Adjacent View Latent Distances', fontsize=12)
        ax3.set_xlabel('View transition')
        ax3.set_ylabel('L2 distance')
        ax3.grid(True, alpha=0.3)

        # Add statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        ax3.axhline(y=mean_dist, color='r', linestyle='--', alpha=0.5,
                   label=f'Mean: {mean_dist:.3f}')
        ax3.axhline(y=mean_dist + std_dist, color='orange', linestyle=':', alpha=0.5)
        ax3.axhline(y=mean_dist - std_dist, color='orange', linestyle=':', alpha=0.5)
        ax3.legend()

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight', dpi=150)
            plt.savefig(f"{save_path}.png", bbox_inches='tight', dpi=150)
            print(f"  ✓ Saved to {save_path}.pdf/.png")

        plt.close()

        # Compute and return statistics
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        avg_distance = np.mean(distances) if distances else 0.0

        return {
            'avg_cosine_similarity': float(avg_similarity),
            'avg_l2_distance': float(avg_distance),
            'num_views_analyzed': num_views
        }
