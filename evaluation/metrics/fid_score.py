"""
FID (Fréchet Inception Distance) score computation.
"""

import torch
import numpy as np
from typing import Optional
from tqdm import tqdm
from scipy import linalg

try:
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    PYTORCH_FID_AVAILABLE = True
except ImportError:
    PYTORCH_FID_AVAILABLE = False
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")


class FIDCalculator:
    """Calculate FID score between real and reconstructed images."""

    def __init__(self, device='cuda', dims=2048):
        """
        Initialize FID calculator.

        Args:
            device: Device to run on
            dims: Dimensionality of Inception features (2048 for pool3)
        """
        if not PYTORCH_FID_AVAILABLE:
            raise ImportError("pytorch-fid is required. Install with: pip install pytorch-fid")

        self.device = device
        self.dims = dims

        # Load Inception model
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception = InceptionV3([block_idx]).to(device)
        self.inception.eval()

    def extract_features(self, images):
        """
        Extract Inception features from images.

        Args:
            images: Tensor of images [B, C, H, W] in range [-1, 1] or [0, 1]

        Returns:
            Features [B, dims]
        """
        with torch.no_grad():
            # Ensure images are in [0, 1] range for Inception
            if images.min() < 0:
                images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]

            # Resize to 299x299 for Inception
            if images.shape[2] != 299 or images.shape[3] != 299:
                images = torch.nn.functional.interpolate(
                    images,
                    size=(299, 299),
                    mode='bilinear',
                    align_corners=False
                )

            features = self.inception(images)[0]

            # Flatten
            if features.size(2) != 1 or features.size(3) != 1:
                features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=(1, 1))

            features = features.squeeze(3).squeeze(2)

        return features

    def compute_statistics(self, features):
        """
        Compute mean and covariance of features.

        Args:
            features: [N, dims] array of features

        Returns:
            mu: Mean vector
            sigma: Covariance matrix
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def compute_fid(self, dataloader, model, num_samples=5000):
        """
        Compute FID between real images and VAE reconstructions.

        Args:
            dataloader: DataLoader with real images
            model: VAE model
            num_samples: Number of samples to use

        Returns:
            FID score (float)
        """
        real_features = []
        recon_features = []

        model.eval()
        samples_processed = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="  Computing FID"):
                if samples_processed >= num_samples:
                    break

                images = batch['image'].to(self.device)
                batch_size = images.size(0)

                # Get reconstructions
                if hasattr(model, 'ema_scope') and hasattr(model, 'model_ema'):
                    with model.ema_scope():
                        reconstructions, _ = model(images)
                else:
                    reconstructions, _ = model(images)

                # Extract features
                real_feat = self.extract_features(images).cpu().numpy()
                recon_feat = self.extract_features(reconstructions).cpu().numpy()

                real_features.append(real_feat)
                recon_features.append(recon_feat)

                samples_processed += batch_size

        # Concatenate all features
        real_features = np.concatenate(real_features, axis=0)
        recon_features = np.concatenate(recon_features, axis=0)

        # Compute statistics
        mu_real, sigma_real = self.compute_statistics(real_features)
        mu_recon, sigma_recon = self.compute_statistics(recon_features)

        # Compute FID
        fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_recon, sigma_recon)

        print(f"  ✓ FID: {fid_score:.2f} (computed on {samples_processed} samples)")

        return float(fid_score)
