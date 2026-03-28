"""
FID (Frechet Inception Distance) and s-FID (spatial FID) score computation.

g-FID: Standard FID using final pooling features (2048-d) from Inception v3.
s-FID: Spatial FID using intermediate spatial features (7x7x768) from Inception v3,
       which better captures spatial/structural quality.
"""

import torch
import numpy as np
from typing import Optional, Dict, Callable
from tqdm import tqdm
from scipy import linalg

try:
    from pytorch_fid.inception import InceptionV3
    PYTORCH_FID_AVAILABLE = True
except ImportError:
    PYTORCH_FID_AVAILABLE = False
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Frechet distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    )


class FIDCalculator:
    """Calculate g-FID and s-FID between real and reconstructed images.

    g-FID uses the final pool3 features (2048-d global vector).
    s-FID uses mixed_6/conv spatial features (768 channels, 17x17 spatial),
    sampled over spatial positions to keep covariance tractable.
    """

    # Inception block indices from pytorch_fid
    BLOCK_IDX_POOL3 = 3       # Final pooling layer -> [B, 2048, 1, 1]
    BLOCK_IDX_SPATIAL = 2     # Mixed 6e conv features -> [B, 768, 17, 17]

    def __init__(
        self,
        device='cuda',
        reconstruct_fn: Optional[Callable] = None,
        spatial_samples_per_image: int = 16,
    ):
        """
        Args:
            device: Device to run on
            reconstruct_fn: Optional callable (model, images) -> reconstructions.
                           If None, uses model(images)[0].
        """
        if not PYTORCH_FID_AVAILABLE:
            raise ImportError("pytorch-fid is required. Install with: pip install pytorch-fid")

        self.device = device
        self.reconstruct_fn = reconstruct_fn
        self.spatial_samples_per_image = max(1, int(spatial_samples_per_image))

        # Load Inception with both pool3 (g-FID) and spatial (s-FID) outputs
        self.inception = InceptionV3(
            [self.BLOCK_IDX_SPATIAL, self.BLOCK_IDX_POOL3]
        ).to(device)
        self.inception.eval()

    def _get_reconstructions(self, model, images):
        """Get reconstructions using the configured reconstruct_fn or default."""
        if self.reconstruct_fn is not None:
            return self.reconstruct_fn(model, images)
        reconstructions, *_ = model(images)
        return reconstructions

    def extract_features(self, images):
        """Extract both global and spatial Inception features.

        Args:
            images: [B, C, H, W] in range [-1, 1] or [0, 1]

        Returns:
            (global_features [B, 2048], spatial_features [B*K, 768]) where
            K is the number of sampled spatial positions per image.
        """
        with torch.no_grad():
            if images.min() < 0:
                images = (images + 1) / 2

            if images.shape[2] != 299 or images.shape[3] != 299:
                images = torch.nn.functional.interpolate(
                    images, size=(299, 299), mode='bilinear', align_corners=False
                )

            outputs = self.inception(images)
            # outputs[0] = spatial features (block 2), outputs[1] = pool3 (block 3)
            spatial_feat = outputs[0]  # [B, 768, 17, 17]
            global_feat = outputs[1]   # [B, 2048, 1, 1]

            # Pool global features
            if global_feat.dim() == 4:
                if global_feat.size(2) != 1 or global_feat.size(3) != 1:
                    global_feat = torch.nn.functional.adaptive_avg_pool2d(
                        global_feat, output_size=(1, 1)
                    )
                global_feat = global_feat.squeeze(3).squeeze(2)  # [B, 2048]

            # For s-FID, treat sampled spatial positions as samples and channels as features.
            # This avoids constructing a huge covariance over flattened CxHxW vectors.
            b, c, h, w = spatial_feat.shape
            spatial_feat = spatial_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, HW, C]
            num_positions = h * w
            k = min(self.spatial_samples_per_image, num_positions)
            if k < num_positions:
                idx = torch.randperm(num_positions, device=spatial_feat.device)[:k]
                spatial_feat = spatial_feat[:, idx, :]
            spatial_feat = spatial_feat.reshape(b * k, c)  # [B*K, C]

        return global_feat, spatial_feat

    def compute_statistics(self, features):
        """Compute mean and covariance of features."""
        features = np.asarray(features, dtype=np.float64)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def compute(self, dataloader, model, num_samples=5000) -> Dict[str, float]:
        """Compute g-FID and s-FID between real images and VAE reconstructions.

        Args:
            dataloader: DataLoader with real images
            model: VAE model
            num_samples: Number of samples to use

        Returns:
            Dict with 'g_fid' and 's_fid' scores
        """
        real_global, real_spatial = [], []
        recon_global, recon_spatial = [], []

        model.eval()
        samples_processed = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="  Computing FID"):
                if samples_processed >= num_samples:
                    break

                images = batch['image'].to(self.device)
                batch_size = images.size(0)

                reconstructions = self._get_reconstructions(model, images)

                rg, rs = self.extract_features(images)
                recg, recs = self.extract_features(reconstructions)

                real_global.append(rg.cpu().numpy())
                real_spatial.append(rs.cpu().numpy())
                recon_global.append(recg.cpu().numpy())
                recon_spatial.append(recs.cpu().numpy())

                samples_processed += batch_size

        real_global = np.concatenate(real_global, axis=0)
        real_spatial = np.concatenate(real_spatial, axis=0)
        recon_global = np.concatenate(recon_global, axis=0)
        recon_spatial = np.concatenate(recon_spatial, axis=0)

        # g-FID (standard)
        mu_rg, sig_rg = self.compute_statistics(real_global)
        mu_recg, sig_recg = self.compute_statistics(recon_global)
        g_fid = calculate_frechet_distance(mu_rg, sig_rg, mu_recg, sig_recg)

        # s-FID (spatial)
        mu_rs, sig_rs = self.compute_statistics(real_spatial)
        mu_recs, sig_recs = self.compute_statistics(recon_spatial)
        s_fid = calculate_frechet_distance(mu_rs, sig_rs, mu_recs, sig_recs)

        print(f"  g-FID: {g_fid:.2f} | s-FID: {s_fid:.2f} ({samples_processed} samples)")

        return {'g_fid': float(g_fid), 's_fid': float(s_fid)}

    # Backward-compatible alias
    def compute_fid(self, dataloader, model, num_samples=5000):
        """Compute g-FID only (backward compatible)."""
        result = self.compute(dataloader, model, num_samples)
        return result['g_fid']
