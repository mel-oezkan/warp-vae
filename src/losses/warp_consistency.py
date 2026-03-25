"""
Warp-based consistency losses for multi-view VAE training.

Provides loss functions that enforce consistency between VAE latent codes
across different views using dense correspondences from RoMaV2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class WarpConsistencyLoss(nn.Module):
    """
    Multi-view consistency loss using dense warp correspondences.

    Enforces that corresponding pixels in different views have similar
    latent representations in the VAE's encoding.

    The loss works in both directions:
    1. Warp latent A to B using warp_ab, compare with latent B
    2. Warp latent B to A using warp_ba, compare with latent A

    Args:
        loss_type: Type of loss ("l1", "l2", "cosine", "combined")
        bidirectional: Whether to compute loss in both directions
        confidence_weighted: Whether to weight loss by warp confidence
        confidence_threshold: Minimum confidence for loss computation
        normalize_latents: Whether to normalize latents before comparison
        feature_scale: Scale factor for feature-level vs pixel-level loss
    """

    def __init__(
        self,
        loss_type: str = "l1",
        bidirectional: bool = True,
        confidence_weighted: bool = True,
        confidence_threshold: float = 0.1,
        normalize_latents: bool = False,
        feature_scale: float = 1.0,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.bidirectional = bidirectional
        self.confidence_weighted = confidence_weighted
        self.confidence_threshold = confidence_threshold
        self.normalize_latents = normalize_latents
        self.feature_scale = feature_scale

    def warp_features(
        self,
        features: torch.Tensor,
        warp: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Warp feature map using dense correspondence field.

        Args:
            features: Feature tensor (B, C, H, W)
            warp: Warp field (B, H_w, W_w, 2) in normalized [-1, 1] coords
            confidence: Confidence map (B, H_w, W_w)

        Returns:
            Tuple of (warped_features, validity_mask)
        """
        B, C, H, W = features.shape

        # Resize warp to feature resolution if needed
        if warp.shape[1] != H or warp.shape[2] != W:
            warp = F.interpolate(
                warp.permute(0, 3, 1, 2),  # (B, 2, H_w, W_w)
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1)  # (B, H, W, 2)

            if confidence is not None:
                confidence = F.interpolate(
                    confidence.unsqueeze(1),  # (B, 1, H_w, W_w)
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)  # (B, H, W)

        # Apply grid sample for warping
        warped = F.grid_sample(
            features,
            warp,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )

        # Create validity mask based on confidence and in-bounds check
        if confidence is not None:
            validity_mask = confidence > self.confidence_threshold
        else:
            # Check if warp coordinates are within valid range
            validity_mask = (
                (warp[..., 0] >= -1) & (warp[..., 0] <= 1) &
                (warp[..., 1] >= -1) & (warp[..., 1] <= 1)
            )

        return warped, validity_mask

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss between predicted and target features.

        Args:
            pred: Predicted features (B, C, H, W)
            target: Target features (B, C, H, W)
            mask: Validity mask (B, H, W)
            confidence: Confidence weights (B, H, W)

        Returns:
            Scalar loss value
        """
        if self.normalize_latents:
            pred = F.normalize(pred, dim=1)
            target = F.normalize(target, dim=1)

        if self.loss_type == "l1":
            loss = torch.abs(pred - target)

        elif self.loss_type == "l2":
            loss = (pred - target) ** 2

        elif self.loss_type == "cosine":
            # Cosine similarity loss (1 - cos_sim)
            cos_sim = F.cosine_similarity(pred, target, dim=1)  # (B, H, W)
            loss = 1 - cos_sim
            loss = loss.unsqueeze(1)  # (B, 1, H, W) for consistent shape

        elif self.loss_type == "combined":
            # Combination of L1 and cosine
            l1_loss = torch.abs(pred - target)
            cos_sim = F.cosine_similarity(pred, target, dim=1).unsqueeze(1)
            loss = l1_loss + 0.5 * (1 - cos_sim)
            
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Average over channels
        loss = loss.mean(dim=1)  # (B, H, W)

        # Apply mask
        if mask is not None:
            loss = loss * mask.float()

        # Apply confidence weighting
        if self.confidence_weighted and confidence is not None:
            # Resize confidence to loss resolution if needed
            if confidence.shape[1:] != loss.shape[1:]:
                confidence = F.interpolate(
                    confidence.unsqueeze(1),
                    size=loss.shape[1:],
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)
            loss = loss * confidence

        # Compute mean, accounting for mask
        if mask is not None:
            num_valid = mask.float().sum()
            # Avoid NaN when no valid pixels
            if num_valid > 0:
                loss = loss.sum() / num_valid
            else:
                # Return zero loss if no valid pixels
                loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        else:
            loss = loss.mean()

        return loss

    def forward(
        self,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        warp_ab: torch.Tensor,
        warp_ba: torch.Tensor,
        confidence_ab: Optional[torch.Tensor] = None,
        confidence_ba: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute warp consistency loss between two latent representations.

        Args:
            latent_a: Latent code for image A (B, C, H, W)
            latent_b: Latent code for image B (B, C, H, W)
            warp_ab: Warp field A->B (B, H_w, W_w, 2)
            warp_ba: Warp field B->A (B, H_w, W_w, 2)
            confidence_ab: Confidence map A->B (B, H_w, W_w)
            confidence_ba: Confidence map B->A (B, H_w, W_w)

        Returns:
            Dictionary containing:
            - 'loss': Total consistency loss
            - 'loss_ab': Loss for A->B direction
            - 'loss_ba': Loss for B->A direction (if bidirectional)
        """
        # A->B: Warp latent A using warp_ab, compare with latent B
        warped_a_to_b, mask_ab = self.warp_features(latent_a, warp_ab, confidence_ab)
        loss_ab = self.compute_loss(warped_a_to_b, latent_b, mask_ab, confidence_ab)

        result = {
            "loss_ab": loss_ab,
        }

        if self.bidirectional:
            # B->A: Warp latent B using warp_ba, compare with latent A
            warped_b_to_a, mask_ba = self.warp_features(latent_b, warp_ba, confidence_ba)
            loss_ba = self.compute_loss(warped_b_to_a, latent_a, mask_ba, confidence_ba)

            result["loss_ba"] = loss_ba
            result["loss"] = (loss_ab + loss_ba) / 2.0
        else:
            result["loss"] = loss_ab

        return result


class WarpReconstructionLoss(nn.Module):
    """
    Warp-based image reconstruction loss.

    Instead of comparing reconstruction to input directly, this loss
    warps the reconstruction and compares to the corresponding view.

    This encourages the VAE to learn representations that are consistent
    across viewpoints.

    Args:
        loss_type: Type of loss ("l1", "l2", "perceptual")
        confidence_weighted: Weight by warp confidence
        perceptual_weight: Weight for perceptual loss component
    """

    def __init__(
        self,
        loss_type: str = "l1",
        confidence_weighted: bool = True,
        perceptual_weight: float = 0.0,
        lpips_model: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.confidence_weighted = confidence_weighted
        self.perceptual_weight = perceptual_weight

        # Optional LPIPS model for perceptual loss
        if perceptual_weight > 0 and lpips_model is not None:
            self.lpips = lpips_model
        else:
            self.lpips = None

    def forward(
        self,
        recon_a: torch.Tensor,
        image_b: torch.Tensor,
        warp_ab: torch.Tensor,
        confidence_ab: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute warp reconstruction loss.

        Warps reconstruction of A to match view B.

        Args:
            recon_a: Reconstruction of image A (B, C, H, W) using VAE
            image_b: Target image B (B, C, H, W)
            warp_ab: Warp field A->B (B, H_w, W_w, 2)
            confidence_ab: Confidence map (B, H_w, W_w)

        Returns:
            Dictionary with loss values
        """
        B, C, H, W = recon_a.shape

        # Resize warp to image resolution
        if warp_ab.shape[1] != H:
            warp_ab = F.interpolate(
                warp_ab.permute(0, 3, 1, 2),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1)

            if confidence_ab is not None:
                confidence_ab = F.interpolate(
                    confidence_ab.unsqueeze(1),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)

        # Warp reconstruction A to view B (uses flow fields)
        warped_recon = F.grid_sample(
            recon_a,
            warp_ab,
            mode="bilinear",
            padding_mode="border", 
            align_corners=False
        )

        # Compute pixel-wise loss
        if self.loss_type == "l1":
            pixel_loss = torch.abs(warped_recon - image_b)
        elif self.loss_type == "l2":
            pixel_loss = (warped_recon - image_b) ** 2
        else:
            pixel_loss = torch.abs(warped_recon - image_b)

        pixel_loss = pixel_loss.mean(dim=1)  # (B, H, W)

        # Apply confidence weighting
        if self.confidence_weighted and confidence_ab is not None:
            pixel_loss = pixel_loss * confidence_ab
            loss = pixel_loss.sum() / (confidence_ab.sum() + 1e-8)
        else:
            loss = pixel_loss.mean()

        result = {"loss": loss, "pixel_loss": loss}

        # Add perceptual loss
        if self.perceptual_weight > 0 and self.lpips is not None:
            with torch.no_grad():
                perceptual = self.lpips(warped_recon, image_b).mean()
            result["perceptual_loss"] = perceptual
            result["loss"] = loss + self.perceptual_weight * perceptual

        return result


class CycleConsistencyLoss(nn.Module):
    """
    Cycle consistency loss for warp validation.

    Ensures that warping A->B->A returns to the original location.
    This helps verify warp quality and can serve as a regularization.

    Args:
        tolerance: Pixel tolerance for cycle error
    """

    def __init__(self, tolerance: float = 2.0):
        super().__init__()
        self.tolerance = tolerance

    def forward(
        self,
        warp_ab: torch.Tensor,
        warp_ba: torch.Tensor,
        confidence_ab: Optional[torch.Tensor] = None,
        confidence_ba: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cycle consistency error.

        Args:
            warp_ab: Warp A->B (B, H, W, 2)
            warp_ba: Warp B->A (B, H, W, 2)
            confidence_ab: Confidence A->B (B, H, W)
            confidence_ba: Confidence B->A (B, H, W)

        Returns:
            Dictionary with cycle error metrics
        """
        B, H, W, _ = warp_ab.shape

        # Create identity grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=warp_ab.device),
            torch.linspace(-1, 1, W, device=warp_ab.device),
            indexing='ij'
        )
        identity = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Warp A->B->A
        # First apply warp_ab, then sample warp_ba at those locations
        warp_ba_expanded = warp_ba.permute(0, 3, 1, 2)  # (B, 2, H, W)
        cycle_warp = F.grid_sample(
            warp_ba_expanded,
            warp_ab,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        ).permute(0, 2, 3, 1)  # (B, H, W, 2)

        # Compute cycle error
        cycle_error = torch.norm(cycle_warp - identity, dim=-1)  # (B, H, W)

        # Weight by confidence
        if confidence_ab is not None and confidence_ba is not None:
            # Sample confidence_ba at warp_ab locations
            conf_ba_at_ab = F.grid_sample(
                confidence_ba.unsqueeze(1),
                warp_ab,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False
            ).squeeze(1)

            combined_conf = confidence_ab * conf_ba_at_ab
            weighted_error = (cycle_error * combined_conf).sum() / (combined_conf.sum() + 1e-8)
        else:
            weighted_error = cycle_error.mean()

        # Percentage within tolerance
        within_tolerance = (cycle_error < self.tolerance / max(H, W)).float().mean()

        return {
            "loss": weighted_error,
            "cycle_error": weighted_error,
            "within_tolerance": within_tolerance,
        }
