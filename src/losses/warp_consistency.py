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
                align_corners=False,
            ).permute(0, 2, 3, 1)  # (B, H, W, 2)

            if confidence is not None:
                confidence = F.interpolate(
                    confidence.unsqueeze(1),  # (B, 1, H_w, W_w)
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)  # (B, H, W)

        # Apply grid sample for warping
        warped = F.grid_sample(
            features,
            warp,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        # Always check in-bounds, combine with confidence if available
        in_bounds = (
            (warp[..., 0] >= -1) & (warp[..., 0] <= 1) &
            (warp[..., 1] >= -1) & (warp[..., 1] <= 1)
        )
        if confidence is not None:
            validity_mask = (confidence > self.confidence_threshold) & in_bounds
        else:
            validity_mask = in_bounds

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
            num_valid = mask.float().sum().clamp(min=1.0)
            loss = loss.sum() / num_valid
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
        pixel_weight: Weight for pixel-level (L1/L2) loss component
        perceptual_weight: Weight for perceptual loss component
    """

    def __init__(
        self,
        loss_type: str = "l1",
        confidence_weighted: bool = True,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        lpips_model: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.confidence_weighted = confidence_weighted
        self.pixel_weight = pixel_weight
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

        # Compute in-bounds mask before warping
        in_bounds = (
            (warp_ab[..., 0] >= -1) & (warp_ab[..., 0] <= 1) &
            (warp_ab[..., 1] >= -1) & (warp_ab[..., 1] <= 1)
        )

        # Warp reconstruction A to view B (uses flow fields)
        warped_recon = F.grid_sample(
            recon_a,
            warp_ab,
            mode="bilinear",
            padding_mode="zeros",
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

        # Combine in-bounds mask with confidence weighting
        if self.confidence_weighted and confidence_ab is not None:
            mask = in_bounds.float() * confidence_ab
            pixel_loss = pixel_loss * mask
            loss = pixel_loss.sum() / (mask.sum() + 1e-8)
        else:
            pixel_loss = pixel_loss * in_bounds.float()
            num_valid = in_bounds.float().sum()
            loss = pixel_loss.sum() / (num_valid + 1e-8)

        result = {"pixel_loss": loss}

        # Combine pixel and perceptual losses
        total_loss = self.pixel_weight * loss

        if self.perceptual_weight > 0 and self.lpips is not None:
            perceptual = self.lpips(warped_recon, image_b).mean()
            result["perceptual_loss"] = perceptual
            total_loss = total_loss + self.perceptual_weight * perceptual

        result["loss"] = total_loss
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


class NaiveWarpConsistencyLoss(nn.Module):
    """
    Naive warp equivariance loss inspired by EQ-VAE.

    Instead of encoding two separate views and comparing their latents
    via warped correspondences (like WarpConsistencyLoss), this loss
    checks that encoding commutes with warping:

        encode(warp(image_a→b)) ≈ warp(encode(image_a))

    The left side warps the source image to the target view in pixel space,
    then encodes. The right side encodes the source image first, then warps
    the latent. If the encoder is truly 3D-aware, these should match.

    Args:
        loss_type: Type of loss ("l1", "l2", "cosine")
        confidence_weighted: Whether to weight loss by warp confidence
        confidence_threshold: Minimum confidence for loss computation
        bidirectional: Whether to compute loss in both directions
    """

    def __init__(
        self,
        loss_type: str = "l1",
        confidence_weighted: bool = True,
        confidence_threshold: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.confidence_weighted = confidence_weighted
        self.confidence_threshold = confidence_threshold
        self.bidirectional = bidirectional

    def _warp_tensor(
        self,
        tensor: torch.Tensor,
        warp: torch.Tensor,
    ) -> torch.Tensor:
        """Warp a tensor (image or latent) using a correspondence field."""
        _, _, H, W = tensor.shape
        if warp.shape[1] != H or warp.shape[2] != W:
            warp = F.interpolate(
                warp.permute(0, 3, 1, 2),
                size=(H, W),
                mode="nearest",
            ).permute(0, 2, 3, 1)

        return F.grid_sample(
            tensor, warp,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def _get_mask(
        self,
        warp: torch.Tensor,
        H: int, W: int,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute validity mask at the given spatial resolution."""
        if warp.shape[1] != H or warp.shape[2] != W:
            warp = F.interpolate(
                warp.permute(0, 3, 1, 2),
                size=(H, W),
                mode="nearest",
            ).permute(0, 2, 3, 1)
            if confidence is not None:
                confidence = F.interpolate(
                    confidence.unsqueeze(1),
                    size=(H, W),
                    mode="nearest",
                ).squeeze(1)

        in_bounds = (
            (warp[..., 0] >= -1) & (warp[..., 0] <= 1) &
            (warp[..., 1] >= -1) & (warp[..., 1] <= 1)
        )
        if confidence is not None:
            return (confidence > self.confidence_threshold) & in_bounds
        return in_bounds

    def _masked_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute masked & optionally confidence-weighted loss."""
        if self.loss_type == "l1":
            loss = torch.abs(pred - target)
        elif self.loss_type == "l2":
            loss = (pred - target) ** 2
        elif self.loss_type == "cosine":
            cos_sim = F.cosine_similarity(pred, target, dim=1)
            loss = (1 - cos_sim).unsqueeze(1)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        loss = loss.mean(dim=1)  # (B, H, W)
        loss = loss * mask.float()

        if self.confidence_weighted and confidence is not None:
            if confidence.shape[1:] != loss.shape[1:]:
                confidence = F.interpolate(
                    confidence.unsqueeze(1),
                    size=loss.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            loss = loss * confidence

        num_valid = mask.float().sum()
        if num_valid > 0:
            return loss.sum() / num_valid
        return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

    def _one_direction(
        self,
        image_src: torch.Tensor,
        latent_src: torch.Tensor,
        encoder_fn,
        warp: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss for one direction:
            encode(warp(image_src)) vs warp(encode(image_src))
        """
        # Left side: warp image in pixel space, then encode
        warped_image = self._warp_tensor(image_src, warp)
        posterior = encoder_fn(warped_image)
        latent_of_warped = posterior.sample()

        # Right side: warp the already-computed latent
        warped_latent = self._warp_tensor(latent_src, warp)

        # Mask at latent resolution
        _, _, H, W = latent_src.shape
        mask = self._get_mask(warp, H, W, confidence)

        return self._masked_loss(latent_of_warped, warped_latent, mask, confidence)

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        encoder_fn,
        warp_ab: torch.Tensor,
        warp_ba: torch.Tensor,
        confidence_ab: Optional[torch.Tensor] = None,
        confidence_ba: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute naive warp equivariance loss.

        Args:
            image_a: Source image (B, 3, H, W)
            image_b: Target image (B, 3, H, W)
            latent_a: Latent of source image (B, C, h, w)
            latent_b: Latent of target image (B, C, h, w)
            encoder_fn: model.encode callable (returns posterior)
            warp_ab: Warp field A→B (B, H_w, W_w, 2)
            warp_ba: Warp field B→A (B, H_w, W_w, 2)
            confidence_ab: Confidence map A→B (B, H_w, W_w)
            confidence_ba: Confidence map B→A (B, H_w, W_w)

        Returns:
            Dict with 'loss', 'loss_ab', and optionally 'loss_ba'
        """
        loss_ab = self._one_direction(
            image_a, latent_a, encoder_fn, warp_ab, confidence_ab,
        )

        result = {"loss_ab": loss_ab}

        if self.bidirectional:
            loss_ba = self._one_direction(
                image_b, latent_b, encoder_fn, warp_ba, confidence_ba,
            )
            result["loss_ba"] = loss_ba
            result["loss"] = (loss_ab + loss_ba) / 2.0
        else:
            result["loss"] = loss_ab

        return result
