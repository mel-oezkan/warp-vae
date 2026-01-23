"""
Model-specific VAE trainers.

Each trainer extends BaseVAETrainer with model-specific functionality:
- VanillaVAETrainer: Standard AutoencoderKL
- PluckerVAETrainer: PluckerAutoencoder with Plucker loss
- EQVAETrainer: EQVAEAutoencoder with equivariance regularization
- WarpVAETrainer: VAE with multi-view consistency via RoMaV2 warps
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional

from src.trainer.base_trainer import BaseVAETrainer
from src.losses.warp_consistency import WarpConsistencyLoss, WarpReconstructionLoss


class VanillaVAETrainer(BaseVAETrainer):
    """
    Trainer for standard AutoencoderKL (Vanilla VAE).
    
    This is the simplest trainer - just image reconstruction with KL regularization
    and discriminator loss.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 4.5e-6,
        ema_decay: Optional[float] = None,
        image_key: str = "image",
        log_images_every_n_steps: int = 500,
        checkpoint_path: Optional[str] = None,
        ignore_keys: List[str] = [],
    ):
        """
        Initialize Vanilla VAE trainer.
        
        Args:
            model_config: Config for AutoencoderKL instantiation
            learning_rate: Learning rate for optimizers
            ema_decay: EMA decay rate (None to disable)
            image_key: Key for images in batch
            log_images_every_n_steps: Image logging frequency
            checkpoint_path: Path to pretrained checkpoint
            ignore_keys: Keys to ignore when loading checkpoint
        """
        super().__init__(
            model_config=model_config,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
            image_key=image_key,
            log_images_every_n_steps=log_images_every_n_steps,
            checkpoint_path=checkpoint_path,
            ignore_keys=ignore_keys,
        )
        print("[VanillaVAETrainer] Initialized for AutoencoderKL training")
    
    def _get_model_output(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """
        Get model output (reconstruction and posterior).
        
        Returns:
            Tuple of (reconstructions, posterior)
        """
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self.model(inputs, sample_posterior=True)
        return reconstructions, posterior


class PluckerVAETrainer(BaseVAETrainer):
    """
    Trainer for PluckerAutoencoder.
    
    Extends base training with:
    - Plucker coordinate prediction from encoder
    - Hybrid Plucker loss (reconstruction + constraint + normalization)
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 4.5e-6,
        ema_decay: Optional[float] = None,
        image_key: str = "image",
        plucker_key: str = "plucker_coords",
        log_images_every_n_steps: int = 500,
        checkpoint_path: Optional[str] = None,
        ignore_keys: List[str] = [],
        plucker_loss_weight: float = 1.0,
    ):
        """
        Initialize Plucker VAE trainer.
        
        Args:
            model_config: Config for PluckerAutoencoder instantiation
            learning_rate: Learning rate for optimizers
            ema_decay: EMA decay rate (None to disable)
            image_key: Key for images in batch
            plucker_key: Key for ground truth Plucker coordinates in batch
            log_images_every_n_steps: Image logging frequency
            checkpoint_path: Path to pretrained checkpoint
            ignore_keys: Keys to ignore when loading checkpoint
            plucker_loss_weight: Weight for Plucker loss component
        """
        super().__init__(
            model_config=model_config,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
            image_key=image_key,
            log_images_every_n_steps=log_images_every_n_steps,
            checkpoint_path=checkpoint_path,
            ignore_keys=ignore_keys,
        )
        
        self.plucker_key = plucker_key
        self.plucker_loss_weight = plucker_loss_weight
        print(f"[PluckerVAETrainer] Initialized with plucker_key='{plucker_key}', weight={plucker_loss_weight}")
    
    def _get_model_output(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        """
        Get model output including Plucker predictions.
        
        Returns:
            Tuple of (reconstructions, posterior, predicted_plucker)
        """
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior, pred_plucker = self.model(inputs, sample_posterior=True)
        return reconstructions, posterior, pred_plucker
    
    def _compute_additional_losses(
        self,
        batch: Dict[str, Any],
        model_output: Tuple[Any, ...],
        split: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Plucker coordinate loss.
        
        Args:
            batch: Input batch with ground truth Plucker coordinates
            model_output: (reconstructions, posterior, pred_plucker)
            split: "train" or "val" for logging keys
            
        Returns:
            Tuple of (plucker_loss, log_dict)
        """
        pred_plucker = model_output[2]
        gt_plucker = batch[self.plucker_key]
        
        # Ensure gt_plucker is on correct device
        gt_plucker = gt_plucker.to(pred_plucker.device)
        
        # Use model's hybrid_plucker_loss method
        plucker_loss = self.model.hybrid_plucker_loss(pred_plucker, gt_plucker)
        
        weighted_loss = self.plucker_loss_weight * plucker_loss
        
        log_dict = {
            f"{split}/plucker_loss": plucker_loss.detach(),
            f"{split}/plucker_loss_weighted": weighted_loss.detach(),
        }
        
        return weighted_loss, log_dict
    
    def _get_additional_ae_params(self) -> List[torch.nn.Parameter]:
        """
        Include Plucker head parameters in autoencoder optimization.
        
        Returns:
            List of Plucker prediction head parameters
        """
        params = []
        
        # Add Plucker-specific components
        if hasattr(self.model, 'pluck_head'):
            params.extend(list(self.model.pluck_head.parameters()))
        if hasattr(self.model, 'pluck_norm_in'):
            params.extend(list(self.model.pluck_norm_in.parameters()))
        if hasattr(self.model, 'pluck_proj_layers'):
            params.extend(list(self.model.pluck_proj_layers.parameters()))
        if hasattr(self.model, 'pluck_proj_out'):
            params.extend(list(self.model.pluck_proj_out.parameters()))
        
        return params
    
    def _get_additional_log_images(
        self,
        batch: Dict[str, Any],
        model_output: Tuple[Any, ...]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate Plucker visualization images.
        
        Returns:
            Dictionary with Plucker coordinate visualizations
        """
        # TODO: Possible addition: Plucker ray visualization
        return {}


class EQVAETrainer(BaseVAETrainer):
    """
    Trainer for EQVAEAutoencoder (Equivariant VAE).
    
    Extends base training with:
    - Probabilistic equivariance regularization
    - Latent-space transformations (scaling, rotation)
    - Transformed target generation
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 4.5e-6,
        ema_decay: Optional[float] = None,
        image_key: str = "image",
        log_images_every_n_steps: int = 500,
        checkpoint_path: Optional[str] = None,
        ignore_keys: List[str] = [],
        # EQ-VAE specific
        p_prior: float = 0.9,
        equivariance_weight: float = 1.0,
    ):
        """
        Initialize EQ-VAE trainer.
        
        Args:
            model_config: Config for EQVAEAutoencoder instantiation
            learning_rate: Learning rate for optimizers
            ema_decay: EMA decay rate (None to disable)
            image_key: Key for images in batch
            log_images_every_n_steps: Image logging frequency
            checkpoint_path: Path to pretrained checkpoint
            ignore_keys: Keys to ignore when loading checkpoint
            p_prior: Probability of applying equivariance regularization
            equivariance_weight: Weight for equivariance loss
        """
        super().__init__(
            model_config=model_config,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
            image_key=image_key,
            log_images_every_n_steps=log_images_every_n_steps,
            checkpoint_path=checkpoint_path,
            ignore_keys=ignore_keys,
        )
        
        self.p_prior = p_prior
        self.equivariance_weight = equivariance_weight
        
        # Track whether current step uses equivariance
        self._use_eqvae_this_step = False
        self._current_transformed_target = None
        
        print(f"[EQVAETrainer] Initialized with p_prior={p_prior}, eq_weight={equivariance_weight}")
    
    def _get_model_output(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """
        Get model output, potentially with equivariance transforms.
        
        Randomly decides whether to apply EQ-VAE transformations based on p_prior.
        
        Returns:
            Tuple of (reconstructions, posterior)
        """
        inputs = self.get_input(batch, self.image_key)
        
        # Decide whether to apply equivariance
        self._use_eqvae_this_step = torch.rand(1).item() < self.p_prior
        
        if self._use_eqvae_this_step and self.training:
            # Use EQ-VAE forward with transformations
            reconstructions, posterior, transformed_target = self.model._eqvae_forward(inputs)
            self._current_transformed_target = transformed_target
        else:
            # Standard forward pass
            reconstructions, posterior = self.model(inputs, sample_posterior=True)
            self._current_transformed_target = None
        
        return reconstructions, posterior
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Training step with equivariance-aware loss computation.

        When EQ-VAE is active, the reconstruction target is the transformed input,
        not the original input.

        Uses manual optimization for dual optimizer setup.
        """
        opt_ae, opt_disc = self.optimizers()

        inputs = self.get_input(batch, self.image_key)
        model_output = self._get_model_output(batch)

        reconstructions = model_output[0]
        posterior = model_output[1]

        # Determine target for reconstruction loss
        if self._use_eqvae_this_step and self._current_transformed_target is not None:
            target = self._current_transformed_target
        else:
            target = inputs

        # ========== Optimize Autoencoder ==========
        # Autoencoder loss (against appropriate target)
        aeloss, log_dict_ae = self.model.loss(
            target,  # Use transformed target if EQ-VAE active
            reconstructions,
            posterior,
            0,  # optimizer_idx for autoencoder
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        # Optimize autoencoder
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        # Log EQ-VAE status
        self.log("train/eqvae_active", float(self._use_eqvae_this_step), sync_dist=False)
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)

        # ========== Optimize Discriminator ==========
        # Discriminator loss
        discloss, log_dict_disc = self.model.loss(
            target,
            reconstructions,
            posterior,
            1,  # optimizer_idx for discriminator
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        # Optimize discriminator
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        # Log discriminator losses
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)

        return aeloss
    
    def _validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        postfix: str = ""
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step - always uses standard forward (no transforms).
        """
        # Disable EQ-VAE transforms during validation
        self._use_eqvae_this_step = False
        self._current_transformed_target = None
        
        return super()._validation_step(batch, batch_idx, postfix)
    
    def _get_additional_log_images(
        self,
        batch: Dict[str, Any],
        model_output: Tuple[Any, ...]
    ) -> Dict[str, torch.Tensor]:
        """
        Add transformed target to logged images when EQ-VAE active.
        """
        log = {}
        if self._current_transformed_target is not None:
            log["transformed_target"] = self._current_transformed_target

        return log


class WarpVAETrainer(BaseVAETrainer):
    """
    Trainer for VAE with multi-view consistency via RoMaV2 warps.

    Extends base training with:
    - Paired image processing (source and target views)
    - Latent space warp consistency loss
    - Optional image-space warp reconstruction loss
    - Bidirectional warp supervision

    The key idea is that the latent representations of corresponding
    pixels across views (as determined by RoMaV2 dense correspondences)
    should be similar, encouraging the VAE to learn 3D-aware features.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 4.5e-6,
        ema_decay: Optional[float] = None,
        image_key: str = "image",
        target_key: str = "image_target",
        log_images_every_n_steps: int = 500,
        checkpoint_path: Optional[str] = None,
        ignore_keys: List[str] = [],
        # Warp-specific parameters
        warp_consistency_weight: float = 1.0,
        warp_reconstruction_weight: float = 0.0,
        consistency_loss_type: str = "l1",
        bidirectional: bool = True,
        confidence_weighted: bool = True,
        loss_confidence_threshold: float = 0.1,
        warmup_steps: int = 0,
        vanilla_probability: float = 0.0,
        # Gradient accumulation
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize Warp VAE trainer.

        Args:
            model_config: Config for AutoencoderKL instantiation
            learning_rate: Learning rate for optimizers
            ema_decay: EMA decay rate (None to disable)
            image_key: Key for source images in batch
            target_key: Key for target images in batch
            log_images_every_n_steps: Image logging frequency
            checkpoint_path: Path to pretrained checkpoint
            ignore_keys: Keys to ignore when loading checkpoint
            warp_consistency_weight: Weight for latent consistency loss
            warp_reconstruction_weight: Weight for image-space warp loss
            consistency_loss_type: Type of consistency loss ("l1", "l2", "cosine")
            bidirectional: Compute loss in both directions (A->B and B->A)
            confidence_weighted: Weight loss by RoMaV2 confidence
            loss_confidence_threshold: Minimum confidence for loss computation
            warmup_steps: Steps before enabling warp loss (for stability)
            vanilla_probability: Probability of using vanilla loss only (no warp loss)
            gradient_accumulation_steps: Number of batches to accumulate before updating
        """
        super().__init__(
            model_config=model_config,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
            image_key=image_key,
            log_images_every_n_steps=log_images_every_n_steps,
            checkpoint_path=checkpoint_path,
            ignore_keys=ignore_keys,
        )

        self.target_key = target_key
        self.warp_consistency_weight = warp_consistency_weight
        self.warp_reconstruction_weight = warp_reconstruction_weight
        self.warmup_steps = warmup_steps

        # Initialize warp consistency loss
        self.warp_consistency_loss = WarpConsistencyLoss(
            loss_type=consistency_loss_type,
            bidirectional=bidirectional,
            confidence_weighted=confidence_weighted,
            confidence_threshold=loss_confidence_threshold,
        )

        # Optional warp reconstruction loss
        if warp_reconstruction_weight > 0:
            self.warp_reconstruction_loss = WarpReconstructionLoss(
                loss_type="l1",
                confidence_weighted=confidence_weighted,
            )
        else:
            self.warp_reconstruction_loss = None

        # Probability of using vanilla loss only (skipping warp loss)
        self.vanilla_probability = vanilla_probability

        # Gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps

        print("[WarpVAETrainer] Initialized with:")
        print(f"  - warp_consistency_weight={warp_consistency_weight}")
        print(f"  - warp_reconstruction_weight={warp_reconstruction_weight}")
        print(f"  - consistency_loss_type={consistency_loss_type}")
        print(f"  - bidirectional={bidirectional}")
        print(f"  - warmup_steps={warmup_steps}")
        print(f"  - vanilla_probability={vanilla_probability}")
        print(f"  - gradient_accumulation_steps={gradient_accumulation_steps}")

    def _get_model_output(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """
        Get model output for source image.

        Returns:
            Tuple of (reconstructions, posterior)
        """
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self.model(inputs, sample_posterior=True)
        return reconstructions, posterior

    def _get_target_encoding(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Encode target image to latent space.

        Returns:
            Latent code for target image (B, C, H, W)
        """
        target = self.get_input(batch, self.target_key)
        posterior = self.model.encode(target)
        return posterior.sample()

    def _compute_warp_losses(
        self,
        batch: Dict[str, Any],
        latent_a: torch.Tensor,
        latent_b: torch.Tensor,
        recon_a: torch.Tensor,
        split: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute warp-based consistency losses.

        Args:
            batch: Input batch with warp fields
            latent_a: Latent code for source image
            latent_b: Latent code for target image
            recon_a: Reconstruction of source image
            split: "train" or "val" for logging

        Returns:
            Tuple of (total_warp_loss, log_dict)
        """
        # Extract warp fields from batch
        warp_ab = batch["warp_ab"].to(latent_a.device)
        warp_ba = batch["warp_ba"].to(latent_a.device)
        conf_ab = batch.get("confidence_ab")
        conf_ba = batch.get("confidence_ba")

        if conf_ab is not None:
            conf_ab = conf_ab.to(latent_a.device)
        if conf_ba is not None:
            conf_ba = conf_ba.to(latent_a.device)

        log_dict = {}
        total_loss = torch.tensor(0.0, device=latent_a.device)

        # Check warmup
        warp_factor = 1.0
        if self.warmup_steps > 0 and self.global_step < self.warmup_steps:
            warp_factor = self.global_step / self.warmup_steps

        # Latent consistency loss
        if self.warp_consistency_weight > 0:
            consistency_result = self.warp_consistency_loss(
                latent_a, latent_b,
                warp_ab, warp_ba,
                conf_ab, conf_ba
            )

            consistency_loss = consistency_result["loss"]
            weighted_consistency = self.warp_consistency_weight * warp_factor * consistency_loss
            total_loss = total_loss + weighted_consistency

            log_dict[f"{split}/warp_consistency_loss"] = consistency_loss.detach()
            log_dict[f"{split}/warp_consistency_weighted"] = weighted_consistency.detach()

            if "loss_ab" in consistency_result:
                log_dict[f"{split}/warp_consistency_ab"] = consistency_result["loss_ab"].detach()
            if "loss_ba" in consistency_result:
                log_dict[f"{split}/warp_consistency_ba"] = consistency_result["loss_ba"].detach()

        # Image-space warp reconstruction loss
        if self.warp_reconstruction_loss is not None and self.warp_reconstruction_weight > 0:
            target_img = self.get_input(batch, self.target_key)

            recon_result = self.warp_reconstruction_loss(
                recon_a, target_img, warp_ab, conf_ab
            )

            recon_loss = recon_result["loss"]
            weighted_recon = self.warp_reconstruction_weight * warp_factor * recon_loss
            total_loss = total_loss + weighted_recon

            log_dict[f"{split}/warp_recon_loss"] = recon_loss.detach()
            log_dict[f"{split}/warp_recon_weighted"] = weighted_recon.detach()

        log_dict[f"{split}/warp_factor"] = torch.tensor(warp_factor, device=latent_a.device)

        return total_loss, log_dict

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Training step with warp consistency and gradient accumulation.

        Processes paired images and enforces latent space consistency
        across views using RoMaV2 correspondences.

        With probability `vanilla_probability`, the warp loss is skipped
        and only the vanilla VAE loss is used.

        Gradient accumulation: Accumulates gradients over multiple batches
        before performing an optimizer step, effectively increasing the
        batch size without increasing memory usage.
        """
        opt_ae, opt_disc = self.optimizers()

        # Determine if this is an accumulation step or update step
        is_accumulating = (batch_idx + 1) % self.gradient_accumulation_steps != 0
        accum_steps = self.gradient_accumulation_steps

        # Get source image and encoding
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self.model(inputs, sample_posterior=True)
        latent_a = posterior.sample()

        # Decide whether to use vanilla loss only (skip warp loss)
        use_vanilla_only = torch.rand(1).item() < self.vanilla_probability

        # ========== Optimize Autoencoder ==========
        # Standard reconstruction loss
        aeloss, log_dict_ae = self.model.loss(
            inputs,
            reconstructions,
            posterior,
            0,  # optimizer_idx for autoencoder
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        # Compute warp loss only if not using vanilla mode
        if use_vanilla_only:
            warp_loss = torch.tensor(0.0, device=inputs.device)
            warp_log_dict = {
                "train/warp_consistency_loss": torch.tensor(0.0, device=inputs.device),
                "train/warp_factor": torch.tensor(0.0, device=inputs.device),
            }
        else:
            # Get target encoding
            latent_b = self._get_target_encoding(batch)
            # Warp consistency loss
            warp_loss, warp_log_dict = self._compute_warp_losses(
                batch, latent_a, latent_b, reconstructions, split="train"
            )

        total_ae_loss = aeloss + warp_loss

        # Scale loss for gradient accumulation
        scaled_ae_loss = total_ae_loss / accum_steps

        # Backward pass (accumulates gradients)
        self.manual_backward(scaled_ae_loss)

        # Only step optimizer after accumulating enough gradients
        if not is_accumulating:
            opt_ae.step()
            opt_ae.zero_grad()

        # Log losses (unscaled for interpretability)
        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log("train/warp_loss", warp_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log("train/total_ae_loss", total_ae_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log("train/vanilla_mode", float(use_vanilla_only), prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)
        # Filter out total_loss from log_dict_ae to avoid duplicate logging
        log_dict_ae_filtered = {k: v for k, v in log_dict_ae.items() if "total_loss" not in k}
        self.log_dict(log_dict_ae_filtered, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)
        self.log_dict(warp_log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)

        # Log memory usage periodically
        if batch_idx % 100 == 0:
            mem_stats = self.memory_profiler.snapshot(f"step_{self.global_step}")
            self.log("memory/allocated_mb", mem_stats.get('allocated_mb', 0), logger=True, sync_dist=False)

        # ========== Optimize Discriminator ==========
        discloss, log_dict_disc = self.model.loss(
            inputs,
            reconstructions,
            posterior,
            1,  # optimizer_idx for discriminator
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )

        # Scale discriminator loss for gradient accumulation
        scaled_disc_loss = discloss / accum_steps

        # Backward pass (accumulates gradients)
        self.manual_backward(scaled_disc_loss)

        # Only step optimizer after accumulating enough gradients
        if not is_accumulating:
            opt_disc.step()
            opt_disc.zero_grad()

        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)

        return total_ae_loss

    def _validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        postfix: str = ""
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step with warp consistency metrics.
        """
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self.model(inputs, sample_posterior=True)
        latent_a = posterior.sample()

        # Get target encoding
        latent_b = self._get_target_encoding(batch)

        # Standard losses
        aeloss, log_dict_ae = self.model.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split=f"val{postfix}",
        )

        discloss, log_dict_disc = self.model.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split=f"val{postfix}",
        )

        # Warp consistency loss
        warp_loss, warp_log_dict = self._compute_warp_losses(
            batch, latent_a, latent_b, reconstructions, split=f"val{postfix}"
        )

        # Log metrics
        self.log(f"val{postfix}/rec_loss", log_dict_ae.get(f"val{postfix}/rec_loss", aeloss), sync_dist=True)
        self.log(f"val{postfix}/warp_loss", warp_loss, sync_dist=True)
        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)
        self.log_dict(warp_log_dict, sync_dist=True)

        return {**log_dict_ae, **log_dict_disc, **warp_log_dict}

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict[str, Any],
        only_inputs: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate images for logging, including warped visualizations.
        """
        log = {}

        # Source image
        inputs = self.get_input(batch, self.image_key)
        log["source"] = inputs

        # Target image
        if self.target_key in batch:
            target = self.get_input(batch, self.target_key)
            log["target"] = target

        if not only_inputs:
            # Reconstruction
            reconstructions, posterior = self.model(inputs, sample_posterior=True)
            log["reconstruction"] = reconstructions

            # Warp visualization if available
            if "warp_ab" in batch:
                warp_ab = batch["warp_ab"].to(inputs.device)

                # Resize warp to image resolution
                H, W = inputs.shape[2:]
                if warp_ab.shape[1] != H:
                    warp_ab = F.interpolate(
                        warp_ab.permute(0, 3, 1, 2),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False
                    ).permute(0, 2, 3, 1)

                # Warp source to target view
                warped_source = F.grid_sample(
                    inputs,
                    warp_ab,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False
                )
                log["warped_source_to_target"] = warped_source

                # Warp reconstruction to target view
                warped_recon = F.grid_sample(
                    reconstructions,
                    warp_ab,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False
                )
                log["warped_recon_to_target"] = warped_recon

        return log