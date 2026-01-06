"""
Model-specific VAE trainers.

Each trainer extends BaseVAETrainer with model-specific functionality:
- VanillaVAETrainer: Standard AutoencoderKL
- PluckerVAETrainer: PluckerAutoencoder with Plucker loss
- EQVAETrainer: EQVAEAutoencoder with equivariance regularization
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional

from src.trainer.base_trainer import BaseVAETrainer


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