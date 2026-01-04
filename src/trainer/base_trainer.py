"""
Base trainer class for VAE models.

Provides common functionality for training different VAE variants:
- AutoencoderKL (Vanilla VAE)
- PluckerAutoencoder (Plucker-aware VAE)
- EQVAEAutoencoder (Equivariant VAE)
"""

import torch
import pytorch_lightning as pl
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple, List

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma


class BaseVAETrainer(pl.LightningModule):
    """
    Base trainer for VAE models.
    
    Handles common training infrastructure:
    - Model wrapping and forward passes
    - Optimizer configuration (dual optimizer for VAE + discriminator)
    - EMA (Exponential Moving Average) support
    - Logging and checkpointing
    - Image logging utilities
    
    Subclasses must implement:
    - _get_model_output(batch): Extract model-specific outputs
    - _compute_additional_losses(batch, model_output): Compute model-specific losses
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
        Initialize base VAE trainer.
        
        Args:
            model_config: Configuration dict for model instantiation
                         (passed to instantiate_from_config)
            learning_rate: Learning rate for optimizers
            ema_decay: EMA decay rate (None to disable EMA)
            image_key: Key for images in batch dict
            log_images_every_n_steps: Frequency of image logging
            checkpoint_path: Path to checkpoint for initialization
            ignore_keys: Keys to ignore when loading checkpoint
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])
        
        self.learning_rate = learning_rate
        self.image_key = image_key
        self.log_images_every_n_steps = log_images_every_n_steps
        
        # Instantiate the underlying VAE model
        self.model = instantiate_from_config(model_config)
        
        # Setup EMA if requested
        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            self.model_ema = LitEma(self.model, decay=ema_decay)
            print(f"[BaseVAETrainer] Using EMA with decay {ema_decay}")
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.init_from_ckpt(checkpoint_path, ignore_keys)
    
    def init_from_ckpt(self, path: str, ignore_keys: List[str] = []):
        """
        Load model weights from checkpoint.
        
        Args:
            path: Path to checkpoint file
            ignore_keys: List of key prefixes to ignore
        """
        sd = torch.load(path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in sd:
            sd = sd["state_dict"]
        
        # Filter ignored keys
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"[BaseVAETrainer] Deleting key {k} from state_dict")
                    del sd[k]
        
        # Handle 'model.' prefix if loading from trainer checkpoint
        model_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                model_sd[k[6:]] = v  # Remove 'model.' prefix
            else:
                model_sd[k] = v
        
        missing, unexpected = self.model.load_state_dict(model_sd, strict=False)
        if missing:
            print(f"[BaseVAETrainer] Missing keys: {missing}")
        if unexpected:
            print(f"[BaseVAETrainer] Unexpected keys: {unexpected}")
        print(f"[BaseVAETrainer] Restored from {path}")
    
    @contextmanager
    def ema_scope(self, context: Optional[str] = None):
        """
        Context manager for using EMA weights.
        
        Args:
            context: Optional context string for logging
        """
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"[EMA] {context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"[EMA] {context}: Restored training weights")
    
    def on_train_batch_end(self, *args, **kwargs):
        """Update EMA weights after each training batch."""
        if self.use_ema:
            self.model_ema(self.model)
    
    # ==================== Input Processing ====================
    
    def get_input(self, batch: Dict[str, Any], key: str) -> torch.Tensor:
        """
        Extract and preprocess input tensor from batch.
        
        Args:
            batch: Batch dictionary
            key: Key to extract
            
        Returns:
            Preprocessed tensor in (B, C, H, W) format
        """
        x = batch[key]
        
        # Add batch dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # Ensure (B, C, H, W) format and contiguous memory
        if x.shape[-1] in [1, 3, 4]:  # Likely (B, H, W, C)
            x = x.permute(0, 3, 1, 2)
        
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    # ==================== Abstract Methods ====================
    
    def _get_model_output(self, batch: Dict[str, Any]) -> Tuple[Any, ...]:
        """
        Get model output for a batch.
        
        Must be implemented by subclasses.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Tuple of model outputs (reconstruction, posterior, ...)
        """
        raise NotImplementedError("Subclasses must implement _get_model_output")
    
    def _compute_additional_losses(
        self, 
        batch: Dict[str, Any], 
        model_output: Tuple[Any, ...],
        split: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute model-specific additional losses.
        
        Override in subclasses for model-specific losses (e.g., Plucker loss).
        
        Args:
            batch: Input batch dictionary
            model_output: Output from _get_model_output
            split: "train" or "val" for logging
            
        Returns:
            Tuple of (additional_loss, log_dict)
        """
        return torch.tensor(0.0, device=self.device), {}
    
    # ==================== Training Steps ====================
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int, optimizer_idx: int = 0):
        """
        Execute one training step.
        
        Uses dual optimizer setup:
        - optimizer_idx=0: Autoencoder (encoder + decoder)
        - optimizer_idx=1: Discriminator
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            optimizer_idx: Which optimizer to use
            
        Returns:
            Loss tensor
        """
        inputs = self.get_input(batch, self.image_key)
        model_output = self._get_model_output(batch)
        
        # Unpack common outputs
        reconstructions = model_output[0]
        posterior = model_output[1]
        
        if optimizer_idx == 0:
            # Autoencoder loss
            aeloss, log_dict_ae = self.model.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            
            # Add model-specific losses
            additional_loss, additional_log_dict = self._compute_additional_losses(
                batch, model_output, split="train"
            )
            
            total_loss = aeloss + additional_loss
            
            # Log everything
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            if additional_log_dict:
                self.log_dict(additional_log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            
            return total_loss
        
        if optimizer_idx == 1:
            # Discriminator loss
            discloss, log_dict_disc = self.model.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            
            return discloss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Execute one validation step.
        
        Runs validation with both regular and EMA weights if EMA is enabled.
        """
        log_dict = self._validation_step(batch, batch_idx)
        
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
                log_dict.update(log_dict_ema)
        
        return log_dict
    
    def _validation_step(
        self, 
        batch: Dict[str, Any], 
        batch_idx: int, 
        postfix: str = ""
    ) -> Dict[str, torch.Tensor]:
        """
        Internal validation step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            postfix: Suffix for log keys (e.g., "_ema")
            
        Returns:
            Dictionary of logged metrics
        """
        inputs = self.get_input(batch, self.image_key)
        model_output = self._get_model_output(batch)
        
        reconstructions = model_output[0]
        posterior = model_output[1]
        
        # Autoencoder loss
        aeloss, log_dict_ae = self.model.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split=f"val{postfix}",
        )
        
        # Discriminator loss
        discloss, log_dict_disc = self.model.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split=f"val{postfix}",
        )
        
        # Additional losses
        additional_loss, additional_log_dict = self._compute_additional_losses(
            batch, model_output, split=f"val{postfix}"
        )
        
        # Log metrics
        self.log(f"val{postfix}/rec_loss", log_dict_ae.get(f"val{postfix}/rec_loss", aeloss))
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        if additional_log_dict:
            self.log_dict(additional_log_dict)
        
        return {**log_dict_ae, **log_dict_disc, **additional_log_dict}
    
    # ==================== Optimizer Configuration ====================
    
    def configure_optimizers(self):
        """
        Configure dual optimizers for VAE training.
        
        Returns:
            List of optimizers: [autoencoder_optimizer, discriminator_optimizer]
        """
        lr = self.learning_rate
        
        # Autoencoder parameters
        ae_params = (
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()) +
            list(self.model.quant_conv.parameters()) +
            list(self.model.post_quant_conv.parameters())
        )
        
        # Add any additional autoencoder parameters from subclass
        ae_params.extend(self._get_additional_ae_params())
        
        opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(
            self.model.loss.discriminator.parameters(), 
            lr=lr, 
            betas=(0.5, 0.9)
        )
        
        return [opt_ae, opt_disc], []
    
    def _get_additional_ae_params(self) -> List[torch.nn.Parameter]:
        """
        Get additional autoencoder parameters for optimization.
        
        Override in subclasses to include model-specific parameters.
        
        Returns:
            List of additional parameters
        """
        return []
    
    # ==================== Utilities ====================
    
    def get_last_layer(self) -> torch.Tensor:
        """Get the last layer weights for adaptive loss weighting."""
        return self.model.decoder.conv_out.weight
    
    @torch.no_grad()
    def log_images(
        self, 
        batch: Dict[str, Any], 
        only_inputs: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate images for logging.
        
        Args:
            batch: Input batch
            only_inputs: If True, only return input images
            
        Returns:
            Dictionary of images to log
        """
        log = {}
        inputs = self.get_input(batch, self.image_key)
        log["inputs"] = inputs
        
        if not only_inputs:
            model_output = self._get_model_output(batch)
            reconstructions = model_output[0]
            log["reconstructions"] = reconstructions
            
            # Add model-specific images
            log.update(self._get_additional_log_images(batch, model_output))
        
        return log
    
    def _get_additional_log_images(
        self, 
        batch: Dict[str, Any], 
        model_output: Tuple[Any, ...]
    ) -> Dict[str, torch.Tensor]:
        """
        Get additional images for logging.
        
        Override in subclasses for model-specific visualizations.
        
        Returns:
            Dictionary of additional images
        """
        return {}