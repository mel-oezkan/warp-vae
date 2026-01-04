import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma


class AutoencoderKL(pl.LightningModule):
    """
    Base Variational Autoencoder with KL divergence regularization.

    This is the standard VAE implementation from Stable Diffusion,
    without any Plucker-specific modifications.
    """

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        ema_decay=None,
        learn_logvar=False,
    ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0.0 < ema_decay < 1.0
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        """
        Encode input image to VAE latent space.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            posterior: DiagonalGaussianDistribution for VAE latent
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        """
        Decode latent representation to image space.

        Args:
            z: Latent representation (B, embed_dim, H', W')

        Returns:
            Reconstructed image (B, C, H, W)
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        """
        Forward pass through VAE.

        Args:
            input: Input image tensor
            sample_posterior: Whether to sample from posterior or use mode

        Returns:
            dec: Reconstructed image
            posterior: Posterior distribution
        """
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log(
                "aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log(
                "discloss",
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params_list = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters())
        )
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
            if log_ema or self.use_ema:
                with self.ema_scope():
                    xrec_ema, posterior_ema = self(x)
                    if x.shape[1] > 3:
                        # colorize with random projection
                        assert xrec_ema.shape[1] > 3
                        xrec_ema = self.to_rgb(xrec_ema)
                    log["samples_ema"] = self.decode(
                        torch.randn_like(posterior_ema.sample())
                    )
                    log["reconstructions_ema"] = xrec_ema
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


class PluckerAutoencoder(AutoencoderKL):
    """
    AutoencoderKL extended with Plucker coordinate prediction.

    This class adds auxiliary Plucker ray prediction to the standard VAE,
    enabling camera-aware representation learning for multi-view consistency.
    """

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        n_patches: int,
        plucker_key: str = "pluck_ray",
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        ema_decay=None,
        learn_logvar=False,
        plucker_hidden_dim=512,
        plucker_dropout=0.1,
        plucker_weights=None,
    ):
        """
        Initialize Plucker-aware autoencoder.

        Args:
            ddconfig: Encoder/decoder configuration
            lossconfig: Loss function configuration
            embed_dim: Embedding dimension for VAE latent
            n_patches: Number of patches per dimension (e.g., 8 for 8x8 grid)
            plucker_key: Key for Plucker coordinates in batch dict
            plucker_hidden_dim: Hidden dimension for Plucker MLP
            plucker_dropout: Dropout rate for Plucker MLP
            plucker_weights: Dict with keys "recon", "constraint", "norm" for loss weights
        """
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=embed_dim,
            ckpt_path=None,  # Don't load checkpoint yet
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
            ema_decay=ema_decay,
            learn_logvar=learn_logvar,
        )

        self.n_patches = n_patches
        self.plucker_key = plucker_key

        # Plucker loss weights
        if plucker_weights is None:
            plucker_weights = {"recon": 1.0, "constraint": 0.1, "norm": 0.1}
        self.plucker_weights = plucker_weights

        # Plucker prediction head
        encoder_out_channels = 2 * ddconfig["z_channels"]
        self.pluck_head = torch.nn.Conv2d(encoder_out_channels, 6, kernel_size=1)
        self.act = torch.nn.SiLU()

        # Plucker MLP for refinement
        self.pluck_norm_in = torch.nn.LayerNorm(6)
        self.pluck_proj_layers = torch.nn.ModuleList(
            [
                self._make_projection_layer(6, plucker_hidden_dim, plucker_dropout),
                self._make_projection_layer(
                    plucker_hidden_dim, plucker_hidden_dim, plucker_dropout
                ),
            ]
        )
        self.pluck_proj_out = torch.nn.Linear(plucker_hidden_dim, 6)

        # Initialize with small weights for stability
        for module in self.pluck_proj_layers.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.pluck_proj_out.weight, gain=0.01)

        # Load checkpoint after Plucker components are initialized
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def _make_projection_layer(self, in_dim, out_dim, dropout_rate):
        """
        Create a projection layer with normalization and dropout.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            dropout_rate: Dropout probability

        Returns:
            Sequential module: Linear -> LayerNorm -> SiLU -> Dropout
        """
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
        )

    def encode(self, x):
        """
        Encode input image to VAE latent and Plucker coordinates.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            posterior: DiagonalGaussianDistribution for VAE latent
            pluck: Plucker coordinates (B, n_patches*n_patches, 6)
        """
        B = x.shape[0]
        h = self.encoder(x)

        # Generate Plucker coordinates from encoder output
        pluck = self.pluck_head(h)  # (B, 6, H, W)
        pluck = self.act(pluck)

        # Interpolate to n_patches × n_patches spatial size
        pluck = F.interpolate(
            pluck,
            size=(self.n_patches, self.n_patches),
            mode="bilinear",
            align_corners=False,
        )  # (B, 6, n_patches, n_patches)

        # Reshape to (B, n_patches*n_patches, 6) for MLP processing
        pluck = pluck.permute(0, 2, 3, 1).reshape(B, -1, 6).contiguous()

        # Apply MLP with normalization and dropout
        pluck = self.pluck_norm_in(pluck)

        # Apply projection layers sequentially
        for layer in self.pluck_proj_layers:
            pluck = layer(pluck)

        # Final projection to 6D Plucker space
        pluck = self.pluck_proj_out(pluck)  # (B, n_patches*n_patches, 6)

        # Generate VAE latent posterior
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        return posterior, pluck

    def forward(self, input, sample_posterior=True):
        """
        Forward pass through Plucker-aware VAE.

        Args:
            input: Input image tensor
            sample_posterior: Whether to sample from posterior or use mode

        Returns:
            dec: Reconstructed image
            posterior: Posterior distribution
            pluck: Predicted Plucker coordinates
        """
        posterior, pluck = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, pluck

    def hybrid_plucker_loss(self, pred, gt):
        """
        Compute hybrid Plucker loss with reconstruction, constraint, and normalization.

        The loss combines:
        1. Reconstruction: MSE between predicted and ground truth
        2. Constraint: Enforces d·m = 0 (Plucker constraint)
        3. Normalization: Encourages unit direction vectors

        Args:
            pred: Predicted Plucker coordinates (B, n_patches*n_patches, 6)
            gt: Ground truth Plucker coordinates (B, n_patches*n_patches, 6)

        Returns:
            Weighted combination of loss terms
        """
        pred_d, pred_m = pred[..., :3], pred[..., 3:]
        gt_d, gt_m = gt[..., :3], gt[..., 3:]

        # Reconstruction loss (MSE)
        loss_d = F.mse_loss(pred_d, gt_d)
        loss_m = F.mse_loss(pred_m, gt_m)
        recon_loss = loss_d + loss_m

        # Constraint: d·m = 0 (orthogonality)
        constraint_loss = torch.mean((pred_d * pred_m).sum(dim=-1) ** 2)

        # Normalization: encourage unit direction vectors
        norm_loss = F.mse_loss(
            torch.norm(pred_d, dim=-1), torch.ones_like(torch.norm(pred_d, dim=-1))
        )

        return (
            self.plucker_weights["recon"] * recon_loss
            + self.plucker_weights["constraint"] * constraint_loss
            + self.plucker_weights["norm"] * norm_loss
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        gt_plucker = batch[self.plucker_key]

        reconstructions, posterior, pred_ray = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            # Compute Plucker loss
            plucker_loss = self.hybrid_plucker_loss(pred_ray, gt_plucker.detach())

            total_loss = aeloss + plucker_loss

            self.log(
                "aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "plucker_loss",
                plucker_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return total_loss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log(
                "discloss",
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return discloss

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs = self.get_input(batch, self.image_key)
        gt_plucker = batch[self.plucker_key]
        reconstructions, posterior, pred_plucker = self(inputs)

        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        plucker_loss = self.hybrid_plucker_loss(pred_plucker, gt_plucker)

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log(f"val{postfix}/plucker_loss", plucker_loss)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


class EQVAEAutoencoder(AutoencoderKL):
    """
    Equivariant VAE extending AutoencoderKL with latent-space transformations.

    Implements the EQ-VAE approach: applies random scaling and rotation transformations
    to latent codes during training to learn equivariant representations.

    Key features:
    - Isotropic scaling transformations on latent codes
    - 90-degree rotation transformations
    - Probabilistic regularization (p_prior)
    - LPIPS + adversarial discriminator loss (via LPIPSWithDiscriminator)
    """

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        ema_decay=None,
        learn_logvar=False,
        # EQ-VAE specific parameters
        p_prior=0.9,              # Probability of applying equivariance loss
        scale_range=None,         # Isotropic scaling range [min, max]
        use_rotation=True,        # Enable 90-degree rotations
        equivariance_weight=1.0,  # Weight for equivariance loss term
    ):
        """
        Initialize Equivariant VAE.

        Args:
            ddconfig: Encoder/decoder configuration
            lossconfig: Loss function configuration (should be LPIPSWithDiscriminator)
            embed_dim: Embedding dimension for VAE latent
            ckpt_path: Path to checkpoint for initialization
            ignore_keys: Keys to ignore when loading checkpoint
            image_key: Key for images in batch dict
            colorize_nlabels: Number of labels for colorization
            monitor: Metric to monitor for checkpointing
            ema_decay: Exponential moving average decay rate
            learn_logvar: Whether to learn log variance
            p_prior: Probability of applying equivariance regularization (0 to 1)
            scale_range: List [min_scale, max_scale] for isotropic scaling (default: [0.25, 1.0])
            use_rotation: Whether to apply 90-degree rotations
            equivariance_weight: Weight for equivariance loss component
        """
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=embed_dim,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
            ema_decay=ema_decay,
            learn_logvar=learn_logvar,
        )

        # EQ-VAE parameters
        self.p_prior = p_prior
        self.scale_range = scale_range if scale_range is not None else [0.25, 1.0]
        self.use_rotation = use_rotation
        self.equivariance_weight = equivariance_weight

        # Verify loss is LPIPSWithDiscriminator
        from ldm.modules.losses import LPIPSWithDiscriminator
        if not isinstance(self.loss, LPIPSWithDiscriminator):
            print(f"Warning: EQVAEAutoencoder expects LPIPSWithDiscriminator loss, got {type(self.loss)}")

    def _sample_transformation(self):
        """
        Sample random transformation parameters.

        Returns:
            dict: {
                'scale': float in [scale_range[0], scale_range[1]],
                'rotation': int in [0, 1, 2, 3] (multiples of 90 degrees)
            }
        """
        # Sample scale uniformly from range
        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()

        # Sample rotation (0, 90, 180, or 270 degrees)
        rotation = 0
        if self.use_rotation:
            rotation = torch.randint(0, 4, (1,)).item()

        return {'scale': scale, 'rotation': rotation}

    def _transform_latent(self, z, transform_params):
        """
        Apply transformation to latent code.

        For scaling: interpolate latent feature map, then pad/crop to maintain shape
        For rotation: rotate latent feature map by 90-degree multiples

        Args:
            z: Latent code tensor [B, C, H, W]
            transform_params: Dict with 'scale' and 'rotation' keys

        Returns:
            Transformed latent code with same shape as input
        """
        z_out = z
        B, C, H, W = z.shape

        # Apply scaling
        if transform_params['scale'] != 1.0:
            new_size = int(H * transform_params['scale'])

            if new_size > 0:  # Ensure valid size
                # Interpolate to new size
                z_out = F.interpolate(z_out, size=(new_size, new_size),
                                      mode='bilinear', align_corners=False)

                # Pad or crop back to original size
                if new_size < H:
                    # Pad to original size
                    pad = (H - new_size) // 2
                    pad_remainder = (H - new_size) % 2
                    z_out = F.pad(z_out, (pad, pad + pad_remainder, pad, pad + pad_remainder),
                                  mode='constant', value=0)
                elif new_size > H:
                    # Center crop to original size
                    start = (new_size - H) // 2
                    z_out = z_out[:, :, start:start+H, start:start+W]

        # Apply rotation (90-degree multiples)
        if self.use_rotation and transform_params['rotation'] > 0:
            z_out = torch.rot90(z_out, k=transform_params['rotation'], dims=[2, 3])

        return z_out

    def _transform_image(self, x, transform_params):
        """
        Apply SAME transformation to input image (for target).

        Critical: Must match latent transformation exactly to ensure valid learning signal.

        Args:
            x: Input image tensor [B, C, H, W]
            transform_params: Dict with 'scale' and 'rotation' keys

        Returns:
            Transformed image with same shape as input
        """
        x_out = x
        B, C, H, W = x.shape

        # Apply scaling
        if transform_params['scale'] != 1.0:
            new_size = int(H * transform_params['scale'])

            if new_size > 0:  # Ensure valid size
                # Interpolate to new size
                x_out = F.interpolate(x_out, size=(new_size, new_size),
                                      mode='bilinear', align_corners=False)

                # Pad or crop back to original size
                if new_size < H:
                    # Pad to original size
                    pad = (H - new_size) // 2
                    pad_remainder = (H - new_size) % 2
                    x_out = F.pad(x_out, (pad, pad + pad_remainder, pad, pad + pad_remainder),
                                  mode='constant', value=0)
                elif new_size > H:
                    # Center crop to original size
                    start = (new_size - H) // 2
                    x_out = x_out[:, :, start:start+H, start:start+W]

        # Apply rotation
        if self.use_rotation and transform_params['rotation'] > 0:
            x_out = torch.rot90(x_out, k=transform_params['rotation'], dims=[2, 3])

        return x_out

    def _eqvae_forward(self, x):
        """
        EQ-VAE forward pass with latent transformations.

        Steps:
        1. Encode input to latent space
        2. Sample from posterior
        3. Apply random transformation to latent code
        4. Apply SAME transformation to input image (for target comparison)
        5. Decode transformed latent
        6. Return reconstruction and posterior

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Tuple of (reconstruction, posterior, transformed_input)
        """
        # Encode
        posterior = self.encode(x)
        z = posterior.sample()

        # Generate random transformation
        transform_params = self._sample_transformation()

        # Transform latent code
        z_transformed = self._transform_latent(z, transform_params)

        # Transform input image (for target)
        x_transformed = self._transform_image(x, transform_params)

        # Decode transformed latent
        dec = self.decode(z_transformed)

        # Return: reconstruction, posterior, and transformed input (as target)
        return dec, posterior, x_transformed

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """
        Training step with probabilistic equivariance regularization.

        With probability p_prior, applies EQ-VAE transformations.
        Otherwise, uses standard VAE training.

        Args:
            batch: Batch dict with 'image' key
            batch_idx: Batch index
            optimizer_idx: 0 for autoencoder, 1 for discriminator

        Returns:
            Loss tensor
        """
        inputs = self.get_input(batch, self.image_key)

        # Decide whether to apply equivariance regularization
        apply_eqvae = torch.rand(1).item() < self.p_prior

        if optimizer_idx == 0:  # Generator (autoencoder)
            if apply_eqvae:
                # EQ-VAE path: encode -> transform latent -> decode
                reconstructions, posterior, transformed_inputs = self._eqvae_forward(inputs)
                # Use transformed inputs as target
                target = transformed_inputs
            else:
                # Standard VAE path
                reconstructions, posterior = self(inputs)
                target = inputs

            # Compute losses (discriminator will be called internally by loss module)
            aeloss, log_dict_ae = self.loss(
                target, reconstructions, posterior, optimizer_idx,
                self.global_step, last_layer=self.get_last_layer(), split="train"
            )

            # Add equivariance flag to logging
            log_dict_ae["train/equivariance_applied"] = float(apply_eqvae)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True,
                          on_step=True, on_epoch=True)

            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:  # Discriminator
            # Standard forward pass (no transforms for discriminator)
            reconstructions, posterior = self(inputs)

            discloss, log_dict_disc = self.loss(
                inputs, reconstructions, posterior, optimizer_idx,
                self.global_step, last_layer=self.get_last_layer(), split="train"
            )

            self.log_dict(log_dict_disc, prog_bar=False, logger=True,
                          on_step=True, on_epoch=True)

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            return discloss

    def configure_optimizers(self):
        """
        Setup dual optimizers: one for autoencoder, one for discriminator.

        Returns:
            Tuple of ([opt_ae, opt_disc], [])
        """
        lr = self.learning_rate

        # Optimizer 0: Autoencoder (encoder + decoder + quant layers)
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )

        # Optimizer 1: Discriminator
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )

        return [opt_ae, opt_disc], []


class IdentityFirstStage(torch.nn.Module):
    """Identity first stage for compatibility."""

    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
