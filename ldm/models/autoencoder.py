import random

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

    def forward(self, input, sample_posterior=True, return_latent=False):
        """
        Forward pass through VAE.

        Args:
            input: Input image tensor
            sample_posterior: Whether to sample from posterior or use mode
            return_latent: If True, also return the sampled latent z

        Returns:
            dec: Reconstructed image
            posterior: Posterior distribution
            z: (only if return_latent=True) The latent sample used for decoding
        """
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if return_latent:
            return dec, posterior, z
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

    Implements the EQ-VAE approach from zelaki/eqvae: applies random scaling and
    rotation transformations to latent codes during training to learn equivariant
    representations. Output spatial dimensions change with the transform (no
    padding/cropping).

    Key features:
    - Isotropic or anisotropic scaling transformations on latent codes
    - 90-degree rotation transformations (k ∈ {1,2,3}, never identity)
    - Probabilistic regularization (p_prior for EQ-VAE, p_prior_s for low-res prior preservation)
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
        p_prior=0.5,              # Probability of applying equivariance regularization
        p_prior_s=0.25,           # Probability of low-res prior preservation (when not EQ-VAE)
        anisotropic=False,        # Allow independent x/y scaling
        uniform_sample_scale=True, # Uniform scale sampling (discrete steps of 1/32)
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
            p_prior_s: Probability of low-res prior preservation when NOT applying EQ-VAE
            anisotropic: If True, sample independent x/y scales
            uniform_sample_scale: Use discrete scale steps (s/32 for s in 8..31)
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
        self.p_prior_s = p_prior_s
        self.anisotropic = anisotropic
        self.uniform_sample_scale = uniform_sample_scale
        self.use_rotation = use_rotation
        self.equivariance_weight = equivariance_weight

        # Discrete scale choices: {8/32, 9/32, ..., 31/32} = {0.25, ..., 0.96875}
        self._scale_choices = [s / 32 for s in range(8, 32)]

        # Verify loss is LPIPSWithDiscriminator
        from ldm.modules.losses import LPIPSWithDiscriminator
        if not isinstance(self.loss, LPIPSWithDiscriminator):
            print(f"Warning: EQVAEAutoencoder expects LPIPSWithDiscriminator loss, got {type(self.loss)}")

    def forward(self, input, scale=1, angle=0):
        """
        Forward pass with optional latent-space scaling and rotation.

        Matches the reference EQ-VAE implementation: scale and angle change the
        output spatial dimensions (no padding/cropping to maintain original size).

        Args:
            input: Input image tensor [B, C, H, W]
            scale: Scale factor (float or tuple for anisotropic). 1 = no scaling.
            angle: Rotation angle as k for rot90 (0, 1, 2, 3)

        Returns:
            Tuple of (reconstruction, posterior, z_transformed)
        """
        posterior = self.encode(input)
        z = posterior.sample()

        if scale != 1:
            z = F.interpolate(z, scale_factor=scale, mode='bilinear', align_corners=False)

        if angle != 0:
            z = torch.rot90(z, k=angle, dims=[-1, -2])

        dec = self.decode(z)
        return dec, posterior, z

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """
        Training step with probabilistic equivariance regularization.

        With probability p_prior, applies EQ-VAE transformations (scale + rotation)
        to both latent and ground truth. Otherwise, prior preservation: with
        probability p_prior_s, trains on downscaled images; else full resolution.

        Matches the reference zelaki/eqvae implementation.

        Args:
            batch: Batch dict with 'image' key
            batch_idx: Batch index
            optimizer_idx: 0 for autoencoder, 1 for discriminator

        Returns:
            Loss tensor
        """
        inputs = self.get_input(batch, self.image_key)

        # EQ-VAE regularization
        if random.random() < self.p_prior:
            mode = "latent"
            if self.anisotropic:
                scale_x = random.choice(self._scale_choices)
                scale_y = random.choice(self._scale_choices)
                scale = (scale_x, scale_y)
            else:
                scale = random.choice(self._scale_choices)

            # Rotation: k ∈ {1, 2, 3} (never identity)
            angle = random.choice([1, 2, 3])
            reconstructions, posterior, z_after = self(inputs, scale=scale, angle=angle)

            # Apply same transforms to ground truth
            inputs = F.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=False)
            inputs = torch.rot90(inputs, k=angle, dims=[-1, -2])

        # Prior preservation
        else:
            mode = "image"
            if random.random() < self.p_prior_s:
                # Low-resolution prior preservation
                scale = random.choice(self._scale_choices)
                inputs = F.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=False)
                reconstructions, posterior, _ = self(inputs)
            else:
                # Full-resolution standard forward
                scale = 1
                reconstructions, posterior, _ = self(inputs)

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(
                inputs, reconstructions, posterior, optimizer_idx,
                self.global_step, last_layer=self.get_last_layer(), split="train"
            )

            self.log(f"aeloss_scale-{scale}-{mode}", aeloss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True,
                          on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(
                inputs, reconstructions, posterior, optimizer_idx,
                self.global_step, last_layer=self.get_last_layer(), split="train"
            )

            self.log(f"discloss_scale-{scale}-{mode}", discloss,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True,
                          on_step=True, on_epoch=False)
            return discloss

    def _eqvae_forward(self, x):
        """
        EQ-VAE forward pass with latent transformations.

        Convenience method for use by EQVAETrainer. Samples random scale + rotation,
        applies to latent and ground truth image. Output dimensions change with the
        transform.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Tuple of (reconstruction, posterior, transformed_input)
        """
        if self.anisotropic:
            scale_x = random.choice(self._scale_choices)
            scale_y = random.choice(self._scale_choices)
            scale = (scale_x, scale_y)
        else:
            scale = random.choice(self._scale_choices)

        angle = random.choice([1, 2, 3])

        reconstructions, posterior, _ = self(x, scale=scale, angle=angle)

        # Apply same transforms to input for target
        x_transformed = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        x_transformed = torch.rot90(x_transformed, k=angle, dims=[-1, -2])

        return reconstructions, posterior, x_transformed

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            if random.random() < 0.5:
                xrec, posterior, _ = self(x)
            else:
                xrec, posterior, _ = self(x, scale=0.5)

            if x.shape[1] > 3:
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

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


# =============================================================================
# PluckerVAE Variants
# =============================================================================


class PluckerConditionedVAE(AutoencoderKL):
    """
    Variant 3: VAE conditioned on Plucker coordinates.

    Plucker rays are concatenated with the input image (9 channels total).
    The encoder learns from this combined input but does NOT predict Plucker rays.
    The decoder reconstructs both the image AND the Plucker rays.

    This variant tests whether conditioning alone improves 3D awareness.

    Input: image (3ch) + plucker (6ch) = 9 channels
    Output: reconstructed image (3ch) + reconstructed plucker (6ch)
    """

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        plucker_key: str = "plucker_coords",
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        ema_decay=None,
        learn_logvar=False,
        plucker_recon_weight: float = 0.5,
        plucker_constraint_weight: float = 0.1,
    ):
        """
        Initialize PluckerConditionedVAE.

        Args:
            ddconfig: Encoder/decoder config. Should have in_channels=9.
            lossconfig: Loss function configuration
            embed_dim: Embedding dimension for VAE latent
            plucker_key: Key for Plucker coordinates in batch dict
            plucker_recon_weight: Weight for Plucker reconstruction loss
            plucker_constraint_weight: Weight for Plucker constraints
        """
        # Validate input channels
        if ddconfig.get("in_channels", 3) != 9:
            print(f"Warning: PluckerConditionedVAE expects in_channels=9, "
                  f"got {ddconfig.get('in_channels', 3)}")

        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=embed_dim,
            ckpt_path=None,  # Load later
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
            ema_decay=ema_decay,
            learn_logvar=learn_logvar,
        )

        self.plucker_key = plucker_key
        self.plucker_recon_weight = plucker_recon_weight
        self.plucker_constraint_weight = plucker_constraint_weight

        # Get decoder output channel count (before final conv)
        # The decoder's final block_in = ch * ch_mult[0] = ch
        decoder_ch = ddconfig["ch"]

        # Multi-head output: image (3ch) + plucker (6ch)
        # Replace the default decoder conv_out with separate heads
        self.decoder_img_head = torch.nn.Conv2d(
            decoder_ch, 3, kernel_size=3, stride=1, padding=1
        )
        self.decoder_plucker_head = torch.nn.Conv2d(
            decoder_ch, 6, kernel_size=3, stride=1, padding=1
        )

        # Store original conv_out for reference, but don't use it
        self._original_conv_out = self.decoder.conv_out

        # Load checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def decode(self, z):
        """
        Decode latent to image and Plucker coordinates.

        Args:
            z: Latent representation (B, embed_dim, H', W')

        Returns:
            Tuple of (reconstructed_image, reconstructed_plucker)
        """
        z = self.post_quant_conv(z)

        # Run decoder with give_pre_end to get features before conv_out
        # We need to manually do the final steps
        h = self.decoder.conv_in(z)

        # Middle
        h = self.decoder.mid.block_1(h, None)
        h = self.decoder.mid.attn_1(h)
        h = self.decoder.mid.block_2(h, None)

        # Upsampling
        for i_level in reversed(range(self.decoder.num_resolutions)):
            for i_block in range(self.decoder.num_res_blocks + 1):
                h = self.decoder.up[i_level].block[i_block](h, None)
                if len(self.decoder.up[i_level].attn) > 0:
                    h = self.decoder.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.decoder.up[i_level].upsample(h)

        # Final normalization and activation
        h = self.decoder.norm_out(h)
        h = h * torch.sigmoid(h)  # swish/SiLU

        # Multi-head outputs
        recon_img = self.decoder_img_head(h)
        recon_plucker = self.decoder_plucker_head(h)

        return recon_img, recon_plucker

    def forward(self, image, plucker, sample_posterior=True):
        """
        Forward pass through PluckerConditionedVAE.

        Args:
            image: Input image tensor (B, 3, H, W)
            plucker: Input Plucker coordinates (B, 6, H, W)
            sample_posterior: Whether to sample from posterior or use mode

        Returns:
            Tuple of (recon_img, recon_plucker, posterior)
        """
        # Concatenate image and plucker for encoder input
        x = torch.cat([image, plucker], dim=1)  # (B, 9, H, W)

        # Encode
        posterior = self.encode(x)

        # Sample or use mode
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        # Decode to both outputs
        recon_img, recon_plucker = self.decode(z)

        return recon_img, recon_plucker, posterior

    def plucker_constraint_loss(self, plucker):
        """
        Compute Plucker geometric constraints.

        Args:
            plucker: Plucker coordinates (B, 6, H, W)

        Returns:
            Constraint loss (orthogonality + normalization)
        """
        d = plucker[:, :3]  # Direction (B, 3, H, W)
        m = plucker[:, 3:]  # Moment (B, 3, H, W)

        # Orthogonality: d . m = 0
        dot_product = (d * m).sum(dim=1)  # (B, H, W)
        ortho_loss = torch.mean(dot_product ** 2)

        # Unit direction: ||d|| = 1
        d_norm = torch.norm(d, dim=1)  # (B, H, W)
        norm_loss = F.mse_loss(d_norm, torch.ones_like(d_norm))

        return ortho_loss + norm_loss

    def get_last_layer(self):
        """Return image head weights for discriminator gradient scaling."""
        return self.decoder_img_head.weight

    def configure_optimizers(self):
        """Setup optimizers including multi-head decoder parameters."""
        lr = self.learning_rate
        ae_params_list = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters())
            + list(self.decoder_img_head.parameters())
            + list(self.decoder_plucker_head.parameters())
        )
        if self.learn_logvar:
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []


class DirectPluckerVAE(AutoencoderKL):
    """
    Variant 2: VAE with unified latent space predicting image + Plucker.

    Similar to PluckerConditionedVAE but conceptually treats Plucker
    as a direct prediction target alongside the image.

    Input: image (3ch) + plucker (6ch) = 9 channels
    Output: reconstructed image (3ch) + reconstructed plucker (6ch)

    Key difference from Variant 3: The loss formulation emphasizes
    the Plucker prediction as an auxiliary task rather than just conditioning.
    """

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        plucker_key: str = "plucker_coords",
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        ema_decay=None,
        learn_logvar=False,
        plucker_recon_weight: float = 0.5,
        plucker_constraint_weight: float = 0.1,
    ):
        """
        Initialize DirectPluckerVAE.

        Args:
            ddconfig: Encoder/decoder config. Should have in_channels=9.
            lossconfig: Loss function configuration
            embed_dim: Embedding dimension for VAE latent
            plucker_key: Key for Plucker coordinates in batch dict
            plucker_recon_weight: Weight for Plucker reconstruction loss
            plucker_constraint_weight: Weight for Plucker constraints
        """
        if ddconfig.get("in_channels", 3) != 9:
            print(f"Warning: DirectPluckerVAE expects in_channels=9, "
                  f"got {ddconfig.get('in_channels', 3)}")

        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=embed_dim,
            ckpt_path=None,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
            ema_decay=ema_decay,
            learn_logvar=learn_logvar,
        )

        self.plucker_key = plucker_key
        self.plucker_recon_weight = plucker_recon_weight
        self.plucker_constraint_weight = plucker_constraint_weight

        decoder_ch = ddconfig["ch"]

        # Multi-head output
        self.decoder_img_head = torch.nn.Conv2d(
            decoder_ch, 3, kernel_size=3, stride=1, padding=1
        )
        self.decoder_plucker_head = torch.nn.Conv2d(
            decoder_ch, 6, kernel_size=3, stride=1, padding=1
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def decode(self, z):
        """Decode latent to image and Plucker coordinates."""
        z = self.post_quant_conv(z)

        h = self.decoder.conv_in(z)
        h = self.decoder.mid.block_1(h, None)
        h = self.decoder.mid.attn_1(h)
        h = self.decoder.mid.block_2(h, None)

        for i_level in reversed(range(self.decoder.num_resolutions)):
            for i_block in range(self.decoder.num_res_blocks + 1):
                h = self.decoder.up[i_level].block[i_block](h, None)
                if len(self.decoder.up[i_level].attn) > 0:
                    h = self.decoder.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.decoder.up[i_level].upsample(h)

        h = self.decoder.norm_out(h)
        h = h * torch.sigmoid(h)

        recon_img = self.decoder_img_head(h)
        recon_plucker = self.decoder_plucker_head(h)

        return recon_img, recon_plucker

    def forward(self, image, plucker, sample_posterior=True):
        """Forward pass through DirectPluckerVAE."""
        x = torch.cat([image, plucker], dim=1)
        posterior = self.encode(x)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        recon_img, recon_plucker = self.decode(z)
        return recon_img, recon_plucker, posterior

    def plucker_constraint_loss(self, plucker):
        """Compute Plucker geometric constraints."""
        d = plucker[:, :3]
        m = plucker[:, 3:]

        dot_product = (d * m).sum(dim=1)
        ortho_loss = torch.mean(dot_product ** 2)

        d_norm = torch.norm(d, dim=1)
        norm_loss = F.mse_loss(d_norm, torch.ones_like(d_norm))

        return ortho_loss + norm_loss

    def get_last_layer(self):
        return self.decoder_img_head.weight

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params_list = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters())
            + list(self.decoder_img_head.parameters())
            + list(self.decoder_plucker_head.parameters())
        )
        if self.learn_logvar:
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []


class ConcatPluckerVAE(AutoencoderKL):
    """
    Variant 1: VAE with separate latent nodes for image and Plucker components.

    This variant uses three separate latent distributions:
    1. Image latent (z_img)
    2. Plucker direction latent (z_d)
    3. Plucker moment latent (z_m)

    Additionally includes an encoder-side Plucker prediction head.

    Input: image (3ch) + plucker (6ch) = 9 channels
    Output: reconstructed image (3ch) + reconstructed direction (3ch) + reconstructed moment (3ch)
    Plus: encoder Plucker prediction
    """

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        plucker_key: str = "plucker_coords",
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        ema_decay=None,
        learn_logvar=False,
        # Loss weights
        img_recon_weight: float = 1.0,
        d_recon_weight: float = 0.5,
        m_recon_weight: float = 0.5,
        encoder_plucker_weight: float = 0.3,
        plucker_constraint_weight: float = 0.1,
        # Latent dimensions
        latent_dim_img: int = 4,
        latent_dim_d: int = 3,
        latent_dim_m: int = 3,
        # Encoder Plucker head config
        plucker_hidden_dim: int = 512,
        plucker_dropout: float = 0.1,
        n_patches: int = 8,
    ):
        """
        Initialize ConcatPluckerVAE.

        Args:
            ddconfig: Encoder/decoder config. Should have in_channels=9.
            lossconfig: Loss function configuration
            embed_dim: Embedding dimension for image VAE latent
            plucker_key: Key for Plucker coordinates in batch dict
            latent_dim_img: Channels for image latent (default: 4)
            latent_dim_d: Channels for direction latent (default: 3)
            latent_dim_m: Channels for moment latent (default: 3)
            n_patches: Number of patches for encoder Plucker prediction
        """
        if ddconfig.get("in_channels", 3) != 9:
            print(f"Warning: ConcatPluckerVAE expects in_channels=9, "
                  f"got {ddconfig.get('in_channels', 3)}")

        # Initialize base class (creates encoder, decoder, single quant_conv)
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=embed_dim,
            ckpt_path=None,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
            ema_decay=ema_decay,
            learn_logvar=learn_logvar,
        )

        self.plucker_key = plucker_key
        self.img_recon_weight = img_recon_weight
        self.d_recon_weight = d_recon_weight
        self.m_recon_weight = m_recon_weight
        self.encoder_plucker_weight = encoder_plucker_weight
        self.plucker_constraint_weight = plucker_constraint_weight

        self.latent_dim_img = latent_dim_img
        self.latent_dim_d = latent_dim_d
        self.latent_dim_m = latent_dim_m
        self.n_patches = n_patches

        # Encoder output channels
        encoder_out_channels = 2 * ddconfig["z_channels"]

        # Three separate quantization convolutions
        # Replace the single quant_conv with three
        self.quant_conv_img = torch.nn.Conv2d(
            encoder_out_channels, 2 * latent_dim_img, 1
        )  # 2x for mean + logvar
        self.quant_conv_d = torch.nn.Conv2d(
            encoder_out_channels, 2 * latent_dim_d, 1
        )
        self.quant_conv_m = torch.nn.Conv2d(
            encoder_out_channels, 2 * latent_dim_m, 1
        )

        # Combined latent: img + d + m
        total_latent_dim = latent_dim_img + latent_dim_d + latent_dim_m

        # Post-quantization conv takes combined latent
        self.post_quant_conv = torch.nn.Conv2d(
            total_latent_dim, ddconfig["z_channels"], 1
        )

        # Decoder multi-head output
        decoder_ch = ddconfig["ch"]
        self.decoder_img_head = torch.nn.Conv2d(
            decoder_ch, 3, kernel_size=3, stride=1, padding=1
        )
        self.decoder_d_head = torch.nn.Conv2d(
            decoder_ch, 3, kernel_size=3, stride=1, padding=1
        )
        self.decoder_m_head = torch.nn.Conv2d(
            decoder_ch, 3, kernel_size=3, stride=1, padding=1
        )

        # Encoder Plucker prediction head (similar to PluckerAutoencoder)
        self.pluck_head = torch.nn.Conv2d(encoder_out_channels, 6, kernel_size=1)
        self.pluck_act = torch.nn.SiLU()

        # Plucker MLP for refinement
        self.pluck_norm_in = torch.nn.LayerNorm(6)
        self.pluck_proj_layers = torch.nn.ModuleList([
            self._make_projection_layer(6, plucker_hidden_dim, plucker_dropout),
            self._make_projection_layer(
                plucker_hidden_dim, plucker_hidden_dim, plucker_dropout
            ),
        ])
        self.pluck_proj_out = torch.nn.Linear(plucker_hidden_dim, 6)

        # Initialize with small weights
        for module in self.pluck_proj_layers.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.pluck_proj_out.weight, gain=0.01)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def _make_projection_layer(self, in_dim, out_dim, dropout_rate):
        """Create a projection layer with normalization and dropout."""
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
        )

    def encode(self, x):
        """
        Encode input to three separate latent distributions + encoder Plucker prediction.

        Args:
            x: Concatenated input (B, 9, H, W)

        Returns:
            Tuple of (posterior_img, posterior_d, posterior_m, pluck_pred)
        """
        B = x.shape[0]
        h = self.encoder(x)

        # Three separate latent distributions
        moments_img = self.quant_conv_img(h)
        moments_d = self.quant_conv_d(h)
        moments_m = self.quant_conv_m(h)

        posterior_img = DiagonalGaussianDistribution(moments_img)
        posterior_d = DiagonalGaussianDistribution(moments_d)
        posterior_m = DiagonalGaussianDistribution(moments_m)

        # Encoder Plucker prediction (auxiliary task)
        pluck = self.pluck_head(h)
        pluck = self.pluck_act(pluck)

        # Interpolate to n_patches
        pluck = F.interpolate(
            pluck,
            size=(self.n_patches, self.n_patches),
            mode="bilinear",
            align_corners=False,
        )

        # MLP processing
        pluck = pluck.permute(0, 2, 3, 1).reshape(B, -1, 6).contiguous()
        pluck = self.pluck_norm_in(pluck)

        for layer in self.pluck_proj_layers:
            pluck = layer(pluck)

        pluck_pred = self.pluck_proj_out(pluck)

        return posterior_img, posterior_d, posterior_m, pluck_pred

    def decode(self, z_img, z_d, z_m):
        """
        Decode combined latent to image and Plucker components.

        Args:
            z_img: Image latent (B, latent_dim_img, H', W')
            z_d: Direction latent (B, latent_dim_d, H', W')
            z_m: Moment latent (B, latent_dim_m, H', W')

        Returns:
            Tuple of (recon_img, recon_d, recon_m)
        """
        # Concatenate latents
        z_combined = torch.cat([z_img, z_d, z_m], dim=1)
        z = self.post_quant_conv(z_combined)

        # Decode
        h = self.decoder.conv_in(z)
        h = self.decoder.mid.block_1(h, None)
        h = self.decoder.mid.attn_1(h)
        h = self.decoder.mid.block_2(h, None)

        for i_level in reversed(range(self.decoder.num_resolutions)):
            for i_block in range(self.decoder.num_res_blocks + 1):
                h = self.decoder.up[i_level].block[i_block](h, None)
                if len(self.decoder.up[i_level].attn) > 0:
                    h = self.decoder.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.decoder.up[i_level].upsample(h)

        h = self.decoder.norm_out(h)
        h = h * torch.sigmoid(h)

        # Three separate output heads
        recon_img = self.decoder_img_head(h)
        recon_d = self.decoder_d_head(h)
        recon_m = self.decoder_m_head(h)

        return recon_img, recon_d, recon_m

    def forward(self, image, plucker, sample_posterior=True):
        """
        Forward pass through ConcatPluckerVAE.

        Args:
            image: Input image (B, 3, H, W)
            plucker: Input Plucker (B, 6, H, W)
            sample_posterior: Whether to sample from posteriors

        Returns:
            Tuple of (recon_img, recon_d, recon_m, posteriors, pluck_pred)
            where posteriors = (posterior_img, posterior_d, posterior_m)
        """
        x = torch.cat([image, plucker], dim=1)

        posterior_img, posterior_d, posterior_m, pluck_pred = self.encode(x)

        if sample_posterior:
            z_img = posterior_img.sample()
            z_d = posterior_d.sample()
            z_m = posterior_m.sample()
        else:
            z_img = posterior_img.mode()
            z_d = posterior_d.mode()
            z_m = posterior_m.mode()

        recon_img, recon_d, recon_m = self.decode(z_img, z_d, z_m)

        posteriors = (posterior_img, posterior_d, posterior_m)
        return recon_img, recon_d, recon_m, posteriors, pluck_pred

    def plucker_constraint_loss(self, d, m):
        """
        Compute Plucker geometric constraints on separate d and m.

        Args:
            d: Direction (B, 3, H, W)
            m: Moment (B, 3, H, W)
        """
        dot_product = (d * m).sum(dim=1)
        ortho_loss = torch.mean(dot_product ** 2)

        d_norm = torch.norm(d, dim=1)
        norm_loss = F.mse_loss(d_norm, torch.ones_like(d_norm))

        return ortho_loss + norm_loss

    def get_last_layer(self):
        return self.decoder_img_head.weight

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params_list = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv_img.parameters())
            + list(self.quant_conv_d.parameters())
            + list(self.quant_conv_m.parameters())
            + list(self.post_quant_conv.parameters())
            + list(self.decoder_img_head.parameters())
            + list(self.decoder_d_head.parameters())
            + list(self.decoder_m_head.parameters())
            + list(self.pluck_head.parameters())
            + list(self.pluck_norm_in.parameters())
            + list(self.pluck_proj_layers.parameters())
            + list(self.pluck_proj_out.parameters())
        )
        if self.learn_logvar:
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []
