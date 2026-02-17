"""Model loading and encoding utilities for VAE analysis."""

import torch
from pathlib import Path

try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    from diffusers import AutoencoderKL as DiffusersVAE
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1] range."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def load_model(checkpoint_path: str, config_path: str, model_type: str = "auto"):
    """Load VAE model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt, .pt, or .safetensors)
        config_path: Path to config file (JSON for diffusers-style or YAML for LDM-style)
        model_type: One of 'auto', 'ldm', 'eqvae', 'diffusers'

    Returns:
        model: Loaded VAE model
        model_type: Detected or specified model type
    """
    checkpoint_path_str = str(checkpoint_path)
    config_path_str = str(config_path)

    print(f"Loading checkpoint from {checkpoint_path_str}")
    print(f"Loading config from {config_path_str}")

    # Detect model type from config if auto
    if model_type == "auto":
        if config_path_str.endswith('.json'):
            model_type = "diffusers"
        else:
            from omegaconf import OmegaConf
            yaml_config = OmegaConf.load(config_path_str)
            if 'model' in yaml_config:
                target = yaml_config.model.get('target', '')
                if 'EQVAEAutoencoder' in target:
                    model_type = "eqvae"
                else:
                    model_type = "ldm"
            else:
                model_type = "ldm"
        print(f"  Detected model type: {model_type}")

    # Load state dict from checkpoint
    if checkpoint_path_str.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors package required")
        state_dict = load_safetensors(checkpoint_path_str)
    else:
        ckpt = torch.load(checkpoint_path_str, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

    # Handle key prefixes
    if any("first_stage_model" in k for k in state_dict.keys()):
        state_dict = {
            k.replace("first_stage_model.", ""): v
            for k, v in state_dict.items()
            if "first_stage_model" in k
        }
    elif any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

    # Load model based on type
    if model_type == "diffusers":
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers package required for loading diffusers-format models")

        checkpoint_dir = Path(checkpoint_path_str).parent
        print(f"  Loading diffusers VAE from directory: {checkpoint_dir}")

        model = DiffusersVAE.from_pretrained(str(checkpoint_dir))
        print("  Diffusers VAE loaded successfully")
        return model, "sdvae"

    elif model_type == "eqvae":
        from omegaconf import OmegaConf
        from ldm.models.autoencoder import EQVAEAutoencoder

        config = OmegaConf.load(config_path_str)
        OmegaConf.resolve(config)
        model_params = config.model.params

        model = EQVAEAutoencoder(
            ddconfig=OmegaConf.to_container(model_params.ddconfig),
            lossconfig=OmegaConf.to_container(model_params.lossconfig),
            embed_dim=model_params.embed_dim,
            p_prior=model_params.get("p_prior", 0.5),
            p_prior_s=model_params.get("p_prior_s", 0.25),
            anisotropic=model_params.get("anisotropic", False),
            use_rotation=model_params.get("use_rotation", True),
        )

    else:  # ldm
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config

        yaml_config = OmegaConf.load(config_path_str)
        OmegaConf.resolve(yaml_config)
        yaml_config = OmegaConf.to_container(yaml_config, resolve=True)

        if 'model' in yaml_config:
            model = instantiate_from_config(yaml_config['model'])
        else:
            raise ValueError("YAML config must contain 'model' key")

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    print("  Checkpoint loaded successfully")

    return model, model_type


def load_sd_vae(device: str = "cuda"):
    """Load SD-VAE (ft-mse) from HuggingFace.

    This loads the fine-tuned MSE variant of the Stable Diffusion VAE.
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers package required for SD-VAE")

    model = DiffusersVAE.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    )
    model = model.to(device)
    model.eval()
    print("Loaded SD-VAE (ft-mse) from HuggingFace")
    return model


@torch.no_grad()
def encode_images(model, images: torch.Tensor, device: str, model_type: str = "ldm") -> torch.Tensor:
    """Encode images with VAE model.

    Args:
        model: VAE model (either diffusers or LDM-style)
        images: Input images tensor in [-1, 1] range
        device: Device to run on
        model_type: Either "sdvae" for diffusers or "ldm"/"eqvae" for LDM-style

    Returns:
        Latent tensor (unscaled, raw latent space)
    """
    images = images.to(device)

    if model_type == "sdvae":
        latent = model.encode(images).latent_dist.sample()
    else:
        if hasattr(model, 'ema_scope') and hasattr(model, 'model_ema') and model.model_ema is not None:
            with model.ema_scope():
                posterior = model.encode(images)
        else:
            posterior = model.encode(images)
        latent = posterior.sample()

    return latent


@torch.no_grad()
def decode_latents(model, latents: torch.Tensor, device: str, model_type: str = "ldm") -> torch.Tensor:
    """Decode latents with VAE model.

    Args:
        model: VAE model (either diffusers or LDM-style)
        latents: Latent tensor (unscaled)
        device: Device to run on
        model_type: Either "sdvae" for diffusers or "ldm"/"eqvae" for LDM-style

    Returns:
        Reconstructed images tensor in [-1, 1] range
    """
    latents = latents.to(device)

    if model_type == "sdvae":
        scaling_factor = getattr(model.config, 'scaling_factor', 0.18215)
        latents_scaled = latents / scaling_factor
        recon = model.decode(latents_scaled).sample
    else:
        if hasattr(model, 'ema_scope') and hasattr(model, 'model_ema') and model.model_ema is not None:
            with model.ema_scope():
                recon = model.decode(latents)
        else:
            recon = model.decode(latents)

    return recon
