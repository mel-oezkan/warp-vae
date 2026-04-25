"""
Generic model wrapper for unified reconstruction across all VAE variants.

Handles the different forward() signatures so that evaluation code
only needs to call wrapper.reconstruct(images) -> reconstructions.
"""

import re
import torch
import yaml
from pathlib import Path
from typing import Optional

from ldm.util import instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL


def _resolve_hydra_refs(config, root=None):
    """Resolve ${section.key} references in a loaded YAML config dict."""
    if root is None:
        root = config
    if isinstance(config, dict):
        return {k: _resolve_hydra_refs(v, root) for k, v in config.items()}
    if isinstance(config, list):
        return [_resolve_hydra_refs(v, root) for v in config]
    if isinstance(config, str) and "${" in config:
        def _replace(match):
            path = match.group(1)
            node = root
            for part in path.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return match.group(0)  # leave unresolved
            return str(node) if not isinstance(node, (dict, list)) else match.group(0)
        resolved = re.sub(r'\$\{([^}]+)\}', _replace, config)
        # Try to cast back to numeric types
        try:
            if '.' in resolved:
                return float(resolved)
            return int(resolved)
        except (ValueError, TypeError):
            return resolved
    return config


def _detect_model_type(model):
    """Return a string tag for the model variant."""
    cls_name = type(model).__name__
    # Order matters: subclasses before base classes
    type_map = [
        ("ConcatPluckerVAE", "concat_plucker"),
        ("DirectPluckerVAE", "direct_plucker"),
        ("PluckerConditionedVAE", "conditioned_plucker"),
        ("PluckerAutoencoder", "plucker"),
        ("EQVAEAutoencoder", "eqvae"),
        ("AutoencoderKL", "vanilla"),
    ]
    for name, tag in type_map:
        if cls_name == name:
            return tag
    return "vanilla"


class VAEModelWrapper:
    """Unified interface for all VAE model variants.

    Usage:
        wrapper = VAEModelWrapper.from_config(config_path, checkpoint_path, device)
        recon = wrapper.reconstruct(images)
        posterior = wrapper.encode(images)
    """

    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device
        self.model_type = _detect_model_type(model)

    @classmethod
    def from_config(
        cls,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda',
    ) -> 'VAEModelWrapper':
        """Load a model from config + checkpoint.

        Handles both:
        - Trainer checkpoints (state_dict with 'model.' prefix)
        - Direct model checkpoints (no prefix)
        """
        config_path = Path(config_path)
        checkpoint_path = Path(checkpoint_path)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config = _resolve_hydra_refs(config)

        # Instantiate model from config
        model = instantiate_from_config(config['model'])

        # Load checkpoint weights
        ckpt = torch.load(str(checkpoint_path), map_location='cpu')

        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt

        # Strip 'model.' prefix (trainer checkpoints) and filter loss/discriminator keys
        model_sd = {}
        for k, v in sd.items():
            key = k[6:] if k.startswith("model.") else k
            # Skip loss module, discriminator, EMA buffers from trainer
            if any(key.startswith(p) for p in ("loss.", "model_ema.", "discriminator.")):
                continue
            model_sd[key] = v

        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        if missing:
            # Filter out loss-related missing keys (expected)
            real_missing = [k for k in missing if not k.startswith("loss.")]
            if real_missing:
                print(f"[VAEModelWrapper] Missing keys: {real_missing}")
        if unexpected:
            print(f"[VAEModelWrapper] Unexpected keys: {len(unexpected)}")

        print(f"[VAEModelWrapper] Loaded {type(model).__name__} from {checkpoint_path.name}")
        return cls(model, device)

    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """Get image reconstructions regardless of model variant.

        Args:
            images: [B, 3, H, W] input images

        Returns:
            [B, 3, H, W] reconstructed images
        """
        images = images.to(self.device)

        if self.model_type == "vanilla" or self.model_type == "eqvae":
            recon, *_ = self.model(images)
        elif self.model_type == "plucker":
            recon, _posterior, _pluck = self.model(images)
        elif self.model_type in ("direct_plucker", "conditioned_plucker"):
            # These need plucker input; use zeros as dummy for reconstruction eval
            B, _, H, W = images.shape
            dummy_plucker = torch.zeros(B, 6, H, W, device=self.device)
            recon, _recon_pluck, _posterior = self.model(images, dummy_plucker)
        elif self.model_type == "concat_plucker":
            B, _, H, W = images.shape
            dummy_plucker = torch.zeros(B, 6, H, W, device=self.device)
            recon, _d, _m, _posteriors, _pluck = self.model(images, dummy_plucker)
        else:
            recon, *_ = self.model(images)

        return recon

    @torch.no_grad()
    def encode(self, images: torch.Tensor):
        """Encode images to posterior distribution.

        Returns the main posterior (first one for ConcatPlucker).
        """
        images = images.to(self.device)

        if self.model_type == "plucker":
            posterior, _pluck = self.model.encode(images)
        elif self.model_type == "concat_plucker":
            posterior, *_ = self.model.encode(images)
        else:
            posterior = self.model.encode(images)

        return posterior

    @property
    def name(self):
        return type(self.model).__name__
