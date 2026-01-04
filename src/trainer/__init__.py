"""
VAE Trainers module.

Provides trainers for different VAE variants:
- VanillaVAETrainer: Standard AutoencoderKL
- PluckerVAETrainer: Plucker-aware VAE
- EQVAETrainer: Equivariant VAE
"""

from src.trainers.base_trainer import BaseVAETrainer
from src.trainers.vae_trainers import (
    VanillaVAETrainer,
    PluckerVAETrainer,
    EQVAETrainer,
)

__all__ = [
    "BaseVAETrainer",
    "VanillaVAETrainer",
    "PluckerVAETrainer",
    "EQVAETrainer",
]