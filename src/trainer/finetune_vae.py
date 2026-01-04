"""
Legacy FinetuneVAE trainer.

DEPRECATED: Use PluckerVAETrainer instead.
Kept for backward compatibility with existing configs.
"""

import warnings
from src.trainer.vae_trainers import PluckerVAETrainer


class FinetuneVAE(PluckerVAETrainer):
    """
    Legacy trainer class - wraps PluckerVAETrainer for backward compatibility.
    
    .. deprecated::
        Use `PluckerVAETrainer` directly instead.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FinetuneVAE is deprecated. Use PluckerVAETrainer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)