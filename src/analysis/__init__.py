"""Analysis module for VAE latent space evaluation and visualization."""

from .model_utils import (
    load_model,
    load_sd_vae,
    encode_images,
    decode_latents,
    denormalize,
)

from .latent_metrics import (
    compute_latent_similarity,
    compute_sequence_consistency,
    compute_latent_stats,
)

from .camera_utils import (
    load_camera_data,
    extract_camera_positions,
    compute_angular_separation,
    find_overlapping_pairs,
    find_view_sequences,
)

from .visualization import (
    latent_to_pca_rgb,
    visualize_reconstructions,
    visualize_latent_channels,
    visualize_latent_pca,
)

__all__ = [
    # Model utilities
    "load_model",
    "load_sd_vae",
    "encode_images",
    "decode_latents",
    "denormalize",
    # Latent metrics
    "compute_latent_similarity",
    "compute_sequence_consistency",
    "compute_latent_stats",
    # Camera utilities
    "load_camera_data",
    "extract_camera_positions",
    "compute_angular_separation",
    "find_overlapping_pairs",
    "find_view_sequences",
    # Visualization
    "latent_to_pca_rgb",
    "visualize_reconstructions",
    "visualize_latent_channels",
    "visualize_latent_pca",
]
