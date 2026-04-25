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
    compute_pairwise_similarity_matrices,
    compute_latent_stats,
)

from .camera_utils import (
    load_camera_data,
    extract_camera_positions,
    compute_angular_separation,
    compute_euclidean_distance_matrix,
    find_overlapping_pairs,
    find_view_sequences,
    load_co3d_annotations,
    extract_co3d_camera_positions,
)

from .visualization import (
    latent_to_pca_rgb,
    latent_to_pca_jet,
    visualize_reconstructions,
    visualize_latent_channels,
    visualize_latent_pca,
)

from .roma_metrics import (
    load_roma_model,
    compute_roma_correspondences,
    warp_to_latent_warp,
    confidence_to_latent_mask,
    warp_latent,
    compute_region_similarity,
    compute_bidirectional_region_similarity,
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
    "compute_pairwise_similarity_matrices",
    "compute_latent_stats",
    # Camera utilities
    "load_camera_data",
    "extract_camera_positions",
    "compute_angular_separation",
    "compute_euclidean_distance_matrix",
    "find_overlapping_pairs",
    "find_view_sequences",
    "load_co3d_annotations",
    "extract_co3d_camera_positions",
    # Visualization
    "latent_to_pca_rgb",
    "latent_to_pca_jet",
    "visualize_reconstructions",
    "visualize_latent_channels",
    "visualize_latent_pca",
    # RoMA metrics
    "load_roma_model",
    "compute_roma_correspondences",
    "warp_to_latent_warp",
    "confidence_to_latent_mask",
    "warp_latent",
    "compute_region_similarity",
    "compute_bidirectional_region_similarity",
]
