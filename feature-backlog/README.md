# Documentation Index

This directory contains design documents and guides for the 3D-Aware VAE project. Each document covers a specific model variant, tool, or evaluation method.

## Getting Started

- [QUICKSTART.md](QUICKSTART.md) -- Setup, training commands, troubleshooting, dataset preparation

## Model Variants

These documents describe the different approaches to improving 3D awareness in the VAE:

| Document | Model | Summary |
|----------|-------|---------|
| [EQ-VAE_Implementation.md](EQ-VAE_Implementation.md) | EQ-VAE | Equivariance regularization via random scale + rotation transforms on latent codes. Probabilistic mixing of EQ-VAE, low-res, and full-res training paths. |
| [Warp_VAE_Training.md](Warp_VAE_Training.md) | Warp VAE, Naive Warp VAE, Depth Warp VAE | Multi-view consistency via RoMA or depth-based correspondences. Includes warp/naive/depth variants, NaiveWarpConsistencyLoss, depth warp precomputation, `return_latent` API, and experiment configs. |
| [PluckerVAE_Variants.md](PluckerVAE_Variants.md) | Plucker VAE | Three variants (Concat, Direct, Conditioned) for integrating full-resolution Plucker ray coordinates into the VAE. Includes architecture diagrams, loss functions, and known issues. |

## Evaluation and Analysis

| Document | Script/Tool | Summary |
|----------|-------------|---------|
| [MultiViewConsistencyEval.md](MultiViewConsistencyEval.md) | `scripts/analyze_multiview_latent_consistency.py` | Measures latent consistency across viewpoints on CO3D/OmniObject3D. Supports global and RoMA-based analysis modes. |
| [roma_region_metrics.md](roma_region_metrics.md) | (same script, RoMA mode) | Region-based metrics using dense correspondences to compare only overlapping 3D regions. Explains min-pooling, bidirectional averaging, and expected results. |
| [ROMA_VISUALIZATION.md](ROMA_VISUALIZATION.md) | `scripts/visualize_roma_warps.py` | Generates per-pair visualizations of RoMA warps, confidence maps, PCA-projected latents, and difference maps. |
| [warp_quality_evaluation.md](warp_quality_evaluation.md) | `scripts/visualize_distance_sampling.py` | Evaluates RoMA warp quality at different camera distance ranges. Use before running full precomputation. |

## Data Pipeline

| Document | Script/Tool | Summary |
|----------|-------------|---------|
| [ROMA_PRECOMPUTE_SPEEDUP.md](ROMA_PRECOMPUTE_SPEEDUP.md) | `precompute_warps.py`, `precompute_depth_warps.py` | Precompute warp fields to disk (RoMA-based or depth-based). Distance-based pair selection, multi-GPU support. |

## Reading Order for New Contributors

1. **QUICKSTART.md** -- Understand how to run training
2. **Warp_VAE_Training.md** -- The primary active approach (Warp VAE, Naive Warp, Depth Warp variants)
3. **EQ-VAE_Implementation.md** -- The equivariance approach
4. **MultiViewConsistencyEval.md** -- How we evaluate models (model-agnostic metrics)
5. **PluckerVAE_Variants.md** -- Alternative geometric prior approach
