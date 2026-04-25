# TODO: Delete this
# CO3D Multi-View Latent Consistency Evaluation

## Overview

The evaluation script `scripts/analyze_multiview_latent_consistency.py` now supports CO3D alongside OmniObject3D. It measures how consistent VAE latent representations are across different viewpoints of the same scene, using CO3D hydrant sequences as test data.

## What it does

1. **Loads CO3D sequences** from preprocessed annotation files (`.jgz`), each sequence being a video of a hydrant from varying camera angles
2. **Extracts camera positions** from CO3D's world-to-camera (R, T) format and computes Euclidean camera distance between all view pairs
3. **Selects view pairs** within a camera distance range (default 0.5-3.0) to compare nearby but distinct viewpoints
4. **Encodes each view** through one or more VAE models to get latent representations
5. **Computes latent similarity metrics** (cosine similarity, MSE, MAE) between view pairs as a function of camera distance
6. **Generates visualizations**: scatter plots, trend lines, box plots, binned bar charts, PCA latent visualizations, and similarity matrices

## Analysis modes

- **Global mode** (`--mode global`): Compares full latent tensors directly between view pairs
- **RoMA mode** (`--mode roma`): Uses RoMA dense correspondences to compare only overlapping regions between views, giving a more precise measure of 3D consistency. Supports `--roma_setting` (`precise`, `fast`, `turbo`, `base`) and `--roma_confidence_threshold` (default 0.8)
- **Both** (`--mode both`): Runs both analyses

## Key CLI arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoints` | One or more checkpoint paths | required |
| `--configs` | One config per checkpoint | required |
| `--model_names` | Display names for each model | auto from checkpoint |
| `--model_types` | Model type per checkpoint (`auto`, `ldm`, `eqvae`, `diffusers`) | `auto` |
| `--compare_baseline` | Auto-include f8 baseline VAE | off |
| `--mode` | `global`, `roma`, or `both` | `global` |
| `--roma_setting` | RoMaV2 variant (`precise`, `fast`, `turbo`, `base`) | `precise` |
| `--roma_confidence_threshold` | Min confidence for valid correspondences | `0.8` |
| `--max_distance` | Maximum camera Euclidean distance for pair selection | `3.0` |
| `--min_distance` | Minimum camera Euclidean distance for pair selection | `0.5` |
| `--image_size` | Image size for encoding | `256` |
| `--num_objects` | Number of objects for aggregate stats | `50` |
| `--num_detailed_objects` | Objects for per-object visualizations | `5` |
| `--num_workers` | Parallel workers for multi-GPU (RoMA only) | `1` |
| `--gpu_ids` | GPU IDs to use for multi-GPU | auto |

## CO3D data flow

```
hydrant_train.jgz (preprocessed annotations)
    -> {sequence_name: [{filepath, R, T, focal_length, principal_point, bbox}, ...]}
    -> extract camera world positions: pos = -R^T @ T
    -> compute Euclidean camera distance matrix
    -> find view pairs in [min_distance, max_distance] range
    -> load images from co3d_full/hydrant/<seq>/images/frame000XXX.jpg
    -> encode with VAE -> latent tensors
    -> compute pairwise metrics
```

## Usage

The script now uses `--checkpoints`, `--configs`, and `--model_names` to compare multiple models in a single run. Dataset defaults to CO3D (co3d_dir and co3d_annotations have sensible defaults).

```bash
# Single model evaluation
python scripts/analyze_multiview_latent_consistency.py \
    --checkpoints checkpoints/massive-accurate-okapi-of-blizzard/last.ckpt \
    --configs config/warp_vae_co3d_precomputed.yaml \
    --model_names "WARP-VAE" \
    --output_name warp_vae_analysis \
    --num_objects 50

# Full 3-model comparison with RoMA on CO3D (recommended)
python scripts/analyze_multiview_latent_consistency.py \
    --dataset co3d \
    --checkpoints \
        checkpoints/massive-accurate-okapi-of-blizzard/last.ckpt \
        checkpoints/eq-vae/diffusion_pytorch_model.safetensors \
        weights/f8/model.ckpt \
    --configs \
        config/warp_vae_co3d_precomputed.yaml \
        checkpoints/eq-vae/config.json \
        config/baseVAE.yaml \
    --model_names "WARP-VAE" "EQ-VAE" "SD-VAE" \
    --mode roma \
    --roma_setting fast \
    --max_distance 5.0 \
    --min_distance 0.5 \
    --output_name full_comparison_co3d

# RoMA mode with precise setting and confidence filtering
python scripts/analyze_multiview_latent_consistency.py \
    --checkpoints weights/f8/model.ckpt \
    --configs config/baseVAE.yaml \
    --model_names "Baseline" \
    --mode roma \
    --roma_setting precise \
    --roma_confidence_threshold 0.8 \
    --output_name roma_analysis

# Both global and RoMA analysis
python scripts/analyze_multiview_latent_consistency.py \
    --checkpoints model1.ckpt model2.ckpt \
    --configs config1.yaml config2.yaml \
    --model_names "Model A" "Model B" \
    --mode both \
    --output_name full_comparison
```

## Key differences from OmniObject3D evaluation

| Aspect | OmniObject3D | CO3D |
|--------|-------------|------|
| Views per object | 24 (fixed, synthetic) | ~100 (variable, real video frames) |
| Angular step | ~15 degrees (uniform) | ~0.1 degrees (dense, from video) |
| Image format | 800x800 PNG | ~700x1200 JPG (non-square) |
| Camera format | 4x4 C2W transform matrix | 3x3 R + 3D T (W2C) |
| Annotations | transforms.json per object | Single .jgz per category |

## Outputs

Results are saved to `eval_outputs/<output_name>/`:
- `model_comparison.png` -- aggregate metrics across all sequences
- `comparison_stats.txt` -- summary statistics
- `sequence_<seq_name>.png` -- PCA latent visualization for individual sequences
- `matrices_<seq_name>.png` -- pairwise similarity matrices
- `angle_vs_sim_<seq_name>.png` -- per-sequence angle vs similarity scatter plots
- `roma_model_comparison.png` -- RoMA region-based metrics (if `--mode roma`)
- `roma_comparison_stats.txt` -- RoMA summary statistics

## Prerequisites

Before running, ensure the CO3D preprocessing is complete:

```bash
# Step 1: Precompute bounding boxes (run once, slow)
python preprocess_co3d.py \
    --category hydrant --precompute_bbox \
    --co3d_v2_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \
    --output_dir /visinf/projects_students/dlcv2025_groupZ/co3d_annotations

# Step 2: Process poses (uses precomputed bboxes)
python preprocess_co3d.py \
    --category hydrant \
    --co3d_v2_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \
    --output_dir /visinf/projects_students/dlcv2025_groupZ/co3d_annotations
```

## Reconstruction Metrics

The `ReconstructionMetrics` class (`evaluation/metrics/reconstruction_metrics.py`) has been refactored to be model-agnostic:

- **Old API**: `ReconstructionMetrics(model, device)` — tightly coupled to a specific model
- **New API**: `ReconstructionMetrics(device, reconstruct_fn=None)` — accepts any VAE variant via callback

```python
# Default: model(images)[0]
metrics = ReconstructionMetrics(device)
results = metrics.compute(dataloader, model)

# Custom reconstruction for WarpVAE or NaiveWarpVAE:
metrics = ReconstructionMetrics(device, reconstruct_fn=lambda m, imgs: my_reconstruct(m, imgs))
results = metrics.compute(dataloader, model)
```

This allows evaluating any trainer variant (WarpVAE, NaiveWarpVAE, etc.) without modifying the metrics module.

## Implementation

CO3D support is implemented via a `DatasetAdapter` pattern in the evaluation script. The `CO3DAdapter` class handles:
- Loading annotations from `.jgz` files (`src/analysis/camera_utils.load_co3d_annotations`)
- Converting W2C camera poses to world positions (`src/analysis/camera_utils.extract_co3d_camera_positions`)
- Resolving image file paths from annotation metadata

The existing `OmniObjectAdapter` wraps the original OmniObject3D logic. All downstream analysis (Euclidean camera distance, pair finding, encoding, metrics, visualization) is shared between both datasets.
