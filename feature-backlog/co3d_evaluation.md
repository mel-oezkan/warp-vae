# CO3D Multi-View Latent Consistency Evaluation

## Overview

The evaluation script `scripts/analyze_multiview_latent_consistency.py` now supports CO3D alongside OmniObject3D. It measures how consistent VAE latent representations are across different viewpoints of the same scene, using CO3D hydrant sequences as test data.

## What it does

1. **Loads CO3D sequences** from preprocessed annotation files (`.jgz`), each sequence being a video of a hydrant from varying camera angles
2. **Extracts camera positions** from CO3D's world-to-camera (R, T) format and computes angular separation between all view pairs
3. **Selects view pairs** within an angular range (default 2-30 degrees) to compare nearby but distinct viewpoints
4. **Encodes each view** through one or more VAE models to get latent representations
5. **Computes latent similarity metrics** (cosine similarity, MSE, MAE) between view pairs as a function of angular separation
6. **Generates visualizations**: scatter plots, trend lines, box plots, binned bar charts, PCA latent visualizations, and similarity matrices

## Analysis modes

- **Global mode** (`--mode global`): Compares full latent tensors directly between view pairs
- **RoMA mode** (`--mode roma`): Uses RoMA dense correspondences to compare only overlapping regions between views, giving a more precise measure of 3D consistency
- **Both** (`--mode both`): Runs both analyses

## CO3D data flow

```
hydrant_train.jgz (preprocessed annotations)
    -> {sequence_name: [{filepath, R, T, focal_length, principal_point, bbox}, ...]}
    -> extract camera world positions: pos = -R^T @ T
    -> compute angular separation matrix
    -> find view pairs in [min_angle, max_angle] range
    -> load images from co3d_full/hydrant/<seq>/images/frame000XXX.jpg
    -> encode with VAE -> latent tensors
    -> compute pairwise metrics
```

## Usage

```bash
# Basic evaluation with baseline VAE on CO3D hydrants
python scripts/analyze_multiview_latent_consistency.py \
    --dataset co3d \
    --co3d_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \
    --co3d_annotations /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \
    --checkpoints weights/f8/model.ckpt \
    --configs config/baseVAE.yaml \
    --model_names "f8 Baseline" \
    --output_name co3d_hydrant_baseline \
    --num_objects 50

# Compare custom model vs baseline
python scripts/analyze_multiview_latent_consistency.py \
    --dataset co3d \
    --co3d_annotations /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \
    --checkpoints outputs/my_model/checkpoints/last.ckpt weights/f8/model.ckpt \
    --configs config/my_config.yaml config/baseVAE.yaml \
    --model_names "My Model" "f8 Baseline" \
    --output_name co3d_hydrant_comparison \
    --num_objects 30 --num_detailed_objects 3

# RoMA region-based analysis
python scripts/analyze_multiview_latent_consistency.py \
    --dataset co3d \
    --co3d_annotations /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \
    --checkpoints weights/f8/model.ckpt \
    --configs config/baseVAE.yaml \
    --model_names "f8 Baseline" \
    --mode roma \
    --output_name co3d_hydrant_roma
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

## Implementation

CO3D support is implemented via a `DatasetAdapter` pattern in the evaluation script. The `CO3DAdapter` class handles:
- Loading annotations from `.jgz` files (`src/analysis/camera_utils.load_co3d_annotations`)
- Converting W2C camera poses to world positions (`src/analysis/camera_utils.extract_co3d_camera_positions`)
- Resolving image file paths from annotation metadata

The existing `OmniObjectAdapter` wraps the original OmniObject3D logic. All downstream analysis (angular separation, pair finding, encoding, metrics, visualization) is shared between both datasets.
