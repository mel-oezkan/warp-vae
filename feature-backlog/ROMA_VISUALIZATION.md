# RoMA Warp and Latent Visualization Script

## Overview

Created `scripts/visualization/visualize_roma_warps.py` - a visualization tool for RoMA correspondences, confidence maps, and VAE latent embeddings for multi-view image pairs.

## Purpose

Generate presentation-ready visualizations showing:
- RoMA warp fields between image pairs
- Correspondence confidence maps
- VAE latent embeddings (using EQ-VAE)
- Warped latent representations
- Latent difference analysis in corresponding regions

## Usage

```bash
conda activate cv
python scripts/visualization/visualize_roma_warps.py
```

## Configuration

- **RoMA setting**: `fast` (configurable: "precise", "fast", "turbo", "base")
- **VAE checkpoint**: `checkpoints/eq-vae/diffusion_pytorch_model.safetensors`
- **Confidence threshold**: 70% mean confidence for "high quality" pairs
- **Image size**: 256x256

## Output Structure

All outputs saved to `eval_outputs/roma_visualization/`

### Per-Pair Folders

Each image pair gets its own folder with 12 individual plots + metadata:

```
eval_outputs/roma_visualization/
├── co3d_toytruck_575_84788_167772_f000001-000004/
│   ├── 01_image_a.png              # Input Image A
│   ├── 02_image_b.png              # Input Image B
│   ├── 03_warp_a_to_b.png          # RoMA warp field A→B (color-coded)
│   ├── 04_warp_b_to_a.png          # RoMA warp field B→A
│   ├── 05_confidence_a_to_b.png    # Confidence map A→B (with colorbar)
│   ├── 06_confidence_b_to_a.png    # Confidence map B→A
│   ├── 07_latent_a_pca.png         # Latent A (PCA-projected to RGB)
│   ├── 08_latent_b_pca.png         # Latent B (PCA-projected)
│   ├── 09_latent_b_warped_pca.png  # Latent B warped to A's coordinates
│   ├── 10_latent_difference.png    # |Latent A - Latent B warped|
│   ├── 11_valid_mask.png           # Valid correspondence regions
│   ├── 12_latent_diff_masked.png   # Difference in valid regions only
│   └── metadata.txt                # Confidence values and metrics
├── co3d_toytruck_575_84788_167772_f000001-000004.png  # Combined figure
├── summary_pairs.png               # Grid of top pairs by confidence
└── ...
```

### Metadata Contents

```
Pair: 575_84788_167772_f000001-000004
Source: co3d_toytruck
Mean Confidence A→B: 84.89%
Mean Confidence B→A: 86.55%
Mean Confidence: 85.72%
Valid Fraction A→B: 72.75%
Valid Fraction B→A: 78.32%
Meets 70% threshold: True
```

## Data Sources

### CO3D (Common Objects in 3D)
- Loads from `/data/lab_moezkan/co3d_data/`
- Categories: toytruck, apple, ball
- Real-world video sequences with camera motion
- Higher confidence scores due to natural textures and backgrounds

### OmniObject3D
- Loads from `/data/lab_moezkan/omni_obj/blender_renders_24_views/`
- Rendered objects with 24 views each
- RGBA images composited onto white background
- Lower confidence due to uniform backgrounds and small object coverage

## Key Functions

### `load_co3d_pairs()`
- Loads consecutive frames from CO3D sequences
- Frame spacing: 3, 5, or 8 frames apart for varying overlap

### `load_omniobject_pairs()`
- Loads adjacent views from OmniObject renders
- Handles RGBA → RGB conversion with white background
- View spacing: 1-2 views apart

### `visualize_roma_pair()`
- Computes RoMA correspondences using `roma_metrics.py`
- Encodes images with VAE
- Warps latent B to A's coordinate frame
- Saves both individual plots and combined figure

### `save_individual_plot()`
- Saves single visualization as standalone image
- Supports colormaps and colorbars
- 6x6 inch figures at 150 DPI

## Results (Example Run)

**High-confidence pairs (>=70%):**
1. `co3d_toytruck_575_84788_167772_f000001-000004` - 85.7%
2. `co3d_toytruck_346_36113_66551_f000007-000010` - 82.0%
3. `co3d_toytruck_346_36113_66551_f000001-000004` - 78.6%
4. `co3d_toytruck_346_36113_66551_f000019-000022` - 74.5%
5. `co3d_toytruck_346_36113_66551_f000013-000016` - 73.9%

## Dependencies

- RoMA v2 (`src/analysis/roma_metrics.py`)
- EQ-VAE / Diffusers AutoencoderKL
- PyTorch, PIL, matplotlib, sklearn (PCA)

## Notes

- OmniObject images have low RoMA confidence because objects occupy small portion of frame
- CO3D real-world images work much better for correspondence matching
- All visualizations use consistent PCA projection (fitted on Latent A, applied to B and B_warped)
