# RoMA Region-Based Latent Consistency Metrics

## Overview

This document explains the new RoMA-based region evaluation metrics added to `analyze_multiview_latent_consistency.py`. These metrics measure how consistently a VAE encodes the **same 3D surface regions** across different viewpoints, using dense correspondences from RoMA (Robust Dense Feature Matching).

## Motivation

The original **global metrics** (cosine similarity, MSE, MAE) compare entire latent representations between view pairs. However, when comparing two views of the same object from different angles:
- Only a portion of the image content overlaps (the same 3D surface seen from both views)
- Non-overlapping regions (occluded areas, different background) add noise to the comparison
- A 3D-consistent VAE should encode the *same* 3D surface similarly, regardless of viewpoint

**Region-based metrics** address this by:
1. Using RoMA to find dense pixel correspondences between views
2. Filtering to high-confidence correspondences (>80% confidence)
3. Comparing only the latent regions that correspond to the same 3D surface

## How It Works

### Data Flow

```
Image A (256x256)                    Image B (256x256)
       |                                    |
       +-------- RoMA matching -------------+
                       |
              warp_AB (256x256x2)   [dense correspondences]
              overlap_AB (256x256)  [confidence per pixel]
                       |
              Filter: confidence > 0.8 AND in-bounds
                       |
              Downsample to latent space (32x32)
              - warp: bilinear interpolation
              - mask: min-pooling (conservative)
                       |
       +---------------+---------------+
       |                               |
   VAE encode                      VAE encode
       |                               |
   latent_A (4x32x32)           latent_B (4x32x32)
       |                               |
       |     grid_sample(latent_B, warp_AB)
       |               |
       |        latent_B_warped (4x32x32)
       |               |
       +-------Compare in valid regions only-------+
                       |
              region_mse, region_mae, region_cosine
```

### Coordinate Mapping

Both image and latent spaces use normalized [-1, 1] coordinates:
- **Image space**: 256x256 pixels → each pixel is ~0.0078 in normalized coords
- **Latent space**: 32x32 cells → each cell corresponds to 8x8 image pixels

The warp field is downsampled using bilinear interpolation. The confidence mask uses **min-pooling**: a latent cell is only marked valid if ALL 8 corresponding image pixels have confidence > threshold.

## Metrics Explained

### Region Metrics (New)

| Metric | Description |
|--------|-------------|
| `region_cosine` | Cosine similarity computed only over valid (corresponding) latent regions. Higher = more consistent encoding of the same 3D surface. |
| `region_mse` | Mean Squared Error over valid regions only. Lower = more consistent. |
| `region_mae` | Mean Absolute Error over valid regions only. Lower = more consistent. |
| `valid_fraction` | Fraction of latent space (32x32=1024 cells) with valid RoMA correspondences. Typically 30-70% for nearby views, decreasing with angular separation. |
| `valid_fraction_ab` | Valid fraction for A→B direction |
| `valid_fraction_ba` | Valid fraction for B→A direction |

### Global Metrics (Existing, for comparison)

| Metric | Description |
|--------|-------------|
| `global_cosine` | Cosine similarity over the entire flattened latent (no warping) |
| `global_mse` | MSE over entire latent |
| `global_mae` | MAE over entire latent |

### Interpretation

- **Region cosine > Global cosine**: The VAE encodes corresponding 3D regions more consistently than the overall image. This is expected for a 3D-aware model.
- **Region cosine ≈ Global cosine**: The VAE treats all regions similarly, no special 3D consistency.
- **Region cosine < Global cosine**: Unusual; could indicate issues with the correspondence or encoding.

## Usage

```bash
# RoMA mode only
python scripts/analysis/analyze_multiview_latent_consistency.py \
    --checkpoints weights/f8/model.ckpt \
    --configs config/baseVAE.yaml \
    --model_names "Baseline" \
    --mode roma \
    --roma_setting precise \
    --roma_confidence_threshold 0.8 \
    --output_name roma_analysis

# Both global and RoMA analysis
python scripts/analysis/analyze_multiview_latent_consistency.py \
    --checkpoints model1.ckpt model2.ckpt \
    --configs config1.yaml config2.yaml \
    --model_names "Model A" "Model B" \
    --mode both \
    --output_name full_comparison
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `global` | Analysis mode: `global`, `roma`, or `both` |
| `--roma_setting` | `precise` | RoMA quality: `precise` (best), `fast`, `turbo`, `base` |
| `--roma_confidence_threshold` | `0.8` | Minimum confidence for valid correspondences (0-1) |

## Output Files

When running with `--mode roma` or `--mode both`:

| File | Description |
|------|-------------|
| `roma_model_comparison.png` | 3x3 visualization grid with region metrics |
| `roma_comparison_stats.txt` | Detailed statistics including region/global comparison |

### Visualization Plots

1. **Region Cosine vs Angle**: Scatter plot showing how region similarity decreases with angular separation
2. **Region MSE vs Angle**: Error increases with angle
3. **Valid Fraction vs Angle**: Coverage decreases as views diverge
4. **Region vs Global Cosine**: Scatter comparing the two metric types
5. **Box plots**: Distribution of region cosine and valid fraction per model
6. **Binned bar charts**: Average metrics by angle bin (0-15°, 15-30°, etc.)
7. **Summary table**: Key statistics for each model

## Expected Results

For a well-behaved VAE on multi-view data:

1. **Valid fraction**: 40-70% for views <30° apart, decreasing to 10-30% for views >45° apart
2. **Region cosine**: Should be higher than global cosine for similar angles
3. **Correlation with angle**: Region cosine should decrease as angular separation increases (same surface seen from more different angles = harder to encode identically)

## Technical Details

### RoMA Settings

| Setting | Resolution | Speed | Quality |
|---------|------------|-------|---------|
| `precise` | 800→1280 | Slow | Best |
| `fast` | 512 | Medium | Good |
| `turbo` | 320 | Fast | Acceptable |
| `base` | 560 | Medium | Good |

For evaluation, `precise` is recommended as accuracy matters more than speed.

### Bidirectional Averaging

The metrics use bidirectional correspondences:
1. Warp B→A frame, compare with A
2. Warp A→B frame, compare with B
3. Average the results

This provides more robust estimates than single-direction comparison.

### Min-Pooling for Confidence

Using min-pooling (instead of average pooling) for the confidence mask is conservative: a latent cell requires ALL 8x8 image pixels to have valid correspondences. This ensures high-quality region comparisons but may result in lower valid fractions.
