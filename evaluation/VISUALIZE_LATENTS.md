# Latent Space Visualization Script

This script visualizes the latent space of VAE models by extracting latent codes from images across multiple object subcategories and generating visualizations.

## Overview

The script performs two main visualizations across multiple subcategories:

1. **Latent Samples** (`latent_samples.png`)
   - For each subcategory: input images (Row 0) and individual latent channels (Rows 1-4)
   - Each channel is normalized independently using percentile clipping for better contrast
   - Shows multiple subcategories stacked vertically

2. **PCA Visualization** (`latent_pca.png`)
   - For each subcategory: input images, raw latent (channels 0-2 as RGB), and PCA-transformed latents
   - Reports average explained variance across all samples

## How It Works

### 1. Model Loading
The script auto-detects the model architecture from the checkpoint:
- Detects SD checkpoint format (`first_stage_model.` prefix) vs training format (`model.` prefix)
- Determines architecture parameters (channel size, channel multipliers) from weight shapes
- Supports vanilla SD architecture (ch=128, ch_mult=[1,2,4,4]) and smaller models (ch=64, ch_mult=[1,2,4])

### 2. Dataset Loading
Supports three dataset types:
- `co3d`: CO3D dataset (requires bounding box file)
- `omniobject`: OmniObject3D dataset
- `warp_co3d`: Warp CO3D dataset (requires bounding box file)

The dataset internally applies transforms (resize, to tensor, normalize to [-1, 1]) so images are ready for the VAE encoder.

### 3. Latent Extraction
- Encodes pre-transformed images through the VAE encoder
- Stores flattened latents (spatial average) for distribution analysis
- Stores spatial latents (first 100 samples) for PCA visualization

### 4. PCA Transformation
For each image independently:
1. Reshape latent from (C, H, W) to (H×W, C)
2. Fit PCA on the spatial pixels of that single image
3. Transform to 3 principal components
4. Reshape back to (H, W, 3) for RGB visualization
5. Normalize using percentile clipping for better contrast

## Usage

### Basic Usage

```bash
python evaluation/visualize_latents.py \
    --checkpoint <path_to_checkpoint> \
    --output_name <subfolder_name>
```

### Full Example (Warp VAE)

```bash
python evaluation/visualize_latents.py \
    --checkpoint checkpoints/last.ckpt \
    --output_name warp_vae_latents \
    --dataset_type co3d \
    --data_dir /data/lab_moezkan/co3d_full \
    --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz \
    --num_samples 500 \
    --image_size 128 \
    --batch_size 4
```

### Vanilla SD VAE

```bash
python evaluation/visualize_latents.py \
    --checkpoint sd_model/v1-5-pruned.ckpt \
    --output_name sd_vae_baseline \
    --dataset_type co3d \
    --data_dir /data/lab_moezkan/co3d_full \
    --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz \
    --num_samples 500 \
    --image_size 256 \
    --vanilla_sd \
    --batch_size 4
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | Yes | - | Path to model checkpoint |
| `--output_name` | Yes | - | Subfolder name under `eval_outputs/` |
| `--dataset_type` | No | `co3d` | Dataset type: `co3d`, `omniobject`, `warp_co3d` |
| `--data_dir` | No | `/data/lab_moezkan/co3d_full/toybus` | Dataset root directory |
| `--bb_file` | No | Auto-detected | Bounding box file for CO3D datasets |
| `--config` | No | None | Optional config YAML for model instantiation |
| `--vanilla_sd` | No | False | Force vanilla SD architecture |
| `--num_samples` | No | 1000 | Number of samples for latent extraction |
| `--image_size` | No | 256 | Input image size |
| `--batch_size` | No | 16 | Batch size for inference |
| `--skip_pca` | No | False | Skip PCA visualization |

## Output

Results are saved to `eval_outputs/<output_name>/`:

```
eval_outputs/
└── <output_name>/
    ├── latent_samples.png         # Latent samples
    └── latent_pca.png             # PCA visualization
```

## Example Output Interpretation

### Latent Samples
- **Row 0 (Input)**: Original images from the dataset
- **Rows 1-4 (Latent Ch 0-3)**: Individual latent channels visualized as grayscale heatmaps
- **Multiple Subcategories**: Stacked vertically for comparison

### PCA Visualization
- **Row 1 (Input)**: Original images from the dataset
- **Row 2 (Latent Ch 0-2)**: Raw latent channels visualized as RGB
- **Row 3 (Latent PCA)**: PCA-transformed latents showing the dominant variance directions
- **Title**: Reports explained variance ratios for the 3 principal components across all subcategories
