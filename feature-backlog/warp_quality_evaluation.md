# Warp Quality Evaluation

Script: `scripts/visualize_distance_sampling.py`

Evaluates RoMA warp quality for CO3D image pairs at a given camera distance
range. For each pair it computes bidirectional warps, measures pixel-space MSE,
and saves visualizations with L2 error heatmaps.

Useful for choosing good `--distance_min` / `--distance_max` values before
running `precompute_warps.py` on the full dataset.

## Usage

```bash
# Default: hydrant pairs at distance 2-5, fast RoMA
python scripts/visualize_distance_sampling.py

# Custom distance range
python scripts/visualize_distance_sampling.py --distance_min 1.5 --distance_max 4.0

# More samples
python scripts/visualize_distance_sampling.py --num_sequences 20 --pairs_per_sequence 5

python scripts/visualize_distance_sampling.py --distance_min 1.5 --distance_max 2.0 --num_sequences 20 --pairs_per_sequence 5

# Use precise RoMA (needs expandable_segments on 1080 Ti)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/visualize_distance_sampling.py --roma_setting precise
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--annotation_path` | `hydrant_train.jgz` | CO3D annotation file (with R/T poses) |
| `--co3d_root` | `co3d_full` | CO3D images root |
| `--output_dir` | `eval_outputs/warp_quality` | Output directory |
| `--distance_min` | 2.0 | Min camera Euclidean distance |
| `--distance_max` | 5.0 | Max camera Euclidean distance |
| `--num_sequences` | 10 | Sequences to sample from |
| `--pairs_per_sequence` | 3 | Pairs per sequence |
| `--image_size` | 256 | Image resolution |
| `--roma_setting` | fast | RoMA setting (precise/fast/turbo/base) |
| `--confidence_threshold` | 0.8 | RoMA confidence threshold |
| `--device` | cuda | GPU device |

## Output

Per-pair PNG in `eval_outputs/warp_quality/dist_MIN_MAX/` showing: source,
target, warped images, L2 error heatmaps with MSE values and camera distance.
Console summary with forward/backward/overall MSE mean and std.

## Notes

- `precise` RoMA OOMs on 1080 Ti (11GB) without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `fast` works without memory issues and is the recommended default
- RoMA may output warps at a different resolution than `--image_size`; resizing is handled automatically
- Camera distance is Euclidean distance between world positions: `pos = -R^T @ T`

## Workflow

1. Run this script at a few distance ranges to see MSE vs distance trade-off
2. Pick `--distance_min` / `--distance_max` with acceptable warp quality
3. Pass those values to `precompute_warps.py` for full-dataset precomputation
