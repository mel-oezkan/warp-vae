# RoMA Warp Precomputation Guide

## Overview

`precompute_warps.py` precomputes all RoMaV2 warp fields to disk, removing the
RoMA computation from the training loop. Pairs are selected using **camera
Euclidean distance** (derived from R/T poses), so only geometrically meaningful
pairs are computed regardless of capture order.

Supports multi-GPU acceleration for 1.8-2.0x speedup with 2 GPUs.

## Input Format

The script now takes the preprocessed `.jgz` annotation files produced by
`preprocess_co3d.py` (format: `{seq_name: [{filepath, R, T, ...}]}`), which
contain camera pose data needed for distance-based pair selection.

## Quick Start

### Single GPU
```bash
python precompute_warps.py \
    --annotation_file /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \
    --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant \
    --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \
    --romav2_setting turbo \
    --distance_min 0.5 \
    --distance_max 3.0
```

### Dual GPU (2x Speedup)
```bash
python precompute_warps.py \
    --annotation_file /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \
    --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant \
    --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \
    --num_workers 2 \
    --gpu_ids 0 1 \
    --romav2_setting turbo
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--annotation_file` | required | Preprocessed CO3D `.jgz` file with R/T poses |
| `--distance_min` | 0.5 | Minimum camera Euclidean distance for pairs |
| `--distance_max` | 3.0 | Maximum camera Euclidean distance for pairs |
| `--num_pairs_per_sample` | 3 | Max pairs per source frame |
| `--num_workers` | 1 | Parallel workers (1 = single GPU, 2+ = multi-GPU) |
| `--gpu_ids` | Auto | GPU IDs to use (e.g., `0 1`) |
| `--romav2_setting` | turbo | RoMA setting (turbo/outdoor/indoor) |
| `--resume` | False | Skip already computed pairs |

## Distance-Based Pair Selection

Pairs are selected per sequence using camera Euclidean distance:
```
camera_pos = -R^T @ T      # world position from CO3D W2C convention
dist(i, j) = ||pos_i - pos_j||_2
```
Only pairs with `distance_min <= dist <= distance_max` are computed. This
avoids near-duplicate views and views with too little overlap, producing
better-quality training pairs than frame-index proximity.

Use `scripts/visualization/visualize_distance_sampling.py` to inspect warp quality at
different distance ranges before running a large precomputation job.

## Performance Expectations

| Workers | GPUs | Expected Speedup |
|---------|------|------------------|
| 1 | 1 | 1.0x (baseline) |
| 2 | 2 | 1.8-2.0x |
| 3 | 3 | 2.4-2.6x |

## Memory Requirements

Each worker: ~3-5GB (RoMA model + image buffers). For 2 GPUs: ~6-10GB total.

## Usage Examples

### Resume interrupted computation
```bash
python precompute_warps.py \
    --annotation_file hydrant_train.jgz \
    --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant \
    --root_dir /data/co3d_full \
    --resume
```

### Use specific non-adjacent GPUs
```bash
python precompute_warps.py \
    --annotation_file hydrant_train.jgz \
    --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant \
    --root_dir /data/co3d_full \
    --num_workers 2 \
    --gpu_ids 2 3
```

## Output

- `warp_AAAAA_BBBBB.pt` per pair with keys: `warp_ab`, `confidence_ab`, `warp_ba`, `confidence_ba`
- `metadata.json` recording all run parameters:
```json
{
  "annotation_file": "...",
  "distance_min": 0.5,
  "distance_max": 3.0,
  "num_workers": 2,
  "gpu_ids": [0, 1],
  "num_pairs": 1000,
  ...
}
```

## How It Works

### Single-Worker Mode
```
Main Process (GPU 0)
├─ Load annotations (with R/T)
├─ Compute per-sequence distance matrices
├─ Select pairs by camera distance
└─ Process sequentially
```

### Multi-Worker Mode
```
Main Process
├─ Load annotations, compute distance matrices, collect pairs
├─ Worker 0 (GPU 0): Load RoMA, process batch 0
└─ Worker 1 (GPU 1): Load RoMA, process batch 1
```

## Troubleshooting

**Out of Memory**: reduce `--num_workers`, or use `--romav2_setting turbo`

**No pairs found**: widen `--distance_min` / `--distance_max` range (check
typical distances with `visualize_distance_sampling.py`)

**Workers seem stuck**: check `nvidia-smi`, try single-worker mode first
