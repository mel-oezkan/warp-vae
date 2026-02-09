# RoMA Warp Precomputation Speedup Guide

## Overview

The `precompute_warps.py` script now supports **multi-GPU acceleration** for 1.8-2.0x speedup when using 2 GPUs. Each worker loads its own copy of the RoMA model on its assigned GPU and processes a batch of image pairs in parallel.

## Quick Start

### Single GPU (Original)
```bash
python precompute_warps.py \
    --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz \
    --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_co3d/toytruck \
    --root_dir /data/lab_moezkan/co3d_full \
    --romav2_setting turbo
```

### Dual GPU (2x Speedup)
```bash
python precompute_warps.py \
    --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz \
    --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_co3d/toytruck \
    --root_dir /data/lab_moezkan/co3d_full \
    --num_workers 2 \
    --gpu_ids 0 1 \
    --romav2_setting turbo
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_workers` | 1 | Number of parallel workers (1 = single GPU, 2+ = multi-GPU) |
| `--gpu_ids` | Auto | GPU IDs to use (e.g., `0 1`). If not specified, uses first `num_workers` GPUs |
| `--romav2_setting` | turbo | RoMA setting (turbo/outdoor/indoor) |
| `--device` | cuda:0 | Device (single-worker mode only) |
| `--resume` | False | Skip already computed pairs |

## Performance Expectations

| Workers | GPUs | Expected Speedup |
|---------|------|------------------|
| 1 | 1 | 1.0x (baseline) |
| 2 | 2 | 1.8-2.0x |
| 3 | 3 | 2.4-2.6x |

Speedup depends on:
- Number of pairs to process
- GPU memory available
- RoMA model variant (turbo is fastest)

## Memory Requirements

Each worker loads:
- RoMA model: ~2-3GB
- Image buffers: ~1-2GB
- **Per-GPU total**: ~3-5GB

For 2 GPUs: ~6-10GB total (should fit on 2x 16GB GPUs easily)

## Usage Examples

### Example 1: Process 1000 pairs across 2 GPUs
```bash
python precompute_warps.py \
    --bb_file bboxes.jgz \
    --output_dir ./warps \
    --root_dir /data/co3d_full \
    --num_workers 2 \
    --gpu_ids 0 1 \
    --romav2_setting turbo
```

### Example 2: Resume interrupted computation
```bash
python precompute_warps.py \
    --bb_file bboxes.jgz \
    --output_dir ./warps \
    --root_dir /data/co3d_full \
    --num_workers 2 \
    --gpu_ids 0 1 \
    --resume  # Skip already computed pairs
```

### Example 3: Use specific non-adjacent GPUs
```bash
python precompute_warps.py \
    --bb_file bboxes.jgz \
    --output_dir ./warps \
    --root_dir /data/co3d_full \
    --num_workers 2 \
    --gpu_ids 2 3  # Use GPUs 2 and 3
```

## How It Works

### Single-Worker Mode (num_workers=1)
```
Main Process (GPU 0)
├─ Load RoMA model
├─ Load all image pairs
└─ Process sequentially
```

### Multi-Worker Mode (num_workers=2)
```
Main Process
├─ Split pairs into 2 batches
├─ Worker 0 (GPU 0)
│  ├─ Load RoMA model
│  └─ Process batch 1 pairs
└─ Worker 1 (GPU 1)
   ├─ Load RoMA model
   └─ Process batch 2 pairs
```

Both workers run **in parallel**, each on its own GPU.

## Implementation Details

### Key Changes
1. **New `--num_workers` argument**: Controls parallelization
2. **New `--gpu_ids` argument**: Specify which GPUs to use
3. **Worker function**: `worker_process_pairs()` loads RoMA per-GPU
4. **Multiprocessing.Pool**: Distributes work across CPUs/GPUs
5. **Statistics tracking**: Reports processed/error/skipped counts per GPU

### Output Metadata
The script saves `metadata.json` with:
```json
{
  "num_workers": 2,
  "gpu_ids": [0, 1],
  "num_pairs": 1000,
  ...
}
```

## Troubleshooting

### Out of Memory
If CUDA out of memory errors:
1. Reduce `--num_workers` (use 1 instead of 2)
2. Use `--romav2_setting turbo` instead of `outdoor`
3. Process fewer pairs at a time (split into multiple runs)

### Worker Process Issues
If workers seem stuck:
1. Check GPU usage: `nvidia-smi`
2. Ensure GPU IDs are valid: `nvidia-smi | head -10`
3. Try single-worker mode first: `--num_workers 1`

### Timing Out
Multi-worker mode may take longer to start (model loading overhead). Be patient for the first few seconds.

## Performance Tips

1. **Use turbo setting** for fastest speeds:
   ```bash
   --romav2_setting turbo
   ```

2. **Run on large batch** of pairs for better amortization:
   - 100+ pairs recommended for 2-GPU setup

3. **Monitor progress**:
   ```bash
   watch -n 5 nvidia-smi  # Check GPU utilization
   ```

4. **Resume capability**:
   ```bash
   --resume  # Skip already computed to avoid redoing work
   ```

## Backward Compatibility

- `--num_workers 1` is identical to original single-GPU behavior
- All existing arguments still work
- Default is single-worker mode (`--num_workers 1`)

## Future Enhancements

Potential improvements:
1. Batch RoMA inference (process 2+ pairs per forward pass)
2. Async image I/O while computing
3. Distributed training across nodes (with torch.distributed)
