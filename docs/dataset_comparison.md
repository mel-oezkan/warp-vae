# Dataset Comparison: Toybus (old good run) vs Hydrant (recent runs)

Generated: 2026-03-06

## Dataset Overview

| Metric | Toybus (old) | Hydrant (new) |
|--------|-------------|---------------|
| Annotation file | `/data/lab_moezkan/co3d_bboxes/toybus_test.jgz` | `/visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz` |
| Sequences | 26 | 604 |
| Total frames | 4,798 | 60,655 |
| Avg frames/seq | 184.5 | 100.4 |
| Has R/T poses | Yes | Yes |
| Has bboxes | Yes | Yes |

## Image Properties

| Metric | Toybus | Hydrant |
|--------|--------|---------|
| Resolution range | 328x342 to 1923x1902 | 692x1068 to 1900x1941 |
| Mean resolution | 922 x 1072 | 1040 x 1657 |
| Mean bbox coverage | 14.1% of image area | 23.4% of image area |
| Bbox coverage range | 4.2% - 30.5% | 8.5% - 64.4% |
| Image cropping used | **No** (full image -> 256x256) | **Yes** (bbox crop -> 256x256) |

## Precomputed Warps

### Warp Directories

| Directory | Pair selection | Distance range | Pairs | Cycle filter |
|-----------|---------------|----------------|-------|-------------|
| `/data/lab_moezkan/precomputed_warps/toybus` | Frame-index proximity (max_pair_distance=20) | Unrestricted | 15,596 | No |
| `.../hydrant_cropped` | Camera Euclidean distance, stratified bins | [0.5, 3.0) | 43,625 | Yes (threshold=0.1) |
| `.../hydrant_cropped_close` | Camera Euclidean distance, stratified bins | [0.5, 1.5) | 33,861 | Yes (threshold=0.1) |

### Camera Distance Distribution

#### Toybus (frame-index based pairing)
```
Distance range    Pairs    Fraction
[0.0, 0.5)         8       4.0%
[0.5, 1.0)        14       7.0%
[1.0, 1.5)        12       6.0%
[1.5, 2.0)        11       5.5%
[2.0, 2.5)        10       5.0%
[2.5, 3.0)        19       9.5%
[3.0, 5.0)        35      17.5%
[5.0, 10.0)       54      27.0%
[10.0, 50.0)      37      18.5%

Min: 0.128,  Mean: 6.009,  Median: 4.516,  Max: 29.414
10th percentile: 0.927    25th: 2.419    75th: 8.361    90th: 12.999
```

**63% of toybus pairs have camera distance > 3.0** -- wide baselines with strong parallax.

#### Hydrant cropped close
All pairs in **[0.5, 1.5)** by construction -- very close views, minimal parallax.

#### Hydrant cropped (wide)
Pairs in **[0.5, 1.5)** and **[1.5, 3.0)** -- moderate baselines but still much narrower than toybus.

### Warp Quality (RoMA Confidence)

| Metric | Toybus | Hydrant close |
|--------|--------|--------------|
| Mean confidence | 0.459 | **0.730** |
| Std confidence | 0.187 | 0.121 |
| Fraction conf > 0.2 | 56.5% | **81.6%** |
| Fraction conf > 0.5 | 48.0% | **76.5%** |

Hydrant warps have significantly higher confidence due to closer camera distances. However, this does **not** mean they provide better training signal -- close views have near-identity warps that are trivially consistent.

### Warp Computation Details

| Setting | Toybus | Hydrant (both dirs) |
|---------|--------|---------------------|
| RoMA setting | turbo | turbo |
| Image size | 256 | 256 |
| Warp resolution | 256 | 256 |
| Crop images | **false** | **true** |
| Pairs per sample | 3 (per frame) | 1 per bin per frame |
| Cycle-consistency filter | No | Yes (threshold=0.1) |

## WandB Runs by Camera Distance Range

### 1. Toybus — Unrestricted distance (frame-index pairing, `max_pair_distance=20`)
- **Runs**: 5 runs from Jan 27–30 (`slick-sidewinder`, `grumpy-statuesque-jellyfish`, `massive-accurate-okapi`, `statuesque-super-cuckoo`)
- **Warp dir**: `/data/lab_moezkan/precomputed_warps/toybus`
- **Distance range**: No camera-distance filtering — 63% of pairs had distance > 3.0, range up to ~29. Mean distance: 6.0.
- These are the "good" baseline runs.

### 2. Hydrant — Full range `[0.5, 3.0)` (old pair selection, no cropping)
- **Runs**: Feb 17 (`utopian-adamant-limpet`, `cunning-lyrical-kakapo`) — both **failed**
- **Warp dir**: `.../hydrant` (distance_min=0.5, distance_max=3.0)

### 3. Hydrant cropped — `[0.5, 1.5)` ∪ `[1.5, 3.0)` (stratified bins `[0.5, 1.5, 3.0]`)
- **Runs**: Feb 23 (`little-knowing-earthworm` — **finished**), Feb 27 (`uptight-beige-gecko` — **failed**), Mar 1 (`quaint-boisterous-bullfinch` — **finished**)
- **Warp dir**: `.../hydrant_cropped`
- **Distance bins**: `[0.5, 1.5, 3.0)` — moderate baselines

### 4. Hydrant cropped close — `[0.5, 1.5)` only (close views only)
- **Runs**: Mar 3 (`new-ultra-caracal` — **failed**, `roaring-eminent-smilodon` — **crashed**, `complex-fragrant-peacock` — **finished**)
- **Warp dir**: `.../hydrant_cropped_close`
- **Distance bins**: `[0.5, 1.5)` — very close views, minimal parallax
- Tags: `close-warps`, `rebalanced`

### 5. Hydrant 50seq nocrop — Uniform pairing (no distance filtering)
- **Run**: Mar 7 (`micro-worm-of-perpetual-storm` — **failed**)
- **Warp dir**: `.../hydrant_50seq_nocrop`
- **Pair mode**: `uniform` — no distance bins, unrestricted (like toybus approach)

## Implications for Training

1. **Geometric signal strength**: Toybus pairs at distance 5-30 provide strong multi-view constraints that force the encoder to learn view-invariant features. Hydrant close pairs ([0.5, 1.5)) are nearly identical views where warp consistency is trivially satisfiable without real 3D understanding.

2. **Dataset scale**: Hydrant is 12x larger (60k vs 5k frames). With 10 epochs on hydrant vs 20 on toybus, the model sees each hydrant image ~10x fewer times relative to total training steps.

3. **Image content**: Toybus uses full uncropped images (object = ~14% of frame, lots of background). Hydrant uses bbox-cropped images (object fills the frame). Cropped images should be easier to reconstruct.

4. **Recommendation**: Use wider distance bins for hydrant (e.g., `--distance_bins 0.5 1.5 3.0 5.0 10.0`) to match toybus's geometric diversity. The current hydrant_cropped ([0.5, 3.0)) is a step in the right direction but still much tighter than toybus.
