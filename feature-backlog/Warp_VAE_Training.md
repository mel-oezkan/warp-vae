# Warp VAE Training Documentation

This document describes the warp-based VAE training pipeline that uses RoMaV2 dense correspondences to enforce multi-view consistency in the VAE's latent space.

## Overview

The core idea is to train a VAE that produces **3D-aware latent representations**. Given two images of the same object from different viewpoints, we want corresponding pixels to have similar latent representations.

```
Image A (view 1) ──► Encoder ──► Latent A ──┐
                                            ├──► Warp Consistency Loss
Image B (view 2) ──► Encoder ──► Latent B ──┘
                         ▲
                         │
              RoMaV2 Warp (A→B)
```

## Architecture

### 1. Dataset: `WarpCO3DDataset`

**File:** `src/data/warp_dataset.py`

The dataset provides paired images from the CO3D dataset along with dense correspondences computed by RoMaV2.

```python
# Each sample contains:
{
    "image": Tensor[3, H, W],           # Source image (normalized to [-1, 1])
    "image_target": Tensor[3, H, W],    # Target image from different viewpoint
    "warp_ab": Tensor[H, W, 2],         # Grid coords to warp A→B
    "warp_ba": Tensor[H, W, 2],         # Grid coords to warp B→A
    "confidence_ab": Tensor[H, W],      # Confidence/overlap mask A→B
    "confidence_ba": Tensor[H, W],      # Confidence/overlap mask B→A
}
```

**Key Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `root_dir` | Path to CO3D dataset | `/data/lab_moezkan/co3d_full` |
| `bb_file` | Bounding box annotations | `.jgz` file |
| `image_size` | Output image resolution | 128 or 256 |
| `romav2_setting` | RoMaV2 model variant | `"turbo"` (fastest) |
| `pair_sampling` | How to select pairs | `"random"` or `"sequential"` |
| `max_pair_distance` | Max frame distance for pairs | 10 |
| `warp_resolution` | Resolution of warp field | Same as image_size |
| `confidence_threshold` | Min confidence for valid regions | 0.5 |

**Warp Computation:**
```python
def _compute_warp(self, img_a, img_b):
    # Compute forward and backward warps using RoMaV2
    pred_ab = self.romav2_model.match(img_a, img_b)
    pred_ba = self.romav2_model.match(img_b, img_a)

    # warp_AB contains normalized grid coordinates [-1, 1]
    # Can be used directly with F.grid_sample()
    warp_ab = pred_ab["warp_AB"]  # Shape: [1, H, W, 2]

    # overlap_AB indicates which pixels have valid correspondences
    confidence_ab = pred_ab["overlap_AB"]  # Shape: [1, H, W, 1]
```

### 2. Loss Functions: `WarpConsistencyLoss`

**File:** `src/losses/warp_consistency.py`

The warp consistency loss enforces that warped latent representations match.

```python
class WarpConsistencyLoss(nn.Module):
    """
    Computes consistency loss between warped and target latent representations.

    Loss = mean(|warp(latent_A, flow_AB) - latent_B| * confidence_AB)
    """

    def forward(self, latent_a, latent_b, warp_ab, warp_ba,
                confidence_ab=None, confidence_ba=None):
        # 1. Resize warp field to latent resolution
        warp_latent = F.interpolate(warp_ab, size=latent_a.shape[2:])

        # 2. Warp latent A to view B
        warped_a_to_b = F.grid_sample(latent_a, warp_latent)

        # 3. Compute weighted L1 loss
        diff = |warped_a_to_b - latent_b|
        loss = (diff * confidence_ab).sum() / confidence_ab.sum()

        # 4. If bidirectional, also compute B→A loss
        if self.bidirectional:
            loss += compute_loss(latent_b, latent_a, warp_ba, confidence_ba)
            loss /= 2

        return loss
```

**Loss Types Available:**
- `WarpConsistencyLoss`: Enforces latent consistency across views
- `WarpReconstructionLoss`: Reconstructs warped image (not used by default)
- `CycleConsistencyLoss`: Enforces A→B→A cycle consistency

### 3. Trainer: `WarpVAETrainer`

**File:** `src/trainer/vae_trainers.py`

The trainer combines standard VAE losses with warp consistency loss.

```python
class WarpVAETrainer(BaseVAETrainer):
    """
    VAE trainer with warp consistency loss for multi-view consistency.

    Total Loss = AE_Loss + warp_weight * Warp_Consistency_Loss

    Where AE_Loss = Reconstruction + KL + Perceptual + (Discriminator)
    """

    def training_step(self, batch, batch_idx):
        # 1. Get source and target images
        img_a = batch["image"]
        img_b = batch["image_target"]

        # 2. Encode and reconstruct source
        recon_a, posterior_a = self.model(img_a)
        latent_a = posterior_a.sample()

        # 3. Encode target (no reconstruction needed)
        latent_b = self.model.encode(img_b).sample()

        # 4. Compute standard VAE loss
        ae_loss = self.model.loss(img_a, recon_a, posterior_a, ...)

        # 5. Compute warp consistency loss
        warp_loss = self.warp_loss(
            latent_a, latent_b,
            batch["warp_ab"], batch["warp_ba"],
            batch["confidence_ab"], batch["confidence_ba"]
        )

        # 6. Combine losses with warmup
        warp_weight = self.get_warp_weight(global_step)  # Ramps up
        total_loss = ae_loss + warp_weight * warp_loss

        return total_loss
```

**Key Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `warp_consistency_weight` | Weight for warp loss | 0.5 - 1.0 |
| `warp_reconstruction_weight` | Weight for warp recon loss | 0.0 |
| `consistency_loss_type` | `"l1"` or `"l2"` | `"l1"` |
| `bidirectional` | Compute both A→B and B→A | `true` |
| `confidence_weighted` | Weight by RoMaV2 confidence | `true` |
| `confidence_threshold` | Min confidence for loss | 0.1 |
| `warmup_steps` | Steps to ramp up warp weight | 1000 |
| `vanilla_probability` | Probability of using vanilla loss only (no warp loss) | 0.0 |

## Configuration

**File:** `config/warp_vae_co3d_small.yaml`

```yaml
model:
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    embed_dim: 4
    ddconfig:
      resolution: 128
      ch: 64
      ch_mult: [1, 2, 4]
      # ... standard VAE config

trainer:
  target: src.trainer.vae_trainers.WarpVAETrainer
  params:
    target_key: "image_target"
    warp_consistency_weight: 0.5
    warp_reconstruction_weight: 0.0
    consistency_loss_type: "l1"
    bidirectional: true
    confidence_weighted: true
    warmup_steps: 1000
    vanilla_probability: 0.5              # Set > 0 to randomly skip warp loss

data:
  target: src.data.datamodule.VAEDataModule
  params:
    dataset_config:
      type: warp_co3d              # Uses WarpCO3DDataset
      params:
        root_dir: "/data/lab_moezkan/co3d_full"
        bb_file: "/data/lab_moezkan/co3d_bboxes/toybus_test.jgz"
        image_size: 128
        romav2_setting: "turbo"
        pair_sampling: "random"
        max_pair_distance: 10
    num_workers: 0                 # MUST be 0 for RoMaV2 (CUDA in workers)
```

## Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Step                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DataLoader loads batch:                                     │
│     ┌──────────────────────────────────────────────┐            │
│     │ WarpCO3DDataset.__getitem__():               │            │
│     │   - Load image pair from CO3D                │            │
│     │   - Compute RoMaV2 warp on-the-fly           │            │
│     │   - Return {image, image_target, warps...}   │            │
│     └──────────────────────────────────────────────┘            │
│                           │                                     │
│                           ▼                                     │
│  2. Forward pass:                                               │
│     ┌──────────────────────────────────────────────┐            │
│     │ img_a ──► Encoder ──► posterior_a ──► latent_a            │
│     │ img_b ──► Encoder ──► posterior_b ──► latent_b            │
│     │ latent_a ──► Decoder ──► recon_a                          │
│     └──────────────────────────────────────────────┘            │
│                           │                                     │
│                           ▼                                     │
│  3. Loss computation:                                           │
│     ┌──────────────────────────────────────────────┐            │
│     │ AE Loss:                                     │            │
│     │   - L1/L2 reconstruction: |recon_a - img_a|  │            │
│     │   - KL divergence: KL(posterior || N(0,1))   │            │
│     │   - Perceptual (LPIPS): lpips(recon_a, img_a)│            │
│     │   - Discriminator: D(recon_a) (after warmup) │            │
│     │                                              │            │
│     │ Warp Loss:                                   │            │
│     │   - Resize warp to latent resolution         │            │
│     │   - warped_a = grid_sample(latent_a, warp)   │            │
│     │   - loss = |warped_a - latent_b| * confidence│            │
│     └──────────────────────────────────────────────┘            │
│                           │                                     │
│                           ▼                                     │
│  4. Backward + Optimize:                                        │
│     total_loss = ae_loss + warp_weight * warp_loss              │
│     optimizer.step()                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## RoMaV2 Integration Details

### Model Variants

| Setting | Resolution | Speed | Memory | Accuracy |
|---------|------------|-------|--------|----------|
| `turbo` | Low | Fastest | ~2GB | Good |
| `outdoor` | Medium | Medium | ~4GB | Better |
| `indoor` | Medium | Medium | ~4GB | Better (indoor) |

### Warp Field Format

RoMaV2 returns warps as normalized grid coordinates compatible with `torch.nn.functional.grid_sample()`:

```python
# warp_AB[i, j] = (x, y) where:
#   x, y ∈ [-1, 1] are normalized coordinates
#   (-1, -1) = top-left corner
#   (+1, +1) = bottom-right corner

# To warp image A to view B:
warped_a = F.grid_sample(
    img_a,              # [B, C, H, W]
    warp_ab,            # [B, H, W, 2]
    mode="bilinear",
    padding_mode="border",
    align_corners=False
)
```

### Confidence/Overlap Mask

RoMaV2 provides confidence scores indicating which pixels have valid correspondences:

```python
# High confidence: pixel visible in both views with reliable match
# Low confidence: occlusion, out-of-view, or uncertain match

# Used to weight the warp consistency loss:
loss = (|warped_latent - target_latent| * confidence).sum() / confidence.sum()
```

## Important Notes

### 1. CUDA in DataLoader Workers

RoMaV2 uses CUDA, which cannot be used in forked DataLoader workers. **Must set `num_workers=0`**.

```yaml
data:
  params:
    num_workers: 0  # Required for RoMaV2
```

### 2. Memory Considerations

**Small Model (warp_vae_co3d_small):**
- VAE model (ch=64): ~0.1GB VRAM
- RoMaV2 model: ~2GB VRAM
- Training batch (bs=8, 128×128): ~3GB VRAM
- **Total**: ~5-6GB per GPU

**Full Model (warp_vae_co3d):**
- VAE model (ch=128): ~0.3GB VRAM
- RoMaV2 model: ~2GB VRAM
- Training batch (bs=2, 256×256): ~6GB VRAM
- **Total**: ~8-10GB per GPU (fits on GTX 1080 Ti)

| Config | Batch Size | Peak VRAM (training) |
|--------|------------|----------------------|
| `warp_vae_co3d_small` | 8 | ~5GB |
| `warp_vae_co3d` | 2 | ~8GB |
| `warp_vae_co3d` | 4 | OOM on 11GB |

### 3. Warp Warmup

The warp loss weight ramps up gradually to prevent early training instability:

```python
def get_warp_weight(self, step):
    if step < self.warmup_steps:
        return self.warp_weight * (step / self.warmup_steps)
    return self.warp_weight
```

### 4. Latent Space Resolution

The warp field is computed at image resolution but must be resized for latent space:

```python
# Image: 128x128 → Latent: 32x32 (with downsampling factor 4)
warp_latent = F.interpolate(
    warp_ab.permute(0, 3, 1, 2),  # [B, 2, H, W]
    size=(32, 32),
    mode="bilinear"
).permute(0, 2, 3, 1)  # [B, 32, 32, 2]
```

### 5. FP16 Numerical Stability

When training with mixed precision (`precision: 16`), several measures are required to prevent NaN losses:

#### Gradient Clipping

The trainer uses **manual optimization** (dual optimizers for autoencoder and discriminator), which means PyTorch Lightning's `gradient_clip_val` parameter cannot be used. Instead, gradient clipping is applied manually after `manual_backward()`:

```python
# In WarpVAETrainer.training_step():

# Autoencoder optimization with gradient clipping
opt_ae.zero_grad()
self.manual_backward(total_ae_loss)
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
opt_ae.step()

# Discriminator optimization with gradient clipping
opt_disc.zero_grad()
self.manual_backward(discloss)
torch.nn.utils.clip_grad_norm_(self.model.loss.discriminator.parameters(), max_norm=1.0)
opt_disc.step()
```

#### Logvar Clamping in Loss Function

The `LPIPSWithDiscriminator` loss function computes NLL loss using a learned `logvar` parameter. Without clamping, extreme `logvar` values can cause overflow/underflow in FP16:

```python
# In ldm/modules/losses/contperceptual.py:

# Clamp logvar to prevent extreme values that cause overflow/underflow
logvar_clamped = torch.clamp(self.logvar.float(), min=-10.0, max=10.0)
nll_loss = rec_loss.float() / torch.exp(logvar_clamped) + logvar_clamped
```

**Why this matters:**
- If `logvar` becomes very negative (e.g., -20), `torch.exp(logvar)` is ~1e-9
- Dividing `rec_loss` by this tiny value causes overflow in FP16 (max ~65504)
- The clamping range `[-10, 10]` keeps `exp(logvar)` in a safe range (~4.5e-5 to ~22026)

#### Summary of FP16 Stability Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| Exploding gradients | NaN loss after many steps | Manual gradient clipping (max_norm=1.0) |
| Logvar overflow | NaN in nll_loss computation | Clamp logvar to [-10, 10] |
| FP16 precision loss | Accumulated errors | Cast rec_loss to float32 before division |

## Evaluation

**File:** `eval_warp_vae.py`

The evaluation script visualizes:
1. **Dataset samples**: Source/target pairs with warps
2. **Warp flow fields**: Dense correspondence visualization
3. **Model outputs**: Reconstructions and warped latents
4. **Latent consistency**: Statistics over many samples

Run evaluation:
```bash
# Without checkpoint (random init baseline)
python eval_warp_vae.py --num_samples 4

# With trained checkpoint
python eval_warp_vae.py --checkpoint ./outputs/warp_vae_co3d_small/<run>/checkpoints/last.ckpt

# Skip model evaluation (dataset-only)
python eval_warp_vae.py --skip_model
```

### What the Evaluation Script Does

The script produces 4 visualizations saved to `./eval_outputs/`:

1. **Dataset Samples** (`dataset_samples.png`)
   - Displays source/target image pairs from the CO3D dataset
   - Warps the source image to the target view using RoMaV2 correspondences via `F.grid_sample()`
   - Shows confidence masks (RoMaV2's certainty about correspondences)
   - Computes pixel-wise warp error: `|warped_source - target|`
   - Useful for validating that the dataset and warp fields are working correctly

2. **Warp Flow Fields** (`warp_flow.png`)
   - Converts RoMaV2's normalized grid coordinates to pixel displacements
   - Visualizes flow magnitude as a heatmap (how far each pixel moves)
   - Draws quiver plots showing flow direction vectors overlaid on the source image
   - Helps diagnose warp quality and identify problematic regions

3. **Model Outputs** (`model_outputs.png`)
   - Runs the VAE encoder/decoder on source and target images
   - Shows: source → reconstruction → target → warped source → warped reconstruction
   - Computes **latent space difference maps**: warps `latent_a` to target view and compares with `latent_b`
   - The latent diff heatmap is the key visualization - lower values = better 3D consistency

4. **Latent Consistency Stats** (`latent_consistency.png`)
   - Aggregates statistics over 100 random samples
   - Plots histograms for three metrics:
     - **Latent Consistency**: confidence-weighted L1 between `warp(latent_a) - latent_b` (primary metric)
     - **Reconstruction L1**: standard VAE reconstruction quality
     - **Warp L1 Error**: image-space warp quality (baseline sanity check)
   - Reports mean ± std for each metric
   - Use this to compare random init (~4.0 consistency) vs trained (<1.0 target)

### Key Implementation Details

- Loads `WarpCO3DDataset` with hardcoded paths matching training config
- Creates VAE with same architecture as `warp_vae_co3d_small.yaml` (ch=64, ch_mult=[1,2,4])
- Auto-detects latest checkpoint in `./outputs/warp_vae_co3d_small/` if none specified
- Handles checkpoint format variations (strips `model.` prefix if present)
- Disables `torch.compile` for older GPU compatibility
## Training Commands

### Small Model (ch=64, for testing/limited VRAM)
```bash
# Small model at 128x128
python train.py --config-name=warp_vae_co3d_small
```

### Full SD-VAE 2.1 Architecture (ch=128, production)
```bash
# Full SD-VAE architecture at 256x256
# Requires ~11GB VRAM per GPU (fits on GTX 1080 Ti with batch_size=2)
python train.py --config-name=warp_vae_co3d
```

### Precomputed Warps (Recommended for Production)
```bash
# Step 1: Precompute RoMaV2 warps (one-time, can run overnight)
python precompute_warps.py \
    --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz \
    --output_dir /data/lab_moezkan/precomputed_warps/toybus \
    --root_dir /data/lab_moezkan/co3d_full \
    --romav2_setting turbo \
    --max_pair_distance 20 \
    --num_pairs_per_sample 3 \
    --warp_resolution 256

# Step 2: Train with precomputed warps (much faster!)
python train.py --config-name=warp_vae_co3d_precomputed
```

### Specifying GPUs (optional)
```bash
# Use specific GPU(s)
CUDA_VISIBLE_DEVICES=0 python train.py --config-name=warp_vae_co3d
CUDA_VISIBLE_DEVICES=0,1 python train.py --config-name=warp_vae_co3d
```

## Precomputed Warps Training

### Overview

The precomputed warp pipeline separates the expensive RoMaV2 correspondence computation from the training loop, providing significant speedups and enabling:

- **Parallel data loading** (`num_workers > 0`)
- **Larger batch sizes** (~2GB VRAM freed from RoMaV2)
- **Faster iterations** (no RoMaV2 forward pass during training)
- **Better GPU utilization** (single model on GPU)

### Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PRECOMPUTATION PHASE (One-time)                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CO3D Dataset                                                        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐ │
│  │ Image Pairs │ ──► │   RoMaV2    │ ──► │  Precomputed Warps      │ │
│  │  (A, B)     │     │   Model     │     │  warp_XXXXX_YYYYY.pt    │ │
│  └─────────────┘     └─────────────┘     └─────────────────────────┘ │
│                                                                      │
│  Output per pair:                                                    │
│    - warp_ab: (H, W, 2) normalized grid coordinates                  │
│    - confidence_ab: (H, W) correspondence confidence                 │
│    - warp_ba: (H, W, 2) reverse direction                            │
│    - confidence_ba: (H, W) reverse confidence                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE (Fast!)                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐                                                 │
│  │ DataLoader      │  num_workers=4, pin_memory=True                 │
│  │ (parallel)      │                                                 │
│  └────────┬────────┘                                                 │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ PrecomputedWarpDataset                                          │ │
│  │   - Load image pair from disk                                   │ │
│  │   - Load precomputed warp from .pt file                         │ │
│  │   - Resize warp if needed (with warning)                        │ │
│  │   - Apply confidence threshold                                   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ WarpVAETrainer (same as online training)                        │ │
│  │   - Standard VAE forward pass                                   │ │
│  │   - Warp consistency loss in latent space                       │ │
│  │   - Gradient accumulation if needed                              │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Precomputation Script

**File:** `precompute_warps.py`

```bash
python precompute_warps.py \
    --bb_file <path_to_bbox_file.jgz> \
    --output_dir <output_directory> \
    --root_dir <co3d_root> \
    --romav2_setting turbo \
    --max_pair_distance 20 \
    --num_pairs_per_sample 3 \
    --warp_resolution 256 \
    --resume  # Continue from existing files
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--bb_file` | Path to CO3D bounding box .jgz file | Required |
| `--output_dir` | Directory to save precomputed warps | Required |
| `--root_dir` | CO3D dataset root directory | `/data/lab_moezkan/co3d_full` |
| `--romav2_setting` | RoMaV2 model variant | `turbo` |
| `--image_size` | Image size for warp computation | 256 |
| `--warp_resolution` | Output warp field resolution | 256 |
| `--max_pair_distance` | Maximum frame distance for pairs | 20 |
| `--num_pairs_per_sample` | Number of pairs per sample | 3 |
| `--resume` | Skip already computed warps | False |

**Output Structure:**

```
output_dir/
├── metadata.json           # Precomputation settings
├── warp_00000_00001.pt     # Warp data for pair (0, 1)
├── warp_00000_00005.pt     # Warp data for pair (0, 5)
├── warp_00001_00003.pt     # ...
└── ...
```

### Dataset: `PrecomputedWarpDataset`

**File:** `src/data/warp_dataset.py`

The dataset automatically:
1. Discovers available warp pairs from the output directory
2. Loads and resizes warps to match training image size (with warning if mismatch)
3. Applies confidence thresholding
4. Returns the same format as `WarpCO3DDataset`

**Key Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `root_dir` | CO3D dataset root | Required |
| `bb_file` | Bounding box file | Required |
| `warp_dir` | Directory with precomputed warps | Required |
| `image_size` | Training image size | 256 |
| `confidence_threshold` | Min confidence for valid matches | 0.1 |

### Configuration

**File:** `config/warp_vae_co3d_precomputed.yaml`

Key differences from online training:

```yaml
data:
  params:
    dataset_config:
      type: precomputed_warp        # Uses PrecomputedWarpDataset
      params:
        warp_dir: "/path/to/precomputed_warps"
    num_workers: 4                  # Can use workers now!
    pin_memory: true                # Enable for faster transfers
    persistent_workers: true        # Keep workers alive

training:
  batch_size: 4                     # Larger batches possible
  gradient_accumulation_steps: 4    # Can reduce (larger effective batch)
```

### Performance Comparison

| Metric | Online (WarpCO3DDataset) | Precomputed |
|--------|--------------------------|-------------|
| Data loading workers | 0 (required) | 4+ |
| RoMaV2 VRAM usage | ~2GB | 0 |
| Batch size (11GB GPU) | 2 | 4-6 |
| Iteration time | ~1.5s | ~0.3s |
| Disk space | - | ~50MB per 1000 pairs |

### Best Practices

1. **Match resolutions**: Precompute warps at the same resolution as training `image_size` to avoid scaling
2. **Use `--resume`**: For large datasets, precomputation can take hours; use resume if interrupted
3. **Monitor disk space**: Each warp file is ~50KB, plan accordingly for large datasets
4. **Validate first**: Run a small test with 10-20 pairs before full precomputation

## Model Architecture Comparison

| Config | ch | ch_mult | z_channels | Downsampling | Image Size | Latent Size | Params |
|--------|-----|---------|------------|--------------|------------|-------------|--------|
| `warp_vae_co3d_small` | 64 | [1,2,4] | 4 | 8× | 128×128 | 16×16×4 | ~21M |
| `warp_vae_co3d` | 128 | [1,2,4,4] | 4 | 8× | 256×256 | 32×32×4 | ~84M |
| SD-VAE 2.1 (reference) | 128 | [1,2,4,4] | 4 | 8× | 768×768 | 96×96×4 | ~84M |

**Note:** The `warp_vae_co3d` config matches the SD-VAE 2.1 architecture exactly, just trained at 256×256 resolution for memory constraints.

## Expected Results

### Metrics to Monitor

| Metric | Description | Expected Trend |
|--------|-------------|----------------|
| `train/aeloss` | Total AE loss | Decrease |
| `train/warp_loss` | Warp consistency | Decrease |
| `train/rec_loss` | Reconstruction L1 | Decrease |
| `train/kl_loss` | KL divergence | Stable/slight increase |
| `train/discloss` | Discriminator | Activate at step 50001 |

### Baseline vs Trained

| Metric | Random Init | After Training |
|--------|-------------|----------------|
| Latent Consistency | ~4.0 | <1.0 (target) |
| Reconstruction L1 | ~0.45 | <0.15 |
| Warp L1 Error | ~0.30 | ~0.30 (depends on data) |

## File Summary

| File | Purpose |
|------|---------|
| `src/data/warp_dataset.py` | WarpCO3DDataset (online) and PrecomputedWarpDataset |
| `src/losses/warp_consistency.py` | Warp consistency loss functions |
| `src/trainer/vae_trainers.py` | WarpVAETrainer class |
| `precompute_warps.py` | Script to precompute RoMaV2 warps |
| `config/warp_vae_co3d_small.yaml` | Small model config (ch=64, 128×128) |
| `config/warp_vae_co3d.yaml` | Full SD-VAE config (ch=128, 256×256, online warps) |
| `config/warp_vae_co3d_precomputed.yaml` | Precomputed warps config (recommended) |
| `eval_warp_vae.py` | Evaluation and visualization |
| `compare_latents.py` | Compare latent representations across models |
