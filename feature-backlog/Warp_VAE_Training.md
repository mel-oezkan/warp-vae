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

### 2. Loss Functions

#### 2a. AE Loss: `LPIPSWithDiscriminator`

**File:** `ldm/modules/losses/contperceptual.py`

The main autoencoder loss combining reconstruction, perceptual, KL regularization, and adversarial objectives.

**Total loss formula (generator/encoder-decoder optimizer):**

```
L_AE = L_rec + kl_weight * L_KL + d_weight * disc_factor * L_G

where:
  L_rec = L_L1 + perceptual_weight * L_LPIPS
  L_L1  = mean(|inputs - reconstructions|)           # pixel-wise L1
  L_KL  = KL(posterior || N(0,1))                    # latent regularization
  L_G   = -mean(discriminator(reconstructions))      # generator adversarial loss
  d_weight = EMA-normalised gradient ratio            # adaptive balancing weight (GradNorm-inspired)
  disc_factor = 0 until disc_start steps, then 1     # discriminator warmup
```

The reconstruction loss is further weighted by a learned log-variance `logvar` (NLL formulation):

```python
nll_loss = rec_loss / exp(logvar) + logvar   # learnable uncertainty weighting
loss = sum(nll_loss) + kl_weight * kl_loss + d_weight * disc_factor * g_loss
```

**Discriminator loss (separate optimizer):**

```
L_disc = disc_factor * L_hinge(logits_real, logits_fake)

Hinge loss:  0.5 * (mean(relu(1 - logits_real)) + mean(relu(1 + logits_fake)))
Vanilla GAN: 0.5 * (mean(softplus(-logits_real)) + mean(softplus(logits_fake)))
```

**Adaptive Weight (`d_weight`) — EMA-Normalised GradNorm:**

The original `d_weight = ‖nll_grads‖ / ‖g_grads‖` is scale-sensitive: when the generator is pre-warmed by the warp loss, `g_grads` at discriminator activation are near-zero, making the ratio explode. The current implementation normalises each gradient norm by its own EMA before taking the ratio:

```python
# Current implementation (scale-invariant):
ema_nll = decay * ema_nll + (1 - decay) * norm(nll_grads)
ema_g   = decay * ema_g   + (1 - decay) * norm(g_grads)
d_weight = (norm(nll_grads) / ema_nll) / (norm(g_grads) / ema_g + 1e-4)
d_weight = d_weight * discriminator_weight   # no hard clamp
```

Both EMAs are stored as `nn.Module` buffers (move with `.to(device)`, included in `state_dict`). The hard clamp from earlier versions has been removed entirely. This is the GradNorm normalisation step (Chen et al., 2018) applied to the two-term GAN+reconstruction balance.

**Key hyperparameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `disc_start` | Step at which discriminator activates | 50001 |
| `kl_weight` | KL divergence regularization weight | 1e-6 |
| `perceptual_weight` | LPIPS loss weight | 1.0 |
| `disc_weight` | Multiplier on d_weight after normalisation | 0.5 |
| `disc_factor` | Base discriminator scaling factor | 1.0 |
| `disc_loss` | GAN loss type | `"hinge"` or `"vanilla"` |
| `grad_norm_ema_decay` | EMA decay for GradNorm normalisation | 0.99 |

#### 2b. Perceptual Loss: `LPIPS`

**File:** `taming/modules/losses/lpips.py`

Measures perceptual similarity using deep features from a frozen VGG16 backbone.

```
L_LPIPS = Σ(k=0..4) spatial_avg( conv1x1_k( (normalize(feat_k_input) - normalize(feat_k_recon))^2 ) )
```

- **Backbone:** Pre-trained VGG16 (frozen, `requires_grad=False`)
- **Feature levels:** 5 layers — `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`, `relu5_3`
- **Channel dims:** `[64, 128, 256, 512, 512]`
- Features are L2-normalized before computing squared differences
- Per-level learned `1×1` conv layers weight the contribution of each feature level
- Results are spatially averaged and summed across all 5 levels

#### 2c. Warp Consistency Loss: `WarpConsistencyLoss`

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

**Warp loss types available:**

| Type | Description | Used by default |
|------|-------------|-----------------|
| `WarpConsistencyLoss` | Enforces latent consistency across views | Yes |
| `WarpReconstructionLoss` | Compares warped reconstruction to target image (pixel + optional LPIPS) | No (`warp_reconstruction_weight: 0.0`) |
| `CycleConsistencyLoss` | Validates warp quality via A→B→A cycle error | No |

### 3. Trainer: `WarpVAETrainer`

**File:** `src/trainer/vae_trainers.py`

The trainer combines standard VAE losses with warp consistency loss.

```python
class WarpVAETrainer(BaseVAETrainer):
    """
    VAE trainer with warp consistency loss + gradient accumulation.

    Total Loss = AE_Loss + warp_weight * warp_factor * Warp_Consistency_Loss

    Where AE_Loss = Reconstruction + KL + Perceptual + (Discriminator)
    """

    def training_step(self, batch, batch_idx):
        # 1. Encode source → reconstruction + posterior + latent_a
        # Use sample_posterior=False so reconstructions and latent_a reference the
        # same deterministic posterior mean. sample_posterior=True would draw z1
        # for the decode and z2 for latent_a, giving the encoder two gradient
        # signals from different latent points — incoherent and noisy.
        recon_a, posterior_a = self.model(img_a, sample_posterior=False)
        latent_a = posterior_a.mode()

        # 2. Decide whether to skip warp loss (vanilla_probability)
        use_vanilla_only = torch.rand(1).item() < self.vanilla_probability

        # 3. Compute standard VAE loss
        ae_loss = self.model.loss(img_a, recon_a, posterior_a, ...)

        # 4. Compute warp consistency loss (if not vanilla mode)
        if not use_vanilla_only:
            latent_b = self.model.encode(img_b).mode()  # deterministic mode, not sample()
            warp_loss = self.warp_consistency_loss(
                latent_a, latent_b, warp_ab, warp_ba, conf_ab, conf_ba
            )

        # 5. Combine with warmup factor
        warp_factor = min(1.0, global_step / warmup_steps)
        total_loss = ae_loss + warp_weight * warp_factor * warp_loss

        # 6. Gradient accumulation: scale and accumulate
        scaled_loss = total_loss / gradient_accumulation_steps
        self.manual_backward(scaled_loss)
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            opt_ae.step()
            opt_ae.zero_grad()

        # 7. Same pattern for discriminator optimizer
```

**Key Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `warp_consistency_weight` | Weight for warp loss | 1.0 |
| `warp_reconstruction_weight` | Weight for image-space warp loss | 0.0 |
| `consistency_loss_type` | `"l1"`, `"l2"`, `"cosine"`, `"combined"` | `"l1"` |
| `bidirectional` | Compute both A→B and B→A | `true` |
| `confidence_weighted` | Weight by RoMaV2 confidence | `true` |
| `loss_confidence_threshold` | Min confidence for loss computation | 0.2 |
| `warmup_steps` | Steps to linearly ramp up warp weight | 5000 |
| `vanilla_probability` | Probability of skipping warp loss | 0.0 |
| `gradient_accumulation_steps` | Mini-batches to accumulate before update | 4 |

## Configuration

**Primary config:** `config/warp_vae_hydrant.yaml`

```yaml
model:
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    embed_dim: 4
    ddconfig:
      resolution: ${training.image_size}
      ch: 128
      ch_mult: [1, 2, 4, 4]
      # ... standard VAE config (matches SD-VAE 2.1)

    lossconfig:
      target: ldm.modules.losses.LPIPSWithDiscriminator
      params:
        disc_start: 50001
        kl_weight: 0.000001
        disc_weight: 0.5
        grad_norm_ema_decay: 0.99  # EMA-normalised adaptive weight (GradNorm-inspired)

trainer:
  target: src.trainer.vae_trainers.WarpVAETrainer
  params:
    target_key: "image_target"
    warp_consistency_weight: 1.0
    warp_reconstruction_weight: 0.0
    consistency_loss_type: "l1"
    bidirectional: true
    confidence_weighted: true
    loss_confidence_threshold: 0.2
    warmup_steps: 5000
    vanilla_probability: 0              # Always use warp loss
    gradient_accumulation_steps: 1      # No accumulation (matches EQ-VAE reference)

data:
  target: src.data.datamodule.VAEDataModule
  params:
    dataset_config:
      type: precomputed_warp             # Uses PrecomputedWarpDataset (no RoMaV2!)
      params:
        root_dir: "/visinf/projects_students/dlcv2025_groupZ/co3d_full"
        bb_file: "...hydrant_train.jgz"
        warp_dir: "...precomputed_warps/hydrant_cropped"
        image_size: ${training.image_size}
        confidence_threshold: 0.25
        crop_images: true               # Warps computed on cropped images
    num_workers: 4                      # Parallel loading OK (no CUDA in workers)
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 4

training:
  batch_size: 2                         # Per-GPU (2 GPUs × 2 = 4 effective)
  precision: 32                         # fp16 not used (stability)
  limit_train_batches: 21000            # Cap epoch to match EQ-VAE wall time
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

RoMaV2 uses CUDA, which cannot be used in forked DataLoader workers. **Must set `num_workers=0`** for online warps (`WarpCO3DDataset`).

```yaml
# Online warps (WarpCO3DDataset):
data:
  params:
    num_workers: 0  # Required for RoMaV2

# Precomputed warps (PrecomputedWarpDataset) - no restriction:
data:
  params:
    num_workers: 4  # Parallel loading OK, no CUDA in dataloader
```

**This is the primary motivation for precomputing warps** - it eliminates the CUDA-in-workers constraint and enables parallel data loading.

### 2. Memory Considerations

**Precomputed Config (warp_vae_co3d_precomputed) - Recommended:**
- VAE model (ch=128): ~0.3GB VRAM
- No RoMaV2 during training: 0 GB
- Training batch (bs=2, 256×256): ~6GB VRAM
- **Total**: ~6-8GB per GPU

**Online Full Model (warp_vae_co3d):**
- VAE model (ch=128): ~0.3GB VRAM
- RoMaV2 model: ~2GB VRAM
- Training batch (bs=2, 256×256): ~6GB VRAM
- **Total**: ~8-10GB per GPU (fits on GTX 1080 Ti)

**Online Small Model (warp_vae_co3d_small):**
- VAE model (ch=64): ~0.1GB VRAM
- RoMaV2 model: ~2GB VRAM
- Training batch (bs=8, 128×128): ~3GB VRAM
- **Total**: ~5-6GB per GPU

| Config | Batch Size | RoMaV2 | Peak VRAM |
|--------|------------|--------|-----------|
| `warp_vae_co3d_precomputed` | 2 | No | ~6-8GB |
| `warp_vae_co3d_small` | 8 | Yes | ~5GB |
| `warp_vae_co3d` | 2 | Yes | ~8GB |
| `warp_vae_co3d` | 4 | Yes | OOM on 11GB |

### 3. Gradient Accumulation

The `WarpVAETrainer` supports gradient accumulation to simulate larger effective batch sizes without increasing memory:

```python
# gradient_accumulation_steps = 4, batch_size = 2
# Effective batch size = 4 * 2 = 8

# Loss is scaled: scaled_loss = total_loss / accumulation_steps
# Optimizer steps only every N batches:
if (batch_idx + 1) % gradient_accumulation_steps != 0:
    # Accumulate gradients (no optimizer step)
else:
    opt.step()
    opt.zero_grad()
```

Both the autoencoder and discriminator optimizers follow this pattern.

### 4. Warp Warmup

The warp loss weight ramps up linearly to prevent early training instability:

```python
# warp_factor = min(1.0, global_step / warmup_steps)
# effective_weight = warp_consistency_weight * warp_factor
```

With `warmup_steps=5000` (precomputed config), the warp loss reaches full weight at step 5000.

### 5. Latent Space Resolution

The warp field is computed at image resolution but `WarpConsistencyLoss.warp_features()` automatically resizes it to latent resolution:

```python
# Image: 256x256 → Latent: 32x32 (8x downsampling via ch_mult=[1,2,4,4])
# The loss function handles this automatically:
#   - Detects shape mismatch between warp and features
#   - Resizes warp and confidence via F.interpolate (bilinear)
```

### 6. FP16 Numerical Stability

**Note:** The precomputed config defaults to `precision: 32` to avoid these issues entirely. This section applies when using `precision: 16`.

## Evaluation

Evaluation is done via the `WarpVAETrainer.log_images()` method during training (logged to WandB), which generates:
- **Source image**: Original input view
- **Target image**: Paired view from different viewpoint
- **Reconstruction**: VAE reconstruction of source
- **Warped source**: Source image warped to target view via `F.grid_sample(warp_ab)`
- **Warped reconstruction**: VAE reconstruction warped to target view

Key metrics to track in WandB:
- `train/warp_consistency_loss`: Primary multi-view consistency metric
- `train/aeloss`: Standard VAE reconstruction quality
- `val/warp_loss`: Validation warp consistency (generalization check)
## Training Commands

### Warp-consistency only (primary config)
```bash
# Full SD-VAE 2.1 on CO3D hydrant with precomputed warps, warp-consistency loss only
CUDA_VISIBLE_DEVICES=0,1 python train.py --config-name=warp_vae_hydrant
```

### Warp-consistency + image-space reconstruction loss
```bash
# Adds WarpReconstructionLoss on top of warp consistency
CUDA_VISIBLE_DEVICES=1 python train.py --config-name=warp_vae_hydrant_recon
```

### Precompute warps first (one-time, see ROMA_PRECOMPUTE_SPEEDUP.md)
```bash
# Step 1: Precompute RoMaV2 warps with stratified distance sampling
CUDA_VISIBLE_DEVICES=0,1 python precompute_warps.py \
    --annotation_file /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \
    --output_dir /visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant_cropped \
    --root_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \
    --num_workers 2 --gpu_ids 0 1 \
    --romav2_setting turbo \
    --crop_images \
    --distance_bins 0.5 1.5 2.5 3.0 \
    --num_pairs_per_bin 1 \
    --cycle_consistency_threshold 0.1

# Step 2: Train (warp_dir must match output_dir above, with crop_images: true in config)
CUDA_VISIBLE_DEVICES=0,1 python train.py --config-name=warp_vae_hydrant
```

### Specifying GPUs
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config-name=warp_vae_hydrant
CUDA_VISIBLE_DEVICES=0,1 python train.py --config-name=warp_vae_hydrant
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
│                 PRECOMPUTATION PHASE (One-time, multi-GPU)            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CO3D Dataset ──► Generate pair list ──► Split across N GPUs         │
│                                                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐                    │
│  │ Worker 0 (GPU 0)    │  │ Worker 1 (GPU 1)    │  ...               │
│  │ ├─ Load RoMaV2      │  │ ├─ Load RoMaV2      │                    │
│  │ └─ Process batch 0  │  │ └─ Process batch 1  │                    │
│  └─────────────────────┘  └─────────────────────┘                    │
│                          │                                           │
│                          ▼                                           │
│  Output: warp_XXXXX_YYYYY.pt files + metadata.json                   │
│    - warp_ab: (H, W, 2) normalized grid coordinates                  │
│    - confidence_ab: (H, W) correspondence confidence                 │
│    - warp_ba / confidence_ba: reverse direction                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE (Fast!)                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DataLoader (num_workers=4, pin_memory, persistent_workers)          │
│           │                                                          │
│           ▼                                                          │
│  PrecomputedWarpDataset                                              │
│    - Load image pair + precomputed warp from .pt file                │
│    - Auto-resize warp if resolution mismatch                         │
│    - Apply soft confidence threshold                                 │
│           │                                                          │
│           ▼                                                          │
│  WarpVAETrainer                                                      │
│    - Encode source/target → latent_a, latent_b                       │
│    - AE loss (recon + KL + LPIPS + discriminator)                    │
│    - Warp consistency loss (with warmup ramp)                        │
│    - Gradient accumulation (effective_bs = bs * accum_steps)         │
│    - Dual optimizer step: AE + discriminator                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Precomputation Script

**File:** `precompute_warps.py`

Supports **single-GPU** and **multi-GPU** modes. Each worker loads its own RoMaV2 model on its assigned GPU and processes pairs in parallel via `multiprocessing.Pool`.

```bash
python precompute_warps.py \
    --bb_file <path_to_bbox_file.jgz> \
    --output_dir <output_directory> \
    --root_dir <co3d_root> \
    --romav2_setting turbo \
    --max_pair_distance 20 \
    --num_pairs_per_sample 3 \
    --warp_resolution 256 \
    --num_workers 2 --gpu_ids 0 1 \
    --resume  # Continue from existing files
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--bb_file` | Path to CO3D bounding box .jgz file | Required |
| `--output_dir` | Directory to save precomputed warps | Required |
| `--root_dir` | CO3D dataset root directory | `/data/lab_moezkan/co3d_full` |
| `--romav2_setting` | RoMaV2 model variant (`turbo`/`outdoor`/`indoor`) | `turbo` |
| `--image_size` | Image size for warp computation | 256 |
| `--warp_resolution` | Output warp field resolution | 256 |
| `--max_pair_distance` | Maximum frame distance for pairs | 20 |
| `--num_pairs_per_sample` | Number of pairs per sample | 3 |
| `--resume` | Skip already computed warps | False |
| `--num_workers` | Number of parallel GPU workers (1 = single GPU) | 1 |
| `--gpu_ids` | GPU IDs to use (e.g., `0 1 2`) | Auto (first N GPUs) |

**Multi-GPU Architecture:**

```
Single-Worker (--num_workers 1):       Multi-Worker (--num_workers 2):
  Main Process (GPU 0)                   Main Process
  ├─ Load RoMaV2                         ├─ Split pairs into N batches
  └─ Process all pairs                   ├─ Worker 0 (GPU 0)
                                         │  ├─ Load own RoMaV2
                                         │  └─ Process batch 0
                                         └─ Worker 1 (GPU 1)
                                            ├─ Load own RoMaV2
                                            └─ Process batch 1
```

**Expected Speedups:**

| Workers | GPUs | Speedup | Memory per GPU |
|---------|------|---------|----------------|
| 1 | 1 | 1.0x | ~3-5GB |
| 2 | 2 | 1.8-2.0x | ~3-5GB each |
| 3 | 3 | 2.4-2.6x | ~3-5GB each |

**Pair Generation Logic:**
1. Each sample belongs to a CO3D sequence (object + viewpoint track)
2. For each sample, `num_pairs_per_sample` targets are randomly selected within `max_pair_distance` frames
3. Pairs stored as ordered tuples `(min_idx, max_idx)` to avoid duplicates
4. With `--resume`, existing `.pt` files are detected and skipped

**Output Structure:**

```
output_dir/
├── metadata.json           # Settings (bb_file, resolution, num_workers, gpu_ids, etc.)
├── warp_00000_00001.pt     # Warp data for pair (0, 1)
├── warp_00000_00005.pt     # Warp data for pair (0, 5)
├── warp_00001_00003.pt     # ...
└── ...
```

Each `.pt` file contains:
```python
{
    "warp_ab": Tensor[H, W, 2],       # Normalized grid coords A→B
    "confidence_ab": Tensor[H, W],    # Match confidence A→B
    "warp_ba": Tensor[H, W, 2],       # Normalized grid coords B→A
    "confidence_ba": Tensor[H, W],    # Match confidence B→A
}
```

### Dataset: `PrecomputedWarpDataset`

**File:** `src/data/warp_dataset.py`

The dataset automatically:
1. Discovers available warp pairs from the output directory (supports both 5-digit `warp_00000_00001.pt` and 4-digit `warp_0000_0001.pt` naming)
2. Loads warp resolution from `metadata.json` if available
3. Resizes warps to match training `image_size` if resolution differs (logs a warning on first mismatch)
4. Applies **soft confidence thresholding** (not hard cutoff)
5. Returns the same format as `WarpCO3DDataset`

**Key Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `root_dir` | CO3D dataset root | Required |
| `bb_file` | Bounding box file | Required |
| `warp_dir` | Directory with precomputed warps | Required |
| `image_size` | Training image size | 256 |
| `warp_resolution` | Auto-detected from `metadata.json` | `None` |
| `confidence_threshold` | Soft confidence threshold | 0.1 |

**Soft Confidence Thresholding:**

Unlike a hard cutoff, the confidence threshold applies a soft ramp that suppresses low-confidence correspondences while preserving gradients:

```python
# Values below threshold become 0; values above are rescaled to [0, 1]
confidence = clamp(confidence - threshold, min=0) / (1 - threshold)
```

For example, with `confidence_threshold=0.25`:
- Confidence 0.1 → 0.0 (suppressed)
- Confidence 0.25 → 0.0 (boundary)
- Confidence 0.5 → 0.33
- Confidence 1.0 → 1.0

### Configuration

**File:** `config/warp_vae_hydrant.yaml`

Key settings for the precomputed-warp training pipeline:

```yaml
model:
  params:
    lossconfig:
      params:
        disc_start: 50001              # Discriminator activates at step 50k
        kl_weight: 0.000001            # Matches EQ-VAE reference
        grad_norm_ema_decay: 0.99      # EMA for adaptive weight normalisation

trainer:
  params:
    warp_consistency_weight: 1.0
    loss_confidence_threshold: 0.2
    warmup_steps: 5000                 # 5k steps to ramp up warp loss
    vanilla_probability: 0             # Always use warp loss
    gradient_accumulation_steps: 1     # No accumulation (matches EQ-VAE reference)

data:
  params:
    dataset_config:
      type: precomputed_warp           # Uses PrecomputedWarpDataset (no RoMaV2!)
      params:
        warp_dir: ".../hydrant_cropped"  # Must use crop_images: true warp set
        confidence_threshold: 0.25       # Soft threshold for correspondences
        crop_images: true                # Must match how warps were computed
    num_workers: 4                     # Parallel data loading (no CUDA in workers)
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 4                 # Pre-load 4 batches per worker

training:
  batch_size: 2                        # Per-GPU; 2 GPUs → 4 effective
  lr: 4.5e-6
  precision: 32                        # FP32 (fp16 not used)
  gradient_accumulation_steps: 1
  limit_train_batches: 21000           # Cap epoch to ~same wall time as EQ-VAE
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
| `warp_vae_hydrant` | 128 | [1,2,4,4] | 4 | 8× | 256×256 | 32×32×4 | ~84M |
| `warp_vae_hydrant_recon` | 128 | [1,2,4,4] | 4 | 8× | 256×256 | 32×32×4 | ~84M |
| SD-VAE 2.1 (reference) | 128 | [1,2,4,4] | 4 | 8× | 768×768 | 96×96×4 | ~84M |

Both hydrant configs share the same SD-VAE 2.1 architecture, differing only in whether the `WarpReconstructionLoss` is enabled.

## Expected Results

### Metrics to Monitor

| Metric | Description | Expected Trend |
|--------|-------------|----------------|
| `train/aeloss` | Standard AE loss (recon + KL + perceptual) | Decrease |
| `train/warp_loss` | Warp consistency loss (weighted) | Decrease |
| `train/total_ae_loss` | aeloss + warp_loss | Decrease |
| `train/warp_consistency_loss` | Raw consistency (unweighted) | Decrease |
| `train/warp_factor` | Warmup multiplier (0→1) | Ramp to 1.0 over warmup_steps |
| `train/discloss` | Discriminator | Activate at `disc_start` (15000) |
| `train/vanilla_mode` | Warp loss skipped (0/1) | Depends on `vanilla_probability` |

### Baseline vs Trained

| Metric | Random Init | After Training |
|--------|-------------|----------------|
| Latent Consistency | ~4.0 | <1.0 (target) |
| Reconstruction L1 | ~0.45 | <0.15 |
| Warp L1 Error | ~0.30 | ~0.30 (depends on data) |

## File Summary

| File | Purpose |
|------|---------|
| `src/data/warp_dataset.py` | `WarpCO3DDataset` (online) and `PrecomputedWarpDataset` |
| `src/losses/warp_consistency.py` | `WarpConsistencyLoss`, `WarpReconstructionLoss`, `CycleConsistencyLoss` |
| `src/trainer/vae_trainers.py` | `WarpVAETrainer` class with gradient accumulation |
| `ldm/modules/losses/contperceptual.py` | `LPIPSWithDiscriminator` with EMA-normalised adaptive weight |
| `precompute_warps.py` | Multi-GPU script to precompute RoMaV2 warps |
| `config/warp_vae_hydrant.yaml` | Primary config: warp-consistency only, hydrant, precomputed |
| `config/warp_vae_hydrant_recon.yaml` | Variant: adds image-space warp reconstruction loss |
