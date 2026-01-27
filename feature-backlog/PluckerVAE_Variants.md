# PluckerVAE Variants - Design Document

**Status: IMPLEMENTED**

## Overview

This document describes three new variants of the PluckerVAE model that explore different ways of integrating Plücker ray coordinates into the VAE architecture to improve 3D awareness.

## Implementation Summary

All three variants have been implemented:

| Variant | Model Class | Trainer Class | Config File |
|---------|-------------|---------------|-------------|
| 1 | `ConcatPluckerVAE` | `ConcatPluckerVAETrainer` | `concat_plucker_vae_co3d.yaml` |
| 2 | `DirectPluckerVAE` | `DirectPluckerVAETrainer` | `direct_plucker_vae_co3d.yaml` |
| 3 | `PluckerConditionedVAE` | `PluckerConditionedVAETrainer` | `plucker_conditioned_vae_co3d.yaml` |

### Key Files Modified/Created:
- `ldm/models/autoencoder.py` - Added three new model classes
- `src/trainer/vae_trainers.py` - Added three new trainer classes
- `data_process/plucker.py` - Added `plucker_full_image()` function
- `src/data/co3d_dataset.py` - Added `full_resolution_plucker` parameter
- `config/*.yaml` - Three new config files

## Current Implementation (Baseline)

The current `PluckerAutoencoder` in `ldm/models/autoencoder.py`:
- Takes image (3 channels) as input
- Encoder produces features, then a `pluck_head` predicts Plücker coordinates
- Interpolates to `n_patches × n_patches` (e.g., 8×8 = 64 patches)
- MLP refines the Plücker predictions
- Decoder only reconstructs the image
- Loss: MSE reconstruction + hybrid Plücker loss (recon + orthogonality constraint + normalization)

```
Image (B, 3, H, W)
       |
   [Encoder]
       |
  h (encoder features)
     /    \
[quant_conv]  [pluck_head + MLP]
     |              |
  z ~ N(μ,σ)   Plücker (B, 64, 6)
     |
  [Decoder]
     |
Recon Image
```

---

## Variant 1: ConcatPluckerVAE (Separate Latent Nodes)

### Concept
Full-resolution Plücker rays are concatenated with the image as input. The encoder produces three separate latent distributions: one for image features, one for Plücker direction (d), and one for Plücker moment (m). The decoder has three corresponding output heads.

### Architecture
```
Image (B, 3, H, W)  +  Plücker_GT (B, 6, H, W)
               \        /
            [Concatenate] → (B, 9, H, W)
                   |
            [Encoder (in_ch=9)]
                   |
            h (encoder features)
           /       |       \
  [quant_conv_img] [quant_conv_d] [quant_conv_m]
         |           |           |
    z_img ~ N     z_d ~ N     z_m ~ N
    (4 ch)       (3 ch)      (3 ch)
                \    |    /
             [Concatenate] → z_combined (10 ch)
                    |
            [Decoder (modified)]
                    |
           [Multi-Head Output]
            /       |       \
     Recon_img  Recon_d   Recon_m
    (B,3,H,W)  (B,3,H,W) (B,3,H,W)

Additional: pluck_head on encoder for direct prediction
```

### Losses
1. **MSE Image Reconstruction**: `MSE(recon_img, input_img)`
2. **MSE Direction Reconstruction**: `MSE(recon_d, gt_d)`
3. **MSE Moment Reconstruction**: `MSE(recon_m, gt_m)`
4. **L1 Encoder Plücker**: `L1(encoder_pluck_pred, gt_pluck)` - from auxiliary head
5. **KL Divergence**: Three separate KL terms for each latent space
6. **Plücker Constraints**: Orthogonality (d·m=0) and normalization (||d||=1)

### Key Features
- Disentangled latent space for image vs. geometry
- Explicit supervision at multiple points (encoder + decoder)
- Most complex variant with most parameters

---

## Variant 2: DirectPluckerVAE (Combined Latent)

### Concept
Simplified version of Variant 1. Concatenated input but with a single unified latent space (no separate mean/variance nodes for Plücker components).

### Architecture
```
Image (B, 3, H, W)  +  Plücker_GT (B, 6, H, W)
               \        /
            [Concatenate] → (B, 9, H, W)
                   |
            [Encoder (in_ch=9)]
                   |
            h (encoder features)
                   |
             [quant_conv]
                   |
              z ~ N(μ,σ)
            (unified latent)
                   |
             [Decoder]
                   |
           [Multi-Head Output]
              /         \
        Recon_img    Recon_plücker
       (B, 3, H, W)  (B, 6, H, W)
```

### Losses
1. **MSE Image Reconstruction**: `MSE(recon_img, input_img)`
2. **MSE Plücker Reconstruction**: `MSE(recon_pluck, gt_pluck)`
3. **KL Divergence**: Single KL term
4. **Plücker Constraints**: Orthogonality and normalization on decoder output

### Key Features
- Simpler than Variant 1
- Entangled latent space (image + geometry mixed)
- Fewer parameters, potentially faster training

---

## Variant 3: PluckerConditionedVAE (Conditioning Only)

### Concept
Plücker rays are provided as conditioning input only. The encoder does NOT predict Plücker rays. The decoder is expected to reconstruct both image and Plücker rays, but Plücker supervision is only on the decoder output.

### Architecture
```
Image (B, 3, H, W)  +  Plücker_GT (B, 6, H, W)
               \        /
            [Concatenate] → (B, 9, H, W)
                   |
            [Encoder (in_ch=9)]
                   |
            h (encoder features)
                   |
             [quant_conv]
                   |
              z ~ N(μ,σ)
                   |
             [Decoder]
                   |
           [Multi-Head Output]
              /         \
        Recon_img    Recon_plücker
       (B, 3, H, W)  (B, 6, H, W)

NO encoder Plücker prediction head
```

### Losses
1. **MSE Image Reconstruction**: `MSE(recon_img, input_img)`
2. **MSE Plücker Reconstruction**: `MSE(recon_pluck, gt_pluck)` - decoder only
3. **KL Divergence**: Single KL term
4. **Plücker Constraints**: Orthogonality and normalization

### Key Features
- Simplest variant
- Tests whether conditioning alone improves 3D awareness
- Hypothesis: if the VAE can reconstruct geometry, it has learned 3D-aware features

---

## Implementation Components

### 1. Model Classes (ldm/models/autoencoder.py)

| Variant | Class Name | Key Changes |
|---------|-----------|-------------|
| 1 | `ConcatPluckerVAE` | 3 quant_conv, 3 posteriors, multi-head decoder |
| 2 | `DirectPluckerVAE` | 9-ch input, multi-head decoder output |
| 3 | `PluckerConditionedVAE` | 9-ch input, multi-head decoder, no encoder pluck head |

### 2. Trainer Classes (src/trainer/vae_trainers.py)

| Variant | Class Name | Key Differences |
|---------|-----------|-----------------|
| 1 | `ConcatPluckerVAETrainer` | 3 KL losses, encoder pluck L1, 3 recon MSE |
| 2 | `DirectPluckerVAETrainer` | 1 KL, combined pluck recon |
| 3 | `PluckerConditionedVAETrainer` | 1 KL, decoder pluck recon only |

### 3. Data Pipeline (data_process/plucker.py, src/data/co3d_dataset.py)

**New Function:**
```python
def plucker_from_full_image(R, T, fl, pp, H, W, device):
    """Compute Plücker coordinates for every pixel."""
    pixel_grid = create_grid(H, W, device, patch_num=None)
    plucker = plucker_from_all_pixels(R, T, pixel_grid, fl, pp)
    return plucker.view(H, W, 6).permute(2, 0, 1)  # (6, H, W)
```

**Dataset Changes:**
- Add `full_resolution_plucker: bool` parameter
- Return Plücker as `(6, H, W)` tensor for concatenation

### 4. Encoder/Decoder Modifications (ldm/modules/diffusionmodules/model.py)

**Encoder:**
- Modify `in_channels` from 3 to 9 in config

**Decoder:**
- Create `MultiHeadDecoder` wrapper or modify to output multiple tensors:
```python
class MultiHeadDecoder(nn.Module):
    def __init__(self, base_config, output_heads):
        # output_heads: {"image": 3, "direction": 3, "moment": 3}
        self.decoder = Decoder(**base_config)
        self.heads = nn.ModuleDict({
            name: nn.Conv2d(ch, out_ch, 3, 1, 1)
            for name, out_ch in output_heads.items()
        })
```

---

## Config Examples

### Variant 1: config/concat_plucker_vae_co3d.yaml
```yaml
model:
  target: ldm.models.autoencoder.ConcatPluckerVAE
  params:
    embed_dim: 4
    latent_dim_img: 4
    latent_dim_d: 3
    latent_dim_m: 3
    ddconfig:
      in_channels: 9
      # ...

trainer:
  target: src.trainer.vae_trainers.ConcatPluckerVAETrainer
  params:
    img_recon_weight: 1.0
    d_recon_weight: 0.5
    m_recon_weight: 0.5
    encoder_plucker_weight: 0.3
    kl_weight_img: 1e-6
    kl_weight_d: 1e-6
    kl_weight_m: 1e-6

data:
  params:
    dataset_config:
      params:
        full_resolution_plucker: true
```

### Variant 2: config/direct_plucker_vae_co3d.yaml
```yaml
model:
  target: ldm.models.autoencoder.DirectPluckerVAE
  params:
    embed_dim: 4
    ddconfig:
      in_channels: 9

trainer:
  target: src.trainer.vae_trainers.DirectPluckerVAETrainer
  params:
    img_recon_weight: 1.0
    plucker_recon_weight: 0.5
```

### Variant 3: config/plucker_conditioned_vae_co3d.yaml
```yaml
model:
  target: ldm.models.autoencoder.PluckerConditionedVAE
  params:
    embed_dim: 4
    ddconfig:
      in_channels: 9

trainer:
  target: src.trainer.vae_trainers.PluckerConditionedVAETrainer
  params:
    img_recon_weight: 1.0
    plucker_recon_weight: 0.3
```

---

## Memory Considerations

Full-resolution Plücker at 256×256:
- Per sample: 6 × 256 × 256 × 4 bytes = **1.5 MB** (float32)
- Batch of 4: ~6 MB additional

**Mitigation Strategies:**
1. Precompute and cache Plücker coordinates
2. Use mixed precision (fp16)
3. Gradient checkpointing in encoder
4. Gradient accumulation if needed

---

## Experimental Comparison

| Aspect | Baseline | Variant 1 | Variant 2 | Variant 3 |
|--------|----------|-----------|-----------|-----------|
| Input channels | 3 | 9 | 9 | 9 |
| Latent spaces | 1 | 3 | 1 | 1 |
| Encoder pluck head | Yes | Yes | No | No |
| Decoder pluck output | No | Yes | Yes | Yes |
| Parameters | Medium | High | Medium | Medium |
| Complexity | Medium | High | Medium | Low |

---

## Research Questions

1. **Disentanglement**: Does separating latent spaces (Variant 1) lead to better geometric understanding?
2. **Conditioning vs. Prediction**: Is providing geometry as input (all variants) better than predicting it (baseline)?
3. **Reconstruction Signal**: Does decoder Plücker reconstruction improve image quality?
4. **Trade-offs**: How do the variants compare in terms of:
   - Image reconstruction quality (PSNR, SSIM, LPIPS)
   - Geometric accuracy (Plücker MSE, constraint satisfaction)
   - Multi-view consistency
   - Training stability and convergence speed

---

## Implementation Order (Recommended)

1. **Data Pipeline**: Full-resolution Plücker computation
2. **Variant 3** (PluckerConditionedVAE): Simplest, validates input pipeline
3. **Variant 2** (DirectPluckerVAE): Adds encoder Plücker, tests unified latent
4. **Variant 1** (ConcatPluckerVAE): Most complex, tests disentangled latents

---

## References

- Current PluckerAutoencoder: `ldm/models/autoencoder.py:292-602`
- Plücker computation: `data_process/plucker.py`
- Encoder/Decoder: `ldm/modules/diffusionmodules/model.py`
- Base trainer: `src/trainer/base_trainer.py`
- VAE trainers: `src/trainer/vae_trainers.py`
