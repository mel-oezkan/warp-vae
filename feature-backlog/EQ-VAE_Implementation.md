# EQ-VAE Implementation Documentation

## Overview

EQ-VAE (Equivariant Variational Autoencoder) is an extension of the standard Stable Diffusion VAE architecture that learns **equivariant representations** by applying random geometric transformations to latent codes during training. The core idea is to enforce that the VAE's latent space respects geometric transformations - if you transform the latent code, the decoded image should be transformed in the same way.

This implementation aims to improve the 3D awareness of 2D VAEs by making them robust to geometric transformations like scaling and rotation.

---

## Architecture

### Model Class: `EQVAEAutoencoder`

**Location:** [ldm/models/autoencoder.py:605-912](ldm/models/autoencoder.py#L605-L912)

EQ-VAE extends the `AutoencoderKL` base class (standard Stable Diffusion VAE) with:

1. **Latent-space transformations** - Random scaling and rotation applied to encoded representations
2. **Probabilistic regularization** - Controlled application of equivariance training via `p_prior`
3. **Dual optimizer training** - Separate optimizers for autoencoder and discriminator
4. **LPIPS + Adversarial loss** - Perceptual quality via LPIPS with GAN-based training

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p_prior` | 0.9 | Probability of applying equivariance regularization (0-1) |
| `scale_range` | [0.25, 1.0] | Isotropic scaling range for latent transformations |
| `use_rotation` | true | Enable 90-degree rotation transformations |
| `equivariance_weight` | 1.0 | Weight for equivariance loss component |
| `embed_dim` | 4 | Latent embedding dimension |

### Encoder-Decoder Configuration

The model uses a U-Net-style encoder-decoder with configurable depth:

```yaml
ddconfig:
  double_z: true          # Output mean + variance
  z_channels: 4           # Latent channels
  resolution: 256         # Input resolution
  in_channels: 3          # RGB input
  out_ch: 3               # RGB output
  ch: 128                 # Base channel count
  ch_mult: [1, 2, 4, 4]   # Channel multipliers per level
  num_res_blocks: 2       # Residual blocks per level
  attn_resolutions: []    # Attention at these resolutions
  dropout: 0.0
```

---

## How EQ-VAE Works

### Core Mechanism: Equivariance Regularization

The fundamental insight of EQ-VAE is that a well-structured latent space should be **equivariant** to geometric transformations. This means:

```
decode(transform(encode(x))) ≈ transform(x)
```

If you encode an image, transform the latent code, and decode it, the result should be the same as if you had transformed the original image.

### Training Flow

#### Standard VAE Path (probability = 1 - p_prior)

```
Input Image (x)
    ↓
Encode → Posterior Distribution
    ↓
Sample z ~ N(μ, σ²)
    ↓
Decode z → Reconstruction
    ↓
Loss = ||Reconstruction - x||
```

#### EQ-VAE Path (probability = p_prior)

```
Input Image (x)
    ↓
Encode → Posterior Distribution
    ↓
Sample z ~ N(μ, σ²)
    ↓
Sample Random Transform T (scale + rotation)
    ↓
Apply T to latent: z' = T(z)
Apply T to image: x' = T(x)
    ↓
Decode z' → Reconstruction
    ↓
Loss = ||Reconstruction - x'||  (compare to TRANSFORMED target)
```

The critical insight is that **the same transformation is applied to both the latent code and the input image**. This provides a valid learning signal - the model learns that transforming the latent should produce a correspondingly transformed output.

### Transformation Implementation

#### Transformation Sampling

**Method:** `_sample_transformation()` ([autoencoder.py:680-698](ldm/models/autoencoder.py#L680-L698))

```python
def _sample_transformation(self):
    # Scale: uniform in [0.25, 1.0]
    scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()

    # Rotation: one of {0, 90, 180, 270} degrees
    rotation = torch.randint(0, 4, (1,)).item() if self.use_rotation else 0

    return {'scale': scale, 'rotation': rotation}
```

#### Latent Transformation

**Method:** `_transform_latent()` ([autoencoder.py:700-742](ldm/models/autoencoder.py#L700-L742))

**Scaling:**
1. Interpolate latent feature map to new size: `new_size = H * scale`
2. If scaled down (scale < 1): Zero-pad to maintain original dimensions
3. If scaled up (scale > 1): Center-crop back to original dimensions
4. Shape is preserved: `[B, C, H, W] → [B, C, H, W]`

**Rotation:**
- Uses `torch.rot90()` with k ∈ {0, 1, 2, 3}
- k=1 rotates 90° counterclockwise
- Exact transformations (no interpolation artifacts)

#### Image Transformation

**Method:** `_transform_image()` ([autoencoder.py:744-785](ldm/models/autoencoder.py#L744-L785))

Applies the **identical** transformation to the input image to generate the training target. This is critical - mismatched transformations would provide invalid learning signals.

---

## Loss Functions

### LPIPSWithDiscriminator

**Location:** [ldm/modules/losses/contperceptual.py:7-180](ldm/modules/losses/contperceptual.py#L7-L180)

The loss module combines multiple components:

#### 1. Reconstruction Loss

```python
rec_loss = L1(inputs, reconstructions) + perceptual_weight * LPIPS(inputs, reconstructions)
```

- **L1 Loss:** Pixel-level reconstruction
- **LPIPS Loss:** Perceptual similarity using pretrained VGG features (frozen, no gradients)

#### 2. KL Divergence Loss

```python
kl_loss = KL(posterior || N(0, I))
```

- Weight: `0.000001` (minimal - nearly negligible)
- Encourages posterior to stay close to standard normal prior
- Very small to prioritize reconstruction quality

#### 3. Adversarial Loss (Discriminator)

```python
g_loss = -mean(discriminator(reconstructions))  # Generator wants fake to look real
d_loss = hinge_loss(real, fake)                  # Discriminator classifies
```

- Starts after `disc_start` steps (default: 50,001)
- Uses hinge loss or vanilla GAN loss
- Weight: 0.5

#### 4. Adaptive Weighting

The discriminator loss weight is computed adaptively based on gradient magnitudes:

```python
d_weight = ||∇(nll_loss)|| / ||∇(g_loss)||
```

This balances reconstruction and adversarial components automatically.

#### Combined Loss

```python
total_loss = nll_loss + kl_weight * kl_loss + d_weight * disc_factor * g_loss
```

---

## Training Process

### Trainer: `EQVAETrainer`

**Location:** [src/trainer/vae_trainers.py:201-377](src/trainer/vae_trainers.py#L201-L377)

The trainer uses **manual optimization** with dual optimizers for generator-discriminator training.

### Training Step

```python
def training_step(self, batch, batch_idx):
    # 1. Get optimizers
    opt_ae, opt_disc = self.optimizers()

    # 2. Decide: use EQ-VAE or standard VAE?
    use_eqvae = random() < p_prior

    # 3. Forward pass
    if use_eqvae:
        reconstructions, posterior, transformed_target = model._eqvae_forward(inputs)
        target = transformed_target
    else:
        reconstructions, posterior = model(inputs)
        target = inputs

    # 4. Optimize autoencoder
    aeloss = loss(target, reconstructions, posterior, optimizer_idx=0)
    opt_ae.zero_grad()
    backward(aeloss)
    opt_ae.step()

    # 5. Optimize discriminator
    discloss = loss(target, reconstructions, posterior, optimizer_idx=1)
    opt_disc.zero_grad()
    backward(discloss)
    opt_disc.step()
```

### Validation

During validation, EQ-VAE transformations are **disabled**. Standard forward pass is used to compute reconstruction metrics on untransformed images.

---

## Configuration

### Full Model Configuration

**File:** [config/eqvae_omniobject.yaml](config/eqvae_omniobject.yaml)

```yaml
model:
  base_learning_rate: 4.5e-6
  target: ldm.models.autoencoder.EQVAEAutoencoder
  params:
    embed_dim: 4
    p_prior: 0.9              # 90% EQ-VAE, 10% standard VAE
    scale_range: [0.25, 1.0]  # Scale down to 25% to 100%
    use_rotation: true         # Enable 90° rotations
    equivariance_weight: 1.0

    ddconfig:
      ch: 128
      ch_mult: [1, 2, 4, 4]
      resolution: 256

    lossconfig:
      target: ldm.modules.losses.LPIPSWithDiscriminator
      params:
        disc_start: 50001     # Start discriminator after 50k steps
        kl_weight: 0.000001
        disc_weight: 0.5
        perceptual_weight: 1.0
```

### Small Model Configuration (Memory-Efficient)

**File:** [config/eqvae_omniobject_small.yaml](config/eqvae_omniobject_small.yaml)

For testing on limited GPU memory (e.g., 11GB):

```yaml
model:
  params:
    p_prior: 0.5              # Less aggressive equivariance

    ddconfig:
      ch: 64                  # Halved channel count
      ch_mult: [1, 2, 4]      # 3 levels instead of 4
      resolution: 128         # Reduced resolution
```

Memory usage:
- Full model: ~10GB GPU memory
- Small model: ~2.6GB GPU memory

---

## Dataset: OmniObject3D

### Overview

OmniObject3D is a large-scale 3D object dataset with rendered multi-view images. It provides 24 views per object with camera parameters.

**Location:** [src/data/omniobject3d_dataset.py](src/data/omniobject3d_dataset.py)

### Data Structure

```
omniobject/
└── img/
    ├── object_001/
    │   ├── 000.png
    │   ├── 001.png
    │   ├── ...
    │   ├── 023.png
    │   └── transforms.json
    ├── object_002/
    └── ...
```

Each `transforms.json` contains camera parameters:
- Rotation matrix (R)
- Translation vector (T)
- Field of view (for computing focal length)

### Sample Modes

#### Single View Mode (Default for EQ-VAE)

```yaml
sample_mode: single
```

Returns one view per sample:
- `image`: tensor (C, H, W)
- `camera`: dict with R, T, focal_length
- `index`: sample index

#### Paired View Mode (For Multi-View Consistency)

```yaml
sample_mode: pairs
```

Returns two views of the same object:
- `image`, `image2`: both views
- `camera`, `camera2`: parameters for both
- `R_rel`, `T_rel`: relative camera transformation

### View Pairing Strategies

| Strategy | Description |
|----------|-------------|
| `sequential` | Consecutive views: (0,1), (1,2), ..., (23,0) |
| `random` | Random pairs (different views of same object) |
| `fixed_interval` | Opposite views: (0,12), (1,13), ... |

### Plucker Coordinates (Optional)

The dataset can compute Plucker coordinates - a 6D representation of camera rays useful for 3D-aware models:

```yaml
include_plucker: true
n_patches: 8  # 8x8 grid of ray samples
```

For EQ-VAE training, Plucker coordinates are typically **disabled** since the model uses synthetic 2D transformations rather than real camera movements.

---

## Key Implementation Details

### Memory Optimization

1. **LPIPS Frozen:** Perceptual loss network parameters are frozen (no gradients)
2. **Gradient Computation:** Uses `create_graph=False` in adaptive weight calculation
3. **Immediate Cleanup:** Latent tensors are deleted immediately after use

### Numerical Stability

1. **Log Variance in FP32:** Kept in full precision for stability
2. **Gradient Clipping:** Via adaptive weighting mechanism
3. **Delayed Discriminator:** Starts after 50k steps for stable early training

### Why Probabilistic (p_prior)?

Using `p_prior < 1.0` ensures:
1. The model doesn't overfit to transformed representations
2. Standard reconstruction capability is maintained
3. Training signal variety (both transformed and untransformed)

Default `p_prior = 0.9` means:
- 90% of training uses EQ-VAE transformations
- 10% uses standard VAE training

---

## Training Commands

### Full Model

```bash
python train.py --config-name=eqvae_omniobject
```

### Small Model (Testing/Debugging)

```bash
python train.py --config-name=eqvae_omniobject_small
```

### Custom Configuration

```bash
python train.py --config-name=eqvae_omniobject \
    model.params.p_prior=0.85 \
    model.params.scale_range="[0.5, 1.0]" \
    training.num_epochs=50
```

---

## Future Improvements

See [eqvae-future-features.md](feature-backlog/(2)eqvae-future-features.md) for planned enhancements:

1. **Real Camera Transformations** - Use actual camera poses from OmniObject instead of synthetic 2D transforms
2. **Prior Preservation Mode** - Additional `p_prior_s` for random downscaling only
3. **Anisotropic Scaling** - Different scale factors for x and y
4. **Arbitrary Angle Rotations** - Continuous rotation angles
5. **Learning Rate Scheduling** - Cosine annealing for better convergence

---

## Summary

EQ-VAE improves upon standard VAE training by:

1. **Applying random geometric transformations** to latent codes during training
2. **Training against transformed targets** to learn equivariant representations
3. **Probabilistically mixing** equivariance training with standard reconstruction
4. **Using perceptual + adversarial losses** for high-quality outputs

The result is a VAE with better geometric awareness that should generalize better to transformed inputs and maintain consistent representations across scales and rotations.
