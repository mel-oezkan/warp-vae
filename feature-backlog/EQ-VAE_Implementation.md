# EQ-VAE Implementation Documentation

## Overview

EQ-VAE (Equivariant Variational Autoencoder) is an extension of the standard Stable Diffusion VAE architecture that learns **equivariant representations** by applying random geometric transformations to latent codes during training. The core idea is to enforce that the VAE's latent space respects geometric transformations - if you transform the latent code, the decoded image should be transformed in the same way.

This implementation matches the reference [zelaki/eqvae](https://github.com/zelaki/eqvae/blob/master/train_eqvae/ldm/models/autoencoder.py) and improves the 3D awareness of 2D VAEs by making them robust to geometric transformations like scaling and rotation.

---

## Architecture

### Model Class: `EQVAEAutoencoder`

**Location:** [ldm/models/autoencoder.py:606](ldm/models/autoencoder.py#L606)

EQ-VAE extends the `AutoencoderKL` base class (standard Stable Diffusion VAE) with:

1. **Latent-space transformations** - Random scaling and rotation applied to encoded representations (output dimensions change, no padding/cropping)
2. **Probabilistic regularization** - Controlled application of equivariance training via `p_prior`
3. **Prior preservation** - Low-resolution and full-resolution standard training via `p_prior_s`
4. **Dual optimizer training** - Separate optimizers for autoencoder and discriminator
5. **LPIPS + Adversarial loss** - Perceptual quality via LPIPS with GAN-based training

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p_prior` | 0.5 | Probability of applying equivariance regularization (0-1) |
| `p_prior_s` | 0.25 | Probability of low-res prior preservation when NOT applying EQ-VAE |
| `anisotropic` | false | If true, sample independent x/y scales |
| `uniform_sample_scale` | true | Use discrete scale steps (s/32 for s in 8..31) |
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

Each training step follows one of three paths:

#### EQ-VAE Path (probability = p_prior)

Both scaling and rotation are always applied together. The output spatial dimensions change (no padding/cropping to maintain original size).

```
Input Image (x)
    ↓
Encode → Posterior Distribution
    ↓
Sample z ~ N(μ, σ²)
    ↓
Sample Random Scale s ∈ {8/32, 9/32, ..., 31/32}  (discrete, never 1.0)
Sample Random Rotation k ∈ {1, 2, 3}               (never identity)
    ↓
Scale latent:   z' = interpolate(z, scale_factor=s)
Rotate latent:  z' = rot90(z', k=k)
    ↓
Scale GT image: x' = interpolate(x, scale_factor=s)
Rotate GT:      x' = rot90(x', k=k)
    ↓
Decode z' → Reconstruction (variable spatial size)
    ↓
Loss = LPIPS(x', Reconstruction) + KL + Discriminator
```

#### Low-Res Prior Preservation Path (probability = (1 - p_prior) × p_prior_s)

Standard VAE training on downscaled images to maintain reconstruction quality at lower resolutions:

```
Input Image (x)
    ↓
Sample Random Scale s ∈ {8/32, ..., 31/32}
Downscale: x' = interpolate(x, scale_factor=s)
    ↓
Encode x' → Posterior → Sample z → Decode → Reconstruction
    ↓
Loss = LPIPS(x', Reconstruction) + KL + Discriminator
```

#### Full-Res Prior Preservation Path (probability = (1 - p_prior) × (1 - p_prior_s))

Standard VAE training at original resolution:

```
Input Image (x)
    ↓
Encode → Posterior → Sample z → Decode → Reconstruction
    ↓
Loss = LPIPS(x, Reconstruction) + KL + Discriminator
```

### Transformation Details

#### Scale Sampling

Scales are sampled from a **discrete set**: `{8/32, 9/32, ..., 31/32}` = `{0.25, 0.28125, ..., 0.96875}`.

- Scale `1.0` is never sampled in the EQ-VAE branch
- When `anisotropic=True`, x and y scales are sampled independently: `scale = (scale_x, scale_y)`
- Scaling uses `F.interpolate(scale_factor=scale)` — output spatial dimensions change accordingly

#### Rotation Sampling

Rotations are sampled from `k ∈ {1, 2, 3}` (90°, 180°, 270°):

- Identity rotation (`k=0`) is never sampled in the EQ-VAE branch
- Uses `torch.rot90(tensor, k=k, dims=[-1, -2])`
- Exact transformation (no interpolation artifacts)

#### Key Design: No Padding/Cropping

Unlike some equivariance implementations, the reference EQ-VAE does **not** pad or crop tensors back to the original size after scaling. The scaled latent produces a smaller/larger output, and the loss is computed between the reconstruction and the identically-scaled ground truth. This avoids artifacts from zero-padding and provides a cleaner equivariance signal.

---

## Loss Functions

### LPIPSWithDiscriminator

**Location:** [ldm/modules/losses/contperceptual.py](ldm/modules/losses/contperceptual.py)

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

**Location:** [src/trainer/vae_trainers.py:203](src/trainer/vae_trainers.py#L203)

The trainer uses **manual optimization** with dual optimizers for generator-discriminator training.

### Training Step

```python
def training_step(self, batch, batch_idx):
    opt_ae, opt_disc = self.optimizers()
    inputs = self.get_input(batch, self.image_key)

    if random() < p_prior:
        # EQ-VAE path: scale + rotate latent and GT
        scale = random.choice([s/32 for s in range(8, 32)])
        angle = random.choice([1, 2, 3])
        reconstructions, posterior, _ = model(inputs, scale=scale, angle=angle)

        # Apply same transforms to ground truth
        target = F.interpolate(inputs, scale_factor=scale)
        target = torch.rot90(target, k=angle, dims=[-1, -2])
    else:
        if random() < p_prior_s:
            # Low-res prior preservation
            scale = random.choice([s/32 for s in range(8, 32)])
            inputs = F.interpolate(inputs, scale_factor=scale)
            reconstructions, posterior, _ = model(inputs)
        else:
            # Full-res standard forward
            reconstructions, posterior, _ = model(inputs)
        target = inputs

    # Optimize autoencoder
    aeloss = loss(target, reconstructions, posterior, optimizer_idx=0)
    opt_ae.zero_grad(); backward(aeloss); opt_ae.step()

    # Optimize discriminator
    discloss = loss(target, reconstructions, posterior, optimizer_idx=1)
    opt_disc.zero_grad(); backward(discloss); opt_disc.step()
```

### Validation

During validation, EQ-VAE transformations are **disabled**. Standard forward pass is used to compute reconstruction metrics on untransformed images.

---

## Configuration

### Full Model Configuration

**File:** [config/eqvae_omniobject.yaml](config/eqvae_omniobject.yaml)

```yaml
model:
  target: ldm.models.autoencoder.EQVAEAutoencoder
  params:
    embed_dim: 4
    p_prior: 0.5             # 50% EQ-VAE, 50% prior preservation
    p_prior_s: 0.25          # 25% of prior preservation is low-res
    anisotropic: false        # Isotropic scaling
    use_rotation: true        # Enable 90° rotations
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
    p_prior: 0.5              # Match reference default

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

### Differences from Some Other EQ-VAE Implementations

Our implementation matches the reference [zelaki/eqvae](https://github.com/zelaki/eqvae):

1. **No padding/cropping:** Scaling changes output spatial dimensions. The loss is computed between variable-size reconstruction and identically-scaled ground truth.
2. **Discrete scale sampling:** Scales come from `{s/32 : s ∈ {8, ..., 31}}`, never `1.0`.
3. **Non-identity rotation only:** `k ∈ {1, 2, 3}`, never `0`.
4. **Coupled transforms:** Both scale AND rotation are always applied together in the EQ-VAE branch.
5. **Prior preservation:** Two-level fallback when not doing EQ-VAE — low-res with probability `p_prior_s`, or full-res otherwise.
6. **Anisotropic scaling:** Optional independent x/y scale factors.

### Memory Optimization

1. **LPIPS Frozen:** Perceptual loss network parameters are frozen (no gradients)
2. **Gradient Computation:** Uses `create_graph=False` in adaptive weight calculation

### Numerical Stability

1. **Log Variance in FP32:** Kept in full precision for stability
2. **Gradient Clipping:** Via adaptive weighting mechanism
3. **Delayed Discriminator:** Starts after 50k steps for stable early training

### Why Probabilistic (p_prior)?

Using `p_prior < 1.0` ensures:
1. The model doesn't overfit to transformed representations
2. Standard reconstruction capability is maintained
3. Training signal variety (both transformed and untransformed)

Default `p_prior = 0.5` means:
- 50% of training uses EQ-VAE transformations (scale + rotation)
- 50% uses prior preservation (low-res or full-res standard VAE)

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
    model.params.p_prior=0.5 \
    model.params.p_prior_s=0.25 \
    model.params.anisotropic=true \
    training.num_epochs=50
```

---

## Summary

EQ-VAE improves upon standard VAE training by:

1. **Applying random geometric transformations** (scaling + rotation) to latent codes during training, with output dimensions changing accordingly
2. **Training against identically-transformed ground truth** to learn equivariant representations
3. **Probabilistically mixing** three training modes: EQ-VAE, low-res prior preservation, and full-res standard VAE
4. **Using perceptual + adversarial losses** for high-quality outputs
5. **Supporting anisotropic scaling** for richer transformation variety

The result is a VAE with better geometric awareness that generalizes better to transformed inputs and maintains consistent representations across scales and rotations.
