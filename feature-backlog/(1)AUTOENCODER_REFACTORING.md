# Autoencoder Refactoring Summary

## Overview

The autoencoder implementation has been refactored to separate concerns between the base VAE functionality and Plucker-specific extensions. This makes the code more modular, maintainable, and follows the original Stable Diffusion architecture more closely.

## Changes Made

### 1. Base AutoencoderKL Class (Clean VAE)

**Location**: `ldm/models/autoencoder.py` (lines 13-289)

This is the standard Variational Autoencoder implementation from Stable Diffusion, **without any Plucker-specific modifications**.

**Features**:
- Standard encoder-decoder architecture
- KL divergence regularization
- EMA (Exponential Moving Average) support
- Discriminator-based adversarial training
- Compatible with original Stable Diffusion checkpoints

**Methods**:
- `encode(x)` → Returns `posterior` (DiagonalGaussianDistribution)
- `decode(z)` → Returns reconstructed image
- `forward(input, sample_posterior)` → Returns `(reconstruction, posterior)`

**Use case**: When you need a standard VAE without camera-aware features.

### 2. PluckerAutoencoder Class (Extended VAE)

**Location**: `ldm/models/autoencoder.py` (lines 292-602)

This class **extends AutoencoderKL** and adds Plucker coordinate prediction for camera-aware multi-view learning.

**Additional Features**:
- Plucker ray prediction head (Conv2d)
- MLP refinement network with LayerNorm and Dropout
- Hybrid Plucker loss (reconstruction + constraint + normalization)
- Camera-aware representation learning

**Methods**:
- `encode(x)` → Returns `(posterior, plucker_coords)`
- `forward(input, sample_posterior)` → Returns `(reconstruction, posterior, plucker_coords)`
- `hybrid_plucker_loss(pred, gt)` → Computes Plucker loss with geometric constraints

**Plucker Loss Components**:
1. **Reconstruction**: MSE between predicted and ground truth
2. **Constraint**: Enforces d·m = 0 (Plucker orthogonality)
3. **Normalization**: Encourages unit direction vectors (||d|| = 1)

**Use case**: When training with multi-view data and camera parameters (CO3D, OmniObject, etc.)

## Configuration Updates

### vae_config.yaml

**Before**:
```yaml
model:
    target: ldm.models.autoencoder.AutoencoderKL
    params:
        embed_dim: 4
        n_patches: 8
        plucker_key: "pluck"
```

**After**:
```yaml
model:
    target: ldm.models.autoencoder.PluckerAutoencoder
    params:
        embed_dim: 4
        n_patches: 8
        plucker_key: "pluck_ray"  # Updated key name
```

## Class Hierarchy

```
AutoencoderKL (Base VAE)
    ├── Standard encoder-decoder
    ├── KL divergence
    ├── EMA support
    └── Discriminator training

PluckerAutoencoder (extends AutoencoderKL)
    ├── All base VAE features
    ├── + Plucker prediction head
    ├── + Plucker MLP refinement
    └── + Hybrid Plucker loss
```

## Usage Examples

### Using Base AutoencoderKL

```python
from ldm.models.autoencoder import AutoencoderKL

# Create standard VAE
model = AutoencoderKL(
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    embed_dim=4,
)

# Forward pass
reconstruction, posterior = model(image)

# Encode only
posterior = model.encode(image)
z = posterior.sample()

# Decode only
reconstruction = model.decode(z)
```

### Using PluckerAutoencoder

```python
from ldm.models.autoencoder import PluckerAutoencoder

# Create Plucker-aware VAE
model = PluckerAutoencoder(
    ddconfig=ddconfig,
    lossconfig=lossconfig,
    embed_dim=4,
    n_patches=8,
    plucker_key="pluck_ray",
    plucker_weights={"recon": 1.0, "constraint": 0.1, "norm": 0.1}
)

# Forward pass (returns Plucker coordinates too)
reconstruction, posterior, plucker_rays = model(image)

# Encode with Plucker
posterior, plucker_rays = model.encode(image)

# Compute Plucker loss
gt_plucker = batch["pluck_ray"]
loss = model.hybrid_plucker_loss(plucker_rays, gt_plucker)
```

## Key Parameters

### PluckerAutoencoder Additional Parameters

- `n_patches` (int): Number of patches per dimension (e.g., 8 for 8×8 grid = 64 rays)
- `plucker_key` (str): Key for Plucker coordinates in batch dict (default: "pluck_ray")
- `plucker_hidden_dim` (int): Hidden dimension for MLP (default: 512)
- `plucker_dropout` (float): Dropout rate for MLP (default: 0.1)
- `plucker_weights` (dict): Loss weights with keys:
  - `"recon"`: Reconstruction term weight (default: 1.0)
  - `"constraint"`: Constraint term weight (default: 0.1)
  - `"norm"`: Normalization term weight (default: 0.1)

## Verification

Run the verification script to ensure everything works:

```bash
python test_autoencoder_refactor.py
```

**Expected output**: All tests should pass with ✅

## Benefits of Refactoring

1. **Modularity**: Clean separation between base VAE and Plucker extensions
2. **Reusability**: Base AutoencoderKL can be used for standard VAE tasks
3. **Maintainability**: Easier to understand and modify each component
4. **Compatibility**: Base class follows original Stable Diffusion design
5. **Extensibility**: Easy to add other extensions by subclassing AutoencoderKL

## Migration Guide

### For Existing Code

If you're using the old AutoencoderKL with Plucker features:

1. **Update config**: Change `target: ldm.models.autoencoder.AutoencoderKL` to `target: ldm.models.autoencoder.PluckerAutoencoder`
2. **Update plucker_key**: Change `"pluck"` to `"pluck_ray"` in config
3. **No code changes needed**: The API is backward compatible

### For New Projects

- Use `AutoencoderKL` for standard VAE tasks
- Use `PluckerAutoencoder` for multi-view learning with camera awareness

## Files Modified

1. **ldm/models/autoencoder.py**:
   - Refactored `AutoencoderKL` to be clean base class
   - Created new `PluckerAutoencoder` class extending `AutoencoderKL`
   - Moved `hybrid_plucker_loss` from train.py to `PluckerAutoencoder`

2. **vae_config.yaml**:
   - Updated target to `PluckerAutoencoder`
   - Updated plucker_key to "pluck_ray"

3. **test_autoencoder_refactor.py** (new):
   - Verification script for refactoring

## References

- **Original EQ-VAE**: https://github.com/zelaki/eqvae
- **Stable Diffusion**: https://github.com/CompVis/stable-diffusion
- **Plucker Coordinates**: Used for geometric ray representation in 3D vision

## Notes

- The base `AutoencoderKL` is now compatible with standard Stable Diffusion workflows
- `PluckerAutoencoder` maintains all functionality from the previous implementation
- All existing checkpoints should work with `PluckerAutoencoder` (with `strict=False` for missing Plucker weights)
