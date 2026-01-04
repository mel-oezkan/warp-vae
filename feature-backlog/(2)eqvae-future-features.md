# EQ-VAE Feature Backlog

This document tracks potential future enhancements and features for the EQ-VAE implementation. These are deliberately excluded from the initial implementation to keep it simple and focused on core functionality.

## Current Implementation Status (v1.0)

### ✅ Implemented Features
- **Basic EQ-VAE Pipeline**: Core equivariance training with latent-space transformations
- **Isotropic Scaling**: Random scaling in range [0.25, 1.0] applied uniformly in x and y dimensions
- **90-Degree Rotations**: Exact rotations at 0°, 90°, 180°, 270° using `torch.rot90()`
- **Probabilistic Regularization**: p_prior parameter (default 0.9) controls EQ-VAE vs standard VAE training ratio
- **LPIPS + Discriminator Loss**: Perceptual quality via LPIPS, adversarial training with discriminator starting at step 50k
- **Dual Optimizers**: Separate Adam optimizers for autoencoder and discriminator
- **OmniObject Dataset Integration**: Uses existing OmniObjectDataModule (view pairs available but only first view used)
- **Validation Tools**: Unit tests and equivariance validation script with visualizations

---

## Future Features & Enhancements

### 1. Anisotropic Scaling Transformations

**Priority**: Medium
**Effort**: Low (2-3 hours)
**Status**: Not Implemented

**Description**:
Extend scaling transformations to allow different factors for x and y dimensions, enabling more diverse augmentations that test equivariance under non-uniform scaling.

**Implementation Details**:
```python
def _sample_transformation(self):
    if self.anisotropic:
        scale_x = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        scale_y = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        scale = (scale_x, scale_y)
    else:
        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()

    return {'scale': scale, 'rotation': rotation}

def _transform_latent(self, z, transform_params):
    if isinstance(transform_params['scale'], tuple):
        # Anisotropic scaling
        scale_x, scale_y = transform_params['scale']
        new_h = int(H * scale_y)
        new_w = int(W * scale_x)
        z_out = F.interpolate(z_out, size=(new_h, new_w), ...)
        # Pad/crop to original size...
```

**Configuration**:
```yaml
model:
  params:
    anisotropic: true
    scale_range: [0.25, 1.0]  # Applied independently to x and y
```

**Benefits**:
- More comprehensive equivariance testing
- Better handling of aspect ratio variations
- Matches original EQ-VAE implementation more closely

**Risks**:
- May introduce more interpolation artifacts
- Could make training less stable initially

---

### 2. Arbitrary Angle Rotations

**Priority**: Low
**Effort**: Medium (4-6 hours)
**Status**: Not Implemented

**Description**:
Support continuous rotation angles instead of just 90-degree multiples, using `torch.nn.functional.affine_grid` and `grid_sample`.

**Implementation Details**:
```python
def _sample_transformation(self):
    if self.use_arbitrary_rotation:
        # Sample angle in radians
        rotation = torch.empty(1).uniform_(0, 2 * math.pi).item()
    else:
        # 90-degree multiples
        rotation = torch.randint(0, 4, (1,)).item()

    return {'scale': scale, 'rotation': rotation}

def _transform_latent_continuous_rotation(self, z, angle_radians):
    B, C, H, W = z.shape

    # Create rotation matrix
    theta = torch.zeros(B, 2, 3)
    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a

    # Apply rotation
    grid = F.affine_grid(theta, z.size(), align_corners=False)
    z_rotated = F.grid_sample(z, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    return z_rotated
```

**Configuration**:
```yaml
model:
  params:
    use_arbitrary_rotation: true
    rotation_range: [0, 360]  # Degrees
```

**Benefits**:
- More general equivariance property
- Better coverage of SO(2) rotation group
- Useful for datasets with arbitrary viewpoint changes

**Risks**:
- Interpolation artifacts from bilinear sampling
- Harder to verify equivariance (no exact 360° = identity)
- May require more careful initialization

---

### 3. Real Camera Transformations from OmniObject

**Priority**: High
**Effort**: High (1-2 days)
**Status**: Not Implemented (Deferred to iteration 2)

**Description**:
Use actual camera transformation matrices from OmniObject dataset instead of synthetic scaling/rotation. Apply relative pose transformations between view pairs to learn true 3D equivariance.

**Implementation Details**:
```python
def _eqvae_forward_with_camera_transform(self, x, batch):
    # Encode first view
    posterior = self.encode(x)
    z = posterior.sample()

    # Get camera parameters from batch
    R_rel = batch['R_rel']  # [B, 3, 3]
    T_rel = batch['T_rel']  # [B, 3]

    # Apply 3D transformation to latent (requires learning transform mapping)
    z_transformed = self.transform_latent_3d(z, R_rel, T_rel)

    # Get second view from batch
    x2 = batch['image2']

    # Decode transformed latent
    dec = self.decode(z_transformed)

    # Compare with actual second view
    return dec, posterior, x2  # Use real second view as target
```

**Required Components**:
1. **Learnable 3D Transform Module**: Map (R, T) to latent space transformations
2. **View Pair Training**: Use both views from OmniObject dataset
3. **Multi-View Consistency Loss**: Ensure decoded latent matches actual second view
4. **Pose Encoder**: Optional network to predict pose from latent differences

**Configuration**:
```yaml
data:
  use_view_pairs: true
  apply_real_transforms: true

model:
  params:
    use_camera_transforms: true
    transform_module:
      type: "learnable_3d"  # or "direct_mapping"
      hidden_dim: 256
      num_layers: 3
```

**Benefits**:
- **Real-world equivariance**: Learns from actual 3D geometry
- **Multi-view consistency**: Leverages paired views
- **3D understanding**: Model learns camera-aware representations
- **Better generalization**: Trained on real camera movements

**Challenges**:
- **Complexity**: Requires learning transformation mapping
- **Latent space structure**: Need compatible latent representation for 3D transforms
- **Training stability**: More complex optimization landscape
- **Compute cost**: Processing view pairs doubles data loading

**Phased Approach**:
1. **Phase 1**: Simple learned affine mapping from (R, T) → latent transform
2. **Phase 2**: Add multi-view consistency loss
3. **Phase 3**: Full 3D equivariant architecture with pose encoder

---

### 4. Prior Preservation Mode

**Priority**: Medium
**Effort**: Low (1-2 hours)
**Status**: Not Implemented

**Description**:
Implement `p_prior_s` parameter from original EQ-VAE for random downscaling with a separate probability, helping preserve prior distribution during equivariance training.

**Implementation Details**:
```python
def training_step(self, batch, batch_idx, optimizer_idx=0):
    inputs = self.get_input(batch, self.image_key)

    # Decide mode
    rand = torch.rand(1).item()

    if rand < self.p_prior:
        # Standard equivariance mode
        reconstructions, posterior, transformed_inputs = self._eqvae_forward(inputs)
        target = transformed_inputs
    elif rand < (self.p_prior + self.p_prior_s):
        # Prior preservation mode: random downscaling only
        scale = torch.empty(1).uniform_(0.5, 1.0).item()
        transform_params = {'scale': scale, 'rotation': 0}
        # Apply to latent only
        posterior = self.encode(inputs)
        z = posterior.sample()
        z_scaled = self._transform_latent(z, transform_params)
        reconstructions = self.decode(z_scaled)
        target = inputs  # Compare with original
    else:
        # Standard VAE mode
        reconstructions, posterior = self(inputs)
        target = inputs

    # Compute loss...
```

**Configuration**:
```yaml
model:
  params:
    p_prior: 0.8       # Equivariance probability
    p_prior_s: 0.1     # Prior preservation probability
    # Remaining 0.1: standard VAE
```

**Benefits**:
- Better latent distribution regularization
- Prevents mode collapse
- Balances equivariance and reconstruction quality

---

### 5. Gradient-Based Adaptive Loss Weighting

**Priority**: Low
**Effort**: Medium (3-4 hours)
**Status**: Not Implemented

**Description**:
Implement adaptive weighting of discriminator loss based on gradient magnitudes, as in original EQ-VAE and Stable Diffusion.

**Implementation Details**:
```python
def training_step(self, batch, batch_idx, optimizer_idx=0):
    if optimizer_idx == 0:  # Autoencoder
        # ... forward pass ...

        # Compute losses
        nll_loss = F.mse_loss(reconstructions, target)

        # Get last layer for gradient calculation
        last_layer = self.get_last_layer()

        # Calculate discriminator loss
        g_loss = -torch.mean(self.loss.discriminator(reconstructions))

        # Adaptive weighting
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        # Combined loss
        loss = nll_loss + d_weight * g_loss + kl_loss
```

**Benefits**:
- Automatic balancing of loss components
- More stable GAN training
- Prevents discriminator from dominating

**Risks**:
- Additional computational cost (gradient calculations)
- May require tuning clamp values

---

### 6. Exponential Moving Average (EMA) for Inference

**Priority**: Medium
**Effort**: Low (implemented in base class, just needs configuration)
**Status**: Partially implemented (base class has support)

**Description**:
Enable EMA of model weights for better inference quality, already supported by base `AutoencoderKL` class.

**Configuration**:
```yaml
model:
  params:
    ema_decay: 0.9999  # Currently set to null in base implementation
```

**Implementation**:
No code changes needed, just enable in config. EMA is already handled by base class via `LitEma` module.

**Benefits**:
- Smoother, higher-quality reconstructions
- More stable inference
- Standard practice for diffusion models

---

### 7. Flip and Reflection Augmentations

**Priority**: Low
**Effort**: Low (2-3 hours)
**Status**: Not Implemented

**Description**:
Add horizontal/vertical flips as additional transformation types, as implemented in original EQ-VAE's `flip_or_rotate_image()` utility.

**Implementation Details**:
```python
def _sample_transformation(self):
    scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
    rotation = torch.randint(0, 4, (1,)).item() if self.use_rotation else 0

    # Add flips
    flip_h = torch.rand(1).item() < 0.5 if self.use_flips else False
    flip_v = torch.rand(1).item() < 0.5 if self.use_flips else False

    return {
        'scale': scale,
        'rotation': rotation,
        'flip_h': flip_h,
        'flip_v': flip_v
    }

def _transform_latent(self, z, transform_params):
    # ... existing scale + rotation ...

    if transform_params.get('flip_h', False):
        z_out = torch.flip(z_out, dims=[3])  # Flip width
    if transform_params.get('flip_v', False):
        z_out = torch.flip(z_out, dims=[2])  # Flip height

    return z_out
```

**Configuration**:
```yaml
model:
  params:
    use_flips: true
    flip_probability: 0.5
```

**Benefits**:
- More augmentation diversity
- Tests different symmetries
- Useful for symmetric objects in OmniObject

---

### 8. Mixed Precision Training

**Priority**: Medium
**Effort**: Low (configuration only)
**Status**: Not Implemented

**Description**:
Enable automatic mixed precision (AMP) training for faster training and reduced memory usage.

**Configuration**:
```yaml
training:
  precision: 16  # Already in config, may need trainer flag

# In train.py trainer instantiation:
trainer = Trainer(
    precision='16-mixed',  # or just 16
    ...
)
```

**Benefits**:
- ~2x faster training
- Reduced memory usage (can increase batch size)
- Minimal quality impact with proper loss scaling

**Risks**:
- May require gradient scaling for stability
- Discriminator training can be sensitive to precision

---

### 9. Learning Rate Scheduling

**Priority**: Medium
**Effort**: Low (2-3 hours)
**Status**: Not Implemented (returns empty scheduler list)

**Description**:
Add learning rate scheduling for better convergence, especially useful for longer training runs.

**Implementation Details**:
```python
def configure_optimizers(self):
    lr = self.learning_rate

    opt_ae = torch.optim.Adam(...)
    opt_disc = torch.optim.Adam(...)

    # Add schedulers
    scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_ae, T_max=100000, eta_min=lr * 0.1
    )
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_disc, T_max=100000, eta_min=lr * 0.1
    )

    return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]
```

**Configuration Options**:
- **Cosine Annealing**: Smooth decay with restarts
- **Step Decay**: Drop LR at specific epochs
- **Warmup**: Linear warmup for first N steps
- **Reduce on Plateau**: Adaptive based on validation loss

**Benefits**:
- Better final convergence
- Escape local minima
- Standard practice for long training

---

### 10. Multi-Scale Discriminator

**Priority**: Low
**Effort**: Medium (4-6 hours)
**Status**: Not Implemented (uses single-scale discriminator)

**Description**:
Replace single discriminator with multi-scale architecture operating at different resolutions for better perceptual quality.

**Implementation**:
Modify loss config to use multi-scale discriminator similar to pix2pixHD.

**Benefits**:
- Better detail preservation at multiple scales
- More stable GAN training
- Higher visual quality

**Effort**:
Requires implementing new discriminator architecture or importing from existing codebase.

---

### 11. Latent Code Visualization and Analysis

**Priority**: Low
**Effort**: Medium (half day)
**Status**: Not Implemented

**Description**:
Tools to visualize and analyze learned latent representations, especially how transformations affect latent codes.

**Features**:
- PCA/t-SNE visualization of latent codes
- Latent space interpolation
- Transformation vector visualization
- Disentanglement metrics

**Benefits**:
- Better understanding of learned representations
- Debugging equivariance issues
- Publication-quality figures

---

## Implementation Priority Ranking

### High Priority (After Initial Training)
1. **Real Camera Transformations** - Core research goal
2. **EMA for Inference** - Easy win for quality

### Medium Priority (Quality Improvements)
3. **Prior Preservation Mode** - Better training stability
4. **Learning Rate Scheduling** - Better convergence
5. **Mixed Precision Training** - Speed + memory
6. **Anisotropic Scaling** - More comprehensive equivariance

### Low Priority (Nice to Have)
7. **Gradient-Based Loss Weighting** - Training stability
8. **Flip Augmentations** - More diversity
9. **Arbitrary Rotations** - Research exploration
10. **Multi-Scale Discriminator** - Quality improvement
11. **Latent Visualization** - Analysis tools

---

## Version Roadmap

### v1.0 (Current - Baseline)
- Basic EQ-VAE with isotropic scaling + 90° rotations
- LPIPS + single-scale discriminator
- OmniObject dataset (single view)

### v1.1 (Quick Wins)
- Enable EMA for inference
- Add learning rate scheduling
- Implement mixed precision training

### v1.2 (Stability Improvements)
- Prior preservation mode (p_prior_s)
- Gradient-based loss weighting
- Anisotropic scaling

### v2.0 (Major Feature: Real 3D Transforms)
- Real camera transformations from OmniObject
- View pair training
- Multi-view consistency loss
- Learnable 3D transform module

### v2.1 (Advanced Features)
- Multi-scale discriminator
- Arbitrary angle rotations
- Flip augmentations

### v3.0 (Research Extensions)
- Latent space analysis tools
- Disentanglement metrics
- Integration with downstream tasks

---

## Notes

- Each feature should be implemented on a separate git branch
- All features should include corresponding tests
- Configuration should be backward compatible
- Document performance impact (speed, memory, quality) for each feature

**Last Updated**: 2026-01-04
**Maintainer**: Documented during initial EQ-VAE implementation
