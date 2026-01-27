# FP16 NaN Loss Fix Summary

This document summarizes the debugging and fixes applied to resolve NaN losses during Warp VAE training with FP16 mixed precision.

## Problem

Training with `precision: 16` produced NaN losses after ~8 epochs. The `train/aeloss` metric would suddenly become NaN and training would fail.

## Root Causes Identified

### 1. Config Parameters Not Passed to Trainer

**File:** `train.py`

The `setup_trainer_module()` function was not passing `cfg.trainer.params` to the trainer class. This meant all trainer-specific parameters used their default values instead of config values:

| Parameter | Config Value | Default (Used) | Impact |
|-----------|-------------|----------------|--------|
| `warmup_steps` | 5000 | **0** | Full warp loss from step 0 |
| `warp_consistency_weight` | 0.5 | **1.0** | 2x gradient magnitude |

### 2. FP16 Numerical Instability in Loss Function

**File:** `ldm/modules/losses/contperceptual.py`

The NLL loss computation could overflow in FP16:
```python
nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
```

When `logvar` becomes very negative (e.g., -20), `exp(logvar)` is ~1e-9. Dividing by this tiny value exceeds FP16's max (~65504), causing overflow → NaN.

### 3. No Gradient Clipping with Manual Optimization

**File:** `src/trainer/vae_trainers.py`

The trainer uses manual optimization (dual optimizers), so PyTorch Lightning's `gradient_clip_val` parameter doesn't work. Without gradient clipping, exploding gradients could cause NaN.

## Fixes Applied

### Fix 1: Pass Trainer Params from Config

**File:** `train.py` (lines 111-120)

```python
# Before:
trainer_module = trainer_class(
    model_config=cfg.model,
    learning_rate=cfg.training.lr,
    ema_decay=cfg.training.get("ema_decay", 0.9999),
    image_key="image",
)

# After:
trainer_params = OmegaConf.to_container(cfg.trainer.get("params", {}), resolve=True)
trainer_module = trainer_class(
    model_config=cfg.model,
    learning_rate=cfg.training.lr,
    ema_decay=cfg.training.get("ema_decay", 0.9999),
    image_key="image",
    **trainer_params,  # Now passes warp_consistency_weight, warmup_steps, etc.
)
```

### Fix 2: Logvar Clamping for FP16 Stability

**File:** `ldm/modules/losses/contperceptual.py` (lines 97-100)

```python
# Before:
nll_loss = rec_loss / torch.exp(self.logvar.float()) + self.logvar.float()

# After:
logvar_clamped = torch.clamp(self.logvar.float(), min=-10.0, max=10.0)
nll_loss = rec_loss.float() / torch.exp(logvar_clamped) + logvar_clamped
```

**Why [-10, 10]?**
- `exp(-10)` ≈ 4.5e-5 (safe minimum divisor)
- `exp(10)` ≈ 22026 (within FP16 range)
- Also casts `rec_loss` to float32 before division

### Fix 3: Manual Gradient Clipping

**File:** `src/trainer/vae_trainers.py` (lines 634-635, 664-665)

```python
# After autoencoder backward pass:
opt_ae.zero_grad()
self.manual_backward(total_ae_loss)
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # NEW
opt_ae.step()

# After discriminator backward pass:
opt_disc.zero_grad()
self.manual_backward(discloss)
torch.nn.utils.clip_grad_norm_(self.model.loss.discriminator.parameters(), max_norm=1.0)  # NEW
opt_disc.step()
```

## Key Config Values

**File:** `config/warp_vae_co3d.yaml`

```yaml
training:
  precision: 16                    # FP16 mixed precision

trainer:
  params:
    warmup_steps: 5000             # Gradual warp loss ramp-up
    warp_consistency_weight: 0.5   # Balanced warp loss weight
```

## Testing

After applying fixes:
1. Clear Python cache: `find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} +`
2. Start fresh training: `python train.py --config-name=warp_vae_co3d training.num_epochs=1`
3. Monitor for NaN in first epoch

## Summary Table

| Issue | Symptom | Fix | File |
|-------|---------|-----|------|
| Config params not passed | Wrong warmup/weights | Pass `**trainer_params` | `train.py` |
| Logvar overflow | NaN in nll_loss | Clamp to [-10, 10] + float32 | `contperceptual.py` |
| Exploding gradients | NaN after many steps | Manual `clip_grad_norm_` | `vae_trainers.py` |

## Documentation

The FP16 stability fixes are also documented in `feature-backlog/Warp_VAE_Training.md` under section "### 5. FP16 Numerical Stability".
