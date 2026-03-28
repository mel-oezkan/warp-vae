# Run Comparison: EQ-VAE vs Warp VAE variants

## Runs

| Run | W&B ID | Name |
|-----|--------|------|
| EQ-VAE | `tho4uc2z` | EQ-VAE on CO3D hydrant 50seq, step-matched |
| Warp VAE | `e4ksa79v` | hydrant 50seq nocrop, warp_w=1, disc_w=0.5, disc_start=15k |
| Warp VAE + Recon | `gyaorhum` | hydrant 50seq nocrop, warp_w=1, warp_recon_w=1, disc_w=0.5, disc_start=15k |

## Configuration Comparison

| Setting | EQ-VAE (`tho4uc2z`) | Warp VAE (`e4ksa79v`) | Warp VAE + Recon (`gyaorhum`) |
|---------|----------------------|-----------------------|-------------------------------|
| **Model** | `EQVAEAutoencoder` | `AutoencoderKL` | `AutoencoderKL` |
| **Trainer** | `EQVAETrainer` | `WarpVAETrainer` | `WarpVAETrainer` |
| **Key idea** | Equivariance regularization (scale+rotation in latent space) | Latent warp consistency loss | Latent warp consistency + warp reconstruction loss |
| **Dataset** | CO3D 50seq (4915 single images) | Precomputed warp pairs (14487 pairs, bidirectional) | Same as Warp VAE |
| **lr** | 4.5e-6 | 1e-5 | 1e-5 |
| **kl_weight** | 1e-5 | 1e-5 | 1e-5 |
| **disc_start** | 15,000 | 15,000 | 15,000 |
| **disc_weight** | 0.5 | 0.5 | 0.5 |
| **grad_accum** | none | 4 | 4 |
| **epochs** | 30 | 20 | 20 |
| **optimizer steps** | ~58,980 | 57,939 | 57,939 |
| **warp_consistency_weight** | n/a | 1.0 | 1.0 |
| **warp_reconstruction_weight** | n/a | **0** | **1** |

## Key Differences

### Warp VAE (`e4ksa79v`) vs Warp VAE + Recon (`gyaorhum`)

The only difference is `warp_reconstruction_weight`:
- **Warp VAE**: `warp_reconstruction_weight=0` — only enforces latent consistency across warped views
- **Warp VAE + Recon**: `warp_reconstruction_weight=1` — additionally reconstructs the warped view in pixel space

Everything else (model, dataset, lr, loss weights, epochs, steps) is identical.

### EQ-VAE (`tho4uc2z`) vs both Warp VAE runs

- **Approach**: Equivariance regularization (random scale/rotation of latents) instead of multi-view warp consistency
- **Learning rate**: 4.5e-6 (EQ-VAE default) vs 1e-5
- **Dataset**: Uses single images from the same 50-sequence subset, not warp pairs
- **Training duration**: 30 epochs (vs 20) to match ~58K optimizer steps
- **EQ-VAE specific params**: `p_prior=0.5`, `p_prior_s=0.25`, `use_rotation=true`, `equivariance_weight=1.0`
