# 3D-Aware VAE Research Project

## Project Goal
Improve the **3D awareness of 2D Variational Autoencoders (VAEs)**. The baseline is the Stable Diffusion VAE (AutoencoderKL). We iterate on it with multiple approaches that inject geometric priors so that the latent space respects 3D structure.

## Python Environment
- `conda activate cv`

## Quick Orientation
- **Entry point**: `train.py` (Hydra-based config, PyTorch Lightning training)
- **Configs**: `config/*.yaml` (one per experiment variant)
- **Documentation**: `feature-backlog/` contains detailed design docs for each approach

Start by reading `feature-backlog/README.md` for a documentation index, then `feature-backlog/QUICKSTART.md` for training commands.

## Repository Structure

```
train.py                          # Main training entry point (Hydra + Lightning)
preprocess_co3d.py                # CO3D dataset preprocessing (bboxes + annotations)
precompute_warps.py               # Precompute RoMA warp fields for faster training

config/                           # Hydra YAML configs (one per experiment)
  vanilla_vae_co3d.yaml           #   Baseline VAE
  eqvae_omniobject.yaml           #   EQ-VAE (equivariance regularization)
  warp_vae_co3d_precomputed.yaml  #   Warp VAE (multi-view consistency)
  plucker_vae_co3d.yaml           #   Plucker VAE (ray coordinate prediction)
  *_plucker_vae_co3d.yaml         #   Plucker VAE variants (concat, direct, conditioned)

ldm/                              # Core model code (from Stable Diffusion)
  models/autoencoder.py           #   All VAE model classes
  modules/losses/contperceptual.py#   LPIPSWithDiscriminator loss
  modules/diffusionmodules/model.py#  Encoder/Decoder architecture

src/
  trainer/
    base_trainer.py               #   Base Lightning module for all trainers
    vae_trainers.py               #   All trainer variants (Basic, EQ, Warp, Plucker)
  data/
    datamodule.py                 #   Lightning DataModule
    dataset_factory.py            #   Dataset instantiation from config
    co3d_dataset.py               #   CO3D dataset loader
    omniobject3d_dataset.py       #   OmniObject3D dataset loader
    warp_dataset.py               #   Warp datasets (online + precomputed)
  losses/
    warp_consistency.py           #   Warp consistency loss
  analysis/                       #   Evaluation utilities (metrics, visualization, RoMA)

scripts/                          # Analysis and visualization scripts
  analyze_multiview_latent_consistency.py  # Main evaluation script
  visualize_roma_warps.py                 # RoMA correspondence visualization
  visualize_distance_sampling.py          # Warp quality evaluation

feature-backlog/                  # Design documents (see feature-backlog/README.md)
```

## Model Variants (in order of complexity)

| Variant | Model Class | Key Idea | Dataset |
|---------|------------|----------|---------|
| **Vanilla VAE** | `AutoencoderKL` | Baseline SD-VAE, no geometric priors | CO3D |
| **EQ-VAE** | `EQVAEAutoencoder` | Equivariance to scale + rotation in latent space | OmniObject3D |
| **Warp VAE** | `AutoencoderKL` + `WarpVAETrainer` | Enforce latent consistency across views via RoMA warps | CO3D |
| **Plucker VAE** | `PluckerAutoencoder` | Predict Plucker ray coordinates from encoder features | CO3D |
| **Plucker Variants** | `ConcatPluckerVAE`, `DirectPluckerVAE`, `PluckerConditionedVAE` | Three approaches to integrating full-resolution Plucker rays | CO3D |

## Key Technologies
- **RoMA V2**: Dense feature matching for multi-view correspondences
- **Plucker Coordinates**: 6D ray representation (direction + moment) for 3D geometry
- **LPIPS**: Perceptual loss (frozen VGG16)
- **Hydra**: Configuration management
- **PyTorch Lightning**: Training framework
- **WandB**: Experiment tracking

## Datasets
- **CO3D** (Common Objects in 3D): Real-world video sequences, preprocessed via `preprocess_co3d.py`
- **OmniObject3D**: Synthetic 24-view renders of objects with known camera parameters

## Typical Workflow
1. Preprocess data: `python preprocess_co3d.py --category hydrant`
2. (Optional) Precompute warps: `python precompute_warps.py --annotation_file ...`
3. Train: `python train.py --config-name=<config>`
4. Evaluate: `python scripts/analyze_multiview_latent_consistency.py --checkpoints ... --configs ...`
