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
config/                           # Hydra YAML configs (one per experiment)

third_party/                      # External dependencies (added to sys.path via .pth file)
  ldm/                            #   Stable Diffusion VAE code
  taming/                         #   Taming Transformers (LPIPS weights)
  RoMA2/                          #   RoMA V2 dense matching
  co3d/                           #   CO3D dataset tools (Meta)

src/
  trainer/                        #   Lightning training modules
  data/                           #   Dataset loaders
  losses/                         #   Loss functions
  analysis/                       #   Evaluation utilities

scripts/                          # All scripts (preprocessing, visualization, shell)
  data_process/                   #   Dataset processing utilities (Python package)

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
4. Evaluate: `python scripts/analysis/analyze_multiview_latent_consistency.py --checkpoints ... --configs ...`
