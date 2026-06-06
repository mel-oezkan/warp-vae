# 3D-Aware VAE

**Computer Vision Project Lab -- TU Darmstadt**

Improving the 3D awareness of 2D Variational Autoencoders by injecting geometric priors into the latent space. Starting from the Stable Diffusion VAE (AutoencoderKL) as baseline, we explore approaches that enforce multi-view consistency and geometric structure in the learned representations.

## Approaches

| Approach | Key Idea | Config |
|----------|----------|--------|
| **Vanilla VAE** | Baseline SD-VAE, no geometric priors | `vanilla_vae_co3d` |
| **EQ-VAE** | Equivariance regularization -- enforce latent invariance to scale and rotation transforms | `eqvae_omniobject` |
| **Warp VAE** | Multi-view consistency -- use [RoMA](https://github.com/Parskatt/RoMA) dense correspondences to enforce that the same 3D surface maps to similar latent codes across viewpoints | `warp_vae_co3d_precomputed` |
| **Plucker VAE** | Geometric conditioning -- integrate [Plucker ray coordinates](https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates) (6D camera ray representation) into the encoder/decoder | `plucker_vae_co3d` |

## Setup

```bash
git clone https://github.com/mel-oezkan/warp-vae.git
cd warp-vae
conda activate cv
```

### External Dependencies (`third_party/`)

`third_party/ldm` and `third_party/taming` are vendored in this repo. The two
larger upstream dependencies -- **RoMA V2** and **CO3D** -- are *not* committed;
fetch them (pinned to the commits this project was developed against) with:

```bash
bash scripts/setup_third_party.sh
```

This clones `third_party/RoMA2` and `third_party/co3d`. They must be importable
on `PYTHONPATH` (the `cv` conda env adds `third_party/` and `third_party/RoMA2/src`
via a `.pth` file in its site-packages).

### Dataset Preparation

**CO3D** (used by Vanilla, Warp, and Plucker VAE):
```bash
# 1. Download CO3D: https://github.com/facebookresearch/co3d
# 2. Generate bounding boxes
python preprocess_co3d.py --category hydrant
```

**OmniObject3D** (used by EQ-VAE):
Download from [omniobject3d.github.io](https://omniobject3d.github.io/) and set `data_dir` in config.

### Precompute Warps (for Warp VAE)

```bash
python precompute_warps.py --annotation_file <path_to_annotations>
```

## Training

All training uses [Hydra](https://hydra.cc/) for configuration. Each experiment has a YAML config in [`config/`](config/).

```bash
# Baseline
python train.py --config-name=vanilla_vae_co3d

# Warp VAE (precomputed correspondences)
python train.py --config-name=warp_vae_co3d_precomputed

# Plucker VAE
python train.py --config-name=plucker_vae_co3d

# EQ-VAE
python train.py --config-name=eqvae_omniobject
```

Override any parameter from the command line:
```bash
python train.py --config-name=warp_vae_co3d_precomputed \
    training.batch_size=4 \
    training.lr=1e-5
```

Multi-GPU:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config-name=warp_vae_co3d_precomputed
```

### Loading Pretrained Weights

```bash
python train.py --config-name=vanilla_vae_co3d \
    pretrained_weights=./sd_model/v1-5-pruned.ckpt
```

## Evaluation

Measure multi-view latent consistency across viewpoints:

```bash
python scripts/analyze_multiview_latent_consistency.py \
    --checkpoints <ckpt_path> \
    --configs <config_name>
```

## Project Structure

```
train.py                            # Entry point (Hydra + PyTorch Lightning)
preprocess_co3d.py                  # CO3D preprocessing (bboxes + annotations)
precompute_warps.py                 # Precompute RoMA warp fields

config/                             # Experiment configs (one per variant)
ldm/                                # Core model code (from Stable Diffusion)
  models/autoencoder.py             #   VAE model classes
  modules/losses/contperceptual.py  #   LPIPS + discriminator loss
  modules/diffusionmodules/model.py #   Encoder/Decoder architecture

src/
  trainer/vae_trainers.py           # All trainer variants (Basic, EQ, Warp, Plucker)
  data/                             # Datasets and data loading
  losses/warp_consistency.py        # Warp consistency loss
  analysis/                         # Evaluation utilities

scripts/                            # Analysis and visualization
feature-backlog/                    # Design documents (see feature-backlog/README.md)
```

## Documentation

Detailed design documents live in [`feature-backlog/`](feature-backlog/). See [`feature-backlog/README.md`](feature-backlog/README.md) for an index and recommended reading order.

## Key Technologies

- **[RoMA V2](https://github.com/Parskatt/RoMA)** -- Dense feature matching for multi-view correspondences
- **[PyTorch Lightning](https://lightning.ai/)** -- Training framework
- **[Hydra](https://hydra.cc/)** -- Configuration management
- **[WandB](https://wandb.ai/)** -- Experiment tracking
- **[LPIPS](https://github.com/richzhang/PerceptualSimilarity)** -- Perceptual loss (frozen VGG16)
