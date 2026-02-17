# 3D-Aware VAE -- Computer Vision Project Lab

University of Darmstadt -- Computer Vision Project Lab

Research project to improve the **3D awareness of 2D Variational Autoencoders**. Starting from the Stable Diffusion VAE as a baseline, we explore multiple approaches to inject geometric priors into the latent space.

## Approaches

| Approach | Key Idea |
|----------|----------|
| **Warp VAE** | Multi-view consistency -- use RoMA dense correspondences to enforce that the same 3D surface maps to similar latent codes across viewpoints |
| **Plucker VAE** | Geometric conditioning -- integrate Plucker ray coordinates (6D camera ray representation) into the encoder/decoder |

## Setup

```bash
git clone --recurse-submodules https://github.com/mel-oezkan/computer-vision-proj-lab.git
conda activate cv
```

## Configuration

Configuration is managed with [Hydra](https://hydra.cc/docs/1.3/tutorials/basic/your_first_app/using_config/). Each experiment has a YAML config in `config/`. Override any parameter from the command line:

```bash
python train.py --config-name=eqvae_omniobject training.batch_size=4 training.lr=1e-5
```
