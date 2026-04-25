# Quick Start Guide - Modular VAE Training

## Installation

1. **Activate environment**:
   ```bash
   conda activate cv
   ```

2. **Verify installation**:
   ```bash
   python -c "import torch; import pytorch_lightning; print('✓ Ready')"
   ```

3. **Quick test** (verify modular system works):
   ```bash
   # Test on small model (should complete in ~30 seconds)
   python train.py --config-name=eqvae_omniobject_small \
       training.num_epochs=1 \
       wandb.enabled=false \
       data.params.val_split=0.5
   ```

   **Expected output**: Should show training progress without errors
   ```
   Epoch 0:   0%|          | 10/1000 [00:03<05:30, 3.0it/s, train/aeloss_step=45000]
   ```

---

## Training Examples

### 1. Vanilla VAE on CO3D (Baseline)

Train a standard AutoencoderKL without geometric priors:

```bash
python train.py --config-name=vanilla_vae_co3d
```

**What it does**:
- Model: Standard Stable Diffusion VAE architecture
- Dataset: CO3D with bounding box cropping
- No Plucker coordinates (pure image reconstruction)

**Customize**:
```bash
python train.py --config-name=vanilla_vae_co3d \
    training.batch_size=16 \
    training.num_epochs=50 \
    training.lr=1e-5 \
    co3d_dir=/your/path/to/co3d \
    bb_file=/your/path/to/bboxes.jgz
```

---

### 2. Plucker VAE on CO3D (Geometric Prior)

Train with Plucker ray coordinates for 3D awareness:

```bash
python train.py --config-name=plucker_vae_co3d
```

**What it does**:
- Model: PluckerAutoencoder with geometric reasoning
- Dataset: CO3D with Plucker coordinate computation
- Losses: Reconstruction + KL + Plucker geometric constraints

**Customize loss weights**:
```bash
python train.py --config-name=plucker_vae_co3d \
    training.plucker_loss_weight=0.2 \
    training.plucker_recon_weight=1.5 \
    training.plucker_constraint_weight=0.15
```

---

### 3. EQ-VAE on OmniObject (Equivariance)

Train with equivariance regularization:

```bash
python train.py --config-name=eqvae_omniobject
```

**What it does**:
- Model: EQVAEAutoencoder with probabilistic transforms
- Dataset: OmniObject3D (single views)
- Regularization: Equivariance to rotations and scaling

**Customize**:
```bash
python train.py --config-name=eqvae_omniobject \
    data.data_dir=/your/path/to/omniobject \
    model.params.p_prior=0.85 \
    model.params.equivariance_weight=1.5
```

---

### 4. EQ-VAE Small Model (Memory-Efficient Testing)

For testing on limited GPU memory (GTX 1080 Ti, 11GB):

```bash
python train.py --config-name=eqvae_omniobject_small
```

**What it does**:
- Model: Reduced EQVAEAutoencoder (ch=64, 3 layers instead of 4)
- Dataset: OmniObject3D at 128x128 resolution
- Memory: ~2.6GB GPU memory (vs ~10GB for full model)
- Speed: ~3.7 iterations/second

**Use this config for**:
- Quick prototyping and debugging
- Training on consumer GPUs
- Testing new features before full-scale training

**Scale up when ready**:
```bash
# Gradually increase model capacity
python train.py --config-name=eqvae_omniobject_small \
    model.params.ddconfig.ch=96 \
    data.params.dataset_config.params.image_size=256
```

---

### 5. Warp VAE on CO3D (Multi-View Consistency)

Train with warp consistency loss using precomputed RoMA correspondences:

```bash
python train.py --config-name=warp_vae_hydrant_recon_crop
```

**What it does**:
- Model: Standard AutoencoderKL with warp consistency + reconstruction losses
- Dataset: CO3D with precomputed RoMA warps
- Losses: Reconstruction + KL + Perceptual + Warp Consistency + Warp Reconstruction

**Variants**:
```bash
# Reconstruction loss only (ablation, no consistency loss)
python train.py --config-name=warp_vae_hydrant_recon_only

# Naive warp (EQ-VAE-style equivariance)
python train.py --config-name=naive_warp_vae_hydrant

# Depth-based warps (ground-truth geometry instead of RoMA)
python train.py --config-name=depth_warp_vae_hydrant
```

**Prerequisite**: Precompute warps first (see [ROMA_PRECOMPUTE_SPEEDUP.md](ROMA_PRECOMPUTE_SPEEDUP.md) or [Warp_VAE_Training.md](Warp_VAE_Training.md) for depth-based warps).

---

## Loading Pretrained Weights

Start from Stable Diffusion VAE checkpoint:

```bash
python train.py --config-name=vanilla_vae_co3d \
    pretrained_weights=./sd_model/v1-5-pruned.ckpt
```

Or add to your config file:
```yaml
pretrained_weights: "./sd_model/v1-5-pruned.ckpt"
```

---

## Monitoring Training

### WandB (Enabled by Default)

View training at: https://wandb.ai/

**Configure**:
```yaml
wandb:
  enabled: true
  project: enhanced-VAE
  entity: your-team
  tags: ["experiment-name", "tags"]
```

**Disable**:
```bash
python train.py --config-name=vanilla_vae_co3d wandb.enabled=false
```

### Local Logs

Checkpoints saved to:
```
checkpoints/<run-name>/
├── vae-epoch005.ckpt
├── vae-epoch010.ckpt
└── last.ckpt
```

Output logs saved to:
```
outputs/<model-type>/<run-name>/
└── last_model.pth
```

---

## Common Configurations

### Small Dataset Testing

```bash
python train.py --config-name=vanilla_vae_co3d \
    training.batch_size=4 \
    training.num_epochs=5 \
    training.val_split=0.2
```

### Multi-GPU Training

Automatically detected via `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config-name=plucker_vae_co3d
```

---

## Resuming Training

```bash
python train.py --config-name=vanilla_vae_co3d \
    ckpt_path=checkpoints/<run-name>/last.ckpt
```

---

## Creating Custom Configs

### 1. Copy existing config:
```bash
cp config/vanilla_vae_co3d.yaml config/my_experiment.yaml
```

### 2. Modify parameters:
```yaml
training:
  batch_size: 12
  num_epochs: 200
  lr: 2e-6
  note: "my-custom-experiment"

data:
  params:
    dataset_config:
      params:
        apply_augmentation: true  # Enable data augmentation
```

### 3. Run:
```bash
python train.py --config-name=my_experiment
```

---

## Troubleshooting

### Out of Memory (OOM)

**Option 1**: Use the small model config (recommended for testing):
```bash
python train.py --config-name=eqvae_omniobject_small
```

**Option 2**: Reduce batch size:
```bash
python train.py --config-name=plucker_vae_co3d training.batch_size=2
```

**Option 3**: Reduce model capacity:
```bash
python train.py --config-name=eqvae_omniobject \
    model.params.ddconfig.ch=64 \
    model.params.ddconfig.ch_mult="[1,2,4]"
```

**Option 4**: Lower image resolution:
```bash
python train.py --config-name=eqvae_omniobject \
    data.params.dataset_config.params.image_size=128
```

**Option 5**: Use single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config-name=plucker_vae_co3d
```

**Note**: Gradient accumulation is **not compatible** with manual optimization (used for dual optimizer setup). Use smaller batch sizes instead.

### Check Configuration

Print resolved config:
```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/vanilla_vae_co3d.yaml')
print(OmegaConf.to_yaml(cfg, resolve=True))
"
```

---

**Memory Calculation**:
```
Approximate GPU Memory = Base Model + (Batch Size × Image Memory)
- Base Model (ch=128, 4 layers): ~6 GB
- Base Model (ch=64, 3 layers): ~1.5 GB
- Image Memory (256×256): ~1.1 GB per sample
- Image Memory (128×128): ~0.28 GB per sample
```

---

## Dataset Setup

### CO3D

1. Download CO3D dataset: https://github.com/facebookresearch/co3d
2. Generate bounding boxes (or use provided)
3. Update config:
   ```yaml
   co3d_dir: "/data/lab_moezkan/co3d_full"
   bb_file: "/data/lab_moezkan/co3d_bboxes"
   ```

### OmniObject3D

1. Download from: https://omniobject3d.github.io/
2. Extract to directory with structure:
   ```
   omniobject/
   └── img/
       ├── object_001/
       │   ├── 000.png
       │   ├── 001.png
       │   └── transforms.json
       ├── object_002/
       └── ...
   ```
3. Update config:
   ```yaml
   data_dir: "/data/lab_moezkan/omni_obj/blender_renders_24_views"
   ```

---

## Architecture Summary

```
┌─────────────────────────────────────────────────┐
│                  Config (YAML)                  │
└─────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌───────────────┐              ┌────────────────┐
│   VAE Model   │              │  Data Module   │
│ (from config) │              │ (from config)  │
└───────────────┘              └────────────────┘
        ↓                               ↓
┌─────────────────────────────────────────────────┐
│            Trainer Module (Lightning)           │
│  - Handles training loop                        │
│  - Computes losses                              │
│  - Logs metrics                                 │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│         PyTorch Lightning Trainer               │
│  - Multi-GPU support                            │
│  - Checkpointing                                │
│  - Mixed precision                              │
└─────────────────────────────────────────────────┘
```

---

## Further Reading

- **Model Variants**: See [EQ-VAE_Implementation.md](EQ-VAE_Implementation.md), [Warp_VAE_Training.md](Warp_VAE_Training.md) (includes Naive Warp + Depth Warp variants), [PluckerVAE_Variants.md](PluckerVAE_Variants.md)
- **Evaluation**: See [MultiViewConsistencyEval.md](MultiViewConsistencyEval.md)
- **Documentation Index**: See [README.md](README.md)
