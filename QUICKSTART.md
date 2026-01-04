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

### Mixed Precision (FP16)

Already enabled by default (`precision: 16`):

```bash
python train.py --config-name=vanilla_vae_co3d \
    training.precision=32  # Use FP32 if needed
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

### Out of Memory

Reduce batch size:
```bash
python train.py --config-name=plucker_vae_co3d training.batch_size=2
```

Enable gradient accumulation:
```bash
python train.py --config-name=plucker_vae_co3d \
    training.batch_size=2 \
    training.accumulate_grad_batches=4
```

### Data Loading Too Slow

Increase workers:
```bash
python train.py --config-name=vanilla_vae_co3d \
    data.params.num_workers=8
```

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

## Next Steps

1. **Test on small dataset**: Verify setup works
2. **Run full training**: Use complete dataset
3. **Experiment**: Try different model/dataset combinations
4. **Evaluate**: Check reconstruction quality
5. **Deploy**: Use trained VAE for downstream tasks

---

## Support

- **Implementation Details**: See [MIGRATION_COMPLETE.md](MIGRATION_COMPLETE.md)
- **Detailed Plan**: See [feature-backlog/(4)modular-migration.md](feature-backlog/(4)modular-migration.md)
- **Issues**: Check console warnings and error messages

---

**Happy Training! 🚀**
