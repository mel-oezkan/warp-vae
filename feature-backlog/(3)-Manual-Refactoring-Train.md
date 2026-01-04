1. Move the Trainer Code from the train.py into separate training modules. 

We create a base trainer module in `src/trainers/base_trainer.py` where we define basic training flows.

2. For the different sub-models we create subclasses where we a


# Implementation Plan

## Phase 1: Refactor Trainer Architecture
### 1.1 Create Base Trainer Class
File: `src/trainers/base_trainer.py`
```
BaseVAETrainer(pl.LightningModule)
├── Common functionality:
│   ├── Checkpoint loading/saving
│   ├── Optimizer configuration
│   ├── Logging infrastructure
│   ├── EMA handling
│   └── Image logging
├── Abstract methods:
│   ├── get_input(batch) -> dict
│   └── compute_loss(batch, model_output) -> loss, log_dict
```

### 1.2 Create Model-Specific Trainer Mixins or Subclasses
File: `src/trainers/vae_trainers.py`
```Python
class VanillaVAETrainer(BaseVAETrainer):
    """For AutoencoderKL - standard image reconstruction"""
    
class PluckerVAETrainer(BaseVAETrainer):
    """For PluckerAutoencoder - adds Plucker loss handling"""
    
class EQVAETrainer(BaseVAETrainer):
    """For EQVAEAutoencoder - handles equivariance transforms"""
```

Why separate classes instead of one:
- Clear separation of concerns
- Each trainer can define its own training_step logic
- Config can directly specify which trainer to use

## Phase 2: Unified Dataset Loading
### 2.1 Create Dataset Factory
File: `src/data/dataset_factory.py`
```Python
def get_dataset(config):
    """
    Factory function that returns appropriate dataset based on config.
    
    Config structure:
        data:
            target: src.data.co3d_dataset.CO3DDataset
            params:
                root_dir: ...
                plucker_coords: true/false  # Controls whether to load Plucker data
    """
```

### 2.2 Create Unified DataModule
File: `src/data/datamodule.py`
```Python
class VAEDataModule(pl.LightningDataModule):
    """
    Unified data module that:
    - Instantiates dataset from config
    - Handles train/val/test splits
    - Configures DataLoader settings
    """
```

Example Config files:
``` Yaml
# CO3D dataset configuration for Vanilla VAE (no Plucker)
data:
  target: src.data.datamodule.VAEDataModule
  params:
    dataset_config:
      type: co3d
      params:
        root_dir: ${data.root_dir}
        image_size: ${training.image_size}
        include_plucker: false
        crop_images: true
    batch_size: ${training.batch_size}
    val_split: ${training.val_split}
    num_workers: 4
    pin_memory: true
```

```Yaml
# CO3D dataset configuration for Plucker VAE
data:
  target: src.data.datamodule.PairedVAEDataModule
  params:
    dataset_config:
      type: co3d
      params:
        root_dir: ${data.root_dir}
        bb_file: ${data.bb_file}
        image_size: ${training.image_size}
        include_plucker: true
        n_patches: 8
        crop_images: true
    batch_size: ${training.batch_size}
    val_split: ${training.val_split}
    num_workers: 4
    include_plucker: true
    pair_sampling: random
```

```Yaml
# OmniObject3D dataset configuration for EQ-VAE
data:
  target: src.data.datamodule.VAEDataModule
  params:
    dataset_config:
      type: omniobject
      params:
        root_dir: ${data.root_dir}
        image_size: ${training.image_size}
        include_plucker: false
    batch_size: ${training.batch_size}
    val_split: ${training.val_split}
    num_workers: 4
```

## Phase 4: Config Structure

### 4.1 Example Config for Vanilla VAE
File: `config/train_vanilla_vae.yaml`
```Yaml
model:
  target: ldm.models.autoencoder.AutoencoderKL
  params:
    ddconfig: ...
    lossconfig: ...

trainer:
  target: src.trainers.vae_trainers.VanillaVAETrainer

data:
  target: src.data.datamodule.VAEDataModule
  params:
    dataset:
      target: src.data.co3d_dataset.CO3DDataset
      params:
        include_plucker: false
```

### 4.2 Example Config for Plucker VAE
File: `config/train_plucker_vae.yaml`
```Yaml
model:
  target: ldm.models.autoencoder.PluckerAutoencoder
  params:
    n_patches: 8
    plucker_key: "plucker_coords"
    ...

trainer:
  target: src.trainers.vae_trainers.PluckerVAETrainer

data:
  target: src.data.datamodule.VAEDataModule
  params:
    dataset:
      target: src.data.co3d_dataset.CO3DDataset
      params:
        include_plucker: true
```
### 4.3 Example Config for EQ-VAE
File: `config/train_eqvae.yaml`
```Yaml
```

# Todos
- [ ] Let Claude test the existing dataset implementations
- [ ] Let claude make the datasets work with the new boilerplate code
- [ ] Create the configs 
- [ ] Extend the training script to work with the new models and datasets
- [ ] Write tests for data+model combinations    

- check how good the ImageNet implementation was and possibly also include it into the training

# Migration Path
Phase 1: Create base_trainer.py and vae_trainers.py - refactor existing FinetuneVAE ✅ 
Phase 2: Create VAEDataModule - wrap existing datasets ✅ 
Phase 3: Modify the old Datasets to work withe the new data factory and Data module **CURRENT** 
  - look at the implementations in src/dataset/co3d.py and src/dataset/omni_obj.py and get adapt to existing code
Phase 4: Create example configs for each model type
Phase 5: Test the dataset implementation
Phase 6: Update train.py to use new structure
Phase 7: Test each model/dataset combination