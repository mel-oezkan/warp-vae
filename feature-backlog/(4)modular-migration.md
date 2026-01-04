# Modular Training Pipeline Migration Sub-Plan

## Executive Summary
This document provides a detailed, step-by-step plan to complete the modular refactoring of the training pipeline. The base infrastructure (trainers, data modules, factory pattern) is complete. This plan focuses on **Phase 3-7**: migrating existing datasets, creating configs, and testing the full system.

---

## High-Level Steps

1. **Migrate CO3D Dataset** - Adapt existing CO3D implementation to new BaseVAEDataset interface
2. **Migrate OmniObject Dataset** - Adapt existing OmniObject implementation with paired-view support
3. **Fix Dataset Factory Registry** - Update module paths to point to new implementations
4. **Create Modular Configs** - Write YAML configs for each model/dataset combination
5. **Write Dataset Tests** - Unit tests to validate dataset functionality
6. **Refactor train.py** - Update training script to use modular architecture
7. **Integration Testing** - End-to-end testing of all model/dataset combinations

---

## Phase 3: Dataset Migration (CURRENT)

### Step 3.1: Migrate CO3D Dataset

**Objective**: Create a new CO3D dataset implementation that inherits from `BaseVAEDataset` and integrates with the factory pattern.

**Action Items**:

1. **Create new file**: `src/data/co3d_dataset.py`

2. **Implement `CO3DDataset` class**:
   ```python
   class CO3DDataset(BaseVAEDataset):
       def __init__(
           self,
           root_dir: str,              # Path to CO3D data
           bb_file: str,               # Bounding box JSON file
           image_size: int = 256,
           include_plucker: bool = False,
           n_patches: int = 8,
           crop_images: bool = False,
           apply_augmentation: bool = False,
           **kwargs
       ):
   ```

3. **Port logic from** `src/dataset/co3d.py`:
   - Sample loading from `ProcessedCo3D._load_samples()`
   - Image loading and cropping logic
   - NDC crop parameter computation
   - Plucker coordinate computation (reuse existing `compute_directions_from_sample`, `ray_to_plucker`)

4. **Implement required abstract methods**:
   - `__len__()` → return `len(self.samples)`
   - `_load_image(idx)` → Load, crop, transform image
   - `_load_plucker_coords(idx)` → Compute Plucker rays if `include_plucker=True`
   - `_get_camera_params(idx)` → Return dict with R, T, focal_length, principal_point

5. **Standardize return keys**:
   - Old: `"pluck_ray"` → New: `"plucker_coords"`
   - Add: `"camera"` dict containing R, T, focal_length, principal_point, crop_params
   - Keep: `"image"`, `"index"`

6. **Preserve augmentation support**:
   - Keep `jitter_bbox`, `square_bbox` logic from `data_process/co3d_dataset.py`
   - Controlled by `apply_augmentation` parameter

**Expected Outcome**:
- New `CO3DDataset` class fully compatible with `VAEDataModule` and factory pattern
- Drop-in replacement for `src/dataset/co3d.py::ProcessedCo3D`

---

### Step 3.2: Migrate OmniObject Dataset

**Objective**: Create a new OmniObject dataset that supports both single-view and paired-view modes, inheriting from `BaseVAEDataset`.

**Action Items**:

1. **Create new file**: `src/data/omniobject3d_dataset.py`

2. **Implement `OmniObject3DDataset` class**:
   ```python
   class OmniObject3DDataset(BaseVAEDataset, PairedDatasetMixin):
       def __init__(
           self,
           root_dir: str,              # Path to OmniObject data
           image_size: int = 256,
           include_plucker: bool = False,
           n_patches: int = 8,
           sample_mode: str = "single",    # "single" or "pairs"
           pair_sampling: str = "sequential",  # "sequential", "random", "fixed_interval"
           **kwargs
       ):
   ```

3. **Port logic from** `src/dataset/omni_obj.py`:
   - Object directory discovery
   - transforms.json parsing
   - Camera parameter extraction from transform matrices
   - View pair generation logic

4. **Implement required methods**:
   - `__len__()` → Length based on sample_mode (single views vs pairs)
   - `_load_image(idx)` → Load PNG from object directory
   - `_load_plucker_coords(idx)` → Compute from camera params
   - `_get_camera_params(idx)` → Extract from transforms.json

5. **Handle paired-view mode**:
   - **Option A (Recommended)**: Return ONLY first view in `__getitem__`, use `PairedDatasetMixin` to handle pairing
   - **Option B**: Return dict with `image2`, `R2`, `T2`, etc. keys (current approach in `omni_obj.py`)
   - **Decision needed**: Check how EQ-VAE trainer expects paired data

6. **Standardize return keys**:
   - Single-view: `{"image", "plucker_coords", "camera", "index", "object_name", "view_idx"}`
   - Paired-view (if Option B): Add `image2`, `camera2`, `plucker_coords2`, `R_rel`, `T_rel`

**Expected Outcome**:
- New `OmniObject3DDataset` compatible with modular architecture
- Supports both single-view (for Vanilla/Plucker VAE) and paired-view (for EQ-VAE)

---

### Step 3.3: Update Dataset Factory Registry

**Objective**: Fix registry to point to new dataset implementations.

**Action Items**:

1. **Edit file**: `src/data/dataset_factory.py`

2. **Update `DATASET_REGISTRY`**:
   ```python
   DATASET_REGISTRY: Dict[str, str] = {
       "co3d": "src.data.co3d_dataset.CO3DDataset",
       "omniobject": "src.data.omniobject3d_dataset.OmniObject3DDataset",
       # Remove or update: "mvimgnet": "src.data.mvimgnet_dataset.MVImgNetDataset",
   }
   ```

3. **Test factory instantiation**:
   - Verify `get_dataset({"type": "co3d", "params": {...}})` works
   - Verify `get_dataset({"target": "src.data.co3d_dataset.CO3DDataset", "params": {...}})` works

**Expected Outcome**: Factory can instantiate CO3D and OmniObject datasets via config.

---

## Phase 4: Create Configuration Files

### Step 4.1: Config for Vanilla VAE + CO3D

**Objective**: Create a config for training standard VAE on CO3D (no Plucker).

**Action Items**:

1. **Create file**: `config/vanilla_vae_co3d.yaml`

2. **Structure** (following pattern from `baseVAE.yaml`):
   ```yaml
   model:
     base_learning_rate: 4.5e-6
     target: ldm.models.autoencoder.AutoencoderKL
     params:
       embed_dim: 4
       monitor: val/rec_loss
       ddconfig: {...}
       lossconfig: {...}

   trainer:
     target: src.trainer.vae_trainers.VanillaVAETrainer

   data:
     target: src.data.datamodule.VAEDataModule
     params:
       dataset_config:
         type: co3d
         params:
           root_dir: ${data.co3d_dir}
           bb_file: ${data.bb_file}
           image_size: ${training.image_size}
           include_plucker: false
           crop_images: true
       batch_size: ${training.batch_size}
       val_split: ${training.val_split}
       num_workers: 4

   training:
     batch_size: 8
     image_size: 256
     num_epochs: 100
     lr: 4.5e-6
     kl_weight: 0.000001
     lpips_loss_weight: 1.0
     val_split: 0.1
     ema_decay: 0.9999
     precision: 16
     output_dir: ./outputs/vanilla_vae_co3d

   wandb:
     enabled: true
     project: enhanced-VAE
     tags: ["vanilla-vae", "co3d"]
   ```

**Expected Outcome**: Working config for Vanilla VAE training on CO3D.

---

### Step 4.2: Config for Plucker VAE + CO3D

**Objective**: Create a config for Plucker VAE with CO3D dataset.

**Action Items**:

1. **Create file**: `config/plucker_vae_co3d.yaml`

2. **Structure**:
   ```yaml
   model:
     base_learning_rate: 4.5e-6
     target: ldm.models.autoencoder.PluckerAutoencoder
     params:
       embed_dim: 4
       n_patches: 8
       plucker_key: "plucker_coords"  # Match new standardized key
       monitor: val/rec_loss
       ddconfig: {...}
       lossconfig: {...}

   trainer:
     target: src.trainer.vae_trainers.PluckerVAETrainer

   data:
     target: src.data.datamodule.VAEDataModule
     params:
       dataset_config:
         type: co3d
         params:
           root_dir: ${data.co3d_dir}
           bb_file: ${data.bb_file}
           image_size: ${training.image_size}
           include_plucker: true      # Enable Plucker coords
           n_patches: 8
           crop_images: true
       batch_size: ${training.batch_size}
       val_split: ${training.val_split}
       num_workers: 4

   training:
     batch_size: 4
     image_size: 256
     plucker_loss_weight: 0.1
     plucker_recon_weight: 1.0
     plucker_constraint_weight: 0.1
     plucker_norm_weight: 0.1
     # ... other training params
   ```

**Expected Outcome**: Working config for Plucker VAE with geometric priors.

---

### Step 4.3: Update Config for EQ-VAE + OmniObject

**Objective**: Update existing `config/eqvae_omniobject.yaml` to use new data structure.

**Action Items**:

1. **Edit file**: `config/eqvae_omniobject.yaml`

2. **Update `data` section**:
   ```yaml
   data:
     target: src.data.datamodule.VAEDataModule
     params:
       dataset_config:
         type: omniobject
         params:
           root_dir: ${data.data_dir}
           image_size: ${training.image_size}
           include_plucker: false
           sample_mode: single        # EQ-VAE uses single views
       batch_size: ${training.batch_size}
       val_split: ${training.val_split}
       num_workers: 8

   trainer:
     target: src.trainer.vae_trainers.EQVAETrainer
   ```

3. **Remove old `dataset_type` field**

**Expected Outcome**: Updated config compatible with new modular data structure.

---

## Phase 5: Write Dataset Tests

### Step 5.1: Unit Tests for CO3D Dataset

**Objective**: Validate CO3D dataset functionality.

**Action Items**:

1. **Create file**: `tests/test_co3d_dataset.py`

2. **Test cases**:
   - `test_co3d_loading_without_plucker()` - Load dataset, check keys
   - `test_co3d_loading_with_plucker()` - Verify plucker_coords shape
   - `test_co3d_augmentation()` - Test jitter/crop
   - `test_co3d_return_keys()` - Verify standardized dictionary structure
   - `test_co3d_with_datamodule()` - Test integration with VAEDataModule

3. **Mock data** (if needed): Create small sample CO3D structure for testing

**Expected Outcome**: Automated tests ensuring CO3D dataset correctness.

---

### Step 5.2: Unit Tests for OmniObject Dataset

**Objective**: Validate OmniObject dataset functionality.

**Action Items**:

1. **Create file**: `tests/test_omniobject_dataset.py`

2. **Test cases**:
   - `test_omniobject_single_view_mode()` - Verify single view returns
   - `test_omniobject_paired_view_mode()` - Verify paired view structure
   - `test_omniobject_pair_sampling()` - Test sequential/random/fixed strategies
   - `test_omniobject_camera_extraction()` - Validate camera parameter parsing
   - `test_omniobject_with_datamodule()` - Test with VAEDataModule

**Expected Outcome**: Comprehensive tests for OmniObject dataset.

---

## Phase 6: Refactor train.py

### Step 6.1: Update Training Script Architecture

**Objective**: Refactor `train.py` to use modular trainers and data modules.

**Action Items**:

1. **Edit file**: `train.py`

2. **Replace direct instantiation** with config-based loading:
   ```python
   # OLD (lines 124-145):
   if dataset_type == "co3d":
       data_module = Co3DDataModule(...)
   elif dataset_type == "omniobject":
       data_module = OmniObjectDataModule(...)

   # NEW:
   from ldm.util import instantiate_from_config
   data_module = instantiate_from_config(cfg.data)
   ```

3. **Replace `FinetuneVAE`** with modular trainer:
   ```python
   # OLD (line 154-168):
   model = FinetuneVAE(vae_config=vae_config, ...)

   # NEW:
   # 1. Instantiate model
   vae_model = instantiate_from_config(cfg.model)

   # 2. Load pretrained weights (if specified)
   if cfg.get("pretrained_weights"):
       vae_weights = get_vae_weights(cfg.pretrained_weights)
       vae_model.load_state_dict(vae_weights, strict=False)

   # 3. Instantiate trainer module
   trainer_module = instantiate_from_config(cfg.trainer)
   trainer_module.model = vae_model
   trainer_module.kl_weight = cfg.training.kl_weight
   # ... set other training params
   ```

4. **Update config loading**:
   - Remove hardcoded `./vae_config.yaml` load
   - Use Hydra config exclusively
   - Support both old-style (for compatibility) and new-style configs

5. **Simplify data loading**:
   - Remove `dataset_type` branching
   - Trust `cfg.data.target` to point to correct DataModule

**Expected Outcome**:
- Cleaner training script
- Full config-driven architecture
- Support for any model/dataset combination via config

---

### Step 6.2: Add Backward Compatibility (Optional)

**Objective**: Allow old configs to still work during transition.

**Action Items**:

1. **Add compatibility layer**:
   ```python
   # Detect old-style config
   if "dataset_type" in cfg.data:
       warnings.warn("Old-style config detected, consider migrating to new format")
       # Convert old config to new format on-the-fly
       cfg.data = convert_legacy_config(cfg.data)
   ```

2. **Create `convert_legacy_config()` helper**

**Expected Outcome**: Smooth migration path without breaking existing setups.

---

## Phase 7: Integration Testing

### Step 7.1: Test Each Model/Dataset Combination

**Objective**: End-to-end testing of all supported combinations.

**Action Items**:

1. **Test Matrix**:
   | Model | Dataset | Plucker | Expected Trainer |
   |-------|---------|---------|------------------|
   | AutoencoderKL | CO3D | No | VanillaVAETrainer |
   | PluckerAutoencoder | CO3D | Yes | PluckerVAETrainer |
   | EQVAEAutoencoder | OmniObject | No | EQVAETrainer |
   | PluckerAutoencoder | OmniObject | Yes | PluckerVAETrainer |

2. **For each combination**:
   - Run 1-2 epochs to verify training loop
   - Check that losses are computed correctly
   - Verify image logging works
   - Ensure validation runs without errors

3. **Document results** in a test report

**Expected Outcome**: Verified working system for all use cases.

---

### Step 7.2: Performance Validation

**Objective**: Ensure no performance regression from refactoring.

**Action Items**:

1. **Benchmark data loading**:
   - Compare old vs new dataset loading times
   - Check memory usage

2. **Verify training metrics**:
   - Ensure loss curves match old implementation
   - Check that model converges similarly

**Expected Outcome**: Confidence that refactoring didn't break functionality.

---

## Key Decision Points

### Decision 1: Paired Data Return Format (OmniObject)

**Context**: OmniObject dataset can return view pairs for EQ-VAE training.

**Options**:
- **Option A**: Return only single view, use `PairedDatasetMixin` to sample pairs
- **Option B**: Return both views in single `__getitem__` (current approach)

**Recommendation**: Check how `EQVAETrainer.training_step()` expects data, then decide.

---

### Decision 2: Legacy Dataset Cleanup

**Context**: Old `src/dataset/` files will be superseded by new `src/data/` implementations.

**Options**:
- **Option A**: Delete old files immediately after migration
- **Option B**: Keep as deprecated with warnings for 1-2 releases
- **Option C**: Keep indefinitely for reference

**Recommendation**: Option B for safety, remove in future cleanup pass.

---

### Decision 3: Config Migration Strategy

**Context**: Existing configs use different structure than new modular configs.

**Options**:
- **Option A**: Hard break, require all configs to be updated
- **Option B**: Add compatibility layer in `train.py`
- **Option C**: Support both formats indefinitely

**Recommendation**: Option B for smooth transition.

---

## Success Criteria

✅ Phase 3 Complete When:
- [ ] `src/data/co3d_dataset.py` exists and passes tests
- [ ] `src/data/omniobject3d_dataset.py` exists and passes tests
- [ ] Dataset factory can instantiate both datasets
- [ ] All datasets inherit from `BaseVAEDataset`
- [ ] Return dictionaries use standardized keys

✅ Phase 4 Complete When:
- [ ] Config files exist for all model/dataset combinations
- [ ] Configs follow consistent structure
- [ ] All config paths are valid (no import errors)

✅ Phase 5 Complete When:
- [ ] Unit tests written for both datasets
- [ ] All tests pass
- [ ] Test coverage > 80% for dataset code

✅ Phase 6 Complete When:
- [ ] `train.py` uses config-based instantiation
- [ ] No hardcoded model/dataset selection
- [ ] Backward compatibility maintained (if desired)
- [ ] Training runs successfully with new configs

✅ Phase 7 Complete When:
- [ ] All model/dataset combinations tested
- [ ] No performance regression observed
- [ ] Documentation updated
- [ ] Migration guide written (if needed)

---

## Estimated Complexity

| Phase | Complexity | Time Estimate | Risk |
|-------|-----------|---------------|------|
| Phase 3 | Medium | 4-6 hours | Low - Clear requirements |
| Phase 4 | Low | 2-3 hours | Low - Template-based |
| Phase 5 | Medium | 3-4 hours | Medium - Depends on test data |
| Phase 6 | High | 4-6 hours | Medium - Integration complexity |
| Phase 7 | Medium | 3-4 hours | Low - Validation only |

**Total**: ~16-23 hours of development work

---

## Notes and Considerations

1. **Import Path Fix**: There's an inconsistency where trainers import from `src.trainers` but directory is `src.trainer` (singular). Fix this.

2. **Plucker Key Naming**: Legacy datasets use `"pluck_ray"`, new standard is `"plucker_coords"`. Ensure PluckerAutoencoder model expects correct key.

3. **Camera Parameter Structure**: Standardize camera dict structure across both datasets.

4. **EMA Handling**: BaseVAETrainer already handles EMA - ensure configs set correct `ema_decay` values.

5. **Discriminator Loss**: PluckerVAETrainer inherits discriminator handling from base - verify it works.

6. **Multi-GPU**: Test with DDP to ensure data loading works correctly in distributed setting.

---

## Next Actions

1. Review this plan with team
2. Clarify any ambiguous decision points
3. Begin Phase 3 implementation
4. Create tracking issues for each phase (optional)
