# https://github.com/Leminhbinh0209/FinetuneVAE-SD/tree/main
import os
import warnings
from pathlib import Path

import hydra
import torch
from coolname import generate_slug
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from ldm.util import instantiate_from_config

# Legacy imports for backward compatibility
from data_process.omniobject_dataset import OmniObjectDataModule
from src.dataset.co3d import Co3DDataModule

torch.cuda.empty_cache()


class ImageLoggerCallback(Callback):
    """Periodically logs input/reconstruction image grids to wandb."""

    def __init__(self, every_n_steps=500, max_images=4):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.max_images = max_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps != 0 or trainer.global_step == 0:
            return
        if trainer.logger is None:
            return

        self._log(trainer, pl_module, batch, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0:
            return
        if trainer.logger is None:
            return

        self._log(trainer, pl_module, batch, "val")

    @torch.no_grad()
    def _log(self, trainer, pl_module, batch, split):
        images = pl_module.log_images(batch)
        for key, img_tensor in images.items():
            img_tensor = img_tensor[:self.max_images].detach().cpu()
            # Clamp to [0, 1] for logging (model outputs may be in [-1, 1])
            img_tensor = (img_tensor + 1.0) / 2.0
            img_tensor = img_tensor.clamp(0, 1)
            trainer.logger.experiment.log(
                {f"{split}/{key}": [wandb.Image(img) for img in img_tensor]},
                step=trainer.global_step,
            )



def get_vae_weights(input_path):
    pretrained_weights = torch.load(input_path, weights_only=False)
    if "state_dict" in pretrained_weights:
        pretrained_weights = pretrained_weights["state_dict"]
    vae_weight = {}

    for k in pretrained_weights.keys():
        if "first_stage_model" in k:
            vae_weight[k.replace("first_stage_model.", "")] = pretrained_weights[k]

    return vae_weight


def get_device_config(force_single_gpu: bool = False, training_device: int = 0):
    """Get device configuration for training.

    Args:
        force_single_gpu: If True, use only one GPU even if multiple are available.
                         This is useful for model parallelism where RoMaV2 runs on
                         a different GPU than the VAE training.
        training_device: Which GPU to use for training (default: 0).
    """
    n_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    devices = torch.cuda.device_count()

    # Use DDP strategy with find_unused_parameters for multi-GPU training
    # This is needed because EMA buffers are not used in forward pass
    if devices > 1 and not force_single_gpu:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        # Single GPU mode - use specified device
        devices = [training_device] if torch.cuda.is_available() else 1
        strategy = "auto"

    return n_gpus, devices, strategy


def is_legacy_config(cfg: DictConfig) -> bool:
    """Check if config uses old-style dataset_type specification."""
    return "dataset_type" in cfg.data and "target" not in cfg.data


def setup_legacy_data_module(cfg: DictConfig):
    """Create data module using old-style config (for backward compatibility)."""
    dataset_type = cfg.data.get("dataset_type", "co3d")
    warnings.warn(
        f"Using legacy config format with dataset_type='{dataset_type}'. "
        "Consider migrating to new modular format with data.target",
        DeprecationWarning
    )

    if dataset_type == "co3d":
        return Co3DDataModule(
            co3d_dir=cfg.data.co3d_dir,
            bb_file=cfg.data.bb_file,
            batch_size=cfg.training.batch_size,
            val_size=cfg.training.val_split,
            size=cfg.training.image_size,
            apply_augmentation=cfg.data.get("apply_augmentation", False),
            crop_images=cfg.data.get("crop_images", False),
            patch_num=cfg.data.get("patch_num", None),
        )
    elif dataset_type == "omniobject":
        return OmniObjectDataModule(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.training.batch_size,
            val_size=cfg.training.val_split,
            size=cfg.training.image_size,
            patch_num=cfg.data.get("patch_num", None),
            pair_sampling=cfg.data.get("pair_sampling", "sequential"),
        )
    else:
        raise NotImplementedError(f"Unknown dataset type: {dataset_type}")


def setup_trainer_module(cfg: DictConfig, log_dir: str, use_wandb: bool):
    """Create trainer module using new modular architecture or legacy FinetuneVAE."""

    # Check if using new modular trainer config
    if "trainer" in cfg and "target" in cfg.trainer:
        print(f"[INFO] Using modular trainer: {cfg.trainer.target}")

        # Get trainer class
        from ldm.util import get_obj_from_str
        trainer_class = get_obj_from_str(cfg.trainer.target)

        # Instantiate trainer with model_config (trainer will instantiate model internally)
        # Pass trainer-specific params from config (e.g., warp_consistency_weight, warmup_steps)
        trainer_params = OmegaConf.to_container(cfg.trainer.get("params", {}), resolve=True)
        trainer_module = trainer_class(
            model_config=cfg.model,
            learning_rate=cfg.training.lr,
            ema_decay=cfg.training.get("ema_decay", 0.9999),
            image_key="image",
            **trainer_params,
        )

        # Set Plucker weights if applicable (for PluckerVAETrainer)
        if hasattr(trainer_module, 'plucker_loss_weight'):
            trainer_module.plucker_loss_weight = cfg.training.get("plucker_loss_weight", 0.1)
            trainer_module.plucker_weights = {
                "recon": cfg.training.get("plucker_recon_weight", 1.0),
                "constraint": cfg.training.get("plucker_constraint_weight", 0.1),
                "norm": cfg.training.get("plucker_norm_weight", 0.1),
            }

        return trainer_module

    else:
        # Legacy mode: use FinetuneVAE
        warnings.warn(
            "No trainer.target specified in config. Using legacy FinetuneVAE. "
            "Consider adding 'trainer.target' to your config.",
            DeprecationWarning
        )

        # Import here to avoid issues if FinetuneVAE is removed later
        from src.trainer.finetune_vae import FinetuneVAE

        plucker_weights = {
            "recon": cfg.training.get("plucker_recon_weight", 1.0),
            "constraint": cfg.training.get("plucker_constraint_weight", 0.1),
            "norm": cfg.training.get("plucker_norm_weight", 0.1),
        }

        return FinetuneVAE(
            vae_config=cfg.model,
            vae_weights=None,  # Will load separately
            kl_weight=cfg.training.kl_weight,
            lpips_loss_weight=cfg.training.lpips_loss_weight,
            plucker_loss_weight=cfg.training.get("plucker_loss_weight", 0.1),
            lr=cfg.training.lr,
            disc_lr=cfg.training.get("disc_lr", cfg.training.lr),
            optim=cfg.training.get("optim", "adam"),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            log_dir=log_dir,
            ema_decay=cfg.training.ema_decay,
            use_wandb=use_wandb,
            plucker_weights=plucker_weights,
        )


@hydra.main(version_base=None, config_path="config", config_name="vanilla_vae_co3d")
def main(cfg: DictConfig):
    print(f"[INFO] Starting training with config:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize wandb
    use_wandb = cfg.get("wandb", {}).get("enabled", True)
    wandb_logger = None

    run_name = generate_slug()

    if use_wandb:
        if hasattr(cfg.training, "note") and cfg.training.note:
            run_name += f"_{cfg.training.note}"

        wandb_logger = WandbLogger(
            project=cfg.get("wandb", {}).get("project", "vae-finetuning"),
            entity=cfg.get("wandb", {}).get("entity", None),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.get("wandb", {}).get("tags", ["vae", "modular-training"]),
        )
        print(f"[INFO] Initialized wandb logger with project: {wandb_logger.experiment.project}")
    else:
        print("[INFO] Wandb logging disabled")

    # Get device configuration
    # Check if model parallelism is enabled (romav2_device specified)
    force_single_gpu = False
    training_device = 0
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'params'):
        dataset_params = cfg.data.params.get('dataset_config', {}).get('params', {})
        romav2_device = dataset_params.get('romav2_device', None)
        if romav2_device:
            # Model parallelism: VAE on one GPU, RoMaV2 on another
            force_single_gpu = True
            training_device = cfg.training.get('training_device', 0)
            print(f"[INFO] Model parallelism: VAE on cuda:{training_device}, RoMaV2 on {romav2_device}")

    n_gpus, devices, strategy = get_device_config(force_single_gpu, training_device)
    print(f"[INFO] Using devices: {devices}, Strategy: {strategy}")

    # Create log directory
    log_dir = f"{cfg.training.output_dir}/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Log directory: {log_dir}")

    # Instantiate data module
    if is_legacy_config(cfg):
        data_module = setup_legacy_data_module(cfg)
    else:
        print(f"[INFO] Instantiating data module: {cfg.data.target}")
        data_module = instantiate_from_config(cfg.data)

    # Instantiate trainer module (Lightning module)
    # Note: Trainer instantiates the model internally from cfg.model
    trainer_module = setup_trainer_module(cfg, log_dir, use_wandb)

    # Load pretrained weights if specified
    if "pretrained_weights" in cfg and cfg.pretrained_weights:
        print(f"[INFO] Loading pretrained weights from {cfg.pretrained_weights}")
        vae_weights = get_vae_weights(cfg.pretrained_weights)
        trainer_module.model.load_state_dict(vae_weights, strict=False)
        print(f"[INFO] Loaded {len(vae_weights)} pretrained parameters")

    # Create callbacks
    checkpoint_callbacks = []

    checkpoint_dir = f"checkpoints/{run_name}"

    every_n_steps = cfg.training.get("checkpoint_every_n_steps", 0)
    if every_n_steps > 0:
        checkpoint_callbacks.append(ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="vae-step{step:06d}",
            save_top_k=-1,
            every_n_train_steps=every_n_steps,
            save_last=True,
        ))
    else:
        checkpoint_callbacks.append(ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="vae-epoch{epoch:03d}",
            save_top_k=-1,
            every_n_epochs=cfg.training.get("checkpoint_every_n_epochs", 5),
            save_last=True,
        ))

    # Add image logging callback
    if use_wandb:
        log_img_every = cfg.training.get("log_images_every_n_steps", 5000)
        checkpoint_callbacks.append(ImageLoggerCallback(every_n_steps=log_img_every))

    # Create PyTorch Lightning Trainer
    # Note: accumulate_grad_batches is NOT used here because we use manual optimization
    # Gradient accumulation is handled manually in training_step
    pl_trainer = Trainer(
        max_epochs=cfg.training.num_epochs,
        precision=cfg.training.precision,
        strategy=strategy,
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=cfg.training.get("gradient_clip_val", None),
        logger=wandb_logger if use_wandb else None,
        log_every_n_steps=cfg.training.get("log_every_n_steps", 50),
        check_val_every_n_epoch=cfg.training.get("check_val_every_n_epoch", 1),
        limit_train_batches=cfg.training.get("limit_train_batches", 1.0),
        callbacks=checkpoint_callbacks,
    )

    # Train (optionally resume from checkpoint)
    resume_ckpt = cfg.training.get("resume_from_checkpoint", None)
    if resume_ckpt:
        print(f"[INFO] Resuming from checkpoint: {resume_ckpt}")
    print(f"[INFO] Starting training for {cfg.training.num_epochs} epochs")
    pl_trainer.fit(trainer_module, datamodule=data_module, ckpt_path=resume_ckpt)

    # Save final model
    final_model_path = f"{log_dir}/last_model.pth"
    print(f"[INFO] Saving model to {final_model_path}")
    torch.save(trainer_module.model.state_dict(), final_model_path)
    print("[INFO] Model saved successfully")

    # Log artifact to wandb
    if use_wandb and wandb_logger:
        artifact = wandb.Artifact(
            name="vae_model_final",
            type="model",
            description=f"Final VAE model after {cfg.training.num_epochs} epochs",
        )
        artifact.add_file(final_model_path)
        wandb_logger.experiment.log_artifact(artifact)
        print("[INFO] Model artifact logged to wandb")
        wandb.finish()


if __name__ == "__main__":
    main()
