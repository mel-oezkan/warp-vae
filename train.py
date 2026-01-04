# https://github.com/Leminhbinh0209/FinetuneVAE-SD/tree/main
import gzip
import json
import os
from contextlib import contextmanager   
from dataclasses import dataclass
from pathlib import Path
from typing import IO, List, Optional, cast

import hydra
import lpips
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from coolname import generate_slug
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, random_split
from torchvision import transforms

import wandb
from data_process.co3d_dataset import jitter_bbox, square_bbox
from data_process.omniobject_dataset import OmniObjectDataModule
from data_process.plucker import compute_directions_from_sample, ray_to_plucker
from ldm.models.autoencoder import AutoencoderKL, PluckerAutoencoder, EQVAEAutoencoder
from ldm.modules.ema import LitEma
from ldm.util import instantiate_from_config

from src.dataset.co3d import Co3DDataModule

torch.cuda.empty_cache()


def get_vae_weights(input_path):
    pretrained_weights = torch.load(input_path, weights_only=False)
    if "state_dict" in pretrained_weights:
        pretrained_weights = pretrained_weights["state_dict"]
    vae_weight = {}

    for k in pretrained_weights.keys():
        if "first_stage_model" in k:
            vae_weight[k.replace("first_stage_model.", "")] = pretrained_weights[k]

    return vae_weight


def get_device_config():
    n_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    devices = torch.cuda.device_count()
    strategy = "ddp" if n_gpus > 1 else "auto"
    return n_gpus, devices, strategy


@hydra.main(version_base=None, config_path="config", config_name="finetuneVAE")
def main(cfg: DictConfig):
    print(f"[DEBUG] Starting training with config: {OmegaConf.to_yaml(cfg)}")

    # Initialize wandb
    use_wandb = cfg.get("wandb", {}).get("enabled", True)
    wandb_logger = None

    run_name = generate_slug()

    if use_wandb:  
        run_name = run_name
        if hasattr(cfg.training, "note") and cfg.training.note:
            run_name += f"_{cfg.training.note}"

        wandb_logger = WandbLogger(
            project=cfg.get("wandb", {}).get("project", "vae-finetuning"),
            entity=cfg.get("wandb", {}).get("entity", None),
            name=run_name,
            config={
                "lr": cfg.training.lr,
                "batch_size": cfg.training.batch_size,
                "num_epochs": cfg.training.num_epochs,
                "image_size": cfg.training.image_size,
                "kl_weight": cfg.training.kl_weight,
                "lpips_loss_weight": cfg.training.lpips_loss_weight,
                "plucker_loss_weight": cfg.training.get("plucker_loss_weight", 0.1),
                "ema_decay": cfg.training.ema_decay,
                "val_size": cfg.training.val_size,
                "precision": cfg.training.precision,
                "data_dir": cfg.data.get("data_dir", cfg.data.get("co3d_dir", "")),
            },
            tags=cfg.get("wandb", {}).get(
                "tags", ["vae", "finetuning", "stable-diffusion", "plucker"]
            ),
        )
        print(
            f"[DEBUG] Initialized wandb logger with project: {wandb_logger.experiment.project}"
        )
    else:
        print("[DEBUG] Wandb logging disabled")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus, devices, strategy = get_device_config()
    print(f"[DEBUG] Using device: {device}, GPUs: {n_gpus}, Strategy: {strategy}")

    file_names = f"size_({cfg.training.image_size})_val({cfg.training.val_size})_ema({cfg.training.ema_decay})_bs({cfg.training.batch_size})_lr({cfg.training.lr})_epochs({cfg.training.num_epochs})_kl({cfg.training.kl_weight})_lpips({cfg.training.lpips_loss_weight})_{cfg.training.note}"
    log_dir = f"{cfg.training.output_dir}/{file_names}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[DEBUG] Log directory: {log_dir}")

    config = OmegaConf.load("./vae_config.yaml")
    vae_config = config.model
    print("[DEBUG] Loaded VAE config from ./vae_config.yaml")

    input_path = "./sd_model/v1-5-pruned.ckpt"
    vae_weight = get_vae_weights(input_path)
    print(
        f"[DEBUG] Loaded VAE weights from {input_path}, found {len(vae_weight)} parameters"
    )

    # Select dataset based on config
    dataset_type = cfg.data.get("dataset_type", "co3d")
    print(f"[DEBUG] Using dataset type: {dataset_type}")

    if dataset_type == "co3d":
        data_module = Co3DDataModule(
            co3d_dir=cfg.data.co3d_dir,
            bb_file=cfg.data.bb_file,
            batch_size=cfg.training.batch_size,
            val_size=cfg.training.val_size,
            size=cfg.training.image_size,
            apply_augmentation=cfg.data.get("apply_augmentation", False),
            crop_images=cfg.data.get("crop_images", False),
            patch_num=cfg.data.get("patch_num", None),
        )
    elif dataset_type == "omniobject":
        data_module = OmniObjectDataModule(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.training.batch_size,
            val_size=cfg.training.get("val_size", 0.1),
            size=cfg.training.image_size,
            patch_num=cfg.data.get("patch_num", None),
            pair_sampling=cfg.data.get("pair_sampling", "sequential"),
        )
    else:
        raise NotImplementedError(f"Unknown dataset type: {dataset_type}")

    # Plücker loss weights from config
    plucker_weights = {
        "recon": cfg.training.get("plucker_recon_weight", 1.0),
        "constraint": cfg.training.get("plucker_constraint_weight", 0.1),
        "norm": cfg.training.get("plucker_norm_weight", 0.1),
    }

    model = FinetuneVAE(
        vae_config=vae_config,
        vae_weights=vae_weight,
        kl_weight=cfg.training.kl_weight,
        lpips_loss_weight=cfg.training.lpips_loss_weight,
        plucker_loss_weight=cfg.training.get("plucker_loss_weight", 0.1),
        lr=cfg.training.lr,
        disc_lr=cfg.training.get("disc_lr", cfg.training.lr),
        optim=cfg.training.get("optim", "adam"),
        device=device,
        log_dir=log_dir,
        ema_decay=cfg.training.ema_decay,
        use_wandb=use_wandb,
        plucker_weights=plucker_weights,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints" + "/" + run_name,
        filename="vae-epoch{epoch:03d}",
        save_top_k=-1,  # save all checkpoints
        every_n_epochs=5,  # 🔑 save every 5 epochs
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=cfg.training.num_epochs,
        precision=cfg.training.precision,
        strategy=strategy,
        devices=devices,
        accelerator="gpu",
        accumulate_grad_batches=1,
        logger=wandb_logger if use_wandb else None,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
    )

    print(f"[DEBUG] Starting training for {cfg.training.num_epochs} epochs")
    trainer.fit(model, datamodule=data_module)

    print(f"[DEBUG] Training completed, saving model to {log_dir}/last_model.pth")
    torch.save(model.model.state_dict(), f"{log_dir}/last_model.pth")
    print("[DEBUG] Model saved successfully")

    if use_wandb and wandb_logger:
        artifact = wandb.Artifact(
            name="vae_model_final",
            type="model",
            description=f"Final VAE model after {cfg.training.num_epochs} epochs",
        )
        artifact.add_file(f"{log_dir}/last_model.pth")
        wandb_logger.experiment.log_artifact(artifact)
        print("[DEBUG] Model artifact logged to wandb")

        wandb.finish()


if __name__ == "__main__":
    main()
