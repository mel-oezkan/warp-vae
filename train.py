# https://github.com/Leminhbinh0209/FinetuneVAE-SD/tree/main
import os
from contextlib import contextmanager
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import lpips
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from ldm.modules.ema import LitEma
from ldm.util import instantiate_from_config

from typing import Optional, List
import gzip
from data_process.co3d_dataset import jitter_bbox, square_bbox
from dataclasses import dataclass
from pathlib import Path

from data_process.plucker import plucker_encodeing, plucker_to_rays, simple_rays
from typing import cast, IO
import json

torch.cuda.empty_cache()


def sign_invariant_l1(pred, gt):
    pos = torch.abs(pred - gt).mean(dim=-1)
    neg = torch.abs(pred + gt).mean(dim=-1)
    return torch.minimum(pos, neg).mean()


@dataclass
class Co3DSample:
    """Dataclass representing a single CO3D sample."""

    filepath: str
    R: List[List[float]]
    T: List[float]
    focal_length: List[float]
    principal_point: List[float]
    bbox: List[int]


class ProcessedCo3D(Dataset):
    def __init__(
        self,
        co3d_dir: str,
        bb_file: str,
        apply_augmentation: bool = False,
        transform: Optional[transforms.Compose] = None,
        patch_num: Optional[int] = None,
        crop_images: bool = False,
        device: Optional[str] = None,
    ):
        """Dataset for the preprocessed CO3D data.

        Args:
            co3d_dir (str): Path to the image directory.
            bb_file (str): Path to the single object bbox file.
            apply_augmentation (bool, optional): Determines if jitter augs should be applied to bboxes. Defaults to False.
            transform (Optional[transforms.Compose], optional): Transform obj containing image trans. Defaults to Norm+See Code.
            patch_num (Optional[int], optional): Number of patches for one dim. Final plücker dim will be (n_patches, n_patches). Defaults to None (use only one).
        """
        super().__init__()
        self.co3d_dir = Path(co3d_dir)
        self.bbox_file = bb_file
        self.crop_images = crop_images
        self.device = device
        self.patch_num = patch_num

        self.samples = []
        with gzip.GzipFile(self.bbox_file, "rb") as f:
            obj_dict = json.loads(cast(IO, f).read().decode("utf8"))

        for subdir in obj_dict.keys():
            for sample in obj_dict[subdir]:
                self.samples.append(sample)

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((512, 512), antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        self.apply_augmentation = apply_augmentation
        if self.apply_augmentation:
            self.jitter_scale = (1.1, 1.2)
            self.jitter_trans = (-0.07, 0.07)
        else:
            self.jitter_scale = (1, 1)
            self.jitter_trans = (0.0, 0.0)

    def _crop_image(self, image, bbox):
        image_crop = transforms.functional.crop(
            image,
            top=bbox[1],
            left=bbox[0],
            height=bbox[3] - bbox[1],
            width=bbox[2] - bbox[0],
        )
        return image_crop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # convert to torch tensor
        source_image = Image.open(self.co3d_dir / sample["filepath"])
        image = transforms.ToTensor()(source_image)  # c w h
        orig_h, orig_w = image.size()[1:]

        bbox = sample["bbox"]
        bbox_init = bbox if self.crop_images else [0, 0, orig_w, orig_h]
        bbox = square_bbox(np.array(bbox_init))

        if self.apply_augmentation:
            bbox = jitter_bbox(
                bbox,
                jitter_scale=self.jitter_scale,
                jitter_trans=self.jitter_trans,
            )

        rounded_bb = np.around(bbox).astype(int)

        # crop handling
        crop_center = (bbox[:2] + bbox[2:]) / 2
        max_dimension = max(orig_w, orig_h)
        crop_center_adjusted = (
            crop_center + (max_dimension - np.array([orig_w, orig_h])) / 2
        )

        scale_fact = max_dimension / min(orig_w, orig_h)
        ndc_center = scale_fact - 2 * scale_fact * crop_center_adjusted / max_dimension
        crop_width_ndc = (
            2 * scale_fact * (rounded_bb[2] - rounded_bb[0]) / max_dimension
        )
        crop_params = torch.tensor(
            [-ndc_center[0], -ndc_center[1], crop_width_ndc, scale_fact],
            dtype=torch.float32,
        )

        # crop and normalize the image
        image_cropped = self._crop_image(image=source_image, bbox=rounded_bb)
        image_cropped = self.transform(image_cropped)
        cropped_size = (image_cropped.shape[1], image_cropped.shape[2])

        rot = torch.tensor(sample["R"], dtype=torch.float32)
        trans = torch.tensor(sample["T"], dtype=torch.float32)
        focal_length = torch.tensor(sample["focal_length"], dtype=torch.float32)
        principle_point = torch.tensor(sample["principal_point"], dtype=torch.float32)

        plucker = plucker_encodeing(
            R=rot,
            T=trans,
            fl=focal_length,
            pp=principle_point,
            crop_params=crop_params,
            original_size=(orig_h, orig_w),
            cropped_size=cropped_size,
            device=self.device,
            patch_num=self.patch_num,
        )

        rays = simple_rays(plucker[..., :3], trans)

        # Don't return PIL Image - only return tensors
        return {
            "image": image_cropped,
            "rays": rays,
            "crop_params": crop_params,
            "R": rot,
            "T": trans,
            "focal_length": focal_length,
            "principal_point": principle_point,
        }


class Co3DDataModule(pl.LightningDataModule):
    """DataModule for CO3D dataset."""

    def __init__(
        self,
        co3d_dir: str,
        bb_file: str,
        batch_size: int = 2,
        val_size: float = 0.1,
        size: int = 384,
        apply_augmentation: bool = False,
        crop_images: bool = False,
        patch_num: Optional[int] = None,
    ):
        super().__init__()
        self.co3d_dir = co3d_dir
        self.bb_file = bb_file
        self.batch_size = batch_size
        self.val_size = val_size
        self.size = size
        self.apply_augmentation = apply_augmentation
        self.crop_images = crop_images
        self.patch_num = patch_num
        self.setup("fit")
        print(
            f"[DEBUG] Co3DDataModule initialized with co3d_dir={co3d_dir}, batch_size={batch_size}, val_size={val_size}, size={size}"
        )

    def setup(self, stage):
        transform = transforms.Compose(
            [
                transforms.Resize((self.size, self.size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        full_ds = ProcessedCo3D(
            co3d_dir=self.co3d_dir,
            bb_file=self.bb_file,
            apply_augmentation=self.apply_augmentation,
            transform=transform,
            crop_images=self.crop_images,
            patch_num=self.patch_num,
        )

        train_size = int((1 - self.val_size) * len(full_ds))
        val_size = len(full_ds) - train_size
        self.train_ds, self.val_ds = random_split(full_ds, [train_size, val_size])

        print(
            f"[DEBUG] Co3DDataModule setup complete - Train size: {len(self.train_ds)}, Val size: {len(self.val_ds)}"
        )

    def train_dataloader(self):
        print(
            f"[DEBUG] Creating CO3D train dataloader with batch_size={self.batch_size}"
        )
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        print(
            f"[DEBUG] Creating CO3D validation dataloader with batch_size={self.batch_size}"
        )
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


class FinetuneVAE(pl.LightningModule):
    def __init__(
        self,
        kl_weight=0.1,
        lpips_loss_weight=0.1,
        plucker_loss_weight=0.1,
        lr=1e-4,
        disc_lr=1e-4,
        momentum=0.9,
        weight_decay=5e-4,
        optim="adam",
        vae_config=None,
        vae_weights=None,
        device=torch.device("cuda"),
        ema_decay=0.999,
        precision=32,
        log_dir=None,
        use_wandb=True,
        plucker_weights=None,
        freeze_decoder=False,
    ):
        super().__init__()
        print(
            f"[DEBUG] FinetuneVAE initializing with lr={lr}, kl_weight={kl_weight}, lpips_weight={lpips_loss_weight}"
        )
        self.kl_weight = kl_weight
        self.lpips_loss_weight = lpips_loss_weight
        self.plucker_loss_weight = plucker_loss_weight
        self.lpips_loss_fn = lpips.LPIPS(net="alex").to(device)
        self.lpips_loss_fn.eval()

        for param in self.lpips_loss_fn.parameters():
            param.requires_grad = False

        self.lr = lr
        self.disc_lr = disc_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optim = optim
        self.model = instantiate_from_config(vae_config)

        # Load pretrained weights with strict=False to allow new layers
        if vae_weights is not None:
            missing, unexpected = self.model.load_state_dict(vae_weights, strict=False)
            print(
                f"[DEBUG] Loaded VAE weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
            if missing:
                print(
                    f"[DEBUG] Missing keys (new layers): {missing[:10]}..."
                )  # Print first 10

        self.model.train()
        self.precision = precision
        self.log_dir = log_dir
        self.log_one_batch = False
        self.use_ema = ema_decay > 0

        # --- NEW: Freeze Decoder Logic ---
        self.freeze_decoder = freeze_decoder
        if self.freeze_decoder:
            print("--> Freezing VAE decoder parameters.")
            # Standard LDM VAE structure usually has 'decoder' and 'post_quant_conv'
            if hasattr(self.model, "decoder"):
                for param in self.model.decoder.parameters():
                    param.requires_grad = False

        # Plücker loss weights
        self.plucker_weights = plucker_weights or {
            "recon": 1.0,
            "constraint": 0.1,
            "norm": 0.1,
        }
        # Assign to model as well for hybrid_plucker_loss
        self.model.plucker_weights = self.plucker_weights

        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0.0 < ema_decay < 1.0
            self.model_ema = LitEma(self.model, decay=ema_decay)

        self.use_wandb = use_wandb

        # Enable automatic optimization to be False for manual optimizer stepping
        self.automatic_optimization = True
        self.validation_step_outputs = []

    def configure_optimizers(self):
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if len(params) == 0:
            print("[WARNING] No parameters to optimize! Check freezing logic.")

        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        return opt

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = self.trainer.datamodule.train_ds
            self.val_ds = self.trainer.datamodule.val_ds

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        target = batch["image"]

        if self.precision == 16:
            target = target.half()

        posterior, rays = self.model.encode(target)
        z = posterior.sample()

        pred = self.model.decode(z)

        # 2. calucalte loss
        rec_loss = torch.abs(target - pred)
        if self.current_epoch < self.trainer.max_epochs // 3 * 2:
            rec_loss = rec_loss.mean() * rec_loss.size(1)
        else:
            rec_loss = rec_loss.pow(2).mean() * rec_loss.size(1)

        with torch.no_grad():
            lpips_loss = self.lpips_loss_fn(pred, target).mean()

        pluck_loss = self.hybrid_plucker_loss(batch["pluck"], rays)
        kl_loss = posterior.kl().mean() * self.kl_weight

        # 3. aggregate loss
        loss = (
            rec_loss
            + (self.lpips_loss_weight * lpips_loss)
            + (self.plucker_loss_weight * pluck_loss)
            + (self.kl_weight * kl_loss)
        )

        # 4. Logging
        self.log_dict(
            {
                "train/rec_loss": rec_loss,
                "train/lpips_loss": lpips_loss,
                "train/ray_loss": self.plucker_loss_weight * pluck_loss,
                "train/kl_loss": kl_loss,
                "train/total_loss": loss,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def hybrid_plucker_loss(self, pred, gt):
        """
        Compute hybrid Plücker loss with reconstruction, constraint, and normalization terms.

        Args:
            pred: Predicted Plücker coordinates (B, n_patches*n_patches, 6)
            gt: Ground truth Plücker coordinates (B, n_patches*n_patches, 6)
        """
        pred_d, pred_m = pred[..., :3], pred[..., 3:]
        gt_d, gt_m = gt[..., :3], gt[..., 3:]

        # Simple reconstruction loss (with sign ambiguity handling)
        loss_d = sign_invariant_l1(pred_d, gt_d)
        loss_m = sign_invariant_l1(pred_m, gt_m)

        recon_loss = loss_d + loss_m

        # Constraint: d·m = 0
        constraint_loss = torch.mean((pred_d * pred_m).sum(dim=-1) ** 2)

        # Encourage unit direction vectors
        norm_loss = F.mse_loss(
            torch.norm(pred_d, dim=-1), torch.ones_like(torch.norm(pred_d, dim=-1))
        )

        return (
            self.plucker_weights["recon"] * recon_loss
            + self.plucker_weights["constraint"] * constraint_loss
            + self.plucker_weights["norm"] * norm_loss
        )

    def validation_step(self, batch, batch_idx):
        # Handle both dataset formats
        target = batch["image"]
        gt_plucker = batch.get("pluck", None)
        name = torch.arange(target.shape[0])

        if self.precision == 16:
            target = target.half()

        # Forward pass
        pred, posterior, pred_plucker = self.model(target, sample_posterior=False)

        rec_loss = torch.abs(target - pred).mean()
        lpips_loss = self.lpips_loss_fn(pred, target).mean()
        kl_loss = posterior.kl().mean()

        plucker_loss = torch.tensor(0.0, device=target.device)
        if gt_plucker is not None and pred_plucker is not None:
            plucker_loss = self.hybrid_plucker_loss(gt_plucker, pred_plucker)

        loss = (
            rec_loss
            + self.lpips_loss_weight * lpips_loss
            + self.plucker_loss_weight * plucker_loss
            + self.kl_weight * kl_loss
        )

        output = {
            "val_loss": loss.detach(),
            "rec_loss": rec_loss.detach(),
            "plucker_loss": plucker_loss.detach(),
        }

        self.validation_step_outputs.append(output)

        # Trigger image logging on first batch
        if batch_idx == 0 and self.use_wandb:
            self.log_images_wandb(
                target, pred, torch.zeros(target.shape[0])
            )  # Dummy labels

        return output

    def log_images(self, input, output, names):
        if self.log_one_batch:
            return
        print(
            f"[DEBUG] Logging images for epoch {self.current_epoch} - {len(names)} images"
        )

        # Limit to just a few samples
        max_log = min(4, len(names))
        for idx, (img1, img2, label) in enumerate(
            zip(input[:max_log], output[:max_log], names[:max_log])
        ):
            img1 = img1.cpu().detach().numpy().transpose(1, 2, 0)
            img2 = img2.cpu().detach().numpy().transpose(1, 2, 0)
            img1 = (img1 + 1) / 2
            img2 = (img2 + 1) / 2
            diff = abs(img1 - img2)
            img = np.concatenate([img1, img2, diff], axis=1)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            os.makedirs(self.log_dir + "/" + str(self.current_epoch), exist_ok=True)
            # Convert label tensor to int and create a proper filename
            label_id = label.item() if torch.is_tensor(label) else label
            filename = f"class_{label_id}_sample_{idx}.png"
            img.save(os.path.join(self.log_dir, str(self.current_epoch), filename))

        self.log_one_batch = True
        print(f"[DEBUG] Images saved to {self.log_dir}/{str(self.current_epoch)}")

    def log_images_wandb(self, input, output, labels, max_samples=4):
        """Log sample images to wandb."""
        if not self.use_wandb:
            return

        print(
            f"[DEBUG] Logging {min(max_samples, len(input))} images to wandb for epoch {self.current_epoch}"
        )

        images_to_log = []
        max_samples = min(max_samples, len(input))

        for i in range(max_samples):
            # Convert tensors to numpy and denormalize
            input_img = input[i].cpu().detach().numpy().transpose(1, 2, 0)
            output_img = output[i].cpu().detach().numpy().transpose(1, 2, 0)

            # Denormalize from [-1, 1] to [0, 1]
            input_img = (input_img + 1) / 2
            output_img = (output_img + 1) / 2

            # Clip to valid range
            input_img = np.clip(input_img, 0, 1)
            output_img = np.clip(output_img, 0, 1)

            # Create difference image
            diff_img = np.abs(input_img - output_img)

            # Create side-by-side comparison
            comparison = np.concatenate([input_img, output_img, diff_img], axis=1)

            label_id = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]

            images_to_log.append(
                wandb.Image(
                    comparison,
                    caption=f"Class {label_id} | Input | Reconstruction | Difference",
                )
            )

        self.logger.experiment.log(
            {f"validation_images_epoch_{self.current_epoch}": images_to_log}
        )

    def on_train_epoch_end(self):
        if self.use_ema:
            self.model_ema(self.model)

        if self.use_wandb and self.use_ema:
            self.log("ema_decay", self.ema_decay, on_epoch=True, logger=True)

        if self.current_epoch == self.trainer.max_epochs // 3 * 2:
            old_weight = self.lpips_loss_weight
            print(
                f"[DEBUG] Reducing LPIPS weight from {self.lpips_loss_weight} to {self.lpips_loss_weight * 0.1}"
            )
            self.lpips_loss_weight = self.lpips_loss_weight * 0.1

            if self.use_wandb:
                self.log(
                    "lpips/old_weight",
                    old_weight,
                    logger=True,
                )
                self.log(
                    "lpips/new_weight",
                    self.lpips_loss_weight,
                    logger=True,
                )
                self.log(
                    "lpips/epoch",
                    self.curret_epoch,
                    logger=True,
                )

    def on_validation_epoch_end(self):
        if self.use_ema:
            self.model_ema.restore(self.model.parameters())

        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_rec = torch.stack(
            [x["rec_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_pluck = torch.stack(
            [x["plucker_loss"] for x in self.validation_step_outputs]
        ).mean()

        self.log("val/loss", avg_loss, on_epoch=True, logger=True)
        self.log("val/rec_loss", avg_rec, on_epoch=True, logger=True)
        self.log("val/plucker_loss", avg_pluck, on_epoch=True, logger=True)

        print(f"[DEBUG] Validation Complete. Loss: {avg_loss:.4f}")
        self.validation_step_outputs.clear()


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

    if use_wandb:
        run_name = f"vae_finetune_lr{cfg.training.lr}_bs{cfg.training.batch_size}_ep{cfg.training.num_epochs}"
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
    else:
        raise NotImplementedError

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
        dirpath="checkpoints",
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
