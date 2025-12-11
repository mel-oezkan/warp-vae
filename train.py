#https://github.com/Leminhbinh0209/FinetuneVAE-SD/tree/main
import os
import random
from contextlib import contextmanager

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
from torchvision.datasets import ImageFolder

from ldm.modules.ema import LitEma
from ldm.util import instantiate_from_config

from typing import Optional, List
import gzip
from data_process.co3d_dataset import jitter_bbox, square_bbox
from dataclasses import dataclass
from pathlib import Path
from torchvision import transforms

from data_process.plucker import plucker_encodeing
from typing import cast, IO
import json

torch.cuda.empty_cache()



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
        bbox_init = bbox if self.crop_images else [0, 0, orig_h, orig_w]
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

        return {"source": source_image, "image": image_cropped, "pluck": plucker}



class Co3DDataModule(pl.LightningDataModule):
    """DataModule for CO3D dataset."""
    
    def __init__(
        self,
        co3d_dir: str,
        bb_file: str,
        batch_size: int = 64,
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
        print(f"[DEBUG] Co3DDataModule initialized with co3d_dir={co3d_dir}, batch_size={batch_size}, val_size={val_size}, size={size}")

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
        
        print(f"[DEBUG] Co3DDataModule setup complete - Train size: {len(self.train_ds)}, Val size: {len(self.val_ds)}")

    def train_dataloader(self):
        print(f"[DEBUG] Creating CO3D train dataloader with batch_size={self.batch_size}")
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        print(f"[DEBUG] Creating CO3D validation dataloader with batch_size={self.batch_size}")
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

class FinetuneFaceData(Dataset):
    """PyTorch Dataset for face finetuning with ImageFolder structure."""
    
    def __init__(
        self,
        data_dir: str,
        size: int = 384,
        class_filter: list = None,
        max_samples: int = None,
    ):
        """
        Args:
            data_dir: Path to the dataset folder with class subfolders
            size: Size to resize images to
            class_filter: Optional list of class indices to include
            max_samples: Optional max number of samples to load
        """
        super().__init__()
        self.data_dir = data_dir
        self.size = size
        self.class_filter = class_filter
        self.max_samples = max_samples
        
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        
        # Use ImageFolder to load dataset structure
        self.dataset = ImageFolder(data_dir)
        
        # Filter by class if specified
        if self.class_filter is not None:
            self.samples = [
                (path, label) for path, label in self.dataset.samples
                if label in self.class_filter
            ]
        else:
            self.samples = self.dataset.samples
        
        # Limit samples if specified
        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]
        
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
    
    def get_class_samples(self, class_id, max_samples=None):
        """Get all samples for a specific class."""
        class_samples = [(p, l) for p, l in self.samples if l == class_id]
        if max_samples:
            class_samples = class_samples[:max_samples]
        return class_samples
    


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, val_size=0.1, size=384):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.size = size
        self.setup("fit")
        print(f"[DEBUG] DataModule initialized with data_dir={data_dir}, batch_size={batch_size}, val_size={val_size}, size={size}")

    def setup(self, stage):
        all_images = sorted(
            [
                u
                for u in os.listdir(self.data_dir)
                if u.endswith(".png") or u.endswith(".jpg")
            ]
        )
        random.shuffle(all_images)
        train_size = int((1 - self.val_size) * len(all_images))
        full_ds = FinetuneFaceData(
            data_dir=self.data_dir,
            size=self.size,
            max_samples=10,
        )
        train_size = int(0.9 * len(full_ds))
        val_size = len(full_ds) - train_size
        self.train_ds, self.val_ds = random_split(full_ds, [train_size, val_size])
        
        print(f"Train size: {len(self.train_ds)}, Val size: {len(self.val_ds)}")
        print(f"[DEBUG] DataModule setup complete - Train size: {len(self.train_ds)}, Val size: {len(self.val_ds)}")

    def train_dataloader(self):
        print(f"[DEBUG] Creating train dataloader with batch_size={self.batch_size}")
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        print(f"[DEBUG] Creating validation dataloader with batch_size={self.batch_size}")
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4
        )



class FinetuneVAE(pl.LightningModule):
    def __init__(
        self,
        kl_weight=0.1,
        lpips_loss_weight=0.1,
        lr=1e-4,
        momentum=0.9,
        weight_decay=5e-4,
        optim="sgd",
        vae_config=None,
        vae_weights=None,
        device=torch.device("cuda"),
        ema_decay=0.999,
        precision=32,
        log_dir=None,
        use_wandb=True,
    ):
        super().__init__()
        print(f"[DEBUG] FinetuneVAE initializing with lr={lr}, kl_weight={kl_weight}, lpips_weight={lpips_loss_weight}")
        self.kl_weight = kl_weight
        self.lpips_loss_weight = lpips_loss_weight
        self.lpips_loss_fn = lpips.LPIPS(net="alex").to(device)
        self.lpips_loss_fn.eval()

        for param in self.lpips_loss_fn.parameters():
            param.requires_grad = False

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optim = optim
        self.model = instantiate_from_config(vae_config)
        self.model.load_state_dict(vae_weights, strict=True)
        self.model.train()
        self.precision = precision
        self.log_dir = log_dir
        self.log_one_batch = False
        self.use_ema = ema_decay > 0
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0.0 < ema_decay < 1.0
            self.model_ema = LitEma(self.model, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.use_wandb = use_wandb
        print(f"[DEBUG] FinetuneVAE initialization complete. Using EMA: {self.use_ema}, Using wandb: {self.use_wandb}")
        
        # Store validation outputs for epoch end aggregation
        self.validation_step_outputs = []

        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Assuming the DataModule is attached to the Trainer and accessible
            self.train_ds = self.trainer.datamodule.train_ds
            self.val_ds = self.trainer.datamodule.val_ds
            print("Warning: The setup method is called")

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
        # Handle both dataset formats
        if isinstance(batch, dict):
            # CO3D dataset format
            target = batch["image"]
            label = None
        else:
            # FinetuneFaceData format
            target, label = batch
            
        if batch_idx % 50 == 0:
            print(f"[DEBUG] Training step - Epoch: {self.current_epoch}, Batch: {batch_idx}, Target shape: {target.shape}")
        
        if self.precision == 16:
            target = target.half()

        posterior = self.model.encode(target)
        z = posterior.sample()
        pred = self.model.decode(z)

        rec_loss = torch.abs(target.contiguous() - pred.contiguous())
        if self.current_epoch < self.trainer.max_epochs // 3 * 2:
            rec_loss = rec_loss.mean() * rec_loss.size(1)
        else:
            rec_loss = rec_loss.pow(2).mean() * rec_loss.size(1)

        with torch.no_grad():
            lpips_loss = self.lpips_loss_fn(pred, target).mean()

        loss = rec_loss + self.lpips_loss_weight * lpips_loss
        
        if batch_idx % 50 == 0:
            print(f"[DEBUG] Training losses - Rec: {rec_loss:.4f}, LPIPS: {lpips_loss:.4f}, Total: {loss:.4f}")
        
        self.log("rec_loss", rec_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("lpips_loss", lpips_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if batch_idx % 100 == 0:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log("learning_rate", current_lr, on_step=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Handle both dataset formats
        if isinstance(batch, dict):
            # CO3D dataset format
            target = batch["image"]
            name = torch.arange(target.shape[0])  # Use indices as labels
        else:
            # FinetuneFaceData format
            target, name = batch
            
        if batch_idx == 0:
            print(f"[DEBUG] Validation step - Epoch: {self.current_epoch}, Batch: {batch_idx}, Target shape: {target.shape}")
        
        if self.precision == 16:
            target = target.half()

        posterior = self.model.encode(target)
        z = posterior.mode()
        pred = self.model.decode(z)

        rec_loss = torch.abs(target.contiguous() - pred.contiguous())
        rec_loss = rec_loss.mean()

        lpips_loss = self.lpips_loss_fn(pred, target).mean()
        loss = rec_loss + self.lpips_loss_weight * lpips_loss
        
        if batch_idx == 0:
            print(f"[DEBUG] Validation losses - Rec: {rec_loss:.4f}, LPIPS: {lpips_loss:.4f}, Total: {loss:.4f}")

        output = {"val_loss": loss.detach(), "rec_loss": rec_loss.detach(), "lpips_loss": lpips_loss.detach()}
        
        if batch_idx == 0 and self.use_wandb:
            self.log_images_wandb(target, pred, name)
        
        self.validation_step_outputs.append(output)
        
        del pred, name, target
        torch.cuda.empty_cache()

        return output


    def configure_optimizers(self):
        if self.optim == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError
        return optimizer

    
    def log_images(self, input, output, names):
        if self.log_one_batch:
            return
        print(f"[DEBUG] Logging images for epoch {self.current_epoch} - {len(names)} images")

        # Limit to just a few samples
        max_log = min(4, len(names))
        for idx, (img1, img2, label) in enumerate(zip(input[:max_log], output[:max_log], names[:max_log])):
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
            
        print(f"[DEBUG] Logging {min(max_samples, len(input))} images to wandb for epoch {self.current_epoch}")
        
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
                    caption=f"Class {label_id} | Input | Reconstruction | Difference"
                )
            )
        
        self.logger.experiment.log({
            f"validation_images_epoch_{self.current_epoch}": images_to_log
        })

    def on_train_epoch_end(self):
        print(f"[DEBUG] Training epoch {self.current_epoch} completed")
        if self.use_ema:
            self.model_ema(self.model)
        
        # Log EMA decay
        if self.use_wandb and self.use_ema:
            self.log("ema_decay", self.ema_decay, on_epoch=True, logger=True)
            
        if self.current_epoch == self.trainer.max_epochs // 3 * 2:
            old_weight = self.lpips_loss_weight
            print(f"[DEBUG] Reducing LPIPS weight from {self.lpips_loss_weight} to {self.lpips_loss_weight * 0.1}")
            self.lpips_loss_weight = self.lpips_loss_weight * 0.1
            
            # Log the weight change to wandb
            if self.use_wandb:
                self.log("lpips_weight_change", {
                    "old_weight": old_weight,
                    "new_weight": self.lpips_loss_weight,
                    "epoch": self.current_epoch
                }, logger=True)

    def on_validation_epoch_end(self):

        if self.use_ema:
            self.model_ema.restore(self.model.parameters())
    
        print(f"[DEBUG] Validation epoch {self.current_epoch} completed with {len(self.validation_step_outputs)} batches")
        self.log_one_batch = False
        val_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        rec_loss = torch.stack([x["rec_loss"] for x in self.validation_step_outputs]).mean()
        lpips_loss = torch.stack(
            [x["lpips_loss"] for x in self.validation_step_outputs]
        ).mean()
        # kl_loss = torch.stack([x['kl_loss'] for x in self.validation_step_outputs]).mean()
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_rec_loss",
            rec_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "val_lpips_loss",
            lpips_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("current_lpips_weight", self.lpips_loss_weight, on_epoch=True, logger=True)
        print(f"[DEBUG] Epoch {self.current_epoch} validation metrics - Loss: {val_loss:.4f}, Rec: {rec_loss:.4f}, LPIPS: {lpips_loss:.4f}")
        
        # Clear outputs for next epoch
        self.validation_step_outputs.clear()
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
    use_wandb = cfg.get('wandb', {}).get('enabled', True)
    wandb_logger = None
    
    if use_wandb:
        # Create wandb run name
        run_name = f"vae_finetune_lr{cfg.training.lr}_bs{cfg.training.batch_size}_ep{cfg.training.num_epochs}"
        if hasattr(cfg.training, 'note') and cfg.training.note:
            run_name += f"_{cfg.training.note}"
        
        wandb_logger = WandbLogger(
            project=cfg.get('wandb', {}).get('project', 'vae-finetuning'),
            entity=cfg.get('wandb', {}).get('entity', None),
            name=run_name,
            config={
                "lr": cfg.training.lr,
                "batch_size": cfg.training.batch_size,
                "num_epochs": cfg.training.num_epochs,
                "image_size": cfg.training.image_size,
                "kl_weight": cfg.training.kl_weight,
                "lpips_loss_weight": cfg.training.lpips_loss_weight,
                "ema_decay": cfg.training.ema_decay,
                "val_size": cfg.training.val_size,
                "precision": cfg.training.precision,
                "data_dir": cfg.data.data_dir,
            },
            tags=cfg.get('wandb', {}).get('tags', ["vae", "finetuning", "stable-diffusion"]),
        )
        print(f"[DEBUG] Initialized wandb logger with project: {wandb_logger.experiment.project}")
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
    print(f"[DEBUG] Loaded VAE weights from {input_path}, found {len(vae_weight)} parameters")

    # Select dataset based on config
    dataset_type = cfg.data.get('dataset_type', 'face')
    print(f"[DEBUG] Using dataset type: {dataset_type}")
    
    if dataset_type == 'co3d':
        data_module = Co3DDataModule(
            co3d_dir=cfg.data.co3d_dir,
            bb_file=cfg.data.bb_file,
            batch_size=cfg.training.batch_size,
            val_size=cfg.training.val_size,
            size=cfg.training.image_size,
            apply_augmentation=cfg.data.get('apply_augmentation', False),
            crop_images=cfg.data.get('crop_images', False),
            patch_num=cfg.data.get('patch_num', None),
        )
    else:
        data_module = DataModule(
            cfg.data.data_dir, 
            batch_size=cfg.training.batch_size, 
            val_size=cfg.training.val_size, 
            size=cfg.training.image_size
        )

    model = FinetuneVAE(
        vae_config=vae_config,
        vae_weights=vae_weight,
        kl_weight=cfg.training.kl_weight,
        lpips_loss_weight=cfg.training.lpips_loss_weight,
        lr=cfg.training.lr,
        device=device,
        log_dir=log_dir,
        ema_decay=cfg.training.ema_decay,
        use_wandb=use_wandb,
    )

    trainer = Trainer(
        max_epochs=cfg.training.num_epochs,
        precision=cfg.training.precision,
        strategy="ddp",
        devices=2,
        accelerator="gpu",
        accumulate_grad_batches=1,
        logger=wandb_logger if use_wandb else None,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
    )


    print(f"[DEBUG] Starting training for {cfg.training.num_epochs} epochs")
    trainer.fit(model, datamodule=data_module)
    
    print(f"[DEBUG] Training completed, saving model to {log_dir}/last_model.pth")
    torch.save(model.model.state_dict(), f"{log_dir}/last_model.pth")
    print("[DEBUG] Model saved successfully")
    
    # Log final model artifact to wandb
    if use_wandb and wandb_logger:
        artifact = wandb.Artifact(
            name=f"vae_model_final",
            type="model",
            description=f"Final VAE model after {cfg.training.num_epochs} epochs"
        )
        artifact.add_file(f"{log_dir}/last_model.pth")
        wandb_logger.experiment.log_artifact(artifact)
        print("[DEBUG] Model artifact logged to wandb")
        
        # Finish the wandb run
        wandb.finish()


if __name__ == "__main__":
    main()