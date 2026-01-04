import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO, List, Optional, cast

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from data_process.co3d_dataset import jitter_bbox, square_bbox
from data_process.plucker import compute_directions_from_sample, ray_to_plucker
from typing import Dict
from torch.utils.data import DataLoader

@dataclass
class Co3DSample:
    filepath: str
    R: List[List[float]]
    T: List[float]
    focal_length: List[float]
    principal_point: List[float]
    bbox: List[int]

class ProcessedCo3D(Dataset):
    """Dataset handler for Co3D data with Plücker ray computation."""
    
    def __init__(
        self,
        co3d_dir: str,
        bb_file: str,
        apply_augmentation: bool = False,
        transform: Optional[transforms.Compose] = None,
        patch_num: Optional[int] = None,
        crop_images: bool = False,
    ):
        super().__init__()
        self.co3d_dir = Path(co3d_dir)
        self.crop_images = crop_images
        self.patch_num = patch_num
        self.apply_augmentation = apply_augmentation

        # Augmentation settings
        self.jitter_scale = (1.1, 1.2) if apply_augmentation else (1.0, 1.0)
        self.jitter_trans = (-0.07, 0.07) if apply_augmentation else (0.0, 0.0)

        # Load Transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Load Samples
        self.samples = self._load_samples(bb_file)

    def _load_samples(self, bb_file: str) -> List[Dict]:
        """Parses the gzipped JSON bounding box file."""
        samples = []
        with gzip.GzipFile(bb_file, "rb") as f:
            obj_dict = json.loads(cast(IO, f).read().decode("utf8"))
        
        for subdir in obj_dict.values():
            samples.extend(subdir)
        return samples

    def _compute_crop_params(self, bbox, orig_w, orig_h):
        """Calculates Normalized Device Coordinates (NDC) crop parameters."""
        crop_center = (bbox[:2] + bbox[2:]) / 2
        max_dim = max(orig_w, orig_h)
        
        # Adjust center relative to square canvas
        crop_center_adj = crop_center + (max_dim - np.array([orig_w, orig_h])) / 2
        
        scale_fact = max_dim / min(orig_w, orig_h)
        ndc_center = scale_fact - 2 * scale_fact * crop_center_adj / max_dim
        crop_width_ndc = 2 * scale_fact * (bbox[2] - bbox[0]) / max_dim
        
        return torch.tensor(
            [-ndc_center[0], -ndc_center[1], crop_width_ndc, scale_fact],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_data = self.samples[idx]
        
        # Load Image
        img_path = self.co3d_dir / sample_data["filepath"]
        source_image = Image.open(img_path)
        img_tensor = transforms.ToTensor()(source_image)
        _, orig_h, orig_w = img_tensor.shape

        # Handle Bounding Box
        bbox_init = sample_data["bbox"] if self.crop_images else [0, 0, orig_w, orig_h]
        bbox = square_bbox(np.array(bbox_init))

        if self.apply_augmentation:
            bbox = jitter_bbox(bbox, self.jitter_scale, self.jitter_trans)
        
        rounded_bbox = np.around(bbox).astype(int)

        # Crop & Transform Image
        img_cropped = transforms.functional.crop(
            source_image, 
            top=rounded_bbox[1], left=rounded_bbox[0], 
            height=rounded_bbox[3]-rounded_bbox[1], width=rounded_bbox[2]-rounded_bbox[0]
        )
        img_final = self.transform(img_cropped)

        # Prepare Output Dict
        result = {
            "image": img_final,
            "crop_params": self._compute_crop_params(rounded_bbox, orig_w, orig_h),
            "R": torch.Tensor(sample_data["R"]),
            "T": torch.Tensor(sample_data["T"]),
            "focal_length": torch.Tensor(sample_data["focal_length"]),
            "principal_point": torch.Tensor(sample_data["principal_point"]),
        }

        # Compute Plücker Rays
        # 
        rays = compute_directions_from_sample(result, self.patch_num)
        result["pluck_ray"] = ray_to_plucker(rays)

        return result
    
class Co3DDataModule(pl.LightningDataModule):
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
        self.save_hyperparameters()
        self.transform = transforms.Compose([
            transforms.Resize((size, size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def setup(self, stage=None):
        full_ds = ProcessedCo3D(
            co3d_dir=self.hparams.co3d_dir,
            bb_file=self.hparams.bb_file,
            apply_augmentation=self.hparams.apply_augmentation,
            transform=self.transform,
            crop_images=self.hparams.crop_images,
            patch_num=self.hparams.patch_num,
        )
        
        train_len = int((1 - self.hparams.val_size) * len(full_ds))
        self.train_ds, self.val_ds = random_split(full_ds, [train_len, len(full_ds) - train_len])
        print(f"[INFO] Data loaded. Train: {len(self.train_ds)}, Val: {len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)