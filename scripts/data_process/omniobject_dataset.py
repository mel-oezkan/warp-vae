"""OmniObject Dataset for EQ-VAE training.

This module provides a PyTorch Dataset for loading the OmniObject dataset
with view pairs, camera parameters, and Plucker coordinate encoding.
"""

import json
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import pytorch_lightning as pl

from data_process.plucker import compute_directions_from_sample, ray_to_plucker


class OmniObjectDataset(Dataset):
    """OmniObject dataset with view pairs and camera parameters.

    Args:
        data_dir: Path to OmniObject dataset root
        transform: Optional image transforms
        patch_num: Number of patches per dimension for Plucker encoding (e.g., 8 for 8x8 grid)
        image_size: Target image size after transforms
        sample_mode: How to sample views ("pairs" for view pairs, "single" for single views)
        pair_sampling: Strategy for pairing views ("sequential", "random", "fixed_interval")
        device: Device for tensor operations
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        patch_num: Optional[int] = None,
        image_size: int = 512,
        sample_mode: str = "pairs",
        pair_sampling: str = "sequential",
        device: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir) / "img"
        self.transform = transform
        self.patch_num = patch_num
        self.image_size = image_size
        self.sample_mode = sample_mode
        self.pair_sampling = pair_sampling
        self.device = device or "cpu"

        # Discover all object directories
        self.objects = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        print(f"Found {len(self.objects)} object directories")

        # Build sample list
        self.samples = []
        self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found in dataset")

        print(f"Total samples: {len(self.samples)}")

    def _build_samples(self):
        """Build list of samples by scanning object directories."""
        for obj_dir in self.objects:
            transforms_file = obj_dir / "transforms.json"
            if not transforms_file.exists():
                warnings.warn(f"Missing transforms.json for {obj_dir}")
                continue

            # Load camera data
            try:
                with open(transforms_file) as f:
                    camera_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                warnings.warn(f"Failed to load {transforms_file}: {e}")
                continue

            num_views = len(camera_data["frames"])

            if self.sample_mode == "pairs":
                # Generate view pairs
                pairs = self._generate_view_pairs(num_views)
                for view1_idx, view2_idx in pairs:
                    self.samples.append({
                        "obj_dir": obj_dir,
                        "view1_idx": view1_idx,
                        "view2_idx": view2_idx,
                        "camera_data": camera_data
                    })
            else:
                # Single view mode
                for view_idx in range(num_views):
                    self.samples.append({
                        "obj_dir": obj_dir,
                        "view_idx": view_idx,
                        "camera_data": camera_data
                    })

    def _generate_view_pairs(self, num_views: int) -> List[Tuple[int, int]]:
        """Generate view pairs for training.

        Args:
            num_views: Number of views available (typically 24)

        Returns:
            List of (view1_idx, view2_idx) tuples
        """
        if self.pair_sampling == "sequential":
            # Consecutive pairs: (0,1), (1,2), ..., (22,23), (23,0)
            return [(i, (i + 1) % num_views) for i in range(num_views)]

        elif self.pair_sampling == "random":
            # Random pairs excluding same view
            pairs = []
            for i in range(num_views):
                j = np.random.choice([x for x in range(num_views) if x != i])
                pairs.append((i, j))
            return pairs

        elif self.pair_sampling == "fixed_interval":
            # Fixed interval (e.g., 12 views apart for opposite views)
            interval = num_views // 2
            return [(i, (i + interval) % num_views) for i in range(num_views)]

        else:
            raise ValueError(f"Unknown pair_sampling: {self.pair_sampling}")

    def _extract_camera_params(
        self,
        frame_data: dict,
        camera_angle_x: float,
        image_size: int
    ) -> Dict[str, torch.Tensor]:
        """Extract R, T, focal_length, principal_point from transform matrix.

        Args:
            frame_data: Single frame dict with transform_matrix
            camera_angle_x: FOV in radians
            image_size: Image resolution (assumed square)

        Returns:
            dict with R, T, focal_length, principal_point as tensors
        """
        # Transform matrix is camera-to-world (C2W)
        # Format: [[R11, R12, R13, Tx],
        #          [R21, R22, R23, Ty],
        #          [R31, R32, R33, Tz],
        #          [  0,   0,   0,  1]]

        transform_matrix = np.array(frame_data["transform_matrix"])

        # Extract rotation (3x3) and translation (3,)
        R_c2w = transform_matrix[:3, :3]  # Camera-to-world rotation
        T_c2w = transform_matrix[:3, 3]    # Camera-to-world translation

        # Convert to world-to-camera (W2C) for consistency with CO3D
        # W2C rotation: R_w2c = R_c2w.T
        # W2C translation: T_w2c = -R_c2w.T @ T_c2w
        R = R_c2w.T  # (3, 3)
        T = -R_c2w.T @ T_c2w  # (3,)

        # Validate rotation matrix
        det_R = np.linalg.det(R)
        if np.abs(det_R - 1.0) > 0.01:
            warnings.warn(f"Invalid rotation matrix determinant: {det_R}")

        # Compute focal length from FOV
        # focal_length = (image_width / 2) / tan(FOV / 2)
        focal_length = (image_size / 2) / np.tan(camera_angle_x / 2)

        # Principal point (assume image center)
        principal_point = np.array([image_size / 2, image_size / 2])

        return {
            "R": torch.tensor(R, dtype=torch.float32),
            "T": torch.tensor(T, dtype=torch.float32),
            "focal_length": torch.tensor([focal_length, focal_length], dtype=torch.float32),
            "principal_point": torch.tensor(principal_point, dtype=torch.float32)
        }

    def _compute_relative_pose(
        self,
        cam1_params: dict,
        cam2_params: dict
    ) -> Dict[str, torch.Tensor]:
        """Compute relative transformation from camera 1 to camera 2.

        Args:
            cam1_params: {R, T, focal_length, principal_point} for view 1
            cam2_params: {R, T, focal_length, principal_point} for view 2

        Returns:
            dict with relative R and T
        """
        R1, T1 = cam1_params["R"], cam1_params["T"]
        R2, T2 = cam2_params["R"], cam2_params["T"]

        # Relative rotation: R_rel = R2 @ R1.T
        R_rel = R2 @ R1.T

        # Relative translation: T_rel = T2 - R_rel @ T1
        T_rel = T2 - R_rel @ T1

        return {
            "R_rel": R_rel,
            "T_rel": T_rel
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return a view pair with camera parameters and Plucker coordinates."""
        sample = self.samples[idx]

        obj_dir = sample["obj_dir"]
        camera_data = sample["camera_data"]

        if self.sample_mode == "pairs":
            view1_idx = sample["view1_idx"]
            view2_idx = sample["view2_idx"]

            # Load images
            img1_path = obj_dir / f"{view1_idx:03d}.png"
            img2_path = obj_dir / f"{view2_idx:03d}.png"

            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            # Apply transforms
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # Get image size after transform (assumes square images)
            if isinstance(img1, torch.Tensor):
                img_size = img1.shape[-1]
            else:
                img_size = self.image_size

            # Extract camera parameters
            cam1_params = self._extract_camera_params(
                camera_data["frames"][view1_idx],
                camera_data["camera_angle_x"],
                img_size
            )

            cam2_params = self._extract_camera_params(
                camera_data["frames"][view2_idx],
                camera_data["camera_angle_x"],
                img_size
            )

            # Compute relative pose
            rel_pose = self._compute_relative_pose(cam1_params, cam2_params)

            # Create identity crop params (no crop, no NDC adjustment)
            # Format: [-cc[0], -cc[1], crop_width, s]
            crop_params1 = torch.tensor([0.0, 0.0, 2.0, 1.0], dtype=torch.float32)
            crop_params2 = torch.tensor([0.0, 0.0, 2.0, 1.0], dtype=torch.float32)

            # Compute Plucker rays for view 1
            if self.patch_num is not None:
                sample1 = {
                    "R": cam1_params["R"],
                    "T": cam1_params["T"],
                    "focal_length": cam1_params["focal_length"],
                    "principal_point": cam1_params["principal_point"],
                    "crop_params": crop_params1
                }
                rays1 = compute_directions_from_sample(sample1, self.patch_num)
                pluck_rays1 = ray_to_plucker(rays1)

                # Compute Plucker rays for view 2
                sample2 = {
                    "R": cam2_params["R"],
                    "T": cam2_params["T"],
                    "focal_length": cam2_params["focal_length"],
                    "principal_point": cam2_params["principal_point"],
                    "crop_params": crop_params2
                }
                rays2 = compute_directions_from_sample(sample2, self.patch_num)
                pluck_rays2 = ray_to_plucker(rays2)
            else:
                pluck_rays1 = torch.zeros(64, 6)  # Placeholder
                pluck_rays2 = torch.zeros(64, 6)

            return {
                # View 1
                "image": img1,
                "crop_params": crop_params1,
                "R": cam1_params["R"],
                "T": cam1_params["T"],
                "focal_length": cam1_params["focal_length"],
                "principal_point": cam1_params["principal_point"],
                "pluck_ray": pluck_rays1,

                # View 2
                "image2": img2,
                "crop_params2": crop_params2,
                "R2": cam2_params["R"],
                "T2": cam2_params["T"],
                "focal_length2": cam2_params["focal_length"],
                "principal_point2": cam2_params["principal_point"],
                "pluck_ray2": pluck_rays2,

                # Relative pose
                "R_rel": rel_pose["R_rel"],
                "T_rel": rel_pose["T_rel"],

                # Metadata
                "object_name": obj_dir.name,
                "view1_idx": view1_idx,
                "view2_idx": view2_idx,
            }

        else:
            # Single view mode
            view_idx = sample["view_idx"]

            # Load image
            img_path = obj_dir / f"{view_idx:03d}.png"
            img = Image.open(img_path).convert("RGB")

            # Apply transforms
            if self.transform:
                img = self.transform(img)

            # Get image size after transform
            if isinstance(img, torch.Tensor):
                img_size = img.shape[-1]
            else:
                img_size = self.image_size

            # Extract camera parameters
            cam_params = self._extract_camera_params(
                camera_data["frames"][view_idx],
                camera_data["camera_angle_x"],
                img_size
            )

            # Create identity crop params
            crop_params = torch.tensor([0.0, 0.0, 2.0, 1.0], dtype=torch.float32)

            # Compute Plucker rays
            if self.patch_num is not None:
                sample_dict = {
                    "R": cam_params["R"],
                    "T": cam_params["T"],
                    "focal_length": cam_params["focal_length"],
                    "principal_point": cam_params["principal_point"],
                    "crop_params": crop_params
                }
                rays = compute_directions_from_sample(sample_dict, self.patch_num)
                pluck_rays = ray_to_plucker(rays)
            else:
                pluck_rays = torch.zeros(64, 6)

            return {
                "image": img,
                "crop_params": crop_params,
                "R": cam_params["R"],
                "T": cam_params["T"],
                "focal_length": cam_params["focal_length"],
                "principal_point": cam_params["principal_point"],
                "pluck_ray": pluck_rays,
                "object_name": obj_dir.name,
                "view_idx": view_idx,
            }


class OmniObjectDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for OmniObject dataset.

    Args:
        data_dir: Path to OmniObject dataset root
        batch_size: Batch size for dataloaders
        val_size: Fraction of data to use for validation
        size: Target image size
        patch_num: Number of patches per dimension for Plucker encoding
        pair_sampling: Strategy for pairing views
        num_workers: Number of dataloader workers
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        val_size: float = 0.1,
        size: int = 512,
        patch_num: Optional[int] = None,
        pair_sampling: str = "sequential",
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.size = size
        self.patch_num = patch_num
        self.pair_sampling = pair_sampling
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Setup train and validation datasets."""
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        full_ds = OmniObjectDataset(
            data_dir=self.data_dir,
            transform=transform,
            patch_num=self.patch_num,
            image_size=self.size,
            pair_sampling=self.pair_sampling,
        )

        # Split into train and validation
        train_size = int((1 - self.val_size) * len(full_ds))
        val_size = len(full_ds) - train_size
        self.train_ds, self.val_ds = random_split(
            full_ds,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"Train samples: {len(self.train_ds)}, Val samples: {len(self.val_ds)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )
