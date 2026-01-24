"""
CO3D Dataset for VAE training.

Provides a PyTorch Dataset for loading CO3D (Common Objects in 3D) data
with camera parameters and optional Plucker coordinate encoding.
"""

import gzip
import json
from pathlib import Path
from typing import IO, List, Optional, Dict, Any, cast

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data.base_dataset import BaseVAEDataset
from data_process.co3d_dataset import jitter_bbox, square_bbox
from data_process.plucker import compute_directions_from_sample, ray_to_plucker


class CO3DDataset(BaseVAEDataset):
    """
    CO3D dataset with camera parameters and optional Plucker coordinates.

    This dataset loads CO3D images with bounding box information and camera
    parameters. Optionally computes Plucker ray coordinates for geometric
    reasoning in VAE training.

    Args:
        root_dir: Path to CO3D dataset root directory
        bb_file: Path to gzipped JSON file containing bounding box annotations
        image_size: Target image size after transforms (default: 256)
        include_plucker: Whether to compute Plucker coordinates (default: False)
        n_patches: Number of patches per dimension for Plucker encoding (default: 8)
        crop_images: Whether to crop images using bounding boxes (default: False)
        apply_augmentation: Whether to apply jitter augmentation (default: False)
        transform: Optional custom image transform (default: resize + normalize)
        **kwargs: Additional arguments passed to BaseVAEDataset
    """

    def __init__(
        self,
        root_dir: str,
        bb_file: str,
        image_size: int = 256,
        include_plucker: bool = False,
        n_patches: Optional[int] = None,
        crop_images: bool = False,
        apply_augmentation: bool = False,
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        # Initialize base class
        super().__init__(
            root_dir=root_dir,
            image_size=image_size,
            include_plucker=include_plucker,
            n_patches=n_patches or 8,
            transform=transform,
            **kwargs
        )

        self.bb_file = bb_file
        self.crop_images = crop_images
        self.apply_augmentation = apply_augmentation

        # Augmentation settings
        self.jitter_scale = (1.1, 1.2) if apply_augmentation else (1.0, 1.0)
        self.jitter_trans = (-0.07, 0.07) if apply_augmentation else (0.0, 0.0)

        # Load samples from bounding box file
        self.samples = self._load_samples(bb_file)

        print(f"[CO3DDataset] Loaded {len(self.samples)} samples from {bb_file}")
        print(f"[CO3DDataset] include_plucker={include_plucker}, "
              f"crop_images={crop_images}, augmentation={apply_augmentation}")

    def _load_samples(self, bb_file: str) -> List[Dict[str, Any]]:
        """
        Parse the gzipped JSON bounding box file.

        Args:
            bb_file: Path to gzipped JSON file

        Returns:
            List of sample dictionaries containing filepath, camera params, bbox
        """
        samples = []

        with gzip.GzipFile(bb_file, "rb") as f:
            obj_dict = json.loads(cast(IO, f).read().decode("utf8"))

        # Flatten all samples from all objects
        for subdir in obj_dict.values():
            samples.extend(subdir)

        return samples

    def _compute_crop_params(
        self,
        bbox: np.ndarray,
        orig_w: int,
        orig_h: int
    ) -> torch.Tensor:
        """
        Calculate Normalized Device Coordinates (NDC) crop parameters.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            orig_w: Original image width
            orig_h: Original image height

        Returns:
            Crop parameters tensor [-cc[0], -cc[1], crop_width, scale_fact]
        """
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

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def _load_image(self, idx: int) -> torch.Tensor:
        """
        Load and transform image at given index.

        Args:
            idx: Sample index

        Returns:
            Transformed image tensor of shape (C, H, W)
        """
        sample_data = self.samples[idx]

        # Load image
        img_path = Path(self.root_dir) / sample_data["filepath"]
        source_image = Image.open(img_path).convert("RGB")

        # Get original dimensions
        img_tensor = transforms.ToTensor()(source_image)
        _, orig_h, orig_w = img_tensor.shape

        # Handle bounding box
        bbox_init = sample_data["bbox"] if self.crop_images else [0, 0, orig_w, orig_h]
        bbox = square_bbox(np.array(bbox_init))

        if self.apply_augmentation:
            bbox = jitter_bbox(bbox, self.jitter_scale, self.jitter_trans)

        rounded_bbox = np.around(bbox).astype(int)

        # Store bbox for later use in _get_camera_params
        self._current_bbox = rounded_bbox
        self._current_orig_w = orig_w
        self._current_orig_h = orig_h

        # Crop and transform image
        img_cropped = transforms.functional.crop(
            source_image,
            top=rounded_bbox[1],
            left=rounded_bbox[0],
            height=rounded_bbox[3] - rounded_bbox[1],
            width=rounded_bbox[2] - rounded_bbox[0]
        )

        img_final = self.transform(img_cropped)

        return img_final

    def _load_plucker_coords(self, idx: int) -> Optional[torch.Tensor]:
        """
        Compute Plucker coordinates for the sample.

        Args:
            idx: Sample index

        Returns:
            Plucker coordinates tensor of shape (n_patches*n_patches, 6)
            or None if not available
        """
        if not self.include_plucker:
            return None

        sample_data = self.samples[idx]

        # Get camera parameters
        camera_params = self._get_camera_params(idx)

        # Compute ray directions
        rays = compute_directions_from_sample(camera_params, self.n_patches)

        # Convert to Plucker coordinates
        plucker_coords = ray_to_plucker(rays)

        return plucker_coords

    def _get_camera_params(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get camera parameters for sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with camera parameters:
            - R: Rotation matrix (3, 3)
            - T: Translation vector (3,)
            - focal_length: Focal length (2,)
            - principal_point: Principal point (2,)
            - crop_params: NDC crop parameters (4,)
        """
        sample_data = self.samples[idx]

        # Compute crop params using cached bbox info from _load_image
        crop_params = self._compute_crop_params(
            self._current_bbox,
            self._current_orig_w,
            self._current_orig_h
        )

        return {
            "R": torch.tensor(sample_data["R"], dtype=torch.float32),
            "T": torch.tensor(sample_data["T"], dtype=torch.float32),
            "focal_length": torch.tensor(sample_data["focal_length"], dtype=torch.float32),
            "principal_point": torch.tensor(sample_data["principal_point"], dtype=torch.float32),
            "crop_params": crop_params,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample at index.

        Returns:
            Dictionary containing:
            - 'image': Image tensor (C, H, W)
            - 'plucker_coords': Plucker coords (n_patches*n_patches, 6) if include_plucker
            - 'camera': Camera params dict (R, T, focal_length, principal_point, crop_params)
            - 'index': Sample index
            - 'filepath': Original filepath string
        """
        # Call parent implementation which handles the standard flow
        sample = super().__getitem__(idx)

        # Add filepath for visualization/debugging
        sample['filepath'] = self.samples[idx].get('filepath', '')

        return sample
