"""
OmniObject3D Dataset for VAE training.

Provides a PyTorch Dataset for loading OmniObject3D multi-view data
with camera parameters and optional Plucker coordinate encoding.
"""

import json
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data.base_dataset import BaseVAEDataset, PairedDatasetMixin
from data_process.plucker import compute_directions_from_sample, ray_to_plucker


class OmniObject3DDataset(BaseVAEDataset, PairedDatasetMixin):
    """
    OmniObject3D dataset with view pairs and camera parameters.

    Supports both single-view and paired-view modes. In single-view mode,
    each sample is a single rendered view. In paired-view mode, each sample
    contains two views of the same object for multi-view consistency learning.

    Args:
        root_dir: Path to OmniObject3D dataset root (will append /img)
        image_size: Target image size after transforms (default: 256)
        include_plucker: Whether to compute Plucker coordinates (default: False)
        n_patches: Number of patches per dimension for Plucker encoding (default: 8)
        sample_mode: "single" for single views, "pairs" for view pairs (default: "single")
        pair_sampling: Strategy for pairing views - "sequential", "random", "fixed_interval"
        transform: Optional custom image transform
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        include_plucker: bool = False,
        n_patches: Optional[int] = None,
        sample_mode: str = "single",
        pair_sampling: str = "sequential",
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        # Initialize base classes
        BaseVAEDataset.__init__(
            self,
            root_dir=root_dir,
            image_size=image_size,
            include_plucker=include_plucker,
            n_patches=n_patches or 8,
            transform=transform,
            **kwargs
        )

        PairedDatasetMixin.__init__(
            self,
            pair_sampling=pair_sampling,
            max_pair_distance=24,  # OmniObject has 24 views
        )

        self.data_dir = Path(root_dir) / "img"
        self.sample_mode = sample_mode

        # Discover all object directories
        self.objects = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if len(self.objects) == 0:
            raise RuntimeError(f"No object directories found in {self.data_dir}")

        print(f"[OmniObject3DDataset] Found {len(self.objects)} object directories")

        # Build sample list
        self.samples = []
        self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found in dataset")

        print(f"[OmniObject3DDataset] Total samples: {len(self.samples)}")
        print(f"[OmniObject3DDataset] sample_mode={sample_mode}, "
              f"pair_sampling={pair_sampling}, include_plucker={include_plucker}")

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
        """
        Generate view pairs for training.

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
        """
        Extract R, T, focal_length, principal_point from transform matrix.

        Args:
            frame_data: Single frame dict with transform_matrix
            camera_angle_x: FOV in radians
            image_size: Image resolution (assumed square)

        Returns:
            dict with R, T, focal_length, principal_point as tensors
        """
        # Transform matrix is camera-to-world (C2W)
        transform_matrix = np.array(frame_data["transform_matrix"])

        # Extract rotation (3x3) and translation (3,)
        R_c2w = transform_matrix[:3, :3]  # Camera-to-world rotation
        T_c2w = transform_matrix[:3, 3]    # Camera-to-world translation

        # Convert to world-to-camera (W2C) for consistency with CO3D
        R = R_c2w.T  # (3, 3)
        T = -R_c2w.T @ T_c2w  # (3,)

        # Validate rotation matrix
        det_R = np.linalg.det(R)
        if np.abs(det_R - 1.0) > 0.01:
            warnings.warn(f"Invalid rotation matrix determinant: {det_R}")

        # Compute focal length from FOV
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
        """
        Compute relative transformation from camera 1 to camera 2.

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
        sample = self.samples[idx]
        obj_dir = sample["obj_dir"]

        if self.sample_mode == "pairs":
            view_idx = sample["view1_idx"]
        else:
            view_idx = sample["view_idx"]

        # Load image
        img_path = obj_dir / f"{view_idx:03d}.png"
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img

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

        sample = self.samples[idx]
        camera_data = sample["camera_data"]
        obj_dir = sample["obj_dir"]

        if self.sample_mode == "pairs":
            view_idx = sample["view1_idx"]
        else:
            view_idx = sample["view_idx"]

        # Get image size (assume square after transform)
        img_size = self.image_size

        # Extract camera parameters
        cam_params = self._extract_camera_params(
            camera_data["frames"][view_idx],
            camera_data["camera_angle_x"],
            img_size
        )

        # Create identity crop params (no crop for OmniObject)
        crop_params = torch.tensor([0.0, 0.0, 2.0, 1.0], dtype=torch.float32)

        # Create sample dict for Plucker computation
        sample_dict = {
            "R": cam_params["R"],
            "T": cam_params["T"],
            "focal_length": cam_params["focal_length"],
            "principal_point": cam_params["principal_point"],
            "crop_params": crop_params
        }

        # Compute Plucker rays
        rays = compute_directions_from_sample(sample_dict, self.n_patches)
        plucker_coords = ray_to_plucker(rays)

        return plucker_coords

    def _get_camera_params(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get camera parameters for sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with camera parameters (R, T, focal_length, principal_point, crop_params)
        """
        sample = self.samples[idx]
        camera_data = sample["camera_data"]

        if self.sample_mode == "pairs":
            view_idx = sample["view1_idx"]
        else:
            view_idx = sample["view_idx"]

        # Get image size
        img_size = self.image_size

        # Extract camera parameters
        cam_params = self._extract_camera_params(
            camera_data["frames"][view_idx],
            camera_data["camera_angle_x"],
            img_size
        )

        # Add identity crop params
        cam_params["crop_params"] = torch.tensor([0.0, 0.0, 2.0, 1.0], dtype=torch.float32)

        return cam_params

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample at index.

        For single-view mode:
            Returns standard dict with: image, plucker_coords, camera, index

        For paired-view mode:
            Returns dict with both views:
            - View 1: image, plucker_coords, camera (standard keys)
            - View 2: image2, plucker_coords2, camera2 (separate keys)
            - Relative pose: R_rel, T_rel
            - Metadata: object_name, view1_idx, view2_idx
        """
        sample_info = self.samples[idx]

        if self.sample_mode == "single":
            # Standard single-view mode - use parent implementation
            result = super().__getitem__(idx)

            # Add metadata
            result["object_name"] = sample_info["obj_dir"].name
            result["view_idx"] = sample_info["view_idx"]

            return result

        else:
            # Paired-view mode - need to manually handle both views
            obj_dir = sample_info["obj_dir"]
            camera_data = sample_info["camera_data"]
            view1_idx = sample_info["view1_idx"]
            view2_idx = sample_info["view2_idx"]
            img_size = self.image_size

            # Load View 1
            img1_path = obj_dir / f"{view1_idx:03d}.png"
            img1 = Image.open(img1_path).convert("RGB")
            if self.transform:
                img1 = self.transform(img1)

            # Load View 2
            img2_path = obj_dir / f"{view2_idx:03d}.png"
            img2 = Image.open(img2_path).convert("RGB")
            if self.transform:
                img2 = self.transform(img2)

            # Extract camera parameters for both views
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

            # Identity crop params
            crop_params1 = torch.tensor([0.0, 0.0, 2.0, 1.0], dtype=torch.float32)
            crop_params2 = torch.tensor([0.0, 0.0, 2.0, 1.0], dtype=torch.float32)

            # Compute Plucker rays if needed
            if self.include_plucker:
                sample1 = {**cam1_params, "crop_params": crop_params1}
                rays1 = compute_directions_from_sample(sample1, self.n_patches)
                pluck_rays1 = ray_to_plucker(rays1)

                sample2 = {**cam2_params, "crop_params": crop_params2}
                rays2 = compute_directions_from_sample(sample2, self.n_patches)
                pluck_rays2 = ray_to_plucker(rays2)
            else:
                pluck_rays1 = torch.zeros(self.n_patches * self.n_patches, 6)
                pluck_rays2 = torch.zeros(self.n_patches * self.n_patches, 6)

            # Build result dict with paired-view structure
            result = {
                # View 1 (standard keys for compatibility with base trainer)
                "image": img1,
                "plucker_coords": pluck_rays1,
                "camera": {**cam1_params, "crop_params": crop_params1},
                "index": idx,

                # View 2 (separate keys)
                "image2": img2,
                "plucker_coords2": pluck_rays2,
                "camera2": {**cam2_params, "crop_params": crop_params2},

                # For backward compatibility with old OmniObject implementation
                "crop_params": crop_params1,
                "crop_params2": crop_params2,
                "R": cam1_params["R"],
                "T": cam1_params["T"],
                "focal_length": cam1_params["focal_length"],
                "principal_point": cam1_params["principal_point"],
                "R2": cam2_params["R"],
                "T2": cam2_params["T"],
                "focal_length2": cam2_params["focal_length"],
                "principal_point2": cam2_params["principal_point"],

                # Relative pose
                "R_rel": rel_pose["R_rel"],
                "T_rel": rel_pose["T_rel"],

                # Metadata
                "object_name": obj_dir.name,
                "view1_idx": view1_idx,
                "view2_idx": view2_idx,
            }

            return result
