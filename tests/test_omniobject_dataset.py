"""Unit tests for OmniObject dataset.

These tests verify the correctness of camera parameter extraction,
relative pose computation, and dataset loading.
"""

import json
import numpy as np
import pytest
import torch
from pathlib import Path

from data_process.omniobject_dataset import OmniObjectDataset, OmniObjectDataModule


class TestOmniObjectDataset:
    """Test suite for OmniObjectDataset."""

    @pytest.fixture
    def sample_transform_matrix(self):
        """Create a sample 4x4 transformation matrix."""
        # Simple rotation around Z-axis by 90 degrees
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        T = np.array([1.2, 0.0, 0.5])

        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[:3, 3] = T
        return matrix

    @pytest.fixture
    def sample_frame_data(self, sample_transform_matrix):
        """Create sample frame data."""
        return {
            "file_path": "000.png",
            "transform_matrix": sample_transform_matrix.tolist()
        }

    def test_camera_parameter_extraction_determinant(self, sample_frame_data):
        """Test that extracted rotation matrix has determinant of 1."""
        dataset = OmniObjectDataset(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            image_size=512,
            patch_num=8,
        )

        params = dataset._extract_camera_params(sample_frame_data, 0.8575, 512)

        R = params["R"]
        det = torch.det(R).item()

        assert abs(det - 1.0) < 1e-5, f"Determinant should be 1.0, got {det}"

    def test_camera_parameter_extraction_orthogonality(self, sample_frame_data):
        """Test that rotation matrix is orthogonal (R @ R.T = I)."""
        dataset = OmniObjectDataset(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            image_size=512,
            patch_num=8,
        )

        params = dataset._extract_camera_params(sample_frame_data, 0.8575, 512)

        R = params["R"]
        identity_error = torch.norm(R @ R.T - torch.eye(3)).item()

        assert identity_error < 1e-5, f"Rotation should be orthogonal, error: {identity_error}"

    def test_camera_parameter_extraction_focal_length(self, sample_frame_data):
        """Test focal length computation."""
        dataset = OmniObjectDataset(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            image_size=512,
            patch_num=8,
        )

        camera_angle_x = 0.8575  # radians
        image_size = 512

        params = dataset._extract_camera_params(sample_frame_data, camera_angle_x, image_size)

        # Expected focal length: (image_size / 2) / tan(FOV / 2)
        expected_fl = (image_size / 2) / np.tan(camera_angle_x / 2)

        fl = params["focal_length"][0].item()

        assert abs(fl - expected_fl) < 1e-3, f"Focal length mismatch: {fl} vs {expected_fl}"

    def test_relative_pose_computation(self):
        """Test relative pose computation between two cameras."""
        dataset = OmniObjectDataset(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            image_size=512,
            patch_num=8,
        )

        # Camera 1: identity
        cam1 = {
            "R": torch.eye(3, dtype=torch.float32),
            "T": torch.zeros(3, dtype=torch.float32)
        }

        # Camera 2: 90 degree rotation around Z
        R2 = torch.tensor([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        T2 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        cam2 = {
            "R": R2,
            "T": T2
        }

        rel_pose = dataset._compute_relative_pose(cam1, cam2)

        # Relative rotation should equal R2
        assert torch.allclose(rel_pose["R_rel"], R2, atol=1e-5)

        # Relative translation should equal T2
        assert torch.allclose(rel_pose["T_rel"], T2, atol=1e-5)

    def test_view_pair_generation_sequential(self):
        """Test sequential view pair generation."""
        dataset = OmniObjectDataset(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            image_size=512,
            patch_num=8,
            pair_sampling="sequential"
        )

        pairs = dataset._generate_view_pairs(24)

        # Should have 24 pairs
        assert len(pairs) == 24

        # First pair should be (0, 1)
        assert pairs[0] == (0, 1)

        # Last pair should be (23, 0) - wraps around
        assert pairs[-1] == (23, 0)

        # Check all pairs are consecutive
        for i, (v1, v2) in enumerate(pairs):
            assert v2 == (v1 + 1) % 24

    def test_view_pair_generation_fixed_interval(self):
        """Test fixed interval view pair generation."""
        dataset = OmniObjectDataset(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            image_size=512,
            patch_num=8,
            pair_sampling="fixed_interval"
        )

        pairs = dataset._generate_view_pairs(24)

        # Should have 24 pairs
        assert len(pairs) == 24

        # Interval should be 12 (half of 24)
        for i, (v1, v2) in enumerate(pairs):
            assert v2 == (v1 + 12) % 24

    def test_sample_format_pairs(self):
        """Test that dataset returns correct format for view pairs."""
        dataset = OmniObjectDataset(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            image_size=512,
            patch_num=8,
            sample_mode="pairs"
        )

        if len(dataset) == 0:
            pytest.skip("No samples in dataset")

        sample = dataset[0]

        required_keys = [
            "image", "crop_params", "R", "T", "focal_length",
            "principal_point", "pluck_ray",
            "image2", "crop_params2", "R2", "T2", "focal_length2",
            "principal_point2", "pluck_ray2",
            "R_rel", "T_rel", "object_name", "view1_idx", "view2_idx"
        ]

        for key in required_keys:
            assert key in sample, f"Missing key: {key}"

    def test_plucker_ray_shape(self):
        """Test that Plucker rays have correct shape."""
        dataset = OmniObjectDataset(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            image_size=512,
            patch_num=8,  # 8x8 grid = 64 patches
        )

        if len(dataset) == 0:
            pytest.skip("No samples in dataset")

        sample = dataset[0]

        pluck_rays = sample["pluck_ray"]

        # Should be (64, 6) for 8x8 grid
        assert pluck_rays.shape == (64, 6), f"Expected shape (64, 6), got {pluck_rays.shape}"

    def test_plucker_constraint(self):
        """Test that Plucker rays satisfy the constraint d · m ≈ 0."""
        dataset = OmniObjectDataset(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            image_size=512,
            patch_num=8,
        )

        if len(dataset) == 0:
            pytest.skip("No samples in dataset")

        sample = dataset[0]
        pluck_rays = sample["pluck_ray"]

        # Extract directions and moments
        directions = pluck_rays[:, :3]
        moments = pluck_rays[:, 3:]

        # Compute d · m for each ray
        dot_products = (directions * moments).sum(dim=-1)

        # All should be close to 0
        max_violation = torch.abs(dot_products).max().item()

        assert max_violation < 0.1, f"Plucker constraint violated: max |d·m| = {max_violation}"

    def test_direction_normalization(self):
        """Test that ray directions are approximately unit vectors."""
        dataset = OmniObjectDataset(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            image_size=512,
            patch_num=8,
        )

        if len(dataset) == 0:
            pytest.skip("No samples in dataset")

        sample = dataset[0]
        pluck_rays = sample["pluck_ray"]

        # Extract directions
        directions = pluck_rays[:, :3]

        # Compute norms
        norms = torch.norm(directions, dim=-1)

        # Most should be close to 1 (allowing some variation)
        mean_norm = norms.mean().item()

        assert abs(mean_norm - 1.0) < 0.5, f"Direction norms should be ~1, got mean {mean_norm}"


class TestOmniObjectDataModule:
    """Test suite for OmniObjectDataModule."""

    def test_datamodule_creation(self):
        """Test that datamodule can be created."""
        dm = OmniObjectDataModule(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            batch_size=2,
            val_size=0.1,
            size=512,
            patch_num=8,
        )

        assert dm is not None
        assert dm.batch_size == 2
        assert dm.val_size == 0.1

    def test_datamodule_setup(self):
        """Test that datamodule setup creates train/val splits."""
        dm = OmniObjectDataModule(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            batch_size=2,
            val_size=0.1,
            size=512,
            patch_num=8,
        )

        dm.setup()

        assert hasattr(dm, 'train_ds')
        assert hasattr(dm, 'val_ds')

        # Train should be larger than val
        assert len(dm.train_ds) > len(dm.val_ds)

    def test_dataloader_creation(self):
        """Test that dataloaders can be created."""
        dm = OmniObjectDataModule(
            data_dir="/data/lab_moezkan/omni_obj/blender_renders_24_views",
            batch_size=2,
            val_size=0.1,
            size=512,
            patch_num=8,
        )

        dm.setup()

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
