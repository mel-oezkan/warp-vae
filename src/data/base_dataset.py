"""
Base dataset class with optional Plucker coordinate support.

Provides a consistent interface for all VAE datasets.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class BaseVAEDataset(Dataset, ABC):
    """
    Abstract base class for VAE datasets.
    
    Defines the expected interface for datasets used in VAE training.
    Subclasses must implement:
    - __len__
    - __getitem__
    - _load_image
    
    Optional to implement:
    - _load_plucker_coords (if include_plucker=True)
    """
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        include_plucker: bool = False,
        n_patches: Optional[int] = None,
        transform: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize base dataset.
        
        Args:
            root_dir: Root directory containing the data
            image_size: Target image size (square)
            include_plucker: Whether to compute/load Plucker coordinates
            n_patches: Number of patches per dimension for Plucker coords
            transform: Optional image transform (default: resize + normalize)
            **kwargs: Additional dataset-specific arguments
        """
        super().__init__()
        
        self.root_dir = root_dir
        self.image_size = image_size
        self.include_plucker = include_plucker
        self.n_patches = n_patches or 8  # Default 8x8 patches
        self.transform = transform
        
        # Set up default transform if not provided
        if self.transform is None:
            self.transform = self._get_default_transform()
        
        # Storage for dataset items
        self.samples = []
    
    def _get_default_transform(self):
        """Get default image transform."""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] range
        ])
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        pass
    
    @abstractmethod
    def _load_image(self, idx: int) -> torch.Tensor:
        """
        Load and transform image at given index.
        
        Args:
            idx: Sample index
            
        Returns:
            Image tensor of shape (C, H, W)
        """
        pass
    
    def _load_plucker_coords(self, idx: int) -> Optional[torch.Tensor]:
        """
        Load or compute Plucker coordinates for sample.
        
        Override in subclasses that support Plucker coordinates.
        
        Args:
            idx: Sample index
            
        Returns:
            Plucker coordinates tensor of shape (n_patches*n_patches, 6)
            or None if not available
        """
        return None
    
    def _get_camera_params(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get camera parameters for sample.
        
        Override in subclasses with camera information.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with camera parameters (R, T, focal_length, etc.)
            or None if not available
        """
        return None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample at index.
        
        Returns:
            Dictionary containing:
            - 'image': Image tensor (C, H, W)
            - 'plucker_coords': Plucker coords (n_patches*n_patches, 6) if include_plucker
            - 'camera': Camera params dict if available
            - 'index': Sample index
        """
        sample = {
            "image": self._load_image(idx),
            "index": idx,
        }
        
        # Add Plucker coordinates if requested
        if self.include_plucker:
            plucker = self._load_plucker_coords(idx)
            if plucker is not None:
                sample["plucker_coords"] = plucker
        
        # Add camera parameters if available
        camera = self._get_camera_params(idx)
        if camera is not None:
            sample["camera"] = camera
        
        return sample


class PairedDatasetMixin:
    """
    Mixin for datasets that provide paired samples.

    ? This is a helper class to extend BaseVAEDataset for paired data scenarios.
    ? A possible usecase for this is when we want to train the mdoel using image pairs.
    ? The image pairs are currently used for training eq-vae with real world datasets like CO3D/omni_obj.
    
    Adds support for:
    - Source/target image pairs
    - Relative pose computation
    - View consistency sampling
    """
    
    def __init__(
        self,
        pair_sampling: str = "random",
        max_pair_distance: int = 10,
        **kwargs
    ):
        """
        Initialize paired dataset mixin.
        
        Args:
            pair_sampling: Sampling strategy ("random", "sequential", "fixed")
            max_pair_distance: Maximum frame distance for pairs
        """
        self.pair_sampling = pair_sampling
        self.max_pair_distance = max_pair_distance
    
    def _get_pair_index(self, idx: int, object_indices: list) -> int:
        """
        Get paired sample index based on sampling strategy.
        
        Args:
            idx: Source sample index
            object_indices: List of valid indices for the same object
            
        Returns:
            Target sample index
        """
        import random
        
        if self.pair_sampling == "sequential":
            # Next frame (wrap around)
            pos = object_indices.index(idx)
            target_pos = (pos + 1) % len(object_indices)
            return object_indices[target_pos]
        
        elif self.pair_sampling == "random":
            # Random frame from same object (excluding self)
            candidates = [i for i in object_indices if i != idx]
            if not candidates:
                return idx
            return random.choice(candidates)
        
        elif self.pair_sampling == "fixed":
            # Fixed offset (e.g., always 5 frames ahead)
            pos = object_indices.index(idx)
            offset = min(5, len(object_indices) - 1)
            target_pos = (pos + offset) % len(object_indices)
            return object_indices[target_pos]
        
        else:
            raise ValueError(f"Unknown pair_sampling: {self.pair_sampling}")