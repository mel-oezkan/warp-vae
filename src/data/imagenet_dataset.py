"""
ImageNet-256 Dataset for VAE training and evaluation.

Simple dataset loader for ImageNet-style directory structure (class subdirectories).
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
from torchvision import transforms
from PIL import Image

from src.data.base_dataset import BaseVAEDataset


class ImageNetDataset(BaseVAEDataset):
    """
    ImageNet-style dataset loader.
    
    Expects directory structure:
        root_dir/
            class_1/
                image_1.jpg
                image_2.jpg
                ...
            class_2/
                ...
    
    Args:
        root_dir: Path to directory containing class subdirectories
        image_size: Target image size (default: 256)
        transform: Optional custom image transform
        **kwargs: Additional arguments passed to BaseVAEDataset
    """
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        """Initialize ImageNet dataset."""
        super().__init__(
            root_dir=root_dir,
            image_size=image_size,
            include_plucker=False,  # ImageNet doesn't support Plucker
            transform=transform,
            **kwargs
        )
        
        # Discover all image files from class subdirectories
        self.samples = self._discover_images()
        print(f"[ImageNetDataset] Discovered {len(self.samples)} images from {root_dir}")
    
    def _discover_images(self) -> List[Path]:
        """
        Recursively discover all image files in class subdirectories.
        
        Returns:
            List of Path objects pointing to image files
        """
        root = Path(self.root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")
        
        # Supported image extensions
        img_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        images = []
        for img_path in sorted(root.rglob('*')):
            if img_path.suffix.lower() in img_extensions:
                images.append(img_path)
        
        if not images:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                f"Expected class subdirectories with .jpg/.png files."
            )
        
        return images
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def _load_image(self, idx: int) -> torch.Tensor:
        """Load and transform image at given index."""
        img_path = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index."""
        image = self._load_image(idx)
        
        return {
            'image': image,
            'path': str(self.samples[idx]),
        }
