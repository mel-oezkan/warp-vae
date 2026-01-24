"""
Warp-enabled Dataset for VAE training with RoMaV2 dense correspondences.

Provides paired image samples with precomputed or on-the-fly warp fields
from RoMaV2 for multi-view consistency training.
"""

import gzip
import json
import random
from pathlib import Path
from typing import IO, List, Optional, Dict, Any, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data.base_dataset import BaseVAEDataset


class WarpCO3DDataset(BaseVAEDataset):
    """
    CO3D dataset with paired images and RoMaV2 warp fields.

    Each sample returns a pair of images from the same object/sequence
    along with the warp field computed by RoMaV2 for multi-view
    consistency training.

    Args:
        root_dir: Path to CO3D dataset root directory
        bb_file: Path to gzipped JSON file containing bounding box annotations
        image_size: Target image size after transforms (default: 256)
        romav2_model: Optional pre-loaded RoMaV2 model (loaded lazily if None)
        romav2_setting: RoMaV2 setting ("turbo", "fast", "base", "precise")
        pair_sampling: How to sample pairs ("random", "sequential", "fixed")
        max_pair_distance: Maximum frame distance for random pairs
        precompute_warps: Whether to cache warps (memory intensive)
        warp_resolution: Resolution for warp field computation
        warp_confidence_threshold: Minimum confidence for valid correspondences
        **kwargs: Additional arguments passed to BaseVAEDataset
    """

    def __init__(
        self,
        root_dir: str,
        bb_file: str,
        image_size: int = 256,
        romav2_model: Optional[Any] = None,
        romav2_setting: str = "turbo",
        romav2_device: Optional[str] = None,  # GPU device for RoMaV2 (e.g., "cuda:1")
        pair_sampling: str = "random",
        max_pair_distance: int = 20,
        precompute_warps: bool = False,
        warp_resolution: int = 256,
        warp_confidence_threshold: float = 0.5,
        include_plucker: bool = False,
        n_patches: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        super().__init__(
            root_dir=root_dir,
            image_size=image_size,
            include_plucker=include_plucker,
            n_patches=n_patches or 8,
            transform=transform,
            **kwargs
        )

        self.bb_file = bb_file
        self.romav2_setting = romav2_setting
        self.romav2_device = romav2_device  # None means default CUDA device
        self.pair_sampling = pair_sampling
        self.max_pair_distance = max_pair_distance
        self.precompute_warps = precompute_warps
        self.warp_resolution = warp_resolution
        self.warp_confidence_threshold = warp_confidence_threshold

        # Load samples and organize by sequence
        self.samples, self.sequence_to_indices = self._load_samples(bb_file)

        # RoMaV2 model - loaded lazily to avoid GPU memory issues during init
        self._romav2_model = romav2_model
        self._romav2_loaded = romav2_model is not None

        # Warp cache if precomputing
        self._warp_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        print(f"[WarpCO3DDataset] Loaded {len(self.samples)} samples from {len(self.sequence_to_indices)} sequences")
        print(f"[WarpCO3DDataset] pair_sampling={pair_sampling}, romav2_setting={romav2_setting}")

    def _load_samples(self, bb_file: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        """
        Parse the gzipped JSON bounding box file and organize by sequence.

        Returns:
            Tuple of (samples list, sequence_to_indices mapping)
        """
        samples = []
        sequence_to_indices: Dict[str, List[int]] = {}

        with gzip.GzipFile(bb_file, "rb") as f:
            obj_dict = json.loads(cast(IO, f).read().decode("utf8"))

        # Flatten samples and track sequence membership
        idx = 0
        for seq_key, subdir in obj_dict.items():
            sequence_to_indices[seq_key] = []
            for sample in subdir:
                sample["sequence_key"] = seq_key
                samples.append(sample)
                sequence_to_indices[seq_key].append(idx)
                idx += 1

        return samples, sequence_to_indices

    def _get_romav2_model(self):
        """Lazily load RoMaV2 model on first use."""
        if not self._romav2_loaded:
            import sys
            import os
            import torch

            # Disable torch.compile for older GPUs
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.disable()

            # Add RoMaV2 to path
            romav2_src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "RoMA2", "src")
            if romav2_src not in sys.path:
                sys.path.insert(0, romav2_src)

            from romav2 import RoMaV2

            # Determine device for RoMaV2
            if self.romav2_device:
                device = torch.device(self.romav2_device)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            cfg = RoMaV2.Cfg(compile=False, setting=self.romav2_setting)
            self._romav2_model = RoMaV2(cfg=cfg)
            self._romav2_model = self._romav2_model.to(device)
            self._romav2_model.eval()
            self._romav2_loaded = True
            print(f"[WarpCO3DDataset] RoMaV2 loaded with setting={self.romav2_setting} on device={device}")

        return self._romav2_model

    def _get_pair_index(self, idx: int) -> int:
        """Get paired sample index based on sampling strategy."""
        sample_data = self.samples[idx]
        seq_key = sample_data["sequence_key"]
        seq_indices = self.sequence_to_indices[seq_key]

        if len(seq_indices) < 2:
            return idx  # Only one sample in sequence

        if self.pair_sampling == "sequential":
            # Next frame (wrap around)
            pos = seq_indices.index(idx)
            target_pos = (pos + 1) % len(seq_indices)
            return seq_indices[target_pos]

        elif self.pair_sampling == "random":
            # Random frame within max_pair_distance
            pos = seq_indices.index(idx)
            min_idx = max(0, pos - self.max_pair_distance)
            max_idx = min(len(seq_indices) - 1, pos + self.max_pair_distance)
            candidates = [seq_indices[i] for i in range(min_idx, max_idx + 1) if seq_indices[i] != idx]
            if not candidates:
                candidates = [i for i in seq_indices if i != idx]
            return random.choice(candidates) if candidates else idx

        elif self.pair_sampling == "fixed":
            # Fixed offset (5 frames ahead)
            pos = seq_indices.index(idx)
            offset = min(5, len(seq_indices) - 1)
            target_pos = (pos + offset) % len(seq_indices)
            return seq_indices[target_pos]

        else:
            raise ValueError(f"Unknown pair_sampling: {self.pair_sampling}")

    def _load_single_image(self, idx: int) -> Tuple[torch.Tensor, Image.Image]:
        """
        Load image at given index.

        Returns:
            Tuple of (transformed tensor, PIL image for RoMaV2)
        """
        sample_data = self.samples[idx]
        img_path = Path(self.root_dir) / sample_data["filepath"]

        pil_image = Image.open(img_path).convert("RGB")

        # Resize for consistency
        pil_image_resized = pil_image.resize(
            (self.image_size, self.image_size),
            Image.LANCZOS
        )

        # Apply transform for tensor output
        img_tensor = self.transform(pil_image_resized)

        return img_tensor, pil_image_resized

    def _compute_warp(
        self,
        img_a: Image.Image,
        img_b: Image.Image
    ) -> Dict[str, torch.Tensor]:
        """
        Compute RoMaV2 warp field between two images.

        Args:
            img_a: Source image (PIL)
            img_b: Target image (PIL)

        Returns:
            Dictionary with:
            - warp_ab: Warp field from A to B (H, W, 2) in [-1, 1] coordinates
            - confidence_ab: Confidence map (H, W)
            - warp_ba: Warp field from B to A (H, W, 2)
            - confidence_ba: Confidence map (H, W)
        """
        model = self._get_romav2_model()

        with torch.no_grad():
            # RoMaV2 expects PIL images
            # Compute A->B warp
            pred_ab = model.match(img_a, img_b)

            # Compute B->A warp (reverse direction)
            pred_ba = model.match(img_b, img_a)

        # Extract warp fields (they are in normalized [-1, 1] coordinates)
        warp_ab = pred_ab["warp_AB"].squeeze(0).cpu()  # (H, W, 2)
        # confidence_AB is (H, W, 4) - use overlap_AB which is (H, W, 1) for simpler confidence
        # or take mean of confidence channels
        if pred_ab["overlap_AB"] is not None:
            confidence_ab = pred_ab["overlap_AB"].squeeze(0).squeeze(-1).cpu()  # (H, W)
        else:
            confidence_ab = pred_ab["confidence_AB"].squeeze(0).mean(dim=-1).cpu()  # (H, W)

        warp_ba = pred_ba["warp_AB"].squeeze(0).cpu()  # (H, W, 2) - note: still warp_AB from reversed call
        if pred_ba["overlap_AB"] is not None:
            confidence_ba = pred_ba["overlap_AB"].squeeze(0).squeeze(-1).cpu()  # (H, W)
        else:
            confidence_ba = pred_ba["confidence_AB"].squeeze(0).mean(dim=-1).cpu()  # (H, W)

        # Resize warps to target resolution if needed
        if warp_ab.shape[0] != self.warp_resolution:
            warp_ab = self._resize_warp(warp_ab, self.warp_resolution)
            confidence_ab = F.interpolate(
                confidence_ab.unsqueeze(0).unsqueeze(0),
                size=(self.warp_resolution, self.warp_resolution),
                mode="bilinear",
                align_corners=False
            ).squeeze()

            warp_ba = self._resize_warp(warp_ba, self.warp_resolution)
            confidence_ba = F.interpolate(
                confidence_ba.unsqueeze(0).unsqueeze(0),
                size=(self.warp_resolution, self.warp_resolution),
                mode="bilinear",
                align_corners=False
            ).squeeze()

        return {
            "warp_ab": warp_ab,
            "confidence_ab": confidence_ab,
            "warp_ba": warp_ba,
            "confidence_ba": confidence_ba,
        }

    def _resize_warp(self, warp: torch.Tensor, target_size: int) -> torch.Tensor:
        """Resize warp field while maintaining normalized coordinates."""
        # warp is (H, W, 2)
        warp = warp.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
        warp = F.interpolate(
            warp,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False
        )
        return warp.squeeze(0).permute(1, 2, 0)  # (H, W, 2)

    def _load_image(self, idx: int) -> torch.Tensor:
        """Load and transform image at given index (required by base class)."""
        img_tensor, _ = self._load_single_image(idx)
        return img_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get paired sample with warp field.

        Returns:
            Dictionary containing:
            - 'image': Source image tensor (C, H, W)
            - 'image_target': Target image tensor (C, H, W)
            - 'warp_ab': Warp field A->B (H, W, 2)
            - 'confidence_ab': Confidence map A->B (H, W)
            - 'warp_ba': Warp field B->A (H, W, 2)
            - 'confidence_ba': Confidence map B->A (H, W)
            - 'index': Source sample index
            - 'index_target': Target sample index
        """
        # Get pair index
        idx_target = self._get_pair_index(idx)

        # Load both images
        img_a_tensor, img_a_pil = self._load_single_image(idx)
        img_b_tensor, img_b_pil = self._load_single_image(idx_target)

        # Check cache or compute warp
        cache_key = (min(idx, idx_target), max(idx, idx_target))

        if self.precompute_warps and cache_key in self._warp_cache:
            warp_data = self._warp_cache[cache_key]
            # Swap if needed
            if idx > idx_target:
                warp_data = {
                    "warp_ab": warp_data["warp_ba"],
                    "confidence_ab": warp_data["confidence_ba"],
                    "warp_ba": warp_data["warp_ab"],
                    "confidence_ba": warp_data["confidence_ab"],
                }
        else:
            warp_data = self._compute_warp(img_a_pil, img_b_pil)
            if self.precompute_warps:
                self._warp_cache[cache_key] = warp_data

        sample = {
            "image": img_a_tensor,
            "image_target": img_b_tensor,
            "warp_ab": warp_data["warp_ab"],
            "confidence_ab": warp_data["confidence_ab"],
            "warp_ba": warp_data["warp_ba"],
            "confidence_ba": warp_data["confidence_ba"],
            "index": idx,
            "index_target": idx_target,
        }

        return sample


class PrecomputedWarpDataset(BaseVAEDataset):
    """
    Dataset that loads precomputed warp fields from disk.

    Use this for faster training when warps have been precomputed
    using the precompute_warps.py script.

    Args:
        root_dir: Path to CO3D dataset root directory
        bb_file: Path to gzipped JSON file containing bounding box annotations
        warp_dir: Directory containing precomputed warp files
        image_size: Target image size after transforms (default: 256)
        **kwargs: Additional arguments passed to BaseVAEDataset
    """

    def __init__(
        self,
        root_dir: str,
        bb_file: str,
        warp_dir: str,
        image_size: int = 256,
        include_plucker: bool = False,
        n_patches: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        **kwargs
    ):
        super().__init__(
            root_dir=root_dir,
            image_size=image_size,
            include_plucker=include_plucker,
            n_patches=n_patches or 8,
            transform=transform,
            **kwargs
        )

        self.bb_file = bb_file
        self.warp_dir = Path(warp_dir)

        # Load samples and pair mappings
        self.samples, self.pairs = self._load_samples_and_pairs(bb_file)

        print(f"[PrecomputedWarpDataset] Loaded {len(self.pairs)} pairs from {bb_file}")

    def _load_samples_and_pairs(
        self,
        bb_file: str
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
        """Load samples and find available precomputed pairs."""
        samples = []
        pairs = []

        with gzip.GzipFile(bb_file, "rb") as f:
            obj_dict = json.loads(cast(IO, f).read().decode("utf8"))

        # Flatten samples
        for subdir in obj_dict.values():
            samples.extend(subdir)

        # Find precomputed warp files
        for warp_file in self.warp_dir.glob("warp_*.pt"):
            # Parse indices from filename: warp_XXXX_YYYY.pt
            name = warp_file.stem
            parts = name.split("_")
            if len(parts) == 3:
                try:
                    idx_a = int(parts[1])
                    idx_b = int(parts[2])
                    if idx_a < len(samples) and idx_b < len(samples):
                        pairs.append((idx_a, idx_b))
                except ValueError:
                    continue

        return samples, pairs

    def _load_image(self, idx: int) -> torch.Tensor:
        """Load and transform image at given index."""
        sample_data = self.samples[idx]
        img_path = Path(self.root_dir) / sample_data["filepath"]

        pil_image = Image.open(img_path).convert("RGB")
        pil_image = pil_image.resize((self.image_size, self.image_size), Image.LANCZOS)

        return self.transform(pil_image)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get paired sample with precomputed warp."""
        idx_a, idx_b = self.pairs[idx]

        # Load images
        img_a = self._load_image(idx_a)
        img_b = self._load_image(idx_b)

        # Load precomputed warp
        warp_file = self.warp_dir / f"warp_{idx_a:04d}_{idx_b:04d}.pt"
        warp_data = torch.load(warp_file)

        return {
            "image": img_a,
            "image_target": img_b,
            "warp_ab": warp_data["warp_ab"],
            "confidence_ab": warp_data["confidence_ab"],
            "warp_ba": warp_data["warp_ba"],
            "confidence_ba": warp_data["confidence_ba"],
            "index": idx_a,
            "index_target": idx_b,
        }
