#!/usr/bin/env python
"""
Compare latent consistency across multiple VAE models on multi-view images.

This script compares one or more models on OmniObject3D or CO3D multi-view data:
1. Loads multiple VAE models (custom checkpoints + optional baseline)
2. Encodes views from multiple objects with all models
3. Computes similarity metrics (cosine sim, MSE) vs camera Euclidean distance
4. Generates comparative visualizations

Usage:
    # Compare single model with baseline
    python scripts/analyze_multiview_latent_consistency.py \
        --checkpoints outputs/my_model/checkpoints/last.ckpt \
        --configs config/my_config.yaml \
        --model_names "My Model" \
        --compare_baseline \
        --output_name comparison_test

    # Compare multiple models with multi-GPU RoMA
    python scripts/analyze_multiview_latent_consistency.py \
        --checkpoints model1.ckpt model2.ckpt \
        --configs config1.yaml config2.yaml \
        --model_names "Model A" "Model B" \
        --mode roma \
        --num_workers 2 \
        --gpu_ids 0 1 \
        --output_name roma_comparison

    # Run on CO3D hydrants
    python scripts/analyze_multiview_latent_consistency.py \
        --dataset co3d \
        --co3d_dir /visinf/projects_students/dlcv2025_groupZ/co3d_full \
        --co3d_annotations /visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz \
        --checkpoints weights/f8/model.ckpt \
        --configs config/baseVAE.yaml \
        --model_names "f8 Baseline" \
        --output_name co3d_hydrant_test
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import multiprocessing as mp
from multiprocessing import Queue

import torch
import torch._dynamo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.decomposition import PCA

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Set path before importing local modules
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.analysis import (
    load_model,
    encode_images,
    denormalize,
    compute_latent_similarity,
    compute_pairwise_similarity_matrices,
    load_camera_data,
    extract_camera_positions,
    compute_euclidean_distance_matrix,
    find_overlapping_pairs,
    find_view_sequences,
    latent_to_pca_rgb,
    # CO3D utilities
    load_co3d_annotations,
    extract_co3d_camera_positions,
    # RoMA utilities
    load_roma_model,
    compute_roma_correspondences,
    compute_bidirectional_region_similarity,
)

# Default paths for f8 baseline VAE
F8_BASELINE_CHECKPOINT = "weights/f8/model.ckpt"
F8_BASELINE_CONFIG = "config/baseVAE.yaml"

# Color palette for models
MODEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def compute_distance_bins(all_results: Dict[str, List[Dict]], n_bins: int = 5) -> List[Tuple[float, float]]:
    """Compute distance bins dynamically from the data."""
    all_distances = []
    for results in all_results.values():
        all_distances.extend([r["camera_distance"] for r in results])
    if not all_distances:
        return [(0, 1)]
    min_d, max_d = min(all_distances), max(all_distances)
    edges = np.linspace(min_d, max_d, n_bins + 1)
    return [(round(edges[i], 2), round(edges[i+1], 2)) for i in range(n_bins)]


class DatasetAdapter:
    """Unified interface for iterating over multi-view objects/sequences."""

    def get_object_ids(self) -> list:
        raise NotImplementedError

    def get_object_name(self, obj_id) -> str:
        raise NotImplementedError

    def get_camera_positions(self, obj_id) -> np.ndarray:
        raise NotImplementedError

    def get_num_views(self, obj_id) -> int:
        raise NotImplementedError

    def load_view_image(self, obj_id, view_idx: int, transform, device: str) -> torch.Tensor:
        raise NotImplementedError

    def load_view_image_pil(self, obj_id, view_idx: int, size: int = 256) -> Image.Image:
        raise NotImplementedError


class OmniObjectAdapter(DatasetAdapter):
    """Adapter for OmniObject3D dataset (transforms.json + numbered PNGs)."""

    def __init__(self, data_dir: str, num_objects: int, seed: int):
        img_dir = Path(data_dir) / "img"
        object_dirs = sorted([d for d in img_dir.iterdir() if d.is_dir()])
        np.random.seed(seed)
        if len(object_dirs) > num_objects:
            np.random.shuffle(object_dirs)
            object_dirs = object_dirs[:num_objects]
        self.object_dirs = object_dirs

    def get_object_ids(self):
        return self.object_dirs

    def get_object_name(self, obj_dir):
        return obj_dir.name

    def get_camera_positions(self, obj_dir):
        transforms_path = obj_dir / "transforms.json"
        if not transforms_path.exists():
            return None
        camera_data = load_camera_data(str(transforms_path))
        return extract_camera_positions(camera_data)

    def get_num_views(self, obj_dir):
        transforms_path = obj_dir / "transforms.json"
        if not transforms_path.exists():
            return 0
        camera_data = load_camera_data(str(transforms_path))
        return len(camera_data["frames"])

    def load_view_image(self, obj_dir, view_idx, transform, device):
        img_path = obj_dir / f"{view_idx:03d}.png"
        img = Image.open(img_path).convert("RGB")
        return transform(img).unsqueeze(0).to(device)

    def load_view_image_pil(self, obj_dir, view_idx, size=256):
        img_path = obj_dir / f"{view_idx:03d}.png"
        img = Image.open(img_path).convert("RGB")
        if size is not None:
            img = img.resize((size, size), Image.LANCZOS)
        return img


class CO3DAdapter(DatasetAdapter):
    """Adapter for CO3D dataset (preprocessed .jgz annotations)."""

    def __init__(self, co3d_dir: str, annotations_path: str, num_objects: int, seed: int):
        self.co3d_dir = Path(co3d_dir)
        self.annotations = load_co3d_annotations(annotations_path)

        # Build flat index mapping (same order as precompute_warps.py)
        self.sequence_to_flat_indices = {}
        flat_idx = 0
        for seq_name in sorted(self.annotations.keys()):
            indices = []
            for _ in self.annotations[seq_name]:
                indices.append(flat_idx)
                flat_idx += 1
            self.sequence_to_flat_indices[seq_name] = indices

        seq_names = sorted(self.annotations.keys())
        np.random.seed(seed)
        if len(seq_names) > num_objects:
            np.random.shuffle(seq_names)
            seq_names = seq_names[:num_objects]
        self.seq_names = seq_names

    def get_object_ids(self):
        return self.seq_names

    def get_object_name(self, seq_name):
        return seq_name

    def get_camera_positions(self, seq_name):
        frames = self.annotations[seq_name]
        if not frames:
            return None
        return extract_co3d_camera_positions(frames)

    def get_num_views(self, seq_name):
        return len(self.annotations[seq_name])

    def load_view_image(self, seq_name, view_idx, transform, device):
        frames = self.annotations[seq_name]
        filepath = frames[view_idx]["filepath"]
        img_path = self.co3d_dir / filepath
        img = Image.open(img_path).convert("RGB")
        return transform(img).unsqueeze(0).to(device)

    def load_view_image_pil(self, seq_name, view_idx, size=256):
        frames = self.annotations[seq_name]
        filepath = frames[view_idx]["filepath"]
        img_path = self.co3d_dir / filepath
        img = Image.open(img_path).convert("RGB")
        if size is not None:
            img = img.resize((size, size), Image.LANCZOS)
        return img


class NativeCO3DAdapter(DatasetAdapter):
    """Adapter for native CO3D dataset (frame_annotations.jgz across all categories)."""

    def __init__(self, co3d_root: str, num_objects: int, seed: int):
        self.co3d_root = Path(co3d_root)
        self.sequences = {}  # (category, seq_name) -> list of frame dicts

        # Scan all category directories
        for cat_dir in sorted(self.co3d_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            ann_path = cat_dir / "frame_annotations.jgz"
            if not ann_path.exists():
                continue

            with gzip.open(str(ann_path), "rt") as f:
                frames = json.load(f)

            # Group frames by sequence
            by_seq = defaultdict(list)
            for fr in frames:
                by_seq[fr["sequence_name"]].append(fr)

            for seq_name, seq_frames in by_seq.items():
                # Sort by frame number for consistent ordering
                seq_frames.sort(key=lambda x: x["frame_number"])
                # Only include sequences whose image directory exists
                seq_dir = cat_dir / seq_name
                if seq_dir.is_dir():
                    self.sequences[(cat_dir.name, seq_name)] = seq_frames

        # Sample sequences
        all_keys = sorted(self.sequences.keys())
        np.random.seed(seed)
        if len(all_keys) > num_objects:
            np.random.shuffle(all_keys)
            all_keys = all_keys[:num_objects]
        self.selected_keys = all_keys
        print(f"NativeCO3DAdapter: {len(self.sequences)} total sequences, selected {len(self.selected_keys)}")

    def get_object_ids(self):
        return self.selected_keys

    def get_object_name(self, obj_id):
        category, seq_name = obj_id
        return f"{category}/{seq_name}"

    def get_camera_positions(self, obj_id):
        frames = self.sequences[obj_id]
        positions = []
        for fr in frames:
            vp = fr["viewpoint"]
            R = np.array(vp["R"])
            T = np.array(vp["T"])
            cam_pos = -R.T @ T
            positions.append(cam_pos)
        return np.array(positions)

    def get_num_views(self, obj_id):
        return len(self.sequences[obj_id])

    def load_view_image(self, obj_id, view_idx, transform, device):
        frames = self.sequences[obj_id]
        img_path = self.co3d_root / frames[view_idx]["image"]["path"]
        img = Image.open(img_path).convert("RGB")
        return transform(img).unsqueeze(0).to(device)

    def load_view_image_pil(self, obj_id, view_idx, size=256):
        frames = self.sequences[obj_id]
        img_path = self.co3d_root / frames[view_idx]["image"]["path"]
        img = Image.open(img_path).convert("RGB")
        if size is not None:
            img = img.resize((size, size), Image.LANCZOS)
        return img


def load_f8_baseline_vae(device="cuda"):
    """Load the f8 SD-VAE baseline model."""
    print(f"Loading f8 baseline VAE from {F8_BASELINE_CHECKPOINT}")
    model, model_type = load_model(
        checkpoint_path=F8_BASELINE_CHECKPOINT,
        config_path=F8_BASELINE_CONFIG,
        model_type="ldm"
    )
    model = model.to(device)
    model.eval()
    print("Loaded f8 baseline VAE (SD 2.x compatible)")
    return model, model_type


def _create_adapter_from_config(dataset_config: Dict, object_ids: list) -> DatasetAdapter:
    """Create a dataset adapter in a worker process from serializable config."""
    if dataset_config['type'] == 'co3d':
        adapter = CO3DAdapter(
            co3d_dir=dataset_config['co3d_dir'],
            annotations_path=dataset_config['co3d_annotations'],
            num_objects=999999,  # don't filter, object_ids already filtered
            seed=0,
        )
        adapter.seq_names = object_ids
        return adapter
    elif dataset_config['type'] == 'co3d_native':
        adapter = NativeCO3DAdapter(
            co3d_root=dataset_config['co3d_native_dir'],
            num_objects=999999,
            seed=0,
        )
        adapter.selected_keys = object_ids
        return adapter
    else:
        adapter = OmniObjectAdapter.__new__(OmniObjectAdapter)
        adapter.object_dirs = object_ids
        return adapter


def worker_process_roma_objects(
    gpu_id: int,
    object_ids: list,
    models_config: Dict,
    roma_config: Dict,
    transform_config: Dict,
    dataset_config: Dict,
    output_queue: Queue,
    worker_id: int = 0
):
    """Worker function for one GPU - processes batch of objects with RoMA analysis.

    Args:
        gpu_id: GPU device ID to use
        object_ids: List of object identifiers to process
        models_config: Dict with keys: checkpoints, configs, names, types, baseline
        roma_config: Dict with roma_setting, confidence_threshold, image_size
        transform_config: Dict with image_size
        dataset_config: Dict with dataset type and paths for adapter reconstruction
        output_queue: Queue to put results in
        worker_id: Worker ID for logging
    """
    # Set device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    try:
        # Reconstruct adapter in worker
        adapter = _create_adapter_from_config(dataset_config, object_ids)

        # Load models on assigned GPU
        models = []
        for ckpt, cfg, name, mtype in zip(
            models_config['checkpoints'],
            models_config['configs'],
            models_config['names'],
            models_config['types']
        ):
            model, model_type = load_model(
                checkpoint_path=ckpt,
                config_path=cfg,
                model_type=mtype
            )
            model = model.to(device)
            model.eval()
            models.append((model, model_type, name))

        # Add baseline if requested
        if models_config.get('baseline', False):
            baseline_model, baseline_type = load_f8_baseline_vae(str(device))
            models.append((baseline_model, baseline_type, "f8 Baseline"))

        print(f"[Worker {worker_id}] GPU {gpu_id}: Loaded {len(models)} models")

        # Load RoMA model
        roma_model = load_roma_model(
            setting=roma_config['roma_setting'],
            device=str(device),
            compile=False
        )
        print(f"[Worker {worker_id}] GPU {gpu_id}: Loaded RoMA model")

        # Create transform
        transform = transforms.Compose([
            transforms.Resize((transform_config['image_size'], transform_config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Process objects
        roma_results = {name: [] for _, _, name in models}

        for obj_id in object_ids:
            results_by_model = analyze_object_with_roma(
                models, roma_model, adapter, obj_id, transform, str(device),
                max_distance=roma_config.get('max_distance', 60),
                min_distance=roma_config.get('min_distance', 2),
                max_pairs=roma_config.get('max_pairs', 50),
                confidence_threshold=roma_config['confidence_threshold'],
                image_size=roma_config['image_size'],
                precomputed_warps_dir=Path(roma_config['precomputed_warps_dir']) if roma_config.get('precomputed_warps_dir') else None
            )
            for model_name, results in results_by_model.items():
                roma_results[model_name].extend(results)

        print(f"[Worker {worker_id}] GPU {gpu_id}: Processed {len(object_ids)} objects")

        # Put results in queue
        output_queue.put({
            'worker_id': worker_id,
            'gpu_id': gpu_id,
            'results': roma_results
        })

    except Exception as e:
        print(f"[Worker {worker_id}] ERROR on GPU {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        output_queue.put({
            'worker_id': worker_id,
            'gpu_id': gpu_id,
            'error': str(e)
        })


@torch.no_grad()
def analyze_object_with_models(
    models: List[Tuple],  # List of (model, model_type, model_name)
    adapter: DatasetAdapter,
    obj_id,
    transform,
    device: str,
    max_distance: float = None,
    min_distance: float = None,
    max_pairs: int = 50
) -> Dict[str, List[Dict]]:
    """Analyze latent consistency for a single object across all models.

    Returns:
        Dictionary mapping model_name -> list of pair results
    """
    positions = adapter.get_camera_positions(obj_id)
    if positions is None:
        return {}

    dist_matrix = compute_euclidean_distance_matrix(positions)
    pairs = find_overlapping_pairs(dist_matrix, max_distance=max_distance, min_distance=min_distance)

    if len(pairs) > max_pairs:
        pairs = pairs[::len(pairs)//max_pairs][:max_pairs]

    results_by_model = {}

    for model, model_type, model_name in models:
        results = []
        for view1_idx, view2_idx, angle in pairs:
            img1 = adapter.load_view_image(obj_id, view1_idx, transform, device)
            img2 = adapter.load_view_image(obj_id, view2_idx, transform, device)

            latent1 = encode_images(model, img1, device, model_type)
            latent2 = encode_images(model, img2, device, model_type)

            similarity = compute_latent_similarity(latent1, latent2)

            results.append({
                "view1_idx": view1_idx,
                "view2_idx": view2_idx,
                "camera_distance": angle,
                **similarity
            })

        results_by_model[model_name] = results

    return results_by_model


@torch.no_grad()
def encode_object_views(
    models: List[Tuple],
    adapter: DatasetAdapter,
    obj_id,
    transform,
    device: str,
    max_views: int = 24
) -> Dict[str, Dict]:
    """Encode all views of an object with all models.

    Returns:
        Dictionary mapping model_name -> {latents, images, positions, dist_matrix, ...}
    """
    positions = adapter.get_camera_positions(obj_id)
    if positions is None:
        return {}

    dist_matrix = compute_euclidean_distance_matrix(positions)
    n_views = min(len(positions), max_views)

    # Load images once
    images = []
    for view_idx in range(n_views):
        img = adapter.load_view_image(obj_id, view_idx, transform, device)
        images.append(img)

    results = {}
    for model, model_type, model_name in models:
        latents = []
        for img in images:
            latent = encode_images(model, img, device, model_type)
            latents.append(latent)

        # Compute pairwise similarity matrices
        matrices = compute_pairwise_similarity_matrices(latents)

        results[model_name] = {
            "latents": latents,
            **matrices,
        }

    # Add shared data (same for all models)
    shared = {
        "images": images,
        "positions": positions,
        "dist_matrix": dist_matrix,
        "n_views": n_views,
        "object_name": adapter.get_object_name(obj_id),
    }

    return {"models": results, "shared": shared}


def load_precomputed_warp(
    warps_dir: Path,
    flat_idx_a: int,
    flat_idx_b: int,
    confidence_threshold: float = 0.8,
    latent_resolution: int = 32
) -> Dict[str, torch.Tensor]:
    """Load a precomputed warp file and convert to latent-space format.

    The precomputed files contain raw warps at image resolution. This function
    applies the same downsampling and confidence masking as compute_roma_correspondences.

    Args:
        warps_dir: Directory containing warp_XXXXX_YYYYY.pt files
        flat_idx_a: Global flat index of first view
        flat_idx_b: Global flat index of second view
        confidence_threshold: Minimum confidence for valid correspondences
        latent_resolution: Target latent resolution (32 for 8x downsampling)

    Returns:
        Dictionary matching compute_roma_correspondences output format, or None if file not found
    """
    # Warp files use sorted (min, max) indices
    idx_lo, idx_hi = min(flat_idx_a, flat_idx_b), max(flat_idx_a, flat_idx_b)
    warp_file = warps_dir / f"warp_{idx_lo:05d}_{idx_hi:05d}.pt"

    if not warp_file.exists():
        return None

    data = torch.load(warp_file, map_location="cpu", weights_only=True)

    # Determine direction: if flat_idx_a < flat_idx_b, file's ab = our ab
    if flat_idx_a <= flat_idx_b:
        warp_ab_raw = data["warp_ab"]      # (H, W, 2)
        conf_ab_raw = data["confidence_ab"]  # (H, W)
        warp_ba_raw = data["warp_ba"]
        conf_ba_raw = data["confidence_ba"]
    else:
        # Swap directions
        warp_ab_raw = data["warp_ba"]
        conf_ab_raw = data["confidence_ba"]
        warp_ba_raw = data["warp_ab"]
        conf_ba_raw = data["confidence_ab"]

    # Add batch dimension: (H, W, 2) -> (1, H, W, 2), (H, W) -> (1, H, W)
    warp_ab = warp_ab_raw.unsqueeze(0)
    warp_ba = warp_ba_raw.unsqueeze(0)
    conf_ab = conf_ab_raw.unsqueeze(0)
    conf_ba = conf_ba_raw.unsqueeze(0)

    image_resolution = warp_ab.shape[1]

    # Downsample warps to latent resolution
    from src.analysis.roma_metrics import warp_to_latent_warp, confidence_to_latent_mask

    warp_ab_latent = warp_to_latent_warp(warp_ab, image_resolution, latent_resolution)
    warp_ba_latent = warp_to_latent_warp(warp_ba, image_resolution, latent_resolution)

    # Create valid masks at latent resolution
    valid_mask_ab = confidence_to_latent_mask(
        conf_ab, confidence_threshold, image_resolution, latent_resolution
    )
    valid_mask_ba = confidence_to_latent_mask(
        conf_ba, confidence_threshold, image_resolution, latent_resolution
    )

    valid_fraction_ab = valid_mask_ab.float().mean().item()
    valid_fraction_ba = valid_mask_ba.float().mean().item()

    return {
        "warp_ab_latent": warp_ab_latent,
        "warp_ba_latent": warp_ba_latent,
        "valid_mask_ab": valid_mask_ab,
        "valid_mask_ba": valid_mask_ba,
        "valid_fraction_ab": valid_fraction_ab,
        "valid_fraction_ba": valid_fraction_ba,
    }


@torch.no_grad()
def analyze_object_with_roma(
    models: List[Tuple],  # List of (model, model_type, model_name)
    roma_model,
    adapter: DatasetAdapter,
    obj_id,
    transform_vae,
    device: str,
    max_distance: float = None,
    min_distance: float = None,
    max_pairs: int = 50,
    confidence_threshold: float = 0.8,
    image_size: int = 256,
    precomputed_warps_dir: Path = None
) -> Dict[str, List[Dict]]:
    """Analyze latent consistency using RoMA region-based comparison.

    For each view pair:
    1. Load both images
    2. Load precomputed warps OR compute RoMA correspondences on-the-fly
    3. Filter by confidence threshold
    4. Map warp to latent space
    5. Encode images with each VAE model
    6. Compare corresponding latent regions

    Returns:
        Dictionary mapping model_name -> list of pair results with:
        - camera_distance
        - global_cosine, global_mse, global_mae (full latent comparison)
        - region_cosine, region_mse, region_mae (valid region comparison)
        - valid_fraction (fraction of latent with valid correspondences)
    """
    positions = adapter.get_camera_positions(obj_id)
    if positions is None:
        return {}

    dist_matrix = compute_euclidean_distance_matrix(positions)
    pairs = find_overlapping_pairs(dist_matrix, max_distance=max_distance, min_distance=min_distance)

    if len(pairs) > max_pairs:
        pairs = pairs[::len(pairs)//max_pairs][:max_pairs]

    # Get flat index mapping for precomputed warps
    flat_indices = None
    if precomputed_warps_dir is not None and hasattr(adapter, 'sequence_to_flat_indices'):
        flat_indices = adapter.sequence_to_flat_indices.get(obj_id)

    results_by_model = {name: [] for _, _, name in models}

    for view1_idx, view2_idx, angle in pairs:
        # Try loading precomputed warp
        roma_results = None
        if precomputed_warps_dir is not None and flat_indices is not None:
            flat_a = flat_indices[view1_idx]
            flat_b = flat_indices[view2_idx]
            roma_results = load_precomputed_warp(
                precomputed_warps_dir, flat_a, flat_b,
                confidence_threshold=confidence_threshold,
                latent_resolution=32
            )

        # Fall back to on-the-fly computation
        if roma_results is None:
            if roma_model is None:
                continue  # No RoMA model and no precomputed warp
            img1_pil = adapter.load_view_image_pil(obj_id, view1_idx, size=image_size)
            img2_pil = adapter.load_view_image_pil(obj_id, view2_idx, size=image_size)
            roma_results = compute_roma_correspondences(
                roma_model,
                img1_pil,
                img2_pil,
                confidence_threshold=confidence_threshold,
                latent_resolution=32
            )

        # Skip if no valid regions
        if roma_results["valid_fraction_ab"] < 0.01:
            continue

        # Load images for VAE (normalized to [-1, 1])
        img1_vae = adapter.load_view_image(obj_id, view1_idx, transform_vae, device)
        img2_vae = adapter.load_view_image(obj_id, view2_idx, transform_vae, device)

        # Process each model
        for model, model_type, model_name in models:
            latent1 = encode_images(model, img1_vae, device, model_type)
            latent2 = encode_images(model, img2_vae, device, model_type)

            # Move warp data to same device as latents
            warp_ab_latent = roma_results["warp_ab_latent"].to(device)
            warp_ba_latent = roma_results["warp_ba_latent"].to(device)
            valid_mask_ab = roma_results["valid_mask_ab"].to(device)
            valid_mask_ba = roma_results["valid_mask_ba"].to(device)

            # Compute bidirectional region similarity
            similarity = compute_bidirectional_region_similarity(
                latent1,
                latent2,
                warp_ab_latent,
                warp_ba_latent,
                valid_mask_ab,
                valid_mask_ba
            )

            results_by_model[model_name].append({
                "view1_idx": view1_idx,
                "view2_idx": view2_idx,
                "camera_distance": angle,
                "region_cosine": similarity["region_cosine"],
                "region_mse": similarity["region_mse"],
                "region_mae": similarity["region_mae"],
                "global_cosine": similarity["global_cosine"],
                "global_mse": similarity["global_mse"],
                "global_mae": similarity["global_mae"],
                "valid_fraction": similarity["valid_fraction"],
                "valid_fraction_ab": similarity["valid_fraction_ab"],
                "valid_fraction_ba": similarity["valid_fraction_ba"],
            })

    return results_by_model


def visualize_model_comparison(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
    model_colors: Dict[str, str]
):
    """Create main comparison visualization across all models."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    model_names = list(all_results.keys())

    # Plot 1: Cosine Similarity vs Camera Distance (scatter + trend)
    ax1 = axes[0, 0]
    for model_name in model_names:
        results = all_results[model_name]
        angles = [r["camera_distance"] for r in results]
        cos_values = [r["cosine_similarity"] for r in results]
        color = model_colors[model_name]

        ax1.scatter(angles, cos_values, alpha=0.3, s=10, color=color, label=model_name)

        # Trend line
        if len(angles) > 3:
            z = np.polyfit(angles, cos_values, 2)
            x_line = np.linspace(min(angles), max(angles), 100)
            ax1.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax1.set_xlabel("Camera Distance (Euclidean)")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Latent Cosine Similarity vs Camera Distance")
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.4, 1.0])

    # Plot 2: MSE vs Camera Distance (scatter + trend)
    ax2 = axes[0, 1]
    for model_name in model_names:
        results = all_results[model_name]
        angles = [r["camera_distance"] for r in results]
        mse_values = [r["mse"] for r in results]
        color = model_colors[model_name]

        ax2.scatter(angles, mse_values, alpha=0.3, s=10, color=color, label=model_name)

        if len(angles) > 3:
            z = np.polyfit(angles, mse_values, 2)
            x_line = np.linspace(min(angles), max(angles), 100)
            ax2.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax2.set_xlabel("Camera Distance (Euclidean)")
    ax2.set_ylabel("Latent MSE")
    ax2.set_title("Latent MSE vs Camera Distance")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: MAE vs Camera Distance (scatter + trend)
    ax3 = axes[0, 2]
    for model_name in model_names:
        results = all_results[model_name]
        angles = [r["camera_distance"] for r in results]
        mae_values = [r["mae"] for r in results]
        color = model_colors[model_name]

        ax3.scatter(angles, mae_values, alpha=0.3, s=10, color=color, label=model_name)

        if len(angles) > 3:
            z = np.polyfit(angles, mae_values, 2)
            x_line = np.linspace(min(angles), max(angles), 100)
            ax3.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax3.set_xlabel("Camera Distance (Euclidean)")
    ax3.set_ylabel("Latent MAE")
    ax3.set_title("Latent MAE vs Camera Distance")
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Box plot comparison (Cosine Similarity)
    ax4 = axes[1, 0]
    box_data = [
        [r["cosine_similarity"] for r in all_results[name]]
        for name in model_names
    ]
    bp = ax4.boxplot(box_data, labels=model_names, patch_artist=True)
    for patch, name in zip(bp['boxes'], model_names):
        patch.set_facecolor(to_rgba(model_colors[name], 0.6))
    ax4.set_ylabel("Cosine Similarity")
    ax4.set_title("Cosine Similarity Distribution")
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=20, ha='right')

    # Plot 5: Box plot comparison (MAE)
    ax5 = axes[1, 1]
    box_data_mae = [
        [r["mae"] for r in all_results[name]]
        for name in model_names
    ]
    bp_mae = ax5.boxplot(box_data_mae, labels=model_names, patch_artist=True)
    for patch, name in zip(bp_mae['boxes'], model_names):
        patch.set_facecolor(to_rgba(model_colors[name], 0.6))
    ax5.set_ylabel("MAE")
    ax5.set_title("MAE Distribution")
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=20, ha='right')

    # Plot 6: Binned cosine similarity comparison
    ax6 = axes[1, 2]
    dist_bins = compute_distance_bins(all_results)
    bin_labels = [f"{low}-{high}" for low, high in dist_bins]
    x = np.arange(len(bin_labels))
    width = 0.8 / len(model_names)

    for idx, model_name in enumerate(model_names):
        results = all_results[model_name]
        means = []
        stds = []
        for low, high in dist_bins:
            bin_vals = [r["cosine_similarity"] for r in results if low <= r["camera_distance"] < high]
            means.append(np.mean(bin_vals) if bin_vals else 0)
            stds.append(np.std(bin_vals) if bin_vals else 0)

        offset = (idx - len(model_names)/2 + 0.5) * width
        ax6.bar(x + offset, means, width, yerr=stds, label=model_name,
                color=model_colors[model_name], alpha=0.7, capsize=2)

    ax6.set_xticks(x)
    ax6.set_xticklabels(bin_labels)
    ax6.set_xlabel("Camera Distance Range")
    ax6.set_ylabel("Mean Cosine Similarity")
    ax6.set_title("Cosine Similarity by Distance Bin")
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # Plot 7: Binned MSE comparison
    ax7 = axes[2, 0]
    for idx, model_name in enumerate(model_names):
        results = all_results[model_name]
        means = []
        stds = []
        for low, high in dist_bins:
            bin_vals = [r["mse"] for r in results if low <= r["camera_distance"] < high]
            means.append(np.mean(bin_vals) if bin_vals else 0)
            stds.append(np.std(bin_vals) if bin_vals else 0)

        offset = (idx - len(model_names)/2 + 0.5) * width
        ax7.bar(x + offset, means, width, yerr=stds, label=model_name,
                color=model_colors[model_name], alpha=0.7, capsize=2)

    ax7.set_xticks(x)
    ax7.set_xticklabels(bin_labels)
    ax7.set_xlabel("Camera Distance Range")
    ax7.set_ylabel("Mean MSE")
    ax7.set_title("Latent MSE by Distance Bin")
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # Plot 8: Binned MAE comparison
    ax8 = axes[2, 1]
    for idx, model_name in enumerate(model_names):
        results = all_results[model_name]
        means = []
        stds = []
        for low, high in dist_bins:
            bin_vals = [r["mae"] for r in results if low <= r["camera_distance"] < high]
            means.append(np.mean(bin_vals) if bin_vals else 0)
            stds.append(np.std(bin_vals) if bin_vals else 0)

        offset = (idx - len(model_names)/2 + 0.5) * width
        ax8.bar(x + offset, means, width, yerr=stds, label=model_name,
                color=model_colors[model_name], alpha=0.7, capsize=2)

    ax8.set_xticks(x)
    ax8.set_xticklabels(bin_labels)
    ax8.set_xlabel("Camera Distance Range")
    ax8.set_ylabel("Mean MAE")
    ax8.set_title("Latent MAE by Distance Bin")
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # Plot 9: Summary statistics table
    ax9 = axes[2, 2]
    ax9.axis('off')

    table_data = []
    headers = ["Metric"] + [name[:12] for name in model_names]

    # Compute stats for each model
    stats = {}
    for model_name in model_names:
        results = all_results[model_name]
        angles = [r["camera_distance"] for r in results]
        cos_vals = [r["cosine_similarity"] for r in results]
        mse_vals = [r["mse"] for r in results]
        mae_vals = [r["mae"] for r in results]

        stats[model_name] = {
            "n_pairs": len(results),
            "cos_mean": np.mean(cos_vals),
            "cos_std": np.std(cos_vals),
            "mse_mean": np.mean(mse_vals),
            "mae_mean": np.mean(mae_vals),
            "mae_std": np.std(mae_vals),
            "dist_cos_corr": np.corrcoef(angles, cos_vals)[0, 1] if len(angles) > 1 else 0,
        }

    table_data.append(["N pairs"] + [f"{stats[n]['n_pairs']}" for n in model_names])
    table_data.append(["Cos Sim (mean)"] + [f"{stats[n]['cos_mean']:.4f}" for n in model_names])
    table_data.append(["Cos Sim (std)"] + [f"{stats[n]['cos_std']:.4f}" for n in model_names])
    table_data.append(["MSE (mean)"] + [f"{stats[n]['mse_mean']:.4f}" for n in model_names])
    table_data.append(["MAE (mean)"] + [f"{stats[n]['mae_mean']:.4f}" for n in model_names])
    table_data.append(["MAE (std)"] + [f"{stats[n]['mae_std']:.4f}" for n in model_names])
    table_data.append(["Dist-Cos Corr"] + [f"{stats[n]['dist_cos_corr']:.4f}" for n in model_names])

    table = ax9.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    ax9.set_title("Summary Statistics", pad=20)

    plt.suptitle("Multi-View Latent Consistency: Model Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison to {output_dir / 'model_comparison.png'}")


def visualize_sequence_comparison(
    object_data: Dict,
    output_dir: Path,
    model_colors: Dict[str, str],
    seq_length: int = 5
):
    """Visualize a sequence of views with latent PCA for all models."""
    shared = object_data["shared"]
    models_data = object_data["models"]

    obj_name = shared["object_name"]
    images = shared["images"]
    positions = shared["positions"]
    dist_matrix = shared["dist_matrix"]
    n_views = shared["n_views"]

    model_names = list(models_data.keys())
    n_models = len(model_names)

    # Find a good sequence
    sequences = find_view_sequences(positions, dist_matrix, seq_length=seq_length, max_pairwise_angle=3.0)

    if not sequences:
        if n_views >= seq_length:
            sequences = [(tuple(range(seq_length)), dist_matrix[0, seq_length-1], 15.0)]
        else:
            print(f"Not enough views for sequence of length {seq_length}")
            return

    view_indices, total_span, avg_step = sequences[0]

    # Fit PCA on combined latents from all models for fair visualization
    all_lat_flat = []
    for model_name in model_names:
        latents = models_data[model_name]["latents"]
        for view_idx in view_indices:
            lat = latents[view_idx][0].cpu().numpy()
            C, H, W = lat.shape
            all_lat_flat.append(lat.reshape(C, -1).T)
    all_lat_flat = np.vstack(all_lat_flat)

    pca_model = PCA(n_components=3)
    pca_model.fit(all_lat_flat)

    # Create figure: (1 + n_models) rows x seq_length columns
    n_rows = 1 + n_models
    fig, axes = plt.subplots(n_rows, seq_length, figsize=(4*seq_length, int(2.75*n_rows)))
    
    print(model_names)
    for col, view_idx in enumerate(view_indices):
        # Row 0: Original images
        img_np = denormalize(images[view_idx][0]).permute(1, 2, 0).cpu().numpy()
        axes[0, col].imshow(np.clip(img_np, 0, 1))
        axes[0, col].set_title(f"View {view_idx}")
        axes[0, col].axis('off')

        # Rows 1 to n_models: Latent PCA for each model
        for row, model_name in enumerate(model_names, start=1):
            latents = models_data[model_name]["latents"]
            lat_rgb, _ = latent_to_pca_rgb(latents[view_idx], pca_model)
            axes[row, col].imshow(lat_rgb)
            if col == 0:
                axes[row, col].set_ylabel(model_name, fontsize=11, fontweight='bold')

            axes[row, col].axis('off')

    axes[0, 0].set_ylabel("Images", fontsize=11, fontweight='bold')

    plt.suptitle(f"Latent Comparison: {obj_name}\nSpan: {total_span:.1f}°, Avg Step: {avg_step:.1f}°",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"sequence_{obj_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sequence comparison to {output_dir / f'sequence_{obj_name}.png'}")


def visualize_similarity_matrices(
    object_data: Dict,
    output_dir: Path,
    model_colors: Dict[str, str]
):
    """Visualize similarity matrices for all models side by side."""
    shared = object_data["shared"]
    models_data = object_data["models"]

    obj_name = shared["object_name"]
    dist_matrix = shared["dist_matrix"]

    model_names = list(models_data.keys())
    n_models = len(model_names)

    # Create figure: 3 rows (cos_sim, mse, mae) x (1 + n_models) columns
    fig, axes = plt.subplots(3, 1 + n_models, figsize=(4*(1+n_models), 11))

    # Column 0: Camera distance matrix
    im0 = axes[0, 0].imshow(dist_matrix, cmap='viridis', aspect='equal')
    axes[0, 0].set_title("Camera Distance")
    axes[0, 0].set_xlabel("View")
    axes[0, 0].set_ylabel("View")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    axes[1, 0].axis('off')  # Empty cell
    axes[2, 0].axis('off')  # Empty cell

    # Columns 1 to n_models: Similarity matrices for each model
    for col, model_name in enumerate(model_names, start=1):
        cos_matrix = models_data[model_name]["cos_sim_matrix"]
        mse_matrix = models_data[model_name]["mse_matrix"]
        mae_matrix = models_data[model_name]["mae_matrix"]

        # Cosine similarity
        im1 = axes[0, col].imshow(cos_matrix, cmap='RdYlGn', aspect='equal', vmin=0.5, vmax=1.0)
        axes[0, col].set_title(f"{model_name}\nCosine Sim")
        axes[0, col].set_xlabel("View")
        plt.colorbar(im1, ax=axes[0, col], shrink=0.8)

        # MSE
        im2 = axes[1, col].imshow(mse_matrix, cmap='hot', aspect='equal')
        axes[1, col].set_title(f"{model_name}\nMSE")
        axes[1, col].set_xlabel("View")
        plt.colorbar(im2, ax=axes[1, col], shrink=0.8)

        # MAE
        im3 = axes[2, col].imshow(mae_matrix, cmap='hot', aspect='equal')
        axes[2, col].set_title(f"{model_name}\nMAE")
        axes[2, col].set_xlabel("View")
        plt.colorbar(im3, ax=axes[2, col], shrink=0.8)

    plt.suptitle(f"Similarity Matrices: {obj_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"matrices_{obj_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity matrices to {output_dir / f'matrices_{obj_name}.png'}")


def visualize_angle_vs_similarity_per_object(
    object_data: Dict,
    output_dir: Path,
    model_colors: Dict[str, str]
):
    """Scatter plot of distance vs similarity for a single object, all models."""
    shared = object_data["shared"]
    models_data = object_data["models"]

    obj_name = shared["object_name"]
    dist_matrix = shared["dist_matrix"]
    n_views = shared["n_views"]

    model_names = list(models_data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Extract upper triangle indices
    triu_idx = np.triu_indices(n_views, k=1)
    angles_flat = dist_matrix[triu_idx]

    # Plot cosine similarity
    ax1 = axes[0]
    for model_name in model_names:
        cos_matrix = models_data[model_name]["cos_sim_matrix"]
        cos_flat = cos_matrix[triu_idx]
        color = model_colors[model_name]

        ax1.scatter(angles_flat, cos_flat, alpha=0.5, s=30, color=color, label=model_name)

        # Trend line
        if len(angles_flat) > 3:
            z = np.polyfit(angles_flat, cos_flat, 2)
            x_line = np.linspace(min(angles_flat), max(angles_flat), 100)
            ax1.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax1.set_xlabel("Camera Distance")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Cosine Similarity vs Distance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot MSE
    ax2 = axes[1]
    for model_name in model_names:
        mse_matrix = models_data[model_name]["mse_matrix"]
        mse_flat = mse_matrix[triu_idx]
        color = model_colors[model_name]

        ax2.scatter(angles_flat, mse_flat, alpha=0.5, s=30, color=color, label=model_name)

        if len(angles_flat) > 3:
            z = np.polyfit(angles_flat, mse_flat, 2)
            x_line = np.linspace(min(angles_flat), max(angles_flat), 100)
            ax2.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax2.set_xlabel("Camera Distance")
    ax2.set_ylabel("MSE")
    ax2.set_title("MSE vs Distance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot MAE
    ax3 = axes[2]
    for model_name in model_names:
        mae_matrix = models_data[model_name]["mae_matrix"]
        mae_flat = mae_matrix[triu_idx]
        color = model_colors[model_name]

        ax3.scatter(angles_flat, mae_flat, alpha=0.5, s=30, color=color, label=model_name)

        if len(angles_flat) > 3:
            z = np.polyfit(angles_flat, mae_flat, 2)
            x_line = np.linspace(min(angles_flat), max(angles_flat), 100)
            ax3.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax3.set_xlabel("Camera Distance")
    ax3.set_ylabel("MAE")
    ax3.set_title("MAE vs Distance")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f"Distance vs Similarity: {obj_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"angle_vs_sim_{obj_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distance vs similarity to {output_dir / f'angle_vs_sim_{obj_name}.png'}")


def visualize_roma_model_comparison(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
    model_colors: Dict[str, str]
):
    """Create comparison visualization for RoMA-based region analysis."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    model_names = list(all_results.keys())

    # Plot 1: Region Cosine Similarity vs Camera Distance
    ax1 = axes[0, 0]
    for model_name in model_names:
        results = all_results[model_name]
        if not results:
            continue
        angles = [r["camera_distance"] for r in results]
        cos_values = [r["region_cosine"] for r in results if not np.isnan(r["region_cosine"])]
        angles_valid = [r["camera_distance"] for r in results if not np.isnan(r["region_cosine"])]
        color = model_colors[model_name]

        if angles_valid:
            ax1.scatter(angles_valid, cos_values, alpha=0.3, s=10, color=color, label=model_name)
            if len(angles_valid) > 3:
                z = np.polyfit(angles_valid, cos_values, 2)
                x_line = np.linspace(min(angles_valid), max(angles_valid), 100)
                ax1.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax1.set_xlabel("Camera Distance (Euclidean)")
    ax1.set_ylabel("Region Cosine Similarity")
    ax1.set_title("Region Cosine Similarity vs Camera Distance")
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.4, 1.0])

    # Plot 2: Region MSE vs Camera Distance
    ax2 = axes[0, 1]
    for model_name in model_names:
        results = all_results[model_name]
        if not results:
            continue
        mse_values = [r["region_mse"] for r in results if not np.isnan(r["region_mse"])]
        angles_valid = [r["camera_distance"] for r in results if not np.isnan(r["region_mse"])]
        color = model_colors[model_name]

        if angles_valid:
            ax2.scatter(angles_valid, mse_values, alpha=0.3, s=10, color=color, label=model_name)
            if len(angles_valid) > 3:
                z = np.polyfit(angles_valid, mse_values, 2)
                x_line = np.linspace(min(angles_valid), max(angles_valid), 100)
                ax2.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax2.set_xlabel("Camera Distance (Euclidean)")
    ax2.set_ylabel("Region MSE")
    ax2.set_title("Region MSE vs Camera Distance")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Valid Fraction vs Camera Distance
    ax3 = axes[0, 2]
    for model_name in model_names:
        results = all_results[model_name]
        if not results:
            continue
        angles = [r["camera_distance"] for r in results]
        valid_fracs = [r["valid_fraction"] for r in results]
        color = model_colors[model_name]

        ax3.scatter(angles, valid_fracs, alpha=0.3, s=10, color=color, label=model_name)
        if len(angles) > 3:
            z = np.polyfit(angles, valid_fracs, 2)
            x_line = np.linspace(min(angles), max(angles), 100)
            ax3.plot(x_line, np.poly1d(z)(x_line), '-', color=color, linewidth=2, alpha=0.8)

    ax3.set_xlabel("Camera Distance (Euclidean)")
    ax3.set_ylabel("Valid Fraction")
    ax3.set_title("RoMA Valid Region Fraction vs Distance")
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.0])

    # Plot 4: Region vs Global Cosine Similarity
    ax4 = axes[1, 0]
    for model_name in model_names:
        results = all_results[model_name]
        if not results:
            continue
        global_cos = [r["global_cosine"] for r in results if not np.isnan(r["region_cosine"])]
        region_cos = [r["region_cosine"] for r in results if not np.isnan(r["region_cosine"])]
        color = model_colors[model_name]

        if global_cos:
            ax4.scatter(global_cos, region_cos, alpha=0.5, s=20, color=color, label=model_name)

    # Add diagonal line
    ax4.plot([0.4, 1.0], [0.4, 1.0], 'k--', alpha=0.3, label='y=x')
    ax4.set_xlabel("Global Cosine Similarity")
    ax4.set_ylabel("Region Cosine Similarity")
    ax4.set_title("Region vs Global Similarity")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0.4, 1.0])
    ax4.set_ylim([0.4, 1.0])

    # Plot 5: Box plot - Region Cosine
    ax5 = axes[1, 1]
    box_data = []
    valid_names = []
    for name in model_names:
        vals = [r["region_cosine"] for r in all_results[name] if not np.isnan(r["region_cosine"])]
        if vals:
            box_data.append(vals)
            valid_names.append(name)
    if box_data:
        bp = ax5.boxplot(box_data, labels=valid_names, patch_artist=True)
        for patch, name in zip(bp['boxes'], valid_names):
            patch.set_facecolor(to_rgba(model_colors[name], 0.6))
    ax5.set_ylabel("Region Cosine Similarity")
    ax5.set_title("Region Cosine Distribution")
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=20, ha='right')

    # Plot 6: Box plot - Valid Fraction
    ax6 = axes[1, 2]
    box_data = []
    valid_names = []
    for name in model_names:
        vals = [r["valid_fraction"] for r in all_results[name]]
        if vals:
            box_data.append(vals)
            valid_names.append(name)
    if box_data:
        bp = ax6.boxplot(box_data, labels=valid_names, patch_artist=True)
        for patch, name in zip(bp['boxes'], valid_names):
            patch.set_facecolor(to_rgba(model_colors[name], 0.6))
    ax6.set_ylabel("Valid Fraction")
    ax6.set_title("Valid Fraction Distribution")
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=20, ha='right')

    # Plot 7: Binned Region Cosine
    ax7 = axes[2, 0]
    dist_bins = compute_distance_bins(all_results)
    bin_labels = [f"{low}-{high}" for low, high in dist_bins]
    x = np.arange(len(bin_labels))
    width = 0.8 / max(len(model_names), 1)

    for idx, model_name in enumerate(model_names):
        results = all_results[model_name]
        means = []
        stds = []
        for low, high in dist_bins:
            bin_vals = [r["region_cosine"] for r in results
                       if low <= r["camera_distance"] < high and not np.isnan(r["region_cosine"])]
            means.append(np.mean(bin_vals) if bin_vals else 0)
            stds.append(np.std(bin_vals) if bin_vals else 0)

        offset = (idx - len(model_names)/2 + 0.5) * width
        ax7.bar(x + offset, means, width, yerr=stds, label=model_name,
                color=model_colors[model_name], alpha=0.7, capsize=2)

    ax7.set_xticks(x)
    ax7.set_xticklabels(bin_labels)
    ax7.set_xlabel("Camera Distance Range")
    ax7.set_ylabel("Mean Region Cosine")
    ax7.set_title("Region Cosine by Distance Bin")
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # Plot 8: Binned Region MSE
    ax8 = axes[2, 1]
    for idx, model_name in enumerate(model_names):
        results = all_results[model_name]
        means = []
        stds = []
        for low, high in dist_bins:
            bin_vals = [r["region_mse"] for r in results
                       if low <= r["camera_distance"] < high and not np.isnan(r["region_mse"])]
            means.append(np.mean(bin_vals) if bin_vals else 0)
            stds.append(np.std(bin_vals) if bin_vals else 0)

        offset = (idx - len(model_names)/2 + 0.5) * width
        ax8.bar(x + offset, means, width, yerr=stds, label=model_name,
                color=model_colors[model_name], alpha=0.7, capsize=2)

    ax8.set_xticks(x)
    ax8.set_xticklabels(bin_labels)
    ax8.set_xlabel("Camera Distance Range")
    ax8.set_ylabel("Mean Region MSE")
    ax8.set_title("Region MSE by Distance Bin")
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # Plot 9: Summary statistics table
    ax9 = axes[2, 2]
    ax9.axis('off')

    table_data = []
    headers = ["Metric"] + [name[:12] for name in model_names]

    stats = {}
    for model_name in model_names:
        results = all_results[model_name]
        if not results:
            stats[model_name] = {"n_pairs": 0}
            continue

        region_cos = [r["region_cosine"] for r in results if not np.isnan(r["region_cosine"])]
        region_mse = [r["region_mse"] for r in results if not np.isnan(r["region_mse"])]
        valid_fracs = [r["valid_fraction"] for r in results]

        stats[model_name] = {
            "n_pairs": len(results),
            "region_cos_mean": np.mean(region_cos) if region_cos else float('nan'),
            "region_cos_std": np.std(region_cos) if region_cos else float('nan'),
            "region_mse_mean": np.mean(region_mse) if region_mse else float('nan'),
            "valid_frac_mean": np.mean(valid_fracs) if valid_fracs else float('nan'),
        }

    table_data.append(["N pairs"] + [f"{stats[n].get('n_pairs', 0)}" for n in model_names])
    table_data.append(["Region Cos"] + [f"{stats[n].get('region_cos_mean', 0):.4f}" for n in model_names])
    table_data.append(["Region Cos (std)"] + [f"{stats[n].get('region_cos_std', 0):.4f}" for n in model_names])
    table_data.append(["Region MSE"] + [f"{stats[n].get('region_mse_mean', 0):.4f}" for n in model_names])
    table_data.append(["Valid Frac"] + [f"{stats[n].get('valid_frac_mean', 0):.4f}" for n in model_names])

    table = ax9.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    ax9.set_title("Summary Statistics", pad=20)

    plt.suptitle("RoMA Region-Based Latent Consistency: Model Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "roma_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RoMA model comparison to {output_dir / 'roma_model_comparison.png'}")


def save_roma_comparison_stats(
    all_results: Dict[str, List[Dict]],
    output_path: Path
):
    """Save RoMA-based comparison statistics to a text file."""
    model_names = list(all_results.keys())

    with open(output_path, 'w') as f:
        f.write("RoMA Region-Based Latent Consistency: Model Comparison\n")
        f.write("=" * 70 + "\n\n")

        for model_name in model_names:
            results = all_results[model_name]
            if not results:
                f.write(f"Model: {model_name}\n")
                f.write("  No valid pairs found.\n\n")
                continue

            angles = [r["camera_distance"] for r in results]
            region_cos = [r["region_cosine"] for r in results if not np.isnan(r["region_cosine"])]
            region_mse = [r["region_mse"] for r in results if not np.isnan(r["region_mse"])]
            region_mae = [r["region_mae"] for r in results if not np.isnan(r["region_mae"])]
            global_cos = [r["global_cosine"] for r in results]
            valid_fracs = [r["valid_fraction"] for r in results]

            f.write(f"Model: {model_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Total pairs analyzed: {len(results)}\n")
            f.write(f"  Camera distance range: {min(angles):.2f} - {max(angles):.2f}\n")

            f.write("\n  Region Cosine Similarity (corresponding areas only):\n")
            if region_cos:
                f.write(f"    Mean: {np.mean(region_cos):.4f}\n")
                f.write(f"    Std:  {np.std(region_cos):.4f}\n")
                f.write(f"    Min:  {min(region_cos):.4f}\n")
                f.write(f"    Max:  {max(region_cos):.4f}\n")
            else:
                f.write("    No valid measurements\n")

            f.write("\n  Region MSE:\n")
            if region_mse:
                f.write(f"    Mean: {np.mean(region_mse):.4f}\n")
                f.write(f"    Std:  {np.std(region_mse):.4f}\n")

            f.write("\n  Region MAE:\n")
            if region_mae:
                f.write(f"    Mean: {np.mean(region_mae):.4f}\n")
                f.write(f"    Std:  {np.std(region_mae):.4f}\n")

            f.write("\n  Global Cosine Similarity (for comparison):\n")
            f.write(f"    Mean: {np.mean(global_cos):.4f}\n")
            f.write(f"    Std:  {np.std(global_cos):.4f}\n")

            f.write("\n  Valid Fraction (RoMA correspondence coverage):\n")
            f.write(f"    Mean: {np.mean(valid_fracs):.4f}\n")
            f.write(f"    Std:  {np.std(valid_fracs):.4f}\n")
            f.write(f"    Min:  {min(valid_fracs):.4f}\n")
            f.write(f"    Max:  {max(valid_fracs):.4f}\n")

            if region_cos and len(angles) > 1:
                corr = np.corrcoef(
                    [a for a, r in zip(angles, results) if not np.isnan(r["region_cosine"])],
                    region_cos
                )[0, 1]
                f.write(f"\n  Correlation(distance, region_cos): {corr:.4f}\n")
            f.write("\n")

        # Binned comparison
        f.write("\nBinned Comparison (Region Cosine Similarity)\n")
        f.write("=" * 70 + "\n")

        dist_bins = compute_distance_bins(all_results)

        header = f"{'Bin':<12}" + "".join([f"{name[:12]:<14}" for name in model_names])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for low, high in dist_bins:
            row = f"{low}-{high}".ljust(12)
            for model_name in model_names:
                results = all_results[model_name]
                bin_vals = [r["region_cosine"] for r in results
                           if low <= r["camera_distance"] < high and not np.isnan(r["region_cosine"])]
                if bin_vals:
                    row += f"{np.mean(bin_vals):.3f}±{np.std(bin_vals):.3f}".ljust(14)
                else:
                    row += "N/A".ljust(14)
            f.write(row + "\n")

        # Binned valid fraction
        f.write("\n\nBinned Valid Fraction\n")
        f.write("=" * 70 + "\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for low, high in dist_bins:
            row = f"{low}-{high}".ljust(12)
            for model_name in model_names:
                results = all_results[model_name]
                bin_vals = [r["valid_fraction"] for r in results if low <= r["camera_distance"] < high]
                if bin_vals:
                    row += f"{np.mean(bin_vals):.3f}±{np.std(bin_vals):.3f}".ljust(14)
                else:
                    row += "N/A".ljust(14)
            f.write(row + "\n")

    print(f"Saved RoMA statistics to {output_path}")


def save_comparison_stats(
    all_results: Dict[str, List[Dict]],
    output_path: Path
):
    """Save comparison statistics to a text file."""
    model_names = list(all_results.keys())

    with open(output_path, 'w') as f:
        f.write("Multi-View Latent Consistency: Model Comparison\n")
        f.write("=" * 70 + "\n\n")

        for model_name in model_names:
            results = all_results[model_name]
            angles = [r["camera_distance"] for r in results]
            cos_values = [r["cosine_similarity"] for r in results]
            mse_values = [r["mse"] for r in results]
            mae_values = [r["mae"] for r in results]

            f.write(f"Model: {model_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Total pairs analyzed: {len(results)}\n")
            f.write(f"  Camera distance range: {min(angles):.2f} - {max(angles):.2f}\n")
            f.write("\n  Cosine Similarity:\n")
            f.write(f"    Mean: {np.mean(cos_values):.4f}\n")
            f.write(f"    Std:  {np.std(cos_values):.4f}\n")
            f.write(f"    Min:  {min(cos_values):.4f}\n")
            f.write(f"    Max:  {max(cos_values):.4f}\n")
            f.write("\n  MSE:\n")
            f.write(f"    Mean: {np.mean(mse_values):.4f}\n")
            f.write(f"    Std:  {np.std(mse_values):.4f}\n")
            f.write("\n  MAE:\n")
            f.write(f"    Mean: {np.mean(mae_values):.4f}\n")
            f.write(f"    Std:  {np.std(mae_values):.4f}\n")
            f.write(f"    Min:  {min(mae_values):.4f}\n")
            f.write(f"    Max:  {max(mae_values):.4f}\n")

            corr = np.corrcoef(angles, cos_values)[0, 1] if len(angles) > 1 else 0
            f.write(f"\n  Correlation(distance, cos_sim): {corr:.4f}\n")
            f.write("\n")

        # Binned comparison (Cosine Similarity)
        f.write("\nBinned Comparison (Cosine Similarity)\n")
        f.write("=" * 70 + "\n")

        dist_bins = compute_distance_bins(all_results)

        header = f"{'Bin':<12}" + "".join([f"{name[:12]:<14}" for name in model_names])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for low, high in dist_bins:
            row = f"{low}-{high}".ljust(12)
            for model_name in model_names:
                results = all_results[model_name]
                bin_vals = [r["cosine_similarity"] for r in results if low <= r["camera_distance"] < high]
                if bin_vals:
                    row += f"{np.mean(bin_vals):.3f}±{np.std(bin_vals):.3f}".ljust(14)
                else:
                    row += "N/A".ljust(14)
            f.write(row + "\n")

        # Binned comparison (MAE)
        f.write("\n\nBinned Comparison (MAE)\n")
        f.write("=" * 70 + "\n")

        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for low, high in dist_bins:
            row = f"{low}-{high}".ljust(12)
            for model_name in model_names:
                results = all_results[model_name]
                bin_vals = [r["mae"] for r in results if low <= r["camera_distance"] < high]
                if bin_vals:
                    row += f"{np.mean(bin_vals):.3f}±{np.std(bin_vals):.3f}".ljust(14)
                else:
                    row += "N/A".ljust(14)
            f.write(row + "\n")

    print(f"Saved statistics to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare latent consistency across multiple VAE models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model inputs (can specify multiple)
    parser.add_argument(
        "--checkpoints", type=str, nargs='+', required=True,
        help="Paths to model checkpoints (one or more)"
    )
    parser.add_argument(
        "--configs", type=str, nargs='+', required=True,
        help="Paths to config files (one per checkpoint)"
    )
    parser.add_argument(
        "--model_names", type=str, nargs='+', default=None,
        help="Names for each model (defaults to checkpoint names)"
    )
    parser.add_argument(
        "--model_types", type=str, nargs='+', default=None,
        choices=["auto", "ldm", "eqvae", "diffusers"],
        help="Model types (one per checkpoint, defaults to 'auto')"
    )

    # Baseline comparison
    parser.add_argument(
        "--compare_baseline", action="store_true",
        help="Include f8 baseline VAE in comparison"
    )

    # Output
    parser.add_argument(
        "--output_name", type=str, required=True,
        help="Output subfolder name under eval_outputs/"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset", type=str, default="omniobject",
        choices=["omniobject", "co3d", "co3d_native"],
        help="Dataset type to use"
    )
    parser.add_argument(
        "--co3d_native_dir", type=str,
        default="/visinf/home/lab_mozkan/computer-vision-proj-lab/data/co3d_data",
        help="Root directory of native CO3D data with per-category frame_annotations.jgz "
             "(only used with --dataset co3d_native)"
    )
    parser.add_argument(
        "--co3d_dir", type=str,
        default="/visinf/projects_students/dlcv2025_groupZ/co3d_full",
        help="CO3D dataset root directory (only used with --dataset co3d)"
    )
    parser.add_argument(
        "--co3d_annotations", type=str,
        default="/visinf/projects_students/dlcv2025_groupZ/co3d_annotations/hydrant_train.jgz",
        help="Path to CO3D preprocessed annotations .jgz file (only used with --dataset co3d)"
    )

    # Data options
    parser.add_argument(
        "--data_dir", type=str,
        default="/data/lab_moezkan/omni_obj/blender_renders_24_views",
        help="OmniObject3D dataset directory (only used with --dataset omniobject)"
    )
    parser.add_argument(
        "--num_objects", type=int, default=50,
        help="Number of objects to analyze for aggregate statistics"
    )
    parser.add_argument(
        "--num_detailed_objects", type=int, default=5,
        help="Number of objects for detailed per-object visualizations"
    )
    parser.add_argument(
        "--max_distance", type=float, default=None,
        help="Maximum camera Euclidean distance for pair selection (default: no limit)"
    )
    parser.add_argument(
        "--min_distance", type=float, default=None,
        help="Minimum camera Euclidean distance for pair selection (default: no limit)"
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Image size for encoding"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    # RoMA mode options
    parser.add_argument(
        "--mode", type=str, default="global",
        choices=["global", "roma", "both"],
        help="Analysis mode: 'global' (existing), 'roma' (region-based), 'both'"
    )
    parser.add_argument(
        "--roma_setting", type=str, default="precise",
        choices=["precise", "fast", "turbo", "base"],
        help="RoMaV2 setting (only used with --mode roma or both)"
    )
    parser.add_argument(
        "--roma_confidence_threshold", type=float, default=0.8,
        help="Minimum RoMA confidence for valid correspondences"
    )
    parser.add_argument(
        "--precomputed_warps_dir", type=str, default=None,
        help="Directory with precomputed warp .pt files (from precompute_warps.py). "
             "Skips on-the-fly RoMA computation when provided."
    )

    # Multi-GPU parallelization options
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel workers for multi-GPU processing (RoMA mode only)"
    )
    parser.add_argument(
        "--gpu_ids", type=int, nargs='+', default=None,
        help="GPU IDs to use (e.g., 0 1 for GPUs 0 and 1). If not specified, uses num_workers GPUs starting from 0"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    n_models = len(args.checkpoints)
    if len(args.configs) != n_models:
        raise ValueError(f"Number of configs ({len(args.configs)}) must match checkpoints ({n_models})")

    if args.model_names is None:
        args.model_names = [Path(ckpt).parent.parent.name for ckpt in args.checkpoints]
    elif len(args.model_names) != n_models:
        raise ValueError(f"Number of model names ({len(args.model_names)}) must match checkpoints ({n_models})")

    if args.model_types is None:
        args.model_types = ["auto"] * n_models
    elif len(args.model_types) != n_models:
        raise ValueError(f"Number of model types ({len(args.model_types)}) must match checkpoints ({n_models})")

    # Setup
    output_dir = Path("eval_outputs") / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load all models
    models = []
    print("\nLoading models...")
    for ckpt, cfg, name, mtype in zip(args.checkpoints, args.configs, args.model_names, args.model_types):
        print(f"\n  Loading {name} from {ckpt}")
        model, model_type = load_model(checkpoint_path=ckpt, config_path=cfg, model_type=mtype)
        model = model.to(device)
        model.eval()
        models.append((model, model_type, name))

    # Add baseline if requested
    if args.compare_baseline:
        baseline_model, baseline_type = load_f8_baseline_vae(device)
        models.append((baseline_model, baseline_type, "f8 Baseline"))

    # Assign colors
    model_colors = {name: MODEL_COLORS[i % len(MODEL_COLORS)]
                   for i, (_, _, name) in enumerate(models)}

    # Create dataset adapter
    if args.dataset == "co3d":
        adapter = CO3DAdapter(
            co3d_dir=args.co3d_dir,
            annotations_path=args.co3d_annotations,
            num_objects=args.num_objects,
            seed=args.seed,
        )
    elif args.dataset == "co3d_native":
        adapter = NativeCO3DAdapter(
            co3d_root=args.co3d_native_dir,
            num_objects=args.num_objects,
            seed=args.seed,
        )
    else:
        adapter = OmniObjectAdapter(
            data_dir=args.data_dir,
            num_objects=args.num_objects,
            seed=args.seed,
        )

    object_ids = adapter.get_object_ids()

    print(f"\nDataset: {args.dataset}")
    print(f"Analyzing {len(object_ids)} objects with {len(models)} models...")
    print(f"Analysis mode: {args.mode}")

    # Load RoMA model if needed (skip if using precomputed warps)
    roma_model = None
    if args.mode in ["roma", "both"]:
        if args.precomputed_warps_dir:
            print(f"\nUsing precomputed warps from {args.precomputed_warps_dir}")
            print("Skipping RoMA model loading")
        else:
            print(f"\nLoading RoMA model (setting={args.roma_setting})...")
            roma_model = load_roma_model(
                setting=args.roma_setting,
                device=str(device),
                compile=False
            )

    # === Global analysis (existing behavior) ===
    if args.mode in ["global", "both"]:
        print("\n" + "=" * 60)
        print("Running GLOBAL latent analysis...")
        print("=" * 60)

        all_results = {name: [] for _, _, name in models}
        for obj_id in tqdm(object_ids, desc="Processing objects (global)"):
            results_by_model = analyze_object_with_models(
                models, adapter, obj_id, transform, device,
                max_distance=args.max_distance, min_distance=args.min_distance
            )
            for model_name, results in results_by_model.items():
                all_results[model_name].extend(results)

        # Generate aggregate visualizations
        print("\nGenerating global analysis visualizations...")
        visualize_model_comparison(all_results, output_dir, model_colors)
        save_comparison_stats(all_results, output_dir / "comparison_stats.txt")

        # Detailed per-object visualizations
        print(f"\nGenerating detailed visualizations for {args.num_detailed_objects} objects...")
        detailed_ids = object_ids[:args.num_detailed_objects]

        for obj_id in tqdm(detailed_ids, desc="Detailed analysis"):
            object_data = encode_object_views(models, adapter, obj_id, transform, device)
            if object_data:
                visualize_sequence_comparison(object_data, output_dir, model_colors)
                visualize_similarity_matrices(object_data, output_dir, model_colors)
                visualize_angle_vs_similarity_per_object(object_data, output_dir, model_colors)

    # === RoMA region-based analysis ===
    if args.mode in ["roma", "both"]:
        print("\n" + "=" * 60)
        print("Running RoMA REGION-BASED latent analysis...")
        print(f"Confidence threshold: {args.roma_confidence_threshold}")
        print("=" * 60)

        # Determine GPU IDs and number of workers
        num_workers = args.num_workers
        if args.gpu_ids is not None:
            gpu_ids = args.gpu_ids
            num_workers = len(gpu_ids)
        else:
            num_workers = min(num_workers, torch.cuda.device_count())
            gpu_ids = list(range(num_workers))
        
        print(f"Using {num_workers} workers on GPUs: {gpu_ids}")

        roma_results = {name: [] for _, _, name in models}

        # Single-process mode (original behavior)
        if num_workers <= 1:
            print("Running in single-process mode (num_workers=1)")
            roma_model_main = roma_model  # Use already loaded roma_model
            
            for obj_id in tqdm(object_ids, desc="Processing objects (RoMA)"):
                precomputed_dir = Path(args.precomputed_warps_dir) if args.precomputed_warps_dir else None
                results_by_model = analyze_object_with_roma(
                    models, roma_model_main, adapter, obj_id, transform, str(device),
                    max_distance=args.max_distance, min_distance=args.min_distance,
                    confidence_threshold=args.roma_confidence_threshold,
                    image_size=args.image_size,
                    precomputed_warps_dir=precomputed_dir
                )
                for model_name, results in results_by_model.items():
                    roma_results[model_name].extend(results)

        # Multi-process mode (parallel across GPUs)
        else:
            print(f"Running in multi-process mode with {num_workers} workers")

            # Split objects across workers
            object_batches = []
            batch_size = (len(object_ids) + num_workers - 1) // num_workers
            for i in range(num_workers):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(object_ids))
                object_batches.append(object_ids[start_idx:end_idx])

            print(f"Split {len(object_ids)} objects into {num_workers} batches:")
            for i, batch in enumerate(object_batches):
                print(f"  Worker {i} (GPU {gpu_ids[i]}): {len(batch)} objects")

            # Prepare config dicts for workers
            models_config = {
                'checkpoints': args.checkpoints,
                'configs': args.configs,
                'names': args.model_names,
                'types': args.model_types,
                'baseline': args.compare_baseline
            }

            roma_config = {
                'roma_setting': args.roma_setting,
                'confidence_threshold': args.roma_confidence_threshold,
                'image_size': args.image_size,
                'max_distance': args.max_distance,
                'min_distance': args.min_distance,
                'max_pairs': 50,
                'precomputed_warps_dir': args.precomputed_warps_dir,
            }

            transform_config = {
                'image_size': args.image_size
            }

            dataset_config = {
                'type': args.dataset,
                'data_dir': args.data_dir,
                'co3d_dir': args.co3d_dir,
                'co3d_annotations': args.co3d_annotations,
                'co3d_native_dir': args.co3d_native_dir,
            }

            # Launch worker processes
            processes = []
            output_queue = mp.Queue()

            for worker_id, (batch, gpu_id) in enumerate(zip(object_batches, gpu_ids)):
                p = mp.Process(
                    target=worker_process_roma_objects,
                    args=(
                        gpu_id,
                        batch,
                        models_config,
                        roma_config,
                        transform_config,
                        dataset_config,
                        output_queue,
                        worker_id
                    )
                )
                p.start()
                processes.append(p)
            
            # Collect results from workers
            print("Waiting for workers to complete...")
            worker_results_list = []
            for p in processes:
                p.join()
            
            # Get results from queue
            for _ in range(num_workers):
                worker_result = output_queue.get(timeout=10)
                if 'error' in worker_result:
                    print(f"Worker {worker_result['worker_id']} error: {worker_result['error']}")
                else:
                    worker_results_list.append(worker_result)
            
            # Aggregate results
            for worker_result in worker_results_list:
                for model_name, results in worker_result['results'].items():
                    roma_results[model_name].extend(results)
            
            print(f"Aggregated results from {len(worker_results_list)} workers")
            for model_name in roma_results:
                print(f"  {model_name}: {len(roma_results[model_name])} pair results")

        # Generate RoMA visualizations
        print("\nGenerating RoMA analysis visualizations...")
        visualize_roma_model_comparison(roma_results, output_dir, model_colors)
        save_roma_comparison_stats(roma_results, output_dir / "roma_comparison_stats.txt")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
