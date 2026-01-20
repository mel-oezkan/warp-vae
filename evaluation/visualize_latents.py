#!/usr/bin/env python
"""
Latent Space Visualization Script for VAE Models.

Visualizes:
1. PCA analysis of the latent space 
    - show original image
    - show original latent channels mean
    - show PCA-reduced latent channels


Usage:
    python evaluation/visualize_latents.py \
        --checkpoint outputs/warp_vae/checkpoints/last.ckpt \
        --config configs/vae_config.yaml \
        --output_name my_experiment \
        --dataset_type co3d \
        --data_dir /data/lab_moezkan/co3d_full/toybus \
        --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz
"""

import os
import sys
import argparse
import gzip
import json
from typing import IO, cast

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Optional safetensors import
try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] range."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def tensor_to_numpy(tensor):
    """Convert tensor to numpy for visualization."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    return denormalize(tensor).permute(1, 2, 0).cpu().numpy()


def detect_architecture(state_dict):
    """Detect model architecture from state dict keys and weight shapes.

    Returns:
        tuple: (ch, ch_mult) for model config
    """
    # Check for encoder down blocks to determine architecture
    # Small model: ch=64, ch_mult=[1,2,4] -> 3 down blocks
    # Full SD: ch=128, ch_mult=[1,2,4,4] -> 4 down blocks

    has_down_3 = any("encoder.down.3" in k for k in state_dict.keys())

    # Check actual channel size from conv_in layer (most reliable)
    ch = None
    for k, v in state_dict.items():
        if k == 'encoder.conv_in.weight' or k.endswith('encoder.conv_in.weight'):
            ch = v.shape[0]  # Output channels of first conv
            break
    
    # Fallback: check from down blocks
    if ch is None:
        for k, v in state_dict.items():
            if 'encoder.down.0.block.0.conv1.weight' in k:
                ch = v.shape[0]  # Output channels
                break
    
    # Default to SD architecture if we can't detect
    if ch is None:
        ch = 128
        print("  Warning: Could not detect ch from state_dict, defaulting to 128")

    if has_down_3:
        # Full SD architecture (4 down blocks)
        return ch, [1, 2, 4, 4]
    else:
        # 3-block architecture
        return ch, [1, 2, 4]


def load_model(checkpoint_path, config_path):
    """Load VAE model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt, .pt, or .safetensors)
        config_path: Path to config file (JSON for diffusers-style or YAML for LDM-style)

    Returns:
        model: Loaded VAE model
    """
    from ldm.models.autoencoder import AutoencoderKL

    # Convert to string for extension checking
    checkpoint_path_str = str(checkpoint_path)
    config_path_str = str(config_path)
    
    print(f"Loading checkpoint from {checkpoint_path_str}")
    print(f"Loading config from {config_path_str}")
    
    # Load state dict from checkpoint
    if checkpoint_path_str.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors package is required to load .safetensors files. "
                "Install it with: pip install safetensors"
            )
        print("  Loading safetensors format...")
        state_dict = load_safetensors(checkpoint_path_str)
    else:
        print("  Loading PyTorch checkpoint format...")
        ckpt = torch.load(checkpoint_path_str, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

    # Handle key prefixes
    if any("first_stage_model" in k for k in state_dict.keys()):
        print("  Detected SD checkpoint format (first_stage_model prefix)")
        state_dict = {
            k.replace("first_stage_model.", ""): v
            for k, v in state_dict.items()
            if "first_stage_model" in k
        }
    elif any(k.startswith("model.") for k in state_dict.keys()):
        print("  Detected training checkpoint format (model. prefix)")
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

    # Load config and create model
    if config_path_str.endswith('.json'):
        # Diffusers-style config.json
        with open(config_path_str, 'r') as f:
            config = json.load(f)
        
        # Convert diffusers config to ldm ddconfig
        block_out_channels = config.get('block_out_channels', [128, 256, 512, 512])
        ch = block_out_channels[0]
        ch_mult = [c // ch for c in block_out_channels]
        
        ddconfig = {
            "double_z": True,
            "z_channels": config.get('latent_channels', 4),
            "resolution": config.get('sample_size', 256),
            "in_channels": config.get('in_channels', 3),
            "out_ch": config.get('out_channels', 3),
            "ch": ch,
            "ch_mult": ch_mult,
            "num_res_blocks": config.get('layers_per_block', 2),
            "attn_resolutions": [],
            "dropout": 0.0,
        }
        
        print(f"  Parsed diffusers config: ch={ch}, ch_mult={ch_mult}")
        print(f"    latent_channels: {ddconfig['z_channels']}")
        
        model = AutoencoderKL(
            ddconfig=ddconfig,
            lossconfig={"target": "torch.nn.Identity"},
            embed_dim=ddconfig['z_channels'],
        )
        
    elif config_path_str.endswith('.yaml') or config_path_str.endswith('.yml'):
        # LDM-style YAML config with OmegaConf for variable interpolation
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config

        # Load with OmegaConf to resolve ${} interpolations
        yaml_config = OmegaConf.load(config_path_str)
        
        # Resolve all interpolations
        OmegaConf.resolve(yaml_config)
        
        # Convert to plain dict for instantiate_from_config
        yaml_config = OmegaConf.to_container(yaml_config, resolve=True)

        if 'model' in yaml_config:
            model = instantiate_from_config(yaml_config['model'])
            print(f"  Model instantiated from YAML config")
        else:
            raise ValueError("YAML config must contain 'model' key")
    else:
        raise ValueError(f"Unknown config file format: {config_path_str}. Use .json or .yaml/.yml")

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
        if len(missing) <= 10:
            for k in missing:
                print(f"    - {k}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
        if len(unexpected) <= 10:
            for k in unexpected:
                print(f"    - {k}")
    print("  Checkpoint loaded successfully")

    return model


def get_subcategories_from_bb_file(bb_file: str, min_samples: int = 5) -> list:
    """Extract subcategory names from bounding box file.
    
    Args:
        bb_file: Path to gzipped JSON bounding box file
        min_samples: Minimum number of samples required per subcategory
        
    Returns:
        List of subcategory names with at least min_samples samples
    """
    with gzip.GzipFile(bb_file, "rb") as f:
        obj_dict = json.loads(cast(IO, f).read().decode("utf8"))
    
    # Filter subcategories with enough samples
    subcategories = [name for name, samples in obj_dict.items() 
                     if len(samples) >= min_samples]
    
    return sorted(subcategories)


def load_dataset(dataset_type, data_dir, image_size=256, bb_file=None, crop_images=True, **kwargs):
    """Load dataset based on type.

    Args:
        dataset_type: One of 'co3d', 'omniobject', 'warp_co3d'
        data_dir: Root directory of the dataset
        image_size: Input image size
        bb_file: Bounding box file (required for co3d datasets)
        crop_images: Whether to crop images based on bounding boxes
        **kwargs: Additional dataset arguments

    Returns:
        dataset: PyTorch dataset
    """
    if dataset_type == "co3d":
        from src.data.co3d_dataset import CO3DDataset

        if bb_file is None:
            raise ValueError("bb_file is required for CO3D dataset")

        dataset = CO3DDataset(
            root_dir=data_dir,
            bb_file=bb_file,
            image_size=image_size,
            include_plucker=False,
            crop_images=crop_images,
            **kwargs
        )
    elif dataset_type == "omniobject":
        from src.data.omniobject3d_dataset import OmniObject3DDataset

        dataset = OmniObject3DDataset(
            root_dir=data_dir,
            image_size=image_size,
            **kwargs
        )
    elif dataset_type == "warp_co3d":
        from src.data.warp_dataset import WarpCO3DDataset

        if bb_file is None:
            raise ValueError("bb_file is required for WarpCO3D dataset")

        dataset = WarpCO3DDataset(
            root_dir=data_dir,
            bb_file=bb_file,
            image_size=image_size,
            crop_images=crop_images,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return dataset


def extract_latents_by_subcategory(model, datasets_by_subcat, num_samples_per_subcat, device):
    """Extract latent codes from multiple subcategories.

    Args:
        model: VAE model
        datasets_by_subcat: Dict mapping subcategory name to dataset
        num_samples_per_subcat: Number of samples to extract per subcategory
        device: Torch device

    Returns:
        dict mapping subcategory name to dict with:
            - latents_spatial: (N, C, H, W) spatial latents
            - images: (N, 3, H, W) input images
    """
    from torch.utils.data import DataLoader
    
    results = {}
    model.eval()
    
    for subcat_name, dataset in datasets_by_subcat.items():
        print(f"  Processing subcategory: {subcat_name} ({len(dataset)} samples available)")
        
        dataloader = DataLoader(
            dataset,
            batch_size=min(8, num_samples_per_subcat),
            shuffle=True,  # Shuffle to get variety
            num_workers=2,
            pin_memory=True
        )
        
        latents_spatial = []
        images = []
        samples_processed = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_processed >= num_samples_per_subcat:
                    break

                # Get images from batch
                if isinstance(batch, dict):
                    imgs = batch.get('image', batch.get('images'))
                else:
                    imgs = batch[0]

                imgs = imgs.to(device)

                # Encode
                if hasattr(model, 'ema_scope') and hasattr(model, 'model_ema') and model.model_ema is not None:
                    with model.ema_scope():
                        posterior = model.encode(imgs)
                else:
                    posterior = model.encode(imgs)

                z = posterior.sample()  # (B, C, H', W')

                latents_spatial.append(z.cpu())
                images.append(imgs.cpu())

                samples_processed += imgs.size(0)
        
        if latents_spatial:
            results[subcat_name] = {
                'latents_spatial': torch.cat(latents_spatial, dim=0)[:num_samples_per_subcat].numpy(),
                'images': torch.cat(images, dim=0)[:num_samples_per_subcat].numpy(),
            }
    
    return results


def visualize_latents_by_subcategory(data_by_subcat, save_dir, n_samples_per_subcat=5):
    """Create latent visualization showing samples from multiple subcategories.

    Args:
        data_by_subcat: Dict mapping subcategory name to dict with latents_spatial and images
        save_dir: Directory to save outputs
        n_samples_per_subcat: Number of samples to show per subcategory
    """
    save_dir = Path(save_dir)
    latents_dir = save_dir / 'latent_samples'
    latents_dir.mkdir(parents=True, exist_ok=True)

    if not data_by_subcat:
        print("  Warning: No data available for visualization")
        return

    for subcat_name, subcat_data in data_by_subcat.items():
        images = subcat_data['images']
        latents_spatial = subcat_data['latents_spatial']
        
        n_channels = latents_spatial.shape[1]
        n_available = min(n_samples_per_subcat, images.shape[0])
        
        # Create figure: (1 + n_channels) rows, n_samples columns
        n_rows = 1 + n_channels  # input + each latent channel
        
        fig, axes = plt.subplots(n_rows, n_available, 
                                  figsize=(n_available * 2.5, n_rows * 2))
        
        # Handle case where axes might be 1D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_available == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_available):
            # Row 0: Input image
            img = images[i]
            img = (img * 0.5 + 0.5)  # Denormalize from [-1, 1] to [0, 1]
            img = np.clip(img, 0, 1)
            img = np.transpose(img, (1, 2, 0))
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Sample {i}', fontsize=10)
            
            # Rows 1 to n_channels: Individual latent channels
            for c in range(n_channels):
                lat_channel = latents_spatial[i, c]  # (H, W)
                vmin, vmax = np.percentile(lat_channel, [2, 98])
                lat_norm = np.clip((lat_channel - vmin) / (vmax - vmin + 1e-8), 0, 1)
                axes[c + 1, i].imshow(lat_norm, cmap='viridis')
            
            # Turn off axes for all
            for row in range(n_rows):
                axes[row, i].axis('off')
        
        # Add y-labels for the first column
        axes[0, 0].set_ylabel('Input', fontsize=10)
        for c in range(n_channels):
            axes[c + 1, 0].set_ylabel(f'Ch {c}', fontsize=10)
        
        # Re-enable y-axis label display
        for row in range(n_rows):
            axes[row, 0].yaxis.set_visible(True)
            axes[row, 0].yaxis.label.set_visible(True)

        plt.suptitle(f'Latent Samples: {subcat_name} (C={n_channels})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(latents_dir / f'{subcat_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved latent_samples/{subcat_name}.png")


def visualize_latent_pca_by_subcategory(data_by_subcat, save_dir, n_samples_per_subcat=5):
    """Create PCA visualization showing samples from multiple subcategories.

    Args:
        data_by_subcat: Dict mapping subcategory name to dict with latents_spatial and images
        save_dir: Directory to save outputs
        n_samples_per_subcat: Number of samples to show per subcategory
    """
    save_dir = Path(save_dir)
    pca_dir = save_dir / 'latent_pca'
    pca_dir.mkdir(parents=True, exist_ok=True)

    if not data_by_subcat:
        print("  Warning: No data available for PCA visualization")
        return

    for subcat_name, subcat_data in data_by_subcat.items():
        images = subcat_data['images']
        latents_spatial = subcat_data['latents_spatial']
        
        n_channels = latents_spatial.shape[1]
        n_available = min(n_samples_per_subcat, images.shape[0])
        _, _, h, w = latents_spatial.shape
        
        # Create figure: 3 rows (input, raw latent, PCA), n_samples columns
        n_rows = 3
        
        fig, axes = plt.subplots(n_rows, n_available, 
                                  figsize=(n_available * 2.5, n_rows * 2.5))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_available == 1:
            axes = axes.reshape(-1, 1)

        explained_variances = []

        for i in range(n_available):
            # Row 0: Input image
            img = images[i]
            img = (img * 0.5 + 0.5)
            img = np.clip(img, 0, 1)
            img = np.transpose(img, (1, 2, 0))
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Sample {i}', fontsize=10)

            # Row 1: Raw latent (first 3 channels as RGB)
            lat_raw = latents_spatial[i][:3]
            lat_raw = np.transpose(lat_raw, (1, 2, 0))
            lat_raw = (lat_raw - lat_raw.min()) / (lat_raw.max() - lat_raw.min() + 1e-8)
            axes[1, i].imshow(lat_raw)

            # Row 2: PCA latent
            lat_single = latents_spatial[i]
            lat_flat = lat_single.transpose(1, 2, 0).reshape(-1, n_channels)

            pca = PCA(n_components=3)
            lat_pca_flat = pca.fit_transform(lat_flat)
            explained_variances.append(pca.explained_variance_ratio_)

            lat_pca = lat_pca_flat.reshape(h, w, 3)
            lat_pca_norm = np.zeros_like(lat_pca)
            for c in range(3):
                channel = lat_pca[..., c]
                vmin, vmax = np.percentile(channel, [2, 98])
                lat_pca_norm[..., c] = np.clip((channel - vmin) / (vmax - vmin + 1e-8), 0, 1)

            axes[2, i].imshow(lat_pca_norm)
            
            # Turn off axes
            for row in range(n_rows):
                axes[row, i].axis('off')
        
        # Add y-labels
        axes[0, 0].set_ylabel('Input', fontsize=10)
        axes[1, 0].set_ylabel('Latent', fontsize=10)
        axes[2, 0].set_ylabel('PCA', fontsize=10)
        
        for row in range(n_rows):
            axes[row, 0].yaxis.set_visible(True)
            axes[row, 0].yaxis.label.set_visible(True)

        # Compute average explained variance for this subcategory
        if explained_variances:
            avg_var = np.mean(explained_variances, axis=0)
            var_text = f"Avg PCA: PC1={avg_var[0]:.1%}, PC2={avg_var[1]:.1%}, PC3={avg_var[2]:.1%}"
        else:
            var_text = ""
        
        fig.suptitle(f'Latent PCA: {subcat_name}\n{var_text}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(pca_dir / f'{subcat_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved latent_pca/{subcat_name}.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize VAE latent space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file (JSON or YAML)")
    parser.add_argument("--output_name", type=str, required=True,
                        help="Subfolder name under eval_outputs/")

    # Dataset options
    parser.add_argument("--dataset_type", type=str, default="co3d",
                        choices=["co3d", "omniobject", "warp_co3d"],
                        help="Dataset type to use")
    parser.add_argument("--data_dir", type=str, default="/data/lab_moezkan/co3d_full",
                        help="Dataset root directory (parent of category folders)")
    parser.add_argument("--bb_file", type=str, default=None,
                        help="Bounding box file for CO3D datasets")
    parser.add_argument("--no_crop", action="store_true",
                        help="Disable cropping images based on bounding boxes")

    # Visualization options
    parser.add_argument("--num_subcategories", type=int, default=5,
                        help="Number of subcategories to visualize")
    parser.add_argument("--num_samples_per_subcat", type=int, default=5,
                        help="Number of samples per subcategory")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Input image size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")

    # Optional flags
    parser.add_argument("--skip_pca", action="store_true",
                        help="Skip PCA visualization")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path("eval_outputs") / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )
    model = model.to(device)
    model.eval()

    # Set default bb_file if not provided
    if args.dataset_type in ["co3d", "warp_co3d"] and args.bb_file is None:
        args.bb_file = "/data/lab_moezkan/co3d_bboxes/toybus_test.jgz"
        print(f"  Using default bb_file: {args.bb_file}")

    # Discover subcategories
    print(f"\nDiscovering subcategories from: {args.bb_file}")
    subcategories = get_subcategories_from_bb_file(args.bb_file, min_samples=args.num_samples_per_subcat)
    print(f"  Found {len(subcategories)} subcategories with >= {args.num_samples_per_subcat} samples")
    
    # Select subset of subcategories
    selected_subcats = subcategories[:args.num_subcategories]
    print(f"  Selected subcategories: {selected_subcats}")

    # Load datasets for each subcategory
    print(f"\nLoading datasets for {len(selected_subcats)} subcategories...")
    crop_images = not args.no_crop
    
    # Create temporary bb files for each subcategory
    datasets_by_subcat = {}
    
    # Load full bb file
    with gzip.GzipFile(args.bb_file, "rb") as f:
        full_obj_dict = json.loads(cast(IO, f).read().decode("utf8"))
    
    for subcat_name in selected_subcats:
        # Create a temporary dict with just this subcategory
        subcat_dict = {subcat_name: full_obj_dict[subcat_name]}
        
        # Write temporary bb file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jgz', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            with gzip.GzipFile(fileobj=tmp_file, mode='wb') as gz:
                gz.write(json.dumps(subcat_dict).encode('utf8'))
        
        try:
            dataset = load_dataset(
                dataset_type=args.dataset_type,
                data_dir=args.data_dir,
                bb_file=tmp_path,
                image_size=args.image_size,
                crop_images=crop_images,
            )
            datasets_by_subcat[subcat_name] = dataset
            print(f"    {subcat_name}: {len(dataset)} samples")
        finally:
            os.unlink(tmp_path)

    # Extract latents by subcategory
    print(f"\nExtracting latents ({args.num_samples_per_subcat} samples per subcategory)...")
    data_by_subcat = extract_latents_by_subcategory(
        model, datasets_by_subcat, args.num_samples_per_subcat, device
    )

    # Generate visualizations
    print("\nGenerating latent visualization...")
    visualize_latents_by_subcategory(
        data_by_subcat=data_by_subcat,
        save_dir=output_dir,
        n_samples_per_subcat=args.num_samples_per_subcat
    )

    if not args.skip_pca:
        print("\nGenerating PCA visualization...")
        visualize_latent_pca_by_subcategory(
            data_by_subcat=data_by_subcat,
            save_dir=output_dir,
            n_samples_per_subcat=args.num_samples_per_subcat
        )

    # Summary
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}/")
    print(f"  - latent_samples/ ({len(selected_subcats)} files)")
    if not args.skip_pca:
        print(f"  - latent_pca/ ({len(selected_subcats)} files)")
    print(f"Visualized {len(selected_subcats)} subcategories: {selected_subcats}")
    print("=" * 60)


if __name__ == "__main__":
    main()
