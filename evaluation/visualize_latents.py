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

    # Also check actual channel size from first conv layer
    ch = 64  # default
    for k, v in state_dict.items():
        if 'encoder.down.0.block.0.conv1.weight' in k:
            ch = v.shape[0]  # Output channels
            break

    if has_down_3:
        # Full SD architecture (4 down blocks)
        return ch, [1, 2, 4, 4]
    else:
        # 3-block architecture
        return ch, [1, 2, 4]


def load_model(checkpoint_path, config_path=None, use_vanilla_sd=False):
    """Load VAE model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config YAML for model instantiation
        use_vanilla_sd: Force vanilla SD architecture (ch=128, ch_mult=[1,2,4,4])

    Returns:
        model: Loaded VAE model
    """
    from ldm.models.autoencoder import AutoencoderKL

    # If config provided, use it
    if config_path:
        import yaml
        from ldm.util import instantiate_from_config

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model = instantiate_from_config(config['model'])
        print(f"Model instantiated from config: {config['model']['target']}")
    else:
        # Auto-detect or use specified architecture
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract state dict
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            # Handle first_stage_model prefix (SD format)
            if any("first_stage_model" in k for k in state_dict.keys()):
                print("  Detected SD checkpoint format (first_stage_model prefix)")
                state_dict = {
                    k.replace("first_stage_model.", ""): v
                    for k, v in state_dict.items()
                    if "first_stage_model" in k
                }
            # Handle model. prefix (training format)
            elif any(k.startswith("model.") for k in state_dict.keys()):
                print("  Detected training checkpoint format (model. prefix)")
                state_dict = {
                    k.replace("model.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("model.")
                }
        else:
            state_dict = ckpt

        # Detect architecture from state dict
        if use_vanilla_sd:
            ch, ch_mult = 128, [1, 2, 4, 4]
            print("  Using vanilla SD architecture (ch=128, ch_mult=[1,2,4,4])")
        else:
            ch, ch_mult = detect_architecture(state_dict)
            print(f"  Detected architecture: ch={ch}, ch_mult={ch_mult}")

        # Create model config
        ddconfig = {
            "double_z": True,
            "z_channels": 4,
            "resolution": 256 if use_vanilla_sd else 128,
            "in_channels": 3,
            "out_ch": 3,
            "ch": ch,
            "ch_mult": ch_mult,
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }

        lossconfig = {
            "target": "ldm.modules.losses.LPIPSWithDiscriminator",
            "params": {
                "disc_start": 50001,
                "kl_weight": 0.000001,
                "disc_weight": 0.5,
                "perceptual_weight": 1.0,
                "disc_in_channels": 3,
                "disc_num_layers": 3 if use_vanilla_sd else 2,
                "use_actnorm": False,
            },
        }

        model = AutoencoderKL(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            embed_dim=4,
        )

        # Load weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys")
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

    if not data_by_subcat:
        print("  Warning: No data available for visualization")
        return

    subcategories = list(data_by_subcat.keys())
    n_subcats = len(subcategories)
    
    # Get number of latent channels from first subcategory
    first_subcat = subcategories[0]
    n_channels = data_by_subcat[first_subcat]['latents_spatial'].shape[1]
    
    # Create figure: (1 + n_channels) rows per subcategory, n_samples_per_subcat columns
    n_rows_per_subcat = 1 + n_channels  # input + each latent channel
    total_rows = n_subcats * n_rows_per_subcat
    
    fig, axes = plt.subplots(total_rows, n_samples_per_subcat, 
                              figsize=(n_samples_per_subcat * 2.5, total_rows * 2))
    
    # Handle case where axes might be 1D
    if total_rows == 1:
        axes = axes.reshape(1, -1)
    if n_samples_per_subcat == 1:
        axes = axes.reshape(-1, 1)

    for subcat_idx, subcat_name in enumerate(subcategories):
        subcat_data = data_by_subcat[subcat_name]
        images = subcat_data['images']
        latents_spatial = subcat_data['latents_spatial']
        
        n_available = min(n_samples_per_subcat, images.shape[0])
        base_row = subcat_idx * n_rows_per_subcat
        
        for i in range(n_samples_per_subcat):
            if i < n_available:
                # Row 0: Input image
                img = images[i]
                img = (img * 0.5 + 0.5)  # Denormalize from [-1, 1] to [0, 1]
                img = np.clip(img, 0, 1)
                img = np.transpose(img, (1, 2, 0))
                axes[base_row, i].imshow(img)
                
                # Rows 1 to n_channels: Individual latent channels
                for c in range(n_channels):
                    lat_channel = latents_spatial[i, c]  # (H, W)
                    vmin, vmax = np.percentile(lat_channel, [2, 98])
                    lat_norm = np.clip((lat_channel - vmin) / (vmax - vmin + 1e-8), 0, 1)
                    axes[base_row + c + 1, i].imshow(lat_norm, cmap='viridis')
            
            # Turn off axes for all
            for row_offset in range(n_rows_per_subcat):
                axes[base_row + row_offset, i].axis('off')
        
        # Add y-labels for the first column
        axes[base_row, 0].set_ylabel(f'{subcat_name}\nInput', fontsize=10)
        for c in range(n_channels):
            axes[base_row + c + 1, 0].set_ylabel(f'Ch {c}', fontsize=10)
        
        # Re-enable y-axis label display
        for row_offset in range(n_rows_per_subcat):
            axes[base_row + row_offset, 0].yaxis.set_visible(True)
            axes[base_row + row_offset, 0].yaxis.label.set_visible(True)
        
        # Add title for first sample of each subcategory
        if subcat_idx == 0:
            for i in range(n_samples_per_subcat):
                axes[base_row, i].set_title(f'Sample {i}', fontsize=10)

    plt.suptitle(f'Latent Samples by Subcategory ({n_subcats} subcategories, C={n_channels})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'latent_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved latent_samples.png")


def visualize_latent_pca_by_subcategory(data_by_subcat, save_dir, n_samples_per_subcat=5):
    """Create PCA visualization showing samples from multiple subcategories.

    Args:
        data_by_subcat: Dict mapping subcategory name to dict with latents_spatial and images
        save_dir: Directory to save outputs
        n_samples_per_subcat: Number of samples to show per subcategory
    """
    save_dir = Path(save_dir)

    if not data_by_subcat:
        print("  Warning: No data available for PCA visualization")
        return

    subcategories = list(data_by_subcat.keys())
    n_subcats = len(subcategories)
    
    # Get dimensions from first subcategory
    first_subcat = subcategories[0]
    n_channels = data_by_subcat[first_subcat]['latents_spatial'].shape[1]
    
    # Create figure: 3 rows per subcategory (input, raw latent, PCA)
    n_rows_per_subcat = 3
    total_rows = n_subcats * n_rows_per_subcat
    
    fig, axes = plt.subplots(total_rows, n_samples_per_subcat, 
                              figsize=(n_samples_per_subcat * 2.5, total_rows * 2.5))
    
    if total_rows == 1:
        axes = axes.reshape(1, -1)
    if n_samples_per_subcat == 1:
        axes = axes.reshape(-1, 1)

    all_explained_variance = []

    for subcat_idx, subcat_name in enumerate(subcategories):
        subcat_data = data_by_subcat[subcat_name]
        images = subcat_data['images']
        latents_spatial = subcat_data['latents_spatial']
        
        n_available = min(n_samples_per_subcat, images.shape[0])
        _, _, h, w = latents_spatial.shape
        base_row = subcat_idx * n_rows_per_subcat
        
        for i in range(n_samples_per_subcat):
            if i < n_available:
                # Row 0: Input image
                img = images[i]
                img = (img * 0.5 + 0.5)
                img = np.clip(img, 0, 1)
                img = np.transpose(img, (1, 2, 0))
                axes[base_row, i].imshow(img)

                # Row 1: Raw latent (first 3 channels as RGB)
                lat_raw = latents_spatial[i][:3]
                lat_raw = np.transpose(lat_raw, (1, 2, 0))
                lat_raw = (lat_raw - lat_raw.min()) / (lat_raw.max() - lat_raw.min() + 1e-8)
                axes[base_row + 1, i].imshow(lat_raw)

                # Row 2: PCA latent
                lat_single = latents_spatial[i]
                lat_flat = lat_single.transpose(1, 2, 0).reshape(-1, n_channels)

                pca = PCA(n_components=3)
                lat_pca_flat = pca.fit_transform(lat_flat)
                all_explained_variance.append(pca.explained_variance_ratio_)

                lat_pca = lat_pca_flat.reshape(h, w, 3)
                lat_pca_norm = np.zeros_like(lat_pca)
                for c in range(3):
                    channel = lat_pca[..., c]
                    vmin, vmax = np.percentile(channel, [2, 98])
                    lat_pca_norm[..., c] = np.clip((channel - vmin) / (vmax - vmin + 1e-8), 0, 1)

                axes[base_row + 2, i].imshow(lat_pca_norm)
            
            # Turn off axes
            for row_offset in range(n_rows_per_subcat):
                axes[base_row + row_offset, i].axis('off')
        
        # Add y-labels
        axes[base_row, 0].set_ylabel(f'{subcat_name}\nInput', fontsize=10)
        axes[base_row + 1, 0].set_ylabel('Latent', fontsize=10)
        axes[base_row + 2, 0].set_ylabel('PCA', fontsize=10)
        
        for row_offset in range(n_rows_per_subcat):
            axes[base_row + row_offset, 0].yaxis.set_visible(True)
            axes[base_row + row_offset, 0].yaxis.label.set_visible(True)
        
        if subcat_idx == 0:
            for i in range(n_samples_per_subcat):
                axes[base_row, i].set_title(f'Sample {i}', fontsize=10)

    # Compute average explained variance
    if all_explained_variance:
        avg_var = np.mean(all_explained_variance, axis=0)
        var_text = f"Avg PCA explained variance: PC1={avg_var[0]:.1%}, PC2={avg_var[1]:.1%}, PC3={avg_var[2]:.1%}"
    else:
        var_text = ""
    
    fig.suptitle(f'Latent Space PCA by Subcategory ({n_subcats} subcategories)\n{var_text}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'latent_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved latent_pca.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize VAE latent space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
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

    # Model options
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config YAML for model instantiation")
    parser.add_argument("--vanilla_sd", action="store_true",
                        help="Force vanilla SD architecture (ch=128, ch_mult=[1,2,4,4])")

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
        use_vanilla_sd=args.vanilla_sd
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
    print("  - latent_samples.png")
    if not args.skip_pca:
        print("  - latent_pca.png")
    print(f"Visualized {len(selected_subcats)} subcategories: {selected_subcats}")
    print("=" * 60)


if __name__ == "__main__":
    main()
