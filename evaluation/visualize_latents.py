#!/usr/bin/env python
"""
Latent Space Visualization Script for VAE Models.

Visualizes:
1. Per-channel latent distributions and statistics
2. PCA analysis of the latent space

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


def extract_latents(model, dataloader, num_samples, device):
    """Extract latent codes from dataset samples.

    Args:
        model: VAE model
        dataloader: DataLoader with images
        num_samples: Number of samples to extract
        device: Torch device

    Returns:
        dict with:
            - latents: (N, C) flattened latents
            - latents_spatial: (N, C, H, W) spatial latents (first 100)
            - images: (N, 3, H, W) input images (first 100)
    """
    latents = []
    latents_spatial = []
    images = []
    samples_processed = 0

    model.eval()
    
    # Calculate expected number of batches needed
    batch_size = dataloader.batch_size
    batches_needed = (num_samples + batch_size - 1) // batch_size
    total_batches = len(dataloader)
    
    print(f"  Target: {num_samples} samples ({batches_needed} batches needed, {total_batches} total available)")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Extracting latents [0/{num_samples}]", total=min(batches_needed, total_batches))
        for batch in pbar:
            if samples_processed >= num_samples:
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

            # Flatten spatial: average over spatial dims
            z_flat = z.view(z.size(0), z.size(1), -1).mean(dim=2)  # (B, C)
            latents.append(z_flat.cpu())

            # Store spatial latents and images for first 100 samples
            if samples_processed < 100:
                latents_spatial.append(z.cpu())
                images.append(imgs.cpu())

            samples_processed += imgs.size(0)
            
            # Update progress bar description with current count
            pbar.set_description(f"Extracting latents [{min(samples_processed, num_samples)}/{num_samples}]")

    return {
        'latents': torch.cat(latents, dim=0)[:num_samples].numpy(),
        'latents_spatial': torch.cat(latents_spatial, dim=0)[:min(100, num_samples)].numpy() if latents_spatial else None,
        'images': torch.cat(images, dim=0)[:min(100, num_samples)].numpy() if images else None,
    }


def visualize_latents(latents, latents_spatial, images, save_dir):
    """Create comprehensive latent visualization.

    Args:
        latents: (N, C) numpy array of flattened latents
        latents_spatial: (N, C, H, W) numpy array of spatial latents (optional)
        images: (N, 3, H, W) numpy array of images (optional)
        save_dir: Directory to save outputs
    """
    save_dir = Path(save_dir)
    n_samples, n_channels = latents.shape

    fig = plt.figure(figsize=(16, 8))

    # 1. Per-channel distribution histograms (top row)
    for i in range(min(n_channels, 4)):
        ax = fig.add_subplot(2, 4, i + 1)
        ax.hist(latents[:, i], bins=50, density=True, alpha=0.7, color=f'C{i}', edgecolor='black')
        ax.axvline(latents[:, i].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {latents[:, i].mean():.3f}')
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel(f'Channel {i}')
        ax.set_ylabel('Density')
        ax.set_title(f'Channel {i} Distribution')
        ax.legend(fontsize=8)

    # 2. Spatial structure visualization (bottom row) - show latent channels as images
    if latents_spatial is not None and images is not None:
        n_show = min(4, latents_spatial.shape[0])
        for i in range(n_show):
            # Show input image
            ax_img = fig.add_subplot(2, 4, 5 + i)
            img = images[i]
            img = (img * 0.5 + 0.5)  # Denormalize from [-1, 1] to [0, 1]
            img = np.clip(img, 0, 1)
            img = np.transpose(img, (1, 2, 0))
            ax_img.imshow(img)
            ax_img.axis('off')
            ax_img.set_title(f'Input {i}', fontsize=10)

    plt.suptitle(f'Latent Space Analysis (N={n_samples}, C={n_channels})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'latent_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved latent_distributions.png")


def visualize_latent_pca(latents_spatial, images, save_dir):
    """Create 2D PCA visualization of spatial latents.

    Applies PCA per-image to reduce 4-channel latents to 3 channels (RGB) for visualization.
    Shows input images alongside their PCA-reduced latent representations.

    Args:
        latents_spatial: (N, C, H, W) numpy array of spatial latents
        images: (N, 3, H, W) numpy array of input images
        save_dir: Directory to save outputs
    """
    save_dir = Path(save_dir)

    if latents_spatial is None or images is None:
        print("  Warning: No spatial latents available for PCA visualization")
        return

    n_samples, n_channels, h, w = latents_spatial.shape
    n_show = min(8, n_samples)  # Show up to 8 samples

    # Create figure: 3 rows (input, raw latent ch0-2, PCA latent)
    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 3, 9))

    all_explained_variance = []

    for i in range(n_show):
        # Row 1: Input image
        img = images[i]
        img = (img * 0.5 + 0.5)  # Denormalize from [-1, 1] to [0, 1]
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Input', fontsize=12)
        axes[0, i].set_title(f'Sample {i}', fontsize=10)

        # Row 2: Raw latent (first 3 channels as RGB)
        lat_raw = latents_spatial[i][:3]  # (3, H, W)
        lat_raw = np.transpose(lat_raw, (1, 2, 0))  # (H, W, 3)
        lat_raw = (lat_raw - lat_raw.min()) / (lat_raw.max() - lat_raw.min() + 1e-8)
        axes[1, i].imshow(lat_raw)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Latent (Ch 0-2)', fontsize=12)

        # Row 3: PCA latent - apply PCA per image
        # Reshape single image latent: (C, H, W) -> (H*W, C)
        lat_single = latents_spatial[i]  # (C, H, W)
        lat_flat = lat_single.transpose(1, 2, 0).reshape(-1, n_channels)  # (H*W, C)

        # Fit PCA on this single image's latent
        pca = PCA(n_components=3)
        lat_pca_flat = pca.fit_transform(lat_flat)  # (H*W, 3)
        all_explained_variance.append(pca.explained_variance_ratio_)

        # Reshape back: (H*W, 3) -> (H, W, 3)
        lat_pca = lat_pca_flat.reshape(h, w, 3)

        # Normalize to [0, 1] for visualization
        lat_pca_norm = np.zeros_like(lat_pca)
        for c in range(3):
            channel = lat_pca[..., c]
            vmin, vmax = np.percentile(channel, [2, 98])
            lat_pca_norm[..., c] = np.clip((channel - vmin) / (vmax - vmin + 1e-8), 0, 1)

        axes[2, i].imshow(lat_pca_norm)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Latent PCA', fontsize=12)

    # Compute average explained variance
    avg_var = np.mean(all_explained_variance, axis=0)
    var_text = f"Avg PCA explained variance: PC1={avg_var[0]:.1%}, PC2={avg_var[1]:.1%}, PC3={avg_var[2]:.1%}"
    fig.suptitle(f'Latent Space PCA Visualization (per-image)\n{var_text}', fontsize=14, fontweight='bold')
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
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples for latent extraction")
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

    # Load dataset
    print(f"\nLoading {args.dataset_type} dataset from: {args.data_dir}")
    crop_images = not args.no_crop
    print(f"  Crop images: {crop_images}")
    dataset_kwargs = {"image_size": args.image_size, "crop_images": crop_images}

    # Set default bb_file if not provided
    if args.dataset_type in ["co3d", "warp_co3d"] and args.bb_file is None:
        args.bb_file = "/data/lab_moezkan/co3d_bboxes/toybus_test.jgz"
        print(f"  Using default bb_file: {args.bb_file}")

    dataset = load_dataset(
        dataset_type=args.dataset_type,
        data_dir=args.data_dir,
        bb_file=args.bb_file,
        **dataset_kwargs
    )
    print(f"  Dataset size: {len(dataset)}")


    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Extract latents
    print(f"\nExtracting latents from {args.num_samples} samples...")
    latent_data = extract_latents(model, dataloader, args.num_samples, device)
    print(f"  Extracted latents shape: {latent_data['latents'].shape}")

    # Generate visualizations
    print("\nGenerating latent distribution visualization...")
    visualize_latents(
        latents=latent_data['latents'],
        latents_spatial=latent_data['latents_spatial'],
        images=latent_data['images'],
        save_dir=output_dir
    )

    if not args.skip_pca:
        print("\nGenerating PCA visualization...")
        visualize_latent_pca(
            latents_spatial=latent_data['latents_spatial'],
            images=latent_data['images'],
            save_dir=output_dir
        )

    # Summary
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}/")
    print("  - latent_distributions.png")
    if not args.skip_pca:
        print("  - latent_pca.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
