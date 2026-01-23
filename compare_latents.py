#!/usr/bin/env python
"""
Compare latent representations between different VAE models on various datasets.

Visualizes for each dataset (CO3D, ImageNet, OmniObject3D):
1. Reconstruction grid (original vs reconstructed)
2. Latent channel visualization for random samples
3. PCA of latents

Usage:
    python compare_latents.py \
        --checkpoint outputs/my_model/checkpoints/last.ckpt \
        --config configs/vae_config.yaml \
        --output_name my_experiment
"""

import os
import sys
import argparse
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# CUDA_VISIBLE_DEVICES should be set externally, not hardcoded
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Optional imports
try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    from diffusers import AutoencoderKL as DiffusersVAE
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] range."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def tensor_to_numpy(tensor):
    """Convert tensor to numpy for visualization."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    return denormalize(tensor).permute(1, 2, 0).cpu().numpy()


def load_model(checkpoint_path, config_path, model_type="auto"):
    """Load VAE model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt, .pt, or .safetensors)
        config_path: Path to config file (JSON for diffusers-style or YAML for LDM-style)
        model_type: One of 'auto', 'ldm', 'eqvae', 'diffusers'

    Returns:
        model: Loaded VAE model
        model_type: Detected or specified model type
    """
    checkpoint_path_str = str(checkpoint_path)
    config_path_str = str(config_path)

    print(f"Loading checkpoint from {checkpoint_path_str}")
    print(f"Loading config from {config_path_str}")

    # Detect model type from config if auto
    if model_type == "auto":
        if config_path_str.endswith('.json'):
            model_type = "diffusers"
        else:
            # Check YAML for model type
            from omegaconf import OmegaConf
            yaml_config = OmegaConf.load(config_path_str)
            if 'model' in yaml_config:
                target = yaml_config.model.get('target', '')
                if 'EQVAEAutoencoder' in target:
                    model_type = "eqvae"
                else:
                    model_type = "ldm"
            else:
                model_type = "ldm"
        print(f"  Detected model type: {model_type}")

    # Load state dict from checkpoint
    if checkpoint_path_str.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors package required")
        state_dict = load_safetensors(checkpoint_path_str)
    else:
        ckpt = torch.load(checkpoint_path_str, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

    # Handle key prefixes
    if any("first_stage_model" in k for k in state_dict.keys()):
        state_dict = {
            k.replace("first_stage_model.", ""): v
            for k, v in state_dict.items()
            if "first_stage_model" in k
        }
    elif any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

    # Load model based on type
    if model_type == "diffusers":
        # For diffusers format, load using diffusers library directly
        # This handles the different weight naming conventions properly
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers package required for loading diffusers-format models")

        # Get the directory containing config.json and safetensors
        checkpoint_dir = Path(checkpoint_path_str).parent
        print(f"  Loading diffusers VAE from directory: {checkpoint_dir}")

        model = DiffusersVAE.from_pretrained(str(checkpoint_dir))
        print("  Diffusers VAE loaded successfully")

        # Return early - diffusers models don't need state_dict loading
        return model, "sdvae"
        
    elif model_type == "eqvae":
        from omegaconf import OmegaConf
        from ldm.models.autoencoder import EQVAEAutoencoder
        
        config = OmegaConf.load(config_path_str)
        OmegaConf.resolve(config)
        model_params = config.model.params
        
        model = EQVAEAutoencoder(
            ddconfig=OmegaConf.to_container(model_params.ddconfig),
            lossconfig=OmegaConf.to_container(model_params.lossconfig),
            embed_dim=model_params.embed_dim,
            p_prior=model_params.get("p_prior", 0.9),
            scale_range=model_params.get("scale_range", [0.25, 1.0]),
            use_rotation=model_params.get("use_rotation", True),
        )
        
    else:  # ldm
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        
        yaml_config = OmegaConf.load(config_path_str)
        OmegaConf.resolve(yaml_config)
        yaml_config = OmegaConf.to_container(yaml_config, resolve=True)
        
        if 'model' in yaml_config:
            model = instantiate_from_config(yaml_config['model'])
        else:
            raise ValueError("YAML config must contain 'model' key")

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    print("  Checkpoint loaded successfully")

    return model, model_type


def load_sd_vae(device="cuda"):
    """Load SD-VAE (ft-mse) from HuggingFace.

    This loads the fine-tuned MSE variant of the Stable Diffusion VAE,
    which was trained with MSE loss for better reconstruction quality.
    """
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("diffusers package required for SD-VAE")

    model = DiffusersVAE.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    )
    model = model.to(device)
    model.eval()
    print("Loaded SD-VAE (ft-mse) from HuggingFace")
    return model


def load_dataset(dataset_type, data_dir, image_size=256, bb_file=None, crop_images=True, **kwargs):
    """Load dataset based on type."""
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
    elif dataset_type == "imagenet":
        dataset = ImageNetDataset(
            root_dir=data_dir,
            image_size=image_size,
        )
    elif dataset_type == "omniobject":
        from src.data.omniobject3d_dataset import OmniObject3DDataset
        dataset = OmniObject3DDataset(
            root_dir=data_dir,
            image_size=image_size,
            include_plucker=False,
            sample_mode="single",
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


class ImageNetDataset(torch.utils.data.Dataset):
    """Simple ImageNet dataset for loading random images."""
    
    def __init__(self, root_dir, image_size=256, max_images=10000):
        self.root = Path(root_dir)
        self.image_size = image_size
        
        # Collect image paths
        self.image_paths = []
        for subdir in self.root.iterdir():
            if subdir.is_dir():
                for ext in ["*.jpg", "*.JPEG", "*.png"]:
                    self.image_paths.extend(list(subdir.glob(ext)))
                if len(self.image_paths) > max_images:
                    break
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {"image": self.transform(img)}


@torch.no_grad()
def encode_images(model, images, device, model_type="ldm"):
    """Encode images with VAE model.

    Args:
        model: VAE model (either diffusers or LDM-style)
        images: Input images tensor in [-1, 1] range
        device: Device to run on
        model_type: Either "sdvae" for diffusers or "ldm"/"eqvae" for LDM-style

    Returns:
        Latent tensor (unscaled, raw latent space)
    """
    images = images.to(device)

    if model_type == "sdvae":
        # Diffusers VAE returns unscaled latents from encode()
        # We sample from the latent distribution directly (no scaling applied here)
        latent = model.encode(images).latent_dist.sample()
    else:
        # LDM models may have EMA weights
        if hasattr(model, 'ema_scope') and hasattr(model, 'model_ema') and model.model_ema is not None:
            with model.ema_scope():
                posterior = model.encode(images)
        else:
            posterior = model.encode(images)
        latent = posterior.sample()

    return latent


@torch.no_grad()
def decode_latents(model, latents, device, model_type="ldm"):
    """Decode latents with VAE model.

    Args:
        model: VAE model (either diffusers or LDM-style)
        latents: Latent tensor (unscaled)
        device: Device to run on
        model_type: Either "sdvae" for diffusers or "ldm"/"eqvae" for LDM-style

    Returns:
        Reconstructed images tensor in [-1, 1] range
    """
    latents = latents.to(device)

    if model_type == "sdvae":
        # Diffusers VAE expects scaled latents for decoding
        scaling_factor = getattr(model.config, 'scaling_factor', 0.18215)
        latents_scaled = latents / scaling_factor
        recon = model.decode(latents_scaled).sample
    else:
        if hasattr(model, 'ema_scope') and hasattr(model, 'model_ema') and model.model_ema is not None:
            with model.ema_scope():
                recon = model.decode(latents)
        else:
            recon = model.decode(latents)

    return recon


def extract_samples(model, dataset, num_samples, device, model_type="ldm", seed=42):
    """Extract random samples from dataset with their latents and reconstructions."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    images = []
    for idx in indices:
        sample = dataset[int(idx)]
        if isinstance(sample, dict):
            img = sample.get('image', sample.get('images'))
        else:
            img = sample[0]
        images.append(img)
    
    images = torch.stack(images).to(device)
    
    # Encode
    latents = encode_images(model, images, device, model_type)
    
    # Decode
    recons = decode_latents(model, latents, device, model_type)
    
    return {
        'images': images.cpu(),
        'latents': latents.cpu(),
        'reconstructions': recons.cpu(),
    }


def visualize_reconstructions(data, save_path, n_samples=10):
    """Create reconstruction grid: original vs reconstructed."""
    images = data['images']
    recons = data['reconstructions']
    
    n_available = min(n_samples, images.shape[0])
    
    fig, axes = plt.subplots(2, n_available, figsize=(n_available * 2.5, 5))
    
    if n_available == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_available):
        # Original
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        
        # Reconstruction
        rec = denormalize(recons[i]).permute(1, 2, 0).numpy()
        rec = np.clip(rec, 0, 1)
        axes[1, i].imshow(rec)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Recon', fontsize=12)
    
    # Compute MSE for title
    mse = F.mse_loss(recons[:n_available], images[:n_available]).item()
    
    plt.suptitle(f'Reconstructions (MSE: {mse:.4f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def visualize_latent_channels(data, save_path, n_samples=5):
    """Visualize individual latent channels for random samples."""
    images = data['images']
    latents = data['latents'].numpy()
    
    n_channels = latents.shape[1]
    n_available = min(n_samples, images.shape[0])
    
    # Create figure: (1 + n_channels) rows, n_samples columns
    n_rows = 1 + n_channels
    
    fig, axes = plt.subplots(n_rows, n_available, figsize=(n_available * 2.5, n_rows * 2))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_available == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_available):
        # Row 0: Input image
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Sample {i}', fontsize=10)
        axes[0, i].axis('off')
        
        # Rows 1 to n_channels: Individual latent channels
        for c in range(n_channels):
            lat_channel = latents[i, c]
            vmin, vmax = np.percentile(lat_channel, [2, 98])
            lat_norm = np.clip((lat_channel - vmin) / (vmax - vmin + 1e-8), 0, 1)
            axes[c + 1, i].imshow(lat_norm, cmap='viridis')
            axes[c + 1, i].axis('off')
    
    # Add y-labels
    axes[0, 0].set_ylabel('Input', fontsize=10)
    for c in range(n_channels):
        axes[c + 1, 0].set_ylabel(f'Ch {c}', fontsize=10)
    
    for row in range(n_rows):
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].yaxis.label.set_visible(True)
    
    plt.suptitle(f'Latent Channels (C={n_channels})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def visualize_latent_pca(data, save_path, n_samples=5):
    """Visualize PCA of latents for random samples."""
    images = data['images']
    latents = data['latents'].numpy()
    
    n_channels = latents.shape[1]
    n_available = min(n_samples, images.shape[0])
    _, _, h, w = latents.shape
    
    # Create figure: 3 rows (input, raw latent RGB, PCA), n_samples columns
    n_rows = 3
    
    fig, axes = plt.subplots(n_rows, n_available, figsize=(n_available * 2.5, n_rows * 2.5))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_available == 1:
        axes = axes.reshape(-1, 1)
    
    explained_variances = []
    
    for i in range(n_available):
        # Row 0: Input image
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Sample {i}', fontsize=10)
        axes[0, i].axis('off')
        
        # Row 1: Raw latent (first 3 channels as RGB)
        lat_raw = latents[i][:min(3, n_channels)]
        if lat_raw.shape[0] < 3:
            # Pad with zeros if less than 3 channels
            lat_raw = np.concatenate([lat_raw, np.zeros((3 - lat_raw.shape[0], h, w))], axis=0)
        lat_raw = np.transpose(lat_raw, (1, 2, 0))
        lat_raw = (lat_raw - lat_raw.min()) / (lat_raw.max() - lat_raw.min() + 1e-8)
        axes[1, i].imshow(lat_raw)
        axes[1, i].axis('off')
        
        # Row 2: PCA latent
        lat_single = latents[i]
        lat_flat = lat_single.transpose(1, 2, 0).reshape(-1, n_channels)
        
        pca = PCA(n_components=min(3, n_channels))
        lat_pca_flat = pca.fit_transform(lat_flat)
        explained_variances.append(pca.explained_variance_ratio_)
        
        # Pad to 3 components if needed
        if lat_pca_flat.shape[1] < 3:
            lat_pca_flat = np.concatenate([
                lat_pca_flat, 
                np.zeros((lat_pca_flat.shape[0], 3 - lat_pca_flat.shape[1]))
            ], axis=1)
        
        lat_pca = lat_pca_flat.reshape(h, w, 3)
        lat_pca_norm = np.zeros_like(lat_pca)
        for c in range(3):
            channel = lat_pca[..., c]
            vmin, vmax = np.percentile(channel, [2, 98])
            lat_pca_norm[..., c] = np.clip((channel - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        axes[2, i].imshow(lat_pca_norm)
        axes[2, i].axis('off')
    
    # Add y-labels
    axes[0, 0].set_ylabel('Input', fontsize=10)
    axes[1, 0].set_ylabel('Latent', fontsize=10)
    axes[2, 0].set_ylabel('PCA', fontsize=10)
    
    for row in range(n_rows):
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].yaxis.label.set_visible(True)
    
    # Compute average explained variance
    if explained_variances:
        avg_var = np.mean(explained_variances, axis=0)
        n_pcs = len(avg_var)
        var_parts = [f"PC{i+1}={avg_var[i]:.1%}" for i in range(min(3, n_pcs))]
        var_text = f"Avg PCA: {', '.join(var_parts)}"
    else:
        var_text = ""
    
    plt.suptitle(f'Latent PCA\n{var_text}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")


def compute_latent_stats(latents, name: str):
    """Compute statistics for latent representations."""
    return {
        "name": name,
        "shape": list(latents.shape),
        "mean": latents.mean().item(),
        "std": latents.std().item(),
        "min": latents.min().item(),
        "max": latents.max().item(),
        "channel_means": latents.mean(dim=(0, 2, 3)).tolist(),
        "channel_stds": latents.std(dim=(0, 2, 3)).tolist(),
    }


def print_stats(stats: dict):
    """Pretty print latent statistics."""
    print(f"\n  {stats['name']}:")
    print(f"    Shape: {stats['shape']}")
    print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    print(f"    Channel means: {[f'{x:.3f}' for x in stats['channel_means']]}")
    print(f"    Channel stds: {[f'{x:.3f}' for x in stats['channel_stds']]}")


def save_stats_to_file(stats_list, save_path):
    """Save latent statistics to a text file."""
    with open(save_path, 'w') as f:
        for stats in stats_list:
            f.write(f"{stats['name']}:\n")
            f.write(f"  Shape: {stats['shape']}\n")
            f.write(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}\n")
            f.write(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}\n")
            f.write(f"  Channel means: {[f'{x:.3f}' for x in stats['channel_means']]}\n")
            f.write(f"  Channel stds: {[f'{x:.3f}' for x in stats['channel_stds']]}\n")
            f.write("\n")
    print(f"  Saved {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare VAE latent representations across multiple datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file (JSON or YAML)")
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "ldm", "eqvae", "diffusers"],
                        help="Model type")
    parser.add_argument("--output_name", type=str, required=True,
                        help="Subfolder name under eval_outputs/")

    # Dataset paths (with defaults)
    parser.add_argument("--co3d_dir", type=str, default="/data/lab_moezkan/co3d_full",
                        help="CO3D dataset root directory")
    parser.add_argument("--co3d_bb_file", type=str, default="/data/lab_moezkan/co3d_bboxes/toybus_test.jgz",
                        help="CO3D bounding box file")
    parser.add_argument("--imagenet_dir", type=str, default="/data/lab_moezkan/imagenet-256",
                        help="ImageNet dataset root directory")
    parser.add_argument("--omniobject_dir", type=str, default="/data/lab_moezkan/omni_obj/blender_renders_24_views",
                        help="OmniObject3D dataset root directory")

    # Dataset selection
    parser.add_argument("--skip_co3d", action="store_true",
                        help="Skip CO3D dataset")
    parser.add_argument("--skip_imagenet", action="store_true",
                        help="Skip ImageNet dataset")
    parser.add_argument("--skip_omniobject", action="store_true",
                        help="Skip OmniObject3D dataset")

    # Comparison with SD-VAE
    parser.add_argument("--compare_sdvae", action="store_true",
                        help="Also compare with SD-VAE 2.1")

    # Visualization options
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples for reconstruction grid")
    parser.add_argument("--num_latent_samples", type=int, default=5,
                        help="Number of samples for latent visualization")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Input image size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Optional flags
    parser.add_argument("--skip_pca", action="store_true",
                        help="Skip PCA visualization")
    parser.add_argument("--skip_latents", action="store_true",
                        help="Skip latent channel visualization")
    parser.add_argument("--no_crop", action="store_true",
                        help="Disable cropping images based on bounding boxes")

    return parser.parse_args()


def process_dataset(model, dataset, dataset_name, output_dir, args, device, model_type):
    """Process a single dataset and generate visualizations."""
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} ({len(dataset)} samples)")
    print(f"{'='*60}")
    
    # Extract samples
    print(f"  Extracting {args.num_samples} random samples...")
    data = extract_samples(
        model, dataset, args.num_samples, device, 
        model_type=model_type, seed=args.seed
    )
    
    # Compute and save stats
    stats = compute_latent_stats(data['latents'], dataset_name)
    print_stats(stats)
    save_stats_to_file([stats], dataset_dir / "latent_stats.txt")
    
    # Visualize reconstructions
    print("  Generating reconstruction visualization...")
    visualize_reconstructions(
        data, 
        dataset_dir / "reconstructions.png",
        n_samples=args.num_samples
    )
    
    # Visualize latent channels
    if not args.skip_latents:
        print("  Generating latent channel visualization...")
        visualize_latent_channels(
            data,
            dataset_dir / "latent_channels.png",
            n_samples=args.num_latent_samples
        )
    
    # Visualize PCA
    if not args.skip_pca:
        print("  Generating PCA visualization...")
        visualize_latent_pca(
            data,
            dataset_dir / "latent_pca.png",
            n_samples=args.num_latent_samples
        )
    
    return data, stats


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path("eval_outputs") / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, model_type = load_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        model_type=args.model_type,
    )
    model = model.to(device)
    model.eval()

    # Load SD-VAE if requested
    sdvae = None
    if args.compare_sdvae:
        print("\nLoading SD-VAE for comparison...")
        sdvae = load_sd_vae(device)

    # Define datasets to process
    datasets_config = []
    
    if not args.skip_co3d:
        datasets_config.append({
            "name": "co3d",
            "type": "co3d",
            "data_dir": args.co3d_dir,
            "bb_file": args.co3d_bb_file,
            "crop_images": not args.no_crop,
        })
    
    if not args.skip_imagenet:
        datasets_config.append({
            "name": "imagenet",
            "type": "imagenet",
            "data_dir": args.imagenet_dir,
        })
    
    if not args.skip_omniobject:
        datasets_config.append({
            "name": "omniobject",
            "type": "omniobject",
            "data_dir": args.omniobject_dir,
        })

    # Process each dataset
    all_stats = []
    processed_datasets = []
    
    for ds_config in datasets_config:
        dataset_name = ds_config["name"]
        dataset_type = ds_config["type"]
        
        print(f"\n{'='*60}")
        print(f"Loading {dataset_name} dataset...")
        print(f"{'='*60}")
        
        # Build kwargs for load_dataset
        load_kwargs = {
            "dataset_type": dataset_type,
            "data_dir": ds_config["data_dir"],
            "image_size": args.image_size,
        }
        if "bb_file" in ds_config:
            load_kwargs["bb_file"] = ds_config["bb_file"]
        if "crop_images" in ds_config:
            load_kwargs["crop_images"] = ds_config["crop_images"]
        
        try:
            dataset = load_dataset(**load_kwargs)
            print(f"  Dataset size: {len(dataset)}")
        except Exception as e:
            print(f"  Warning: Failed to load {dataset_name} dataset: {e}")
            print(f"  Skipping {dataset_name}...")
            continue
        
        processed_datasets.append(dataset_name)
        
        # Process with main model
        data, stats = process_dataset(
            model, dataset, dataset_name,
            output_dir, args, device, model_type
        )
        all_stats.append(stats)
        
        # Process with SD-VAE if requested
        if sdvae is not None:
            sdvae_name = f"{dataset_name}_sdvae"
            sdvae_dir = output_dir / sdvae_name
            sdvae_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"Processing SD-VAE on {dataset_name}")
            print(f"{'='*60}")
            
            # Re-extract with SD-VAE
            sdvae_data = extract_samples(
                sdvae, dataset, args.num_samples, device,
                model_type="sdvae", seed=args.seed
            )
            
            sdvae_stats = compute_latent_stats(sdvae_data['latents'], sdvae_name)
            print_stats(sdvae_stats)
            save_stats_to_file([sdvae_stats], sdvae_dir / "latent_stats.txt")
            all_stats.append(sdvae_stats)
            
            visualize_reconstructions(
                sdvae_data,
                sdvae_dir / "reconstructions.png",
                n_samples=args.num_samples
            )
            
            if not args.skip_latents:
                visualize_latent_channels(
                    sdvae_data,
                    sdvae_dir / "latent_channels.png",
                    n_samples=args.num_latent_samples
                )
            
            if not args.skip_pca:
                visualize_latent_pca(
                    sdvae_data,
                    sdvae_dir / "latent_pca.png",
                    n_samples=args.num_latent_samples
                )

    # Save combined stats
    if all_stats:
        save_stats_to_file(all_stats, output_dir / "all_latent_stats.txt")

    # Summary
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Results saved to: {output_dir}/")
    
    for ds_name in processed_datasets:
        print(f"  - {ds_name}/")
        print("      - reconstructions.png")
        if not args.skip_latents:
            print("      - latent_channels.png")
        if not args.skip_pca:
            print("      - latent_pca.png")
        print("      - latent_stats.txt")
        if sdvae is not None:
            print(f"  - {ds_name}_sdvae/")
            print("      - (same files)")
    
    print("  - all_latent_stats.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
