"""
Compare latent representations between EQ-VAE and SD-VAE 2.1 on CO3D and ImageNet.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL as DiffusersVAE
from omegaconf import OmegaConf

from ldm.models.autoencoder import EQVAEAutoencoder


def load_eqvae(checkpoint_path: str, config_path: str, device: str = "cuda", use_ema: bool = True):
    """Load EQ-VAE model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config YAML
        device: Device to load model on
        use_ema: Whether to use EMA weights if available

    Returns:
        Tuple of (model, ema_mode) where ema_mode indicates if EMA weights are being used
    """
    config = OmegaConf.load(config_path)
    model_params = config.model.params

    # Create model without checkpoint first
    model = EQVAEAutoencoder(
        ddconfig=OmegaConf.to_container(model_params.ddconfig),
        lossconfig=OmegaConf.to_container(model_params.lossconfig),
        embed_dim=model_params.embed_dim,
        p_prior=model_params.get("p_prior", 0.9),
        scale_range=model_params.get("scale_range", [0.25, 1.0]),
        use_rotation=model_params.get("use_rotation", True),
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Remove 'model.' prefix if present (PyTorch Lightning checkpoint format)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned_state_dict[k[6:]] = v
        else:
            cleaned_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing_keys:
        print(f"  Warning: Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys: {len(unexpected_keys)}")

    model = model.to(device)
    model.eval()

    # Check for EMA weights
    ema_mode = False
    if use_ema and hasattr(model, 'model_ema') and model.model_ema is not None:
        print(f"Loaded EQ-VAE from {checkpoint_path} (using EMA weights)")
        ema_mode = True
    else:
        print(f"Loaded EQ-VAE from {checkpoint_path} (no EMA weights)")

    return model, ema_mode


def load_sd_vae(device: str = "cuda"):
    """Load SD-VAE 2.1 from HuggingFace."""
    model = DiffusersVAE.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    )
    model = model.to(device)
    model.eval()
    print("Loaded SD-VAE 2.1 (ft-mse) from HuggingFace")
    return model


def get_co3d_images(n_images: int = 10):
    """Load CO3D images using config."""
    from src.data.co3d_dataset import CO3DDataset

    config = OmegaConf.load("config/plucker_vae_co3d.yaml")

    dataset = CO3DDataset(
        root_dir=config.co3d_dir,
        bb_file=config.bb_file,
        image_size=256,
        include_plucker=False,
        crop_images=True,
        apply_augmentation=False,
    )

    indices = np.random.choice(len(dataset), min(n_images, len(dataset)), replace=False)
    images = []
    for idx in indices:
        sample = dataset[int(idx)]
        images.append(sample["image"])

    return torch.stack(images), "CO3D"


def get_imagenet_images(root_dir: str, n_images: int = 10, image_size: int = 256):
    """Load random ImageNet images."""
    root = Path(root_dir)

    # Get all image files from subdirectories
    all_images = []
    for subdir in root.iterdir():
        if subdir.is_dir():
            for img_file in subdir.glob("*.jpg"):
                all_images.append(img_file)
            for img_file in subdir.glob("*.JPEG"):
                all_images.append(img_file)
            for img_file in subdir.glob("*.png"):
                all_images.append(img_file)
            if len(all_images) > 1000:  # Limit search for efficiency
                break

    # Sample random images
    selected = np.random.choice(len(all_images), min(n_images, len(all_images)), replace=False)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    images = []
    for idx in selected:
        img = Image.open(all_images[idx]).convert("RGB")
        images.append(transform(img))

    return torch.stack(images), "ImageNet"


def get_omniobject_images(root_dir: str, n_images: int = 10, image_size: int = 256):
    """Load random OmniObject3D images."""
    from src.data.omniobject3d_dataset import OmniObject3DDataset

    dataset = OmniObject3DDataset(
        root_dir=root_dir,
        image_size=image_size,
        include_plucker=False,
        sample_mode="single",
    )

    indices = np.random.choice(len(dataset), min(n_images, len(dataset)), replace=False)
    images = []
    for idx in indices:
        sample = dataset[int(idx)]
        images.append(sample["image"])

    return torch.stack(images), "OmniObject3D"


@torch.no_grad()
def encode_with_eqvae(model, images, device="cuda", ema_mode=False):
    """Encode images with EQ-VAE model."""
    images = images.to(device)
    if ema_mode and hasattr(model, 'ema_scope'):
        with model.ema_scope():
            posterior = model.encode(images)
    else:
        posterior = model.encode(images)
    latent = posterior.sample()
    return latent


@torch.no_grad()
def decode_with_eqvae(model, latents, device="cuda", ema_mode=False):
    """Decode latents with EQ-VAE model."""
    latents = latents.to(device)
    if ema_mode and hasattr(model, 'ema_scope'):
        with model.ema_scope():
            recon = model.decode(latents)
    else:
        recon = model.decode(latents)
    return recon


@torch.no_grad()
def encode_with_sdvae(model, images, device="cuda"):
    """Encode images with SD-VAE model."""
    images = images.to(device)
    latent = model.encode(images).latent_dist.sample()
    return latent


@torch.no_grad()
def decode_with_sdvae(model, latents, device="cuda"):
    """Decode latents with SD-VAE model."""
    latents = latents.to(device)
    recon = model.decode(latents).sample
    return recon


def compute_latent_stats(latents, name: str):
    """Compute statistics for latent representations."""
    stats = {
        "name": name,
        "shape": list(latents.shape),
        "mean": latents.mean().item(),
        "std": latents.std().item(),
        "min": latents.min().item(),
        "max": latents.max().item(),
        "channel_means": latents.mean(dim=(0, 2, 3)).tolist(),
        "channel_stds": latents.std(dim=(0, 2, 3)).tolist(),
    }
    return stats


def compare_latents(latent1, latent2, name1: str, name2: str):
    """Compare two latent representations."""
    # Resize if shapes differ
    if latent1.shape != latent2.shape:
        # Resize latent1 to match latent2 spatial dimensions
        latent1_resized = torch.nn.functional.interpolate(
            latent1, size=latent2.shape[-2:], mode="bilinear", align_corners=False
        )
        print(f"  Resized {name1} from {list(latent1.shape)} to {list(latent1_resized.shape)}")
        latent1 = latent1_resized

    # MSE between latents
    mse = torch.nn.functional.mse_loss(latent1, latent2).item()

    # Cosine similarity (flatten spatial dims)
    flat1 = latent1.flatten(start_dim=1)
    flat2 = latent2.flatten(start_dim=1)
    cos_sim = torch.nn.functional.cosine_similarity(flat1, flat2, dim=1).mean().item()

    # Correlation
    flat1_centered = flat1 - flat1.mean(dim=1, keepdim=True)
    flat2_centered = flat2 - flat2.mean(dim=1, keepdim=True)
    corr = (flat1_centered * flat2_centered).sum(dim=1) / (
        flat1_centered.norm(dim=1) * flat2_centered.norm(dim=1) + 1e-8
    )
    corr = corr.mean().item()

    return {
        "mse": mse,
        "cosine_similarity": cos_sim,
        "correlation": corr,
    }


def print_stats(stats: dict):
    """Pretty print latent statistics."""
    print(f"\n  {stats['name']}:")
    print(f"    Shape: {stats['shape']}")
    print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    print(f"    Channel means: {[f'{x:.3f}' for x in stats['channel_means']]}")
    print(f"    Channel stds: {[f'{x:.3f}' for x in stats['channel_stds']]}")


def save_images(images: torch.Tensor, output_dir: Path, prefix: str):
    """Save tensor images to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = img.clamp(0, 1)

        # Convert to PIL
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Save
        save_path = output_dir / f"{prefix}_{i:02d}.png"
        pil_img.save(save_path)

    print(f"Saved {len(images)} images to {output_dir}/{prefix}_*.png")


def save_latent_images(latents: torch.Tensor, output_dir: Path, prefix: str):
    """Save latent representations as images (one per channel, normalized)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # latents shape: [B, C, H, W]
    for i, latent in enumerate(latents):
        # Create a grid of channels for each sample
        n_channels = latent.shape[0]

        # Save each channel as a separate grayscale image
        for c in range(n_channels):
            channel = latent[c].cpu().numpy()

            # Normalize to [0, 255]
            channel_min, channel_max = channel.min(), channel.max()
            if channel_max - channel_min > 0:
                channel_norm = (channel - channel_min) / (channel_max - channel_min)
            else:
                channel_norm = np.zeros_like(channel)

            img_np = (channel_norm * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode='L')

            # Upscale for better visibility
            pil_img = pil_img.resize((128, 128), Image.NEAREST)

            save_path = output_dir / f"{prefix}_{i:02d}_ch{c}.png"
            pil_img.save(save_path)

        # Also save all channels as a combined RGB-like image (first 3 channels)
        combined = latent[:3].cpu().numpy()  # Take first 3 channels
        combined_norm = np.zeros((3, latent.shape[1], latent.shape[2]))
        for c in range(3):
            ch = combined[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 0:
                combined_norm[c] = (ch - ch_min) / (ch_max - ch_min)

        combined_rgb = (combined_norm.transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_combined = Image.fromarray(combined_rgb)
        pil_combined = pil_combined.resize((128, 128), Image.NEAREST)

        save_path = output_dir / f"{prefix}_{i:02d}_combined.png"
        pil_combined.save(save_path)

    print(f"Saved {len(latents)} latent images to {output_dir}/{prefix}_*.png")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths
    eqvae_ckpt = "checkpoints/loose-mushroom-of-algebraic-tempering_EQ-VAE small model for GPU memory testing/last.ckpt"
    eqvae_config = "config/eqvae_omniobject_small.yaml"
    imagenet_root = "/data/lab_moezkan/imagenet-256"
    omniobject_root = "/data/lab_moezkan/omni_obj/blender_renders_24_views"

    # Load models
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)
    eqvae, ema_mode = load_eqvae(eqvae_ckpt, eqvae_config, device, use_ema=True)
    sdvae = load_sd_vae(device)

    # Load datasets
    print("\n" + "="*60)
    print("Loading datasets (10 images each)...")
    print("="*60)

    np.random.seed(42)
    co3d_images, co3d_name = get_co3d_images(n_images=10)
    imagenet_images, imagenet_name = get_imagenet_images(imagenet_root, n_images=10)
    omniobj_images, omniobj_name = get_omniobject_images(omniobject_root, n_images=10)

    print(f"CO3D images shape: {co3d_images.shape}")
    print(f"ImageNet images shape: {imagenet_images.shape}")
    print(f"OmniObject3D images shape: {omniobj_images.shape}")

    # Save images
    output_dir = Path("outputs/latent_comparison")
    save_images(co3d_images, output_dir, "co3d")
    save_images(imagenet_images, output_dir, "imagenet")
    save_images(omniobj_images, output_dir, "omniobject")

    # Resize images to match model expectations
    # EQ-VAE expects 128x128, SD-VAE expects 256x256+
    co3d_128 = torch.nn.functional.interpolate(co3d_images, size=(128, 128), mode="bilinear")
    imagenet_128 = torch.nn.functional.interpolate(imagenet_images, size=(128, 128), mode="bilinear")
    omniobj_128 = torch.nn.functional.interpolate(omniobj_images, size=(128, 128), mode="bilinear")

    # Encode with both models
    print("\n" + "="*60)
    print("Encoding CO3D images...")
    print("="*60)

    eqvae_co3d_latent = encode_with_eqvae(eqvae, co3d_128, device, ema_mode)
    sdvae_co3d_latent = encode_with_sdvae(sdvae, co3d_images, device)

    eqvae_co3d_stats = compute_latent_stats(eqvae_co3d_latent, "EQ-VAE on CO3D")
    sdvae_co3d_stats = compute_latent_stats(sdvae_co3d_latent, "SD-VAE on CO3D")

    print_stats(eqvae_co3d_stats)
    print_stats(sdvae_co3d_stats)

    co3d_comparison = compare_latents(
        eqvae_co3d_latent, sdvae_co3d_latent,
        "EQ-VAE", "SD-VAE"
    )
    print("\n  Comparison (EQ-VAE vs SD-VAE on CO3D):")
    print(f"    MSE: {co3d_comparison['mse']:.4f}")
    print(f"    Cosine Similarity: {co3d_comparison['cosine_similarity']:.4f}")
    print(f"    Correlation: {co3d_comparison['correlation']:.4f}")

    print("\n" + "="*60)
    print("Encoding ImageNet images...")
    print("="*60)

    eqvae_imagenet_latent = encode_with_eqvae(eqvae, imagenet_128, device, ema_mode)
    sdvae_imagenet_latent = encode_with_sdvae(sdvae, imagenet_images, device)

    eqvae_imagenet_stats = compute_latent_stats(eqvae_imagenet_latent, "EQ-VAE on ImageNet")
    sdvae_imagenet_stats = compute_latent_stats(sdvae_imagenet_latent, "SD-VAE on ImageNet")

    print_stats(eqvae_imagenet_stats)
    print_stats(sdvae_imagenet_stats)

    imagenet_comparison = compare_latents(
        eqvae_imagenet_latent, sdvae_imagenet_latent,
        "EQ-VAE", "SD-VAE"
    )
    print("\n  Comparison (EQ-VAE vs SD-VAE on ImageNet):")
    print(f"    MSE: {imagenet_comparison['mse']:.4f}")
    print(f"    Cosine Similarity: {imagenet_comparison['cosine_similarity']:.4f}")
    print(f"    Correlation: {imagenet_comparison['correlation']:.4f}")

    print("\n" + "="*60)
    print("Encoding OmniObject3D images...")
    print("="*60)

    eqvae_omniobj_latent = encode_with_eqvae(eqvae, omniobj_128, device, ema_mode)
    sdvae_omniobj_latent = encode_with_sdvae(sdvae, omniobj_images, device)

    eqvae_omniobj_stats = compute_latent_stats(eqvae_omniobj_latent, "EQ-VAE on OmniObject3D")
    sdvae_omniobj_stats = compute_latent_stats(sdvae_omniobj_latent, "SD-VAE on OmniObject3D")

    print_stats(eqvae_omniobj_stats)
    print_stats(sdvae_omniobj_stats)

    omniobj_comparison = compare_latents(
        eqvae_omniobj_latent, sdvae_omniobj_latent,
        "EQ-VAE", "SD-VAE"
    )
    print("\n  Comparison (EQ-VAE vs SD-VAE on OmniObject3D):")
    print(f"    MSE: {omniobj_comparison['mse']:.4f}")
    print(f"    Cosine Similarity: {omniobj_comparison['cosine_similarity']:.4f}")
    print(f"    Correlation: {omniobj_comparison['correlation']:.4f}")

    # Cross-dataset comparison for each model
    print("\n" + "="*60)
    print("Cross-dataset comparison...")
    print("="*60)

    eqvae_cross = compare_latents(
        eqvae_co3d_latent, eqvae_imagenet_latent,
        "EQ-VAE CO3D", "EQ-VAE ImageNet"
    )
    print("\n  EQ-VAE: CO3D vs ImageNet:")
    print(f"    MSE: {eqvae_cross['mse']:.4f}")
    print(f"    Cosine Similarity: {eqvae_cross['cosine_similarity']:.4f}")
    print(f"    Correlation: {eqvae_cross['correlation']:.4f}")

    sdvae_cross = compare_latents(
        sdvae_co3d_latent, sdvae_imagenet_latent,
        "SD-VAE CO3D", "SD-VAE ImageNet"
    )
    print("\n  SD-VAE: CO3D vs ImageNet:")
    print(f"    MSE: {sdvae_cross['mse']:.4f}")
    print(f"    Cosine Similarity: {sdvae_cross['cosine_similarity']:.4f}")
    print(f"    Correlation: {sdvae_cross['correlation']:.4f}")

    eqvae_cross_omni = compare_latents(
        eqvae_co3d_latent, eqvae_omniobj_latent,
        "EQ-VAE CO3D", "EQ-VAE OmniObject3D"
    )
    print("\n  EQ-VAE: CO3D vs OmniObject3D:")
    print(f"    MSE: {eqvae_cross_omni['mse']:.4f}")
    print(f"    Cosine Similarity: {eqvae_cross_omni['cosine_similarity']:.4f}")
    print(f"    Correlation: {eqvae_cross_omni['correlation']:.4f}")

    sdvae_cross_omni = compare_latents(
        sdvae_co3d_latent, sdvae_omniobj_latent,
        "SD-VAE CO3D", "SD-VAE OmniObject3D"
    )
    print("\n  SD-VAE: CO3D vs OmniObject3D:")
    print(f"    MSE: {sdvae_cross_omni['mse']:.4f}")
    print(f"    Cosine Similarity: {sdvae_cross_omni['cosine_similarity']:.4f}")
    print(f"    Correlation: {sdvae_cross_omni['correlation']:.4f}")

    # Save latents
    print("\n" + "="*60)
    print("Saving latents...")
    print("="*60)

    # Save latent images
    print("\n" + "="*60)
    print("Saving latent images...")
    print("="*60)

    save_latent_images(eqvae_co3d_latent, output_dir, "latent_eqvae_co3d")
    save_latent_images(eqvae_imagenet_latent, output_dir, "latent_eqvae_imagenet")
    save_latent_images(eqvae_omniobj_latent, output_dir, "latent_eqvae_omniobj")
    save_latent_images(sdvae_co3d_latent, output_dir, "latent_sdvae_co3d")
    save_latent_images(sdvae_imagenet_latent, output_dir, "latent_sdvae_imagenet")
    save_latent_images(sdvae_omniobj_latent, output_dir, "latent_sdvae_omniobj")

    # Save reconstructions
    print("\n" + "="*60)
    print("Decoding and saving reconstructions...")
    print("="*60)

    recon_dir = Path("outputs/reconstructions")

    # EQ-VAE reconstructions (decode at 128x128, then upscale to 256 for comparison)
    eqvae_co3d_recon = decode_with_eqvae(eqvae, eqvae_co3d_latent, device, ema_mode)
    eqvae_imagenet_recon = decode_with_eqvae(eqvae, eqvae_imagenet_latent, device, ema_mode)
    eqvae_omniobj_recon = decode_with_eqvae(eqvae, eqvae_omniobj_latent, device, ema_mode)

    # Upscale EQ-VAE reconstructions to 256x256 for fair comparison
    eqvae_co3d_recon_256 = torch.nn.functional.interpolate(
        eqvae_co3d_recon, size=(256, 256), mode="bilinear", align_corners=False
    )
    eqvae_imagenet_recon_256 = torch.nn.functional.interpolate(
        eqvae_imagenet_recon, size=(256, 256), mode="bilinear", align_corners=False
    )
    eqvae_omniobj_recon_256 = torch.nn.functional.interpolate(
        eqvae_omniobj_recon, size=(256, 256), mode="bilinear", align_corners=False
    )

    # SD-VAE reconstructions
    sdvae_co3d_recon = decode_with_sdvae(sdvae, sdvae_co3d_latent, device)
    sdvae_imagenet_recon = decode_with_sdvae(sdvae, sdvae_imagenet_latent, device)
    sdvae_omniobj_recon = decode_with_sdvae(sdvae, sdvae_omniobj_latent, device)

    # Save all reconstructions
    save_images(eqvae_co3d_recon_256, recon_dir, "recon_eqvae_co3d")
    save_images(eqvae_imagenet_recon_256, recon_dir, "recon_eqvae_imagenet")
    save_images(eqvae_omniobj_recon_256, recon_dir, "recon_eqvae_omniobj")
    save_images(sdvae_co3d_recon, recon_dir, "recon_sdvae_co3d")
    save_images(sdvae_imagenet_recon, recon_dir, "recon_sdvae_imagenet")
    save_images(sdvae_omniobj_recon, recon_dir, "recon_sdvae_omniobj")

    # Also save original images in the same folder for easy comparison
    save_images(co3d_images, recon_dir, "original_co3d")
    save_images(imagenet_images, recon_dir, "original_imagenet")
    save_images(omniobj_images, recon_dir, "original_omniobj")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
