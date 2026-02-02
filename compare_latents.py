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

import argparse
from pathlib import Path

import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from src.analysis import (
    load_model,
    load_sd_vae,
    encode_images,
    decode_latents,
    compute_latent_stats,
    visualize_reconstructions,
    visualize_latent_channels,
    visualize_latent_pca,
)
from src.analysis.latent_metrics import print_latent_stats, save_stats_to_file

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


class ImageNetDataset(torch.utils.data.Dataset):
    """Simple ImageNet dataset for loading random images."""

    def __init__(self, root_dir, image_size=256, max_images=10000):
        self.root = Path(root_dir)
        self.image_size = image_size

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

    latents = encode_images(model, images, device, model_type)
    recons = decode_latents(model, latents, device, model_type)

    return {
        'images': images.cpu(),
        'latents': latents.cpu(),
        'reconstructions': recons.cpu(),
    }


def process_dataset(model, dataset, dataset_name, output_dir, args, device, model_type):
    """Process a single dataset and generate visualizations."""
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} ({len(dataset)} samples)")
    print(f"{'='*60}")

    print(f"  Extracting {args.num_samples} random samples...")
    data = extract_samples(
        model, dataset, args.num_samples, device,
        model_type=model_type, seed=args.seed
    )

    stats = compute_latent_stats(data['latents'], dataset_name)
    print_latent_stats(stats)
    save_stats_to_file([stats], dataset_dir / "latent_stats.txt")

    print("  Generating reconstruction visualization...")
    visualize_reconstructions(
        data,
        dataset_dir / "reconstructions.png",
        n_samples=args.num_samples
    )

    if not args.skip_latents:
        print("  Generating latent channel visualization...")
        visualize_latent_channels(
            data,
            dataset_dir / "latent_channels.png",
            n_samples=args.num_latent_samples
        )

    if not args.skip_pca:
        print("  Generating PCA visualization...")
        visualize_latent_pca(
            data,
            dataset_dir / "latent_pca.png",
            n_samples=args.num_latent_samples
        )

    return data, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare VAE latent representations across multiple datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file (JSON or YAML)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "ldm", "eqvae", "diffusers"],
        help="Model type",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Subfolder name under eval_outputs/",
    )

    parser.add_argument(
        "--co3d_dir",
        type=str,
        default="/data/lab_moezkan/co3d_full",
        help="CO3D dataset root directory",
    )
    parser.add_argument(
        "--co3d_bb_file",
        type=str,
        default="/data/lab_moezkan/co3d_bboxes/toybus_test.jgz",
        help="CO3D bounding box file",
    )
    parser.add_argument(
        "--imagenet_dir",
        type=str,
        default="/data/lab_moezkan/imagenet-256",
        help="ImageNet dataset root directory",
    )
    parser.add_argument(
        "--omniobject_dir",
        type=str,
        default="/data/lab_moezkan/omni_obj/blender_renders_24_views",
        help="OmniObject3D dataset root directory",
    )

    parser.add_argument(
        "--skip_co3d", action="store_true", help="Skip CO3D dataset"
    )
    parser.add_argument(
        "--skip_imagenet", action="store_true", help="Skip ImageNet dataset"
    )
    parser.add_argument(
        "--skip_omniobject", action="store_true", help="Skip OmniObject3D dataset"
    )

    parser.add_argument(
        "--compare_sdvae", action="store_true", help="Also compare with SD-VAE 2.1"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples for reconstruction grid",
    )
    parser.add_argument(
        "--num_latent_samples",
        type=int,
        default=5,
        help="Number of samples for latent visualization",
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=256, 
        help="Input image size"
    )
    parser.add_argument(
        "--seed", type=int, default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip_pca", 
        action="store_true", 
        help="Skip PCA visualization"
    )
    parser.add_argument(
        "--skip_latents", 
        action="store_true", 
        help="Skip latent channel visualization"
    )
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Disable cropping images based on bounding boxes",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path("eval_outputs") / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\nLoading model from: {args.checkpoint}")
    model, model_type = load_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        model_type=args.model_type,
    )
    model = model.to(device)
    model.eval()

    sdvae = None
    if args.compare_sdvae:
        print("\nLoading SD-VAE for comparison...")
        sdvae = load_sd_vae(device)

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

    all_stats = []
    processed_datasets = []

    for ds_config in datasets_config:
        dataset_name = ds_config["name"]
        dataset_type = ds_config["type"]

        print(f"\n{'='*60}")
        print(f"Loading {dataset_name} dataset...")
        print(f"{'='*60}")

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

        data, stats = process_dataset(
            model, dataset, dataset_name,
            output_dir, args, device, model_type
        )
        all_stats.append(stats)

        if sdvae is not None:
            sdvae_name = f"{dataset_name}_sdvae"
            sdvae_dir = output_dir / sdvae_name
            sdvae_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Processing SD-VAE on {dataset_name}")
            print(f"{'='*60}")

            sdvae_data = extract_samples(
                sdvae, dataset, args.num_samples, device,
                model_type="sdvae", seed=args.seed
            )

            sdvae_stats = compute_latent_stats(sdvae_data['latents'], sdvae_name)
            print_latent_stats(sdvae_stats)
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

    if all_stats:
        save_stats_to_file(all_stats, output_dir / "all_latent_stats.txt")

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
