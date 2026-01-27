#!/usr/bin/env python
"""
Test script for warp-based VAE training integration.

Tests:
1. WarpCO3DDataset - loading paired images and computing warps
2. WarpConsistencyLoss - loss computation
3. WarpVAETrainer - single training step

Usage:
    python test_warp_training.py
"""

import os
import sys

# Set GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Disable torch.compile for older GPUs
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

print("=" * 60)
print("Warp VAE Training Integration Test")
print("=" * 60)


def test_warp_dataset():
    """Test WarpCO3DDataset loading and warp computation."""
    print("\n[1] Testing WarpCO3DDataset...")

    from src.data.warp_dataset import WarpCO3DDataset

    dataset = WarpCO3DDataset(
        root_dir="/data/lab_moezkan/co3d_full",
        bb_file="/data/lab_moezkan/co3d_bboxes/toybus_test.jgz",
        image_size=128,
        romav2_setting="turbo",
        pair_sampling="sequential",
        warp_resolution=128,
    )

    print(f"  Dataset size: {len(dataset)}")

    # Get a sample
    print("  Loading sample (this will compute RoMaV2 warp on first call)...")
    sample = dataset[0]

    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Target image shape: {sample['image_target'].shape}")
    print(f"  Warp A->B shape: {sample['warp_ab'].shape}")
    print(f"  Confidence A->B shape: {sample['confidence_ab'].shape}")
    print(f"  Warp B->A shape: {sample['warp_ba'].shape}")

    # Check values
    conf_ab = sample['confidence_ab']
    print(f"  Confidence range: [{conf_ab.min():.3f}, {conf_ab.max():.3f}]")
    print(f"  Mean confidence: {conf_ab.mean():.3f}")

    print("  [PASS] WarpCO3DDataset works!")
    return sample


def test_warp_consistency_loss(sample):
    """Test WarpConsistencyLoss computation."""
    print("\n[2] Testing WarpConsistencyLoss...")

    from src.losses.warp_consistency import WarpConsistencyLoss

    loss_fn = WarpConsistencyLoss(
        loss_type="l1",
        bidirectional=True,
        confidence_weighted=True,
        confidence_threshold=0.1,
    )

    # Create fake latent codes (would come from VAE encoder)
    B, C, H, W = 1, 4, 16, 16  # Typical latent dimensions
    latent_a = torch.randn(B, C, H, W, requires_grad=True)
    latent_b = torch.randn(B, C, H, W, requires_grad=True)

    # Prepare warp tensors
    warp_ab = sample['warp_ab'].unsqueeze(0)  # Add batch dim
    warp_ba = sample['warp_ba'].unsqueeze(0)
    conf_ab = sample['confidence_ab'].unsqueeze(0)
    conf_ba = sample['confidence_ba'].unsqueeze(0)

    # Compute loss
    result = loss_fn(latent_a, latent_b, warp_ab, warp_ba, conf_ab, conf_ba)

    print(f"  Loss keys: {list(result.keys())}")
    print(f"  Total loss: {result['loss'].item():.4f}")
    print(f"  Loss A->B: {result['loss_ab'].item():.4f}")
    print(f"  Loss B->A: {result['loss_ba'].item():.4f}")

    # Test backward
    result['loss'].backward()
    print("  Backward pass successful!")
    print(f"  Latent A grad exists: {latent_a.grad is not None}")
    print(f"  Latent B grad exists: {latent_b.grad is not None}")

    print("  [PASS] WarpConsistencyLoss works!")
    return loss_fn


def test_warp_vae_trainer():
    """Test WarpVAETrainer with a synthetic batch."""
    print("\n[3] Testing WarpVAETrainer setup...")

    from src.trainer.vae_trainers import WarpVAETrainer

    # Model config matching the small config
    model_config = {
        "target": "ldm.models.autoencoder.AutoencoderKL",
        "params": {
            "embed_dim": 4,
            "ddconfig": {
                "double_z": True,
                "z_channels": 4,
                "resolution": 128,
                "in_channels": 3,
                "out_ch": 3,
                "ch": 64,
                "ch_mult": [1, 2, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
            },
            "lossconfig": {
                "target": "ldm.modules.losses.LPIPSWithDiscriminator",
                "params": {
                    "disc_start": 50001,
                    "kl_weight": 0.000001,
                    "disc_weight": 0.5,
                    "perceptual_weight": 1.0,
                    "disc_in_channels": 3,
                    "disc_num_layers": 2,
                    "use_actnorm": False,
                },
            },
        },
    }

    trainer = WarpVAETrainer(
        model_config=model_config,
        learning_rate=4.5e-6,
        warp_consistency_weight=0.5,
        consistency_loss_type="l1",
        bidirectional=True,
        warmup_steps=0,
    )

    print(f"  Trainer created successfully!")
    print(f"  Model type: {type(trainer.model).__name__}")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = trainer.to(device)
    print(f"  Using device: {device}")

    return trainer


def test_forward_pass(trainer, sample):
    """Test a forward pass through the model with warp loss."""
    print("\n[4] Testing forward pass with warp loss...")

    device = next(trainer.parameters()).device

    # Create batch from sample
    batch = {
        "image": sample["image"].unsqueeze(0).to(device),
        "image_target": sample["image_target"].unsqueeze(0).to(device),
        "warp_ab": sample["warp_ab"].unsqueeze(0).to(device),
        "warp_ba": sample["warp_ba"].unsqueeze(0).to(device),
        "confidence_ab": sample["confidence_ab"].unsqueeze(0).to(device),
        "confidence_ba": sample["confidence_ba"].unsqueeze(0).to(device),
    }

    # Forward pass through VAE
    with torch.no_grad():
        inputs = batch["image"]
        recon, posterior = trainer.model(inputs, sample_posterior=True)
        latent_a = posterior.sample()

        # Get target latent
        posterior_b = trainer.model.encode(batch["image_target"])
        latent_b = posterior_b.sample()

    print(f"  Input shape: {inputs.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Latent A shape: {latent_a.shape}")
    print(f"  Latent B shape: {latent_b.shape}")

    # Compute warp consistency loss
    warp_loss, log_dict = trainer._compute_warp_losses(
        batch, latent_a, latent_b, recon, split="test"
    )

    print(f"  Warp loss: {warp_loss.item():.4f}")
    print(f"  Log dict keys: {list(log_dict.keys())}")

    print("  [PASS] Forward pass works!")


def test_visualization(sample):
    """Test warp visualization."""
    print("\n[5] Testing warp visualization...")

    import matplotlib.pyplot as plt

    # Denormalize images
    img_a = (sample["image"].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
    img_b = (sample["image_target"].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

    # Warp image A to B
    img_a_tensor = sample["image"].unsqueeze(0)
    warp_ab = sample["warp_ab"].unsqueeze(0)

    warped_a = F.grid_sample(
        img_a_tensor,
        warp_ab,
        mode="bilinear",
        padding_mode="border",
        align_corners=False
    )
    warped_a = (warped_a.squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)

    # Confidence map
    conf = sample["confidence_ab"].numpy()

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(img_a)
    axes[0, 0].set_title("Image A (Source)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_b)
    axes[0, 1].set_title("Image B (Target)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(warped_a)
    axes[1, 0].set_title("Image A Warped to B")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(conf, cmap="hot")
    axes[1, 1].set_title("Confidence Map")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("warp_training_test.png", dpi=150)
    print("  Saved visualization to warp_training_test.png")

    print("  [PASS] Visualization works!")


def main():
    try:
        # Test dataset
        sample = test_warp_dataset()

        # Test loss function
        test_warp_consistency_loss(sample)

        # Test trainer
        trainer = test_warp_vae_trainer()

        # Test forward pass
        test_forward_pass(trainer, sample)

        # Test visualization
        test_visualization(sample)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now train with warp consistency using:")
        print("  python train.py --config-name=warp_vae_co3d_small")
        print("\nOr the full resolution version:")
        print("  python train.py --config-name=warp_vae_co3d")

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
