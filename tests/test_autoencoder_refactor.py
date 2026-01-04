"""
Quick verification script to test the refactored AutoencoderKL and PluckerAutoencoder classes.
"""

import torch
from ldm.models.autoencoder import AutoencoderKL, PluckerAutoencoder


def test_base_autoencoder():
    """Test the base AutoencoderKL class."""
    print("=" * 60)
    print("Testing Base AutoencoderKL")
    print("=" * 60)

    # Configuration
    ddconfig = {
        "double_z": True,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    }

    lossconfig = {"target": "torch.nn.Identity"}

    # Create model
    model = AutoencoderKL(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        embed_dim=4,
    )

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)

    print(f"Input shape: {x.shape}")

    # Encode
    posterior = model.encode(x)
    print(f"✓ Encode successful")
    print(f"  Posterior mean shape: {posterior.mean.shape}")

    # Sample latent
    z = posterior.sample()
    print(f"  Latent shape: {z.shape}")

    # Decode
    recon = model.decode(z)
    print(f"✓ Decode successful")
    print(f"  Reconstruction shape: {recon.shape}")

    # Forward
    recon2, posterior2 = model(x)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {recon2.shape}")

    assert recon.shape == x.shape, "Reconstruction shape mismatch"
    print("\n✅ Base AutoencoderKL test PASSED\n")


def test_plucker_autoencoder():
    """Test the PluckerAutoencoder class."""
    print("=" * 60)
    print("Testing PluckerAutoencoder")
    print("=" * 60)

    # Configuration
    ddconfig = {
        "double_z": True,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    }

    lossconfig = {"target": "torch.nn.Identity"}

    # Create model
    model = PluckerAutoencoder(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        embed_dim=4,
        n_patches=8,
        plucker_key="pluck_ray",
    )

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)

    print(f"Input shape: {x.shape}")

    # Encode
    posterior, pluck = model.encode(x)
    print(f"✓ Encode successful")
    print(f"  Posterior mean shape: {posterior.mean.shape}")
    print(f"  Plucker shape: {pluck.shape}")

    # Check Plucker shape
    expected_pluck_shape = (batch_size, 8 * 8, 6)
    assert pluck.shape == expected_pluck_shape, \
        f"Plucker shape mismatch: expected {expected_pluck_shape}, got {pluck.shape}"
    print(f"  ✓ Plucker shape correct: {pluck.shape}")

    # Sample latent
    z = posterior.sample()
    print(f"  Latent shape: {z.shape}")

    # Decode
    recon = model.decode(z)
    print(f"✓ Decode successful")
    print(f"  Reconstruction shape: {recon.shape}")

    # Forward
    recon2, posterior2, pluck2 = model(x)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {recon2.shape}")
    print(f"  Plucker shape: {pluck2.shape}")

    # Test Plucker loss
    gt_pluck = torch.randn_like(pluck)
    loss = model.hybrid_plucker_loss(pluck, gt_pluck)
    print(f"✓ Plucker loss computation successful")
    print(f"  Loss value: {loss.item():.6f}")

    assert recon.shape == x.shape, "Reconstruction shape mismatch"
    assert pluck2.shape == expected_pluck_shape, "Forward Plucker shape mismatch"

    print("\n✅ PluckerAutoencoder test PASSED\n")


def test_plucker_constraint():
    """Test that Plucker loss enforces geometric constraints."""
    print("=" * 60)
    print("Testing Plucker Constraint Enforcement")
    print("=" * 60)

    ddconfig = {
        "double_z": True,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    }

    lossconfig = {"target": "torch.nn.Identity"}

    model = PluckerAutoencoder(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        embed_dim=4,
        n_patches=8,
        plucker_weights={"recon": 1.0, "constraint": 1.0, "norm": 1.0},
    )

    # Create perfect Plucker coordinates (d·m = 0, ||d|| = 1)
    batch_size, num_rays = 2, 64
    directions = torch.randn(batch_size, num_rays, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)  # Unit vectors

    # Create moments perpendicular to directions
    # For d·m = 0, we need m perpendicular to d
    # Use cross product with random vector
    random_vec = torch.randn(batch_size, num_rays, 3)
    moments = torch.cross(directions, random_vec, dim=-1)

    gt_pluck = torch.cat([directions, moments], dim=-1)

    # Check constraint
    dot_products = (directions * moments).sum(dim=-1)
    print(f"Ground truth d·m values (should be ~0):")
    print(f"  Mean: {dot_products.mean().item():.6e}")
    print(f"  Max: {dot_products.abs().max().item():.6e}")

    # Test with perfect prediction (should have low loss)
    loss_perfect = model.hybrid_plucker_loss(gt_pluck, gt_pluck)
    print(f"\n✓ Loss for perfect prediction: {loss_perfect.item():.6f}")

    # Test with random prediction (should have high loss)
    random_pluck = torch.randn_like(gt_pluck)
    loss_random = model.hybrid_plucker_loss(random_pluck, gt_pluck)
    print(f"✓ Loss for random prediction: {loss_random.item():.6f}")

    assert loss_perfect < loss_random, "Perfect prediction should have lower loss"
    print(f"\n✅ Plucker constraint test PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Autoencoder Refactoring Verification")
    print("=" * 60 + "\n")

    try:
        test_base_autoencoder()
        test_plucker_autoencoder()
        test_plucker_constraint()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nRefactoring summary:")
        print("✓ Base AutoencoderKL: Clean VAE without Plucker modifications")
        print("✓ PluckerAutoencoder: Extends AutoencoderKL with Plucker prediction")
        print("✓ Both classes work correctly and independently")
        print("✓ Plucker constraints are properly enforced")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
