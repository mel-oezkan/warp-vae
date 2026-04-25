"""
Unit tests for EQVAEAutoencoder implementation.

Tests cover:
- Model initialization
- Transformation sampling
- Latent transformation correctness
- Image transformation consistency
- Dual optimizer setup
- Training step execution
"""

import torch
import pytest
from ldm.models.autoencoder import EQVAEAutoencoder
from ldm.modules.losses import LPIPSWithDiscriminator


def get_test_config():
    """Get basic configuration for testing."""
    return {
        'ddconfig': {
            'double_z': True,
            'z_channels': 4,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0
        },
        'lossconfig': {
            'target': 'ldm.modules.losses.LPIPSWithDiscriminator',
            'params': {
                'disc_start': 50001,
                'kl_weight': 0.000001,
                'disc_weight': 0.5,
                'perceptual_weight': 1.0,
                'disc_in_channels': 3,
                'disc_num_layers': 3,
                'use_actnorm': False
            }
        },
        'embed_dim': 4,
        'p_prior': 0.9,
        'scale_range': [0.25, 1.0],
        'use_rotation': True
    }


def test_eqvae_initialization():
    """Test EQVAEAutoencoder can be instantiated with correct parameters."""
    config = get_test_config()

    # Instantiate loss module separately
    from ldm.util import instantiate_from_config
    loss_module = instantiate_from_config(config['lossconfig'])

    # Create model
    model = EQVAEAutoencoder(
        ddconfig=config['ddconfig'],
        lossconfig=loss_module,
        embed_dim=config['embed_dim'],
        p_prior=config['p_prior'],
        scale_range=config['scale_range'],
        use_rotation=config['use_rotation']
    )

    # Verify parameters
    assert model.p_prior == 0.9, f"Expected p_prior=0.9, got {model.p_prior}"
    assert model.scale_range == [0.25, 1.0], f"Expected scale_range=[0.25, 1.0], got {model.scale_range}"
    assert model.use_rotation == True, f"Expected use_rotation=True, got {model.use_rotation}"
    assert isinstance(model.loss, LPIPSWithDiscriminator), f"Expected LPIPSWithDiscriminator, got {type(model.loss)}"

    print("✅ test_eqvae_initialization PASSED")


def test_transformation_sampling():
    """Test transformation parameter sampling produces valid ranges."""
    config = get_test_config()
    from ldm.util import instantiate_from_config
    loss_module = instantiate_from_config(config['lossconfig'])

    model = EQVAEAutoencoder(
        ddconfig=config['ddconfig'],
        lossconfig=loss_module,
        embed_dim=config['embed_dim'],
        p_prior=config['p_prior'],
        scale_range=config['scale_range'],
        use_rotation=config['use_rotation']
    )

    # Sample 100 transformations
    scales = []
    rotations = []
    for _ in range(100):
        params = model._sample_transformation()
        scales.append(params['scale'])
        rotations.append(params['rotation'])

    # Verify scale range
    min_scale = min(scales)
    max_scale = max(scales)
    assert min_scale >= 0.25, f"Scale below minimum: {min_scale}"
    assert max_scale <= 1.0, f"Scale above maximum: {max_scale}"

    # Verify rotation values are in {0, 1, 2, 3}
    rotation_set = set(rotations)
    assert rotation_set.issubset({0, 1, 2, 3}), f"Invalid rotations: {rotation_set}"

    print(f"✅ test_transformation_sampling PASSED")
    print(f"   Scale range: [{min_scale:.3f}, {max_scale:.3f}]")
    print(f"   Rotations sampled: {rotation_set}")


def test_latent_transformation_shape():
    """Test latent transformation maintains shape."""
    config = get_test_config()
    from ldm.util import instantiate_from_config
    loss_module = instantiate_from_config(config['lossconfig'])

    model = EQVAEAutoencoder(
        ddconfig=config['ddconfig'],
        lossconfig=loss_module,
        embed_dim=config['embed_dim']
    )

    # Create dummy latent
    z = torch.randn(2, 4, 32, 32)  # [B, C, H, W]

    # Test scaling
    params_scale = {'scale': 0.5, 'rotation': 0}
    z_scaled = model._transform_latent(z, params_scale)
    assert z_scaled.shape == z.shape, f"Shape mismatch after scaling: {z_scaled.shape} vs {z.shape}"

    # Test rotation
    params_rot = {'scale': 1.0, 'rotation': 1}  # 90 degrees
    z_rotated = model._transform_latent(z, params_rot)
    assert z_rotated.shape == z.shape, f"Shape mismatch after rotation: {z_rotated.shape} vs {z.shape}"

    # Test combined
    params_combined = {'scale': 0.75, 'rotation': 2}  # Scale + 180 degrees
    z_combined = model._transform_latent(z, params_combined)
    assert z_combined.shape == z.shape, f"Shape mismatch after combined: {z_combined.shape} vs {z.shape}"

    print("✅ test_latent_transformation_shape PASSED")


def test_rotation_360_degrees():
    """Test that 4x 90-degree rotation returns to original."""
    config = get_test_config()
    from ldm.util import instantiate_from_config
    loss_module = instantiate_from_config(config['lossconfig'])

    model = EQVAEAutoencoder(
        ddconfig=config['ddconfig'],
        lossconfig=loss_module,
        embed_dim=config['embed_dim']
    )

    # Create dummy latent
    z_original = torch.randn(2, 4, 32, 32)

    # Apply 4x 90-degree rotation (360 degrees total)
    z_temp = z_original.clone()
    for _ in range(4):
        z_temp = model._transform_latent(z_temp, {'scale': 1.0, 'rotation': 1})

    # Should return to original (within floating point precision)
    diff = torch.abs(z_temp - z_original).max().item()
    assert diff < 1e-5, f"360-degree rotation not identity: max diff = {diff}"

    print(f"✅ test_rotation_360_degrees PASSED")
    print(f"   Max difference after 360° rotation: {diff:.2e}")


def test_image_transformation_shape():
    """Test image transformation maintains shape."""
    config = get_test_config()
    from ldm.util import instantiate_from_config
    loss_module = instantiate_from_config(config['lossconfig'])

    model = EQVAEAutoencoder(
        ddconfig=config['ddconfig'],
        lossconfig=loss_module,
        embed_dim=config['embed_dim']
    )

    # Create dummy image
    x = torch.randn(2, 3, 256, 256)  # [B, C, H, W]

    # Test various transformations
    test_cases = [
        {'scale': 0.5, 'rotation': 0, 'name': 'scale_only'},
        {'scale': 1.0, 'rotation': 1, 'name': 'rotate_only'},
        {'scale': 0.75, 'rotation': 2, 'name': 'combined'},
    ]

    for params in test_cases:
        x_transformed = model._transform_image(x, params)
        assert x_transformed.shape == x.shape, \
            f"Shape mismatch for {params['name']}: {x_transformed.shape} vs {x.shape}"

    print("✅ test_image_transformation_shape PASSED")


def test_dual_optimizers():
    """Test dual optimizer configuration."""
    config = get_test_config()
    from ldm.util import instantiate_from_config
    loss_module = instantiate_from_config(config['lossconfig'])

    model = EQVAEAutoencoder(
        ddconfig=config['ddconfig'],
        lossconfig=loss_module,
        embed_dim=config['embed_dim']
    )

    optimizers, schedulers = model.configure_optimizers()

    # Verify structure
    assert len(optimizers) == 2, f"Expected 2 optimizers, got {len(optimizers)}"
    assert len(schedulers) == 0, f"Expected 0 schedulers, got {len(schedulers)}"

    # Check optimizer 0 has encoder/decoder params
    opt_ae = optimizers[0]
    ae_param_count = sum(len(pg['params']) for pg in opt_ae.param_groups)
    assert ae_param_count > 0, "Autoencoder optimizer has no parameters"

    # Check optimizer 1 has discriminator params
    opt_disc = optimizers[1]
    disc_param_count = sum(len(pg['params']) for pg in opt_disc.param_groups)
    assert disc_param_count > 0, "Discriminator optimizer has no parameters"

    print(f"✅ test_dual_optimizers PASSED")
    print(f"   Autoencoder params: {ae_param_count}")
    print(f"   Discriminator params: {disc_param_count}")


def test_eqvae_forward():
    """Test EQ-VAE forward pass executes without errors."""
    config = get_test_config()
    from ldm.util import instantiate_from_config
    loss_module = instantiate_from_config(config['lossconfig'])

    model = EQVAEAutoencoder(
        ddconfig=config['ddconfig'],
        lossconfig=loss_module,
        embed_dim=config['embed_dim']
    )
    model.eval()  # Set to eval mode for deterministic behavior

    # Create dummy input
    x = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        # Test EQ-VAE forward
        reconstruction, posterior, x_transformed = model._eqvae_forward(x)

        # Verify outputs
        assert reconstruction.shape == x.shape, \
            f"Reconstruction shape mismatch: {reconstruction.shape} vs {x.shape}"
        assert x_transformed.shape == x.shape, \
            f"Transformed input shape mismatch: {x_transformed.shape} vs {x.shape}"
        assert posterior is not None, "Posterior is None"

    print("✅ test_eqvae_forward PASSED")


def test_training_step_no_crash():
    """Test training step executes without crashing for both optimizers."""
    config = get_test_config()
    from ldm.util import instantiate_from_config
    loss_module = instantiate_from_config(config['lossconfig'])

    model = EQVAEAutoencoder(
        ddconfig=config['ddconfig'],
        lossconfig=loss_module,
        embed_dim=config['embed_dim'],
        p_prior=1.0  # Always apply EQ-VAE for testing
    )

    # Create dummy batch
    batch = {
        'image': torch.randn(2, 3, 256, 256)
    }

    # Test optimizer 0 (autoencoder)
    try:
        loss_ae = model.training_step(batch, 0, optimizer_idx=0)
        assert loss_ae is not None, "Autoencoder loss is None"
        assert not torch.isnan(loss_ae), "Autoencoder loss is NaN"
        print(f"   Autoencoder loss: {loss_ae.item():.4f}")
    except Exception as e:
        pytest.fail(f"Training step failed for optimizer 0: {e}")

    # Test optimizer 1 (discriminator)
    try:
        loss_disc = model.training_step(batch, 0, optimizer_idx=1)
        assert loss_disc is not None, "Discriminator loss is None"
        assert not torch.isnan(loss_disc), "Discriminator loss is NaN"
        print(f"   Discriminator loss: {loss_disc.item():.4f}")
    except Exception as e:
        pytest.fail(f"Training step failed for optimizer 1: {e}")

    print("✅ test_training_step_no_crash PASSED")


if __name__ == '__main__':
    """Run tests manually without pytest."""
    print("\n" + "="*60)
    print("Running EQVAEAutoencoder Unit Tests")
    print("="*60 + "\n")

    tests = [
        test_eqvae_initialization,
        test_transformation_sampling,
        test_latent_transformation_shape,
        test_rotation_360_degrees,
        test_image_transformation_shape,
        test_dual_optimizers,
        test_eqvae_forward,
        test_training_step_no_crash,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"\nRunning {test_func.__name__}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")
        exit(1)
