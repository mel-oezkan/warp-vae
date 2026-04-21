"""
Unit tests for the EMA-normalised adaptive weight in LPIPSWithDiscriminator.

Tests cover:
- EMA buffers are registered and initialised correctly
- EMA values update correctly each call
- d_weight stays near 1 when both norms are at parity
- d_weight stays bounded when g_grads are near-zero (the pre-warmed generator case)
- d_weight stays bounded when nll_grads are large relative to g_grads
- Buffers survive a device round-trip (state_dict save/load)
- d_weight is scale-invariant: scaling both inputs by a constant leaves it unchanged
"""

import torch
import pytest
from ldm.modules.losses import LPIPSWithDiscriminator


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_loss(**kwargs):
    """Instantiate a minimal LPIPSWithDiscriminator for testing."""
    defaults = dict(
        disc_start=50001,
        kl_weight=1e-6,
        disc_weight=1.0,
        perceptual_weight=1.0,
        disc_in_channels=3,
        disc_num_layers=2,
        use_actnorm=False,
    )
    defaults.update(kwargs)
    return LPIPSWithDiscriminator(**defaults).to(DEVICE)


def make_param_and_losses(nll_grad_scale=1.0, g_grad_scale=1.0):
    """
    Return (loss_module, nll_loss, g_loss, last_layer) where the gradient norms
    at last_layer are approximately proportional to nll_grad_scale / g_grad_scale.

    We use a single linear layer as `last_layer` so we control the flow.
    """
    # A single weight tensor that both losses flow through
    w = torch.nn.Parameter(torch.randn(4, 4, device=DEVICE))

    # nll_loss: sum(w) scaled so ∂nll/∂w has norm ~ nll_grad_scale
    nll_loss = nll_grad_scale * w.sum()

    # g_loss: sum(w) scaled so ∂g/∂w has norm ~ g_grad_scale
    g_loss = g_grad_scale * w.sum()

    return w, nll_loss, g_loss


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ema_buffers_registered():
    """EMA buffers exist, are tensors, and are initialised to 1."""
    loss = make_loss()
    assert hasattr(loss, 'ema_nll_grad_norm'), "ema_nll_grad_norm buffer missing"
    assert hasattr(loss, 'ema_g_grad_norm'), "ema_g_grad_norm buffer missing"
    assert loss.ema_nll_grad_norm.item() == pytest.approx(1.0)
    assert loss.ema_g_grad_norm.item() == pytest.approx(1.0)
    # Buffers should NOT be parameters
    param_names = [n for n, _ in loss.named_parameters()]
    assert 'ema_nll_grad_norm' not in param_names
    assert 'ema_g_grad_norm' not in param_names
    print("test_ema_buffers_registered PASSED")


def test_ema_updates_each_call():
    """EMA values change after each call to calculate_adaptive_weight."""
    loss = make_loss(grad_norm_ema_decay=0.9)
    w, nll_loss, g_loss = make_param_and_losses(nll_grad_scale=2.0, g_grad_scale=3.0)

    ema_nll_before = loss.ema_nll_grad_norm.item()
    ema_g_before = loss.ema_g_grad_norm.item()

    loss.calculate_adaptive_weight(nll_loss, g_loss, last_layer=w)

    assert loss.ema_nll_grad_norm.item() != ema_nll_before, "ema_nll did not update"
    assert loss.ema_g_grad_norm.item() != ema_g_before, "ema_g did not update"
    print("test_ema_updates_each_call PASSED")


def test_ema_converges_to_grad_norm():
    """After many calls with constant grad norms the EMAs converge to those norms."""
    decay = 0.9
    loss = make_loss(grad_norm_ema_decay=decay)
    nll_scale, g_scale = 5.0, 3.0

    # Run enough steps that EMA is effectively converged (>5x the time constant)
    steps = int(5 / (1 - decay))
    for _ in range(steps):
        w, nll_loss, g_loss = make_param_and_losses(nll_scale, g_scale)
        loss.calculate_adaptive_weight(nll_loss, g_loss, last_layer=w)

    # For w of shape (4,4) with all-ones gradient: norm = nll_scale * sqrt(16) = nll_scale * 4
    expected_nll = nll_scale * 4.0  # ‖∂(scale * sum(w)) / ∂w‖ = scale * ‖ones(4,4)‖
    expected_g = g_scale * 4.0
    assert loss.ema_nll_grad_norm.item() == pytest.approx(expected_nll, rel=0.05), \
        f"EMA nll {loss.ema_nll_grad_norm.item():.4f} not close to {expected_nll:.4f}"
    assert loss.ema_g_grad_norm.item() == pytest.approx(expected_g, rel=0.05), \
        f"EMA g {loss.ema_g_grad_norm.item():.4f} not close to {expected_g:.4f}"
    print(f"test_ema_converges_to_grad_norm PASSED "
          f"(ema_nll={loss.ema_nll_grad_norm.item():.3f}, ema_g={loss.ema_g_grad_norm.item():.3f})")


def test_d_weight_near_one_at_parity():
    """When both gradient norms are equal and EMAs have converged, d_weight ≈ disc_weight."""
    decay = 0.9
    disc_weight = 0.5
    loss = make_loss(grad_norm_ema_decay=decay, disc_weight=disc_weight)

    # Warm up EMAs with equal norms
    steps = int(5 / (1 - decay))
    for _ in range(steps):
        w, nll_loss, g_loss = make_param_and_losses(1.0, 1.0)
        d_weight = loss.calculate_adaptive_weight(nll_loss, g_loss, last_layer=w)

    # At parity: (norm/ema_nll) / (norm/ema_g) = 1, then * disc_weight
    assert d_weight.item() == pytest.approx(disc_weight, rel=0.05), \
        f"d_weight at parity = {d_weight.item():.4f}, expected ~{disc_weight}"
    print(f"test_d_weight_near_one_at_parity PASSED (d_weight={d_weight.item():.4f})")


def test_d_weight_bounded_when_g_grads_near_zero():
    """
    The pre-warmed generator case: g_grads are near-zero at discriminator activation.
    Old code: ratio → 1e4 (explodes).
    New code: ema_g also near-zero → normalised ratio stays near 1.
    """
    decay = 0.9
    loss = make_loss(grad_norm_ema_decay=decay, disc_weight=1.0)

    # Warm up with near-zero g_grads (pre-warmed generator scenario)
    steps = int(5 / (1 - decay))
    for _ in range(steps):
        w, nll_loss, g_loss = make_param_and_losses(
            nll_grad_scale=1000.0,  # large nll gradient (high nll_loss scale)
            g_grad_scale=0.001,     # near-zero g gradient (generator already good)
        )
        d_weight = loss.calculate_adaptive_weight(nll_loss, g_loss, last_layer=w)

    # With old code this would be ~1e4; with EMA normalisation it should stay near 1
    assert d_weight.item() < 10.0, \
        f"d_weight exploded to {d_weight.item():.1f} — EMA normalisation not working"
    print(f"test_d_weight_bounded_when_g_grads_near_zero PASSED (d_weight={d_weight.item():.4f})")


def test_d_weight_scale_invariant():
    """
    Multiplying both nll_loss and g_loss by the same constant should not change d_weight
    once EMAs have converged, because both norms and their EMAs scale equally.
    """
    decay = 0.9
    steps = int(5 / (1 - decay))

    results = []
    for scale in [1.0, 10.0, 1000.0]:
        loss = make_loss(grad_norm_ema_decay=decay, disc_weight=1.0)
        for _ in range(steps):
            w, nll_loss, g_loss = make_param_and_losses(
                nll_grad_scale=2.0 * scale,
                g_grad_scale=1.0 * scale,
            )
            d_weight = loss.calculate_adaptive_weight(nll_loss, g_loss, last_layer=w)
        results.append(d_weight.item())

    # All three should be approximately equal (scale-invariant)
    assert results[0] == pytest.approx(results[1], rel=0.05), \
        f"d_weight differs between scale=1 ({results[0]:.4f}) and scale=10 ({results[1]:.4f})"
    assert results[0] == pytest.approx(results[2], rel=0.05), \
        f"d_weight differs between scale=1 ({results[0]:.4f}) and scale=1000 ({results[2]:.4f})"
    print(f"test_d_weight_scale_invariant PASSED (d_weights={[f'{r:.4f}' for r in results]})")


def test_buffers_survive_state_dict_roundtrip():
    """EMA buffer values are preserved through state_dict save and load."""
    decay = 0.9
    loss = make_loss(grad_norm_ema_decay=decay)

    # Run a few steps to move EMAs away from 1.0
    for _ in range(20):
        w, nll_loss, g_loss = make_param_and_losses(3.0, 7.0)
        loss.calculate_adaptive_weight(nll_loss, g_loss, last_layer=w)

    saved_nll = loss.ema_nll_grad_norm.item()
    saved_g = loss.ema_g_grad_norm.item()

    # Round-trip through state_dict
    state = loss.state_dict()
    loss2 = make_loss(grad_norm_ema_decay=decay)
    loss2.load_state_dict(state)

    assert loss2.ema_nll_grad_norm.item() == pytest.approx(saved_nll, rel=1e-5), \
        "ema_nll_grad_norm not restored after state_dict load"
    assert loss2.ema_g_grad_norm.item() == pytest.approx(saved_g, rel=1e-5), \
        "ema_g_grad_norm not restored after state_dict load"
    print(f"test_buffers_survive_state_dict_roundtrip PASSED "
          f"(nll={saved_nll:.4f}, g={saved_g:.4f})")


if __name__ == '__main__':
    print(f"\nUsing device: {DEVICE}")
    print("=" * 60)
    print("Running EMA Adaptive Weight Tests")
    print("=" * 60 + "\n")

    tests = [
        test_ema_buffers_registered,
        test_ema_updates_each_call,
        test_ema_converges_to_grad_norm,
        test_d_weight_near_one_at_parity,
        test_d_weight_bounded_when_g_grads_near_zero,
        test_d_weight_scale_invariant,
        test_buffers_survive_state_dict_roundtrip,
    ]

    passed = 0
    failed = 0
    for test_func in tests:
        try:
            print(f"Running {test_func.__name__}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        exit(1)
