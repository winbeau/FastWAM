"""CPU checks of the pilot + automatic gain (``casestudy.agc``): the rescaling path, the gains, the config, the pilot's
x0 convention."""

import numpy as np
import pytest
import torch

from fastwam.casestudy import agc
from fastwam.casestudy.config import AGCConfig, parse_config
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


def _chunk(seed: int, scale: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((32, 7)) * scale).astype(np.float32)
    x[:, 6] = np.sign(x[:, 6]) * rng.uniform(0.2, 1.0, 32).astype(np.float32)
    return x


def test_unit_gain_is_the_identity_bit_for_bit():
    x = _chunk(0)
    x[3, 0], x[5, 4], x[7, 2] = 1.3, -1.7, 1.0000001  # outside the environment range: still untouched at g = 1
    out = agc.apply(x, 1.0, 1.0, 1.0)
    assert out.dtype == x.dtype and out.tobytes() == x.tobytes()
    assert out is not x


def test_gripper_never_changes_and_scaled_dims_are_clipped_products():
    rng = np.random.default_rng(1)
    for i in range(200):
        x = _chunk(i, scale=0.8)
        g_t, g_r = (float(v) for v in rng.uniform(1.0, 1.5, 2))
        if i % 4 == 0:
            g_r = 1.0
        out = agc.apply(x, g_t, g_r, 1.0)
        assert out[:, 6].tobytes() == x[:, 6].tobytes()
        assert np.array_equal(out[:, 0:3], np.clip(x[:, 0:3] * g_t, -1.0, 1.0))
        if g_r == 1.0:
            assert out[:, 3:6].tobytes() == x[:, 3:6].tobytes()
        else:
            assert np.array_equal(out[:, 3:6], np.clip(x[:, 3:6] * g_r, -1.0, 1.0))
        assert np.abs(out[:, :6]).max() <= 1.0 + 1e-7 or np.abs(x[:, :6]).max() > 1.0


def test_gains_pilot_and_const():
    cfg = AGCConfig(mode="pilot", c_t=1.2, c_r=1.1, tau_d=0.05, tau_r=0.01, executed=10)
    out = np.zeros((32, 7), np.float32)
    pilot = np.zeros((32, 7), np.float32)
    out[:10, 0], pilot[:10, 0] = 0.02, 0.025  # |D(out)| = 0.2, |D(p)| = 0.25
    out[:10, 3], pilot[:10, 3] = 0.003, 0.003  # |R| = 0.03 each
    out[10:, :6] = 5.0  # steps after the executed ones do not count
    g = agc.gains(cfg, out, pilot)
    assert g["d_out"] == pytest.approx(0.2) and g["d_pilot"] == pytest.approx(0.25)
    assert g["g_t"] == pytest.approx(1.2 * 0.25 / 0.2) and not g["skip_t"]
    assert g["g_r"] == pytest.approx(1.1) and not g["skip_r"]
    # clipped at hi and at lo
    assert agc.gains(AGCConfig(c_t=3.0, executed=10), out, pilot)["g_t"] == 1.5
    assert agc.gains(AGCConfig(c_t=0.5, executed=10), out, pilot)["g_t"] == 1.0
    # below the threshold (either norm): skipped, g = 1
    small = out.copy()
    small[:10, 0] = 0.004  # |D(out)| = 0.04 < tau_d
    g = agc.gains(cfg, small, pilot)
    assert g["skip_t"] and g["g_t"] == 1.0 and g["raw_t"] is None
    g = agc.gains(AGCConfig(tau_r=0.05), out, pilot)  # |R(p)| = 0.03 < tau_r
    assert g["skip_r"] and g["g_r"] == 1.0
    # const: one clipped gain for both, never skipped
    g = agc.gains(AGCConfig(mode="const", gain=1.25), small, None)
    assert g["g_t"] == g["g_r"] == 1.25 and not g["skip_t"]
    assert agc.gains(AGCConfig(mode="const", gain=2.0), small)["g_t"] == 1.5
    with pytest.raises(ValueError):
        agc.gains(cfg, out, None)


def test_config():
    cfg = parse_config({"method": "prune", "prune": {"keep_ratio": 0.2}, "agc": {"mode": "pilot", "c_t": 1.3, "tau_d": 0.1}})
    assert cfg.agc.c_t == 1.3 and cfg.agc.hi == 1.5 and cfg.agc.executed == 10 and not cfg.agc.force_unit
    assert parse_config({"method": "dense", "record_pilot": True}).record_pilot
    for bad in ({"agc": {"mode": "x"}}, {"agc": {"lo": 1.6}}, {"agc": {"c_t": 0}}, {"agc": {"tau_d": -1}},
                {"agc": {"bogus": 1}}, {"record_pilot": True, "frame0_cache": False}):
        with pytest.raises(ValueError):
            parse_config({"method": "dense", **bad})


def test_no_future_and_extrapolation_config():
    s0 = {"selection": "av", "keep_ratio": 0.2, "warmup_steps": 1}
    assert parse_config({"method": "prune", "prune": {**s0, "action_no_future": True}}).prune.action_no_future
    assert parse_config({"method": "prune", "prune": {**s0, "extrapolate_w": 0.4}}).prune.extrapolate_w == 0.4
    for bad in ({"action_no_future": True, "extrapolate_w": 0.4}, {"extrapolate_w": float("nan")},
                {"extrapolate_w": 0.4, "action_full_kv": True}, {"action_no_future": True, "recon": "r0"}):
        with pytest.raises(ValueError):
            parse_config({"method": "prune", "prune": {**s0, **bad}})


def test_pilot_is_x0_under_the_scheduler_convention():
    """x_sigma = (1 - sigma) x0 + sigma eps and v = eps - x0, so x - sigma v = x0 (sigma_0 = 1 at shift 5)."""
    sch = WanContinuousFlowMatchScheduler(shift=5.0)
    t, _ = sch.build_inference_schedule(10, device="cpu", dtype=torch.float32)
    sigma0 = float(t[0]) / sch.num_train_timesteps
    assert sigma0 == 1.0
    g = torch.Generator().manual_seed(3)
    x0, eps = torch.randn(32, 7, generator=g, dtype=torch.float64), torch.randn(32, 7, generator=g, dtype=torch.float64)
    for sigma in (1.0, 0.7, 0.2):
        x = sch.add_noise(x0, eps, torch.tensor(sigma * 1000.0, dtype=torch.float64))
        v = sch.training_target(x0, eps, None)
        assert torch.allclose(x - sigma * v, x0, atol=1e-12)
