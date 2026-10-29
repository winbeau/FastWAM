"""CPU checks of the action-row attention re-budget (``prune.steer``) of the case-study runner."""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fastwam.casestudy.config import SteerConfig, parse_config
from fastwam.casestudy.joint_runner import JointCaseStudyRunner

HEADS, DIM, F0, NF, NA = 4, 16, 6, 12, 5


def _runner() -> JointCaseStudyRunner:
    runner = object.__new__(JointCaseStudyRunner)  # the helpers only need the head count
    runner.num_heads = HEADS
    return runner


def _qkv(n_keys: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, NA, HEADS * DIM, generator=g)
    k = torch.randn(1, n_keys, HEADS * DIM, generator=g)
    v = torch.randn(1, n_keys, HEADS * DIM, generator=g)
    return q, k, v


def _warmup(runner, seed: int = 0):
    q, k, _ = _qkv(F0 + NF + NA, seed)
    logits = runner._head_logits(q, k)
    return logits, torch.softmax(logits, dim=-1)[..., F0:F0 + NF].contiguous()  # as the runner stores it


def test_lambda_zero_is_plain_attention():
    runner = _runner()
    q, k, v = _qkv(F0 + 4 + NA)
    out, share = runner._steered_attention(q, k, v, slice(F0, F0 + 4), torch.ones(HEADS, NA), 0.0)
    ref = F.scaled_dot_product_attention(*(rearrange(x, "b s (n d) -> b n s d", n=HEADS) for x in (q, k, v)))
    assert torch.allclose(out, rearrange(ref, "b n s d -> b s (n d)"), atol=1e-5)
    assert share.shape == (HEADS, NA)


def test_bias_restores_dense_future_share():
    runner = _runner()
    logits, fut = _warmup(runner)
    kept = torch.tensor([1, 4, 7, 10])
    bias, diag = runner._steer_bias(SteerConfig(lam=0.0, clamp=30.0), {0: fut}, {0: logits}, kept, F0, NF)
    assert diag["clamped_fraction"] == 0.0
    assert diag["check_max_abs"] < 1e-6
    assert torch.all(bias[0] >= 0)
    # The same bias inside a pruned step's key layout [frame-0 | kept | action]:
    cols = torch.cat([torch.arange(F0), kept + F0, torch.arange(F0 + NF, F0 + NF + NA)])
    probs = runner._steered_probs(logits[..., cols], slice(F0, F0 + 4), bias[0], 1.0)
    assert torch.allclose(probs[..., F0:F0 + 4].sum(-1), fut.sum(-1), atol=1e-6)
    # lam 0 leaves the sparse share, lam < 0 lowers it
    p0 = runner._steered_probs(logits[..., cols], slice(F0, F0 + 4), bias[0], 0.0)[..., F0:F0 + 4].sum(-1)
    pm = runner._steered_probs(logits[..., cols], slice(F0, F0 + 4), bias[0], -0.5)[..., F0:F0 + 4].sum(-1)
    assert torch.all(pm <= p0 + 1e-7)


def test_clamp_is_reported():
    runner = _runner()
    logits, fut = _warmup(runner, seed=1)
    _, diag = runner._steer_bias(SteerConfig(lam=1.0, clamp=1e-3), {0: fut}, {0: logits}, torch.tensor([0]), F0, NF)
    assert diag["b_max"] <= 1e-3 + 1e-9
    assert diag["clamped_fraction"] > 0
    assert diag["check_max_abs_clamped"] > 0


def test_keep_all_gives_zero_bias():
    runner = _runner()
    logits, fut = _warmup(runner, seed=2)
    bias, diag = runner._steer_bias(SteerConfig(lam=1.0), {0: fut}, {0: logits}, torch.arange(NF), F0, NF)
    assert diag["b_max"] == 0.0 and diag["b_min"] == 0.0
    assert torch.count_nonzero(bias[0]) == 0


def test_config():
    cfg = parse_config({"method": "prune", "record_future_share": True, "record_motion": True,
                        "prune": {"selection": "av", "keep_ratio": 0.2, "steer": {"lam": 1.5}}})
    assert cfg.prune.steer == SteerConfig(lam=1.5, clamp=3.0)
    assert cfg.record_future_share and cfg.record_motion
    with pytest.raises(ValueError):
        parse_config({"method": "prune", "prune": {"steer": {"lam": 1.0}, "reselect_every_step": True}})
    with pytest.raises(ValueError):
        parse_config({"method": "prune", "prune": {"steer": {"lam": 1.0}, "action_full_kv": True}})
    with pytest.raises(ValueError):
        parse_config({"method": "prune", "prune": {"steer": {"gain": 1.0}}})
