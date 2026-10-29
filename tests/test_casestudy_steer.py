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
    out, probs = runner._steered_attention(q, k, v, slice(F0, F0 + 4), torch.ones(HEADS, NA), 0.0)
    ref = F.scaled_dot_product_attention(*(rearrange(x, "b s (n d) -> b n s d", n=HEADS) for x in (q, k, v)))
    assert torch.allclose(out, rearrange(ref, "b n s d -> b s (n d)"), atol=1e-5)
    assert probs.shape == (HEADS, NA, F0 + 4 + NA)
    # an index tensor selects the same columns as the slice
    _, p_idx = runner._steered_attention(q, k, v, torch.arange(F0, F0 + 4), torch.ones(HEADS, NA), 0.7)
    _, p_sl = runner._steered_attention(q, k, v, slice(F0, F0 + 4), torch.ones(HEADS, NA), 0.7)
    assert torch.equal(p_idx, p_sl)


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
        parse_config({"method": "prune", "prune": {"steer": {"gain": 1.0}}})


def _video_warmup(runner, seed: int = 3):
    """Video rows at step 0: future queries [H, NF, F0 + NF] over keys [frame-0 | future]."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, NF, HEADS * DIM, generator=g)
    k = torch.randn(1, F0 + NF, HEADS * DIM, generator=g)
    logits = runner._head_logits(q, k)
    return logits, torch.softmax(logits, dim=-1)[..., F0:]


def test_video_bias_restores_dense_share():
    runner = _runner()
    logits, fut = _video_warmup(runner)
    kept = torch.tensor([0, 3, 5, 9])
    bias, diag = runner._steer_video_bias(SteerConfig(lam=1.0, clamp=30.0), {0: fut}, {0: logits}, kept, F0)
    assert bias[0].shape == (HEADS, 4)
    assert diag["clamped_fraction"] == 0.0 and diag["check_max_abs"] < 1e-6
    # inside a pruned step: kept queries over [frame-0 | kept], bias on the kept columns
    lg = logits[:, kept][..., torch.cat([torch.arange(F0), kept + F0])]
    share = runner._steered_probs(lg, slice(F0, F0 + 4), bias[0], 1.0)[..., F0:].sum(-1)
    assert torch.allclose(share, fut[:, kept].sum(-1), atol=1e-6)


def test_video_keep_all_gives_zero_bias():
    runner = _runner()
    logits, fut = _video_warmup(runner, seed=4)
    bias, diag = runner._steer_video_bias(SteerConfig(lam=1.0), {0: fut}, {0: logits}, torch.arange(NF), F0)
    assert torch.count_nonzero(bias[0]) == 0 and diag["b_max"] == 0.0


def test_dense_bias_gives_sparse_share():
    runner = _runner()
    logits, fut = _warmup(runner, seed=5)
    kept = torch.tensor([2, 6, 8])
    bias, diag = runner._steer_bias(SteerConfig(lam=-1.0, clamp=30.0), {0: fut}, {0: logits}, kept, F0, NF,
                                    dense_bias=True)
    assert diag["dense_bias_check_max_abs"] < 1e-6
    # the share with -b on every future column equals the share of the row restricted to the kept columns
    dense = runner._steered_probs(logits, slice(F0, F0 + NF), bias[0], -1.0)[..., F0:F0 + NF].sum(-1)
    cols = torch.cat([torch.arange(F0), kept + F0, torch.arange(F0 + NF, F0 + NF + NA)])
    sparse = torch.softmax(logits[..., cols], dim=-1)[..., F0:F0 + 3].sum(-1)
    assert torch.allclose(dense, sparse, atol=1e-6)


def test_context_probe_config():
    base = {"selection": "av", "keep_ratio": 0.2, "warmup_steps": 1, "steer": {"lam": 0.0}}
    v1 = parse_config({"method": "prune", "record_video_frame0": True, "record_motion_steps": True,
                       "prune": {**base, "video_eager": True, "steer_video": {"lam": 1.0}}})
    assert v1.prune.steer_video == SteerConfig(lam=1.0) and v1.record_video_frame0 and v1.record_motion_steps
    v2 = parse_config({"method": "prune", "prune": {**base, "video_eager": True, "action_full_kv": True, "video_full_kv": True}})
    assert v2.prune.video_full_kv and v2.prune.action_full_kv
    assert parse_config({"method": "prune", "prune": {**base, "video_eager": True, "oracle_kv": True}}).prune.oracle_kv
    assert parse_config({"method": "prune", "prune": {**base, "steer": {"lam": -1.0}, "dense_bias": True}}).prune.dense_bias
    for bad in ({"steer_video": {"lam": 1.0}},                                      # needs video_eager
                {"video_eager": True, "video_full_kv": True},                       # needs action_full_kv
                {"video_eager": True, "oracle_kv": True, "action_full_kv": True},   # exclusive
                {"dense_bias": True, "video_eager": True},                          # dense_bias is plain dense
                {"warmup_steps": 2, "video_eager": True, "steer_video": {"lam": 1.0}}):
        with pytest.raises(ValueError):
            parse_config({"method": "prune", "num_inference_steps": 10, "prune": {**base, **bad}})
    with pytest.raises(ValueError):  # every context probe needs the shared action-row path
        parse_config({"method": "prune", "prune": {"selection": "av", "keep_ratio": 0.2, "video_eager": True}})
