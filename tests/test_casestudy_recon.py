"""CPU checks of the reconstruction arms (``prune.recon``, keep 0, ``video_sigma_shift``) of the case-study runner."""

import numpy as np
import pytest
import torch

from fastwam.casestudy.config import parse_config
from fastwam.casestudy.joint_runner import JointCaseStudyRunner
from fastwam.models.wan22.wan_video_dit import precompute_freqs_cis_3d, rope_apply

HEADS, HEAD_DIM, FRAMES, H, W = 2, 128, 3, 7, 14
F0 = H * W


def _freqs() -> torch.Tensor:
    """The video expert's RoPE table for a (FRAMES, H, W) grid, built as in ``pre_dit``."""
    ff = precompute_freqs_cis_3d(HEAD_DIM)
    return torch.cat([
        ff[0][:FRAMES].view(FRAMES, 1, 1, -1).expand(FRAMES, H, W, -1),
        ff[1][:H].view(1, H, 1, -1).expand(FRAMES, H, W, -1),
        ff[2][:W].view(1, 1, W, -1).expand(FRAMES, H, W, -1),
    ], dim=-1).reshape(FRAMES * H * W, 1, -1)


def test_future_freqs_are_the_rows_of_the_full_sequence():
    freqs = _freqs()
    pre = {"freqs": freqs}
    idx = torch.tensor([0, 5, 97, 98, 150, 195])
    got = JointCaseStudyRunner._future_freqs(pre, idx, F0)
    assert torch.equal(got, freqs[idx + F0])
    # A token's own pre-RoPE key through the recon RoPE equals the RoPE of the whole sequence at its row.
    g = torch.Generator().manual_seed(0)
    k_t = torch.randn(1, FRAMES * H * W, HEADS * HEAD_DIM, generator=g).to(torch.bfloat16)
    full = rope_apply(k_t, freqs, HEADS)
    part = rope_apply(k_t[:, idx + F0], got, HEADS)
    assert torch.equal(part, full[:, idx + F0])


def test_r0_key_differs_from_frame0_key_by_the_frame_rotation_only():
    """RoPE_p(k~(p0)) vs RoPE_p0(k~(p0)): same h / w channels, only the frame channels rotate."""
    freqs = _freqs()
    g = torch.Generator().manual_seed(1)
    k_t = torch.randn(1, 1, HEADS * HEAD_DIM, generator=g, dtype=torch.float64)
    p0, p = 17, 2 * F0 + 17  # same (h, w), frame 0 and frame 2
    a = rope_apply(k_t, freqs[p0:p0 + 1], HEADS).reshape(HEADS, HEAD_DIM)
    b = rope_apply(k_t, freqs[p:p + 1], HEADS).reshape(HEADS, HEAD_DIM)
    n_frame = 2 * ((HEAD_DIM - 2 * (HEAD_DIM // 3)) // 2)  # real channels of the frame part
    assert torch.equal(a[:, n_frame:], b[:, n_frame:])
    assert not torch.allclose(a[:, :n_frame], b[:, :n_frame])
    assert torch.allclose(a.norm(dim=-1), b.norm(dim=-1))  # a rotation


def test_median_matches_numpy_and_cosine():
    g = torch.Generator().manual_seed(2)
    for n in (1, 2, 5, 8):
        x = torch.randn(n, 7, generator=g)
        assert np.allclose(JointCaseStudyRunner._median0(x).numpy(), np.median(x.numpy(), axis=0), atol=1e-6)
    a = torch.randn(1, 4, 6, generator=g)
    assert torch.allclose(JointCaseStudyRunner._cos_rows(a, a), torch.ones(4))
    assert torch.allclose(JointCaseStudyRunner._cos_rows(a, -2 * a), -torch.ones(4))


def _arm(**prune):
    return {"method": "prune", "num_inference_steps": 10, "prune": {"selection": "av", "warmup_steps": 1, **prune}}


def test_recon_config():
    for mode in ("r0", "r1", "r2"):
        cfg = parse_config({**_arm(keep_ratio=0.2, recon=mode), "record_recon_quality": True})
        assert cfg.prune.recon == mode and cfg.record_recon_quality
    assert parse_config(_arm(keep_ratio=0.0, recon="r0")).prune.keep_ratio == 0.0
    assert parse_config(_arm(keep_ratio=0.0, action_full_kv=True, video_full_kv=True)).prune.action_full_kv
    with pytest.raises(ValueError):
        parse_config(_arm(keep_ratio=0.0, recon="r1"))  # R1 needs kept tokens
    with pytest.raises(ValueError):
        parse_config(_arm(keep_ratio=0.0))  # S0 at keep 0
    with pytest.raises(ValueError):
        parse_config(_arm(keep_ratio=0.2, recon="r1", action_full_kv=True))
    with pytest.raises(ValueError):
        parse_config(_arm(keep_ratio=0.2, recon="r3"))
    with pytest.raises(ValueError):
        parse_config({**_arm(keep_ratio=0.2), "record_recon_quality": True})
    dense = parse_config({"method": "dense", "video_sigma_shift": 10})
    assert dense.video_sigma_shift == 10.0 and dense.prune is None
    with pytest.raises(ValueError):
        parse_config({"method": "dense", "video_sigma_shift": 0})
