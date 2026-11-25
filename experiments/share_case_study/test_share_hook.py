"""CPU tests of share_hook (run on the server: ``.venv/bin/python -m pytest experiments/share_case_study/test_share_hook.py``)."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import share_hook as sh  # noqa: E402

H, D, L, STEPS, A, F0, N = 2, 8, 3, 4, 5, 6, 10


def sdpa(q, k, v, num_heads, ctx_mask=None, compatibility_mode=True):
    b, sq, _ = q.shape
    qh = q.view(b, sq, num_heads, -1).transpose(1, 2)
    kh = k.view(b, k.shape[1], num_heads, -1).transpose(1, 2)
    vh = v.view(b, v.shape[1], num_heads, -1).transpose(1, 2)
    x = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=ctx_mask)
    return x.transpose(1, 2).reshape(b, sq, -1)


def manual(q, k, v, bias):
    qh = q[0].view(q.shape[1], H, D).transpose(0, 1)
    kh = k[0].view(k.shape[1], H, D).transpose(0, 1)
    vh = v[0].view(v.shape[1], H, D).transpose(0, 1)
    p = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(D) + bias, dim=-1)
    return (p @ vh).transpose(0, 1).reshape(1, q.shape[1], H * D), p


def make(nf=N):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, A, H * D, generator=g)
    k = torch.randn(1, F0 + nf + A, H * D, generator=g)
    v = torch.randn(1, F0 + nf + A, H * D, generator=g)
    return q, k, v


def hooked():
    mod = types.SimpleNamespace(flash_attention=sdpa)
    hk = sh.ShareHook(mod, num_layers=L, num_heads=H, steps=STEPS, action_tokens=A, f0=F0, n_future=N)
    hk.install()
    return mod, hk


def run(mod, hk, cond, nf=N, extra_video=True):
    q, k, v = make(nf)
    outs = []
    hk.begin(cond)
    for _ in range(STEPS):
        for _ in range(L):
            if extra_video:  # a video-row call (q = future rows, no action keys) must pass through untouched
                qv = torch.randn(1, nf, H * D)
                kv = torch.randn(1, F0 + nf, H * D)
                assert torch.equal(mod.flash_attention(q=qv, k=kv, v=kv, num_heads=H), sdpa(qv, kv, kv, H))
            outs.append(mod.flash_attention(q=q, k=k, v=v, num_heads=H))
    return q, k, v, outs, hk.end()


def test_passthrough_bitwise():
    mod, hk = hooked()
    q, k, v, outs, rec = run(mod, hk, sh.Condition())
    assert all(torch.equal(o, sdpa(q, k, v, H)) for o in outs)
    assert rec.masked_calls == 0 and rec.calls == STEPS * L
    _, p = manual(q, k, v, 0.0)
    m = p[..., F0:F0 + N].sum(-1).mean(-1)
    assert torch.allclose(rec.m_last[0], m, atol=1e-6)
    hk.uninstall()
    assert mod.flash_attention is sdpa


def test_bias_all_and_last():
    mod, hk = hooked()
    bias = torch.zeros(F0 + N + A)
    bias[F0:F0 + N] = 1.5
    q, k, v, outs, rec = run(mod, hk, sh.Condition(c=1.5, scope="all"))
    ref, p = manual(q, k, v, bias)
    assert all(torch.allclose(o, ref, atol=1e-5) for o in outs)
    assert torch.allclose(rec.m_last[0], p[..., F0:F0 + N].sum(-1).mean(-1), atol=1e-5)
    assert rec.masked_calls == STEPS * L
    q, k, v, outs, rec = run(mod, hk, sh.Condition(c=1.5, scope="last"))
    assert rec.masked_calls == L
    assert all(torch.equal(o, sdpa(q, k, v, H)) for o in outs[:-L])
    assert all(torch.allclose(o, ref, atol=1e-5) for o in outs[-L:])
    # m rises with c
    _, _, _, _, lo = run(mod, hk, sh.Condition(c=-1.0))
    assert float(lo.m_last.mean()) < float(rec.m_last.mean())


def test_ht_only_when_fewer_keys():
    mod, hk = hooked()
    q, k, v, outs, rec = run(mod, hk, sh.Condition(ht=True))
    assert rec.masked_calls == 0
    q, k, v, outs, rec = run(mod, hk, sh.Condition(ht=True), nf=4)
    bias = torch.zeros(F0 + 4 + A)
    bias[F0:F0 + 4] = math.log(N / 4)
    ref, _ = manual(q, k, v, bias)
    assert all(torch.allclose(o, ref, atol=1e-5) for o in outs)


def test_injected_sparse():
    mod, hk = hooked()
    kk = 3
    q, k, v, outs, rec = run(mod, hk, sh.Condition(sparse_k=kk, select_layer=1, ht=True))
    _, p = manual(q, k, v, 0.0)
    av = p.mean(0)[:, F0:F0 + N].sum(0)
    kept = torch.topk(av, kk).indices.sort().values
    assert torch.equal(rec.kept, kept)
    assert all(torch.equal(o, sdpa(q, k, v, H)) for o in outs[:L])  # step 0 dense
    cols = torch.cat([torch.arange(F0), F0 + kept, torch.arange(F0 + N, F0 + N + A)])
    bias = torch.zeros(len(cols))
    bias[F0:F0 + kk] = math.log(N / kk)
    ref, ps = manual(q, k[:, cols], v[:, cols], bias)
    assert all(torch.allclose(o, ref, atol=1e-5) for o in outs[L:])
    assert torch.allclose(rec.m_last[0], ps[..., F0:F0 + kk].sum(-1).mean(-1), atol=1e-5)
    assert rec.nf_steps == [N] * STEPS


def test_call_count_checked():
    mod, hk = hooked()
    hk.begin(sh.Condition())
    q, k, v = make()
    mod.flash_attention(q=q, k=k, v=v, num_heads=H)
    with pytest.raises(RuntimeError):
        hk.end()


def test_bf16_value():
    assert sh.bf16(1.0) == 1.0 and abs(sh.bf16(math.log(196 / 39)) - 1.6172) < 1e-3
