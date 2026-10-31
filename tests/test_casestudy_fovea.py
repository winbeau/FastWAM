"""CPU checks of the fovea round (``fastwam/casestudy/fovea.py`` and the runner's key bias)."""

import math

import pytest
import torch

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config
from fastwam.models.wan22.wan_video_dit import flash_attention, precompute_freqs_cis_3d

GRID = sel.Grid(frames=3, height=7, width=14)
HEADS, HEAD_DIM = 2, 128


def test_one_token_plus_log_n_equals_n_identical_members():
    g = torch.Generator().manual_seed(0)
    d = HEADS * 16
    q = torch.randn(1, 5, d, generator=g, dtype=torch.float64)
    base_k, base_v = torch.randn(1, 7, d, generator=g, dtype=torch.float64), torch.randn(1, 7, d, generator=g, dtype=torch.float64)
    mk, mv = torch.randn(1, 1, d, generator=g, dtype=torch.float64), torch.randn(1, 1, d, generator=g, dtype=torch.float64)
    for n in (1, 2, 5, 11):
        full = flash_attention(q, torch.cat([base_k, mk.repeat(1, n, 1)], 1), torch.cat([base_v, mv.repeat(1, n, 1)], 1), HEADS)
        mask = torch.zeros(1, 1, 1, 8, dtype=torch.float64)
        mask[..., 7] = math.log(n)
        one = flash_attention(q, torch.cat([base_k, mk], 1), torch.cat([base_v, mv], 1), HEADS, ctx_mask=mask)
        assert torch.allclose(full, one, atol=1e-10, rtol=0)


def test_fractional_rope_equals_the_table_at_integer_positions():
    ff = precompute_freqs_cis_3d(HEAD_DIM)
    f, h, w = GRID.frames, GRID.height, GRID.width
    table = torch.cat([ff[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), ff[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                       ff[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)
    pos = torch.stack(torch.meshgrid(torch.arange(f), torch.arange(h), torch.arange(w), indexing="ij"), -1).reshape(-1, 3).double()
    got = fv.rope_freqs_at(pos, HEAD_DIM)
    assert got.dtype == table.dtype and torch.equal(got, table)
    half = fv.rope_freqs_at(torch.tensor([[1.0, 2.5, 3.25]]), HEAD_DIM)
    assert not torch.equal(half, table[98 + 2 * 14 + 3: 98 + 2 * 14 + 4])
    assert torch.allclose(half.abs(), torch.ones_like(half.abs()))


def test_interlace_slices():
    g = torch.Generator().manual_seed(1)
    for n_per in (173, 114):
        periphery = torch.sort(torch.randperm(196, generator=g)[:n_per]).values
        av = torch.rand(196, generator=g)
        slices, cut = fv.interlace_slices(periphery, av)
        assert len(slices) == 9 and all(s.numel() <= 16 for s in slices)
        got = torch.cat(slices + [cut]).tolist()
        assert sorted(got) == periphery.tolist() and len(set(got)) == len(got)  # every token in exactly one slice or cut
        assert cut.numel() == sum(max(0, periphery[i::9].numel() - 16) for i in range(9))
    assert fv.interlace_slices(torch.arange(114), torch.rand(196))[1].numel() == 0  # keep 50 %: 114 fit 9 x 16


def _scores(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(196, generator=g), torch.rand(196, generator=g), torch.rand(196, generator=g), torch.randn(196, 192, generator=g)


@pytest.mark.parametrize("method", fv.METHODS)
def test_plans(method):
    av, diff, recv, x0 = _scores(2)
    gen = torch.Generator().manual_seed(3)
    p = fv.make_plan(method, 39, GRID, av, diff, recv, x0, gen)
    sinks = set(torch.topk(recv, 3).indices.tolist())
    assert set(p.sinks.tolist()) == sinks
    if method == "sink":
        assert p.fixed.numel() == 39 and sinks <= set(p.fixed.tolist())
        return
    if method == "interlace_ht":
        assert int(p.freq.sum()) == 9 * 39 and sinks <= set(p.fixed.tolist()) and p.truncated.numel() == 0
        assert all(p.real(s).numel() == 39 for s in range(1, 10))
    else:
        assert p.core.numel() == 20 and not (set(p.core.tolist()) & sinks)
    if method not in ("interlace_wfar", "interlace_ht"):  # the far-frame core skips near-frame A->V leaders; HT has no core
        assert set(torch.topk(torch.where(torch.tensor([j in sinks for j in range(196)]), -1.0, av), 10).indices.tolist()) <= set(p.core.tolist())
    if method == "interlace_ht":
        pass
    elif method == "rep":
        assert p.periphery.numel() == 173
        assert len(p.bias_real) <= 16 and p.fixed.numel() == 23 + len(p.bias_real)
    elif method.startswith("fov") or method == "merge":
        assert p.periphery.numel() == 173
        members = torch.cat([g for g, _ in p.regions])
        assert sorted(members.tolist()) == p.periphery.tolist() and len(p.regions) <= 16
        assert p.fixed.numel() == 23 and all(abs(lg - math.log(g.numel())) < 1e-12 for g, lg in p.regions)
        frames = p.region_pos[:, 0]
        assert torch.equal(frames, frames.round())  # one future frame per region
    else:
        assert p.periphery.numel() == 173
        assert p.fixed.numel() == 23 and all(p.real(s).numel() <= 39 for s in range(1, 10))
        assert not p.slice_bias or method in ("interlace_w", "interlace_wfar")  # cal / fill: no count bias in the plan
    off = fv.make_plan(method, 39, GRID, av, diff, recv, x0, gen, off=True)
    assert torch.equal(off.fixed, torch.sort(sel.topk_indices(av, 39)).values) and not off.regions and not off.bias_real
    assert off.freq is None and not off.slices
    full = fv.make_plan(method, 196, GRID, av, diff, recv, x0, gen)
    assert torch.equal(full.fixed, torch.arange(196)) and not full.regions and not full.bias_real and not full.slices
    assert full.bias_at(1) == {}


def test_fill_map():
    computed = torch.tensor([0, 9, 15, 50, 120, 140])  # frame 1: (0,0) agentview, (0,9) wrist, (1,1), (3,8); frame 2: (1,8), (3,0)
    src, outside = fv.fill_map(computed, GRID)
    assert src.shape == (196,) and torch.equal(src[computed], torch.arange(6))  # a computed token maps to itself
    assert int(src[1]) == 0 and int(src[14]) == 0 and int(src[2]) == 2  # (0,1) and (1,0): (0,0) closer; (0,2): (1,1)
    assert int(src[16]) == 2  # (1,2): (1,1) at distance 1
    assert int(src[13]) == 1  # (0,13): wrist camera of frame 1, (0,9) only
    assert int(src[98 + 3]) == 5  # frame 2 agentview (0,3): (3,0) (frame 2, d 18) beats frame 1's (1,1) (other frame)
    assert int(src[98 + 13]) == 4 and outside == 0  # frame 2 wrist (0,13): (1,8) in its own block
    tie = fv.fill_map(torch.tensor([1, 15]), GRID)[0]  # (0,1) and (1,1): (0,0) is nearer (0,1), (1,0) nearer (1,1)
    assert int(tie[0]) == 0 and int(tie[14]) == 1
    sparse_src, sparse_out = fv.fill_map(torch.tensor([3]), GRID)  # one agentview token of frame 1: everything else outside
    assert bool((sparse_src == 0).all()) and sparse_out == 147  # 3 blocks of 49


def test_ht_frequencies_and_schedule():
    for seed, k in ((11, 39), (12, 98), (13, 39)):
        av, diff, recv, _ = _scores(seed)
        if seed == 13:
            av = av ** 8  # peaked
        sinks = torch.sort(torch.topk(recv, 3).indices).values
        f = fv.ht_frequencies(av, sinks, k)
        assert int(f.sum()) == 9 * k and int(f.min()) >= 1 and int(f.max()) <= 9 and bool((f[sinks] == 9).all())
        rest = torch.tensor([j for j in range(196) if j not in set(sinks.tolist())])
        order = torch.argsort(av[rest], descending=True)
        assert bool((f[rest][order][1:] <= f[rest][order][:-1] + 1).all())  # monotone in A->V up to the rounding
        steps = fv.ht_schedule(f, av)
        assert len(steps) == 9 and all(x.numel() == k for x in steps)
        cnt = torch.zeros(196, dtype=torch.long)
        for x in steps:
            assert x.unique().numel() == x.numel()
            cnt[x] += 1
        assert torch.equal(cnt, f)
        p = fv.make_plan("interlace_ht", k, GRID, av, diff, recv)
        assert torch.equal(p.freq, f) and all(torch.equal(p.real(s), steps[s - 1]) for s in range(1, 10))
        for s in range(1, 10):
            b = p.bias_at(s)
            assert set(b) == set(p.slices[s - 1].tolist()) and not (set(b) & set(p.fixed.tolist()))
            assert all(abs(v - math.log(9 / int(f[j]))) < 1e-12 and v > 0 for j, v in b.items())
    with pytest.raises(ValueError):
        fv.ht_frequencies(torch.rand(196), torch.tensor([0, 1, 2]), 20)  # 180 slots cannot cover 196 tokens once


def test_ht_weight_is_unbiased_over_the_steps():
    """Horvitz-Thompson: the mean over the 9 steps of sum_j exp(l_j) e^{b_j} over that step's tokens is sum_j exp(l_j)."""
    av, diff, recv, _ = _scores(14)
    p = fv.make_plan("interlace_ht", 39, GRID, av, diff, recv)
    w = torch.rand(196, generator=torch.Generator().manual_seed(15), dtype=torch.float64)
    est = []
    for s in range(1, 10):
        b = p.bias_at(s)
        est.append(sum(float(w[j]) * math.exp(b.get(int(j), 0.0)) for j in p.real(s).tolist()))
    assert abs(sum(est) / 9 - float(w.sum())) < 1e-9


def test_merge_singleton_plan():
    av, diff, recv, x0 = _scores(16)
    p = fv.make_plan("merge", 39, GRID, av, diff, recv, singleton=True)
    assert p.fixed.numel() == 0 and len(p.regions) == 196 and all(lg == 0.0 for _, lg in p.regions)
    assert [int(g) for g, _ in p.regions] == list(range(196))
    ff = precompute_freqs_cis_3d(HEAD_DIM)
    f, h, w = GRID.frames, GRID.height, GRID.width
    table = torch.cat([ff[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), ff[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                       ff[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)
    assert torch.equal(fv.rope_freqs_at(p.region_pos, HEAD_DIM), table[98:])  # the region RoPE rows are the table's


def test_grid_regions_and_kmeans():
    r = fv.grid_regions(GRID)
    assert r.unique().numel() == 16 and int(r[0]) == 0 and int(r[13]) == 5 and int(r[98 + 6 * 14 + 13]) == 15
    x = torch.randn(40, 8, generator=torch.Generator().manual_seed(4))
    a = fv.kmeans(x, 4, torch.Generator().manual_seed(5))
    b = fv.kmeans(x, 4, torch.Generator().manual_seed(5))
    assert torch.equal(a, b) and a.unique().numel() <= 4


def test_config():
    cfg = parse_config({"method": "prune", "prune": {"selection": "av", "keep_ratio": 0.2, "fovea": "fov_grid"}})
    assert cfg.prune.fovea == "fov_grid"
    for bad in ({"fovea": "bogus"}, {"fovea_off": True}, {"fovea": "rep", "action_full_kv": True}, {"fovea": "rep", "selection": "diff"}):
        with pytest.raises(ValueError):
            parse_config({"method": "prune", "prune": {"selection": "av", "keep_ratio": 0.2, **bad}})
    with pytest.raises(ValueError):
        parse_config({"method": "prune", "record_future_attn": True, "prune": {"keep_ratio": 0.2, "fovea": "rep"}})
    assert parse_config({"method": "prune", "record_future_attn": True, "prune": {"keep_ratio": 0.2, "fovea": "sink"}})


def test_g2_n_of_N_with_log_N_over_n_equals_N_identical_members():
    g = torch.Generator().manual_seed(7)
    d = HEADS * 16
    q = torch.randn(1, 5, d, generator=g, dtype=torch.float64)
    base_k, base_v = torch.randn(1, 7, d, generator=g, dtype=torch.float64), torch.randn(1, 7, d, generator=g, dtype=torch.float64)
    mk, mv = torch.randn(1, 1, d, generator=g, dtype=torch.float64), torch.randn(1, 1, d, generator=g, dtype=torch.float64)
    for big, n in ((173, 16), (114, 13), (19, 2), (5, 5)):
        full = flash_attention(q, torch.cat([base_k, mk.repeat(1, big, 1)], 1), torch.cat([base_v, mv.repeat(1, big, 1)], 1), HEADS)
        mask = torch.zeros(1, 1, 1, 7 + n, dtype=torch.float64)
        mask[..., 7:] = math.log(big / n)
        part = flash_attention(q, torch.cat([base_k, mk.repeat(1, n, 1)], 1), torch.cat([base_v, mv.repeat(1, n, 1)], 1), HEADS,
                               ctx_mask=mask)
        assert torch.allclose(full, part, atol=1e-10, rtol=0)


def test_weighted_and_far_plans():
    av, diff, recv, x0 = _scores(5)
    base = fv.make_plan("interlace", 39, GRID, av, diff, recv)
    w = fv.make_plan("interlace_w", 39, GRID, av, diff, recv)
    assert torch.equal(w.fixed, base.fixed) and all(torch.equal(a, b) for a, b in zip(w.slices, base.slices))
    assert w.periphery.numel() == 173 and len(w.slice_bias) == 9
    assert all(abs(b - math.log(173 / x.numel())) < 1e-12 for b, x in zip(w.slice_bias, w.slices))
    for s in range(1, 10):
        bias = w.bias_at(s)
        assert set(bias) == set(w.slices[(s - 1) % 9].tolist()) and not (set(bias) & set(w.fixed.tolist()))
    assert base.bias_at(3) == {} and not base.slice_bias
    z = fv.make_plan("interlace_w", 39, GRID, av, diff, recv, wzero=True)
    assert not z.slice_bias and z.bias_at(2) == {} and torch.equal(z.fixed, base.fixed)
    far = fv.make_plan("interlace_wfar", 39, GRID, av, diff, recv)
    assert far.core.numel() == 20 and bool((far.core >= GRID.per_frame).all())
    near = [j for j in range(GRID.per_frame) if j not in set(far.sinks.tolist())]
    assert set(near) <= set(far.periphery.tolist()) and len(far.slice_bias) == 9
    nofar = fv.make_plan("interlace_wfar", 39, GRID, av, diff, recv, nofar=True)
    assert torch.equal(nofar.fixed, w.fixed) and nofar.slice_bias == w.slice_bias
    assert all(torch.equal(a, b) for a, b in zip(nofar.slices, w.slices))
    half = fv.make_plan("interlace_wfar", 98, GRID, av, diff, recv)  # keep 50 %: 79 core tokens fit in the far frame
    assert half.core.numel() == 79 and bool((half.core >= GRID.per_frame).all()) and half.periphery.numel() == 114
    for m in ("interlace_w", "interlace_wfar"):
        full = fv.make_plan(m, 196, GRID, av, diff, recv)
        assert torch.equal(full.fixed, torch.arange(196)) and not full.slices and not full.slice_bias


def test_fovea_w_config():
    ok = parse_config({"method": "prune", "record_future_share": True,
                       "prune": {"keep_ratio": 0.2, "fovea": "interlace_wfar", "fovea_nofar": True}})
    assert ok.prune.fovea_nofar and ok.record_future_share
    assert parse_config({"method": "prune", "prune": {"keep_ratio": 0.2, "fovea": "interlace_w", "fovea_wzero": True}}).prune.fovea_wzero
    for bad in ({"fovea": "interlace", "fovea_wzero": True}, {"fovea": "interlace_w", "fovea_nofar": True}):
        with pytest.raises(ValueError):
            parse_config({"method": "prune", "prune": {"keep_ratio": 0.2, **bad}})
    assert parse_config({"method": "dense", "video_freeze": True}).video_freeze
    with pytest.raises(ValueError):
        parse_config({"method": "prune", "video_freeze": True, "prune": {"keep_ratio": 0.2}})


def test_n_methods_config():
    for m in ("interlace_cal", "interlace_fill", "merge", "interlace_ht"):
        cfg = parse_config({"method": "prune", "record_future_share": True, "record_motion": True,
                            "prune": {"keep_ratio": 0.2, "fovea": m}})
        assert cfg.prune.fovea == m and cfg.record_future_share
    assert parse_config({"method": "prune", "prune": {"keep_ratio": 0.2, "fovea": "merge", "fovea_singleton": True}}).prune.fovea_singleton
    for bad in ({"fovea": "interlace_ht", "fovea_singleton": True}, {"fovea": "merge", "fovea_singleton": True, "fovea_off": True}):
        with pytest.raises(ValueError):
            parse_config({"method": "prune", "prune": {"keep_ratio": 0.2, **bad}})
