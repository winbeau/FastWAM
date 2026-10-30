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
    assert p.core.numel() == 20 and not (set(p.core.tolist()) & sinks)
    assert set(torch.topk(torch.where(torch.tensor([j in sinks for j in range(196)]), -1.0, av), 10).indices.tolist()) <= set(p.core.tolist())
    assert p.periphery.numel() == 173
    if method == "rep":
        assert len(p.bias_real) <= 16 and p.fixed.numel() == 23 + len(p.bias_real)
    elif method.startswith("fov"):
        members = torch.cat([g for g, _ in p.regions])
        assert sorted(members.tolist()) == p.periphery.tolist() and len(p.regions) <= 16
        assert p.fixed.numel() == 23 and all(abs(lg - math.log(g.numel())) < 1e-12 for g, lg in p.regions)
        frames = p.region_pos[:, 0]
        assert torch.equal(frames, frames.round())  # one future frame per region
    else:
        assert p.fixed.numel() == 23 and all(p.real(s).numel() <= 39 for s in range(1, 10))
    off = fv.make_plan(method, 39, GRID, av, diff, recv, x0, gen, off=True)
    assert torch.equal(off.fixed, torch.sort(sel.topk_indices(av, 39)).values) and not off.regions and not off.bias_real
    full = fv.make_plan(method, 196, GRID, av, diff, recv, x0, gen)
    assert torch.equal(full.fixed, torch.arange(196)) and not full.regions and not full.bias_real and not full.slices


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
