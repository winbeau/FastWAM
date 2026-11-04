"""CPU checks of the step0-sparse-r2 round (``fovea.py`` neighbours / memory / 10 slices / reselection, config)."""

import math

import pytest
import torch

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config

GRID = sel.Grid(frames=3, height=7, width=14)


def _scores(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(196, generator=g), torch.rand(196, generator=g), torch.rand(196, generator=g)


def test_nearest_sources_and_idw():
    src = torch.tensor([0, 2, 16, 98 + 30, 7])  # block (1, ag): (0,0), (0,2), (1,2); block (2, ag): (2,2); block (1, wr): (0,0)
    nbr, w, has = fv.nearest_sources(src, GRID)
    assert has[0] and int(nbr[0, 0]) == 0 and float(w[0, 0]) == 1.0 and float(w[0, 1:].sum()) == 0.0  # a source: itself
    # token 1 = (0, 1): sources (0,0) d2 1, (0,2) d2 1, (1,2) d2 2 -> weights 1, 1, 1/2 normalised
    assert sorted(int(src[p]) for p, x in zip(nbr[1].tolist(), w[1].tolist()) if x > 0) == [0, 2, 16]
    got = {int(src[p]): x for p, x in zip(nbr[1].tolist(), w[1].tolist()) if x > 0}
    assert abs(got[0] - 0.4) < 1e-12 and abs(got[2] - 0.4) < 1e-12 and abs(got[16] - 0.2) < 1e-12
    assert has[98] and not has[98 + 7]  # (2, ag) has 128; (2, wr) has no source
    vals = torch.randn(196, 3, dtype=torch.float64)
    out, upd = fv.idw_fill(vals, src, GRID)
    assert torch.equal(out[src], vals[src].float())  # exact at the sources
    assert torch.allclose(out[1], (0.4 * vals[0] + 0.4 * vals[2] + 0.2 * vals[16]).float(), atol=1e-6)
    assert torch.equal(out[98 + 7], vals[98 + 7].float()) and 98 + 7 not in set(upd.tolist())  # no source in the block
    out2, upd2 = fv.idw_fill(vals, src, GRID, targets=torch.tensor([1, 3]))
    assert set(upd2.tolist()) == {1, 3} and torch.equal(out2[5], vals[5].float())
    # at most 4 neighbours, ties to the smaller index
    many = torch.tensor([14, 28, 1, 15, 29])  # around (1, 0): (1,0) itself is not a source
    nbr, w, _ = fv.nearest_sources(many, GRID)
    assert sorted(int(many[p]) for p, x in zip(nbr[0].tolist(), w[0].tolist()) if x > 0) == [1, 14, 15, 28]


def test_align_frames_and_memory_map():
    z = torch.arange(196, dtype=torch.float32)[:, None].repeat(1, 2)
    a = fv.align_frames(z, GRID)
    assert torch.equal(a[:98], z[98:]) and torch.equal(a[98:], z[98:] + (z[98:] - z[:98]))
    assert float(a[5, 0]) == 103.0 and float(a[98 + 5, 0]) == 103.0 + 98.0
    comp = torch.tensor([3, 50, 98 + 3])
    m = fv.memory_map(z, comp, GRID)
    filled, _ = fv.idw_fill(z, comp, GRID)
    assert torch.equal(m, fv.align_frames(filled, GRID))
    assert torch.equal(fv.memory_map(z, comp, GRID, direct=True), a)


def test_ten_slices_from_step0():
    av, diff, recv = _scores(6)
    sinks = torch.tensor([182, 20, 60])
    p = fv.make_plan("interlace_w", 39, GRID, av, diff, recv, sinks=sinks, n_slices=10, from_step0=True)
    assert p.periphery.numel() == 173 and len(p.slices) == 10 and all(x.numel() == 16 for x in p.slices)
    assert p.truncated.numel() == 173 - 160
    assert torch.equal(torch.sort(torch.cat(p.slices + [p.truncated])).values, p.periphery)
    raster, _ = fv.interlace_slices(p.periphery, av, n=10)
    for s in range(10):  # step s computes raster slice s
        assert torch.equal(p.real(s), torch.sort(torch.cat([p.fixed, raster[s]])).values)
        assert set(p.bias_at(s)) == set(raster[s].tolist())
    assert all(abs(b - math.log(173 / 16)) < 1e-12 for b in p.slice_bias)
    # the default (9 slices, no rotation) is unchanged
    q = fv.make_plan("interlace_w", 39, GRID, av, diff, recv, sinks=sinks)
    r = fv.make_plan("interlace_w", 39, GRID, av, diff, recv, sinks=sinks, n_slices=9, from_step0=False)
    assert all(torch.equal(x, y) for x, y in zip(q.slices, r.slices)) and len(q.slices) == 9


def test_reselect():
    av, diff, recv = _scores(8)
    sinks = torch.tensor([182, 20, 60])
    p = fv.make_plan("interlace_w", 39, GRID, av, diff, recv, sinks=sinks, n_slices=10, from_step0=True)
    same = fv.reselect(p, av, diff)
    assert torch.equal(same.core, p.core) and all(torch.equal(x, y) for x, y in zip(same.slices, p.slices))
    assert same.slice_bias == p.slice_bias
    diff2 = torch.rand(196, generator=torch.Generator().manual_seed(99))
    q = fv.reselect(p, av, diff2)
    assert torch.equal(q.core, fv.select_core(av, diff2, sinks, 20)) and torch.equal(q.sinks, p.sinks)
    assert not (set(torch.cat(q.slices).tolist()) & set(q.core.tolist()))
    for x, b in zip(q.slices, q.slice_bias):
        assert (b == 0.0 and x.numel() == 0) or abs(b - math.log(173 / x.numel())) < 1e-12
    for s in range(10):  # every step: sinks + core + this step's slice, 39 rows at most
        assert p.real(s).numel() == 39 and q.real(s).numel() <= 39


def test_r2_config():
    base = {"method": "prune", "score_layer": 18, "prune": {"keep_ratio": 0.2, "fovea": "interlace_w"}}
    ok = [{"side0_m": 0, "side0_use": "plan"}, {"side0_m": 4, "side0_use": "both"}, {"draft_from": 4}, {"mem": True},
          {"mem": True, "mem_route": True, "draft_from": 6}, {"mem": True, "mem_route": True, "mem_direct": True, "reselect_step": 5}]
    for extra in ok:
        cfg = parse_config({**base, "prune": {**base["prune"], **extra}})
        assert all(getattr(cfg.prune, k) == v for k, v in extra.items())
    assert parse_config({"method": "dense", "record_r2_steps": [0, 2], "record_r3_ms": [0, 30]}).record_r3_ms == (0, 30)
    bad = [({"side0_m": 0}, {}), ({"side0_m": 0, "side0_use": "x"}, {}), ({"mem_route": True}, {}),
           ({"mem": True, "reselect_step": 5}, {}), ({"mem": True}, {"score_layer": None}),
           ({"draft_from": 4, "fovea": "interlace"}, {}), ({"draft_from": 11}, {}), ({"mem": True, "side0_m": 0, "side0_use": "plan"}, {}),
           ({"draft_from": 4, "step0": "b"}, {})]
    for prune, top in bad:
        with pytest.raises(ValueError):
            parse_config({**base, **top, "prune": {**base["prune"], **prune}})
    with pytest.raises(ValueError):
        parse_config({"method": "prune", "record_r2_steps": [0], "prune": {"keep_ratio": 0.2}})
