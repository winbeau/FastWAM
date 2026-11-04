"""CPU checks of the step0-sparse round (``fovea.py`` probes / interpolation / slicing / tube, the runner's draft layout)."""

import math
from types import SimpleNamespace

import pytest
import torch
from einops import rearrange

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config
from fastwam.casestudy.joint_runner import JointCaseStudyRunner

GRID = sel.Grid(frames=3, height=7, width=14)


def _scores(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(196, generator=g), torch.rand(196, generator=g), torch.rand(196, generator=g)


def test_probe_index():
    p = fv.probe_index(GRID)
    assert p.numel() == 39 and torch.equal(p, torch.sort(p).values) and len(set(p.tolist())) == 39
    assert {182, 7, 105} <= set(p.tolist())
    for f in range(2):
        for cam in range(2):
            block = [f * 98 + r * 14 + cam * 7 + c for r in (1, 3, 5) for c in (1, 3, 5)]
            assert set(block) <= set(p.tolist())


def test_interp_exact_at_probes_and_identity():
    g = torch.Generator().manual_seed(1)
    vals = torch.randn(196, 5, generator=g)
    p = fv.probe_index(GRID)
    out = fv.interp_tokens(vals, GRID, p)
    assert torch.equal(out[p], vals[p])
    assert torch.equal(fv.interp_tokens(vals, GRID, torch.arange(196)), vals)
    # A field linear in (row, column) inside the probe span is reproduced; outside it the nearest probe row / column.
    lin = torch.zeros(196, 1)
    for j in range(196):
        r, c = (j % 98) // 14, (j % 98) % 14 % 7
        lin[j, 0] = 2.0 * min(max(r, 1), 5) - 3.0 * min(max(c, 1), 5) + (j // 98) * 10 + ((j % 98) % 14 >= 7) * 100
    out = fv.interp_tokens(lin, GRID, torch.tensor([j for j in p.tolist() if j not in (182, 7, 105)]))
    assert torch.allclose(out, lin, atol=1e-5)


def test_step0_slices():
    av, diff, recv = _scores(4)
    p = fv.probe_index(GRID)
    sinks = torch.tensor([182, 15, 45])
    plan = fv.make_plan("interlace_w", 39, GRID, av, diff, recv, sinks=sinks, probes=p)
    assert torch.equal(plan.sinks, torch.sort(sinks).values) and plan.core.numel() == 20 and plan.periphery.numel() == 173
    assert len(plan.slices) == 9 and all(x.numel() <= 16 for x in plan.slices)
    parts = torch.cat(plan.slices + [plan.truncated])
    assert torch.equal(torch.sort(parts).values, plan.periphery)
    left = [j for j in plan.periphery.tolist() if j in set(p.tolist())]
    order = sorted(left, key=lambda j: (-float(av[j]), j))
    assert set(plan.slices[8].tolist()) == set(order[:16]) and set(plan.slices[7].tolist()) == set(order[16:32])
    assert set(order[32:]) <= set(plan.slices[6].tolist())
    assert not (set(torch.cat(plan.slices[:6]).tolist()) & set(left))
    assert all(abs(b - math.log(173 / x.numel())) < 1e-12 for b, x in zip(plan.slice_bias, plan.slices))
    # Without probes: M4w's slicing; the given sinks equal the top 3 of recv when they are the top 3.
    top = torch.topk(recv, 3).indices
    a = fv.make_plan("interlace_w", 39, GRID, av, diff, recv)
    b = fv.make_plan("interlace_w", 39, GRID, av, diff, recv, sinks=top)
    assert torch.equal(a.fixed, b.fixed) and all(torch.equal(x, y) for x, y in zip(a.slices, b.slices))
    assert a.slice_bias == b.slice_bias


def test_tube_mask():
    m = fv.tube_mask({1: [(0, 0)], 2: [(3, 6)]}, GRID)
    idx = set(torch.nonzero(m).flatten().tolist())
    assert idx == {0, 1, 14, 15} | {98 + r * 14 + c for r in (2, 3, 4) for c in (5, 6)}
    w = fv.tube_mask({1: []}, GRID, wrist_rows=True)
    assert set(torch.nonzero(w).flatten().tolist()) == {f * 98 + r * 14 + c for f in (0, 1) for r in (5, 6) for c in range(7, 14)}


def test_probe_draft_layout():
    """The runner's draft: with every token a probe the draft velocity is the velocity itself, and the clean estimate is
    x - sigma v in the Diff layout; with the 39 probes the probes keep their own velocity."""
    pt, ph, pw = 1, 2, 2
    unpatch = lambda x, gs: rearrange(x, "b (f h w) (x y z c) -> b c (f x) (h y) (w z)", f=gs[0], h=gs[1], w=gs[2],  # noqa: E731
                                      x=pt, y=ph, z=pw)
    fake = SimpleNamespace(video=SimpleNamespace(unpatchify=unpatch))
    g = torch.Generator().manual_seed(2)
    lat = torch.randn(1, 4, 3, 14, 28, generator=g, dtype=torch.float64).float()
    vel = torch.randn(1, 3 * 98, 16, generator=g)
    vel[:, :98] = 0
    sigma = torch.tensor(0.75)
    gs = (3, 7, 14)
    v, zf = JointCaseStudyRunner._probe_draft(fake, lat, vel, gs, GRID, torch.arange(196), sigma, (pt, ph, pw))
    assert torch.allclose(v[:, 98:], vel[:, 98:], atol=1e-5)
    x0 = lat - sigma * unpatch(vel, gs)
    z = rearrange(x0[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
    assert torch.allclose(zf, z[1:].reshape(196, -1), atol=1e-6)
    p = fv.probe_index(GRID)
    v, _ = JointCaseStudyRunner._probe_draft(fake, lat, vel, gs, GRID, p, sigma, (pt, ph, pw))
    assert torch.allclose(v[:, 98 + p], vel[:, 98 + p], atol=1e-5)


def test_step0_config():
    base = {"method": "prune", "score_layer": 18, "record_future_share": True,
            "prune": {"keep_ratio": 0.2, "fovea": "interlace_w"}}
    for extra in ({"step0": "b"}, {"step0": "d"}, {"step0": "b", "step0_nobias": True}, {"step0": "b", "step0_all": True},
                  {"e9": "early"}, {"e9": "late"}):
        cfg = parse_config({**base, "prune": {**base["prune"], **extra}})
        assert all(getattr(cfg.prune, k) == v for k, v in extra.items())
    assert parse_config({"method": "dense", "record_step0_x0": True}).record_step0_x0
    bad = [({"step0": "b"}, {"score_layer": None}), ({"step0": "c"}, {}), ({"step0": "b", "fovea": "interlace"}, {}),
           ({"step0_all": True}, {}), ({"step0": "d", "step0_all": True}, {}), ({"e9": "early", "step0": "b"}, {}),
           ({"e9": "mid"}, {})]
    for prune, top in bad:
        with pytest.raises(ValueError):
            parse_config({**base, **top, "prune": {**base["prune"], **prune}})
