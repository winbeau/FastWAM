"""CPU checks of the role-select round (``prune.role_select``: fovea.role_plan / spread_order / role_strata, Plan.step_bias)."""

import math

import pytest
import torch

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config

GRID = sel.Grid(frames=3, height=7, width=14)
MEAS = {"method": "prune", "score_layer": 18,
        "prune": {"selection": "av", "warmup_steps": 1, "keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b", "step0_draft": "near",
                  "late": "f1s84", "step0_gain": 1.25, "late_extra": 10, "late_plan_core": "measured"}}
TIPS = (89, 93)  # wrist near frame, row 6 (any two wrist near-frame tokens for the test)


def _near(cam):
    return [r * 14 + cam * 7 + c for r in range(7) for c in range(7)]


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    measured = torch.sort(torch.cat([torch.tensor(_near(0))[torch.randperm(49, generator=g)[:42]],
                                     torch.tensor(_near(1))[torch.randperm(49, generator=g)[:42]]])).values
    av, df = torch.zeros(196), torch.zeros(196)
    av[measured] = torch.rand(84, generator=g)
    df[measured] = torch.rand(84, generator=g)
    sinks = torch.tensor([182, int(measured[3]), int(measured[60])])
    return measured, av, df, sinks


def test_role_config():
    cfg = parse_config({**MEAS, "prune": {**MEAS["prune"], "role_select": "rs", "role_tips": [89, 93], "role_seed": 2}})
    assert cfg.prune.role_select == "rs" and cfg.prune.role_tips == (89, 93) and cfg.prune.role_seed == 2
    assert parse_config({**MEAS, "prune": {**MEAS["prune"], "role_select": "notip"}}).prune.role_tips == ()
    assert parse_config(MEAS).prune.role_select is None
    m4w = {"method": "prune", "prune": {"keep_ratio": 0.2, "warmup_steps": 1, "fovea": "interlace_w", "role_select": "rs", "role_tips": [89, 93]}}
    for bad in (m4w, {**MEAS, "prune": {**MEAS["prune"], "role_select": "rs"}},
                {**MEAS, "prune": {**MEAS["prune"], "role_select": "xx", "role_tips": [89, 93]}},
                {**MEAS, "prune": {**MEAS["prune"], "role_select": "rs", "role_tips": [89, 93], "step_delta": {5: 20}}}):
        with pytest.raises(ValueError):
            parse_config(bad)


def test_spread_order():
    pool = torch.tensor(_near(1))
    o = fv.spread_order(pool, GRID, 0)
    assert torch.equal(torch.sort(o).values, pool) and torch.equal(o, fv.spread_order(pool, GRID, 0))
    assert not torch.equal(o, pool) and not torch.equal(o, fv.spread_order(pool, GRID, 1))
    assert fv.role_strata(o[:4], GRID).unique().numel() == 4  # the first round visits every quadrant once


@pytest.mark.parametrize("variant", fv.ROLE_VARIANTS)
def test_role_plan_groups(variant):
    measured, av, df, sinks = _inputs()
    p = fv.role_plan(measured, av, df, sinks, GRID, TIPS, 39, variant=variant, extra=10, seed=0)
    a_near, w_near = set(_near(0)), set(_near(1))
    score = av if variant == "aav" else df
    cand = [j for j in measured.tolist() if j in a_near and j not in sinks.tolist()]
    ranked = sorted(cand, key=lambda j: (-float(score[j]), j))
    for s in range(2, 10):
        r = set(p.real(s).tolist())
        assert len(r) == (49 if s == 9 else 39)
        assert set(sinks.tolist()) <= r and set(ranked[:12]) <= r
        assert (set(TIPS) <= r) == (variant != "notip")
        far = [j for j in r if j >= 98 and j not in sinks.tolist()]
        assert len(far) == (0 if variant == "nofar" else 4)
        b = p.bias_at(s)
        assert not (set(b) & set(p.fixed.tolist()))  # every-step groups: no bias
        assert all(v > 0 for v in b.values())
    assert set(ranked[12:17]) <= set(p.real(9).tolist())  # the last step's extra arm tokens
    if variant == "wdiff":
        wm = [j for j in measured.tolist() if j in w_near and j not in sinks.tolist() and j not in TIPS]
        wr = sorted(wm, key=lambda j: (-float(df[j]), j))
        assert all(set(wr[:10]) <= set(p.real(s).tolist()) for s in range(2, 10)) and set(wr[10:15]) <= set(p.real(9).tolist())
    moved = p.to("cpu")
    assert moved.step_bias == p.step_bias and torch.equal(moved.real(5), p.real(5))


def test_role_plan_rotation_and_bias():
    measured, av, df, sinks = _inputs(1)
    p = fv.role_plan(measured, av, df, sinks, GRID, TIPS, 39, variant="rs", extra=10, seed=0)
    w_pool = [j for j in _near(1) if j not in sinks.tolist() and j not in TIPS]
    a_rest = [j for j in _near(0) if j not in sinks.tolist() and j not in p.core.tolist()]
    seen = set().union(*(set(p.real(s).tolist()) for s in range(2, 10)))
    assert set(w_pool) <= seen and set(a_rest) <= seen  # 8 steps x 10 (+5) >= 47, 8 x 8 >= the agentview rest
    for s in range(2, 10):
        rows = set(p.real(s).tolist())
        w = [j for j in rows if j in w_pool and j not in p.core.tolist()]
        far = [j for j in rows if j >= 98 and j not in sinks.tolist()]
        b = p.bias_at(s)
        assert len(w) == (15 if s == 9 else 10)
        assert all(abs(b[j] - math.log(len(w_pool) / len(w))) < 1e-12 for j in w)
        assert all(abs(b[j] - math.log((98 - 1) / 4)) < 1e-12 for j in far)  # the far pool: 98 tokens but sink 182
