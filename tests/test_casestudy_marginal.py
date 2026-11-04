"""CPU checks of the step-marginal round (``prune.step_delta``: fovea.stratified_pick / lowest_av / apply_step_delta, Plan.drop)."""

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


def _plan():
    av = torch.rand(196, generator=torch.Generator().manual_seed(7))
    df = torch.rand(196, generator=torch.Generator().manual_seed(8))
    return fv.c2f_plan(torch.arange(84), av, df, torch.tensor([182, 3, 40]), GRID, 39, last_extra=10), av


def _stratum(j):
    return int(j) // 98 * 2 + (int(j) % 98) % 14 // 7


def test_step_delta_config():
    cfg = parse_config({**MEAS, "prune": {**MEAS["prune"], "step_delta": {5: 20, 2: -20}, "step_delta_seed": 3}})
    assert cfg.prune.step_delta == ((2, -20), (5, 20)) and cfg.prune.step_delta_seed == 3
    assert parse_config(MEAS).prune.step_delta == ()
    m4w = {"method": "prune", "prune": {"keep_ratio": 0.2, "warmup_steps": 1, "fovea": "interlace_w", "step_delta": {5: 20}}}
    for bad in (m4w, {**MEAS, "prune": {**MEAS["prune"], "step_delta": {0: 20}}},
                {**MEAS, "prune": {**MEAS["prune"], "step_delta": {10: 20}}}):
        with pytest.raises(ValueError):
            parse_config(bad)


def test_stratified_pick():
    pool = torch.arange(84, 196)  # step 1's rest: 14 near-frame tokens, the whole far frame
    a = fv.stratified_pick(pool, 20, GRID, 0)
    assert torch.equal(a, fv.stratified_pick(pool, 20, GRID, 0)) and a.numel() == 20 and bool(torch.isin(a, pool).all())
    assert torch.equal(a, torch.sort(a).values) and a.unique().numel() == 20
    sizes = torch.bincount(torch.tensor([_stratum(j) for j in pool]), minlength=4).float()
    got = torch.bincount(torch.tensor([_stratum(j) for j in a]), minlength=4).float()
    assert bool(((got - sizes * 20 / pool.numel()).abs() < 1).all())  # proportional allocation (largest remainder)
    assert not torch.equal(a, fv.stratified_pick(pool, 20, GRID, 1))
    assert fv.stratified_pick(pool[:5], 20, GRID, 0).numel() == 5 and fv.stratified_pick(pool, 0, GRID, 0).numel() == 0


def test_lowest_av_keeps_sinks():
    av = torch.arange(196, dtype=torch.float32)
    got = fv.lowest_av(torch.tensor([0, 1, 2, 50, 60, 182]), av, torch.tensor([0, 182]), 3)
    assert got.tolist() == [1, 2, 50]


def test_apply_step_delta():
    base, av = _plan()
    same = fv.apply_step_delta(base, ((5, 0), (1, 20)), av, GRID, 0)  # n = 0 and step 1: no change here
    assert all(torch.equal(same.real(s), base.real(s)) and same.bias_at(s) == base.bias_at(s) for s in range(2, 10))
    add = fv.apply_step_delta(base, ((5, 20),), av, GRID, 0)
    i5 = (5 - 1) % 8
    assert add.real(5).numel() == base.real(5).numel() + 20 and bool(torch.isin(base.real(5), add.real(5)).all())
    assert abs(add.slice_bias[i5] - math.log(173 / add.slices[i5].numel())) < 1e-12
    assert all(torch.equal(add.real(s), base.real(s)) and add.bias_at(s) == base.bias_at(s) for s in range(2, 10) if s != 5)
    cut = fv.apply_step_delta(base, ((3, -20),), av, GRID, 0)
    r3 = cut.real(3)
    gone = base.real(3)[~torch.isin(base.real(3), r3)]
    assert r3.numel() == base.real(3).numel() - 20 and bool(torch.isin(base.sinks, r3).all())
    kept_nonsink = r3[~torch.isin(r3, base.sinks)]
    assert float(av[gone].max()) <= float(av[kept_nonsink].min())  # the lowest A->V left
    assert 3 in cut.drop and all(torch.equal(cut.real(s), base.real(s)) for s in range(2, 10) if s != 3)
    moved = cut.to("cpu")
    assert torch.equal(moved.real(3), r3)
