"""CPU checks of the late-read round (``prune.late_read`` / ``late_read_last`` / ``late_record_z``, fovea.late_read_plan)."""

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


def _arm(**kw):
    return {**MEAS, "prune": {**MEAS["prune"], **kw}}


def _plan(seed=0):
    g = torch.Generator().manual_seed(seed)
    near = [r * 14 + c for r in range(7) for c in range(14)]
    measured = torch.sort(torch.tensor(near)[torch.randperm(98, generator=g)[:84]]).values
    av, df = torch.zeros(196), torch.zeros(196)
    av[measured] = torch.rand(84, generator=g)
    df[measured] = torch.rand(84, generator=g)
    sinks = torch.tensor([182, int(measured[3]), int(measured[60])])
    return fv.c2f_plan(measured, av, df, sinks, GRID, 39, core_all=False, last_extra=10)


def test_late_read_config():
    for mode in ("lr", "jump"):
        for last in ("plan", "near", "all"):
            p = parse_config(_arm(late_read=mode, late_read_last=last)).prune
            assert p.late_read == mode and p.late_read_last == last
    base = parse_config(MEAS).prune
    assert base.late_read is None and base.late_read_last == "plan" and not base.late_record_z
    assert parse_config(_arm(late_record_z=True)).prune.late_record_z
    m4w = {"method": "prune", "prune": {"keep_ratio": 0.2, "warmup_steps": 1, "fovea": "interlace_w", "late_read": "lr"}}
    for bad in (m4w, _arm(late_read="xx"), _arm(late_read_last="near"), _arm(late_read="lr", late_read_last="far"),
                _arm(late_read="lr", role_select="rs", role_tips=[92, 96]), _arm(late_read="jump", step_delta={9: 20}),
                {"method": "prune", "prune": {"keep_ratio": 0.2, "warmup_steps": 1, "fovea": "interlace_w", "late_record_z": True}}):
        with pytest.raises(ValueError):
            parse_config(bad)


@pytest.mark.parametrize("last", ("plan", "near", "all"))
def test_late_read_plan(last):
    plan = _plan()
    p = fv.late_read_plan(plan, last, GRID)
    for s in range(2, 9):
        assert p.real(s).numel() == 0 and p.bias_at(s) == {}
    rows, bias = p.real(9), p.bias_at(9)
    if last == "plan":
        assert torch.equal(rows, plan.real(9)) and rows.numel() == 49
        assert bias == pytest.approx(plan.bias_at(9)) and set(bias) <= set(rows.tolist())
    elif last == "near":
        assert torch.equal(rows, torch.arange(98)) and all(v == pytest.approx(math.log(2.0)) for v in bias.values()) and len(bias) == 98
    else:
        assert torch.equal(rows, torch.arange(196)) and bias == {}
    moved = p.to("cpu")
    assert torch.equal(moved.real(9), rows) and moved.bias_at(9) == bias and moved.fixed.numel() == 0
    with pytest.raises(ValueError):
        fv.late_read_plan(plan, "far", GRID)
