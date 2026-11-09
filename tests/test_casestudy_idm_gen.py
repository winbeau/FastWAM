"""CPU checks of the generality round on FastWAM-IDM (prune.late_read / late_read_last near_half, casestudy.gen, fovea.late_read_plan)."""

import math

import pytest
import torch

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config

GRID = sel.Grid(frames=3, height=7, width=14)
LATE = {"method": "prune", "num_inference_steps": 10, "record_attention": True,
        "prune": {"selection": "av", "warmup_steps": 1, "keep_ratio": 0.2, "fovea": "interlace_w", "late": "f1s84",
                  "step0_draft": "near", "step0_gain": 1.25, "late_extra": 10, "late_plan_core": "measured"}}


def _arm(**kw):
    return {**LATE, "prune": {**LATE["prune"], **kw}}


def test_gen_config():
    p = parse_config(_arm(late_read="jump", late_read_last="near_half", late_read_seed=2)).prune
    assert p.late_read == "jump" and p.late_read_last == "near_half" and p.late_read_seed == 2
    assert parse_config(LATE).prune.late_read is None
    g = parse_config({"method": "dense", "num_inference_steps": 10, "gen": {"eps": [0.05, -0.05], "sets": ["strat84", "f1s84"]}}).gen
    assert g["sets"] == ["strat84", "f1s84"]
    for bad in (_arm(late_read="xx"), _arm(late_read_last="near_half"),
                {"method": "prune", "num_inference_steps": 10, "prune": {"selection": "av", "keep_ratio": 0.2, "late_read": "jump"}},
                {"method": "dense", "num_inference_steps": 10, "gen": {"zzz": 1}},
                {**LATE, "gen": {"eps": [0.05]}}):
        with pytest.raises(ValueError):
            parse_config(bad)


def test_late_read_plan_near_half():
    plan = fv.Plan("interlace_w", torch.tensor([182]), torch.tensor([1, 2]), torch.arange(3, 196), torch.tensor([1, 2, 182]),
                   slices=[torch.tensor([10 + i]) for i in range(8)], slice_bias=[1.0] * 8)
    p = fv.late_read_plan(plan, "near_half", GRID, seed=0)
    for s in range(2, 9):
        assert p.real(s).numel() == 0 and p.bias_at(s) == {}
    rows = p.real(9)
    assert rows.numel() == 49 and bool((rows < 98).all())
    wrist = int(((rows % 14) >= 7).sum())
    assert {wrist, 49 - wrist} == {24, 25}  # stratified by camera in proportion (49 / 49)
    assert all(v == pytest.approx(math.log(196 / 49)) for v in p.bias_at(9).values()) and len(p.bias_at(9)) == 49
    assert torch.equal(fv.late_read_plan(plan, "near_half", GRID, seed=0).real(9), rows)
    assert not torch.equal(fv.late_read_plan(plan, "near_half", GRID, seed=1).real(9), rows)
    moved = p.to("cpu")
    assert moved.bias_at(9) == p.bias_at(9) and moved.bias_at(5) == {}
    assert torch.equal(fv.late_read_plan(plan, "plan", GRID).real(9), plan.real(9))
