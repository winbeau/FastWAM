"""CPU checks of the cam-split round (``prune.cs_order``: fovea.cs_rows / cs_bias / step_normalized / camera_tokens, the near-frame
drafts of a camera anchored on its whole near frame)."""

import math
from types import SimpleNamespace

import pytest
import torch

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config
from fastwam.casestudy.joint_runner import JointCaseStudyRunner as J

GRID = sel.Grid(frames=3, height=7, width=14)
MEAS = {"method": "prune", "score_layer": 18,
        "prune": {"selection": "av", "warmup_steps": 1, "keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b", "step0_draft": "near",
                  "late": "f1s84", "step0_gain": 1.25, "late_extra": 10, "late_plan_core": "measured"}}


def _near(cam):
    return [r * 14 + cam * 7 + c for r in range(7) for c in range(7)]


def test_cs_config():
    cfg = parse_config({**MEAS, "prune": {**MEAS["prune"], "cs_order": "wa", "cs_n": 42, "cs_k": 12, "cs_bias": 196,
                                          "cs_gain_w": 1.05, "cs_gain_a": 1.45}})
    p = cfg.prune
    assert (p.cs_order, p.cs_n, p.cs_k, p.cs_bias, p.cs_gain_w, p.cs_gain_a) == ("wa", 42, 12, "196", 1.05, 1.45)
    assert parse_config(MEAS).prune.cs_order is None and parse_config(MEAS).prune.cs_bias == "98"
    m4w = {"method": "prune", "prune": {"keep_ratio": 0.2, "warmup_steps": 1, "fovea": "interlace_w", "cs_order": "wa"}}
    for bad in (m4w, {**MEAS, "prune": {**MEAS["prune"], "cs_order": "xx"}}, {**MEAS, "prune": {**MEAS["prune"], "cs_order": "wa", "cs_n": 40}},
                {**MEAS, "prune": {**MEAS["prune"], "cs_order": "wa", "cs_bias": 50}},
                {**MEAS, "prune": {**MEAS["prune"], "cs_order": "wa", "role_select": "rs", "role_tips": [78, 82]}}):
        with pytest.raises(ValueError):
            parse_config(bad)


def test_cs_rows_and_bias():
    late = torch.sort(torch.cat([torch.tensor(_near(0))[torch.randperm(49, generator=torch.Generator().manual_seed(1))[:42]],
                                 torch.tensor(_near(1))[torch.randperm(49, generator=torch.Generator().manual_seed(2))[:42]]])).values
    r = fv.cs_rows("wa", 49, 0, late, GRID, 0)
    assert r[1][0].tolist() == _near(1) and r[2][0].tolist() == _near(0) and r[1][1].numel() == 0 == r[2][1].numel()
    r = fv.cs_rows("aw", 42, 12, late, GRID, 0)
    assert set(r[1][0].tolist()) == set(late.tolist()) & set(_near(0)) and set(r[2][0].tolist()) == set(late.tolist()) & set(_near(1))
    assert set(r[1][1].tolist()) <= set(_near(1)) and set(r[2][1].tolist()) <= set(_near(0)) and r[1][1].numel() == 12
    assert fv.role_strata(r[1][1], GRID).unique().numel() == 4  # the context spreads over the quadrants
    assert abs(fv.cs_bias("98", 42) - math.log(196 / 84)) < 1e-12 and fv.cs_bias("0", 49) == 0.0
    assert abs(fv.cs_bias("196", 49) - math.log(4.0)) < 1e-12
    assert fv.camera_tokens(1, GRID).numel() == 98 and set(fv.camera_tokens(1, GRID, near_only=True).tolist()) == set(_near(1))


def test_step_normalized():
    v = torch.arange(196, dtype=torch.float32) + 1
    a, b = torch.tensor(_near(1)), torch.tensor(_near(0))
    out = fv.step_normalized(v, {1: a, 2: b})
    assert abs(float(out[a].mean()) - 1) < 1e-6 and abs(float(out[b].mean()) - 1) < 1e-6
    assert float(out[98:].abs().sum()) == 0.0
    assert torch.equal(torch.argsort(out[a]), torch.argsort(v[a]))  # the order inside a step is kept


def test_whole_near_frame_anchor_drafts_are_measured():
    g = torch.Generator().manual_seed(5)
    values, fallback = torch.randn(196, 8, generator=g), torch.randn(196, 8, generator=g)
    rows = torch.tensor(_near(1))
    out = J._spread(SimpleNamespace(), values, rows, GRID, "near", fallback, 0.5)
    assert torch.equal(out[rows], values[rows])  # n = 49: every near draft of the camera is its measured estimate
    far = torch.tensor(_near(1)) + 98
    assert torch.allclose(out[far], values[rows] + 0.5 * (values[rows] - fallback[rows]))
