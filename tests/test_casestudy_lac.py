"""CPU checks of the late-anchor-confirm round (``prune.late_step`` 0 = X-early, ``late_extra`` on plain interlace_w,
``fovea.add_last_extra``)."""

import math

import pytest
import torch

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config

GRID = sel.Grid(frames=3, height=7, width=14)
LATE = {"method": "prune", "score_layer": 18,
        "prune": {"keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b", "step0_draft": "near", "late": "f1s84",
                  "step0_gain": 1.25, "late_extra": 10}}
M4W = {"method": "prune", "prune": {"keep_ratio": 0.2, "warmup_steps": 1, "fovea": "interlace_w"}}


def test_late_step_config():
    assert parse_config(LATE).prune.late_step == 1  # the frozen late anchor
    assert parse_config({**LATE, "prune": {**LATE["prune"], "late_step": 0}}).prune.late_step == 0
    for bad in ({**LATE["prune"], "late_step": 2},
                {"keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b", "late_step": 0}):
        with pytest.raises(ValueError):
            parse_config({**LATE, "prune": bad})


def test_late_extra_on_m4w_config():
    cfg = parse_config({**M4W, "prune": {**M4W["prune"], "late_extra": 10, "draft_mod": "atten", "draft_mod_scale": 1.25}})
    assert cfg.prune.late_extra == 10 and cfg.prune.late is None
    for bad in ({**M4W["prune"], "fovea": "interlace", "late_extra": 10},
                {**M4W["prune"], "late_extra": 10, "fovea_wzero": True},
                {**M4W["prune"], "late_extra": 10, "step0": "b"},
                {**M4W["prune"], "late_extra": -1}):
        with pytest.raises(ValueError):
            parse_config({**M4W, "prune": bad})


def _m4w_plan():
    g = torch.Generator().manual_seed(11)
    av, df, recv = torch.rand(196, generator=g), torch.rand(196, generator=g), torch.rand(196, generator=g)
    return fv.make_plan("interlace_w", 39, GRID, av, df, recv), av


def test_add_last_extra_on_m4w():
    p, av = _m4w_plan()
    assert len(p.slices) == 9 and p.truncated.numel() == 29
    q = fv.add_last_extra(p, av, 10)
    i9 = (10 - 2) % 9  # real(9) reads slices[8]
    extra = sorted(set(q.slices[i9].tolist()) - set(p.slices[i9].tolist()))
    cut = p.truncated
    want = sorted(cut[torch.argsort(-av[cut], stable=True)][:10].tolist())
    assert extra == want and q.truncated.numel() == 19 and not set(extra) & set(q.truncated.tolist())
    assert torch.equal(torch.sort(torch.cat(q.slices + [q.truncated])).values, q.periphery)
    assert all(torch.equal(x, y) for j, (x, y) in enumerate(zip(q.slices, p.slices)) if j != i9)
    assert abs(q.slice_bias[i9] - math.log(173 / 26)) < 1e-12 and q.slice_bias[:i9] == p.slice_bias[:i9]
    assert [int(q.real(s).numel()) for s in range(1, 10)] == [39] * 8 + [49]
    assert q.bias_at(9) == {int(j): q.slice_bias[i9] for j in q.slices[i9].tolist()}
    assert fv.add_last_extra(p, av, 0) is p


def test_early_plan_from_step0():
    """X-early: the late anchor's plan rule made at step 0 (first = 1): 9 slices for steps 1-9, the computed tokens' best in the
    last two, step 9 + 10."""
    g = torch.Generator().manual_seed(12)
    av, df = torch.rand(196, generator=g), torch.rand(196, generator=g)
    rows = torch.sort(torch.randperm(98, generator=g)[:84]).values
    sinks = torch.tensor([182, int(rows[3]), int(rows[50])])
    p = fv.c2f_plan(rows, av, df, sinks, GRID, 39, first=1, core_all=True, last_extra=10)
    assert len(p.slices) == 9
    assert [int(p.real(s).numel()) for s in range(1, 10)] == [39] * 8 + [49]
    assert torch.equal(torch.sort(torch.cat(p.slices + [p.truncated])).values, p.periphery)
    assert set(p.slices[7].tolist()) <= set(rows.tolist())  # step 8's slice: computed tokens only
    assert len(set(p.slices[8].tolist()) & set(rows.tolist())) >= 16  # step 9: 16 computed + the 10 extra reads
    assert p.truncated.numel() == 19
