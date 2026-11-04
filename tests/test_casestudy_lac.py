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


def test_late_oracle_config():
    """late-anchor-c4 attribution switch: only on the late anchor (late_step 1); record_drafts parses."""
    for o in ("far", "near", "wrist", "agent", "plan", "all"):
        cfg = parse_config({**LATE, "record_drafts": True, "prune": {**LATE["prune"], "late_oracle": o}})
        assert cfg.prune.late_oracle == o and cfg.record_drafts
    assert parse_config(LATE).prune.late_oracle is None and not parse_config(LATE).record_drafts
    for bad in ({**LATE["prune"], "late_oracle": "both"}, {**LATE["prune"], "late_oracle": "far", "late_step": 0},
                {**M4W["prune"], "late_oracle": "far"}):
        with pytest.raises(ValueError):
            parse_config({**LATE, "prune": bad})


def _rows(spec):
    from types import SimpleNamespace
    from fastwam.casestudy.joint_runner import JointCaseStudyRunner as J
    fake = SimpleNamespace()
    fake._explore_rows = lambda s, g, p: J._explore_rows(fake, s, g, p)
    return fake._explore_rows(spec, GRID, None)


def test_far_and_union_rows():
    f2w = _rows("f2w14")
    assert f2w.numel() == 14 and bool((f2w >= 98).all()) and bool(((f2w % 14) >= 7).all())
    f2s = _rows("f2s20")
    assert f2s.numel() == 20 and bool((f2s >= 98).all()) and int(((f2s % 14) >= 7).sum()) == 10
    u = _rows("f1s70+f2w14")
    assert torch.equal(u, torch.sort(torch.unique(torch.cat([_rows("f1s70"), f2w]))).values) and u.numel() == 84
    assert torch.equal(_rows("f1s84"), _rows("f1s84"))  # each part keeps its own fixed generator


def test_spread_nearres_and_wrist_lambda():
    from fastwam.casestudy.joint_runner import JointCaseStudyRunner as J
    g = torch.Generator().manual_seed(21)
    vals, fb = torch.randn(196, 3, generator=g), torch.randn(196, 3, generator=g)
    rows = _rows("f1s70+f2w14")
    near_only = rows[rows < 98]
    base = J._spread(None, vals, near_only, GRID, "near", fb, 0.5)
    own = rows[rows >= 98]
    v2 = vals.clone()
    v2[own] = base[own] + 0.7  # every computed far token sits 0.7 above the extrapolation
    out = J._spread(None, v2, rows, GRID, "nearres", fb, 0.5)
    j = torch.arange(196)
    far_wrist = (j >= 98) & ((j % 14) >= 7)
    far_agent = (j >= 98) & ((j % 14) < 7)
    assert torch.allclose(out[far_wrist], base[far_wrist] + 0.7, atol=1e-5)  # the residual spread over the far wrist block
    assert torch.equal(out[far_agent], base[far_agent]) and torch.equal(out[:98], base[:98])
    assert torch.equal(J._spread(None, vals, near_only, GRID, "near", fb, 0.5), J._spread(None, vals, near_only, GRID, "near", fb, 0.5, None))
    lw = J._spread(None, vals, near_only, GRID, "near", fb, 0.5, 0.1)
    near = base[:98]
    col = (torch.arange(98) % 14) >= 7
    assert torch.allclose(lw[98:][col], (near + 0.1 * (near - fb[:98]))[col]) and torch.equal(lw[98:][~col], base[98:][~col])
    cfg = parse_config({**LATE, "prune": {**LATE["prune"], "late": "f1s70+f2w14", "step0_draft": "nearres", "step0_far_lambda_wrist": 0.25}})
    assert cfg.prune.step0_draft == "nearres" and cfg.prune.step0_far_lambda_wrist == 0.25
    with pytest.raises(ValueError):
        parse_config({**LATE, "prune": {**LATE["prune"], "step0_draft": "idw", "step0_far_lambda_wrist": 0.25}})


def test_late_plan_options_config():
    cfg = parse_config({**LATE, "prune": {**LATE["prune"], "late_plan_far_av": 0.5, "late_plan_far_diff": 0.9, "late_plan_core": "measured"}})
    assert (cfg.prune.late_plan_far_av, cfg.prune.late_plan_far_diff, cfg.prune.late_plan_core) == (0.5, 0.9, "measured")
    for bad in ({**M4W["prune"], "late_plan_far_av": 0.5}, {**LATE["prune"], "late_plan_core": "all"},
                {**LATE["prune"], "late_plan_far_diff": -1.0}, {**LATE["prune"], "late_plan_far_av": 0.5, "late_oracle": "plan"}):
        with pytest.raises(ValueError):
            parse_config({**LATE, "prune": bad})
