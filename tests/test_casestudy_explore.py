"""CPU checks of the step0-explore round (``fovea.block_smooth``, probe grid, layouts, explore / draft_mod config)."""

from types import SimpleNamespace

import pytest
import torch
from einops import rearrange

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config
from fastwam.casestudy.joint_runner import JointCaseStudyRunner

GRID = sel.Grid(frames=3, height=7, width=14)


def test_probe_grid_without_extras():
    g = fv.probe_index(GRID, extras=False)
    assert g.numel() == 36 and set(g.tolist()) <= set(fv.probe_index(GRID).tolist())
    assert not ({fv.PROBE_SINK, *fv.PROBE_CORNERS} & set(g.tolist()))


def test_block_smooth():
    const = torch.full((196, 3), 2.5)
    assert torch.allclose(fv.block_smooth(const, GRID), const)
    v = torch.zeros(196, 1)
    v[3 * 14 + 3] = 9.0  # frame 1, agentview, (3, 3)
    out = fv.block_smooth(v, GRID)
    hit = {j for j in range(196) if float(out[j]) > 0}
    assert hit == {r * 14 + c for r in (2, 3, 4) for c in (2, 3, 4)} and abs(float(out[3 * 14 + 3]) - 1.0) < 1e-6
    w = torch.zeros(196, 1)
    w[0] = 4.0  # corner: the mean of the 4 cells present
    assert abs(float(fv.block_smooth(w, GRID)[0]) - 1.0) < 1e-6 and float(fv.block_smooth(w, GRID)[7]) == 0.0


def test_tok_to_diff_matches_rearrange():
    fake = SimpleNamespace(video=SimpleNamespace(patch_size=(1, 2, 2)))
    lat = torch.randn(1, 4, 3, 14, 28)
    tok = rearrange(lat, "b c (f x) (h y) (w z) -> b (f h w) (x y z c)", x=1, y=2, z=2)[0]
    diff = rearrange(lat[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=1, b=2, e=2).reshape(-1, 16)
    assert torch.equal(JointCaseStudyRunner._tok_to_diff(fake, tok), diff)


def test_explore_config():
    cfg = parse_config({"method": "dense", "record_sinks": True, "explore_track": True,
                        "explore": [{"name": "g", "rows": "grid39"}, {"name": "s", "rows": "all", "layers": "f5l4"},
                                    {"name": "k", "rows": "grid39", "kv": True, "bias": 0.0}]})
    assert cfg.explore[0] == ("g", "grid39", "all", False, 0, 1.0, None) and cfg.explore[2][6] == 0.0
    with pytest.raises(ValueError):
        parse_config({"method": "dense", "explore": [{"name": "g"}]})  # needs record_sinks
    base = {"method": "prune", "prune": {"keep_ratio": 0.2, "fovea": "interlace_w"}}
    for m in ("prior", "white", "low", "high", "wrist", "agent", "far", "near"):
        assert parse_config({**base, "prune": {**base["prune"], "draft_mod": m, "draft_mod_scale": 0.25}}).prune.draft_mod == m
    for bad in ({"draft_mod": "pink"}, {"draft_mod": "white", "draft_from": 4}, {"draft_mod": "white", "fovea": "interlace"}):
        with pytest.raises(ValueError):
            parse_config({**base, "prune": {**base["prune"], **bad}})
    assert parse_config({**base, "score_layer": 18, "prune": {**base["prune"], "side0_m": 0, "side0_use": "video"}}).prune.side0_use == "video"


def test_c2f_survey_and_plan():
    assert fv.neighbours(0, GRID) == [0, 1, 14, 15] and len(fv.neighbours(3 * 14 + 3, GRID)) == 9
    assert fv.neighbours(7, GRID) == [7, 8, 21, 22]  # wrist block (0, 0): never crosses into agentview
    probes = fv.probe_index(GRID)
    av = torch.zeros(196)
    df = torch.zeros(196)
    av[17], av[45], df[17], df[129] = 5.0, 4.0, 3.0, 2.0  # (1, 3) frame 1 ag; (3, 3) frame 1 ag; 129 is not a probe
    sinks = torch.tensor([7, 105, 182])
    s = fv.c2f_survey(probes, av, df, sinks, 2, GRID, 39)
    assert set(sinks.tolist()) <= set(s.tolist()) and set(fv.neighbours(17, GRID)) <= set(s.tolist())
    assert set(fv.neighbours(45, GRID)) <= set(s.tolist()) and s.numel() <= 39 and torch.equal(s, torch.sort(s).values)
    measured = torch.unique(torch.cat([probes, s]))
    avm = torch.rand(196, generator=torch.Generator().manual_seed(1))
    dfm = torch.rand(196, generator=torch.Generator().manual_seed(2))
    p = fv.c2f_plan(measured, avm, dfm, sinks, GRID, 39)
    assert p.core.numel() == 20 and p.periphery.numel() == 173 and set(p.core.tolist()) <= set(measured.tolist())
    assert torch.equal(torch.sort(torch.cat(p.slices + [p.truncated])).values, p.periphery)
    for st in range(2, 10):
        assert p.real(st).numel() <= 39
    last = set(p.slices[(9 - 1) % 8].tolist())  # step 9's slice: measured periphery tokens only
    assert last <= set(measured.tolist()) and len(last) == 16
    assert parse_config({"method": "prune", "score_layer": 18, "prune": {"keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b",
                                                                         "c2f": 3}}).prune.c2f == 3
    with pytest.raises(ValueError):
        parse_config({"method": "prune", "prune": {"keep_ratio": 0.2, "fovea": "interlace_w", "c2f": 3}})


def test_step0_slices_overflow_and_spread():
    av = torch.rand(196, generator=torch.Generator().manual_seed(3))
    many = torch.arange(0, 98)  # the whole near frame computed
    sinks = torch.tensor([182, 20, 60])
    p = fv.make_plan("interlace_w", 39, GRID, av, av, av, sinks=sinks, probes=many)
    assert torch.equal(torch.sort(torch.cat(p.slices + [p.truncated])).values, p.periphery)
    assert all(x.numel() <= 16 for x in p.slices) and set(p.slices[8].tolist()) <= set(many.tolist())
    assert parse_config({"method": "prune", "score_layer": 18, "prune": {"keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b",
                                                                         "step0_rows": "f1s39", "step0_draft": "near"}}).prune.step0_rows == "f1s39"
    with pytest.raises(ValueError):
        parse_config({"method": "prune", "score_layer": 18, "prune": {"keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b",
                                                                      "step0_rows": "f1s39"}})  # grid drafts need the probe grid


def test_late_config_and_core_all():
    base = {"method": "prune", "score_layer": 18, "prune": {"keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b"}}
    cfg = parse_config({**base, "prune": {**base["prune"], "late": "f1s84", "step0_draft": "near", "step0_gain": 1.05}})
    assert cfg.prune.late == "f1s84"
    for bad in ({"late": "f1s84"}, {"late": "f1s84", "step0_draft": "near", "c2f": 3}, {"late": "f1s84", "step0_draft": "near", "step0_rows": "f1s39"}):
        with pytest.raises(ValueError):
            parse_config({**base, "prune": {**base["prune"], **bad}})
    av = torch.rand(196, generator=torch.Generator().manual_seed(5))
    df = torch.rand(196, generator=torch.Generator().manual_seed(6))
    measured = torch.arange(0, 84)
    sinks = torch.tensor([182, 3, 40])
    p = fv.c2f_plan(measured, av, df, sinks, GRID, 39, core_all=True)
    assert torch.equal(p.core, fv.select_core(av, df, sinks, 20))  # the core may hold unmeasured (far-frame) tokens
    assert torch.equal(torch.sort(torch.cat(p.slices + [p.truncated])).values, p.periphery)
    assert set(p.slices[(9 - 1) % 8].tolist()) <= set(measured.tolist())
