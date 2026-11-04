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
