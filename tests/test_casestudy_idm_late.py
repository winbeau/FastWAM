"""CPU checks of IDM-L (late-anchor-confirm): config, the late set, the drafts, the plan of steps 2-9 and fovea.add_last_extra.

Run on the server with the main venv and the IDM sources first on the path:
    PYTHONPATH=$W/repos/FastWAM-idm/src $W/repos/FastWAM/.venv/bin/python -m pytest $W/repos/FastWAM-idm/tests -q
"""

from types import SimpleNamespace

import pytest
import torch
from einops import rearrange

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config
from fastwam.casestudy.idm_runner import IDMCaseStudyRunner as R

GRID = sel.Grid(frames=3, height=7, width=14)
PATCH = (1, 2, 2)
LATE = {"method": "prune", "num_inference_steps": 10, "record_attention": True,
        "prune": {"selection": "av", "keep_ratio": 0.2, "warmup_steps": 1, "fovea": "interlace_w", "late": "f1s84",
                  "step0_draft": "near", "step0_gain": 1.25, "late_extra": 10}}


def test_late_config():
    p = parse_config(LATE).prune
    assert (p.late, p.step0_draft, p.step0_gain, p.step0_far_lambda, p.late_extra) == ("f1s84", "near", 1.25, 0.5, 10)
    m4w = parse_config({**LATE, "prune": {k: v for k, v in LATE["prune"].items()
                                          if k not in ("late", "step0_draft", "step0_gain", "late_extra")}}).prune
    assert m4w.late is None and m4w.late_extra == 0  # every new switch off: IDM M4w
    for bad in ({"late": "grid39"}, {"fovea": None}, {"fovea_off": True}, {"step0_draft": "grid"}, {"step0_gain": 0.0},
                {"late": None}):
        with pytest.raises(ValueError):
            parse_config({**LATE, "prune": {**LATE["prune"], **bad}})


def test_late_rows():
    rows = R._late_rows("f1s84", GRID)
    assert rows.numel() == 84 and torch.equal(rows, torch.sort(torch.unique(rows)).values) and int(rows.max()) < GRID.per_frame
    col = rows % GRID.width
    assert int((col < 7).sum()) == 42 and int((col >= 7).sum()) == 42  # agentview / wrist halves of the near frame
    assert torch.equal(rows, R._late_rows("f1s84", GRID))  # fixed generator
    assert R._late_rows("f1s98", GRID).tolist() == list(range(98))
    st = R._late_rows("strat84", GRID)
    blk = (st // GRID.per_frame) * 2 + ((st % GRID.width) >= 7).long()
    assert st.numel() == 84 and torch.bincount(blk, minlength=4).tolist() == [21, 21, 21, 21]
    with pytest.raises(ValueError):
        R._late_rows("grid39", GRID)


def test_spread_near_and_idw():
    g = torch.Generator().manual_seed(3)
    vals, fb = torch.randn(196, 4, generator=g), torch.randn(196, 4, generator=g)
    rows = R._late_rows("f1s84", GRID)
    out = R._spread(vals, rows, GRID, "near", fb, 0.5)
    assert torch.equal(out[rows], vals[rows].float())
    assert torch.allclose(out[98:], out[:98] + 0.5 * (out[:98] - fb[:98]))  # far = near + 0.5 (near - current frame)
    st = R._late_rows("strat84", GRID)
    o2 = R._spread(vals, st, GRID, "idw", fb, 0.0)
    assert torch.equal(o2[st], vals[st].float()) and not torch.equal(o2[98:], o2[:98] + 0.5 * (o2[:98] - fb[:98]))


def _fake_video():
    pt, ph, pw = PATCH

    def unpatchify(tok, gs):
        f, h, w = gs
        return rearrange(tok, "b (f h w) (x y z c) -> b c (f x) (h y) (w z)", f=f, h=h, w=w, x=pt, y=ph, z=pw)

    return SimpleNamespace(video=SimpleNamespace(unpatchify=unpatchify))


def test_set_draft_gain_and_map():
    """With gain g the drafted clean estimate of a computed row is z + g (x0 - z); every other token gets the spread value, and the
    returned velocity maps the latents exactly onto the spread estimate."""
    fake, (pt, ph, pw) = _fake_video(), PATCH
    c, F, H, W = 3, 3, 7 * ph, 14 * pw
    g = torch.Generator().manual_seed(4)
    lat = torch.randn(1, c, F * pt, H, W, generator=g)
    vel = torch.randn(1, F * 7 * 14, pt * ph * pw * c, generator=g)
    sig = torch.tensor(0.978)
    rows = R._late_rows("f1s84", GRID)
    v, zf = R._set_draft(fake, lat, vel, (F, 7, 14), GRID, rows, sig, PATCH, 1.25, "near", 0.5)
    x0 = lat - sig * fake.video.unpatchify(vel, (F, 7, 14))
    z = rearrange(x0[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
    zc = z[0][GRID.future_obs_index(z.device)]
    zr = z[1:].reshape(-1, z.shape[-1])
    assert torch.allclose(zf[rows], zc[rows] + 1.25 * (zr[rows] - zc[rows]), atol=1e-5)
    back = lat - sig * fake.video.unpatchify(v, (F, 7, 14))
    zb = rearrange(back[0], "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
    assert torch.allclose(zb[1:].reshape(-1, zb.shape[-1]), zf, atol=1e-4)
    assert torch.allclose(v[:, :98], torch.zeros_like(v[:, :98]), atol=1e-6)  # frame 0 does not move


def test_plan_steps_2_to_9_and_extra():
    """The plan IDM-L takes from step 1: 8 slices for steps 2-9, rows per step 39 x 7 and 49 at step 9; the action stage's step s
    set is real(s) of this plan (s >= 2)."""
    g = torch.Generator().manual_seed(5)
    av, df = torch.rand(196, generator=g), torch.rand(196, generator=g)
    rows = R._late_rows("f1s84", GRID)
    sinks = torch.tensor([fv.PROBE_SINK, int(rows[5]), int(rows[60])])
    p = fv.c2f_plan(rows, av, df, sinks, GRID, 39, core_all=True, last_extra=10)
    assert len(p.slices) == 8 and [int(p.real(s).numel()) for s in range(2, 10)] == [39] * 7 + [49]
    assert torch.equal(torch.sort(torch.cat(p.slices + [p.truncated])).values, p.periphery)
    seen = torch.unique(torch.cat([rows] + [p.real(s) for s in range(2, 10)]))
    assert seen.numel() <= 196 and set(rows.tolist()) <= set(seen.tolist())


def test_add_last_extra_present():
    assert callable(fv.add_last_extra) and fv.N_SINKS == 3 and fv.PROBE_SINK == 182
