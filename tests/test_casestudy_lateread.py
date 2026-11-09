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


def test_jump_land_config():
    p = parse_config(_arm(late_read="jump", late_read_last="all", late_read_land=8, late_read_mid="plus10", late_read_seed=3)).prune
    assert p.late_read_land == 8 and p.late_read_mid == "plus10" and p.late_read_seed == 3
    assert parse_config({"method": "dense", "num_inference_steps": 10, "qm": {"grip_eps": [0.2, -0.2], "only_c": True}}).qm["only_c"]
    for bad in (_arm(late_read="lr", late_read_land=8, late_read_mid="core"), _arm(late_read="jump", late_read_land=8),
                _arm(late_read="jump", late_read_land=9, late_read_mid="core"), _arm(late_read="jump", late_read_land=2, late_read_mid="core"),
                _arm(late_read="jump", late_read_land=8, late_read_mid="xx"), _arm(late_read="jump", late_read_mid="core")):
        with pytest.raises(ValueError):
            parse_config(bad)


@pytest.mark.parametrize("mid", ("plus10", "core"))
def test_late_read_plan_land(mid):
    plan = _plan()
    p = fv.late_read_plan(plan, "all", GRID, land=8, mid=mid, seed=0)
    for s in range(2, 8):
        assert p.real(s).numel() == 0 and p.bias_at(s) == {}
    assert torch.equal(p.real(9), torch.arange(196)) and p.bias_at(9) == {}
    rows, bias, own = p.real(8), p.bias_at(8), plan.real(8)
    if mid == "core":
        assert torch.equal(rows, torch.sort(plan.fixed).values) and rows.numel() == 23 and bias == {}
    else:
        extra = sorted(set(rows.tolist()) - set(own.tolist()))
        assert rows.numel() == 49 and set(own.tolist()) <= set(rows.tolist()) and len(extra) == 10 and all(j < 98 for j in extra)
        pool = torch.tensor([j for j in range(98) if j not in set(own.tolist())])  # stratified over the near frame's two cameras
        assert extra == fv.stratified_pick(pool, 10, GRID, 0).tolist()
        slc = [j for j in rows.tolist() if j not in set(plan.fixed.tolist())]
        assert set(bias) == set(slc) and len(slc) == 26
        assert all(v == pytest.approx(math.log(plan.periphery.numel() / 26)) for v in bias.values())
        assert torch.equal(fv.late_read_plan(plan, "all", GRID, land=8, mid=mid, seed=0).real(8), rows)


@pytest.mark.parametrize("last", ("u49", "n49", "flip49"))
def test_read49(last):
    plan = _plan()
    g = torch.Generator().manual_seed(3)
    measured = torch.sort(torch.randperm(98, generator=g)[:84]).values
    d = torch.full((196,), float("nan"))
    d[measured] = torch.rand(84, generator=g)
    p = fv.late_read_plan(plan, last, GRID, d_meas=d, tips=(78, 82), seed=0)
    rows, bias = p.real(9), p.bias_at(9)
    assert rows.numel() == 49 and bool((rows < 98).all()) and set(bias) == set(rows.tolist())
    lg2 = math.log(2.0)
    ag = [j for j in range(98) if j % 14 < 7]
    wr = [j for j in range(98) if j % 14 >= 7]
    r = set(rows.tolist())
    if last == "u49":
        assert len(r & set(ag)) == 25 and len(r & set(wr)) == 24
        assert all(bias[j] == pytest.approx(lg2 + math.log(49 / 25)) for j in r & set(ag))
    if last == "n49":
        am = sorted((j for j in ag if j in set(measured.tolist())), key=lambda j: (-float(d[j]), j))[:12]
        assert set(am) <= r and {78, 82} <= r and len(r & set(ag)) == 20 and len(r & set(wr)) == 29
        assert all(bias[j] == pytest.approx(lg2) for j in am + [78, 82])
        assert all(bias[j] == pytest.approx(lg2 + math.log(37 / 8)) for j in (r & set(ag)) - set(am))
        assert all(bias[j] == pytest.approx(lg2 + math.log(47 / 27)) for j in (r & set(wr)) - {78, 82})
    if last == "flip49":
        wm = sorted((j for j in wr if j in set(measured.tolist()) and j not in (78, 82)), key=lambda j: (-float(d[j]), j))[:27]
        assert set(wm) <= r and {78, 82} <= r and len(r & set(ag)) == 20
        assert all(bias[j] == pytest.approx(lg2 + math.log(49 / 20)) for j in r & set(ag))
        assert all(bias[j] == pytest.approx(lg2) for j in wm + [78, 82])
    with pytest.raises(ValueError):
        fv.late_read_plan(plan, last, GRID)
    assert parse_config(_arm(late_read="jump", late_read_last=last)).prune.late_read_tips == (78, 82)


@pytest.mark.parametrize("last", ("bw20", "bw30", "br30", "bwseg30", "bwc30"))
@pytest.mark.parametrize("with_sink", (False, True))
def test_read_body(last, with_sink):
    plan = _plan()
    lg2 = math.log(2.0)
    ag = [j for j in range(98) if j % 14 < 7]
    wr = [j for j in range(98) if j % 14 >= 7]
    fk = [r * 14 + c for r in range(2, 6) for c in range(1, 6)] + ([84] if with_sink else [])  # 20 (+ the sink) agentview tokens
    body = {"fk": fk, "fk_cover": [r * 14 + c for r in range(3, 5) for c in range(2, 5)], "seg": [r * 14 + c for r in range(1, 5) for c in range(0, 4)]}
    p = fv.late_read_plan(plan, last, GRID, body=body, tips=(78, 82), seed=0)
    rows, bias = p.real(9), p.bias_at(9)
    r = set(rows.tolist())
    b = set(body[fv.BODY_SOURCE.get(last, "fk")])
    n_w = 20 if last == "bw20" else 30
    assert bool((rows < 98).all()) and set(bias) == r and torch.unique(rows).numel() == rows.numel()
    assert b <= r and {78, 82, 84} <= r
    w_sampled = (r & set(wr)) - {78, 82}
    assert len(w_sampled) == n_w and all(bias[j] == pytest.approx(lg2 + math.log(47 / n_w)) for j in w_sampled)
    assert bias[78] == pytest.approx(lg2) and bias[82] == pytest.approx(lg2)
    if 84 not in b:
        assert bias[84] == pytest.approx(lg2)
    a_other = (r & set(ag)) - b - {84}
    if last == "br30":
        pool = len(ag) - len(b) - (0 if 84 in b else 1)
        assert all(bias[j] == pytest.approx(lg2) for j in b)
        assert len(a_other) == 6 and all(bias[j] == pytest.approx(lg2 + math.log(pool / 6)) for j in a_other)
    else:
        assert not a_other and all(bias[j] == pytest.approx(lg2 + math.log(49 / len(b))) for j in b)
    assert rows.numel() == len(b) + (0 if 84 in b else 1) + 2 + n_w + (6 if last == "br30" else 0)
    assert torch.equal(fv.late_read_plan(plan, last, GRID, body=body, tips=(78, 82), seed=0).real(9), rows)
    with pytest.raises(ValueError):
        fv.late_read_plan(plan, last, GRID, tips=(78, 82))
    with pytest.raises(ValueError):
        fv.late_read_plan(plan, last, GRID, body={**body, "fk": [7], "seg": [7], "fk_cover": [7]}, tips=(78, 82))  # a wrist token
    assert parse_config(_arm(late_read="jump", late_read_last=last)).prune.late_read_last == last


def test_body_anchor():
    body = {"fk": [r * 14 + c for r in range(2, 6) for c in range(1, 6)]}  # 20 agentview tokens
    rows = fv.body_anchor(body, GRID, 49, seed=1)
    r = set(rows.tolist())
    wr = r - set(body["fk"])
    assert rows.numel() == 49 and set(body["fk"]) <= r and len(wr) == 29
    assert all(j < 98 and j % 14 >= 7 for j in wr)  # the rest is the wrist's near frame
    assert torch.equal(fv.body_anchor(body, GRID, 49, seed=1), rows)  # fixed seed
    # the blocks are 2 x 2 windows: every wrist token but at most 1 (the cut block) has a chosen 4-neighbour inside its block
    lone = [j for j in wr if not any(k in wr for k in (j - 1, j + 1, j - 14, j + 14) if (k % 14 >= 7) == (j % 14 >= 7))]
    assert len(lone) <= 1
    for bad in (None, {"fk": []}, {"fk": [7]}):  # no mask, empty, a wrist token
        with pytest.raises(ValueError):
            fv.body_anchor(bad, GRID, 49, seed=1)
    with pytest.raises(ValueError):  # the body alone exceeds the budget
        fv.body_anchor(body, GRID, 10, seed=1)
