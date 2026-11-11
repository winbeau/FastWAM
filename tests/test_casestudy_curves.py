"""CPU checks of the chain2-curves reads (2026-11-10, task 014): ``fovea.CURVE_A`` (agentview: drop background / body) and ``CURVE_W``
(wrist: spread / clumped) against Bw-W30 (bw30)."""

import math

import pytest
import torch

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config

GRID = sel.Grid(frames=3, height=7, width=14)
FK = [r * 14 + c for r in range(2, 6) for c in range(1, 6)]  # 20 agentview body tokens
AG = [j for j in range(98) if j % 14 < 7]
WR = [j for j in range(98) if j % 14 >= 7]
LG2 = math.log(2.0)


def _read(last, fk=FK, seed=7):
    rows, bias, sizes = fv.read_body(last, GRID, {"fk": fk, "draw_seed": seed}, (78, 82), 0)
    return set(rows.tolist()), bias, sizes


@pytest.mark.parametrize("last", ("c2a_bg100", "c2w_s30"))
@pytest.mark.parametrize("fk", (FK, FK + [84]))
def test_endpoints_equal_bw30(last, fk):
    """Drop background 100 % and spread 30 are Bw-W30: same rows, same biases (bit for bit)."""
    r0, b0, _ = fv.read_body("bw30", GRID, {"fk": fk}, (78, 82), 0)
    r1, b1, _ = fv.read_body(last, GRID, {"fk": fk, "draw_seed": 123}, (78, 82), 0)
    assert torch.equal(r0, r1) and b0 == b1


def test_agentview_drops_nested_and_bias():
    bg_pool = [j for j in AG if j not in FK and j != 84]  # 28
    prev_bg, prev_bd = None, None
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        name_bg = "c2a_full" if x == 0 else f"c2a_bg{int(x * 100)}"
        r, bias, sizes = _read(name_bg)
        a = r & set(AG)
        assert set(FK) | {84} <= a and len(set(bg_pool) - a) == int(math.floor(x * 28 + 0.5))
        n = len(a - {84})
        assert all(bias[j] == pytest.approx(LG2 + math.log(49 / n)) for j in a - {84}) and bias[84] == pytest.approx(LG2)
        if prev_bg is not None:
            assert a <= prev_bg  # nested
        prev_bg = a
        if x > 0:
            r, bias, _ = _read(f"c2a_bd{int(x * 100)}")
            a = r & set(AG)
            assert set(bg_pool) | {84} <= a and len(set(FK) - a) == int(math.floor(x * 20 + 0.5))
            if prev_bd is not None:
                assert a <= prev_bd
            prev_bd = a
    r, bias, _ = _read("c2a_bd100nb")
    assert r & set(AG) == set(bg_pool) | {84} and all(bias[j] == pytest.approx(LG2) for j in bg_pool)
    r2, b2, _ = _read("c2a_bd100")
    assert r == r2 and all(b2[j] == pytest.approx(LG2 + math.log(49 / 28)) for j in bg_pool)
    # the wrist of the agentview arms is bw30's
    rw, bw, _ = fv.read_body("bw30", GRID, {"fk": FK}, (78, 82), 0)
    assert r & set(WR) == set(rw.tolist()) & set(WR)
    # a different call seed gives a different drop order, the same counts
    assert (_read("c2a_bg50", seed=1)[0] & set(AG)) != (_read("c2a_bg50", seed=2)[0] & set(AG))


def test_sink_inside_body_is_never_dropped():
    fk = FK + [84]
    r, bias, _ = _read("c2a_bd100", fk=fk)
    a = r & set(AG)
    assert 84 in a and not (set(FK) & a)
    n = len(a)
    assert bias[84] == pytest.approx(LG2 + math.log(49 / n))  # inside the body it is an agentview read (bw30's rule)


@pytest.mark.parametrize("n", (4, 8, 16, 30, 47))
def test_wrist_spread_and_clumped(n):
    pool = [j for j in WR if j not in (78, 82)]
    r, bias, _ = _read(f"c2w_s{n}")
    w = (r & set(WR)) - {78, 82}
    assert len(w) == n and all(bias[j] == pytest.approx(LG2 + math.log(47 / n)) for j in w)
    assert set(FK) | {84, 78, 82} <= r and all(bias[j] == pytest.approx(LG2 + math.log(49 / 20)) for j in FK)
    if n == 47:
        assert w == set(pool)
        return
    r, bias, _ = _read(f"c2w_c{n}")
    w = sorted((r & set(WR)) - {78, 82})
    assert len(w) == n and all(bias[j] == pytest.approx(LG2 + math.log(47 / n)) for j in w)
    # clumped: the n nearest to one centre (some read cell has every other read cell within the n-th distance)
    g = torch.Generator().manual_seed(1 + 7)
    centre = pool[int(torch.randint(len(pool), (1,), generator=g))]
    d = lambda j: (j // 14 - centre // 14) ** 2 + (j % 14 - centre % 14) ** 2  # noqa: E731
    assert centre in w and max(d(j) for j in w) <= min(d(j) for j in pool if j not in w)


def test_curve_config():
    base = {"method": "prune", "score_layer": 18,
            "prune": {"selection": "av", "warmup_steps": 1, "keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b", "step0_draft": "near",
                      "late": "f1s84", "late_extra": 10, "late_plan_core": "measured", "late_read": "jump", "step0_gain": 1.05}}
    for last in list(fv.CURVE_A) + list(fv.CURVE_W):
        assert parse_config({**base, "prune": {**base["prune"], "late_read_last": last}}).prune.late_read_last == last
        assert last in fv.BODY_READS
    assert "c2a_full" not in fv.CALL_SEED_READS and "c2w_s4" not in fv.CALL_SEED_READS
    assert {"c2a_bg25", "c2a_bd100nb", "c2w_c4", "bwm_b16"} <= set(fv.CALL_SEED_READS)


# ---- chain2-curves-more (2026-11-11, task 015)
X8 = {"12p5": 0.125, "25": 0.25, "37p5": 0.375, "50": 0.5, "62p5": 0.625, "75": 0.75, "87p5": 0.875, "100": 1.0}


@pytest.mark.parametrize("kind", ("bg", "bd"))
def test_nine_steps_nested(kind):
    """The 014 lines with the four new steps: one drop order per call, so all nine steps nest."""
    prev = _read("c2a_full")[0] & set(AG)
    pool = [j for j in AG if j not in FK and j != 84] if kind == "bg" else FK
    for s, x in X8.items():
        a = _read(f"c2a_{kind}{s}")[0] & set(AG)
        assert len(set(pool) - a) == int(math.floor(x * len(pool) + 0.5)) and a <= prev
        prev = a


@pytest.mark.parametrize("fk", (FK, FK + [84]))
def test_random_drop(fk):
    """c2a_rd<x>: x % of the 48 agentview tokens off the sink, body and background alike, nested; the sink always read; the agentview
    bias log(49 / count) (the sink too when inside the body); rd100 reads only the sink."""
    pool = [j for j in AG if j != 84]
    prev = set(AG)
    for s, x in X8.items():
        r, bias, sizes = _read(f"c2a_rd{s}", fk=fk)
        a = r & set(AG)
        m = int(math.floor(x * 48 + 0.5))
        assert 84 in a and len(set(pool) - a) == m and a <= prev and sizes["agent_pool"] == [48, m]
        prev = a
        biased = a if 84 in fk else a - {84}
        assert all(bias[j] == pytest.approx(LG2 + math.log(49 / len(biased))) for j in biased)
        if 84 not in fk:
            assert bias[84] == pytest.approx(LG2)
        rw, _, _ = fv.read_body("bw30", GRID, {"fk": fk}, (78, 82), 0)
        assert r & set(WR) == set(rw.tolist()) & set(WR)
    assert prev == {84}
    # random over body and background: across call seeds the 50 % drop hits both
    hit_b = hit_g = False
    for seed in range(20):
        a = _read("c2a_rd50", fk=fk, seed=seed)[0] & set(AG)
        hit_b |= bool(set(FK) - a)
        hit_g |= bool(set(pool) - set(FK) - a)
    assert hit_b and hit_g


@pytest.mark.parametrize("s", ("25", "50", "75", "100"))
def test_background_no_bias(s):
    r0, b0, _ = _read(f"c2a_bg{s}")
    r1, b1, _ = _read(f"c2a_bg{s}nb")
    assert r0 == r1
    a = r1 & set(AG)
    assert all(b1[j] == pytest.approx(LG2) for j in a)
    assert all(b1[j] == pytest.approx(b0[j]) for j in r1 - set(AG))


@pytest.mark.parametrize("n", (2, 6, 12, 20, 24, 38))
def test_more_wrist_n(n):
    pool = [j for j in WR if j not in (78, 82)]
    for k in ("s", "c"):
        r, bias, _ = _read(f"c2w_{k}{n}")
        w = (r & set(WR)) - {78, 82}
        assert len(w) == n and w <= set(pool) and all(bias[j] == pytest.approx(LG2 + math.log(47 / n)) for j in w)
    w_s = (_read(f"c2w_s{n}", seed=1)[0] & set(WR)) - {78, 82}
    assert w_s == (_read(f"c2w_s{n}", seed=2)[0] & set(WR)) - {78, 82}  # fixed pattern


def _quads(w):
    return sorted(((j // 14) >= 4) * 2 + ((j % 14 - 7) >= 4) for j in w)


@pytest.mark.parametrize("n", (2, 4, 6, 8, 12, 16))
def test_redraw(n):
    """c2w_r<n>: quad_pick seeded with the call's noise seed: a new set each call, the same strata quota as the fixed pattern."""
    pool = torch.tensor([j for j in WR if j not in (78, 82)], dtype=torch.long)
    fixed = (_read(f"c2w_s{n}")[0] & set(WR)) - {78, 82}
    sets = []
    for seed in (11, 12, 13, 14):
        r, bias, _ = _read(f"c2w_r{n}", seed=seed)
        w = (r & set(WR)) - {78, 82}
        assert w == set(fv.quad_pick(pool, n, GRID, seed).tolist()) and _quads(w) == _quads(fixed)
        assert all(bias[j] == pytest.approx(LG2 + math.log(47 / n)) for j in w)
        assert r & set(AG) == _read(f"c2w_s{n}")[0] & set(AG)
        sets.append(w)
    assert len({tuple(sorted(w)) for w in sets}) > 1
    with pytest.raises(ValueError):
        fv.read_body(f"c2w_r{n}", GRID, {"fk": FK}, (78, 82), 0)


@pytest.mark.parametrize("n", (4, 8, 16, 30))
def test_spread_no_bias(n):
    r0, b0, _ = _read(f"c2w_s{n}")
    r1, b1, _ = _read(f"c2w_s{n}nb")
    assert r0 == r1
    w = (r1 & set(WR)) - {78, 82}
    assert all(b1[j] == pytest.approx(LG2) for j in w)
    assert all(b1[j] == pytest.approx(b0[j]) for j in r1 - w)


def test_more_config():
    new = ([f"c2a_{k}{s}" for k in ("bg", "bd") for s in ("12p5", "37p5", "62p5", "87p5")] + [f"c2a_rd{s}" for s in X8]
           + [f"c2a_bg{s}nb" for s in ("25", "50", "75", "100")] + [f"c2w_{k}{n}" for k in "sc" for n in (2, 6, 12, 20, 24, 38)]
           + [f"c2w_r{n}" for n in (2, 4, 6, 8, 12, 16)] + [f"c2w_s{n}nb" for n in (4, 8, 16, 30)])
    assert len(new) == 42 and set(new) <= set(fv.CURVE_A) | set(fv.CURVE_W)
    assert len(fv.CURVE_A) + len(fv.CURVE_W) == 19 + 42
    assert {f"c2w_r{n}" for n in (2, 4, 6, 8, 12, 16)} <= set(fv.CALL_SEED_READS)
    assert {f"c2a_rd{s}" for s in X8} | {"c2a_bg25nb"} <= set(fv.CALL_SEED_READS)
    assert not {"c2w_s2", "c2w_s4nb", "c2w_s38"} & set(fv.CALL_SEED_READS)
