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
