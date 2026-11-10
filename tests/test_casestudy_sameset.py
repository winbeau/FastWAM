"""CPU checks of the sameset round (2026-11-10): ``prune.late: sameset`` (the think step computes the last step's read set S),
``late_step`` 0 with the late-read jump, the read names bw16 / bw8, ``fovea.pair_bias`` (P3) and the diagnostic flags."""

import math
from types import SimpleNamespace

import pytest
import torch

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config

GRID = sel.Grid(frames=3, height=7, width=14)
BW30 = {"method": "prune", "num_inference_steps": 10, "score_layer": 18,
        "prune": {"selection": "av", "warmup_steps": 1, "keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b", "step0_draft": "near",
                  "late": "f1s84", "late_extra": 10, "late_plan_core": "measured", "late_read": "jump", "late_read_last": "bw30",
                  "step0_gain": 1.05}}
P1 = {**BW30, "prune": {**BW30["prune"], "late": "sameset"}}
P2 = {**BW30, "prune": {k: v for k, v in {**BW30["prune"], "late": "sameset", "late_step": 0}.items() if k != "late_plan_core"}}
P3 = {**P2, "prune": {**P2["prune"], "sameset_vbias": "pair"}}
FK = [r * 14 + c for r in range(2, 6) for c in range(1, 6)]  # 20 agentview body tokens


def test_sameset_config():
    for raw, step, vb in ((P1, 1, "rule"), (P2, 0, "rule"), (P3, 0, "pair")):
        p = parse_config(raw).prune
        assert (p.late, p.late_step, p.sameset_vbias, p.late_read) == ("sameset", step, vb, "jump")
    for last in ("bw16", "bw8"):
        assert parse_config({**P2, "prune": {**P2["prune"], "late_read_last": last}}).prune.late_read_last == last
    assert parse_config(BW30).prune.sameset_vbias == "rule"  # the new switch is off by default
    cfg = parse_config({**P1, "record_read_drafts": True, "record_nbr": True})
    assert cfg.record_read_drafts and cfg.record_nbr and not cfg.record_final_video
    assert parse_config({"method": "dense", "record_final_video": True, "record_nbr": True}).record_final_video
    for bad in ({**BW30["prune"], "late_step": 0},  # late_step 0 with the jump only for sameset
                {**BW30["prune"], "sameset_vbias": "pair"},  # pair needs sameset
                {**P2["prune"], "sameset_vbias": "other"},
                {**P2["prune"], "late_read_last": "near"},  # sameset needs a body read
                {**P2["prune"], "late_read": "lr"},
                {**P1["prune"], "late_read_land": 5, "late_read_mid": "core"}):
        with pytest.raises(ValueError):
            parse_config({**BW30, "prune": bad})
    with pytest.raises(ValueError):
        parse_config({"method": "dense", "record_read_drafts": True})


@pytest.mark.parametrize("last,n_w", (("bw30", 30), ("bw16", 16), ("bw8", 8)))
def test_read_body_wrist_counts(last, n_w):
    rows, bias, sizes = fv.read_body(last, GRID, {"fk": FK}, (78, 82), 0)
    wr = {j for j in range(98) if j % 14 >= 7}
    r = set(rows.tolist())
    assert set(FK) | {84, 78, 82} <= r and len(r) == len(FK) + 1 + 2 + n_w
    sampled = (r & wr) - {78, 82}
    assert len(sampled) == n_w and all(bias[j] == pytest.approx(math.log(2) + math.log(47 / n_w)) for j in sampled)
    assert all(bias[j] == pytest.approx(math.log(2) + math.log(49 / len(FK))) for j in FK)
    if last == "bw30":  # bw30 unchanged by the new names
        assert sizes["3"] == [47, 30, round(math.log(47 / 30), 6)]


def _sameset_rows(last):
    from fastwam.casestudy.joint_runner import JointCaseStudyRunner as J
    fake = SimpleNamespace(cfg=parse_config({**P2, "prune": {**P2["prune"], "late_read_last": last}}), _body_rows={"fk": FK})
    return J._explore_rows(fake, "sameset", GRID, None)


@pytest.mark.parametrize("last", ("bw30", "bw16", "bw8"))
def test_sameset_rows_equal_the_read(last):
    """The think set is the read set: the same read_body call the late-read plan makes for the last step."""
    rows = _sameset_rows(last)
    plan = fv.Plan("interlace_w", *(torch.zeros(0, dtype=torch.long),) * 4, truncated=torch.zeros(0, dtype=torch.long))
    p = fv.late_read_plan(plan, last, GRID, first=1, n_steps=10, body={"fk": FK}, tips=(78, 82), seed=0)
    assert torch.equal(rows, p.real(9)) and all(p.real(s).numel() == 0 for s in range(1, 9))
    assert set(p.bias_at(9)) == set(rows.tolist())


def test_pair_bias():
    rows = _sameset_rows("bw30")
    b0, bf, cnt = fv.pair_bias(rows, FK, GRID)
    present = set(rows.tolist())
    assert sum(cnt["frame0"]) + sum(cnt["future"].values()) == 196 - len(present)
    ag = [j for j in range(98) if j % 14 < 7]
    for p in ag:  # frame-0 stand-ins: agentview off the body, near (if not read) and far
        want = (0 if p in present else 1) + (0 if p in FK else 1)
        assert cnt["frame0"][p] == want and float(b0[p]) == pytest.approx(math.log(1 + want))
    assert all(cnt["frame0"][p] == 0 for p in range(98) if p % 14 >= 7)
    assert cnt["frame0"][84] == 1  # the sink is read (near), its far token stands with frame 0
    for j in FK:  # a body token stands in for its far-frame token
        assert cnt["future"][j] == 1 and bf[j] == pytest.approx(math.log(2))
    reads_w = [j for j in rows.tolist() if j % 14 >= 7]
    assert sum(cnt["future"][j] for j in reads_w) == (49 - len(reads_w)) + 49  # every absent wrist token, near and far
    assert set(bf) == present and float(b0.min()) == 0.0
    # the nearest-read rule on a hand case: wrist reads 77 and 91 only, plus body 1
    b0s, bfs, cs = fv.pair_bias([1, 77, 91], [1], GRID)
    assert cs["future"][77] + cs["future"][91] == 47 + 49 and cs["future"][1] == 1  # 47 near wrist not read + the far 49
    assert sum(cs["frame0"]) == 48 + 48  # agentview: 48 near off-body not read, 48 far off-body
    assert (cs["future"][77], cs["future"][91]) == (41 + 42, 6 + 7)  # 77 = (5, 0) takes rows 0-5, 91 = (6, 0) row 6
    assert cs["future"][77] == sum(1 for t in list(range(98)) + list(range(98, 196)) if t % 98 % 14 >= 7 and t not in (77, 91)
                                   and ((t % 98) // 14 - 5) ** 2 + ((t % 98) % 14 - 7) ** 2 <= ((t % 98) // 14 - 6) ** 2 + ((t % 98) % 14 - 7) ** 2)
    with pytest.raises(ValueError):
        fv.pair_bias([1, 77, 120], [1], GRID)  # a far-frame row
    with pytest.raises(ValueError):
        fv.pair_bias([1, 2], [1], GRID)  # no wrist read
