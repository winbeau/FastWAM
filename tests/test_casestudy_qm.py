"""CPU checks of the quick-metrics round (casestudy.qm, quick_metrics.readout / ratios, role_plan wdiffb)."""

import math

import pytest
import torch

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import quick_metrics as qmm
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config

GRID = sel.Grid(frames=3, height=7, width=14)
MEAS = {"method": "prune", "score_layer": 18,
        "prune": {"selection": "av", "warmup_steps": 1, "keep_ratio": 0.2, "fovea": "interlace_w", "step0": "b", "step0_draft": "near",
                  "late": "f1s84", "step0_gain": 1.25, "late_extra": 10, "late_plan_core": "measured"}}
TIPS = (92, 96)  # wrist near frame, row 6 (any two wrist near-frame tokens for the test)
QM = {"k": 39, "layer": 18, "eps": [0.05, -0.05]}


def _near(cam):
    return [r * 14 + cam * 7 + c for r in range(7) for c in range(7)]


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    measured = torch.sort(torch.cat([torch.tensor(_near(0))[torch.randperm(49, generator=g)[:42]],
                                     torch.tensor(_near(1))[torch.randperm(49, generator=g)[:42]]])).values
    av, df = torch.zeros(196), torch.zeros(196)
    av[measured] = torch.rand(84, generator=g)
    df[measured] = torch.rand(84, generator=g)
    sinks = torch.tensor([182, int(measured[3]), int(measured[60])])
    return measured, av, df, sinks


def test_qm_config():
    cfg = parse_config({"method": "dense", "num_inference_steps": 10, "qm": QM})
    assert cfg.qm == QM
    assert parse_config({"method": "dense", "num_inference_steps": 10}).qm is None
    prune = {"selection": "av", "warmup_steps": 1, "keep_ratio": 0.2}
    for bad in ({"method": "dense", "num_inference_steps": 10, "qm": {**QM, "zzz": 1}},  # unknown key
                {"method": "prune", "num_inference_steps": 10, "prune": prune, "qm": QM},  # not a dense arm
                {"method": "dense", "num_inference_steps": 10, "frame0_cache": False, "qm": QM},  # no frame-0 cache
                {"method": "dense", "num_inference_steps": 2, "qm": QM}):  # fewer than 3 steps
        with pytest.raises(ValueError):
            parse_config(bad)


def test_role_plan_wdiffb():
    measured, av, df, sinks = _inputs()
    p = fv.role_plan(measured, av, df, sinks, GRID, TIPS, 39, variant="wdiffb", extra=10, seed=0)
    q = fv.role_plan(measured, av, df, sinks, GRID, TIPS, 39, variant="wdiff", extra=10, seed=0)
    sink_set, tip_set, w_near = set(sinks.tolist()), set(TIPS), set(_near(1))
    w_pool = [j for j in _near(1) if j not in sink_set and j not in tip_set]
    wm = [j for j in measured.tolist() if j in w_near and j not in sink_set and j not in tip_set]
    wr = sorted(wm, key=lambda j: (-float(df[j]), j))
    assert len(wr) >= 15
    for s in range(2, 10):
        assert torch.equal(p.real(s), q.real(s))  # wdiffb only changes the biases, not the tokens
        n = 15 if s == 9 else 10
        b, bq = p.bias_at(s), q.bias_at(s)
        assert all(b[j] == pytest.approx(math.log(len(w_pool) / n)) for j in wr[:n])
        other = [j for j in b if j not in wr[:n]]
        assert set(other) == set(bq) and all(b[j] == pytest.approx(bq[j]) for j in other)
        assert not (set(bq) & set(wr[:15]))  # wdiff has no bias on the Diff wrist group


def test_readout_identities():
    g = torch.Generator().manual_seed(0)
    f0, kf, a_dim, h, dh = 5, 12, 4, 3, 6
    n = f0 + kf + a_dim
    logits = torch.randn(h, a_dim, n, generator=g)
    vh = torch.randn(h, n, dh, generator=g)
    cams = {"all": torch.ones(kf, dtype=torch.bool),
            "agent": torch.arange(kf) < 6,
            "wrist": torch.arange(kf) >= 6}
    out = qmm.readout(logits, vh, f0, kf, {"full": (torch.arange(kf), torch.zeros(kf))}, cams)
    full = qmm.ratios(out["reads"]["full"]["all"])  # reading every future key with no bias: oS == oD
    assert full["R"] == pytest.approx(1.0, abs=1e-5)
    assert full["M"] == pytest.approx(1.0, abs=1e-5)
    assert full["rho"] == pytest.approx(1.0, abs=1e-5)

    # sum(g) is the action-mean of the head-summed norm of the action read o - cbar
    p = torch.softmax(logits, dim=-1)
    o = torch.einsum("han,hnd->had", p, vh)
    m_d = p[..., f0:f0 + kf].sum(-1)
    p_present = torch.cat([p[..., :f0], p[..., f0 + kf:]], dim=-1)
    v_present = torch.cat([vh[:, :f0], vh[:, f0 + kf:]], dim=1)
    cbar = (p_present @ v_present) / (1.0 - m_d).clamp_min(1e-30)[..., None]
    assert float(out["g"].sum()) == pytest.approx(float((o - cbar).norm(dim=-1).sum(dim=0).mean()), abs=1e-4)
    assert float(out["att"].sum()) == pytest.approx(float(m_d.mean()), abs=1e-6)

    # a single head and row: R is M * rho by construction
    g1 = torch.Generator().manual_seed(1)
    logits1 = torch.randn(1, 1, n, generator=g1)
    vh1 = torch.randn(1, n, dh, generator=g1)
    cols = torch.randperm(kf, generator=g1)[:5]
    bias = torch.randn(5, generator=g1)
    r1 = qmm.ratios(qmm.readout(logits1, vh1, f0, kf, {"five": (cols, bias)}, cams)["reads"]["five"]["all"])
    assert r1["R"] == pytest.approx(r1["M"] * r1["rho"], abs=1e-5)  # ratios rounds to 6 decimals

    z = qmm.ratios({"ro": 0.0, "oo": 0.0, "ms": 0.0, "md": 0.0, "rho_num": 0.0, "rho_den": 0.0})
    assert z["R"] is None and z["M"] is None and z["rho"] is None and z["M_rho"] is None
