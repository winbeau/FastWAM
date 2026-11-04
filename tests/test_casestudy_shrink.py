"""CPU checks of the shrinkage-mechanism round (fovea.roll_blocks, the kappa_val sketch, the tuned lens, the config switches)."""

from types import SimpleNamespace

import pytest
import torch

from fastwam.casestudy import fovea as fv
from fastwam.casestudy import selection as sel
from fastwam.casestudy.config import parse_config
from fastwam.casestudy.joint_runner import JointCaseStudyRunner as J

GRID = sel.Grid(frames=3, height=7, width=14)
M4W = {"method": "prune", "prune": {"keep_ratio": 0.2, "warmup_steps": 1, "fovea": "interlace_w"}}


def test_roll_blocks():
    v = torch.arange(196, dtype=torch.float32)[:, None].repeat(1, 3)
    r = fv.roll_blocks(v, GRID)
    assert torch.equal(torch.sort(r[:, 0]).values, v[:, 0]) and float(r.norm()) == float(v.norm())  # same values, same size
    # frame 1, agentview row 2: column c gets column c - 1 (cyclic inside the 7-wide camera half)
    row = [2 * 14 + c for c in range(7)]
    assert r[row, 0].tolist() == [float(2 * 14 + (c - 1) % 7) for c in range(7)]
    wr = [98 + 3 * 14 + 7 + c for c in range(7)]  # frame 2, wrist row 3
    assert r[wr, 0].tolist() == [float(98 + 3 * 14 + 7 + (c - 1) % 7) for c in range(7)]
    assert torch.equal(fv.roll_blocks(fv.roll_blocks(v, GRID, 3), GRID, 4), v)


def test_draft_mods_and_switches_config():
    for mode in ("shift", "real"):
        assert parse_config({**M4W, "prune": {**M4W["prune"], "draft_mod": mode, "draft_mod_scale": 1.0}}).prune.draft_mod == mode
    assert parse_config({**M4W, "record_kval": True}).record_kval
    dense = {"method": "dense", "num_inference_steps": 10}
    mech = {"steps": [0, 1], "calls": 12, "sets": ["strat39", "f1s84"], "noise": [[0, 30]], "patch_set": "strat39",
            "patch_ranges": [[0, 30]], "selfkey": ["strat39"], "lens_apply": "ckpt/lens.pt"}
    assert parse_config({**dense, "mech": mech}).mech["calls"] == 12
    for bad in ({**M4W, "mech": mech}, {**dense, "mech": {**mech, "unknown": 1}}, {**dense, "mech": {**mech, "lens_fit": "a.pt"}}):
        with pytest.raises(ValueError):
            parse_config(bad)


def test_kval_sketch_is_m_times_fbar_minus_cbar():
    g = torch.Generator().manual_seed(1)
    h, dh, a, f0, nf = 2, 4, 3, 5, 6
    q = torch.randn(1, a, h * dh, generator=g)
    k = torch.randn(1, f0 + nf + a, h * dh, generator=g)
    v = torch.randn(1, f0 + nf + a, h * dh, generator=g)
    fake = SimpleNamespace(num_heads=h, KVAL_DIM=J.KVAL_DIM, _kval_proj=None)
    fake._head_logits = lambda q_, k_: J._head_logits(fake, q_, k_)
    sk, m = J._kval(fake, q, k, v, None, f0, a)
    p = torch.softmax(J._head_logits(fake, q, k), dim=-1)
    vh = v[0].reshape(-1, h, dh).permute(1, 0, 2)
    fut = slice(f0, f0 + nf)
    mm = p[..., fut].sum(-1, keepdim=True)
    fbar = (p[..., fut] @ vh[:, fut]) / mm
    pres = torch.cat([p[..., :f0], p[..., f0 + nf:]], dim=-1)
    vp = torch.cat([vh[:, :f0], vh[:, f0 + nf:]], dim=1)
    cbar = (pres @ vp) / pres.sum(-1, keepdim=True)
    want = (mm * (fbar - cbar)).reshape(-1) @ fake._kval_proj[1]
    assert torch.allclose(sk, want, atol=1e-5) and abs(m - float(mm.mean())) < 1e-6
    sk2, _ = J._kval(fake, q, k, v, None, f0, a)
    assert torch.equal(sk, sk2)  # the projection is fixed


def test_tuned_lens_recovers_an_affine_map():
    g = torch.Generator().manual_seed(2)
    w_true, b_true = torch.randn(16, 5, generator=g), torch.randn(5, generator=g)
    fake = SimpleNamespace(_lens_acc={})
    for _ in range(4):
        h = [torch.randn(200, 16, generator=g)]
        J._lens_update(fake, 0, h, h[0] @ w_true + b_true)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        J._lens_save(fake, d + "/lens.pt", ridge=1e-8)
        lens = torch.load(d + "/lens.pt")
    hx = torch.randn(50, 16, generator=g)
    pred = J._lens_pred(lens[0]["w"][0], hx)
    assert lens[0]["n"] == 800 and torch.allclose(pred, hx @ w_true + b_true, atol=1e-3)
