"""CPU checks of the along-cross round (``fastwam/casestudy/along.py`` and the hybrid modes prog / cross / prog_cross / rot);
D1-D3 of experiments/along-cross/DECISIONS.md."""

import numpy as np
import pytest

from fastwam.casestudy import along
from fastwam.casestudy.hybrid import ALONG, mix

TOL = 1e-6


def _bend():
    """D1: five steps of (0.2, 0, 0), then (0, 0.2, 0); z = 0."""
    a = np.zeros((32, 7))
    a[:5, 0] = 0.2
    a[5:, 1] = 0.2
    return a


def test_d1_bend():
    a = _bend()
    m = along.measure(a, np.array([1.0, 0.5, 0.0]))
    assert abs(m["r_along"] - 0.75) < 1e-12 and abs(m["e"]) < 1e-12 and abs(m["k_star"] - 7.5) < 1e-12
    assert abs(m["theta_total"] - np.degrees(np.arccos(1.5 / np.sqrt(1.25 * 2.0)))) < 1e-9
    assert abs(m["theta_total"] - 18.43) < 0.01 and abs(m["theta_resid"]) < 1e-6
    m = along.measure(a, np.array([1.2, 0.5, 0.0]))
    assert abs(m["e"] - 0.2) < 1e-12 and abs(m["r_along"] - 0.75) < 1e-12
    assert np.allclose(m["delta"], [0.2, 0.0, 0.0], atol=1e-12)
    assert abs(along.psi(a) - 90.0) < 1e-9


def test_projection_ties_and_ends():
    a = np.zeros((32, 7))
    a[:4, 0] = 0.25  # out to x = 1 ...
    a[4:8, 0] = -0.25  # ... and back to 0: x = 0.5 is reached at k = 2 and k = 6
    k, p, d = along.project(along.cumulative(a[:, 0:3]), np.array([0.5, 0.0, 0.0]))
    assert abs(k - 2.0) < 1e-12 and d == 0.0  # ties -> the smaller k
    k, _, d = along.project(along.cumulative(_bend()[:, 0:3]), np.array([-1.0, 0.0, 0.0]))
    assert k == 0.0 and abs(d - 1.0) < 1e-12
    pts = along.cumulative(_bend()[:, 0:3])
    k, p, _ = along.project(pts, pts[-1] + np.array([0.0, 5.0, 0.0]))
    assert k == 32.0 and np.allclose(p, pts[-1])


def test_time_scaled():
    a = _bend()
    a[:, 3:6] = np.linspace(-0.3, 0.3, 32)[:, None] * np.array([1.0, 0.5, -0.2])
    a[:, 6] = np.where(np.arange(32) < 7, -1.0, 1.0)
    one = along.time_scaled(a, 1.0)
    assert np.allclose(one, a[:10], atol=1e-12)
    half = along.time_scaled(a, 0.75)
    pts = along.cumulative(a[:, 0:3])
    assert np.allclose(half[:, 0:3].sum(0), along.at(pts, 7.5), atol=1e-12)
    rot = along.cumulative(a[:, 3:6])
    assert np.allclose(half[:, 3:6].sum(0), along.at(rot, 7.5), atol=1e-12)
    assert [half[i, 6] for i in range(10)] == [a[int(np.floor(0.75 * i + 1e-6)), 6] for i in range(10)]
    fast = along.time_scaled(a, 3.2)  # c up to 32 / 10: the gripper index stays inside the chunk
    assert np.allclose(fast[:, 0:3].sum(0), pts[32], atol=1e-12)


def _pair(seed: int, scale: float = 0.08):
    rng = np.random.default_rng(seed)
    a_d = (rng.standard_normal((32, 7)) * scale).astype(np.float32)
    a_s = (rng.standard_normal((32, 7)) * scale).astype(np.float32)
    a_d[:, 0:3] += np.float32(0.3)  # a path that moves, so |D| clears the small-displacement rule
    a_s[:, 0:3] += np.float32(0.25)
    a_d[:, 6] = np.sign(rng.standard_normal(32)).astype(np.float32)
    a_s[:, 6] = np.sign(rng.standard_normal(32)).astype(np.float32)
    return a_d, a_s


@pytest.mark.parametrize("seed", range(20))
def test_d2_partner_equal_to_dense(seed):
    a_d, _ = _pair(seed)
    for mode in ALONG + ("dir",):  # dir with a_r = a_d is the placebo
        out, info = mix(a_d, a_d.copy(), mode, eps_d=0.1, eps_theta=0.01)
        assert out.dtype == a_d.dtype and float(np.abs(out.astype(np.float64) - a_d).max()) < TOL, mode
        assert info.get("skip") in (None, False) and info.get("skip_t") in (None, False)


@pytest.mark.parametrize("seed", range(20))
def test_d3_endpoints(seed):
    a_d, a_s = _pair(seed)
    e_s = a_s[:10, 0:3].astype(np.float64).sum(0)
    out, info = mix(a_d, a_s, "prog", eps_d=0.1, eps_theta=0.01)
    assert info["clipped"] == 0 and info["end_to_path"] < TOL
    end = out[:10, 0:3].astype(np.float64).sum(0)
    assert along.project(along.cumulative(a_d[:, 0:3]), end)[2] < TOL
    assert np.array_equal(out[10:], a_d[10:])
    out, info = mix(a_d, a_s, "prog_cross", eps_d=0.1, eps_theta=0.01)
    assert info["clipped"] == 0 and info["end_minus_partner"] < TOL
    assert np.linalg.norm(out[:10, 0:3].astype(np.float64).sum(0) - e_s) < TOL
    out, info = mix(a_d, a_s, "cross", eps_d=0.1, eps_theta=0.01)
    m = along.measure(a_d, e_s)
    assert np.allclose(out[:10, 0:3].astype(np.float64).sum(0), a_d[:10, 0:3].astype(np.float64).sum(0) + m["delta"], atol=TOL)
    assert np.array_equal(out[:, 3:], a_d[:, 3:])
    out, _ = mix(a_d, a_s, "rot", eps_d=0.1, eps_theta=0.01)
    assert np.array_equal(out[:10, 3:6], a_s[:10, 3:6]) and np.array_equal(out[:, [0, 1, 2, 6]], a_d[:, [0, 1, 2, 6]])
    assert np.array_equal(out[10:], a_d[10:])


def test_small_displacement_and_clipping():
    a_d, a_s = _pair(0)
    for mode in ("prog", "cross", "prog_cross"):
        out, info = mix(a_d, a_s, mode, eps_d=1e3, eps_theta=0.01)
        assert info["skip"] and np.array_equal(out, a_d)
    out, info = mix(a_d, a_s, "rot", eps_d=0.1, eps_theta=1e3)
    assert info["skip"] and np.array_equal(out, a_d)
    big = a_s.copy()
    big[:10, 0:3] *= 4.0  # a sparse endpoint beyond the dense path's reach on the executed window
    out, info = mix(a_d, big, "prog_cross", eps_d=0.1, eps_theta=0.01)
    assert info["values"] == 60 and np.abs(out[:10, 0:6]).max() <= 1.0
