"""CPU checks of the component study's hybrid execution (``fastwam/casestudy/hybrid.py``)."""

import numpy as np
import pytest

from fastwam.casestudy.hybrid import min_rotation, mix

TOL = 1e-6


def _pair(seed: int, scale: float = 0.08):
    rng = np.random.default_rng(seed)
    a_d = (rng.standard_normal((32, 7)) * scale).astype(np.float32)
    a_s = (rng.standard_normal((32, 7)) * scale).astype(np.float32)
    a_d[:, 6] = np.sign(rng.standard_normal(32)).astype(np.float32)
    a_s[:, 6] = np.sign(rng.standard_normal(32)).astype(np.float32)
    return a_d, a_s


def _sums(a):
    h = np.asarray(a, dtype=np.float64)[:10]
    return h[:, 0:3].sum(0), h[:, 3:6].sum(0)


def _cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def test_min_rotation():
    rng = np.random.default_rng(0)
    for _ in range(100):
        u, v = rng.standard_normal(3), rng.standard_normal(3)
        r = min_rotation(u, v)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1.0)
        assert np.allclose(r @ (u / np.linalg.norm(u)), v / np.linalg.norm(v), atol=1e-12)
        axis = np.cross(u, v)
        assert np.allclose(r @ axis, axis, atol=1e-12)  # the smallest rotation keeps u x v fixed
    u = np.array([0.3, -0.2, 0.5])
    r = min_rotation(u, -2 * u)  # antiparallel: a turn by pi
    assert np.allclose(r @ u, -u, atol=1e-12) and np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(min_rotation(u, 3 * u), np.eye(3))


@pytest.mark.parametrize("seed", range(20))
def test_components(seed):
    a_d, a_s = _pair(seed)
    (dd, td), (ds, ts) = _sums(a_d), _sums(a_s)
    h, info = mix(a_d, a_s, "dist", 0.0, 0.0)
    (dh, th) = _sums(h)
    assert info["clipped"] == 0
    assert abs(_cos(dh, dd) - 1) < TOL and abs(np.linalg.norm(dh) - np.linalg.norm(ds)) < TOL
    assert abs(_cos(th, td) - 1) < TOL and abs(np.linalg.norm(th) - np.linalg.norm(ts)) < TOL
    assert h[:, 6].tobytes() == a_d[:, 6].tobytes()
    h, _ = mix(a_d, a_s, "dir", 0.0, 0.0)
    (dh, th) = _sums(h)
    assert abs(np.linalg.norm(dh) - np.linalg.norm(dd)) < TOL and abs(_cos(dh, ds) - 1) < TOL
    assert abs(np.linalg.norm(th) - np.linalg.norm(td)) < TOL and abs(_cos(th, ts) - 1) < TOL
    assert h[:, 6].tobytes() == a_d[:, 6].tobytes()
    h, _ = mix(a_d, a_s, "grip", 0.0, 0.0)
    assert h[:, :6].tobytes() == a_d[:, :6].tobytes()
    assert h[:10, 6].tobytes() == a_s[:10, 6].tobytes() and h[10:].tobytes() == a_d[10:].tobytes()
    h, _ = mix(a_d, a_s, "all3", 0.0, 0.0)
    (dh, th) = _sums(h)
    assert np.allclose(dh, ds, atol=TOL) and np.allclose(th, ts, atol=TOL)
    assert h[:10, 6].tobytes() == a_s[:10, 6].tobytes()
    for mode in ("dist", "dir", "grip", "all3", "dist_grip"):
        h, _ = mix(a_d, a_s, mode, 0.0, 0.0)
        assert h[10:].tobytes() == a_d[10:].tobytes()  # only the executed steps change
    assert mix(a_d, a_s, "none", 0.0, 0.0)[0].tobytes() == a_d.tobytes()
    assert mix(a_d, a_s, "s0", 0.0, 0.0)[0].tobytes() == a_s.tobytes()


def test_small_displacement_and_clipping():
    a_d, a_s = _pair(3)
    (dd, td), (ds, ts) = _sums(a_d), _sums(a_s)
    eps_d = max(np.linalg.norm(dd), np.linalg.norm(ds)) + 1e-3  # translation below the threshold, rotation not
    h, info = mix(a_d, a_s, "all3", eps_d, 0.0)
    assert info["skip_t"] and not info["skip_r"]
    assert h[:, 0:3].tobytes() == a_d[:, 0:3].tobytes()
    assert np.allclose(_sums(h)[1], ts, atol=TOL)
    big_d, big_s = a_d.copy(), a_s.copy()
    big_d[0, 0], big_d[1:10, 0], big_s[:10, 0] = 0.9, 0.1, 0.95  # |D_s| / |D_d| ~ 5: step 0 leaves [-1, 1]
    h, info = mix(big_d, big_s, "dist", 0.0, 0.0)
    assert info["clipped"] > 0 and np.abs(h[:10, 0:3]).max() <= 1.0
    with pytest.raises(ValueError):
        mix(a_d, a_s, "bogus", 0.0, 0.0)
