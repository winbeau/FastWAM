"""along-cross round: a chunk's endpoint against the dense chunk's path ("going slow" vs "going off the path").

Everything is in simulator units (dims 0-2 translation, 3-5 rotation, 6 gripper), float64. With the dense chunk a_d [T, 7]:

  P(k) = sum_{i<k} a_d[i, 0:3], k in [0, T], linear between integers;  s(k) = arc length of P on [0, k];  D_d = P(10).

An endpoint E (the first ``executed`` steps of another chunk) is projected on the polyline: P(k*) is the nearest point,
ties going to the smaller k. Then r_along = s(k*) / s(10), delta = E - P(k*), e = |delta|, and the angles (degrees)
theta_total = angle(E, D_d), theta_phase = angle(P(k*), D_d), theta_resid = angle(E, P(k*)).

Pure numpy, no package imports: the action-eval analysis imports this file directly.
"""

from __future__ import annotations

import numpy as np

TIE = 1e-12  # a later segment must be nearer by more than this to win (ties -> the smaller k)
GRIP_EPS = 1e-6  # floor(c * i + GRIP_EPS) picks the dense gripper step of a time-scaled step


def cumulative(x: np.ndarray) -> np.ndarray:
    """[T, d] steps -> [T + 1, d] partial sums, row k = sum of the first k steps."""
    x = np.asarray(x, dtype=np.float64)
    return np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)], axis=0)


def at(points: np.ndarray, k: float) -> np.ndarray:
    """The polyline through ``points`` [T + 1, d] at parameter k in [0, T], linear between integers."""
    n = points.shape[0] - 1
    k = min(max(float(k), 0.0), float(n))
    i = min(int(np.floor(k)), n - 1)
    t = k - i
    return points[i] + t * (points[i + 1] - points[i])


def arc(points: np.ndarray) -> np.ndarray:
    """Arc length at the integer parameters [T + 1]."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def arc_at(lengths: np.ndarray, k: float) -> float:
    n = lengths.shape[0] - 1
    k = min(max(float(k), 0.0), float(n))
    i = min(int(np.floor(k)), n - 1)
    return float(lengths[i] + (k - i) * (lengths[i + 1] - lengths[i]))


def project(points: np.ndarray, e: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Nearest point of the polyline to ``e`` -> (k*, P(k*), distance); segments scanned by increasing k, a later one wins
    only when nearer by more than TIE."""
    e = np.asarray(e, dtype=np.float64)
    best = (0.0, points[0].copy(), float(np.linalg.norm(e - points[0])))
    for i in range(points.shape[0] - 1):
        a, seg = points[i], points[i + 1] - points[i]
        ll = float(seg @ seg)
        t = 0.0 if ll == 0.0 else min(max(float((e - a) @ seg) / ll, 0.0), 1.0)
        p = a + t * seg
        d = float(np.linalg.norm(e - p))
        if d < best[2] - TIE:
            best = (i + t, p, d)
    return best


def angle(u: np.ndarray, v: np.ndarray) -> float | None:
    """Angle between two 3-vectors in degrees (atan2 of |u x v| and u . v: accurate near 0 and 180); None when either is
    zero."""
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    if not np.any(u) or not np.any(v):
        return None
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), float(u @ v))))


def measure(a_d: np.ndarray, endpoint: np.ndarray, executed: int = 10) -> dict:
    """Projection of ``endpoint`` (a translation sum) on the path of the dense chunk ``a_d`` [T, >=3]."""
    pts = cumulative(np.asarray(a_d, dtype=np.float64)[:, 0:3])
    lengths = arc(pts)
    d_d = pts[executed]
    k, p, e = project(pts, endpoint)
    s10 = float(lengths[executed])
    endpoint = np.asarray(endpoint, dtype=np.float64)
    return {"k_star": k, "r_along": arc_at(lengths, k) / s10 if s10 > 0 else None,
            "r_old": float(np.linalg.norm(endpoint) / np.linalg.norm(d_d)) if np.linalg.norm(d_d) > 0 else None,
            "e": e, "e_rel": e / s10 if s10 > 0 else None, "delta": (endpoint - p).tolist(), "s10": s10,
            "norm_d": float(np.linalg.norm(d_d)), "theta_total": angle(endpoint, d_d), "theta_phase": angle(p, d_d),
            "theta_resid": angle(endpoint, p)}


def psi(a_d: np.ndarray, executed: int = 10) -> float | None:
    """Turn of the dense chunk over the executed steps: angle between the first and second half displacements."""
    x = np.asarray(a_d, dtype=np.float64)[:executed, 0:3]
    h = executed // 2
    return angle(x[:h].sum(axis=0), x[h:].sum(axis=0))


def time_scaled(a_d: np.ndarray, c: float, executed: int = 10) -> np.ndarray:
    """The dense plan played at speed c for ``executed`` steps [executed, 7]: translation and rotation steps are differences
    of the interpolated partial sums at c (i + 1) and c i, the gripper is a_d[min(T - 1, floor(c i + GRIP_EPS)), 6]."""
    a = np.asarray(a_d, dtype=np.float64)
    n = a.shape[0]
    out = np.zeros((executed, a.shape[1]))
    for sl in (slice(0, 3), slice(3, 6)):
        pts = cumulative(a[:, sl])
        for i in range(executed):
            out[i, sl] = at(pts, c * (i + 1)) - at(pts, c * i)
    for i in range(executed):
        out[i, 6] = a[min(n - 1, int(np.floor(c * i + GRIP_EPS))), 6]
    return out


__all__ = ["angle", "arc", "arc_at", "at", "cumulative", "measure", "project", "psi", "time_scaled"]
