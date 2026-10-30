"""Hybrid execution of the component study: the dense chunk with one component of the sparse (S0) chunk swapped in.

Both chunks are in simulator units with the gripper binarized ([T, 7]: translation = dims 0-2, rotation = dims 3-5,
gripper = dim 6). Only the executed steps (the first ``executed``) change; D / Theta are the translation / rotation sums
over them. Modes (``policy.options.hybrid.mode`` of the adapter):

``none``       the dense chunk (the sparse one is computed and logged only).
``s0``         the whole sparse chunk.
``dist``       dense translation x |D_s| / |D_d| and rotation x |Theta_s| / |Theta_d|, dense gripper.
``dir``        dense translation turned by the smallest rotation taking D_d to D_s (Rodrigues), rotation likewise
               with Theta, dense gripper.
``grip``       dense dims 0-5, sparse gripper.
``all3``       dist, dir and grip together: before clipping D_h = D_s and Theta_h = Theta_s.
``dist_grip``  dist and grip together.

A component is left as it is on a call where |D_d| or |D_s| is below ``eps_d`` (rotation: ``eps_theta``). Replaced
columns are clipped to [-limit, limit]; columns that are not replaced are never written.
"""

from __future__ import annotations

import numpy as np

MODES = ("none", "s0", "dist", "dir", "grip", "all3", "dist_grip")
TRANS, ROT, GRIP = slice(0, 3), slice(3, 6), 6


def min_rotation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Smallest rotation matrix taking the direction of ``u`` to that of ``v`` (both non-zero, float64). Antiparallel
    vectors: a turn by pi about an axis perpendicular to ``u``."""
    u = np.asarray(u, dtype=np.float64) / np.linalg.norm(u)
    v = np.asarray(v, dtype=np.float64) / np.linalg.norm(v)
    c = float(u @ v)
    axis = np.cross(u, v)
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        n = np.cross(u, np.eye(3)[int(np.argmin(np.abs(u)))])
        n /= np.linalg.norm(n)
        return 2.0 * np.outer(n, n) - np.eye(3)
    k = axis / s
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)


def mix(a_d: np.ndarray, a_s: np.ndarray, mode: str, eps_d: float, eps_theta: float,
        executed: int = 10, limit: float = 1.0) -> tuple[np.ndarray, dict]:
    """The executed chunk for ``mode`` from the dense ``a_d`` and sparse ``a_s`` chunks -> (chunk, log)."""
    if mode not in MODES:
        raise ValueError(f"hybrid mode must be one of {MODES}, got {mode!r}")
    if mode == "none":
        return np.array(a_d, copy=True), {"mode": mode}
    if mode == "s0":
        return np.array(a_s, copy=True), {"mode": mode}
    n = min(int(executed), len(a_d))
    d = np.asarray(a_d[:n], dtype=np.float64)
    s = np.asarray(a_s[:n], dtype=np.float64)
    out = np.array(a_d, copy=True)
    scale = mode in ("dist", "all3", "dist_grip")
    turn = mode in ("dir", "all3")
    info: dict = {"mode": mode, "clipped": 0, "values": 0}
    for key, sl, eps in (("t", TRANS, eps_d), ("r", ROT, eps_theta)):
        vd, vs = d[:, sl].sum(axis=0), s[:, sl].sum(axis=0)
        nd, ns = float(np.linalg.norm(vd)), float(np.linalg.norm(vs))
        skip = nd < eps or ns < eps or nd == 0.0 or ns == 0.0
        info.update({f"norm_d_{key}": nd, f"norm_s_{key}": ns, f"skip_{key}": skip})
        if skip or not (scale or turn):
            continue
        x = d[:, sl]
        if turn:
            x = x @ min_rotation(vd, vs).T
        if scale:
            x = x * (ns / nd)
        info["clipped"] += int(np.sum(np.abs(x) > limit))
        info["values"] += int(x.size)
        out[:n, sl] = np.clip(x, -limit, limit).astype(out.dtype)
    if mode in ("grip", "all3", "dist_grip"):
        out[:n, GRIP] = a_s[:n, GRIP]
    return out, info


__all__ = ["MODES", "min_rotation", "mix"]
