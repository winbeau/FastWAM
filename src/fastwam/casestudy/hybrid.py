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

along-cross round (``along.py``; c = k* / executed, k* the projection of the sparse endpoint E_s on the dense path P):

``prog``        the dense plan time-scaled by c: translation and rotation steps from the interpolated partial sums at
               c (i + 1) and c i, gripper a_d[min(T - 1, floor(c i + 1e-6)), 6]; the endpoint lies on P.
``cross``       dense translation + delta / executed (delta = E_s - P(k*)), dense rotation and gripper.
``prog_cross``  prog with delta / executed added to the translation: before clipping the endpoint is E_s.
``rot``         dense translation and gripper, sparse rotation.

prog / cross / prog_cross leave the whole call unchanged when |D_d| or |D_s| is below ``eps_d``; rot when |Theta_d| or
|Theta_s| is below ``eps_theta``.

A component is left as it is on a call where |D_d| or |D_s| is below ``eps_d`` (rotation: ``eps_theta``). Replaced
columns are clipped to [-limit, limit]; columns that are not replaced are never written.
"""

from __future__ import annotations

import numpy as np

from . import along

ALONG = ("prog", "cross", "prog_cross", "rot")
MODES = ("none", "s0", "dist", "dir", "grip", "all3", "dist_grip") + ALONG
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
    if mode in ALONG:
        return mix_along(a_d, a_s, mode, eps_d, eps_theta, executed, limit)
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


def mix_along(a_d: np.ndarray, a_s: np.ndarray, mode: str, eps_d: float, eps_theta: float,
              executed: int = 10, limit: float = 1.0) -> tuple[np.ndarray, dict]:
    """The along-cross modes (``prog``, ``cross``, ``prog_cross``, ``rot``) -> (chunk, log). The log has the sparse endpoint's
    projection (``partner``) and, before clipping, the hybrid endpoint's distance to P (prog) or to E_s (prog_cross)."""
    n = min(int(executed), len(a_d))
    d = np.asarray(a_d, dtype=np.float64)
    s = np.asarray(a_s[:n], dtype=np.float64)
    out = np.array(a_d, copy=True)
    vd, vs = d[:n, TRANS].sum(axis=0), s[:, TRANS].sum(axis=0)
    rd, rs = d[:n, ROT].sum(axis=0), s[:, ROT].sum(axis=0)
    nd, ns, nrd, nrs = (float(np.linalg.norm(x)) for x in (vd, vs, rd, rs))
    m = along.measure(d, vs, n)
    info: dict = {"mode": mode, "clipped": 0, "values": 0, "norm_d_t": nd, "norm_s_t": ns, "norm_d_r": nrd, "norm_s_r": nrs,
                  "partner": {k: m[k] for k in ("k_star", "r_along", "r_old", "e", "theta_total", "theta_phase", "theta_resid")}}
    if mode == "rot":
        info["skip"] = nrd < eps_theta or nrs < eps_theta or nrd == 0.0 or nrs == 0.0
        if not info["skip"]:
            x = s[:, ROT]
            info["clipped"], info["values"] = int(np.sum(np.abs(x) > limit)), int(x.size)
            out[:n, ROT] = np.clip(x, -limit, limit).astype(out.dtype)
        return out, info
    info["skip"] = nd < eps_d or ns < eps_d or nd == 0.0 or ns == 0.0
    if info["skip"]:
        return out, info
    c = m["k_star"] / n
    info["c"] = c
    if mode in ("prog", "prog_cross"):
        ts = along.time_scaled(d, c, n)
        xt, xr = ts[:, TRANS], ts[:, ROT]
    else:
        xt, xr = d[:n, TRANS], None
    if mode in ("cross", "prog_cross"):
        xt = xt + np.asarray(m["delta"]) / n
    end = xt.sum(axis=0)
    if mode == "prog":
        info["end_to_path"] = along.project(along.cumulative(d[:, TRANS]), end)[2]
    elif mode == "prog_cross":
        info["end_minus_partner"] = float(np.linalg.norm(end - vs))
    for sl, x in ((TRANS, xt), (ROT, xr)):
        if x is None:
            continue
        info["clipped"] += int(np.sum(np.abs(x) > limit))
        info["values"] += int(x.size)
        out[:n, sl] = np.clip(x, -limit, limit).astype(out.dtype)
    if mode in ("prog", "prog_cross"):
        out[:n, GRIP] = ts[:, GRIP].astype(out.dtype)
    return out, info


__all__ = ["ALONG", "MODES", "min_rotation", "mix", "mix_along"]
