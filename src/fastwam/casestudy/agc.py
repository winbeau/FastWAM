"""Pilot + automatic gain control (``casestudy.agc``) of an output chunk's amplitude.

The dense warm-up step's clean action estimate (the pilot, ``x - sigma_0 * v_0``, logged by the
runner) has not been through any pruned step. Its displacement over the executed steps, against the
arm's own, says how much the arm's output was shrunk; the chunk's translation and rotation are scaled
back by that factor. Everything here works on chunks in simulator units (``policy_io.postprocess_action``,
[T, 7]): translation = dims 0-2, rotation = dims 3-5, gripper = dim 6, which is never touched.
"""

from __future__ import annotations

import numpy as np

TRANS, ROT = slice(0, 3), slice(3, 6)


def displacements(chunk: np.ndarray, executed: int) -> tuple[float, float]:
    """|D|, |R|: norms of the translation and rotation sums over the first ``executed`` steps (float64)."""
    head = np.asarray(chunk, dtype=np.float64)[:executed]
    return float(np.linalg.norm(head[:, TRANS].sum(axis=0))), float(np.linalg.norm(head[:, ROT].sum(axis=0)))


def gains(cfg, out: np.ndarray, pilot: np.ndarray | None = None) -> dict:
    """Gains of one chunk ``out`` (the arm's output) from ``cfg`` (``AGCConfig``) and the pilot chunk.

    ``pilot``: per component g = clip(c * |pilot| / |out|, lo, hi), and g = 1 ("skip") when either norm
    is below the component's threshold (tau_d for translation, tau_r for rotation) or |out| is 0.
    ``const``: g_t = g_r = clip(gain, lo, hi), never skipped. Returns the gains, the unclipped ratios
    and the norms."""
    d_out, r_out = displacements(out, cfg.executed)
    info: dict = {"mode": cfg.mode, "d_out": d_out, "r_out": r_out}
    if cfg.mode == "const":
        g = min(max(cfg.gain, cfg.lo), cfg.hi)
        for key in ("t", "r"):
            info.update({f"g_{key}": g, f"raw_{key}": cfg.gain, f"skip_{key}": False})
        return info
    if pilot is None:
        raise ValueError("agc mode pilot needs the pilot chunk (record it with the runner)")
    d_p, r_p = displacements(pilot, cfg.executed)
    info.update(d_pilot=d_p, r_pilot=r_p)
    for key, c, tau, n_p, n_o in (("t", cfg.c_t, cfg.tau_d, d_p, d_out), ("r", cfg.c_r, cfg.tau_r, r_p, r_out)):
        skip = n_p < tau or n_o < tau or n_o == 0.0
        raw = None if skip else c * n_p / n_o
        info.update({f"g_{key}": 1.0 if skip else min(max(raw, cfg.lo), cfg.hi), f"raw_{key}": raw, f"skip_{key}": skip})
    return info


def apply(chunk: np.ndarray, g_t: float, g_r: float, limit: float) -> np.ndarray:
    """Copy of ``chunk`` with translation x g_t and rotation x g_r on every step, each clipped to
    [-limit, limit]. A component with gain exactly 1 is left as it is (no clipping either), so
    g_t = g_r = 1 returns the chunk unchanged bit for bit; the gripper column is never written."""
    out = np.array(chunk, copy=True)
    for sl, g in ((TRANS, g_t), (ROT, g_r)):
        if g != 1.0:
            out[:, sl] = np.clip(out[:, sl] * g, -limit, limit)
    return out


__all__ = ["apply", "displacements", "gains"]
