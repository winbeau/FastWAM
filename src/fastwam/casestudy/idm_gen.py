"""generality round (2026-11-09) on FastWAM-IDM: dense diagnostics run after the pass on its recorded states (``casestudy.gen``).

Stage 1 (video): F(s) (the clean estimate of step s against the last step's, change from the current frame) and, on step 1's state,
sparse forwards of the ``sets`` (key + log(Kf / n)) with explicit fp32 attention: kappa_1 against the explicit dense forward and the
late-layer shares of the computed tokens' attention on themselves and on their same-frame 3 x 3 neighbours (same camera block), over
every key and within the future keys. Stage 2 (action, against the dense clean cache): c(s) (step s's executed displacement scaled by
1 + eps about ``a0``) and c_grip(s) (the gripper channel + g), the action stage continued in fp32 (as main's quick_metrics.c_block).
Nothing feeds back into the pass.
"""

from __future__ import annotations

import base64
import math
from typing import Optional

import numpy as np
import torch
from einops import rearrange

from . import selection as sel


def _f32_b64(x: torch.Tensor) -> str:
    return base64.b64encode(x.detach().float().cpu().numpy().astype(np.float32).tobytes()).decode()


def _r6(x) -> Optional[float]:
    x = float(x)
    return round(x, 6) if math.isfinite(x) else None


def explicit_video(r, prep: tuple, kv0: list, rows: torch.Tensor, bias: float, grid: sel.Grid, f0: int, late: range) -> dict:
    """Stage-1 video stack over the future ``rows`` (frame-0 K/V prepended, every row key + ``bias``) with explicit fp32 softmax
    attention -> {vel [1, n, D], shares (late-layer means: self / near over every key and within the future keys)}."""
    xt, tt, tmodt, ctxt, ctxmt, freqst = prep
    dev = xt.device
    rows = rows.to(dev)
    n = int(rows.numel())
    idx = rows + f0
    x = xt[:, idx]
    freqs, tmod, ctxm = freqst[idx], tmodt[:, idx], ctxmt[:, idx]
    hn = r.num_heads
    col_bias = torch.zeros(f0 + n, device=dev)
    col_bias[f0:] = bias
    fr, rem = rows // grid.per_frame, rows % grid.per_frame
    hh, ww = rem // grid.width, rem % grid.width
    cam = ww >= grid.width // 2
    eye = rows[:, None] == rows[None]
    near = (fr[:, None] == fr[None]) & (cam[:, None] == cam[None]) & ((hh[:, None] - hh[None]).abs() <= 1) & (
        (ww[:, None] - ww[None]).abs() <= 1) & ~eye
    acc = {"self_all": [], "near_all": [], "self_fut": [], "near_fut": []}
    for layer in range(r.num_layers):
        block = r.video.blocks[layer]
        io = r.mot._build_expert_attention_io(expert=r.video, block=block, x=x, freqs=freqs, t_mod=tmod)
        k = torch.cat([kv0[layer][0], io[1]], dim=1)
        v = torch.cat([kv0[layer][1], io[2]], dim=1)
        qh = rearrange(io[0], "b s (h d) -> b h s d", h=hn)[0].float()
        kh = rearrange(k, "b s (h d) -> b h s d", h=hn)[0].float()
        vh = rearrange(v, "b s (h d) -> b h s d", h=hn)[0].float()
        p = torch.softmax((qh @ kh.transpose(-1, -2)) * (qh.shape[-1] ** -0.5) + col_bias, dim=-1)
        if layer in late:
            pk = p.mean(dim=0)[:, f0:]  # [n, n] head mean, over every key
            fut = pk.sum(dim=1).clamp_min(1e-12)
            se, ne = (pk * eye).sum(dim=1), (pk * near).sum(dim=1)
            acc["self_all"].append(float(se.mean()))
            acc["near_all"].append(float(ne.mean()))
            acc["self_fut"].append(float((se / fut).mean()))
            acc["near_fut"].append(float((ne / fut).mean()))
        out = rearrange((p @ vh)[None], "b h s d -> b s (h d)").to(x.dtype)
        x = r._post(block, io, out, ctxt, ctxm)
    vel = r.video.head(x, tt[:, idx])
    return {"vel": vel, "shares": {k_: _r6(sum(v_) / len(v_)) if v_ else None for k_, v_ in acc.items()}}


def _continue(r, rec2: list, s: int, xh: torch.Tensor, sig: float, cache: list, context, context_mask) -> torch.Tensor:
    """The fp32 action latents after the last stage-2 step when step ``s``'s action clean estimate is ``xh``."""
    xa = rec2[s]["lat_a"].float()
    x = xa + rec2[s]["d_a"].float() * (xa - xh) / sig
    for s2 in range(s + 1, len(rec2)):
        pred, _ = r._action_layers(x.to(rec2[s2]["lat_a"].dtype), rec2[s2]["ts_a"], context, context_mask, cache, set())
        x = x + rec2[s2]["d_a"].float() * pred.float()
    return x


def run(r, rec1: list, prep1: tuple, rec2: list, kv0: list, cache: list, grid: sel.Grid, f0: int, a0: Optional[torch.Tensor],
        context, context_mask, patch: tuple) -> dict:
    g_cfg = r.cfg.gen
    eps = tuple(float(x) for x in g_cfg.get("eps", (0.05, -0.05)))
    grip_eps = tuple(float(x) for x in g_cfg.get("grip_eps", (0.2, -0.2)))
    sets = tuple(str(x) for x in g_cfg.get("sets", ("strat84", "f1s84")))
    lo, hi = (int(x) for x in g_cfg.get("late_layers", (20, 30)))
    n_exec = int(g_cfg.get("executed", 10))
    pt, ph, pw = patch
    kf, pf, half = grid.num_future, grid.per_frame, grid.width // 2
    dev = rec1[0]["lat_v"].device
    S = len(rec1)

    def tok(lat):
        return rearrange(lat, "b c (f x) (h y) (w z) -> b (f h w) (x y z c)", x=pt, y=ph, z=pw)[0].float()

    allf = torch.arange(kf, device=dev)
    frame, col = allf // pf, (allf % pf) % grid.width
    wrist = col >= half
    groups = {"all": torch.ones(kf, dtype=torch.bool, device=dev), "near": frame == 0, "far": frame == 1, "agent": ~wrist,
              "wrist": wrist, "near_agent": (frame == 0) & ~wrist, "near_wrist": (frame == 0) & wrist,
              "far_agent": (frame == 1) & ~wrist, "far_wrist": (frame == 1) & wrist}
    nts = float(r.model.infer_video_scheduler.num_train_timesteps)
    z0 = tok(rec1[0]["lat_v"])[:f0][allf % pf]
    sig_v = [float(rec1[s]["ts_v"].float() / nts) for s in range(S)]
    dd = [tok(rec1[s]["lat_v"])[f0:] - sig_v[s] * rec1[s]["vel"][0, f0:].float() - z0 for s in range(S)]
    den = (dd[-1] ** 2).sum(dim=-1)
    F = {g: [_r6(((dd[s] * dd[-1]).sum(dim=-1)[m].sum()) / den[m].sum()) for s in range(S)] for g, m in groups.items()}

    # step 1: the flash forward over every token without bias equals the pass bit for bit (self-check); explicit dense + sets
    x1 = tok(rec1[1]["lat_v"])[f0:]
    xt, tt, tmodt, ctxt, ctxmt, freqst = prep1
    xf, _ = r._video_layers(xt[:, allf + f0], freqst[allf + f0], tmodt[:, allf + f0], ctxt, ctxmt[:, allf + f0], kv0, keep_kv=False)
    flash_equal = bool(torch.equal(r.video.head(xf, tt[:, allf + f0]), rec1[1]["vel"][:, f0:]))
    late = range(lo, hi)
    dense = explicit_video(r, prep1, kv0, allf, 0.0, grid, f0, late)
    y = x1 - sig_v[1] * dense["vel"][0].float()
    main = x1 - sig_v[1] * rec1[1]["vel"][0, f0:].float()
    step1 = {"dense": {"shares": dense["shares"], "explicit_vs_flash_rel": _r6((y - main).norm() / main.norm())}}
    for spec in sets:
        rows = r._late_rows(spec, grid).to(dev)
        res = explicit_video(r, prep1, kv0, rows, math.log(kf / int(rows.numel())), grid, f0, late)
        xs = x1[rows] - sig_v[1] * res["vel"][0].float()
        dy = y[rows] - z0[rows]
        step1[spec] = {"n": int(rows.numel()), "shares": res["shares"],
                       "kappa": _r6(((xs - z0[rows]) * dy).sum() / (dy * dy).sum()),
                       "e_x": _r6((xs - y[rows]).norm() / y[rows].norm())}

    # stage 2: c(s) / c_grip(s)
    nta = float(r.model.infer_action_scheduler.num_train_timesteps)
    sig_a = [float(rec2[s]["ts_a"].float() / nta) for s in range(len(rec2))]
    actions, c_grip = {}, {f"{g:g}": [] for g in grip_eps}
    for s in range(len(rec2)):
        xa = rec2[s]["lat_a"].float()
        x0s = xa - sig_a[s] * rec2[s]["pred"].float()
        actions[f"d@{s}"] = x0s[0]
        base = _continue(r, rec2, s, x0s, sig_a[s], cache, context, context_mask)
        if a0 is not None:
            a0d = a0.to(xa.device)
            actions[f"c0@{s}"] = base[0]
            for e in eps:
                xh = x0s.clone()
                xh[0, :n_exec, :3] = a0d + (1.0 + e) * (x0s[0, :n_exec, :3] - a0d)
                actions[f"c{e:g}@{s}"] = _continue(r, rec2, s, xh, sig_a[s], cache, context, context_mask)[0]
        for g in grip_eps:
            xh = x0s.clone()
            xh[0, :n_exec, -1] = x0s[0, :n_exec, -1] + g
            fin = _continue(r, rec2, s, xh, sig_a[s], cache, context, context_mask)
            c_grip[f"{g:g}"].append(_r6((fin[0, :n_exec, -1] - base[0, :n_exec, -1]).mean() / g))
    return {"F": F, "sigma_v": [_r6(x) for x in sig_v], "sigma_a": [_r6(x) for x in sig_a], "step1": step1, "late_layers": [lo, hi],
            "flash_all_equal": flash_equal, "n_exec": n_exec, "eps": list(eps), "grip_eps": list(grip_eps), "c_grip": c_grip,
            "c_skipped": a0 is None, "actions": {k: _f32_b64(v) for k, v in actions.items()}}
