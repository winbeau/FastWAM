"""Quick-metrics round (2026-11-09): dense diagnostics beside a finished dense pass.

A dense pass records its per-step states (``run_call``'s ``rec`` and ``kv_cap``); ``run_call``
then runs, on those states and never feeding back into the pass: F(s) (shape of the dense video
clean estimate against the last step's), kappa_s (single-step V->V shrinkage of a 39-token sparse
forward against the dense forward on the same state), the action's single-step read variants
(a) sparse forward, (b) the 39 tokens read from the dense K/V, (c) everything, with future
shares; c(s) (the executed displacement of step s's action clean estimate scaled by 1 + eps, the
dense action stream continued to the end in fp32); at the last step the per-token contribution
g_j, the attention and the content ratio rho of three 39-token reads.

Every function takes the runner as its first argument and is meant to run under ``no_grad``.
"""

from __future__ import annotations

import base64
import math
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange

from fastwam.models.wan22.wan_video_dit import flash_attention

from . import fovea as fv
from . import selection as sel


def _f32_b64(x: torch.Tensor) -> str:
    return base64.b64encode(x.detach().float().cpu().numpy().astype(np.float32).tobytes()).decode()


def _r6(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return None
    return round(fx, 6) if math.isfinite(fx) else None  # JSON: no NaN


def sparse_forward(
    r,
    pre_v: dict,
    pre_a: dict,
    kv0: list,
    rows: torch.Tensor,
    bias: Optional[torch.Tensor],
    f0: int,
    score_layer: Optional[int] = None,
) -> dict:
    """One joint step with only the future rows ``rows`` as video rows.

    rows : LongTensor of future indices. bias : [n] fp32, or None.
    Returns {"vel", "pred", "share", "recv", "av"} as described in quick_metrics.py.
    """
    dev = pre_v["tokens"].device
    rows = rows.to(dev)
    n = int(rows.numel())
    n_act = int(pre_a["tokens"].shape[1])
    xv = pre_v["tokens"][:, rows + f0]
    xa = pre_a["tokens"]

    share: list[float] = []
    recv = torch.zeros(n, device=dev) if score_layer is not None else None
    av = None

    mask_v = mask_a = None
    if bias is not None and bool((bias != 0).any()):
        b = bias.to(device=dev)
        mask_v = torch.zeros((1, 1, 1, f0 + n), device=dev)
        mask_v[..., f0:] = b
        mask_a = torch.zeros((1, 1, 1, f0 + n + n_act), device=dev)
        mask_a[..., f0:f0 + n] = b
        mask_v, mask_a = mask_v.to(xv.dtype), mask_a.to(xv.dtype)
    for layer in range(r.num_layers):
        bv, ba = r.video.blocks[layer], r.action.blocks[layer]
        io_v = r.mot._build_expert_attention_io(
            expert=r.video, block=bv, x=xv, freqs=pre_v["freqs"][rows + f0],
            t_mod=pre_v["t_mod"][:, rows + f0]
        )
        io_a = r.mot._build_expert_attention_io(
            expert=r.action, block=ba, x=xa, freqs=pre_a["freqs"], t_mod=pre_a["t_mod"]
        )
        k_vid = torch.cat([kv0[layer]["k"], io_v[1]], dim=1)
        v_vid = torch.cat([kv0[layer]["v"], io_v[2]], dim=1)
        k_all = torch.cat([k_vid, io_a[1]], dim=1)
        v_all = torch.cat([v_vid, io_a[2]], dim=1)

        out_v = flash_attention(q=io_v[0], k=k_vid, v=v_vid, num_heads=r.num_heads, ctx_mask=mask_v)
        out_a = flash_attention(q=io_a[0], k=k_all, v=v_all, num_heads=r.num_heads, ctx_mask=mask_a)

        pa = r._masked_probs(io_a[0], k_all, mask_a)  # [A, K]
        share.append(float(pa[:, f0:f0 + n].sum(dim=-1).mean()))

        if score_layer is not None:
            recv += r._masked_probs(io_v[0], k_vid, mask_v)[:, f0:].sum(dim=0)
            recv += pa[:, f0:f0 + n].sum(dim=0)
            if layer == score_layer:
                av = pa[:, f0:f0 + n].sum(dim=0)
                av = av / av.sum()

        xv = r._post(bv, io_v, out_v, {"context": pre_v["context"], "mask": pre_v["context_mask"][:, rows + f0]})
        xa = r._post(ba, io_a, out_a, {"context": pre_a["context"], "mask": pre_a["context_mask"]})

    return {
        "vel": r.video.head(xv, pre_v["t"][:, rows + f0]),
        "pred": r.action.post_dit(xa, pre_a),
        "share": share,
        "recv": recv,
        "av": av,
    }


def action_forward(
    r,
    pre_a: dict,
    kv0: list,
    kv_fut: list,
    cols: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    f0: int,
    capture: bool = False,
    share: bool = True,
) -> dict:
    """One action stream step with future keys taken from ``kv_fut`` for this step.

    kv_fut : list, per layer (k, v) tensors [1, Kf, D].
    cols None: every future column in order; otherwise columns are selected.
    bias : [len(cols)] fp32 or None.
    """
    dev = pre_a["tokens"].device
    xa = pre_a["tokens"]
    n_act = int(xa.shape[1])

    shares: list[float] = []
    captured: list = []
    mask = None

    for layer in range(r.num_layers):
        ba = r.action.blocks[layer]
        io_a = r.mot._build_expert_attention_io(
            expert=r.action, block=ba, x=xa, freqs=pre_a["freqs"], t_mod=pre_a["t_mod"]
        )
        k_f, v_f = kv_fut[layer]
        if cols is not None:
            k_f = k_f[:, cols]
            v_f = v_f[:, cols]
        n = int(k_f.shape[1])

        if bias is not None and bool((bias != 0).any()):
            b = bias.to(device=dev)
            mask = torch.zeros((1, 1, 1, f0 + n + n_act), device=dev)
            mask[..., f0:f0 + n] = b
            mask = mask.to(pre_a["tokens"].dtype)
        else:
            mask = None

        keys = torch.cat([kv0[layer]["k"], k_f, io_a[1]], dim=1)
        values = torch.cat([kv0[layer]["v"], v_f, io_a[2]], dim=1)
        out_a = flash_attention(q=io_a[0], k=keys, v=values, num_heads=r.num_heads, ctx_mask=mask)

        if share:
            pa = r._masked_probs(io_a[0], keys, mask)
            shares.append(float(pa[:, f0:f0 + n].sum(dim=-1).mean()))
        if capture:
            captured.append((io_a[0], io_a[1], io_a[2]))

        xa = r._post(ba, io_a, out_a, {"context": pre_a["context"], "mask": pre_a["context_mask"]})

    return {"pred": r.action.post_dit(xa, pre_a), "share": shares, "capture": captured}


def readout(logits: torch.Tensor, vh: torch.Tensor, f0: int, kf: int, reads: dict, cams: dict) -> dict:
    """One layer's action read statistics at the last step.

    logits : [H, A, f0 + kf + A] fp32. vh : [H, f0 + kf + A, Dh] fp32.
    """
    p = torch.softmax(logits, dim=-1)
    pf = p[..., f0:f0 + kf]
    m_d = pf.sum(dim=-1)
    p_present = torch.cat([p[..., :f0], p[..., f0 + kf:]], dim=-1)
    v_present = torch.cat([vh[:, :f0], vh[:, f0 + kf:]], dim=1)
    denom_present = (1.0 - m_d).clamp_min(1e-30)[..., None]
    cbar = (p_present @ v_present) / denom_present

    vf = vh[:, f0:f0 + kf]
    od = pf @ vf - m_d[..., None] * cbar
    od_norm = od.norm(dim=-1).clamp_min(1e-12)
    u = od / od_norm[..., None]

    vf_dot_u = torch.einsum("hkd,had->hak", vf, u)  # [H, A, kf]
    g = (pf * (vf_dot_u - (cbar * u).sum(dim=-1)[..., None])).sum(dim=0).mean(dim=0)  # [kf]
    att = pf.mean(dim=0).mean(dim=0)  # [kf]

    out_reads = {}
    for name, (cols, bias) in reads.items():
        lg_f = logits[..., f0:f0 + kf].index_select(-1, cols) + bias
        lg = torch.cat([logits[..., :f0], lg_f, logits[..., f0 + kf:]], dim=-1)
        ps = torch.softmax(lg, dim=-1)
        ps_f = ps[..., f0:f0 + int(cols.numel())]
        vs = vf[:, cols]
        out_reads[name] = {}
        for scope, cam_mask in cams.items():
            sel_ = cam_mask[cols]
            ms = ps_f[..., sel_].sum(dim=-1)
            oS = ps_f[..., sel_] @ vs[:, sel_] - ms[..., None] * cbar
            mD = pf[..., cam_mask].sum(dim=-1)
            oD = pf[..., cam_mask] @ vf[:, cam_mask] - mD[..., None] * cbar

            oS_dot_oD = (oS * oD).sum(dim=-1)
            oD_dot_oD = (oD * oD).sum(dim=-1)
            ro = float(oS_dot_oD.sum())
            oo = float(oD_dot_oD.sum())
            ms_sum = float(ms.sum())
            md_sum = float(mD.sum())

            cond = ms > 1e-12
            rho_num = float(torch.where(cond, (mD / ms.clamp_min(1e-30)) * oS_dot_oD, torch.zeros_like(oS_dot_oD)).sum())
            rho_den = float(torch.where(cond, oD_dot_oD, torch.zeros_like(oD_dot_oD)).sum())

            out_reads[name][scope] = {
                "ro": ro,
                "oo": oo,
                "ms": ms_sum,
                "md": md_sum,
                "rho_num": rho_num,
                "rho_den": rho_den,
            }

    return {"g": g, "att": att, "reads": out_reads}


def ratios(sums: dict) -> dict:
    """From raw readout sums, the aggregate ratios R, M, rho, M*rho (rounded to 6 decimals)."""
    ro, oo = float(sums.get("ro", 0.0)), float(sums.get("oo", 0.0))
    ms_, md = float(sums.get("ms", 0.0)), float(sums.get("md", 0.0))
    rho_num, rho_den = float(sums.get("rho_num", 0.0)), float(sums.get("rho_den", 0.0))

    def div(a: float, b: float) -> float:
        return a / b if b != 0.0 and math.isfinite(b) else float("nan")

    R = div(ro, oo)
    M = div(ms_, md)
    rho = div(rho_num, rho_den)
    return {"R": _r6(R), "M": _r6(M), "rho": _r6(rho), "M_rho": _r6(M * rho)}


def run_call(r, rec: list, kv_cap: dict, kv0: list, grid: sel.Grid, f0: int, seed: int,
             a0: Optional[torch.Tensor], context: torch.Tensor, context_mask: torch.Tensor,
             patch: tuple[int, int, int]) -> dict:
    """Quick-metrics dense diagnostics beside a finished dense pass."""
    qm = r.cfg.qm
    k = int(qm.get("k", 39))
    layer = int(qm.get("layer", 18))
    n_exec = int(qm.get("executed", 10))
    eps = tuple(float(x) for x in qm.get("eps", (0.05, -0.05, 0.02, -0.02)))
    tips = tuple(int(x) for x in qm.get("tips", (78, 82)))
    check_enabled = bool(qm.get("check", True))
    late = str(qm.get("late", "f1s84"))

    L, S = r.num_layers, len(rec)
    kf, pf, half = grid.num_future, grid.per_frame, grid.width // 2
    dev = rec[0]["lat_v"].device
    pt, ph, pw = patch

    def tok(lat: torch.Tensor) -> torch.Tensor:
        return rearrange(
            lat, "b c (f x) (h y) (w z) -> b (f h w) (x y z c)", x=pt, y=ph, z=pw
        )[0].float()

    allf = torch.arange(kf, device=dev)
    frame = allf // pf
    col = (allf % pf) % grid.width
    wrist = col >= half
    groups = {
        "all": torch.ones(kf, dtype=torch.bool, device=dev),
        "agent": ~wrist,
        "wrist": wrist,
        "near": frame == 0,
        "far": frame == 1,
        "near_agent": (frame == 0) & (~wrist),
        "near_wrist": (frame == 0) & wrist,
        "far_agent": (frame == 1) & (~wrist),
        "far_wrist": (frame == 1) & wrist,
    }

    z0 = tok(rec[0]["lat_v"])[:f0][allf % pf]
    sig_v: list[float] = []
    sig_a: list[float] = []
    x0v: list[torch.Tensor] = []
    dd: list[torch.Tensor] = []
    for s in range(S):
        sv = float(rec[s]["ts_v"].float() / r.model.infer_video_scheduler.num_train_timesteps)
        sa = float(rec[s]["ts_a"].float() / r.model.infer_action_scheduler.num_train_timesteps)
        sig_v.append(sv)
        sig_a.append(sa)
        tv = tok(rec[s]["lat_v"])
        x0_s = tv[f0:] - sv * rec[s]["vel"][0, f0:].float()
        x0v.append(x0_s)
        dd.append(x0_s - z0)

    dd_last = dd[-1]
    dot_den = (dd_last * dd_last).sum(dim=-1)
    F: dict[str, list] = {}
    for gname, mask in groups.items():
        rows_f = []
        den = dot_den[mask].sum()
        for s in range(S):
            num = (dd[s] * dd_last).sum(dim=-1)[mask].sum()
            rows_f.append(_r6(float(num / den)) if den != 0.0 else None)
        F[gname] = rows_f

    # Plans from the state of step 1, measured with a sparse forward (L84-meas).
    rows1 = r._explore_rows(late, grid, None).to(dev)
    n1 = int(rows1.numel())
    b1 = torch.full((n1,), math.log(kf / n1), dtype=torch.float32, device=dev)
    st1 = sparse_forward(r, rec[1]["pre_v"], rec[1]["pre_a"], kv0, rows1, b1, f0, score_layer=layer)
    av1 = st1["av"]
    recv1 = st1["recv"] / L
    x0_1 = tok(rec[1]["lat_v"])[f0 + rows1] - sig_v[1] * st1["vel"][0].float()
    df1 = (x0_1 - z0[rows1]).norm(dim=-1)

    av_m = torch.zeros(kf)
    df_m = torch.zeros(kf)
    av_m[rows1.cpu()] = av1.cpu()
    df_m[rows1.cpu()] = df1.cpu()

    rr = torch.full((kf,), float("-inf"), device=dev)
    rr[rows1] = recv1
    rr[fv.PROBE_SINK] = float("-inf")
    sinks = torch.cat([torch.tensor([fv.PROBE_SINK], device=dev), torch.topk(rr, fv.N_SINKS - 1).indices])

    l84 = fv.c2f_plan(rows1.cpu(), av_m, df_m, sinks.cpu(), grid, k, n_steps=S, core_all=False, last_extra=0)
    rs = fv.role_plan(rows1.cpu(), av_m, df_m, sinks.cpu(), grid, tips, k, variant="rs", extra=0, seed=0, n_steps=S)

    def plan_read(plan, s_: int):
        rows_p = plan.real(s_).to(dev)
        b_p = torch.zeros(int(rows_p.numel()), dtype=torch.float32, device=dev)
        for j, lg in plan.bias_at(s_).items():
            b_p[rows_p == j] = lg
        return rows_p, b_p

    def strat_read(s_: int):
        rows_s = fv.stratified_pick(torch.arange(kf), k, grid, seed=(int(seed) * 31 + 7 * s_ + 1) & 0x7FFFFFFF).to(dev)
        b_s = torch.full((k,), math.log(kf / k), dtype=torch.float32, device=dev)
        return rows_s, b_s

    actions: dict[str, torch.Tensor] = {}
    share: dict[str, Any] = {"dense": [], "dense_l": [], "a": {"l84": [], "strat": []},
                             "b": {"l84": [], "strat": [], "rs": []}}
    rows_log: dict[str, dict] = {"l84": {}, "strat": {}}
    checks_c_dense: list[bool] = []
    checks_sparse: list[bool] = []
    kappa: dict[str, dict[str, list]] = {name: {gname: [] for gname in groups} for name in ("l84", "strat")}
    e_x: dict[str, list] = {"l84": [], "strat": []}

    for s in range(S):
        pv, pa = rec[s]["pre_v"], rec[s]["pre_a"]
        kv_s = [kv_cap[(s, l)] for l in range(L)]
        xa_lat = rec[s]["lat_a"].float()

        def ax0(pred: torch.Tensor) -> torch.Tensor:
            return (xa_lat - sig_a[s] * pred.float())[0]

        actions[f"d@{s}"] = ax0(rec[s]["pred_a"])

        full = action_forward(r, pa, kv0, kv_s, None, None, f0)
        checks_c_dense.append(bool(torch.equal(full["pred"], rec[s]["pred_a"])))
        if full["share"]:
            share["dense"].append(_r6(float(sum(full["share"]) / len(full["share"]))))
            share["dense_l"].append(_r6(float(full["share"][layer])))
        else:
            share["dense"].append(None)
            share["dense_l"].append(None)

        if check_enabled:
            sp196 = sparse_forward(r, pv, pa, kv0, allf, None, f0)
            checks_sparse.append(bool(torch.equal(sp196["vel"], rec[s]["vel"][:, f0:]) and torch.equal(sp196["pred"], rec[s]["pred_a"])))

        for name in ("l84", "strat"):
            rows_n, bias_n = plan_read(l84, s) if name == "l84" else strat_read(s)
            sp = sparse_forward(r, pv, pa, kv0, rows_n, bias_n, f0)
            xs = tok(rec[s]["lat_v"])[f0 + rows_n] - sig_v[s] * sp["vel"][0].float()
            num_vec = ((xs - z0[rows_n]) * dd[s][rows_n]).sum(dim=-1)
            den_vec = (dd[s][rows_n] * dd[s][rows_n]).sum(dim=-1)
            for gname, gmask in groups.items():
                m = gmask[rows_n]
                if bool(m.any()):
                    kappa[name][gname].append(_r6(float((num_vec[m].sum() / den_vec[m].sum()).item())))
                else:
                    kappa[name][gname].append(None)

            den_norm = x0v[s][rows_n].norm()
            e_x[name].append(_r6(float((xs - x0v[s][rows_n]).norm() / den_norm)) if den_norm != 0.0 else None)

            actions[f"a:{name}@{s}"] = ax0(sp["pred"])
            share["a"][name].append(_r6(float(sum(sp["share"]) / len(sp["share"]))))

            ab = action_forward(r, pa, kv0, kv_s, rows_n, bias_n, f0)
            actions[f"b:{name}@{s}"] = ax0(ab["pred"])
            if ab["share"]:
                share["b"][name].append(_r6(float(sum(ab["share"]) / len(ab["share"]))))
            else:
                share["b"][name].append(None)
            rows_log[name][str(s)] = [int(x) for x in rows_n.tolist()]

    # Role-select read at the last step.
    last_s = S - 1
    rows_rs, bias_rs = plan_read(rs, last_s)
    ab_rs = action_forward(r, rec[last_s]["pre_a"], kv0, [kv_cap[(last_s, l)] for l in range(L)], rows_rs, bias_rs, f0)
    aa_rs = (rec[last_s]["lat_a"].float() - sig_a[last_s] * ab_rs["pred"].float())[0]
    actions[f"b:rs@{last_s}"] = aa_rs
    share["b"]["rs"].append(_r6(float(sum(ab_rs["share"]) / len(ab_rs["share"]))))

    # c(s): dense action stream with an epsilon-scaled first-step displacement, continued in fp32.
    if a0 is not None:
        for s in range(S):
            xa_s = rec[s]["lat_a"].float()
            x0s = xa_s - sig_a[s] * rec[s]["pred_a"].float()
            a0_dev = a0.to(xa_s.device)
            for e in (0.0,) + eps:
                xh = x0s.clone()
                xh[0, :n_exec, :3] = a0_dev + (1.0 + e) * (x0s[0, :n_exec, :3] - a0_dev)
                v = (xa_s - xh) / sig_a[s]
                x = xa_s + rec[s]["d_a"].float() * v
                for s2 in range(s + 1, S):
                    pa2 = r.action.pre_dit(
                        action_tokens=x.to(rec[s2]["lat_a"].dtype),
                        timestep=rec[s2]["ts_a"],
                        context=context,
                        context_mask=context_mask,
                    )
                    pred2 = action_forward(r, pa2, kv0, [kv_cap[(s2, l)] for l in range(L)], None, None, f0, share=False)["pred"]
                    x = x + rec[s2]["d_a"].float() * pred2.float()
                actions[f"c{e:g}@{s}"] = x[0]

    # Last-step readout.
    kv_last = [kv_cap[(last_s, l)] for l in range(L)]
    cap_full = action_forward(r, rec[last_s]["pre_a"], kv0, kv_last, None, None, f0, capture=True, share=False)
    reads_last = {
        "l84": plan_read(l84, last_s),
        "rs": plan_read(rs, last_s),
        "strat": strat_read(last_s),
    }
    cams = {
        "all": torch.ones(kf, dtype=torch.bool, device=dev),
        "agent": ~wrist,
        "wrist": wrist,
    }

    g_list: list[torch.Tensor] = []
    att_list: list[torch.Tensor] = []
    agg_sums: dict[str, dict[str, dict[str, float]]] = {}
    layer_sums: dict[str, dict[str, dict[str, float]]] = {}
    for name in reads_last:
        agg_sums[name] = {scope: {key: 0.0 for key in ("ro", "oo", "ms", "md", "rho_num", "rho_den")}
                          for scope in cams}
        layer_sums[name] = {scope: None for scope in cams}

    for l in range(L):
        q, ka, va = cap_full["capture"][l]
        keys = torch.cat([kv0[l]["k"], kv_last[l][0], ka], dim=1)
        values = torch.cat([kv0[l]["v"], kv_last[l][1], va], dim=1)
        logits = r._head_logits(q, keys)
        vh = rearrange(values, "b s (n d) -> b n s d", n=r.num_heads)[0].float()
        out = readout(logits, vh, f0, kf, reads_last, cams)
        g_list.append(out["g"])
        att_list.append(out["att"])
        for name in reads_last:
            for scope in cams:
                one = out["reads"][name][scope]
                for kk in ("ro", "oo", "ms", "md", "rho_num", "rho_den"):
                    agg_sums[name][scope][kk] += one[kk]
                if l == layer:
                    layer_sums[name][scope] = dict(one)

    g_layer = g_list[layer]
    g_mean = torch.stack(g_list).mean(dim=0)
    att_layer = att_list[layer]
    att_mean = torch.stack(att_list).mean(dim=0)

    rho: dict[str, dict[str, dict]] = {}
    for name in reads_last:
        rho[name] = {
            f"l{layer}": {scope: ratios(layer_sums[name][scope]) for scope in cams},
            "all_layers": {scope: ratios(agg_sums[name][scope]) for scope in cams},
        }

    return {
        "k": k,
        "layer": layer,
        "late": late,
        "n_exec": n_exec,
        "eps": list(eps),
        "sigma_v": [_r6(x) for x in sig_v],
        "sigma_a": [_r6(x) for x in sig_a],
        "dsigma_a": [_r6(-float(rec[s]["d_a"])) for s in range(S)],
        "F": {gname: [v for v in vals] for gname, vals in F.items()},
        "kappa": {name: {gname: [v for v in vals] for gname, vals in groups_.items()} for name, groups_ in kappa.items()},
        "e_x": {name: [v for v in vals] for name, vals in e_x.items()},
        "share": {
            "dense": share["dense"],
            "dense_l": share["dense_l"],
            "a": {"l84": share["a"]["l84"], "strat": share["a"]["strat"]},
            "b": {"l84": share["b"]["l84"], "strat": share["b"]["strat"], "rs": share["b"]["rs"]},
        },
        "plans": {
            "sinks": [int(x) for x in sinks.tolist()],
            "l84_core": l84.core.tolist(),
            "rs_core": rs.core.tolist(),
            "rows": rows_log,
            "rs_rows_last": [int(x) for x in rows_rs.tolist()],
        },
        "actions": {name: _f32_b64(t) for name, t in actions.items()},
        "g_l_f32": _f32_b64(g_layer),
        "g_mean_f32": _f32_b64(g_mean),
        "att_l_f32": _f32_b64(att_layer),
        "att_mean_f32": _f32_b64(att_mean),
        "diff_last_f32": _f32_b64(dd_last.norm(dim=-1)),
        "rho": rho,
        "checks": {
            "c_equal_dense": bool(all(checks_c_dense)) if checks_c_dense else None,
            "c_equal_dense_steps": [bool(x) for x in checks_c_dense],
            "sparse196_equal": bool(all(checks_sparse)) if check_enabled and checks_sparse else None,
            "sparse196_steps": [bool(x) for x in checks_sparse] if check_enabled else None,
        },
        "c_skipped": a0 is None,
    }
