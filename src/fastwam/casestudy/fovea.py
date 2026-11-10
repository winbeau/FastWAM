"""Fovea round: sinks, core and periphery of the future tokens (``prune.fovea``), decided once at the warm-up step 0.

Every method computes on each pruned step as many future rows as S0 (k = round(keep * Kf)):
  sinks (3)  the future tokens receiving the most attention at step 0 (video and action rows, layer / head mean);
  core (C)   A->V top 10 united with Diff top 10 (both without the sinks), filled in A->V order up to C = k - 3 - R;
  periphery  every other future token, with R = 16 slots: one per region, a region being a 2 x 2 block (rows 0-3 / 4-6 x
             columns 0-3 / 4-6 of the camera) of one (future frame, camera) 7 x 7 grid.
Methods: ``sink`` (S0's top-k with the sinks forced in), ``rep`` (one real representative per region, its key logit + log n),
``fov_grid`` / ``fov_cluster`` (one synthetic region token: mean member embedding, fractional RoPE position, logit + log n,
its velocity given to every member; regions by the grid or by step-0 k-means on the clean estimate), ``interlace`` (the
periphery in 9 slices, slice s computed on pruned step s). A periphery that fits its R slots is computed token by token.
fovea-w round: ``interlace_w`` adds log(N / n_k) to the key logits of step k's slice (N = all periphery tokens, truncated
ones included; n_k = the slice's size), every row; ``interlace_wfar`` also takes the core from the far future frame only
(latent frame 2, both cameras; the near frame's tokens go to the periphery), filling from the near frame by A->V only
when the far frame cannot fill it (keep above ~50 %).
N1-N4 round: ``interlace_cal`` (M4's plan, the slice bias calibrated per layer / head / row on step-0 logits by the runner),
``interlace_fill`` (M4's plan, every uncomputed key / value filled by the runner from the nearest computed token of the step,
``fill_map``), ``merge`` (sinks + core, the periphery in Fov-grid's regions merged per layer and un-merged into per-member
residual streams by the runner; ``singleton`` puts every future token in a group of its own) and ``interlace_ht`` (every
token computed f_j >= 1 times over the 9 pruned steps, f proportional to A->V, key logit + log(9 / f_j): ``ht_frequencies``,
``ht_schedule``).
step0-sparse round: step 0 sparse too (``prune.step0``). It computes only the 39 probes (``probe_index``: rows / columns
{1, 3, 5} of every (future frame, camera) 7 x 7 block, the sink 182 and the corners 7 / 105), the rest drafted from the
probes by in-block bilinear interpolation (``interp_tokens``); the plan's leftover probes go to the last slices
(``step0_slices``); plan D marks the end-effector path on the agentview blocks (``tube_mask``).
step0-sparse-r2: block-local inverse-distance neighbours (``nearest_sources``, ``idw_fill``) for the step-by-step drafts and the
previous call's imagination (``memory_map``, ``align_frames``); 10 slices from step 0 (``make_plan(n_slices, from_step0)``);
the step-5 core reselection (``reselect``, ``select_core``).
Pure functions; the runner applies the plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import selection as sel

N_SINKS, N_AV, N_DIFF, R_SLOTS, N_SLICES = 3, 10, 10, 16, 9
METHODS = ("sink", "rep", "fov_grid", "fov_cluster", "interlace", "interlace_w", "interlace_wfar",
           "interlace_cal", "interlace_fill", "merge", "interlace_ht")
INTERLACE = ("interlace", "interlace_w", "interlace_wfar", "interlace_cal", "interlace_fill")
# step0-sparse (DECISIONS B.1): the probe rows / columns of a 7 x 7 block, the fixed sink, and the 2 block corners with the
# highest mean step-0 received attention over the fovea-w M4w probes (both the wrist block's (0, 0)).
PROBE_RC, PROBE_SINK, PROBE_CORNERS = (1, 3, 5), 182, (7, 105)
STEP0 = ("b", "d")


def probe_index(grid: sel.Grid, extras: bool = True) -> torch.Tensor:
    """step0-sparse: the 39 probes (ascending future indices): rows and camera columns in ``PROBE_RC`` of every (future
    frame, camera) block, the sink and the corners (``extras``; without them the 36 grid points)."""
    half = grid.width // 2
    idx = [f * grid.per_frame + r * grid.width + cam * half + c
           for f in range(grid.frames - 1) for cam in (0, 1) for r in PROBE_RC for c in PROBE_RC]
    if extras:
        idx += [PROBE_SINK, *PROBE_CORNERS]
    if len(set(idx)) != len(idx) or max(idx) >= grid.num_future:
        raise ValueError("step0: the probes are not distinct future tokens")
    return torch.tensor(sorted(idx), dtype=torch.long)


def _interp_weights(n: int = 7) -> torch.Tensor:
    """[n, 3] weights of the block rows (or columns) on the probe rows {1, 3, 5}: nearest outside, linear between."""
    w = torch.zeros(n, len(PROBE_RC), dtype=torch.float64)
    for r in range(n):
        if r <= PROBE_RC[0]:
            w[r, 0] = 1.0
        elif r >= PROBE_RC[-1]:
            w[r, -1] = 1.0
        else:
            i = max(i for i, p in enumerate(PROBE_RC) if p <= r)
            t = (r - PROBE_RC[i]) / (PROBE_RC[i + 1] - PROBE_RC[i])
            w[r, i], w[r, i + 1] = 1.0 - t, t
    return w


def interp_tokens(values: torch.Tensor, grid: sel.Grid, computed: torch.Tensor) -> torch.Tensor:
    """values [Kf, D] (float; only the rows of ``computed`` are read) -> [Kf, D] float32: every (future frame, camera) block
    bilinearly interpolated from its 3 x 3 probe grid (separable, nearest at the edges), the ``computed`` rows their own
    values. With every token computed this is the identity."""
    half = grid.width // 2
    out = torch.empty(values.shape, dtype=torch.float32, device=values.device)
    w = _interp_weights(grid.height).float().to(values.device)
    wc = _interp_weights(half).float().to(values.device)
    pr = torch.tensor(PROBE_RC, device=values.device)
    for f in range(grid.frames - 1):
        for cam in (0, 1):
            rows = torch.arange(grid.height, device=values.device)
            cols = cam * half + torch.arange(half, device=values.device)
            block = f * grid.per_frame + rows[:, None] * grid.width + cols[None, :]  # [7, 7]
            g = values.float()[block[pr][:, pr]]  # [3, 3, D]
            out[block] = torch.einsum("ri,ijd,cj->rcd", w, g, wc)
    c = computed.to(values.device)
    out[c] = values[c].float()
    return out


def step0_slices(periphery: torch.Tensor, probes: torch.Tensor, av: torch.Tensor, n: int = N_SLICES, quota: int = R_SLOTS):
    """step0-sparse (DECISIONS B.5) -> (n slices, truncated): the periphery's probes by A->V (descending, ties to the smaller
    index) fill slice n, then n - 1 (``quota`` each), the rest go to slice n - 2; the other periphery tokens (ascending) go
    to slices 1..n - 2 by i mod (n - 2), each cut to its free slots by A->V."""
    is_probe = torch.isin(periphery, probes.to(periphery.device))
    pp, rest = periphery[is_probe], periphery[~is_probe]
    pp = pp[torch.argsort(-av[pp].float(), stable=True)]
    last, prev, third = pp[:quota], pp[quota:2 * quota], pp[2 * quota:]
    if third.numel() > quota:  # step0-explore: more computed tokens than three slices hold; the rest join the raster
        rest = torch.sort(torch.cat([rest, third[quota:]])).values
        third = third[:quota]
    m = n - 2
    slices, cut = [], []
    for i in range(m):
        part = rest[i::m]
        room = quota - (third.numel() if i == m - 1 else 0)
        if part.numel() > room:
            order = torch.argsort(av[part].float(), descending=True, stable=True)
            cut.append(part[order[room:]])
            part = part[order[:room]]
        if i == m - 1:
            part = torch.cat([part, third])
        slices.append(torch.sort(part).values)
    slices += [torch.sort(prev).values, torch.sort(last).values]
    return slices, (torch.cat(cut) if cut else periphery[:0])


def roll_blocks(values: torch.Tensor, grid: sel.Grid, shift: int = 1) -> torch.Tensor:
    """shrinkage-mechanism: values [Kf, D] moved ``shift`` tokens along the row (columns, cyclic) inside every (future frame,
    camera) block: same values, same size, every one at the wrong position."""
    half = grid.width // 2
    j = torch.arange(grid.num_future)
    rem = j % grid.per_frame
    col = rem % grid.width
    cam, c = col // half, col % half
    src = j - col + cam * half + (c - shift) % half
    return values[src.to(values.device)]


def token_latent_mask(mask, height: int, width: int, patch_h: int, patch_w: int) -> torch.Tensor:
    """region-attrib: a token mask [frames, height * width] (tokens flattened as (h w), the video patch embedding's order) ->
    the latent mask [1, 1, frames, height * patch_h, width * patch_w]; token (f, r, c) covers latent rows r * patch_h ...
    (r + 1) * patch_h - 1 and columns c * patch_w ... (c + 1) * patch_w - 1 of latent frame f."""
    mt = torch.as_tensor(mask, dtype=torch.bool).reshape(-1, height, width)
    return mt.repeat_interleave(patch_h, dim=1).repeat_interleave(patch_w, dim=2)[None, None]


def token_norms(d: torch.Tensor, height: int, width: int, patch_h: int, patch_w: int) -> torch.Tensor:
    """region-attrib: latents [1, z, frames, height * patch_h, width * patch_w] -> [frames, height * width], each token's norm
    over its channels x patch (fp32)."""
    ch, nf = int(d.shape[1]), int(d.shape[2])
    x = d[0].float().reshape(ch, nf, height, patch_h, width, patch_w)
    return x.pow(2).sum(dim=(0, 3, 5)).sqrt().reshape(nf, height * width)


def block_smooth(values: torch.Tensor, grid: sel.Grid) -> torch.Tensor:
    """step0-explore: values [Kf, D] smoothed by a 3 x 3 box mean inside every (future frame, camera) 7 x 7 block (edges:
    the mean of the cells present)."""
    half = grid.width // 2
    out = torch.empty_like(values, dtype=torch.float32)
    for f in range(grid.frames - 1):
        for cam in (0, 1):
            idx = torch.tensor([f * grid.per_frame + r * grid.width + cam * half + c for r in range(grid.height) for c in range(half)],
                               device=values.device)
            blk = values.float()[idx].T.reshape(1, -1, grid.height, half)  # [1, D, 7, 7]
            sm = torch.nn.functional.avg_pool2d(blk, 3, stride=1, padding=1, count_include_pad=False)
            out[idx] = sm.reshape(-1, grid.height * half).T
    return out


def nearest_sources(sources: torch.Tensor, grid: sel.Grid, k: int = 4):
    """step0-sparse-r2 block-local neighbours -> (nbr [Kf, k] positions in ``sources``, w [Kf, k] float64, has [Kf] bool): the k
    sources nearest to every future token in its own (future frame, camera) block by (row, camera column) Euclidean
    distance (ties to the smaller future index), weights 1 / d^2 summing to 1 (0 on padding); a source maps to itself."""
    half = grid.width // 2
    src = sources.cpu().tolist()
    pos = {}
    for p, j in enumerate(src):
        rem = j % grid.per_frame
        pos.setdefault((j // grid.per_frame, (rem % grid.width) >= half), []).append((p, j, rem // grid.width, (rem % grid.width) % half))
    nbr = torch.zeros(grid.num_future, k, dtype=torch.long)
    w = torch.zeros(grid.num_future, k, dtype=torch.float64)
    has = torch.zeros(grid.num_future, dtype=torch.bool)
    where = {j: p for p, j in enumerate(src)}
    for j in range(grid.num_future):
        rem = j % grid.per_frame
        r, c = rem // grid.width, (rem % grid.width) % half
        if j in where:
            nbr[j, 0], w[j, 0], has[j] = where[j], 1.0, True
            continue
        cand = sorted(((rr - r) ** 2 + (cc - c) ** 2, jj, p) for p, jj, rr, cc in pos.get((j // grid.per_frame, (rem % grid.width) >= half), []))
        cand = cand[:k]
        if not cand:
            continue
        inv = [1.0 / d for d, _, _ in cand]
        tot = sum(inv)
        for i, ((_, _, p), x) in enumerate(zip(cand, inv)):
            nbr[j, i], w[j, i] = p, x / tot
        has[j] = True
    return nbr, w, has


def idw_fill(values: torch.Tensor, sources: torch.Tensor, grid: sel.Grid, targets: torch.Tensor | None = None,
             k: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """values [Kf, D] -> (float32 copy with every target (default: every non-source) that has a source in its block replaced
    by the inverse-distance-squared mean of its ``k`` nearest sources' values, the replaced targets). Sources and targets
    without a source in their block keep their values."""
    nbr, w, has = nearest_sources(sources, grid, k)
    dev = values.device
    out = values.float().clone()
    src = sources.to(dev)
    if targets is None:
        mask = torch.ones(grid.num_future, dtype=torch.bool)
        mask[sources.cpu()] = False
        targets = torch.nonzero(mask).flatten()
    tg = targets.cpu()
    tg = tg[has[tg]]
    if tg.numel():
        vals = values.float()[src[nbr[tg].to(dev)]]  # [T, k, D]
        out[tg.to(dev)] = (w[tg].float().to(dev)[..., None] * vals).sum(dim=1)
    return out, tg.to(dev)


def align_frames(z: torch.Tensor, grid: sel.Grid) -> torch.Tensor:
    """A previous call's future [Kf, D] moved one call on: new frame 1 = old frame 2, new frame 2 = old 2 + (old 2 - old 1)."""
    if grid.frames != 3:
        raise ValueError("align_frames is defined for two future latent frames")
    f1, f2 = z[:grid.per_frame], z[grid.per_frame:]
    return torch.cat([f2, f2 + (f2 - f1)])


def memory_map(terminal: torch.Tensor, computed: torch.Tensor, grid: sel.Grid, direct: bool = False) -> torch.Tensor:
    """step0-sparse-r2 R6: the previous call's imagination for this call [Kf, D] (float32): the terminal latents of the tokens
    computed on the last step, the others interpolated from them in their block (own terminal without a source there;
    ``direct``: every terminal as it is), then time-aligned."""
    m = terminal.float() if direct else idw_fill(terminal, computed, grid)[0]
    return align_frames(m, grid)


def select_core(av: torch.Tensor, diff: torch.Tensor, sinks: torch.Tensor, c: int) -> torch.Tensor:
    """``make_plan``'s core rule: A->V top 10 united with Diff top 10 (without the sinks), filled to c by A->V (sorted)."""
    core_set: list[int] = []
    for score, n in ((av, N_AV), (diff, N_DIFF)):
        for j in sel.topk_indices(score, min(n, c), exclude=sinks).tolist():
            if j not in core_set:
                core_set.append(j)
    for j in sel.topk_indices(av, av.numel(), exclude=sinks).tolist():
        if len(core_set) >= c:
            break
        if j not in core_set:
            core_set.append(j)
    return torch.sort(torch.tensor(core_set[:c], dtype=torch.long, device=av.device)).values


def reselect(plan: "Plan", av: torch.Tensor, diff: torch.Tensor) -> "Plan":
    """step0-sparse-r2 R7b: the core recomposed from (av, diff) with the plan's sinks; the slices without the new core, each
    slice's bias log(N / its remaining size) with N the original periphery size."""
    core = select_core(av, diff, plan.sinks.cpu(), int(plan.core.numel())).to(plan.core.device)
    fixed = torch.sort(torch.cat([plan.sinks, core])).values
    slices = [x[~torch.isin(x, core)] for x in plan.slices]
    n = int(plan.periphery.numel())
    bias = [math.log(n / int(x.numel())) if x.numel() else 0.0 for x in slices] if plan.slice_bias else []
    return Plan(plan.method, plan.sinks, core, plan.periphery, fixed, slices=slices, truncated=plan.truncated, slice_bias=bias)


def neighbours(j: int, grid: sel.Grid) -> list[int]:
    """step0-explore: the token and its 8-neighbourhood inside its (future frame, camera) 7 x 7 block (ascending)."""
    half = grid.width // 2
    f, rem = divmod(int(j), grid.per_frame)
    r, w = divmod(rem, grid.width)
    cam, c = divmod(w, half)
    return sorted(f * grid.per_frame + rr * grid.width + cam * half + cc
                  for rr in range(max(0, r - 1), min(grid.height, r + 2)) for cc in range(max(0, c - 1), min(half, c + 2)))


def c2f_survey(probes: torch.Tensor, av: torch.Tensor, diff: torch.Tensor, sinks: torch.Tensor, k: int, grid: sel.Grid,
               budget: int) -> torch.Tensor:
    """step0-explore C2F step 1: the sinks, then the neighbourhoods of the probes ranked 1..k by A->V and by Diff (taken in turn:
    A->V #1, Diff #1, A->V #2, ...; sinks excluded from the ranking) until ``budget`` tokens (sorted)."""
    sk = set(int(x) for x in sinks.tolist())
    cand = [int(p) for p in probes.tolist() if int(p) not in sk]
    by_av = sorted(cand, key=lambda p: (-float(av[p]), p))[:k]
    by_df = sorted(cand, key=lambda p: (-float(diff[p]), p))[:k]
    order = [x for pair in zip(by_av, by_df) for x in pair]
    out = list(sk)
    for p in order:
        for j in neighbours(p, grid):
            if len(out) >= budget:
                break
            if j not in out:
                out.append(j)
    return torch.sort(torch.tensor(out[:budget], dtype=torch.long)).values


def c2f_plan(measured: torch.Tensor, av: torch.Tensor, diff: torch.Tensor, sinks: torch.Tensor, grid: sel.Grid, k: int,
             first: int = 2, n_steps: int = 10, quota: int = R_SLOTS, core_all: bool = False, last_extra: int = 0) -> "Plan":
    """step0-explore C2F plan of steps ``first``..n_steps-1, from the scores of the ``measured`` tokens (unmeasured: 0): sinks as
    given; the core by ``make_plan``'s rule (A->V top 10 + Diff top 10, filled by A->V) to k - sinks - quota; the measured
    periphery tokens (best A->V first) fill the last two slices, the rest of the periphery the earlier ones by raster interlace
    (``step0_slices``); slice key bias log(N / n_s). real(s) for s >= first reads the slice of step s."""
    kf = grid.num_future
    m = torch.zeros(kf, dtype=torch.bool)
    m[measured.cpu()] = True
    av_m, df_m = torch.where(m, av.float(), torch.zeros(kf)), torch.where(m, diff.float(), torch.zeros(kf))
    c = max(0, k - int(sinks.numel()) - quota)
    core = select_core(av.float() if core_all else av_m, diff.float() if core_all else df_m, sinks.cpu(), c)  # core_all: spread scores
    mask = torch.ones(kf, dtype=torch.bool)
    mask[sinks.cpu()] = False
    mask[core] = False
    periphery = torch.nonzero(mask).flatten()
    n = n_steps - first
    pm = periphery[m[periphery]]
    best = pm[torch.argsort(-av_m[pm], stable=True)][:2 * quota]
    sl, cut = step0_slices(periphery, best, av.float() if core_all else av_m, n=n, quota=quota)
    # real(s) reads slices[(s - 1) % n]: step s (first..n_steps-1) must read sl[s - first]
    slices = [None] * n
    for s in range(first, n_steps):
        slices[(s - 1) % n] = sl[s - first]
    if last_extra and cut.numel():  # step0-explore: the last step also computes the best never-computed tokens by A->V
        a_ = av.float() if core_all else av_m
        extra = cut[torch.argsort(-a_[cut], stable=True)][:last_extra]
        i9 = (n_steps - 1 - 1) % n
        slices[i9] = torch.sort(torch.cat([slices[i9], extra])).values
        cut = cut[~torch.isin(cut, extra)]
    bias = [math.log(int(periphery.numel()) / int(x.numel())) if x.numel() else 0.0 for x in slices]
    fixed = torch.sort(torch.cat([sinks.cpu(), core])).values
    return Plan("interlace_w", sinks.cpu(), core, periphery, fixed, slices=slices, truncated=cut, slice_bias=bias)


def add_last_extra(plan: "Plan", av: torch.Tensor, n: int, n_steps: int = 10) -> "Plan":
    """late-anchor-confirm (M4w + extra reads): the last step's slice (real(n_steps - 1) reads slices[(n_steps - 2) % len]) also
    computes the ``n`` truncated tokens with the highest ``av`` (ties to the smaller index); they leave the truncated set and that
    slice's key bias becomes log(N / n_s) of its new size (interlace_w). Every other slice is unchanged."""
    if not plan.slices or plan.truncated is None or not plan.truncated.numel() or n <= 0:
        return plan
    cut = plan.truncated
    extra = cut[torch.argsort(-av.float().to(cut.device)[cut], stable=True)][:n]
    i9 = (n_steps - 2) % len(plan.slices)
    slices = list(plan.slices)
    slices[i9] = torch.sort(torch.cat([slices[i9], extra.to(slices[i9].device)])).values
    bias = list(plan.slice_bias)
    if bias:
        bias[i9] = math.log(int(plan.periphery.numel()) / int(slices[i9].numel()))
    return Plan(plan.method, plan.sinks, plan.core, plan.periphery, plan.fixed, slices=slices, truncated=cut[~torch.isin(cut, extra)],
                bias_real=dict(plan.bias_real), regions=list(plan.regions), region_pos=plan.region_pos, slice_bias=bias, freq=plan.freq)


def stratified_pick(pool: torch.Tensor, n: int, grid: sel.Grid, seed: int) -> torch.Tensor:
    """step-marginal: ``n`` tokens of ``pool`` (future indices), stratified by (future frame, camera) in proportion to the pool's
    strata (largest remainder, ties to the earlier stratum), uniform within a stratum (torch generator ``seed``); sorted."""
    pool = pool.cpu()
    n = min(int(n), int(pool.numel()))
    if n <= 0:
        return pool[:0]
    half = grid.width // 2
    key = (pool // grid.per_frame) * 2 + ((pool % grid.per_frame) % grid.width) // half
    strata = [pool[key == k] for k in range(2 * (grid.frames - 1))]
    quota = torch.tensor([float(x.numel()) for x in strata]) * n / float(pool.numel())
    take = torch.floor(quota).long()
    rest = n - int(take.sum())
    for k in sorted(range(len(strata)), key=lambda k: (-float(quota[k] - take[k]), k))[:rest]:
        take[k] += 1
    g = torch.Generator().manual_seed(int(seed))
    picks = [x[torch.randperm(int(x.numel()), generator=g)[:int(t)]] for x, t in zip(strata, take)]
    return torch.sort(torch.cat(picks)).values


def lowest_av(computed: torch.Tensor, av: torch.Tensor, sinks: torch.Tensor, n: int) -> torch.Tensor:
    """step-marginal: the ``n`` tokens of ``computed`` with the lowest ``av`` (ties to the smaller index), the sinks excepted;
    sorted."""
    computed = computed.cpu()
    cand = computed[~torch.isin(computed, sinks.cpu())]
    order = torch.argsort(av.float().cpu()[cand], stable=True)
    return torch.sort(cand[order[:int(n)]]).values


def apply_step_delta(plan: "Plan", deltas, av: torch.Tensor, grid: sel.Grid, seed: int) -> "Plan":
    """step-marginal: steps 2-9 of the late anchor's plan (``c2f_plan``, slices of steps first..9) changed per ``deltas``
    ((step, n), ...; step 1 is the late set's business, n = 0 does nothing). n > 0: step s's slice also computes n tokens the
    step does not compute (``stratified_pick`` over every future token outside real(s)); n < 0: the |n| tokens of real(s) with
    the lowest ``av`` leave it (the sinks stay; slice tokens leave the slice, core tokens are dropped for that step only). The
    slice's key bias becomes log(N / n_s) of its new size (N = the periphery); every other step is unchanged."""
    slices, bias, drop = list(plan.slices), list(plan.slice_bias), dict(plan.drop)
    n_per = int(plan.periphery.numel())
    for s, n in deltas:
        if s < 2 or n == 0:
            continue
        i = (s - 1) % len(slices)
        computed = plan.real(s).cpu()
        if n > 0:
            pool = torch.arange(grid.num_future)
            pool = pool[~torch.isin(pool, computed)]
            slices[i] = torch.sort(torch.cat([slices[i].cpu(), stratified_pick(pool, n, grid, seed)])).values
        else:
            gone = lowest_av(computed, av, plan.sinks, -n)
            slices[i] = slices[i].cpu()[~torch.isin(slices[i].cpu(), gone)]
            core_gone = gone[torch.isin(gone, plan.fixed.cpu())]
            if core_gone.numel():
                drop[s] = core_gone
        bias[i] = math.log(n_per / int(slices[i].numel())) if slices[i].numel() else 0.0
    return Plan(plan.method, plan.sinks, plan.core, plan.periphery, plan.fixed, slices=slices, truncated=plan.truncated,
                bias_real=dict(plan.bias_real), regions=list(plan.regions), region_pos=plan.region_pos, slice_bias=bias,
                freq=plan.freq, drop=drop)


ROLE_VARIANTS = ("rs", "notip", "wdiff", "aav", "nofar", "wdiffb")
ROLE_ARM, ROLE_AGENT, ROLE_FAR = 12, 8, 4


def role_strata(tokens: torch.Tensor, grid: sel.Grid) -> torch.Tensor:
    """role-select: the stratum of each future token, (frame, camera, quadrant) with quadrants rows 0-3 / 4-6 x columns 0-3 / 4-6
    of its 7 x 7 camera block."""
    pf, half = grid.per_frame, grid.width // 2
    rem = tokens % pf
    r, col = rem // grid.width, rem % grid.width
    q = (r >= 4).long() * 2 + ((col % half) >= 4).long()
    return (tokens // pf) * 8 + (col // half) * 4 + q


def spread_order(pool: torch.Tensor, grid: sel.Grid, seed: int) -> torch.Tensor:
    """role-select: the order a rotating group walks its pool in. The pool is split by ``role_strata``, each stratum shuffled and
    the strata visited in a shuffled order, round robin (torch generator ``seed``; not the index order), so that n consecutive
    tokens spread over the strata and the whole pool comes before any token repeats."""
    pool = pool.cpu()
    key = role_strata(pool, grid)
    g = torch.Generator().manual_seed(int(seed))
    strata = [pool[key == k] for k in torch.unique(key).tolist()]
    strata = [x[torch.randperm(int(x.numel()), generator=g)] for x in strata]
    visit = torch.randperm(len(strata), generator=g).tolist()
    out = [int(strata[k][i]) for i in range(max((int(x.numel()) for x in strata), default=0)) for k in visit if i < strata[k].numel()]
    return torch.tensor(out, dtype=torch.long)


def _take(seq: torch.Tensor, start: int, n: int, taken: set) -> list:
    """The next ``n`` tokens of the cyclic ``seq`` from ``start`` that the step does not compute yet (they join ``taken``)."""
    out: list = []
    for t in range(int(seq.numel())):
        if len(out) == n:
            break
        j = int(seq[(start + t) % int(seq.numel())])
        if j not in taken:
            out.append(j)
            taken.add(j)
    return out


def role_plan(measured: torch.Tensor, av: torch.Tensor, diff: torch.Tensor, sinks: torch.Tensor, grid: sel.Grid, tips, k: int,
              variant: str = "rs", extra: int = 10, seed: int = 0, first: int = 2, n_steps: int = 10) -> "Plan":
    """role-select plan of steps ``first``..n_steps-1 (the late anchor's steps 2-9), from step 1's measured scores (``av`` / ``diff``:
    0 where not measured) and no A->V except in variant aav. Per step k tokens:
      sinks   as given, every step;
      arm     the 12 measured agentview near-frame tokens with the largest Diff (aav: A->V), sinks excepted, every step;
      tips    the 2 fixed wrist near-frame tokens ``tips`` (notip: none), sinks excepted, every step;
      wrist   the other wrist near-frame tokens, rotating (``spread_order``), as many as the rest of the budget leaves (10; notip 12,
              nofar 14); wdiff: the measured ones with the largest Diff, every step, no rotation; wdiffb (quick-metrics): wdiff, its
              wrist tokens' keys + log(N_wrist / n) on every step (n = 10, on the last step with the extra Diff ranks);
      agent   the other agentview near-frame tokens (not sinks, not arm), 8 rotating;
      far     the far-frame tokens of both cameras (not sinks), 4 rotating (nofar: none).
    A rotating group starts step i (0, 1, ...) at position i x its size in its order and skips tokens the step already computes; its
    keys get log(N_group / n_group) (pool size over the tokens it computes on that step), the every-step groups nothing. The last
    step also computes ``extra`` more: arm ranks 13.. (extra // 2, no bias) and the rest as more wrist tokens (wdiff: Diff ranks
    11..; rotating: the wrist group grows, its bias with it)."""
    if variant not in ROLE_VARIANTS:
        raise ValueError(f"role_plan: variant must be one of {ROLE_VARIANTS}")
    kf, pf, half = grid.num_future, grid.per_frame, grid.width // 2
    j = torch.arange(kf)
    col = (j % pf) % grid.width
    near = j < pf
    a_near, w_near = near & (col < half), near & (col >= half)
    meas = torch.zeros(kf, dtype=torch.bool)
    meas[measured.cpu()] = True
    sinks = sinks.cpu()
    not_sink = ~torch.isin(j, sinks)
    tip = torch.tensor([] if variant == "notip" else [int(x) for x in tips], dtype=torch.long)
    if variant != "notip" and (tip.numel() != 2 or not bool(w_near[tip].all())):
        raise ValueError("role_plan: tips must be 2 wrist near-frame tokens")
    tip = tip[~torch.isin(tip, sinks)]
    score = (av if variant == "aav" else diff).float().cpu()
    cand = j[a_near & meas & not_sink]
    ranked = cand[torch.argsort(-score[cand], stable=True)]
    arm, arm_extra = ranked[:ROLE_ARM], ranked[ROLE_ARM:ROLE_ARM + extra // 2]
    n_far = 0 if variant == "nofar" else ROLE_FAR
    base = torch.cat([sinks, arm, tip])
    n_w = k - int(base.numel()) - ROLE_AGENT - n_far  # the wrist group takes what the budget leaves (10 with 3 sinks, 2 tips)
    w_extra = extra - extra // 2
    w_pool = j[w_near & not_sink & ~torch.isin(j, tip)]
    wrist_fixed, wrist_extra = j[:0], j[:0]
    wdiff = variant in ("wdiff", "wdiffb")
    if wdiff:
        wm = w_pool[meas[w_pool]]
        wr = wm[torch.argsort(-diff.float().cpu()[wm], stable=True)]
        wrist_fixed, wrist_extra = wr[:n_w], wr[n_w:n_w + w_extra]
    fixed = torch.sort(torch.cat([base, wrist_fixed])).values
    seqs = {"agent": spread_order(j[a_near & not_sink & ~torch.isin(j, arm)], grid, seed + 1),
            "far": spread_order(j[~near & not_sink], grid, seed + 2)}
    sizes = {"agent": ROLE_AGENT, "far": n_far}
    if not wdiff:
        seqs["wrist"], sizes["wrist"] = spread_order(w_pool, grid, seed), n_w
    n = n_steps - first
    slices, step_bias, seen = [None] * n, {}, set()
    for i, s in enumerate(range(first, n_steps)):
        taken = set(fixed.tolist())
        rows, bias = [], {}
        last = s == n_steps - 1
        if last:
            for x in torch.cat([arm_extra, wrist_extra]).tolist():
                if x not in taken:
                    taken.add(x)
                    rows.append(x)
        for name in ("wrist", "agent", "far"):
            if name not in seqs or not sizes[name] or not seqs[name].numel():
                continue
            want = sizes[name] + (w_extra if last and name == "wrist" else 0)
            got = _take(seqs[name], i * sizes[name], want, taken)
            lg = math.log(int(seqs[name].numel()) / len(got)) if got else 0.0
            rows += got
            if lg != 0.0:
                bias.update({x: lg for x in got})
        if variant == "wdiffb":  # the Diff wrist group stands for every wrist near-frame token (as the rotating group of rs)
            wt = wrist_fixed.tolist() + (wrist_extra.tolist() if last else [])
            if wt:
                bias.update({x: math.log(int(w_pool.numel()) / len(wt)) for x in wt})
        slices[(s - 1) % n] = torch.sort(torch.tensor(rows, dtype=torch.long)).values
        step_bias[s] = bias
        seen |= set(rows)
        want_rows = k + (extra if last else 0)
        if int(fixed.numel()) + len(rows) != want_rows:
            raise RuntimeError(f"role_plan: step {s} computes {int(fixed.numel()) + len(rows)} tokens, not {want_rows}")
    periphery = torch.sort(torch.cat([v for v in seqs.values()] + [arm_extra, wrist_extra]).unique()).values
    truncated = periphery[~torch.isin(periphery, torch.tensor(sorted(seen), dtype=torch.long))]
    return Plan("role", sinks, torch.sort(torch.cat([arm, tip, wrist_fixed])).values, periphery, fixed, slices=slices,
                truncated=truncated, step_bias=step_bias)


def quad_pick(pool: torch.Tensor, n: int, grid: sel.Grid, seed: int) -> torch.Tensor:
    """anchor-neff: ``n`` tokens of ``pool`` stratified by ``role_strata`` ((frame, camera, quadrant)) in proportion to the pool
    (largest remainder, ties to the earlier stratum), uniform within a stratum (torch generator ``seed``); sorted."""
    pool = pool.cpu()
    n = min(int(n), int(pool.numel()))
    if n <= 0:
        return pool[:0]
    key = role_strata(pool, grid)
    ks = torch.unique(key).tolist()
    strata = [pool[key == k] for k in ks]
    quota = torch.tensor([float(x.numel()) for x in strata]) * n / float(pool.numel())
    take = torch.floor(quota).long()
    for i in sorted(range(len(strata)), key=lambda i: (-float(quota[i] - take[i]), i))[:n - int(take.sum())]:
        take[i] += 1
    g = torch.Generator().manual_seed(int(seed))
    return torch.sort(torch.cat([x[torch.randperm(int(x.numel()), generator=g)[:int(t)]] for x, t in zip(strata, take)])).values


def read49(last: str, grid: sel.Grid, d_meas: torch.Tensor, tips, seed: int) -> tuple[torch.Tensor, dict, dict]:
    """anchor-neff: the last step's 49 near-frame tokens and key biases. u49: each camera's near frame sampled (25 agentview, 24 wrist);
    n49: agentview the 12 measured tokens with the largest anchor change d (fixed) + 8 sampled from the rest, wrist the fingertips
    (fixed) + 27 sampled from the rest; flip49: agentview 20 sampled, wrist the fingertips + the 27 measured with the largest d (fixed).
    Sampling by ``quad_pick``; a sampled token's key + log(group size / sampled), every key + log(Kf / pf) (the far frame).
    -> (rows, bias, group sizes)."""
    pf, half = grid.per_frame, grid.width // 2
    near = torch.arange(pf)
    ag, wr = near[(near % grid.width) < half], near[(near % grid.width) >= half]
    tip = torch.tensor([int(x) for x in tips], dtype=torch.long)
    lg2 = math.log(grid.num_future / pf)
    d = d_meas.float().cpu()

    def top(pool, k):
        m = pool[torch.isfinite(d[pool])]
        return m[torch.argsort(-d[m], stable=True)][:k]

    if last == "u49":
        groups = [(ag, quad_pick(ag, 25, grid, seed), True), (wr, quad_pick(wr, 24, grid, seed + 1), True)]
    elif last == "n49":
        a_fix = top(ag, 12)
        a_rest = ag[~torch.isin(ag, a_fix)]
        w_rest = wr[~torch.isin(wr, tip)]
        groups = [(a_fix, a_fix, False), (a_rest, quad_pick(a_rest, 8, grid, seed), True), (tip, tip, False),
                  (w_rest, quad_pick(w_rest, 27, grid, seed + 1), True)]
    elif last == "flip49":
        w_fix = top(wr[~torch.isin(wr, tip)], 27)
        groups = [(ag, quad_pick(ag, 20, grid, seed), True), (tip, tip, False), (w_fix, w_fix, False)]
    else:
        raise ValueError(f"read49: {last!r}")
    rows, bias, sizes = [], {}, {}
    for i, (pool, chosen, sampled) in enumerate(groups):
        lg = lg2 + (math.log(int(pool.numel()) / int(chosen.numel())) if sampled and chosen.numel() else 0.0)
        rows.append(chosen)
        bias.update({int(j): lg for j in chosen.tolist()})
        sizes[str(i)] = [int(pool.numel()), int(chosen.numel()), bool(sampled)]
    rows = torch.sort(torch.cat(rows)).values
    if rows.numel() != 49 or torch.unique(rows).numel() != 49:
        raise RuntimeError(f"read49 {last}: {int(torch.unique(rows).numel())} distinct tokens, not 49")
    return rows, bias, sizes


BWM = ("bwm_s16", "bwm_s8", "bwm_all", "bwm_b16", "bwm_g24", "bwm_t24")  # wrist-cover: bwmid30 with another choice of the other 41
BODY_READS = ("bw20", "bw30", "br30", "bwseg30", "bwc30", "bwmid30") + BWM  # body-select: the last step's reads from the body mask
BODY_READS = BODY_READS + ("bw16", "bw8")  # sameset (2026-11-10): bw30 with 16 / 8 sampled wrist tokens (S44 / S36)
# chain2-curves (2026-11-10, task 014): Bw-W30 with one camera changed. Agentview (wrist as bw30): c2a_bg<x> reads the body and drops x %
# of the background, c2a_bd<x> reads the background and drops x % of the body (nb: no agentview bias), c2a_full both whole. Wrist
# (agentview body as bw30): c2w_s<n> n spread (quad_pick), c2w_c<n> the n nearest to a centre drawn with the call's seed.
CURVE_A = {"c2a_full": ("bg", 0.0), "c2a_bg25": ("bg", 0.25), "c2a_bg50": ("bg", 0.5), "c2a_bg75": ("bg", 0.75), "c2a_bg100": ("bg", 1.0),
           "c2a_bd25": ("bd", 0.25), "c2a_bd50": ("bd", 0.5), "c2a_bd75": ("bd", 0.75), "c2a_bd100": ("bd", 1.0), "c2a_bd100nb": ("bd", 1.0)}
CURVE_W = {"c2w_s4": ("s", 4), "c2w_s8": ("s", 8), "c2w_s16": ("s", 16), "c2w_s30": ("s", 30), "c2w_s47": ("s", 47),
           "c2w_c4": ("c", 4), "c2w_c8": ("c", 8), "c2w_c16": ("c", 16), "c2w_c30": ("c", 30)}
BODY_READS = BODY_READS + tuple(CURVE_A) + tuple(CURVE_W)
# reads that draw from the call's noise seed (body["draw_seed"], set by the runner)
CALL_SEED_READS = ("bwm_b16",) + tuple(k for k, (_, x) in CURVE_A.items() if x > 0) + tuple(k for k, (m, _) in CURVE_W.items() if m == "c")
BW_WRIST_N = {"bw20": 20, "bwmid30": 24, "bw16": 16, "bw8": 8}  # sampled wrist tokens of a body read (else 30)
WRIST_MID = (65, 66, 67, 79, 80, 81)  # body-select bwmid30: wrist rows 4-5, columns 2-4 (between and above the fingertips 78 / 82)
BODY_SOURCE = {"bwseg30": "seg", "bwc30": "fk_cover"}  # the mask each read uses (else "fk": the FK body set, dilated)
SINKS_NEAR = (84,)  # body-select: the near-frame sink of diag.late.sinks in >= 50 % of the quick-metrics samples (58 / 60)


def read_body(last: str, grid: sel.Grid, body: dict, tips, seed: int, sinks=SINKS_NEAR) -> tuple[torch.Tensor, dict, dict]:
    """body-select: the last step's near-frame tokens and key biases from the agentview body mask (``body[src]``: near-frame
    agentview token indices, from the robot's joint values + calibration (fk / fk_cover) or the segmentation render (seg)).
    bw20 / bw30 / bwseg30 / bwc30: the body, key + log(agentview tokens / body tokens) (it stands for the whole camera); br30: the
    body without a group weight + 6 sampled from the other agentview tokens (not the sinks), key + log(pool / 6). Then the sinks and
    the fingertips (fixed) and 20 / 30 wrist tokens sampled from the rest, key + log(pool / n); bwmid30: as bw30 with the six
    ``WRIST_MID`` tokens fixed as well and 24 sampled; sameset bw16 / bw8: as bw30 with 16 / 8 sampled. Sampling by ``quad_pick`` (agentview
    ``seed``, wrist ``seed`` + 1); every key + log(Kf / pf) (the far frame). -> (rows, bias, group sizes)."""
    pf, half = grid.per_frame, grid.width // 2
    near = torch.arange(pf)
    ag, wr = near[(near % grid.width) < half], near[(near % grid.width) >= half]
    lg2 = math.log(grid.num_future / pf)
    src = BODY_SOURCE.get(last, "fk")
    if body is None or src not in body:
        raise ValueError(f"read_body {last}: needs the body mask {src!r}")
    b = torch.tensor(sorted({int(j) for j in body[src]}), dtype=torch.long)
    if b.numel() == 0 or not bool(torch.isin(b, ag).all()):
        raise ValueError(f"read_body {last}: the body mask must be a non-empty set of near-frame agentview tokens, got {b.tolist()}")
    tip = torch.tensor([int(x) for x in tips], dtype=torch.long)
    if last == "bwmid30" or last in BWM:  # the six tokens between and above the fingertips are read every call too
        tip = torch.cat([tip, torch.tensor(WRIST_MID, dtype=torch.long)])
    sink = torch.tensor([int(x) for x in sinks if int(x) not in set(b.tolist())], dtype=torch.long)
    w_rest = wr[~torch.isin(wr, tip)]
    if last in CURVE_A or last in CURVE_W:
        return curve_read(last, grid, body, b, ag, tip, sink, w_rest, lg2, seed)
    n_w = BW_WRIST_N.get(last, 30)
    if last == "br30":
        pool = ag[~torch.isin(ag, b) & ~torch.isin(ag, sink)]
        pick = quad_pick(pool, 6, grid, seed)
        groups = [(b, b, 0.0), (pool, pick, math.log(int(pool.numel()) / int(pick.numel())))]
    elif last in BODY_READS:
        groups = [(ag, b, math.log(int(ag.numel()) / int(b.numel())))]
    else:
        raise ValueError(f"read_body: {last!r}")
    wp = wrist_cover_pick(last, w_rest, grid, body, seed) if last in BWM else quad_pick(w_rest, n_w, grid, seed + 1)
    groups += [(sink, sink, 0.0), (tip, tip, 0.0), (w_rest, wp, math.log(int(w_rest.numel()) / int(wp.numel())))]
    rows, bias, sizes = [], {}, {}
    for i, (pool, chosen, lg) in enumerate(groups):
        rows.append(chosen)
        bias.update({int(j): lg2 + lg for j in chosen.tolist()})
        sizes[str(i)] = [int(pool.numel()), int(chosen.numel()), round(lg, 6)]
    rows = torch.sort(torch.cat(rows)).values
    if torch.unique(rows).numel() != rows.numel() or not bool((rows < pf).all()):
        raise RuntimeError(f"read_body {last}: overlapping groups or a far-frame token in {rows.tolist()}")
    return rows, bias, sizes


def curve_read(last: str, grid: sel.Grid, body: dict, b: torch.Tensor, ag: torch.Tensor, tip: torch.Tensor, sink: torch.Tensor,
               w_rest: torch.Tensor, lg2: float, seed: int) -> tuple[torch.Tensor, dict, dict]:
    """chain2-curves: the last step's reads of Bw-W30 with one camera changed (``CURVE_A`` / ``CURVE_W``); called by ``read_body``
    with its sets (b the body, sink the sinks off the body, tip the fingertips, w_rest the other 47 wrist tokens).
    Agentview: the background G = agentview - body - sinks off the body. c2a_bg<x> reads the body and G without its first round(x |G|)
    (half up) in a random order from the generator 3 + ``body["draw_seed"]`` (the call's noise seed; nested over x); c2a_bd<x> reads
    G, the sinks inside the body and the body without its first round(x |B'|) (B' = body - sinks, same generator); every agentview read
    except the sinks off the body gets key + log(49 / its count) (c2a_bd100nb: no agentview bias); the wrist as bw30. Wrist: the body
    as bw30 (key + log(49 / |body|)); c2w_s<n>: ``quad_pick`` (seed + 1, as bw30); c2w_c<n>: the n nearest (squared Euclidean on the
    wrist grid, ties by token) to a centre drawn from w_rest with the generator 1 + draw_seed (as bwm_b16); key + log(47 / n). The sinks
    off the body and the fingertips: no bias. Every key + log(Kf / pf). -> (rows, bias, group sizes)."""
    half = grid.width // 2
    sink_set = {int(x) for x in sink.tolist()}
    if last in CURVE_A:
        kind, x = CURVE_A[last]
        g_bg = ag[~torch.isin(ag, b) & ~torch.isin(ag, sink)]
        b_sink = b[torch.isin(b, torch.tensor(SINKS_NEAR, dtype=torch.long))]  # a sink inside the body: read, never dropped
        b_del = b[~torch.isin(b, b_sink)]
        pool = g_bg if kind == "bg" else b_del
        m = int(math.floor(x * int(pool.numel()) + 0.5))
        drop = pool[:0]
        if m > 0:
            if "draw_seed" not in body:
                raise ValueError(f"{last} needs body['draw_seed'] (the call's noise seed)")
            g = torch.Generator().manual_seed(3 + int(body["draw_seed"]))
            drop = pool[torch.randperm(int(pool.numel()), generator=g)[:m]]
        kept = pool[~torch.isin(pool, drop)]
        a_read = torch.sort(torch.cat([b, kept] if kind == "bg" else [g_bg, b_sink, kept])).values
        lg_a = 0.0 if last.endswith("nb") else math.log(int(ag.numel()) / int(a_read.numel()))
        wp = quad_pick(w_rest, 30, grid, seed + 1)
        sizes = {"agent_pool": [int(pool.numel()), int(m)], "agent_read": int(a_read.numel())}
    else:
        kind, n = CURVE_W[last]
        a_read, lg_a = b, math.log(int(ag.numel()) / int(b.numel()))
        if kind == "s":
            wp = quad_pick(w_rest, n, grid, seed + 1)
        else:
            if "draw_seed" not in body:
                raise ValueError(f"{last} needs body['draw_seed'] (the call's noise seed)")
            pool = w_rest.cpu()
            g = torch.Generator().manual_seed(1 + int(body["draw_seed"]))
            centre = int(pool[int(torch.randint(int(pool.numel()), (1,), generator=g))])
            rc, cc = centre // grid.width, centre % grid.width - half
            d = sorted(((int(j) // grid.width - rc) ** 2 + (int(j) % grid.width - half - cc) ** 2, int(j)) for j in pool.tolist())
            wp = torch.sort(torch.tensor([j for _, j in d[:n]], dtype=torch.long)).values
        sizes = {"wrist_pool": int(w_rest.numel()), "wrist_read": int(wp.numel())}
    lg_w = math.log(int(w_rest.numel()) / int(wp.numel()))
    groups = [(a_read, lg_a), (sink, 0.0), (tip, 0.0), (wp, lg_w)]
    rows, bias = [], {}
    for chosen, lg in groups:
        rows.append(chosen)
        bias.update({int(j): lg2 + lg for j in chosen.tolist()})
    rows = torch.sort(torch.cat(rows)).values
    if torch.unique(rows).numel() != rows.numel() or not bool((rows < grid.per_frame).all()) or sink_set & set(a_read.tolist()):
        raise RuntimeError(f"curve_read {last}: overlapping groups or a far-frame token in {rows.tolist()}")
    return rows, bias, sizes


def pair_bias(rows, body, grid: sel.Grid) -> tuple[torch.Tensor, dict, dict]:
    """sameset P3: the video rows' key bias when the future tokens computed are only ``rows`` (near-frame tokens that include the
    agentview body set ``body``). Every absent future token gets one stand-in among the present keys: an agentview token off the
    body (near or far frame) -> the frame-0 token at the same (h, w); an agentview far-frame token at a body position -> the near-frame
    body token there; a wrist token (the near-frame ones not read and the whole far frame) -> the nearest read near-frame wrist token
    (squared Euclidean on (h, w), ties to the smaller index). Each present key gets + log(1 + the absentees it stands in for).
    -> (frame-0 bias [pf] float32, future bias {token: bias}, counts {"frame0": [pf], "future": {token: n}})."""
    pf, w, half = grid.per_frame, grid.width, grid.width // 2
    present = sorted({int(j) for j in (rows.tolist() if hasattr(rows, "tolist") else rows)})
    pset, b = set(present), {int(j) for j in body}
    if not present or any(j >= pf for j in present):
        raise ValueError(f"pair_bias: rows must be near-frame tokens, got {present}")
    if not b or not b <= pset or any(j % w >= half for j in b):
        raise ValueError("pair_bias: the body set must be non-empty agentview tokens, all of them read")
    reads_w = [j for j in present if j % w >= half]
    if not reads_w:
        raise ValueError("pair_bias: no wrist token is read")
    c0 = [0] * pf
    cf = {j: 0 for j in present}
    for t in range(grid.num_future):
        if t in pset:
            continue
        p = t % pf
        if p % w < half:
            if t >= pf and p in b:
                cf[p] += 1  # far-frame body position: the near-frame body token
            else:
                c0[p] += 1  # off the body: the current frame
        else:
            r, c = p // w, p % w
            cf[min(reads_w, key=lambda j: ((j // w - r) ** 2 + (j % w - c) ** 2, j))] += 1
    if sum(c0) + sum(cf.values()) != grid.num_future - len(present):
        raise RuntimeError("pair_bias: every absent future token needs exactly one stand-in")
    bias0 = torch.tensor([math.log(1.0 + n) for n in c0], dtype=torch.float32)
    return bias0, {j: math.log(1.0 + n) for j, n in cf.items()}, {"frame0": c0, "future": cf}


def wrist_cover_pick(last: str, pool: torch.Tensor, grid: sel.Grid, body: dict, seed: int) -> torch.Tensor:
    """wrist-cover: which of the wrist pool (the 41 near-frame wrist tokens besides the fixed 8) the last step reads. bwm_s16 / s8:
    ``quad_pick`` (seed + 1, as bwmid30); bwm_all: every one; bwm_b16: the 16 nearest (Euclidean on the wrist grid, ties by token)
    to a centre drawn from the pool with the generator 1 + ``body["draw_seed"]`` (the call's noise seed); bwm_g24 / bwm_t24: the 24
    largest of ``body["g9"]`` ({rows, g}: the last-step contribution of J98-G105 at the same call and seed) / ``body["tex"]`` (the
    wrist image's 7 x 7 Sobel texture, ``wrist_texture``), ties by token. Sorted."""
    pool = pool.cpu()
    half = grid.width // 2
    if last == "bwm_s16":
        return quad_pick(pool, 16, grid, seed + 1)
    if last == "bwm_s8":
        return quad_pick(pool, 8, grid, seed + 1)
    if last == "bwm_all":
        return torch.sort(pool).values
    if last == "bwm_b16":
        if "draw_seed" not in body:
            raise ValueError("bwm_b16 needs body['draw_seed'] (the call's noise seed)")
        g = torch.Generator().manual_seed(1 + int(body["draw_seed"]))
        centre = int(pool[int(torch.randint(int(pool.numel()), (1,), generator=g))])
        rc, cc = centre // grid.width, centre % grid.width - half
        d = [((int(j) // grid.width - rc) ** 2 + (int(j) % grid.width - half - cc) ** 2, int(j)) for j in pool.tolist()]
        return torch.sort(torch.tensor([j for _, j in sorted(d)[:16]], dtype=torch.long)).values
    if last == "bwm_g24":
        if "g9" not in body:
            raise ValueError("bwm_g24 needs body['g9'] (J98-G105's last-step g at the same call and seed)")
        gmap = dict(zip((int(j) for j in body["g9"]["rows"]), (float(v) for v in body["g9"]["g"])))
        score = {int(j): gmap[int(j)] for j in pool.tolist()}
    elif last == "bwm_t24":
        if "tex" not in body:
            raise ValueError("bwm_t24 needs body['tex'] (the wrist image's 7 x 7 texture)")
        tex = [float(v) for v in body["tex"]]
        score = {int(j): tex[(int(j) // grid.width) * half + int(j) % grid.width - half] for j in pool.tolist()}
    else:
        raise ValueError(f"wrist_cover_pick: {last!r}")
    top = sorted(score, key=lambda j: (-score[j], j))[:24]
    return torch.sort(torch.tensor(top, dtype=torch.long)).values


def wrist_texture(image: torch.Tensor, grid: sel.Grid) -> list[float]:
    """wrist-cover tex24: the model input image [1, 3, H, W] (agentview | wrist side by side, values in [-1, 1]) -> the wrist half's
    mean Sobel gradient magnitude of the luma (0.299 R + 0.587 G + 0.114 B, replicate padding) over each of its grid cells, row-major
    (grid.height x grid.width / 2)."""
    x = image[0].float()
    h, w = int(x.shape[1]), int(x.shape[2])
    half = grid.width // 2
    if h % grid.height or w % grid.width:
        raise ValueError(f"wrist_texture: image {h} x {w} is not a multiple of the token grid {grid.height} x {grid.width}")
    wr = x[:, :, w // 2:]
    y = (0.299 * wr[0] + 0.587 * wr[1] + 0.114 * wr[2])[None, None]
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=y.device)
    yp = torch.nn.functional.pad(y, (1, 1, 1, 1), mode="replicate")
    gx = torch.nn.functional.conv2d(yp, kx[None, None])
    gy = torch.nn.functional.conv2d(yp, kx.t()[None, None])
    mag = torch.sqrt(gx * gx + gy * gy)
    cells = torch.nn.functional.avg_pool2d(mag, kernel_size=(h // grid.height, (w // 2) // half))[0, 0]
    return [float(v) for v in cells.flatten().cpu()]


def body_anchor(body: dict | None, grid: sel.Grid, n: int, seed: int) -> torch.Tensor:
    """body-select A-B: the anchor's ``n`` near-frame tokens, the agentview body set (``body["fk"]``, from the joint values +
    calibration) and random non-overlapping wrist 2 x 2 blocks (windows of the wrist's near-frame grid in a torch-generator ``seed``
    order, skipping any that touches a chosen token) until n; the last block is cut to fit. Sorted."""
    if body is None or "fk" not in body:
        raise ValueError("body_anchor needs the body mask (body_rows['fk'])")
    half = grid.width // 2
    b = sorted({int(j) for j in body["fk"]})
    if not b or any(j >= grid.per_frame or j % grid.width >= half for j in b) or len(b) > n:
        raise ValueError(f"body_anchor: the body set must be 1 .. {n} near-frame agentview tokens, got {b}")
    wins = [[(r + dr) * grid.width + half + c + dc for dr in (0, 1) for dc in (0, 1)]
            for r in range(grid.height - 1) for c in range(half - 1)]
    g = torch.Generator().manual_seed(int(seed))
    chosen, have = list(b), set(b)
    for i in torch.randperm(len(wins), generator=g).tolist():
        if len(chosen) >= n:
            break
        if have & set(wins[i]):
            continue
        take = wins[i][:n - len(chosen)]
        chosen += take
        have |= set(take)
    if len(chosen) != n:
        raise RuntimeError(f"body_anchor: only {len(chosen)} tokens from the body and the wrist blocks, not {n}")
    return torch.sort(torch.tensor(chosen, dtype=torch.long)).values


def late_read_plan(plan: "Plan", last: str, grid: sel.Grid, first: int = 2, n_steps: int = 10, land: int | None = None,
                   mid: str | None = None, seed: int = 0, d_meas: torch.Tensor | None = None, tips=(78, 82),
                   body: dict | None = None) -> "Plan":
    """late-read: the late anchor's plan with nothing computed on steps ``first`` .. n_steps-2; the last step computes ``last``:
    plan (the plan's own last-step tokens and key bias), near (every near-frame token, key + log(Kf / pf)) or all (every future
    token, no bias). Steps without a token keep an empty bias, so ``bias_at`` reads ``step_bias``.
    jump-tune: ``land`` (< n_steps-1) also computes ``mid``: plus10 (the plan's own set on that step + 10 near-frame tokens it does not
    compute, ``stratified_pick`` with ``seed``, joining the slice: slice key bias log(N / n) of its new size) or core (the plan's
    sinks + core, no bias)."""
    s9 = n_steps - 1
    mid_rows, mid_bias = None, {}
    if land is not None and land < s9:
        own = plan.real(land).cpu()
        if mid == "plus10":
            near = torch.arange(grid.per_frame)
            extra = stratified_pick(near[~torch.isin(near, own)], 10, grid, seed)
            fixed = plan.fixed.cpu()
            slc = torch.cat([own[~torch.isin(own, fixed)], extra])
            mid_rows = torch.sort(torch.cat([own, extra])).values
            lg = math.log(int(plan.periphery.numel()) / int(slc.numel()))
            mid_bias = {int(j): lg for j in slc.tolist()}
        elif mid == "core":
            mid_rows = torch.sort(plan.fixed.cpu()).values
        else:
            raise ValueError(f"late_read_plan: mid must be plus10 / core, got {mid!r}")
    if last == "plan":
        rows, bias = plan.real(s9).cpu(), {int(j): float(v) for j, v in plan.bias_at(s9).items()}
    elif last == "near":
        rows = torch.arange(grid.per_frame)
        bias = {int(j): math.log(grid.num_future / grid.per_frame) for j in rows.tolist()}
    elif last == "all":
        rows, bias = torch.arange(grid.num_future), {}
    elif last in ("u49", "n49", "flip49"):
        if d_meas is None:
            raise ValueError(f"late_read_plan {last} needs the anchor's measured change d_meas")
        rows, bias, _sizes = read49(last, grid, d_meas, tips, seed)
    elif last in BODY_READS:
        rows, bias, _sizes = read_body(last, grid, body, tips, seed)
    else:
        raise ValueError(f"late_read_plan: last must be plan / near / all / u49 / n49 / flip49 / {' / '.join(BODY_READS)}, got {last!r}")
    n = n_steps - first
    empty = torch.zeros(0, dtype=torch.long)
    slices = [empty] * n
    slices[(s9 - 1) % n] = torch.sort(rows.long()).values  # real(s) reads slices[(s - 1) % n]
    step_bias = {s: {} for s in range(first, s9)}
    step_bias[s9] = bias
    if mid_rows is not None:
        slices[(land - 1) % n] = mid_rows.long()
        step_bias[land] = mid_bias
    return Plan(plan.method, plan.sinks.cpu(), plan.core.cpu(), plan.periphery.cpu(), empty, slices=slices,
                truncated=None if plan.truncated is None else plan.truncated.cpu(), step_bias=step_bias)


def tube_mask(cells: dict, grid: sel.Grid, wrist_rows: bool = False, dilate: int = 1) -> torch.Tensor:
    """Plan D: [Kf] bool, the agentview cells (row, column) of every future frame (``cells[f]``, f = 1..F-1) dilated by
    ``dilate`` within the block; ``wrist_rows`` adds the two bottom rows of every wrist block."""
    half = grid.width // 2
    m = torch.zeros(grid.num_future, dtype=torch.bool)
    for f, pts in cells.items():
        for r, c in pts:
            for rr in range(max(0, r - dilate), min(grid.height, r + dilate + 1)):
                for cc in range(max(0, c - dilate), min(half, c + dilate + 1)):
                    m[(f - 1) * grid.per_frame + rr * grid.width + cc] = True
    if wrist_rows:
        for f in range(grid.frames - 1):
            for rr in range(grid.height - 2, grid.height):
                m[f * grid.per_frame + rr * grid.width + half:f * grid.per_frame + (rr + 1) * grid.width] = True
    return m


def grid_regions(grid: sel.Grid) -> torch.Tensor:
    """[Kf] region id 0..15: ((frame - 1) * 2 + camera) * 4 + (row >= 4) * 2 + (camera column >= 4)."""
    j = torch.arange(grid.num_future)
    rem = j % grid.per_frame
    h, w = rem // grid.width, rem % grid.width
    half = grid.width // 2
    cam = (w >= half).long()
    wl = w - cam * half
    return (((j // grid.per_frame) * 2 + cam) * 4 + (h >= 4).long() * 2 + (wl >= 4).long())


def rope_freqs_at(pos: torch.Tensor, head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """RoPE rows [N, 1, head_dim / 2] (complex128) for (frame, h, w) positions [N, 3] that may be fractional, by the same
    float64 operations as ``precompute_freqs_cis_3d`` (equal to its table rows at integer positions)."""
    dims = (head_dim - 2 * (head_dim // 3), head_dim // 3, head_dim // 3)
    parts = []
    for axis, dim in enumerate(dims):
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
        f = torch.outer(pos[:, axis].double(), inv)
        parts.append(torch.polar(torch.ones_like(f), f))
    return torch.cat(parts, dim=-1).unsqueeze(1)


def kmeans(x: torch.Tensor, k: int, gen: torch.Generator, iters: int = 20) -> torch.Tensor:
    """Labels [n] of k-means (k-means++ init from ``gen``, Lloyd ``iters`` times) on rows of x [n, D] (float64)."""
    x = x.double()
    n = x.shape[0]
    k = min(k, n)
    centers = [x[int(torch.randint(n, (1,), generator=gen))]]
    for _ in range(1, k):
        d2 = torch.stack([((x - c) ** 2).sum(-1) for c in centers]).min(0).values
        if float(d2.sum()) == 0.0:
            break
        centers.append(x[int(torch.multinomial(d2 / d2.sum(), 1, generator=gen))])
    c = torch.stack(centers)
    for _ in range(iters):
        lab = torch.cdist(x, c).argmin(1)
        c = torch.stack([x[lab == i].mean(0) if bool((lab == i).any()) else c[i] for i in range(c.shape[0])])
    return torch.cdist(x, c).argmin(1)


def interlace_slices(periphery: torch.Tensor, av: torch.Tensor, n: int = N_SLICES, quota: int = R_SLOTS):
    """Periphery (ascending future indices) -> (n slices, each cut to ``quota`` by A->V, truncated tokens): the i-th token
    goes to slice i mod n, so every slice is spread over the raster."""
    slices, cut = [], []
    for s in range(n):
        part = periphery[s::n]
        if part.numel() > quota:
            order = torch.argsort(av[part].float(), descending=True, stable=True)
            cut.append(part[order[quota:]])
            part = part[order[:quota]]
        slices.append(torch.sort(part).values)
    return slices, (torch.cat(cut) if cut else periphery[:0])


def fill_map(computed: torch.Tensor, grid: sel.Grid) -> tuple[torch.Tensor, int]:
    """N2: for every future token [Kf], the position in ``computed`` (ascending future indices) of its nearest computed
    token by (row, camera column) distance in its own (future frame, camera) block, else in the same camera's other frame,
    else anywhere; ties to the smaller index. A computed token maps to itself. -> (map [Kf], uncomputed tokens filled
    from outside their block)."""
    j = torch.arange(grid.num_future)
    rem = j % grid.per_frame
    h, w = rem // grid.width, rem % grid.width
    half = grid.width // 2
    cam, frame = (w >= half).long(), j // grid.per_frame
    wl = w - cam * half
    c = computed.cpu()
    d = (h[:, None] - h[c][None]) ** 2 + (wl[:, None] - wl[c][None]) ** 2
    d = d + 1000 * (frame[:, None] != frame[c][None]).long() + 2000 * (cam[:, None] != cam[c][None]).long()
    src = torch.argmin(d, dim=1)  # first minimum = smallest future index (computed is ascending)
    out_block = (d[j, src] >= 1000) & ~torch.isin(j, c)
    return src.to(computed.device), int(out_block.sum())


def ht_frequencies(av: torch.Tensor, sinks: torch.Tensor, k: int, n: int = N_SLICES) -> torch.Tensor:
    """N4: computations f_j in {1..n} of every future token over the n pruned steps, sum n * k: the sinks n, the others
    n * min(1, max(1/n, c * a_j)) with a = A->V and c by bisection, rounded by the largest remainder (ties: A->V, index)."""
    kf = av.numel()
    f = torch.full((kf,), n, dtype=torch.long)
    is_sink = torch.zeros(kf, dtype=torch.bool)
    is_sink[sinks.cpu()] = True
    rest = torch.nonzero(~is_sink).flatten()
    target = n * k - n * int(sinks.numel())
    if target >= n * rest.numel():
        return f
    if target < rest.numel():
        raise ValueError(f"interlace_ht: {k} rows per step cannot compute every token once")
    a = av.cpu()[rest].double().clamp_min(0)
    if float(a.max()) <= 0:
        a = torch.ones_like(a)
    tot = lambda c: float(torch.clamp(n * c * a, 1, n).sum())  # noqa: E731
    lo, hi = 0.0, 1.0
    while tot(hi) < target:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tot(mid) < target else (lo, mid)
    fstar = torch.clamp(n * hi * a, 1, n)
    base = torch.floor(fstar).long()
    left = target - int(base.sum())
    frac = torch.where(base < n, fstar - base, torch.full_like(fstar, -1.0))
    o1 = torch.argsort(-a, stable=True)
    order = o1[torch.argsort(-frac[o1], stable=True)]
    if left < 0 or left > int((base < n).sum()):
        raise RuntimeError(f"interlace_ht: rounding left {left} slots")
    base[order[:left]] += 1
    f[rest] = base
    return f


def ht_schedule(f: torch.Tensor, av: torch.Tensor, n: int = N_SLICES) -> list:
    """N4: the tokens of each of the n pruned steps; token by token (f descending, then A->V, then index) on the f_j
    least loaded distinct steps, ties from a rotating pointer, so that every step gets sum(f) / n tokens."""
    kf = f.numel()
    o1 = torch.argsort(-av.cpu().double(), stable=True)
    order = o1[torch.argsort(-f.cpu()[o1], stable=True)].tolist()
    load, steps, p = [0] * n, [[] for _ in range(n)], 0
    for j in order:
        cand = sorted(range(n), key=lambda s: (load[s], (s - p) % n))[:int(f[j])]
        for s in cand:
            steps[s].append(j)
            load[s] += 1
        p = (cand[-1] + 1) % n
    if len(set(load)) != 1 or sum(load) != int(f.sum()) or kf != f.numel():
        raise RuntimeError(f"interlace_ht: unbalanced steps {load}")
    return [torch.tensor(sorted(x), dtype=torch.long, device=f.device) for x in steps]


@dataclass
class Plan:
    """What a pruned step computes: real future tokens (``real(s)``), synthetic region tokens and the key logit bias."""
    method: str
    sinks: torch.Tensor
    core: torch.Tensor
    periphery: torch.Tensor
    fixed: torch.Tensor  # real tokens computed on every pruned step
    slices: list = field(default_factory=list)  # interlace: real tokens of pruned step s = fixed + slices[s - 1]
    truncated: torch.Tensor | None = None
    bias_real: dict = field(default_factory=dict)  # future index -> log n (rep)
    regions: list = field(default_factory=list)  # fov: (members [m], log n) per synthetic token
    region_pos: torch.Tensor | None = None  # fov: [G, 3] (frame, mean h, mean w)
    slice_bias: list = field(default_factory=list)  # interlace_w: log(N / n_k) per slice
    freq: torch.Tensor | None = None  # interlace_ht: computations f_j of every future token over the pruned steps
    drop: dict = field(default_factory=dict)  # step-marginal: step -> fixed tokens that step does not compute
    step_bias: dict = field(default_factory=dict)  # role-select: step -> {token: key bias} (the rotating groups' log(N / n))

    def to(self, dev) -> "Plan":
        mv = lambda t: None if t is None else t.to(dev)  # noqa: E731
        return Plan(self.method, mv(self.sinks), mv(self.core), mv(self.periphery), mv(self.fixed),
                    [x.to(dev) for x in self.slices], mv(self.truncated), dict(self.bias_real),
                    [(g.to(dev), lg) for g, lg in self.regions], self.region_pos, list(self.slice_bias), mv(self.freq),
                    {s: t.to(dev) for s, t in self.drop.items()}, {s: dict(b) for s, b in self.step_bias.items()})

    def real(self, s: int) -> torch.Tensor:
        out = self.fixed
        if self.slices:
            out = torch.sort(torch.cat([self.fixed, self.slices[(s - 1) % len(self.slices)].to(self.fixed.device)])).values
        if s in self.drop:
            out = out[~torch.isin(out, self.drop[s].to(out.device))]
        return out

    def bias_at(self, s: int) -> dict:
        """Future index -> key logit bias on pruned step s: the representatives' log n (rep), this step's slice
        log(N / n_k) (interlace_w) or log(n / f_j) of this step's tokens computed fewer than n times (interlace_ht)."""
        if self.step_bias:
            return dict(self.step_bias.get(s, {}))
        if self.freq is not None and self.slices:
            k = (s - 1) % len(self.slices)
            return {int(j): math.log(len(self.slices) / int(self.freq[j])) for j in self.slices[k].tolist()}
        if self.slice_bias:
            k = (s - 1) % len(self.slices)
            return {int(j): self.slice_bias[k] for j in self.slices[k].tolist()}
        return self.bias_real


def make_plan(method: str, k: int, grid: sel.Grid, av: torch.Tensor, diff: torch.Tensor, recv: torch.Tensor,
              x0_future: torch.Tensor | None = None, gen: torch.Generator | None = None, off: bool = False,
              wzero: bool = False, nofar: bool = False, singleton: bool = False, sinks: torch.Tensor | None = None,
              probes: torch.Tensor | None = None, n_slices: int = N_SLICES, from_step0: bool = False) -> Plan:
    """The plan of ``method`` for budget k from the step-0 scores (A->V, Diff, received attention) [Kf]; ``off`` gives S0
    (A->V top-k, nothing else). Self-checks: ``wzero`` drops interlace_w's slice bias (= interlace), ``nofar`` lifts
    interlace_wfar's far-frame core (= interlace_w), ``singleton`` gives merge one group per future token, nothing else
    (= dense). step0-sparse: ``sinks`` given instead of the top 3 of ``recv``; ``probes`` slices the periphery by
    ``step0_slices``."""
    dev = av.device
    kf = grid.num_future
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    if off:
        top = torch.sort(sel.topk_indices(av, k)).values
        return Plan(method, empty, top, empty, top)
    if singleton:
        allf = torch.arange(kf, device=dev)
        pos = torch.stack([(allf // grid.per_frame + 1).double(), ((allf % grid.per_frame) // grid.width).double(),
                           ((allf % grid.per_frame) % grid.width).double()], dim=1).cpu()
        return Plan(method, empty, empty, allf, empty, regions=[(allf[j:j + 1], 0.0) for j in range(kf)], region_pos=pos)
    sinks = torch.sort(torch.topk(recv.float(), N_SINKS).indices if sinks is None else sinks.to(dev)).values
    if method == "interlace_ht":
        f = ht_frequencies(av, sinks, k).to(dev)
        if bool((f == N_SLICES).all()):
            return Plan(method, sinks, empty, empty, torch.arange(kf, device=dev))
        fixed = torch.nonzero(f == N_SLICES).flatten()
        slices = [x[f[x] < N_SLICES] for x in ht_schedule(f, av)]
        core = fixed[~torch.isin(fixed, sinks)]
        return Plan(method, sinks, core, torch.nonzero(f < N_SLICES).flatten(), fixed, slices=slices, truncated=empty, freq=f)
    if method == "sink":
        kept = sel.topk_indices(av, k)
        missing = [int(x) for x in sinks.tolist() if x not in set(kept.tolist())]
        if missing:
            keep_list = kept.tolist()
            removable = sorted((x for x in keep_list if x not in set(sinks.tolist())), key=lambda j: (float(av[j]), j))
            drop = set(removable[:len(missing)])
            kept = torch.tensor([x for x in keep_list if x not in drop] + missing, device=dev)
        kept = torch.sort(kept).values
        return Plan(method, sinks, kept, empty, kept)
    c = max(0, k - N_SINKS - R_SLOTS)
    far = method == "interlace_wfar" and not nofar
    excl = torch.cat([sinks, torch.arange(grid.per_frame, device=dev)]) if far else sinks  # far: near-frame tokens out
    core_set: list[int] = []
    for score, n in ((av, N_AV), (diff, N_DIFF)):
        for j in sel.topk_indices(score, min(n, c), exclude=excl).tolist():
            if j not in core_set:
                core_set.append(j)
    for j in sel.topk_indices(av, kf, exclude=excl).tolist():
        if len(core_set) >= c:
            break
        if j not in core_set:
            core_set.append(j)
    if far and len(core_set) < c:  # the far frame cannot fill the core (keep above ~50 %): the near frame by A->V
        for j in sel.topk_indices(av, kf, exclude=sinks).tolist():
            if len(core_set) >= c:
                break
            if j not in core_set:
                core_set.append(j)
    core = torch.sort(torch.tensor(core_set[:c], dtype=torch.long, device=dev)).values
    mask = torch.ones(kf, dtype=torch.bool, device=dev)
    mask[sinks] = False
    mask[core] = False
    periphery = torch.nonzero(mask).flatten()
    fixed = torch.sort(torch.cat([sinks, core])).values
    if periphery.numel() <= R_SLOTS:  # the periphery fits its slots: computed token by token
        return Plan(method, sinks, core, periphery, torch.sort(torch.cat([fixed, periphery])).values)
    if method in INTERLACE:
        slices, cut = interlace_slices(periphery, av, n=n_slices) if probes is None else step0_slices(periphery, probes, av)
        if from_step0:  # step0-sparse-r2: step s computes raster slice s (s = 0..n-1); real(s) reads slices[(s - 1) % n]
            slices = slices[1:] + slices[:1]
        weighted = method in ("interlace_w", "interlace_wfar") and not wzero  # interlace_cal's bias is the runner's
        bias = ([math.log(int(periphery.numel()) / int(x.numel())) if x.numel() else 0.0 for x in slices]
                if weighted else [])
        return Plan(method, sinks, core, periphery, fixed, slices=slices, truncated=cut, slice_bias=bias)
    if method in ("rep", "fov_grid", "merge"):
        rid = grid_regions(grid).to(dev)[periphery]
        groups = [periphery[rid == g] for g in torch.unique(rid).tolist()]
    elif method == "fov_cluster":
        rid = grid_regions(grid).to(dev)[periphery] // 4  # (frame, camera) block
        groups = []
        for b in torch.unique(rid).tolist():
            members = periphery[rid == b]
            lab = kmeans(x0_future[members], 4, gen)
            groups += [members[lab == g] for g in torch.unique(lab).tolist()]
    else:
        raise ValueError(f"unknown fovea method {method!r}")
    groups = [g for g in groups if g.numel()]
    if method == "rep":
        reps = [int(g[torch.argmax(av[g].float())]) for g in groups]
        bias = {r: math.log(int(g.numel())) for r, g in zip(reps, groups)}
        return Plan(method, sinks, core, periphery, torch.sort(torch.cat([fixed, torch.tensor(reps, device=dev)])).values,
                    bias_real=bias)
    pos = torch.stack([torch.stack([(g // grid.per_frame + 1).double().mean(),  # every member is in one future frame
                                    ((g % grid.per_frame) // grid.width).double().mean(),
                                    ((g % grid.per_frame) % grid.width).double().mean()]) for g in groups])
    return Plan(method, sinks, core, periphery, fixed, regions=[(g, math.log(int(g.numel()))) for g in groups],
                region_pos=pos)


__all__ = ["INTERLACE", "METHODS", "PROBE_CORNERS", "PROBE_SINK", "Plan", "STEP0", "add_last_extra", "fill_map", "grid_regions",
           "ht_frequencies", "ht_schedule", "interlace_slices", "interp_tokens", "kmeans", "make_plan", "probe_index",
           "roll_blocks", "rope_freqs_at", "step0_slices", "token_latent_mask", "token_norms", "tube_mask", "stratified_pick",
           "lowest_av", "apply_step_delta", "ROLE_VARIANTS", "role_strata", "spread_order", "role_plan", "BODY_READS", "BODY_SOURCE",
           "SINKS_NEAR", "WRIST_MID", "BWM", "read_body", "body_anchor", "wrist_cover_pick", "wrist_texture", "nearest_sources", "idw_fill", "align_frames", "memory_map",
           "select_core", "reselect"]
