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

    def to(self, dev) -> "Plan":
        mv = lambda t: None if t is None else t.to(dev)  # noqa: E731
        return Plan(self.method, mv(self.sinks), mv(self.core), mv(self.periphery), mv(self.fixed),
                    [x.to(dev) for x in self.slices], mv(self.truncated), dict(self.bias_real),
                    [(g.to(dev), lg) for g, lg in self.regions], self.region_pos, list(self.slice_bias), mv(self.freq))

    def real(self, s: int) -> torch.Tensor:
        if self.slices:
            return torch.sort(torch.cat([self.fixed, self.slices[(s - 1) % len(self.slices)].to(self.fixed.device)])).values
        return self.fixed

    def bias_at(self, s: int) -> dict:
        """Future index -> key logit bias on pruned step s: the representatives' log n (rep), this step's slice
        log(N / n_k) (interlace_w) or log(n / f_j) of this step's tokens computed fewer than n times (interlace_ht)."""
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
           "roll_blocks", "rope_freqs_at", "step0_slices", "token_latent_mask", "token_norms", "tube_mask", "nearest_sources", "idw_fill", "align_frames", "memory_map",
           "select_core", "reselect"]
