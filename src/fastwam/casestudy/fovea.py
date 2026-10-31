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
              wzero: bool = False, nofar: bool = False, singleton: bool = False) -> Plan:
    """The plan of ``method`` for budget k from the step-0 scores (A->V, Diff, received attention) [Kf]; ``off`` gives S0
    (A->V top-k, nothing else). Self-checks: ``wzero`` drops interlace_w's slice bias (= interlace), ``nofar`` lifts
    interlace_wfar's far-frame core (= interlace_w), ``singleton`` gives merge one group per future token, nothing else
    (= dense)."""
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
    sinks = torch.sort(torch.topk(recv.float(), N_SINKS).indices).values
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
        slices, cut = interlace_slices(periphery, av)
        weighted = method in ("interlace_w", "interlace_wfar") and not wzero  # interlace_cal's bias is the runner's
        bias = [math.log(int(periphery.numel()) / int(x.numel())) for x in slices] if weighted else []
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


__all__ = ["INTERLACE", "METHODS", "Plan", "fill_map", "grid_regions", "ht_frequencies", "ht_schedule", "interlace_slices",
           "kmeans", "make_plan", "rope_freqs_at"]
