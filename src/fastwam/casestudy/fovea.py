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
Pure functions; the runner applies the plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from . import selection as sel

N_SINKS, N_AV, N_DIFF, R_SLOTS, N_SLICES = 3, 10, 10, 16, 9
METHODS = ("sink", "rep", "fov_grid", "fov_cluster", "interlace")


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

    def to(self, dev) -> "Plan":
        mv = lambda t: None if t is None else t.to(dev)  # noqa: E731
        return Plan(self.method, mv(self.sinks), mv(self.core), mv(self.periphery), mv(self.fixed),
                    [x.to(dev) for x in self.slices], mv(self.truncated), dict(self.bias_real),
                    [(g.to(dev), lg) for g, lg in self.regions], self.region_pos)

    def real(self, s: int) -> torch.Tensor:
        if self.slices:
            return torch.sort(torch.cat([self.fixed, self.slices[(s - 1) % len(self.slices)].to(self.fixed.device)])).values
        return self.fixed


def make_plan(method: str, k: int, grid: sel.Grid, av: torch.Tensor, diff: torch.Tensor, recv: torch.Tensor,
              x0_future: torch.Tensor | None = None, gen: torch.Generator | None = None, off: bool = False) -> Plan:
    """The plan of ``method`` for budget k from the step-0 scores (A->V, Diff, received attention) [Kf]; ``off`` gives S0
    (A->V top-k, nothing else)."""
    dev = av.device
    kf = grid.num_future
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    if off:
        top = torch.sort(sel.topk_indices(av, k)).values
        return Plan(method, empty, top, empty, top)
    sinks = torch.sort(torch.topk(recv.float(), N_SINKS).indices).values
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
    excl = sinks
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
    core = torch.sort(torch.tensor(core_set[:c], dtype=torch.long, device=dev)).values
    mask = torch.ones(kf, dtype=torch.bool, device=dev)
    mask[sinks] = False
    mask[core] = False
    periphery = torch.nonzero(mask).flatten()
    fixed = torch.sort(torch.cat([sinks, core])).values
    if periphery.numel() <= R_SLOTS:  # the periphery fits its slots: computed token by token
        return Plan(method, sinks, core, periphery, torch.sort(torch.cat([fixed, periphery])).values)
    if method == "interlace":
        slices, cut = interlace_slices(periphery, av)
        return Plan(method, sinks, core, periphery, fixed, slices=slices, truncated=cut)
    if method in ("rep", "fov_grid"):
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


__all__ = ["METHODS", "Plan", "grid_regions", "interlace_slices", "kmeans", "make_plan", "rope_freqs_at"]
