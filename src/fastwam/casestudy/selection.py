"""Token scores, kept-set selection and the CF-Prune phase gate.

All functions work on attention probabilities averaged over heads, for batch 1:

``p_act``  [A, Kv0 + Kf + A] action-query probabilities over the joint key axis
           ``[frame-0 video | future video | action]`` (one fused softmax, as in the
           model, so the video mass is measured with the action keys in the denominator).
``p_inst`` [Kv0] cross-attention mass each frame-0 (observation) token places on the
           instruction's real (non-padding) text tokens.

Token layout follows ``rearrange("b c f h w -> b (f h w) c")``: index = f*h*w + y*w + x.
The two cameras are concatenated horizontally (agentview | wrist), so ``x < w // 2`` is
the third-person view and ``x >= w // 2`` the wrist view.

CF-Prune reference: Complementary Future Token Pruning for World Action Models,
arXiv 2610.19957. Adaptations to a joint video/action DiT with text cross-attention are
marked ADAPTATION.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Grid:
    frames: int  # latent frames including frame 0
    height: int  # tokens per column
    width: int  # tokens per row (both cameras)

    @property
    def per_frame(self) -> int:
        return self.height * self.width

    @property
    def num_future(self) -> int:
        return (self.frames - 1) * self.per_frame

    def future_view_masks(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Boolean masks over the future-token axis: (third_person, wrist)."""
        x = torch.arange(self.width, device=device).repeat(self.height)  # [per_frame]
        third = (x < self.width // 2).repeat(self.frames - 1)
        return third, ~third

    def future_frame_index(self, device) -> torch.Tensor:
        """Latent frame (1..F-1) of each future token."""
        return torch.arange(1, self.frames, device=device).repeat_interleave(self.per_frame)

    def future_obs_index(self, device) -> torch.Tensor:
        """Frame-0 token at the same spatial position as each future token."""
        return torch.arange(self.per_frame, device=device).repeat(self.frames - 1)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp_min(0)
    total = x.sum()
    return x / total if total > 0 else torch.full_like(x, 1.0 / x.numel())


def entropy(dist: torch.Tensor) -> float:
    p = _normalize(dist)
    return float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())


def future_mass(p_act: torch.Tensor, grid: Grid) -> torch.Tensor:
    """[Kf] total action-query mass on each future token."""
    start = grid.per_frame
    return p_act[:, start : start + grid.num_future].float().sum(dim=0)


def av_scores(p_act: torch.Tensor, grid: Grid) -> torch.Tensor:
    return future_mass(p_act, grid)


def cf_action_scores(p_act: torch.Tensor, grid: Grid) -> torch.Tensor:
    """CF-Prune action relevance over future tokens.

    Wrist view: action->future-frame attention, using the action queries that belong to
    that future frame's time window (actions split evenly over future latent frames).
    Third-person view: action->observation attention at the same spatial position,
    broadcast over future frames.
    ADAPTATION: the two definitions live on different scales, so each view is normalized
    to unit mass before the joint ranking.
    """
    device = p_act.device
    num_actions = p_act.shape[0]
    third, wrist = grid.future_view_masks(device)
    frame_of = grid.future_frame_index(device)
    obs_of = grid.future_obs_index(device)
    start = grid.per_frame
    fut = p_act[:, start : start + grid.num_future].float()  # [A, Kf]

    groups = torch.arange(num_actions, device=device) * (grid.frames - 1) // num_actions + 1  # [A] in 1..F-1
    in_window = groups.view(-1, 1) == frame_of.view(1, -1)  # [A, Kf]
    wrist_score = (fut * in_window).sum(dim=0)
    obs_mass = p_act[:, : grid.per_frame].float().sum(dim=0)  # [Kv0]
    third_score = obs_mass[obs_of]

    score = torch.zeros(grid.num_future, device=device)
    score[third] = _normalize(third_score[third])
    score[wrist] = _normalize(wrist_score[wrist])
    return score


def cf_instruction_scores(p_inst: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Observation->instruction relevance broadcast over future frames.

    ADAPTATION: FastWAM reads text through cross-attention, whose row always sums to 1
    over the context. The informative quantity is the share of that mass on the real
    instruction tokens (vs. zero-padding slots and the proprio token).
    """
    return p_inst.float()[grid.future_obs_index(p_inst.device)]


def topk_indices(score: torch.Tensor, k: int, exclude: torch.Tensor | None = None) -> torch.Tensor:
    s = score.float().clone()
    if exclude is not None and exclude.numel():
        s[exclude] = float("-inf")
    k = max(0, min(int(k), int(torch.isfinite(s).sum().item())))
    return torch.topk(s, k).indices


def complementary_topk(s_act: torch.Tensor, s_inst: torch.Tensor, k: int) -> torch.Tensor:
    """CF-Prune's disjoint union: half the budget by action relevance, the rest by
    instruction relevance among the tokens not already kept."""
    k_act = (int(k) + 1) // 2
    keep_act = topk_indices(s_act, k_act)
    keep_inst = topk_indices(s_inst, int(k) - keep_act.numel(), exclude=keep_act)
    return torch.cat([keep_act, keep_inst])


def random_indices(num: int, k: int, generator: torch.Generator, device) -> torch.Tensor:
    return torch.randperm(num, generator=generator)[: int(k)].to(device)


def gate_score(p_act: torch.Tensor, grid: Grid) -> dict[str, float]:
    """CF-Prune entropy contrast C = (H_third - H_wrist) + (H_early - H_late).

    High C means action attention concentrates on the wrist view and on later actions:
    a precision phase, where CF-Prune disables pruning.
    """
    device = p_act.device
    third, wrist = grid.future_view_masks(device)
    start = grid.per_frame
    fut = p_act[:, start : start + grid.num_future].float()
    mass = fut.sum(dim=0)
    half = p_act.shape[0] // 2
    h_third = entropy(mass[third])
    h_wrist = entropy(mass[wrist])
    h_early = entropy(fut[:half].sum(dim=0))
    h_late = entropy(fut[half:].sum(dim=0))
    return {
        "h_third": h_third,
        "h_wrist": h_wrist,
        "h_early": h_early,
        "h_late": h_late,
        "c_view": h_third - h_wrist,
        "c_temp": h_early - h_late,
        "c": (h_third - h_wrist) + (h_early - h_late),
    }


__all__ = [
    "Grid",
    "av_scores",
    "cf_action_scores",
    "cf_instruction_scores",
    "complementary_topk",
    "entropy",
    "future_mass",
    "gate_score",
    "random_indices",
    "topk_indices",
]
