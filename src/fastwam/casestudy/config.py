"""Configuration of the sparse-acceleration case study for FastWAM-Joint.

One ``CaseStudyConfig`` describes one experimental arm. Every arm runs the same
exact optimizations (frame-0 K/V reuse, prompt-encoding reuse) so speedups are
measured against an equally optimized dense baseline; the arms differ only in the
``method`` block.

Methods
-------
``dense``     all tokens, all steps.
``prune``     token pruning of *future* video tokens after ``warmup_steps`` full
              steps. ``selection`` picks the kept set:
                ``av``      top-k by action->video attention mass.
                ``vv``      top-k by video->video attention mass (V→V).
                ``cf``      CF-Prune (arXiv 2610.19957): complementary union of
                            action relevance and instruction relevance.
                ``random``  uniform random subset (seeded per call).
              ``gate`` (CF-Prune phase gating) switches a call to dense when the
              attention entropy-contrast score exceeds ``tau``.
``teacache``  step-level residual reuse of the whole joint stack.
``noise``     dense compute plus Gaussian noise on the action stream's attention
              output (perturbation control, no speedup).
Fewer denoising steps is ``dense`` with a smaller ``num_inference_steps``.

``prune.steer`` re-budgets the action rows' attention on pruned steps: the kept future
columns get ``lam * b`` added to their logits, ``b = log(mF / mFk)`` measured at the last
warm-up step (mF: the row's probability on all future tokens, mFk: on the kept ones), so
``lam = 1`` gives back the dense future share. With ``steer`` set the action rows of pruned
steps run in plain fp32 attention, also at ``lam = 0``.

Context probes (all but the cached K/V need ``steer``, so those arms share one path):
``video_eager`` runs the video rows of pruned steps in plain fp32 attention as well;
``steer_video`` adds ``lam * b_v`` on the kept future columns of the kept future queries,
``b_v = log(mF / mFk)`` of each query at the last warm-up step (``lam = 1`` gives the video
rows back their dense future share); ``video_full_kv`` lets the video rows of pruned steps
also see the dropped tokens' K/V of the last full step (with ``action_full_kv``: V2, which runs
on the bf16 SDPA path unless ``video_eager``); ``oracle_kv``
gives the action rows of pruned steps the kept tokens' K/V of a dense pass of the same call,
seed and step; ``dense_bias`` never prunes and adds ``steer.lam * b`` to every future column
of the action rows after warm-up (``lam = -1`` gives the sparse arm's future share with every
token present). ``steer.const`` sets b = 1 (a constant offset ``lam`` on those columns).

``recon`` (pure pruning plus reconstruction): on pruned steps the dropped positions are not
computed or cached; each pruned step infers their K/V from the frame-0 token at the same (h, w)
and they join the keys of the video rows ([frame-0 | kept | reconstructed]) and of the action rows
([frame-0 | kept | reconstructed | action]), without any bias. ``r0``: RoPE_p of the frame-0 token's
pre-RoPE key (after norm_k) and its value. ``r1``: r0 plus, per layer, step, future frame and camera
view, the channel-wise median over the kept tokens of (kept - its frame-0 token), for keys before
RoPE and for values. ``r2``: the frame-0 token's input hidden state of the layer, modulated with
this step's future-frame time modulation, through W_k / norm_k / W_v. ``keep_ratio: 0`` keeps no
future token after warm-up: V2-k0 (``action_full_kv``: the step-0 K/V of all of them) or R0-k0
(``recon: r0``). ``record_recon_quality`` runs a dense pass of the same call first and logs the
cosine of the reconstructed K / V to the dense ones at the dropped positions (per step, layer
and view), plus the RoPE / projection self-checks. ``video_sigma_shift`` replaces the shift of
the video schedule only (the action schedule keeps its own).

``agc`` (pilot + automatic gain, any method on the frame-0-cache path): the dense warm-up step's
clean action estimate ``x - sigma_0 * v_0`` (the pilot) and the arm's output are de-normalized to
simulator units; their translation (dims 0-2) and rotation (dims 3-5) sums over the executed steps
set per component the gain ``g = clip(c * |D(pilot)| / |D(out)|, lo, hi)`` (1 when either norm is
below the component's threshold), or ``const`` gives ``g = clip(gain, lo, hi)``. The whole chunk's
translation and rotation are multiplied by their gains and clipped to the environment range; the
gripper is never touched and a component with ``g = 1`` is left as it is (see ``agc.py``; applied by
the adapter). ``record_pilot`` logs the pilot without any gain (calibration of ``c``).
``prune.selection: diff`` scores each future token by the distance of the warm-up step's clean video estimate
``x - sigma_v * v`` to the current-frame latent at the same (h, w) and keeps the top ceil(k/2) of the third-person
camera and floor(k/2) of the wrist camera (``diff_global``: the top k overall); pruned steps are S0's.
``record_future_attn`` logs per step the action rows' attention on every video key, averaged over layers, heads
and queries.
``prune.action_no_future`` (S0-noF) gives the action rows of pruned steps the keys [frame-0 | action] only;
``prune.extrapolate_w`` (EX) runs the action expert a second time on pruned steps with those keys (its own
token stream; the video side is shared) and steps with ``v_s + w * (v_s - v_noF)``.

``dump`` (any method) writes one call's internals to an ``.npz``: the action query's
attention over the video tokens and the action hidden states after every layer and step,
and the decoded predicted video. Capture does not change the numerics; it is meant for a
single patched call of an episode plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

METHODS = ("dense", "prune", "teacache", "noise")
SELECTIONS = ("av", "vv", "cf", "random", "diff", "diff_global")
STEP0 = ("b", "d")
DRAFT_MODS = ("prior", "atten", "white", "low", "high", "wrist", "agent", "far", "near", "shift", "real")
MECH_KEYS = {"steps", "calls", "sets", "noise", "patch_set", "patch_ranges", "selfkey", "lens_fit", "lens_apply"}
LATE_ORACLES = ("far", "near", "wrist", "agent", "plan", "all")
ROLE_VARIANTS = ("rs", "notip", "wdiff", "aav", "nofar")  # role-select (fovea.role_plan)
FOVEA_METHODS = ("sink", "rep", "fov_grid", "fov_cluster", "interlace", "interlace_w", "interlace_wfar",
                 "interlace_cal", "interlace_fill", "merge", "interlace_ht")
LAYER_RULES = ("min_entropy", "fixed")
SEED_MODES = ("episode_call_hash", "native_fixed")


@dataclass(frozen=True)
class GateConfig:
    #: Precision phase (dense) when C = (H_third - H_wrist) + (H_early - H_late) > tau.
    tau: float


@dataclass(frozen=True)
class SteerConfig:
    #: Multiplier of the bias on the kept future columns (0 = plain sparse attention,
    #: 1 = dense future share, < 0 = less future share than sparse).
    lam: float = 0.0
    #: Upper clamp of b = log(mF / mFk), per layer, head and action row.
    clamp: float = 3.0
    #: b = 1 on every layer, head and row, so the bias is ``lam`` itself (a constant logit offset; with
    #: ``dense_bias``: every future column of the action rows after warm-up). share-sparsity round.
    const: bool = False


@dataclass(frozen=True)
class PruneConfig:
    selection: str = "av"
    #: Fraction of future video tokens kept after warm-up.
    keep_ratio: float = 0.5
    warmup_steps: int = 1
    layer_rule: str = "min_entropy"
    #: Used when ``layer_rule == "fixed"``; applies to both action and instruction scores.
    layer: Optional[int] = None
    gate: Optional[GateConfig] = None
    #: Patch P2 of the case study: recompute scores and the kept set at every pruned step
    #: from that step's own action->video attention instead of fixing it after warm-up.
    reselect_every_step: bool = False
    #: Patch P1 / P1-ctl: after selecting the kept set, also keep this fraction of the
    #: dropped tokens, chosen by A->V score (``av``) or uniformly at random (``random``).
    extra_keep_ratio: float = 0.0
    extra_selection: str = "av"
    #: Patch P3: denoising steps forced to run with all tokens (e.g. late steps only).
    full_steps: tuple[int, ...] = ()
    #: Patch P4: the denoising step whose A->V attention scores the kept set. ``None``
    #: scores at step 0, as before. Late selection sets ``warmup_steps - 1``, so the set
    #: is picked from the attention of the last warm-up step instead of the first.
    score_step: Optional[int] = None
    #: Patch P5: on pruned steps the action rows still see every future token. Kept tokens
    #: give fresh K/V; dropped ones the K/V of the last full step. Video rows and latents
    #: are pruned as usual.
    action_full_kv: bool = False
    #: Patch P6: only future tokens of these latent frames (1 = the first future frame)
    #: can be kept; the count is unchanged.
    score_frames: Optional[tuple[int, ...]] = None
    #: Explicit kept set of future-token indices (e.g. a keep-set dumped from a dense
    #: rollout). When given, ``selection`` is not consulted and no score is computed, so a
    #: dense-derived keep-set can be replayed inside a sparse rollout at that call.
    kept_index: Optional[tuple[int, ...]] = None
    #: Attention re-budget of the action rows on pruned steps (see the module docstring).
    steer: Optional[SteerConfig] = None
    #: Video rows of pruned steps in plain fp32 attention (needed by the two options below).
    video_eager: bool = False
    #: Re-budget of the kept future queries on their kept future columns (module docstring).
    steer_video: Optional[SteerConfig] = None
    #: V2 (video side): the video rows of pruned steps also see the dropped tokens' K/V of the
    #: last full step. With ``action_full_kv`` too, this is V2; alone, V2-vid. A full step
    #: listed in ``full_steps`` refreshes the cached K/V (V2 + Rk).
    video_full_kv: bool = False
    #: Action rows of pruned steps read the kept tokens' K/V from a dense pass (diagnostic).
    oracle_kv: bool = False
    #: E (summary tokens): on pruned steps the dropped tokens are replaced, as keys of the video
    #: and action rows, by one token per spatial region of dropped tokens: K and V = the mean of
    #: the region's step-0 K / V, with ``log(count)`` added to its logit (``summary_bias:
    #: count``) or, per query and head, the region's true step-0 log-sum-exp minus the mean
    #: key's logit (``lse``). Regions: every (future frame, camera view) block split
    #: ``summary_grid`` x ``summary_grid``; 0 = off. ``summary_views`` cameras side by side.
    summary_grid: int = 0
    summary_views: int = 2
    summary_bias: str = "count"
    #: Never prune; the action rows add ``steer.lam * b`` to every future column after warm-up.
    dense_bias: bool = False
    #: Reconstructed K/V of the dropped positions: r0 / r1 / r2 (module docstring); None = off.
    recon: Optional[str] = None
    #: S0-noF: on pruned steps the action rows' keys are [frame-0 | action] only (no future token).
    action_no_future: bool = False
    #: EX: on pruned steps the action expert also runs without future keys (its own token stream,
    #: keys [frame-0 | action]) and the action velocity becomes v_s + w * (v_s - v_noF).
    extrapolate_w: Optional[float] = None
    #: Self-check of the diff selections: score with A->V instead of the clean-estimate distance, through the
    #: same code path (diff_global with it must equal av bit for bit).
    diff_debug_av: bool = False
    #: Fovea round (``fovea.py``): sink | rep | fov_grid | fov_cluster | interlace on top of S0's A->V scoring (fovea-w:
    #: interlace_w | interlace_wfar; N1-N4: interlace_cal | interlace_fill | merge | interlace_ht);
    #: ``fovea_off`` keeps the method's code path but selects S0's A->V top-k only (self-check).
    fovea: Optional[str] = None
    fovea_off: bool = False
    #: fovea-w self-checks: ``fovea_wzero`` drops interlace_w's slice bias (must equal interlace), ``fovea_nofar`` lifts
    #: interlace_wfar's far-frame core (must equal interlace_w).
    fovea_wzero: bool = False
    fovea_nofar: bool = False
    #: N1-N4 self-check: merge with every future token in a group of its own and no sinks / core (must equal dense).
    fovea_singleton: bool = False
    #: step0-sparse (``fovea.py``, interlace_w only): step 0 computes the 39 probes only, every probe key logit + log(Kf / 39),
    #: the rest drafted by interpolation; plan ``b`` (interpolated A->V and Diff) or ``d`` (frame-0 attention and the
    #: end-effector tube of the pilot). ``step0_nobias`` drops the step-0 bias (E8); ``step0_all`` probes every future
    #: token without the bias and slices as M4w (self-check: = interlace_w).
    step0: Optional[str] = None
    step0_nobias: bool = False
    step0_all: bool = False
    #: step0-sparse E9 (interlace_w, dense step 0): the drafts of slices 1-6 (``early``) or 8-9 (``late``) replaced by the
    #: probe interpolation of step 0's clean estimate, from step 0's update on.
    e9: Optional[str] = None
    #: step0-sparse-r2 (interlace_w, dense main step 0): a side forward of the 39 probes on step 0's latents (the first
    #: ``side0_m`` blocks with every future token) gives B's plan (``plan``: R1-plan), step 0's outputs (``draft``: R1-draft)
    #: or both (``both``: = B for m 0; R3b, whose step-0 FLOPs are the side forward's).
    side0_m: Optional[int] = None
    side0_use: Optional[str] = None
    #: R5 / R7: after step s >= draft_from, every periphery token still to be computed takes the inverse-distance mean of the
    #: step's computed tokens' x^0 in its block as its draft, v = (x - d) / sigma_s.
    draft_from: Optional[int] = None
    #: R6 / R7: keep the call's imagination per memory key (terminal latents, last-step tokens, last A->V, step-9 recv) and log
    #: the plan it gives the next call; ``mem_route`` (R7) uses it from the second call on: plan, drafts (``mem_direct``:
    #: every terminal aligned, else interpolated from the last step's tokens), step 0 pruned, 10 slices;
    #: ``reselect_step`` (R7b) recomposes the core before that step.
    mem: bool = False
    mem_route: bool = False
    mem_direct: bool = False
    reselect_step: Optional[int] = None
    #: step0-explore D1 (exploratory, interlace_w): the step-0 drafts of every future token replaced by the dense x^0 + an error
    #: (white / low / high frequency, or white on wrist / agent / far / near tokens) with |e| = draft_mod_scale |x^0|, or by
    #: the current frame (prior); step 0's update redone.
    draft_mod: Optional[str] = None
    draft_mod_scale: float = 0.5
    #: step0-explore: the probes' change from the current frame scaled by this gain before it drafts the other tokens (step0 b / d
    #: and the side0 video / draft / both outputs; 1 = off).
    step0_gain: float = 1.0
    #: step0-explore C2F (step0 b): step 1 computes the sinks + the neighbourhoods of the c2f hottest probes by A->V and by Diff;
    #: the plan of steps 2-9 comes from the scores measured at steps 0 and 1.
    c2f: Optional[int] = None
    #: step0-explore: step 0's computed set by name (``_explore_rows``; None = the 39 probes) and how the other tokens are
    #: drafted from it: grid (bilinear from the 3 x 3 probe grid), idw (block-local inverse distance), near (idw inside the near
    #: frame, far frame = near + step0_far_lambda (near - current frame)).
    step0_rows: Optional[str] = None
    step0_draft: str = "grid"
    step0_far_lambda: float = 0.5
    #: late-anchor-c4: the far-frame extrapolation of the wrist camera (None: step0_far_lambda); step0_draft ``nearres`` adds the
    #: computed far tokens' residual to the extrapolation, spread over their far block.
    step0_far_lambda_wrist: Optional[float] = None
    #: step0-explore late anchor (step0 b): step 0 computes no future token (they stay on pure noise); step 1 computes this set
    #: (``_explore_rows``), drafts every other token from it (step0_draft, step0_gain) and gives the plan of steps 2-9.
    late: Optional[str] = None
    #: late anchor: step 9's slice also computes this many never-computed tokens (best A->V), spending the FLOPs headroom.
    #: late-anchor-confirm: also on plain interlace_w (M4w; the best never-computed tokens by step 0's A->V join step 9's slice).
    late_extra: int = 0
    #: late-anchor-confirm X-early: the step that computes ``late`` and forms the imagination: 1 (the late anchor) or 0 (step 0
    #: computes the set with key bias log(Kf / n) and drafts from it; the plan of steps 1-9 from step 0 by the same rule).
    late_step: int = 1
    #: late-anchor-c4 attribution (diagnostic, not a method): a dense step 0 beside the late anchor (FLOPs not counted) whose
    #: clean estimate (x the gain) replaces the drafts of the far / near (far extrapolated from it) / wrist / agent tokens or
    #: all of them, or whose scores give the plan of steps 2-9 (plan).
    late_oracle: Optional[str] = None
    #: late-anchor-c4 plan of the late anchor (steps 2-9): the uncomputed far tokens' A->V (copies of the near frame) and Diff
    #: (late_plan_far_diff x the near token's at the same (h, w)) scaled before the core is chosen; ``late_plan_core`` measured:
    #: the core from the computed tokens' scores only (c2f_plan core_all False).
    late_plan_far_av: Optional[float] = None
    late_plan_far_diff: Optional[float] = None
    late_plan_core: Optional[str] = None
    #: step-marginal: per-step changes of the late anchor's computed set, ((step, n), ...) with step 1-9: n > 0 computes n more
    #: tokens on that step, drawn from the tokens the step does not compute, stratified by (frame, camera) in proportion to the
    #: pool (generator seed ``step_delta_seed``); n < 0 drops |n| of the step's computed tokens, the sinks excepted, lowest plan
    #: A->V first (step 1, no A->V yet: stratified random). Step 1 changes the late set (key bias log(Kf / n) of its new size),
    #: steps 2-9 the step's slice (slice bias log(N / n_s) of its new size; dropped core tokens leave that step only). n = 0: no-op.
    step_delta: tuple = ()
    step_delta_seed: int = 0
    #: role-select: the late anchor's plan of steps 2-9 by region roles instead of A->V / Diff (``fovea.role_plan``): rs, notip, wdiff,
    #: aav, nofar; ``role_tips`` the 2 fixed wrist near-frame fingertip tokens, ``role_seed`` the rotating groups' order.
    role_select: Optional[str] = None
    role_tips: tuple = ()
    role_seed: int = 0
    #: cam-split: the late anchor spread over two steps by camera. Step 1 computes the near frame of the first camera
    #: (``cs_order`` "wa": wrist first, "aw": agentview first), step 2 the other's; ``cs_n`` tokens per camera (49: all, 42: the
    #: camera's tokens of ``late``), ``cs_k`` near-frame tokens of the other camera as context only (stratified, ``cs_seed``; no
    #: draft, no score), every row's keys + ``cs_bias`` ("0", "98": log(98 / n), "196": log(196 / n)). Each camera drafts on its
    #: anchor step (near: measured, IDW for the missing; far: extrapolated) with its gain ``cs_gain_w`` / ``cs_gain_a`` (None:
    #: step0_gain); before it its tokens stay on x^0 = 0. Steps 3-9: the late anchor's plan (c2f, measured core, 7 slices) from the
    #: two anchor steps' scores, each normalized by its step's mean over its anchor tokens.
    cs_order: Optional[str] = None
    cs_n: int = 49
    cs_k: int = 0
    cs_bias: str = "98"
    cs_gain_w: Optional[float] = None
    cs_gain_a: Optional[float] = None
    cs_seed: int = 0


@dataclass(frozen=True)
class TeaCacheConfig:
    thresh: float = 0.1
    max_consecutive_skips: int = 2
    #: Polynomial rescaling of the relative-L1 signal (numpy.poly1d order, highest first).
    #: Identity by default: published Wan2.1 fits were made on 40-50 step schedules.
    coefficients: tuple[float, ...] = (1.0, 0.0)


@dataclass(frozen=True)
class NoiseConfig:
    #: Noise std relative to the RMS of the action stream's attention output, per layer.
    sigma: float = 0.0


@dataclass(frozen=True)
class AGCConfig:
    #: ``pilot``: per component g = clip(c * |D(pilot)| / |D(out)|, lo, hi), 1 when |D(pilot)| or
    #: |D(out)| is below the component's threshold; ``const``: g_t = g_r = clip(gain, lo, hi).
    mode: str = "pilot"
    c_t: float = 1.0
    c_r: float = 1.0
    #: Thresholds on |D| (translation) and |R| (rotation), simulator units.
    tau_d: float = 0.0
    tau_r: float = 0.0
    gain: float = 1.0
    lo: float = 1.0
    hi: float = 1.5
    #: Leading chunk steps summed into D and R (the protocol's replan_steps).
    executed: int = 10
    #: Environment range of the scaled dimensions (robosuite OSC_POSE input_min / input_max).
    limit: float = 1.0
    #: Compute and log the gains but apply 1 (self-check: bitwise the arm without AGC).
    force_unit: bool = False


AGC_MODES = ("pilot", "const")


@dataclass(frozen=True)
class DumpConfig:
    #: Output ``.npz``; parent directories are created, an existing file is replaced.
    path: str
    attention: bool = True
    hidden: bool = True
    video: bool = True


@dataclass(frozen=True)
class CaseStudyConfig:
    method: str = "dense"
    num_inference_steps: int = 10
    frame0_cache: bool = True
    prompt_cache: bool = True
    seed_mode: str = "episode_call_hash"
    #: Added to the per-call seed; the D' arm (dense, resampled noise) sets this.
    seed_offset: int = 0
    prune: Optional[PruneConfig] = None
    teacache: Optional[TeaCacheConfig] = None
    noise: Optional[NoiseConfig] = None
    #: Capture A->V attention, entropies and gate score at the first step of every call,
    #: also for arms that do not use them, so all arms log the same diagnostics.
    record_attention: bool = True
    #: Layer whose attention is logged when ``method != prune`` (dense score dumps).
    #: ``None`` picks the min-entropy layer, the same rule the pruning arms use.
    score_layer: Optional[int] = None
    #: Also log the V→V mass on future tokens at the captured layer (dense score dumps
    #: and the ``vv`` selection). Requires the selection step to be step 0.
    record_vv: bool = False
    #: Log, per denoising step and layer, the action rows' attention share on future video
    #: tokens (head- and row-mean). Computed beside the forward pass on full steps.
    record_future_share: bool = False
    #: Log, per future token, the distance of the final (clean) latent to the frame-0
    #: latent at the same position: how much the predicted future moves.
    record_motion: bool = False
    #: Log, per denoising step and future query, the video rows' attention share on frame-0
    #: keys (head- and layer-mean; NaN for queries not in the step's sequence).
    record_video_frame0: bool = False
    #: Log, per denoising step and frame-1 token, the distance of that step's clean estimate
    #: x - sigma * v to the frame-0 latent at the same position.
    record_motion_steps: bool = False
    #: Write this call's internals to an ``.npz`` (see the module docstring).
    dump: Optional[DumpConfig] = None
    #: Shift of the video denoising schedule only (None = the call's sigma_shift / the model's).
    video_sigma_shift: Optional[float] = None
    #: With ``prune.recon``: reconstruction quality against a dense pass and the self-checks.
    record_recon_quality: bool = False
    #: Log the pilot (the warm-up step's clean action estimate); implied by ``agc``.
    record_pilot: bool = False
    #: Pilot + automatic gain on the output chunk (see the module docstring).
    agc: Optional[AGCConfig] = None
    #: Log per step the action rows' attention on every video key (frame 0 and future), averaged over layers, heads
    #: and action queries (NaN for keys not in the step's sequence); computed beside the forward pass.
    record_future_attn: bool = False
    #: Log the step-0 attention each future token receives (video and action rows, layer / head mean) and the top 3.
    record_sinks: bool = False
    #: fovea-w G4 (dense only): every denoising step recomputes the video stream from step 0's latents and sigma.
    video_freeze: bool = False
    #: step0-sparse E8: log the step-0 clean estimate x - sigma_0 v at the 39 probes (``fovea.probe_index``); r2: also the
    #: terminal latents there.
    record_step0_x0: bool = False
    #: step0-sparse-r2 R2: on these steps a side forward of the 39 probes (key bias) on the step's latents, its x^0 error at the
    #: probes against the main pass and the main pass's x^0 there; ``record_r2_all``: every future token, no bias (self-check).
    record_r2_steps: tuple[int, ...] = ()
    record_r2_all: bool = False
    #: R3: on step 0 side forwards with every future token on the first m blocks, the 39 probes after (one per m).
    record_r3_ms: tuple[int, ...] = ()
    #: step0-explore (dense, exploratory): one forward beside the main pass per spec on step 0 (name, rows, blocks, kv-only
    #: participants, proxies, video-row bias scale, row bias), logged with the dense step-0 x^0 and the current frame;
    #: ``explore_track`` adds the per-block hidden-state difference to the dense pass.
    explore: tuple = ()
    explore_track: bool = False
    #: late-anchor-c4: log the drafts (every future token's target clean estimate) of the late anchor (step 1) and of draft_mod.
    record_drafts: bool = False
    #: shrinkage-mechanism: the action rows' o - c_bar per step and even layer as a 64-dim random sketch (kappa_val), and m.
    record_kval: bool = False
    #: shrinkage-mechanism (dense only): diagnostic video forwards beside the pass on ``steps`` of the first ``calls`` calls
    #: of an episode: visible ``sets`` (stats), dense ``noise`` injection layer ranges, ``patch_set`` x ``patch_ranges`` path
    #: patching, ``selfkey`` sets; tuned lens ``lens_fit`` / ``lens_apply`` (a path; relative to the FastWAM checkout).
    mech: Optional[dict] = None

    def describe(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        blob = json.dumps(self.describe(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"casestudy config: {msg}")


def _known_keys(block: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(block) - allowed)
    _require(not unknown, f"unknown keys in {where}: {unknown}")


def parse_config(raw: Optional[dict]) -> CaseStudyConfig:
    """Build and validate a config from a plain dict (experiment YAML options)."""
    raw = dict(raw or {})
    _known_keys(raw, {f for f in CaseStudyConfig.__dataclass_fields__}, "casestudy")

    method = str(raw.get("method", "dense"))
    _require(method in METHODS, f"method must be one of {METHODS}, got {method!r}")
    steps = int(raw.get("num_inference_steps", 10))
    _require(steps >= 1, "num_inference_steps must be >= 1")
    seed_mode = str(raw.get("seed_mode", "episode_call_hash"))
    _require(seed_mode in SEED_MODES, f"seed_mode must be one of {SEED_MODES}")

    prune = None
    if method == "prune":
        block = dict(raw.get("prune") or {})
        _known_keys(block, {f for f in PruneConfig.__dataclass_fields__}, "casestudy.prune")
        gate = None
        steer = None
        if block.get("steer") is not None:
            steer = _steer(block["steer"], "casestudy.prune.steer")
        steer_video = None
        if block.get("steer_video") is not None:
            steer_video = _steer(block["steer_video"], "casestudy.prune.steer_video")
        if block.get("gate") is not None:
            gblock = dict(block["gate"])
            _known_keys(gblock, {"tau"}, "casestudy.prune.gate")
            _require("tau" in gblock, "prune.gate requires tau")
            gate = GateConfig(tau=float(gblock["tau"]))
        prune = PruneConfig(
            selection=str(block.get("selection", "av")),
            keep_ratio=float(block.get("keep_ratio", 0.5)),
            warmup_steps=int(block.get("warmup_steps", 1)),
            layer_rule=str(block.get("layer_rule", "min_entropy")),
            layer=None if block.get("layer") is None else int(block["layer"]),
            gate=gate,
            reselect_every_step=bool(block.get("reselect_every_step", False)),
            extra_keep_ratio=float(block.get("extra_keep_ratio", 0.0)),
            extra_selection=str(block.get("extra_selection", "av")),
            full_steps=tuple(int(x) for x in block.get("full_steps", ())),
            score_step=None if block.get("score_step") is None else int(block["score_step"]),
            action_full_kv=bool(block.get("action_full_kv", False)),
            score_frames=None if block.get("score_frames") is None else tuple(int(x) for x in block["score_frames"]),
            kept_index=None if block.get("kept_index") is None else tuple(int(x) for x in block["kept_index"]),
            steer=steer,
            video_eager=bool(block.get("video_eager", False)),
            steer_video=steer_video,
            video_full_kv=bool(block.get("video_full_kv", False)),
            oracle_kv=bool(block.get("oracle_kv", False)),
            dense_bias=bool(block.get("dense_bias", False)),
            summary_grid=int(block.get("summary_grid", 0)),
            summary_views=int(block.get("summary_views", 2)),
            summary_bias=str(block.get("summary_bias", "count")),
            recon=None if block.get("recon") is None else str(block["recon"]),
            action_no_future=bool(block.get("action_no_future", False)),
            extrapolate_w=None if block.get("extrapolate_w") is None else float(block["extrapolate_w"]),
            diff_debug_av=bool(block.get("diff_debug_av", False)),
            fovea=None if block.get("fovea") is None else str(block["fovea"]),
            fovea_off=bool(block.get("fovea_off", False)),
            fovea_wzero=bool(block.get("fovea_wzero", False)),
            fovea_nofar=bool(block.get("fovea_nofar", False)),
            fovea_singleton=bool(block.get("fovea_singleton", False)),
            step0=None if block.get("step0") is None else str(block["step0"]),
            step0_nobias=bool(block.get("step0_nobias", False)),
            step0_all=bool(block.get("step0_all", False)),
            e9=None if block.get("e9") is None else str(block["e9"]),
            side0_m=None if block.get("side0_m") is None else int(block["side0_m"]),
            side0_use=None if block.get("side0_use") is None else str(block["side0_use"]),
            draft_from=None if block.get("draft_from") is None else int(block["draft_from"]),
            mem=bool(block.get("mem", False)),
            mem_route=bool(block.get("mem_route", False)),
            mem_direct=bool(block.get("mem_direct", False)),
            reselect_step=None if block.get("reselect_step") is None else int(block["reselect_step"]),
            draft_mod=None if block.get("draft_mod") is None else str(block["draft_mod"]),
            draft_mod_scale=float(block.get("draft_mod_scale", 0.5)),
            step0_gain=float(block.get("step0_gain", 1.0)),
            c2f=None if block.get("c2f") is None else int(block["c2f"]),
            step0_rows=None if block.get("step0_rows") is None else str(block["step0_rows"]),
            step0_draft=str(block.get("step0_draft", "grid")),
            step0_far_lambda=float(block.get("step0_far_lambda", 0.5)),
            step0_far_lambda_wrist=None if block.get("step0_far_lambda_wrist") is None else float(block["step0_far_lambda_wrist"]),
            late=None if block.get("late") is None else str(block["late"]),
            late_extra=int(block.get("late_extra", 0)),
            late_step=int(block.get("late_step", 1)),
            late_oracle=None if block.get("late_oracle") is None else str(block["late_oracle"]),
            late_plan_far_av=None if block.get("late_plan_far_av") is None else float(block["late_plan_far_av"]),
            late_plan_far_diff=None if block.get("late_plan_far_diff") is None else float(block["late_plan_far_diff"]),
            late_plan_core=None if block.get("late_plan_core") is None else str(block["late_plan_core"]),
            step_delta=tuple(sorted((int(s_), int(n_)) for s_, n_ in (block.get("step_delta") or {}).items())),
            step_delta_seed=int(block.get("step_delta_seed", 0)),
            role_select=None if block.get("role_select") is None else str(block["role_select"]),
            role_tips=tuple(int(x) for x in (block.get("role_tips") or ())),
            role_seed=int(block.get("role_seed", 0)),
            cs_order=None if block.get("cs_order") is None else str(block["cs_order"]),
            cs_n=int(block.get("cs_n", 49)),
            cs_k=int(block.get("cs_k", 0)),
            cs_bias=str(block.get("cs_bias", "98")),
            cs_gain_w=None if block.get("cs_gain_w") is None else float(block["cs_gain_w"]),
            cs_gain_a=None if block.get("cs_gain_a") is None else float(block["cs_gain_a"]),
            cs_seed=int(block.get("cs_seed", 0)),
        )
        if prune.fovea is not None or prune.fovea_off:
            _require(prune.fovea in FOVEA_METHODS, f"prune.fovea must be one of {FOVEA_METHODS} (fovea_off needs a method)")
            _require(not prune.fovea_wzero or prune.fovea in ("interlace_w", "interlace_wfar"),
                     "prune.fovea_wzero is for interlace_w / interlace_wfar")
            _require(not prune.fovea_nofar or prune.fovea == "interlace_wfar", "prune.fovea_nofar is for interlace_wfar")
            _require(not prune.fovea_singleton or (prune.fovea == "merge" and not prune.fovea_off),
                     "prune.fovea_singleton is for merge (without fovea_off)")
            _require(prune.step0 is None or (prune.step0 in STEP0 and prune.fovea == "interlace_w" and not prune.fovea_off
                                             and not prune.fovea_wzero), "prune.step0 is b or d on interlace_w")
            _require(prune.e9 is None or (prune.e9 in ("early", "late") and prune.fovea == "interlace_w"
                                          and not prune.fovea_off and not prune.fovea_wzero and prune.step0 is None),
                     "prune.e9 is early or late on interlace_w (dense step 0)")
            _require(prune.selection == "av" and prune.warmup_steps == 1 and prune.keep_ratio > 0.0 and not (
                prune.action_full_kv or prune.video_full_kv or prune.summary_grid or prune.oracle_kv or prune.recon is not None
                or prune.steer is not None or prune.steer_video is not None or prune.dense_bias or prune.video_eager
                or prune.full_steps or prune.reselect_every_step or prune.gate is not None or prune.extra_keep_ratio
                or prune.kept_index is not None or prune.score_frames is not None or prune.score_step is not None
                or prune.action_no_future or prune.extrapolate_w is not None),
                "prune.fovea is S0 (selection av, warm-up 1, keep > 0) plus the fovea plan only")
        _require(not (prune.step0_nobias or prune.step0_all) or prune.step0 is not None,
                 "prune.step0_nobias / step0_all need prune.step0")
        _require(not prune.step0_all or prune.step0 == "b", "prune.step0_all is the plan-b self-check")
        _require((prune.e9 is None and prune.step0 is None) or prune.fovea is not None,
                 "prune.step0 / prune.e9 need prune.fovea interlace_w")
        r2 = (prune.side0_m is not None or prune.side0_use is not None or prune.draft_from is not None or prune.mem
              or prune.mem_route or prune.mem_direct or prune.reselect_step is not None)
        _require(not r2 or (prune.fovea == "interlace_w" and not prune.fovea_off and not prune.fovea_wzero
                            and prune.step0 is None and prune.e9 is None),
                 "the step0-sparse-r2 options are for interlace_w (no step0 / e9)")
        _require((prune.side0_m is None) == (prune.side0_use is None)
                 and (prune.side0_use is None or (prune.side0_use in ("plan", "draft", "both", "video", "action") and prune.side0_m >= 0)),
                 "prune.side0_m (>= 0) and side0_use (plan / draft / both / video / action) go together")
        _require(prune.draft_mod is None or (prune.draft_mod in DRAFT_MODS and prune.fovea == "interlace_w" and not r2
                                              and prune.step0 is None and prune.e9 is None and prune.draft_mod_scale >= 0),
                 f"prune.draft_mod must be one of {DRAFT_MODS} on plain interlace_w")
        _require(prune.step0_gain == 1.0 or (prune.step0_gain > 0 and (prune.step0 is not None or prune.side0_use in ("draft", "both", "video"))),
                 "prune.step0_gain (> 0) is for step0 or the side0 video / draft / both outputs")
        _require(prune.step0_draft in ("grid", "idw", "near", "nearres") and (prune.step0 is not None or (prune.step0_rows is None
                 and prune.step0_draft == "grid")) and (prune.step0_draft != "grid" or prune.step0_rows in (None, "grid39")),
                 "prune.step0_rows / step0_draft (grid needs the probe grid; idw / near any set) are step0 options")
        _require(prune.late_extra >= 0 and (prune.late_extra == 0 or prune.late is not None or (
            prune.fovea == "interlace_w" and not prune.fovea_off and not prune.fovea_wzero and prune.step0 is None and prune.e9 is None
            and not r2)), "prune.late_extra (>= 0) needs prune.late or plain interlace_w")
        _require(prune.late_step in (0, 1) and (prune.late_step == 1 or prune.late is not None), "prune.late_step (0 or 1) needs prune.late")
        _require(prune.step0_far_lambda_wrist is None or prune.step0_draft in ("near", "nearres"),
                 "prune.step0_far_lambda_wrist needs near / nearres drafts")
        _require((prune.late_plan_far_av is None and prune.late_plan_far_diff is None and prune.late_plan_core is None) or (
            prune.late is not None and prune.late_step == 1 and prune.late_oracle != "plan" and (prune.late_plan_core in (None, "measured"))
            and all(x is None or x >= 0 for x in (prune.late_plan_far_av, prune.late_plan_far_diff))),
                 "prune.late_plan_far_av / far_diff (>= 0) / late_plan_core (measured) are options of the late anchor's plan")
        _require(prune.cs_order is None or (prune.cs_order in ("wa", "aw") and prune.late is not None and prune.late_step == 1
                 and prune.late_oracle is None and not prune.step_delta and prune.role_select is None and prune.late_plan_far_av is None
                 and prune.late_plan_far_diff is None and prune.step0_draft in ("near", "idw") and prune.cs_n in (42, 49)
                 and 0 <= prune.cs_k <= 49 and prune.cs_bias in ("0", "98", "196")
                 and all(g is None or g > 0 for g in (prune.cs_gain_w, prune.cs_gain_a))),
                 "prune.cs_order (wa / aw; cs_n 42 / 49, cs_k 0-49, cs_bias 0 / 98 / 196) spreads the late anchor (late_step 1, no oracle, "
                 "step_delta, role_select or far scaling) over steps 1-2 by camera")
        _require(prune.role_select is None or (prune.role_select in ROLE_VARIANTS and prune.late is not None and prune.late_step == 1
                 and prune.late_oracle is None and not prune.step_delta and prune.late_plan_far_av is None and prune.late_plan_far_diff is None
                 and (prune.role_select == "notip" or len(prune.role_tips) == 2)), f"prune.role_select ({ROLE_VARIANTS}, 2 role_tips but for notip) "
                 "replaces the late anchor's plan of steps 2-9 (late_step 1, no oracle, no step_delta, no far scaling)")
        _require(not prune.step_delta or (prune.late is not None and prune.late_step == 1 and prune.late_oracle is None
                                          and all(1 <= s_ <= 9 for s_, _ in prune.step_delta)
                                          and len({s_ for s_, _ in prune.step_delta}) == len(prune.step_delta)),
                 "prune.step_delta ({step 1-9: n}) is an option of the late anchor (late_step 1)")
        _require(prune.late_oracle is None or (prune.late_oracle in LATE_ORACLES and prune.late is not None and prune.late_step == 1),
                 f"prune.late_oracle must be one of {LATE_ORACLES} on the late anchor (late_step 1)")
        _require(prune.late is None or (prune.step0 == "b" and prune.step0_draft in ("idw", "near", "nearres") and prune.c2f is None
                                        and not prune.step0_all and prune.step0_rows is None), "prune.late needs step0 b with idw / near drafts")
        _require(prune.c2f is None or (prune.step0 == "b" and not prune.step0_all and prune.c2f >= 1), "prune.c2f (>= 1) needs prune.step0 b")
        _require(prune.draft_from is None or 0 <= prune.draft_from <= 10, "prune.draft_from must be in [0, 10]")
        _require(not (prune.mem_route or prune.mem_direct) or prune.mem, "prune.mem_route / mem_direct need prune.mem")
        _require(prune.reselect_step is None or (prune.mem_route and 1 <= prune.reselect_step < 10),
                 "prune.reselect_step (1..9) needs prune.mem_route")
        _require(not (prune.mem and prune.side0_use is not None), "prune.mem and prune.side0 are exclusive")
        _require(prune.selection in SELECTIONS, f"prune.selection must be one of {SELECTIONS}")
        if prune.selection in ("diff", "diff_global"):
            _require(prune.warmup_steps == 1 and prune.score_step is None and prune.kept_index is None
                     and prune.score_frames is None and not prune.reselect_every_step and prune.extra_keep_ratio == 0.0
                     and prune.gate is None, "the diff selections score once at the warm-up step 0 (no other selection option)")
        _require(not prune.diff_debug_av or prune.selection in ("diff", "diff_global"), "prune.diff_debug_av needs a diff selection")
        _require(0.0 <= prune.keep_ratio <= 1.0, "prune.keep_ratio must be in [0, 1]")
        _require(1 <= prune.warmup_steps < steps, "prune.warmup_steps must be in [1, num_inference_steps)")
        _require(prune.layer_rule in LAYER_RULES, f"prune.layer_rule must be one of {LAYER_RULES}")
        _require(prune.layer_rule != "fixed" or prune.layer is not None, "layer_rule=fixed needs prune.layer")
        _require(prune.gate is None or prune.selection == "cf", "phase gating is defined for selection=cf only")
        _require(0.0 <= prune.extra_keep_ratio < 1.0, "prune.extra_keep_ratio must be in [0, 1)")
        _require(prune.extra_selection in ("av", "random"), "prune.extra_selection must be av or random")
        _require(all(0 <= x < steps for x in prune.full_steps), "prune.full_steps out of range")
        if prune.score_step is not None:
            _require(0 <= prune.score_step < prune.warmup_steps,
                     "prune.score_step must be in [0, warmup_steps): the kept set is scored on a full step")
            _require(prune.selection in ("av", "vv", "cf"), "prune.score_step needs an attention-scored selection")
            _require(prune.kept_index is None, "prune.score_step and prune.kept_index are exclusive")
        if prune.score_frames is not None:
            _require(len(prune.score_frames) > 0 and all(x >= 1 for x in prune.score_frames),
                     "prune.score_frames lists future latent frames, 1-based")
            _require(prune.selection == "av", "prune.score_frames is defined for selection=av")
            _require(prune.kept_index is None, "prune.score_frames and prune.kept_index are exclusive")
        if prune.kept_index is not None:
            _require(len(set(prune.kept_index)) == len(prune.kept_index), "prune.kept_index has duplicates")
            _require(all(x >= 0 for x in prune.kept_index), "prune.kept_index must be non-negative")
            _require(0 < len(prune.kept_index), "prune.kept_index must not be empty")
        for name, st in (("steer", prune.steer), ("steer_video", prune.steer_video)):
            if st is not None:
                _require(math.isfinite(st.lam), f"prune.{name}.lam must be finite")
                _require(st.clamp > 0, f"prune.{name}.clamp must be > 0")
                _require(not prune.reselect_every_step, f"prune.{name} needs a kept set fixed after warm-up")
                _require(prune.warmup_steps == 1, f"prune.{name} needs warmup_steps == 1")
                _require(not st.const or (name == "steer" and prune.dense_bias), f"prune.{name}.const needs prune.dense_bias")
        new_path = prune.video_eager or prune.steer_video is not None or prune.oracle_kv or prune.dense_bias
        if new_path:
            _require(prune.steer is not None, "the context probes need prune.steer (the shared action-row path)")
            _require(prune.selection != "vv" and prune.kept_index is None and prune.extra_keep_ratio == 0.0,
                     "the context probes are defined for an attention-scored kept set without extras")
        _require(prune.steer_video is None or prune.video_eager, "prune.steer_video needs prune.video_eager")
        cache = prune.action_full_kv or prune.video_full_kv
        _require(not (cache and prune.reselect_every_step), "cached dropped K/V needs a kept set fixed after warm-up")
        _require(not (prune.oracle_kv and cache), "prune.oracle_kv and the cached dropped K/V are exclusive")
        _require(prune.summary_grid >= 0 and prune.summary_views >= 1, "prune.summary_grid >= 0, summary_views >= 1")
        _require(prune.summary_bias in ("count", "lse"), "prune.summary_bias must be count or lse")
        if prune.summary_grid:
            _require(not (cache or prune.oracle_kv or prune.dense_bias or prune.video_eager or prune.steer is not None
                          or prune.steer_video is not None or prune.full_steps or prune.reselect_every_step
                          or prune.gate is not None),
                     "prune.summary_grid is S0 plus the summary keys (no other option)")
            _require(prune.warmup_steps == 1, "prune.summary_grid needs warmup_steps == 1 (step-0 K/V)")
        if prune.dense_bias:
            _require(not (prune.video_eager or prune.steer_video is not None or cache or prune.oracle_kv
                          or prune.full_steps or prune.gate is not None),
                     "prune.dense_bias is plain dense compute plus the action-row bias")
        _require(prune.recon in (None, "r0", "r1", "r2"), "prune.recon must be r0, r1 or r2")
        if prune.recon is not None:
            _require(not (cache or prune.oracle_kv or prune.dense_bias or prune.video_eager or prune.steer is not None
                          or prune.steer_video is not None or prune.full_steps or prune.reselect_every_step
                          or prune.gate is not None or prune.summary_grid),
                     "prune.recon is S0 plus the reconstructed keys (no other option)")
        if prune.action_no_future or prune.extrapolate_w is not None:
            _require(not (prune.action_no_future and prune.extrapolate_w is not None),
                     "prune.action_no_future (S0-noF) and prune.extrapolate_w (EX) are exclusive")
            _require(prune.extrapolate_w is None or math.isfinite(prune.extrapolate_w), "prune.extrapolate_w must be finite")
            _require(not (cache or prune.oracle_kv or prune.dense_bias or prune.video_eager or prune.steer is not None
                          or prune.steer_video is not None or prune.full_steps or prune.reselect_every_step
                          or prune.gate is not None or prune.summary_grid or prune.recon is not None or prune.keep_ratio == 0.0),
                     "S0-noF / EX are S0 plus the no-future action keys (no other option)")
        if prune.keep_ratio == 0.0:
            _require(prune.recon == "r0" or (prune.action_full_kv and not prune.video_eager and prune.steer is None),
                     "keep_ratio 0 is V2-k0 (action_full_kv) or R0-k0 (recon r0)")
            _require(prune.selection == "av" and prune.extra_keep_ratio == 0.0 and prune.kept_index is None
                     and not prune.reselect_every_step and prune.gate is None and not prune.full_steps
                     and not prune.summary_grid and not prune.oracle_kv and not prune.dense_bias,
                     "keep_ratio 0 takes no other option")
    else:
        _require(raw.get("prune") is None, "prune block given but method != prune")

    teacache = None
    if method == "teacache":
        block = dict(raw.get("teacache") or {})
        _known_keys(block, {f for f in TeaCacheConfig.__dataclass_fields__}, "casestudy.teacache")
        teacache = TeaCacheConfig(
            thresh=float(block.get("thresh", 0.1)),
            max_consecutive_skips=int(block.get("max_consecutive_skips", 2)),
            coefficients=tuple(float(c) for c in block.get("coefficients", (1.0, 0.0))),
        )
        _require(teacache.thresh >= 0.0, "teacache.thresh must be >= 0")
        _require(teacache.max_consecutive_skips >= 0, "teacache.max_consecutive_skips must be >= 0")
    else:
        _require(raw.get("teacache") is None, "teacache block given but method != teacache")

    noise = None
    if method == "noise":
        block = dict(raw.get("noise") or {})
        _known_keys(block, {"sigma"}, "casestudy.noise")
        noise = NoiseConfig(sigma=float(block.get("sigma", 0.0)))
        _require(noise.sigma >= 0.0, "noise.sigma must be >= 0")
    else:
        _require(raw.get("noise") is None, "noise block given but method != noise")

    dump = None
    if raw.get("dump") is not None:
        block = dict(raw["dump"])
        _known_keys(block, {f for f in DumpConfig.__dataclass_fields__}, "casestudy.dump")
        _require(bool(block.get("path")), "casestudy.dump needs a path")
        dump = DumpConfig(path=str(block["path"]), attention=bool(block.get("attention", True)),
                          hidden=bool(block.get("hidden", True)), video=bool(block.get("video", True)))
        _require(dump.path.endswith(".npz"), "casestudy.dump.path must end in .npz")

    agc = None
    if raw.get("agc") is not None:
        block = dict(raw["agc"])
        _known_keys(block, {f for f in AGCConfig.__dataclass_fields__}, "casestudy.agc")
        agc = AGCConfig(
            mode=str(block.get("mode", "pilot")),
            c_t=float(block.get("c_t", 1.0)), c_r=float(block.get("c_r", 1.0)),
            tau_d=float(block.get("tau_d", 0.0)), tau_r=float(block.get("tau_r", 0.0)),
            gain=float(block.get("gain", 1.0)), lo=float(block.get("lo", 1.0)), hi=float(block.get("hi", 1.5)),
            executed=int(block.get("executed", 10)), limit=float(block.get("limit", 1.0)),
            force_unit=bool(block.get("force_unit", False)),
        )
        _require(agc.mode in AGC_MODES, f"agc.mode must be one of {AGC_MODES}")
        _require(0.0 < agc.lo <= agc.hi and math.isfinite(agc.hi), "agc needs 0 < lo <= hi < inf")
        _require(agc.c_t > 0 and agc.c_r > 0 and agc.gain > 0, "agc.c_t, agc.c_r and agc.gain must be > 0")
        _require(agc.tau_d >= 0 and agc.tau_r >= 0, "agc.tau_d and agc.tau_r must be >= 0")
        _require(agc.executed >= 1 and agc.limit > 0, "agc.executed >= 1 and agc.limit > 0")

    cfg = CaseStudyConfig(
        method=method,
        num_inference_steps=steps,
        frame0_cache=bool(raw.get("frame0_cache", True)),
        prompt_cache=bool(raw.get("prompt_cache", True)),
        seed_mode=seed_mode,
        seed_offset=int(raw.get("seed_offset", 0)),
        prune=prune,
        teacache=teacache,
        noise=noise,
        record_attention=bool(raw.get("record_attention", True)),
        score_layer=None if raw.get("score_layer") is None else int(raw["score_layer"]),
        record_vv=bool(raw.get("record_vv", False)),
        record_future_share=bool(raw.get("record_future_share", False)),
        record_motion=bool(raw.get("record_motion", False)),
        record_video_frame0=bool(raw.get("record_video_frame0", False)),
        record_motion_steps=bool(raw.get("record_motion_steps", False)),
        dump=dump,
        video_sigma_shift=None if raw.get("video_sigma_shift") is None else float(raw["video_sigma_shift"]),
        record_recon_quality=bool(raw.get("record_recon_quality", False)),
        record_pilot=bool(raw.get("record_pilot", False)),
        agc=agc,
        record_future_attn=bool(raw.get("record_future_attn", False)),
        record_sinks=bool(raw.get("record_sinks", False)),
        video_freeze=bool(raw.get("video_freeze", False)),
        record_step0_x0=bool(raw.get("record_step0_x0", False)),
        record_r2_steps=tuple(int(x) for x in raw.get("record_r2_steps") or ()),
        record_r2_all=bool(raw.get("record_r2_all", False)),
        record_r3_ms=tuple(int(x) for x in raw.get("record_r3_ms") or ()),
        explore=tuple((str(e["name"]), str(e.get("rows", "grid39")), str(e.get("layers", "all")), bool(e.get("kv", False)),
                       int(e.get("proxies", 0)), float(e.get("vscale", 1.0)), None if e.get("bias") is None else float(e["bias"]))
                      for e in raw.get("explore") or ()),
        explore_track=bool(raw.get("explore_track", False)),
        record_drafts=bool(raw.get("record_drafts", False)),
        record_kval=bool(raw.get("record_kval", False)),
        mech=None if raw.get("mech") is None else dict(raw["mech"]),
    )
    _require(cfg.mech is None or (cfg.method == "dense" and cfg.frame0_cache and set(cfg.mech) <= MECH_KEYS
                                  and not (cfg.mech.get("lens_fit") and cfg.mech.get("lens_apply"))),
             f"casestudy.mech is a dense diagnostic with keys {sorted(MECH_KEYS)} (lens_fit or lens_apply, not both)")
    p = cfg.prune
    _require(p is None or p.fovea in (None, "sink") or p.fovea_off or not (
        cfg.record_future_attn or cfg.record_video_frame0 or cfg.record_motion_steps
        or (cfg.record_future_share and p.fovea not in ("interlace", "interlace_w", "interlace_wfar", "interlace_cal",
                                                         "interlace_fill", "merge", "interlace_ht"))),
        "the per-key / per-query records are defined for fovea sink (or fovea_off) only; future share also for interlace* "
        "and merge")
    _require(not cfg.video_freeze or (cfg.method == "dense" and p is None and cfg.frame0_cache),
             "video_freeze is a dense self-check (casestudy path)")
    _require(not cfg.record_future_attn or p is None or not (
        p.action_full_kv or p.video_full_kv or p.summary_grid or p.oracle_kv or p.recon is not None or p.steer is not None
        or p.dense_bias or p.action_no_future or p.extrapolate_w is not None),
        "record_future_attn is defined on the plain dense / S0 key layout")
    _require(not (cfg.record_pilot or cfg.agc is not None) or cfg.frame0_cache,
             "the pilot (record_pilot, agc) is logged on the frame-0-cache path only")
    _require(cfg.video_sigma_shift is None or cfg.video_sigma_shift > 0, "casestudy.video_sigma_shift must be > 0")
    _require(not cfg.record_recon_quality or (cfg.prune is not None and cfg.prune.recon is not None),
             "record_recon_quality needs prune.recon")
    _require(cfg.dump is None or cfg.prune is None or cfg.prune.recon is None, "dump is not defined for prune.recon")
    _require(cfg.score_layer is None or cfg.score_layer >= 0, "casestudy.score_layer must be >= 0")
    _require(p is None or p.step0 is None or cfg.score_layer is not None,
             "prune.step0 scores on a fixed layer (casestudy.score_layer)")
    _require(not cfg.record_step0_x0 or cfg.frame0_cache, "record_step0_x0 is logged on the frame-0-cache path only")
    _require(p is None or not (p.side0_use is not None or p.mem) or cfg.score_layer is not None,
             "prune.side0 / mem score on a fixed layer (casestudy.score_layer)")
    _require(not (cfg.record_r2_steps or cfg.record_r3_ms) or (cfg.method == "dense" and p is None and cfg.frame0_cache),
             "record_r2_steps / record_r3_ms are dense diagnostics")
    _require(all(0 <= s < steps for s in cfg.record_r2_steps) and all(m >= 0 for m in cfg.record_r3_ms),
             "record_r2_steps in [0, steps), record_r3_ms >= 0")
    _require(not (cfg.explore or cfg.explore_track) or (cfg.method == "dense" and p is None and cfg.frame0_cache and cfg.record_sinks),
             "explore is a dense diagnostic with record_sinks")
    _require(cfg.prune is None or not cfg.record_vv or cfg.prune.warmup_steps == 1,
             "record_vv needs prune.warmup_steps == 1 (V→V is captured at step 0)")
    return cfg


def _steer(raw: Any, where: str) -> SteerConfig:
    block = dict(raw)
    _known_keys(block, {f for f in SteerConfig.__dataclass_fields__}, where)
    return SteerConfig(lam=float(block.get("lam", 0.0)), clamp=float(block.get("clamp", 3.0)),
                       const=bool(block.get("const", False)))


def call_seed(episode_key: str, call_index: int, offset: int = 0) -> int:
    """Diffusion-noise seed for one policy call: a function of (episode, call index) only.

    Dense and sparse arms therefore draw identical noise at the same call index of the
    same episode, independent of what happened earlier in the rollout.
    """
    digest = hashlib.sha256(f"{episode_key}|{int(call_index)}|{int(offset)}".encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


__all__ = [
    "AGCConfig",
    "CaseStudyConfig",
    "DumpConfig",
    "GateConfig",
    "NoiseConfig",
    "PruneConfig",
    "SteerConfig",
    "TeaCacheConfig",
    "call_seed",
    "parse_config",
]
