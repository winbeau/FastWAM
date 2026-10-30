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
token present).

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
SELECTIONS = ("av", "vv", "cf", "random")
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
        )
        _require(prune.selection in SELECTIONS, f"prune.selection must be one of {SELECTIONS}")
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
    )
    _require(not (cfg.record_pilot or cfg.agc is not None) or cfg.frame0_cache,
             "the pilot (record_pilot, agc) is logged on the frame-0-cache path only")
    _require(cfg.video_sigma_shift is None or cfg.video_sigma_shift > 0, "casestudy.video_sigma_shift must be > 0")
    _require(not cfg.record_recon_quality or (cfg.prune is not None and cfg.prune.recon is not None),
             "record_recon_quality needs prune.recon")
    _require(cfg.dump is None or cfg.prune is None or cfg.prune.recon is None, "dump is not defined for prune.recon")
    _require(cfg.score_layer is None or cfg.score_layer >= 0, "casestudy.score_layer must be >= 0")
    _require(cfg.prune is None or not cfg.record_vv or cfg.prune.warmup_steps == 1,
             "record_vv needs prune.warmup_steps == 1 (V→V is captured at step 0)")
    return cfg


def _steer(raw: Any, where: str) -> SteerConfig:
    block = dict(raw)
    _known_keys(block, {f for f in SteerConfig.__dataclass_fields__}, where)
    return SteerConfig(lam=float(block.get("lam", 0.0)), clamp=float(block.get("clamp", 3.0)))


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
