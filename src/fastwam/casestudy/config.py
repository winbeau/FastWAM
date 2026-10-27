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

``dump`` (any method) writes one call's internals to an ``.npz``: the action query's
attention over the video tokens and the action hidden states after every layer and step,
and the decoded predicted video. Capture does not change the numerics; it is meant for a
single patched call of an episode plan.
"""

from __future__ import annotations

import hashlib
import json
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
    #: Write this call's internals to an ``.npz`` (see the module docstring).
    dump: Optional[DumpConfig] = None

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
        )
        _require(prune.selection in SELECTIONS, f"prune.selection must be one of {SELECTIONS}")
        _require(0.0 < prune.keep_ratio <= 1.0, "prune.keep_ratio must be in (0, 1]")
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
        dump=dump,
    )
    _require(cfg.score_layer is None or cfg.score_layer >= 0, "casestudy.score_layer must be >= 0")
    _require(cfg.prune is None or not cfg.record_vv or cfg.prune.warmup_steps == 1,
             "record_vv needs prune.warmup_steps == 1 (V→V is captured at step 0)")
    return cfg


def call_seed(episode_key: str, call_index: int, offset: int = 0) -> int:
    """Diffusion-noise seed for one policy call: a function of (episode, call index) only.

    Dense and sparse arms therefore draw identical noise at the same call index of the
    same episode, independent of what happened earlier in the rollout.
    """
    digest = hashlib.sha256(f"{episode_key}|{int(call_index)}|{int(offset)}".encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


__all__ = [
    "CaseStudyConfig",
    "DumpConfig",
    "GateConfig",
    "NoiseConfig",
    "PruneConfig",
    "TeaCacheConfig",
    "call_seed",
    "parse_config",
]
