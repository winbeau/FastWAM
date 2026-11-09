"""action-eval adapter for FastWAM-Joint (sparse-acceleration case study).

Thin wiring of the platform contract onto FastWAM's own LIBERO pipeline. The model is
built exactly as ``experiments/libero/eval_libero_single.py`` builds it (Hydra
``sim_libero`` + ``task=libero_joint_2cam224_1e-4``) and the observation/action
transforms are the native ones, shared through ``experiments/libero/policy_io.py``.

===========================================  ==========================
concern                                      owner
===========================================  ==========================
image flip                                   action-eval ``libero/backend.py``
center-crop, resize, camera concatenation    FastWAM ``policy_io.compose_model_image``
proprio / action normalisation               FastWAM processor + ``policy_io``
gripper sign and binarisation                FastWAM ``policy_io.postprocess_action``
denoising, sparse/accel arm                  FastWAM ``fastwam.casestudy``
which GPU, timeouts, episode loop, success   action-eval
===========================================  ==========================

The benchmark already applies the ``[::-1, ::-1]`` flip that FastWAM's
``get_libero_image`` applies on the native path, so this adapter must not flip again.

Injected by the runner: ``model_root`` (this checkout), ``model_config`` (the Hydra
entry config, ``configs/sim_libero.yaml``) and ``checkpoint`` (the Joint ``model.pt``).

Options (``policy.options``):
``dataset_stats_path``   required. Explicit, because the native resolver silently falls
                         back to parent directories of the checkpoint.
``diffsynth_model_base_path`` required. Root holding the Wan2.2 VAE, umt5 encoder and
                         tokenizer in DiffSynth layout; downloads are disabled.
``task``                 Hydra task override, default ``libero_joint_2cam224_1e-4``.
``hydra_overrides``      extra Hydra overrides (list of strings).
``casestudy``            the arm, see ``fastwam/casestudy/config.py``.
``casestudy_file``       alternatively, a YAML file ``{name, casestudy}`` holding the arm.
``shadow_casestudy``     optional second arm (normally dense) evaluated on the same observation at
                         every call, with the per-call seed of its own ``seed_offset``; its actions
                         are logged, never executed.
``shadow_executed``      actions per chunk counted as executed in shadow metrics (default 10).
``probe_arms``           optional list of {name, casestudy}: open-loop probes. At every call (or only the
                         calls in ``probe_calls``) each probe arm runs on the same observation once per
                         offset in ``probe_seed_offsets`` (default [0]; offset 0 is the executed arm's
                         noise), sharing the episode's layer choice with the executed arm. Chunks and the
                         arm's attention / motion diagnostics go to ``diagnostics.probes["<name>@<offset>"]``;
                         probe actions are never executed.
``seed_offset_per_repeat`` add the episode's repeat index to every seed offset (executed, shadow, probes),
                         so repeats of one episode are independent noise draws; repeat 0 keeps the noise
                         of a single-repeat run.
``episode_plans``        optional JSON {episode_id: {"calls": {"<k>": casestudy}, "from_call":
                         {"call": k, "casestudy": {...}}}}: per-call arm overrides used by the
                         attribution (patch at one call) and switch-point (arm from call k on)
                         experiments; every other call runs the base arm. An override draws its
                         noise with its own ``seed_offset`` (0 = the base rollout's noise).
                         An episode plan may list ``shadow_calls`` (call indices): the shadow arm then
                         runs only on those calls (patch experiments that need one deviation).
``prompt_embeddings``    optional ``.pt`` from scripts/casestudy/precompute_prompts.py; when set the
                         text encoder is not loaded and unknown instructions are an error.
``hash_checkpoint``      default true.
``checkpoint_sha256``    optional expected digest; a mismatch is fatal.

``hybrid``               optional {mode, partner, calls, eps_d, eps_theta, executed, limit, partner_seed_offset}: hybrid
                         execution of the component study (``fastwam/casestudy/hybrid.py``). On the selected calls the
                         ``partner`` arm (normally S0) also runs on the same observation and per-call seed, with the executed arm's
                         layer choice, and the executed chunk becomes ``hybrid.mix(dense, partner, mode)`` in simulator
                         units (gripper binarized). ``calls``: ``all``, ``{first: m}`` (calls 0..m-1), ``{every: n}``
                         (calls 0, n, 2n, ...) or ``plan`` (the episode plan's ``hybrid_calls``). Other calls run the
                         executed arm alone. ``diagnostics.hybrid`` logs the selection, the norms, skips, clipping and
                         the partner's chunks. ``partner_seed_offset`` (along-cross placebo) gives the partner its own
                         per-call seed offset instead of the executed arm's seed.

``shrink``               optional {frames_dir, metrics}: shrinkage-mechanism. ``<frames_dir>/<episode>.npz`` (future-inject's
                         re-rendered dense rollout) gives probes with ``draft_mod: real`` their drafts (the frames t + round(4 i
                         delta), delta = draft_mod_scale, encoded as a clip; past the rollout's end the probe is skipped), and
                         ``metrics`` adds ``diagnostics.shrink``: every probe's drafts decoded against the dense probe's (pixel
                         projection, RAFT flow ratio / cosine, gripper-disk flow) and the dense flow against the real future.
``inject_future``        optional {frames_dir, deltas, seeds, sparse, decode_dir}: future-injection probes. The executed
                         arm replays a dense rollout whose per-step observations were re-rendered into
                         ``<frames_dir>/<episode>.npz`` (``agentview`` / ``wrist`` [steps + 1, H, W, 3], frame k = after k
                         executed steps; ``deltas``; ``call_ok`` [calls]; ``valid`` [calls, deltas]). On every ``call_ok``
                         call and seed offset: dense (normal), ``sparse`` (normal), F_dense (``infer_inject`` with dense's
                         final future latents) and, for each valid delta, F_real (the clip of the current frame and frames
                         t + round(4 i delta), i = 1..8, through the VAE, injected). ``decode_dir`` saves dense's decoded
                         future at seed offset 0. Logged in ``diagnostics.inject``.
``region_attrib``        optional {frames_dir, plan_dir, seeds}: region-attrib. The executed dense arm replays the rollout
                         re-rendered in ``<frames_dir>/<episode>.npz`` (future-inject). ``<plan_dir>/<episode>.npz`` lists the
                         ``calls``, the condition ``names`` / ``sources`` (static / half / fade), token ``masks`` [calls, conds,
                         2, 98] and ``ok`` [calls, conds]. On a listed call and seed offset: ``base`` (F_real(1), with the action
                         rows' attention on the future keys at every layer), ``Fr0`` (the static future, the current frame x 9)
                         and every ok condition: the masked future tokens of F_real(1) replaced by the static future, the
                         delta = 0.5 future (half) or static + 0.5 (real - static) (fade). Also each future token's latent change
                         against the frame-0 latent and against the static future. Logged in ``diagnostics.region_attrib``.

``step0_tube``           optional {file, suite} for arms with ``prune.step0: d`` (step0-sparse plan D): a JSON with ``M``
                         (3 x 3, summed simulator action xyz -> end-effector displacement in m) and ``cameras``
                         ({"<suite>/<task id>": the agentview world -> pixel matrix, 256 x 256}). The runner gets the
                         pilot's end-effector cells per future latent frame from this observation's ``state[0:3]``.

An arm with ``casestudy.agc`` (pilot + automatic gain, ``fastwam/casestudy/agc.py``) has its de-normalized
chunk rescaled here, for the executed arm, the shadow and the probes alike: ``cont_actions_f32`` and the
executed actions are the rescaled chunk, ``raw_actions_f32`` stays the model's normalized output, and
``pre_agc_cont_actions_f32`` / ``agc`` hold the chunk before the gain and the gains. Arms that log the pilot
(``agc``, ``record_pilot``) add ``pilot_cont_actions_f32``, the pilot in the same units. The shadow's
normalized deviation metrics (``rms_*``) compare the model outputs, before any gain.

The checkpoint is third-party (BadWAM, LIQIIIII/badwam-libero-joint-wam@086a861), not an
official FastWAM release. It is loaded with strict key and shape checks instead of the
native ``load_checkpoint`` (``strict=False``), so a partial load cannot pass silently.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from action_eval_sdk import (
    ACTION_PROFILE_LIBERO_OSC_POSE_V1,
    OBSERVATION_PROFILE_LIBERO_2CAM_V1,
    PolicyDescription,
    Prediction,
)

try:  # added together with this adapter; older SDK checkouts lack it
    from action_eval_sdk import RNG_MODE_EPISODE_CALL_HASH
except ImportError:  # pragma: no cover
    RNG_MODE_EPISODE_CALL_HASH = "episode_call_hash"
from action_eval_sdk import RNG_MODE_FIXED_PER_PREDICT

_DIGESTS: dict[tuple[str, int, int], str] = {}


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(chunk), b""):
                digest.update(block)
        _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def _require_file(value, name: str) -> Path:
    if not value:
        raise ValueError(f"FastWAM adapter requires option `{name}`")
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _verified_load(model, checkpoint: Path) -> dict:
    """Load a Joint checkpoint with strict key/shape checks; return a summary."""
    import torch

    payload = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(payload, dict) or "mot" not in payload:
        raise ValueError(f"checkpoint has no `mot` state dict: {checkpoint}")
    expected = model.mot.state_dict()
    got = payload["mot"]
    missing = sorted(set(expected) - set(got))
    unexpected = sorted(set(got) - set(expected))
    bad_shape = sorted(k for k in set(expected) & set(got) if tuple(expected[k].shape) != tuple(got[k].shape))
    if missing or unexpected or bad_shape:
        raise ValueError(
            "Joint checkpoint does not match FastWAMJoint.mot: "
            f"missing={missing[:8]}({len(missing)}) unexpected={unexpected[:8]}({len(unexpected)}) "
            f"shape={bad_shape[:8]}({len(bad_shape)})"
        )
    model.mot.load_state_dict(got, strict=True)
    if model.proprio_encoder is None:
        raise ValueError("model has no proprio encoder but the LIBERO Joint task expects proprio_dim=8")
    if "proprio_encoder" not in payload:
        raise ValueError("checkpoint lacks `proprio_encoder` weights")
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    return {
        "mot_tensors": len(got),
        "mot_parameters": int(sum(v.numel() for v in got.values())),
        "proprio_tensors": len(payload["proprio_encoder"]),
        "payload_keys": sorted(k for k in payload if k != "optimizer"),
        "step": payload.get("step"),
    }


class FastWAMJointPolicy:
    def __init__(self, config: dict, device: str):
        options = dict(config)
        self.options = options
        self.model_root = Path(options.get("model_root") or os.getcwd()).expanduser()
        for path in (self.model_root, self.model_root / "src"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        model_config = _require_file(options.get("model_config"), "model_config")
        self.checkpoint = _require_file(options.get("checkpoint"), "checkpoint")
        self.stats_path = _require_file(options.get("dataset_stats_path"), "dataset_stats_path")
        base = options.get("diffsynth_model_base_path")
        if not base or not Path(str(base)).is_dir():
            raise FileNotFoundError(f"diffsynth_model_base_path is not a directory: {base}")
        # Base components come from a pre-staged directory; never download at eval time.
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(str(base)).expanduser())
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        from experiments.libero import policy_io
        from fastwam.casestudy import agc as agc_mod
        from fastwam.casestudy import hybrid as hybrid_mod
        from fastwam.casestudy import call_seed, make_runner, parse_config
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from fastwam.utils.config_resolvers import register_default_resolvers

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch sees no GPU; refusing to fall back to CPU")
        self._torch = torch
        self._io = policy_io
        self._agc_mod = agc_mod
        self._hybrid_mod = hybrid_mod
        self._call_seed = call_seed
        self._prompt_template = DEFAULT_PROMPT
        self.device = device

        register_default_resolvers()
        task = str(options.get("task") or "libero_joint_2cam224_1e-4")
        overrides = [f"task={task}", f"ckpt={self.checkpoint}"] + [str(o) for o in options.get("hydra_overrides") or []]
        self.prompt_store = None
        self.prompt_store_sha256 = None
        if options.get("prompt_embeddings"):
            store_path = _require_file(options["prompt_embeddings"], "prompt_embeddings")
            self.prompt_store = torch.load(str(store_path), map_location="cpu")["prompts"]
            self.prompt_store_sha256 = _sha256(store_path)
            overrides.append("model.load_text_encoder=false")
        with initialize_config_dir(config_dir=str(model_config.parent.resolve()), version_base="1.3"):
            cfg = compose(config_name=model_config.stem, overrides=overrides)
        self.cfg = cfg
        allowed_targets = {
            "fastwam.runtime.create_fastwam_joint",
            "fastwam.runtime.create_fastwam_optional_idm",
            "fastwam.runtime.create_fastwam_idm",
        }
        if cfg.model.get("_target_") not in allowed_targets:
            raise ValueError(f"task {task!r} builds {cfg.model.get('_target_')}, not a FastWAM Joint/IDM model")

        raw_arm = options.get("casestudy")
        self.arm_name = None
        self.arm_file_sha256 = None
        if options.get("casestudy_file"):
            if raw_arm is not None:
                raise ValueError("give either `casestudy` or `casestudy_file`, not both")
            import yaml

            arm_file = _require_file(options["casestudy_file"], "casestudy_file")
            arm = yaml.safe_load(arm_file.read_text()) or {}
            unknown = sorted(set(arm) - {"name", "casestudy", "notes"})
            if unknown:
                raise ValueError(f"unknown keys in {arm_file}: {unknown}")
            self.arm_name = arm.get("name")
            self.arm_file_sha256 = _sha256(arm_file)
            raw_arm = arm.get("casestudy")
        self.casestudy = parse_config(raw_arm)
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
        self.load_summary = _verified_load(model, self.checkpoint)
        self.model = model.to(device).eval()

        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(self.stats_path)))
        self.processor = processor

        expected_sha = options.get("checkpoint_sha256")
        self._hash_checkpoint = bool(options.get("hash_checkpoint", True)) or bool(expected_sha)
        self.checkpoint_sha256 = _sha256(self.checkpoint) if self._hash_checkpoint else None
        if expected_sha and self.checkpoint_sha256 != str(expected_sha):
            raise ValueError(f"checkpoint sha256 {self.checkpoint_sha256} != expected {expected_sha}")

        evaluation = cfg.EVALUATION
        self.action_horizon = int(evaluation.get("action_horizon") or (int(cfg.data.train.num_frames) - 1))
        self.num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1
        video_size = list(cfg.data.train.get("video_size", [224, 224]))
        self.input_h, self.input_w = int(video_size[0]), int(video_size[1])
        self.concatenation = str(cfg.data.train.get("concat_multi_camera", "horizontal"))
        self.binarize_gripper = bool(evaluation.get("binarize_gripper", False))
        sigma = evaluation.get("sigma_shift")
        self.sigma_shift = None if sigma is None else float(sigma)
        self.rand_device = str(evaluation.get("rand_device", "cpu"))
        self.native_seed = int(cfg.seed)
        self.runner = make_runner(self.model, self.casestudy, prompt_store=self.prompt_store)
        self.shadow = None
        if options.get("shadow_casestudy") is not None:
            self.shadow = make_runner(self.model, parse_config(options["shadow_casestudy"]), prompt_store=self.prompt_store)
        self.shadow_executed = int(options.get("shadow_executed", 10))
        self.probes: list[tuple[str, object]] = []
        for item in options.get("probe_arms") or []:
            name = str(item["name"])
            if "@" in name or name in {n for n, _ in self.probes}:
                raise ValueError(f"probe arm names must be unique and free of '@': {name!r}")
            self.probes.append((name, make_runner(self.model, parse_config(item["casestudy"]), prompt_store=self.prompt_store)))
        self.probe_seed_offsets = [int(x) for x in options.get("probe_seed_offsets") or [0]]
        self.probe_calls = None if options.get("probe_calls") is None else {int(x) for x in options["probe_calls"]}
        self.seed_offset_per_repeat = bool(options.get("seed_offset_per_repeat", False))
        if (self.probes or self.seed_offset_per_repeat) and self.casestudy.seed_mode != "episode_call_hash":
            raise ValueError("probe_arms and seed_offset_per_repeat need seed_mode=episode_call_hash")
        self.hybrid = None
        self.hybrid_runner = None
        if options.get("hybrid") is not None:
            h = dict(options["hybrid"])
            unknown = sorted(set(h) - {"mode", "partner", "calls", "eps_d", "eps_theta", "executed", "limit",
                                       "partner_seed_offset"})
            if unknown:
                raise ValueError(f"unknown keys in hybrid: {unknown}")
            calls = h.get("calls", "all")
            ok_calls = calls in ("all", "plan") or (isinstance(calls, dict) and len(calls) == 1 and (
                int(calls.get("first", 0)) >= 1 or int(calls.get("every", 0)) >= 1))
            if h.get("mode") not in hybrid_mod.MODES or not ok_calls or h.get("partner") is None:
                raise ValueError(f"hybrid needs mode in {hybrid_mod.MODES}, a partner arm and calls all / plan / "
                                 f"{{first: m}} / {{every: n}}, got {h}")
            self.hybrid = {"mode": str(h["mode"]), "calls": calls, "eps_d": float(h.get("eps_d", 0.0)),
                           "eps_theta": float(h.get("eps_theta", 0.0)), "executed": int(h.get("executed", 10)),
                           "limit": float(h.get("limit", 1.0))}
            if h.get("partner_seed_offset") is not None:  # along-cross placebo: the partner draws its own noise
                self.hybrid["partner_seed_offset"] = int(h["partner_seed_offset"])
            self.hybrid_runner = make_runner(self.model, parse_config(h["partner"]), prompt_store=self.prompt_store)
            if self.casestudy.seed_mode != "episode_call_hash":
                raise ValueError("hybrid needs seed_mode=episode_call_hash")
        self.shrink = None
        self._sm_frames = None
        self._raft = None
        self._sm_cache: dict = {}
        if options.get("shrink") is not None:
            f = dict(options["shrink"])
            unknown = sorted(set(f) - {"frames_dir", "metrics"})
            if unknown or not f.get("frames_dir"):
                raise ValueError(f"shrink needs frames_dir (unknown keys: {unknown})")
            self.shrink = {"frames_dir": Path(str(f["frames_dir"])), "metrics": bool(f.get("metrics", True))}
        self.inject = None
        self.inject_sparse = None
        self._inject_frames = None
        if options.get("inject_future") is not None:
            f = dict(options["inject_future"])
            unknown = sorted(set(f) - {"frames_dir", "deltas", "seeds", "sparse", "decode_dir"})
            if unknown or not f.get("frames_dir") or f.get("sparse") is None:
                raise ValueError(f"inject_future needs frames_dir and sparse (unknown keys: {unknown})")
            self.inject = {"frames_dir": Path(str(f["frames_dir"])), "deltas": [float(x) for x in f.get("deltas") or ()],
                           "seeds": [int(x) for x in f.get("seeds") or [0]],
                           "decode_dir": None if not f.get("decode_dir") else Path(str(f["decode_dir"]))}
            self.inject_sparse = make_runner(self.model, parse_config(f["sparse"]), prompt_store=self.prompt_store)
            if self.casestudy.seed_mode != "episode_call_hash" or self.casestudy.method != "dense":
                raise ValueError("inject_future needs a dense executed arm with seed_mode=episode_call_hash")
        self.region = None
        self._ra_frames = None
        self._ra_plan = None
        if options.get("region_attrib") is not None:
            f = dict(options["region_attrib"])
            unknown = sorted(set(f) - {"frames_dir", "plan_dir", "seeds"})
            if unknown or not f.get("frames_dir") or not f.get("plan_dir"):
                raise ValueError(f"region_attrib needs frames_dir and plan_dir (unknown keys: {unknown})")
            self.region = {"frames_dir": Path(str(f["frames_dir"])), "plan_dir": Path(str(f["plan_dir"])),
                           "seeds": [int(x) for x in f.get("seeds") or [0]]}
            if self.casestudy.seed_mode != "episode_call_hash" or self.casestudy.method != "dense":
                raise ValueError("region_attrib needs a dense executed arm with seed_mode=episode_call_hash")
        self._repeat_offset = 0
        self.episode_plans = {}
        self.episode_plans_sha256 = None
        if options.get("episode_plans"):
            import json as _json

            plans_path = _require_file(options["episode_plans"], "episode_plans")
            self.episode_plans = _json.loads(plans_path.read_text())
            self.episode_plans_sha256 = _sha256(plans_path)
        self._override_runners: dict[str, object] = {}
        self._plan: dict = {}
        self.step0_tube = None
        self._tube_camera = None
        self._state = None
        if options.get("step0_tube") is not None:
            import json as _json

            t = dict(options["step0_tube"])
            unknown = sorted(set(t) - {"file", "suite"})
            if unknown or not t.get("file") or not t.get("suite"):
                raise ValueError(f"step0_tube needs file and suite (unknown keys: {unknown})")
            tube_path = _require_file(t["file"], "step0_tube.file")
            spec = _json.loads(tube_path.read_text())
            self.step0_tube = {"suite": str(t["suite"]), "M": np.asarray(spec["M"], dtype=np.float64),
                               "cameras": {k: np.asarray(v, dtype=np.float64) for k, v in spec["cameras"].items()},
                               "sha256": _sha256(tube_path)}

        self.rng_mode = (
            RNG_MODE_EPISODE_CALL_HASH if self.casestudy.seed_mode == "episode_call_hash" else RNG_MODE_FIXED_PER_PREDICT
        )
        self._episode_key = ""
        self.predict_calls = 0

    # ------------------------------------------------------------------ contract
    def describe(self) -> PolicyDescription:
        fingerprint = {
            "adapter": "fastwam-casestudy",
            "model_class": type(self.model).__name__,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_identity": self.options.get("checkpoint_identity"),
            "checkpoint_load": self.load_summary,
            "dataset_stats": str(self.stats_path),
            "dataset_stats_sha256": _sha256(self.stats_path),
            "hydra_task": str(self.options.get("task") or "libero_joint_2cam224_1e-4"),
            "action_horizon": self.action_horizon,
            "num_video_frames": self.num_video_frames,
            "input_hw": [self.input_h, self.input_w],
            "binarize_gripper": self.binarize_gripper,
            "sigma_shift": self.sigma_shift,
            "rng_mode": self.rng_mode,
            "arm": self.arm_name,
            "arm_file_sha256": self.arm_file_sha256,
            "casestudy": self.casestudy.describe(),
            "casestudy_hash": self.casestudy.config_hash(),
            "shadow": None if self.shadow is None else self.shadow.cfg.describe(),
            "prompt_embeddings_sha256": self.prompt_store_sha256,
            "episode_plans_sha256": self.episode_plans_sha256,
            "episode_plans_count": len(self.episode_plans),
            "probes": [{"name": n, "casestudy": r.cfg.describe()} for n, r in self.probes] or None,
            "probe_seed_offsets": self.probe_seed_offsets if self.probes else None,
            "probe_calls": None if self.probe_calls is None else sorted(self.probe_calls),
            "seed_offset_per_repeat": self.seed_offset_per_repeat,
            "hybrid": None if self.hybrid is None else {**self.hybrid, "partner": self.hybrid_runner.cfg.describe()},
            "inject_future": None if self.inject is None else {
                **{k: str(v) if isinstance(v, Path) else v for k, v in self.inject.items()},
                "sparse": self.inject_sparse.cfg.describe()},
            "region_attrib": None if self.region is None else {
                k: str(v) if isinstance(v, Path) else v for k, v in self.region.items()},
        }
        notes = (
            "FastWAM-Joint through its native Hydra build and shared policy_io transforms; the benchmark "
            "flips images, this adapter does not. Frame-0 K/V and prompt encodings are reused in every arm."
        )
        if self.rng_mode == RNG_MODE_EPISODE_CALL_HASH:
            notes += " Diffusion noise is seeded per call from (instruction, episode id, call index)."
        return PolicyDescription(
            observation_profile=OBSERVATION_PROFILE_LIBERO_2CAM_V1,
            action_profile=ACTION_PROFILE_LIBERO_OSC_POSE_V1,
            rng_mode=self.rng_mode,
            action_horizon=self.action_horizon,
            fingerprint=fingerprint,
            notes=notes,
        )

    def reset(self, episode: dict) -> None:
        """Clears the per-episode prompt cache, CF layer choice and call counter."""
        self.runner.reset_episode()
        if self.shadow is not None:
            self.shadow.reset_episode()
        for runner in self._override_runners.values():
            runner.reset_episode()
        for _name, runner in self.probes:
            runner.reset_episode()
        if self.hybrid_runner is not None:
            self.hybrid_runner.reset_episode()
        self.predict_calls = 0
        self._repeat_offset = int(episode.get("repeat") or 0) if self.seed_offset_per_repeat else 0
        # Seed key = (instruction, task, init, protocol seed); the repeat index is left out so
        # that replays of one episode (patch / switch-point experiments use repeats to index
        # variants) draw exactly the noise of the original rollout at every call index.
        import re as _re

        episode_id = str(episode.get("episode_id", ""))
        self._episode_id = episode_id
        self._episode_key = f"{episode.get('instruction', '')}|{_re.sub(r'-r[0-9]+', '', episode_id)}"
        self._plan = self.episode_plans.get(episode_id, {})
        if self.step0_tube is not None:
            task_id = int(_re.match(r"t(\d+)", episode_id).group(1))
            key = f"{self.step0_tube['suite']}/{task_id}"
            if key not in self.step0_tube["cameras"]:
                raise ValueError(f"step0_tube: no agentview camera for {key}")
            self._tube_camera = self.step0_tube["cameras"][key]
        if self.hybrid is not None and self.hybrid["calls"] == "plan" and "hybrid_calls" not in self._plan:
            raise ValueError(f"hybrid calls: plan, but the episode plan of {episode_id} has no hybrid_calls")
        if self.shrink is not None:
            path = self.shrink["frames_dir"] / f"{_re.sub(r'-r[0-9]+.*$', '', episode_id)}.npz"
            self._sm_frames = None
            if path.is_file():
                with np.load(path) as z:
                    self._sm_frames = {k: z[k] for k in ("agentview", "wrist", "eef", "cam_agentview")}
            self._sm_cache = {}
        if self.inject is not None:
            self.inject_sparse.reset_episode()
            path = self.inject["frames_dir"] / f"{_re.sub(r'-r[0-9]+.*$', '', episode_id)}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"inject_future: no rendered frames {path}")
            with np.load(path) as z:
                self._inject_frames = {k: z[k] for k in ("agentview", "wrist", "deltas", "call_ok", "valid")}
            if [float(x) for x in self._inject_frames["deltas"]] != self.inject["deltas"]:
                raise ValueError(f"inject_future: {path} has deltas {self._inject_frames['deltas']}, not {self.inject['deltas']}")
        if self.region is not None:
            ep = _re.sub(r'-r[0-9]+.*$', '', episode_id)
            fpath, ppath = self.region["frames_dir"] / f"{ep}.npz", self.region["plan_dir"] / f"{ep}.npz"
            if not fpath.is_file() or not ppath.is_file():
                raise FileNotFoundError(f"region_attrib: no rendered frames {fpath} or plan {ppath}")
            with np.load(fpath) as z:
                self._ra_frames = {k: z[k] for k in ("agentview", "wrist")}
            with np.load(ppath) as z:
                self._ra_plan = {"calls": [int(c) for c in z["calls"]], "names": [str(n) for n in z["names"]],
                                 "sources": [str(s) for s in z["sources"]], "masks": z["masks"].astype(bool),
                                 "ok": z["ok"].astype(bool)}
            if not set(self._ra_plan["sources"]) <= {"static", "half", "fade"}:
                raise ValueError(f"region_attrib: unknown sources in {ppath}: {sorted(set(self._ra_plan['sources']))}")

    def predict(self, observation: dict) -> Prediction:
        torch = self._torch
        images = observation.get("images") or {}
        state = np.asarray(observation.get("state"), dtype=np.float32)
        instruction = observation.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("observation is missing a task instruction")
        if set(images) != {"agentview", "wrist"}:
            raise ValueError(f"expected cameras ['agentview', 'wrist'], got {sorted(images)}")
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError(f"expected a finite 8-dim proprio vector, got {state.shape}")

        started = time.perf_counter()
        rgb = self._io.compose_model_image(
            images["agentview"], images["wrist"], processor=self.processor,
            concatenation=self.concatenation, width=self.input_w, height=self.input_h,
        )
        image = self._io.image_to_model_tensor(rgb, device=self.device, dtype=self.model.torch_dtype)
        proprio = self._io.normalize_proprio(state, self.processor)
        prompt = self._prompt_template.format(task=instruction)
        call_index = self.predict_calls
        runner, override = self._runner_for_call(call_index)
        if self.casestudy.seed_mode == "episode_call_hash":
            offset = runner.cfg.seed_offset if override is not None else self.casestudy.seed_offset
            seed = self._call_seed(self._episode_key, call_index, offset + self._repeat_offset)
        else:
            seed = self.native_seed

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_started = time.perf_counter()
        video_cap = {} if self.inject is not None else None
        self._state = state
        action, diag = runner.infer_action(
            prompt=prompt, input_image=image, proprio=proprio,
            action_horizon=self.action_horizon, num_video_frames=self.num_video_frames,
            seed=seed, sigma_shift=self.sigma_shift, rand_device=self.rand_device,
            **({"_capture_video": video_cap} if video_cap is not None else {}), **self._tube_kwargs(runner),
            **self._mem_kwargs(runner, "exec"),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_seconds = time.perf_counter() - model_started
        actions = self._io.postprocess_action(action.clone(), self.processor, binarize_gripper=self.binarize_gripper)
        actions = np.asarray(actions, dtype=np.float32)
        cont, actions, agc_log = self._agc(runner.cfg, diag, self._continuous(action), actions)
        hybrid_log = None
        if self.hybrid is not None:
            actions, hybrid_log = self._hybrid_call(runner, call_index, prompt, image, proprio, seed, actions)
        if not np.isfinite(actions).all():
            raise ValueError("FastWAM produced non-finite actions")
        self.predict_calls += 1
        diag.update({"call_index": call_index, "model_seconds": round(model_seconds, 6)})
        diag.update(self._chunk_logs(action, cont))
        diag.update(agc_log)
        if hybrid_log is not None:
            diag["hybrid"] = hybrid_log
        if diag.get("kept_index") is not None and diag.get("grid") is not None:
            per_frame = int(diag["grid"][1]) * int(diag["grid"][2])
            diag["kept_frame"] = [int(i) // per_frame + 1 for i in diag["kept_index"]]
        if override is not None:
            diag["plan_override"] = override
        shadow_calls = self._plan.get("shadow_calls")
        if self.shadow is not None and (shadow_calls is None or call_index in shadow_calls):
            # The shadow draws its own noise: same per-call seed function, its own seed_offset
            # (a dense-reseed shadow measures the policy's sampling spread; a plain dense shadow
            # on a dense arm reproduces the executed action exactly).
            shadow_offset = getattr(self.shadow.cfg, "seed_offset", self.casestudy.seed_offset)
            shadow_seed = (self._call_seed(self._episode_key, call_index, shadow_offset + self._repeat_offset)
                           if self.casestudy.seed_mode == "episode_call_hash" else seed)
            diag["shadow"] = self._shadow_metrics(prompt, image, proprio, shadow_seed, action, actions)
        if self.probes and (self.probe_calls is None or call_index in self.probe_calls):
            diag["probes"] = self._probe_metrics(prompt, image, proprio, call_index)
            if self.shrink is not None and self.shrink["metrics"]:
                diag["shrink"] = self._shrink_metrics(diag["probes"], image, images, call_index)
        if self.inject is not None:
            diag["inject"] = self._inject_call(runner, call_index, prompt, image, proprio, images, action, video_cap)
        if self.region is not None:
            diag["region_attrib"] = self._region_call(runner, call_index, prompt, image, proprio, images)
        return Prediction(
            actions=actions,
            timing={"predict_seconds": round(time.perf_counter() - started, 6), "model_seconds": round(model_seconds, 6)},
            diagnostics=diag,
        )

    def _inject_call(self, runner, call_index: int, prompt, image, proprio, images: dict, dense_action, video_cap) -> dict:
        """Future-injection probes of one call (option ``inject_future``): dense, sparse, F_dense and F_real(delta) at every
        seed offset, on the observation of the replayed dense rollout."""
        torch = self._torch
        fr, inj = self._inject_frames, self.inject
        if call_index >= len(fr["call_ok"]) or not bool(fr["call_ok"][call_index]):
            return {"used": False}
        t = 10 * call_index  # frame index of this call's observation: after 10 c executed steps
        out: dict = {"used": True, "valid": [bool(x) for x in fr["valid"][call_index]],
                     "current_frame_equal": bool(np.array_equal(fr["agentview"][t], images["agentview"])
                                                 and np.array_equal(fr["wrist"][t], images["wrist"]))}

        def model_frame(j: int):
            rgb = self._io.compose_model_image(fr["agentview"][j], fr["wrist"][j], processor=self.processor,
                                               concatenation=self.concatenation, width=self.input_w, height=self.input_h)
            return self._io.image_to_model_tensor(rgb, device=self.device, dtype=self.model.torch_dtype)

        first = self.model._encode_input_image_latents_tensor(input_image=image, tiled=False)
        latents: dict[float, object] = {}
        for k, delta in enumerate(inj["deltas"]):
            if out["valid"][k]:
                clip = torch.cat([image] + [model_frame(t + int(round(4 * i * delta))) for i in range(1, 9)], dim=0)
                latents[delta] = runner.encode_clip(clip)
        if latents:
            out["frame0_latent_max_abs"] = max(float((z[:, :, 0:1].float() - first.float()).abs().max()) for z in latents.values())
            out["frame0_latent_scale"] = float(first.float().abs().max())
        layers = getattr(runner, "_layers", None)
        entries = {}
        common = dict(prompt=prompt, input_image=image, proprio=proprio, action_horizon=self.action_horizon,
                      num_video_frames=self.num_video_frames, sigma_shift=self.sigma_shift, rand_device=self.rand_device)
        for j in inj["seeds"]:
            seed = self._call_seed(self._episode_key, call_index, j + self._repeat_offset)
            if j == 0 and video_cap:
                a_d, vid = dense_action, video_cap["latents"]
            else:
                cap: dict = {}
                a_d, _ = runner.infer_action(seed=seed, _capture_video=cap, **common)
                vid = cap["latents"]
            entries[f"dense@{j}"] = self._chunk_logs(a_d)
            if layers is not None and int(layers.get("act", -1)) >= 0:
                self.inject_sparse._layers = dict(layers)
            a_s, d_s = self.inject_sparse.infer_action(seed=seed, **common)
            entries[f"S0@{j}"] = {**self._chunk_logs(a_s), "kept_count": d_s.get("kept_count")}
            a_f, d_f = runner.infer_inject(seed=seed, future_latents=vid[:, :, 1:], **common)
            entries[f"Fd@{j}"] = {**self._chunk_logs(a_f), "cost": d_f["cost"]}
            for delta, z in latents.items():
                a_r, _ = runner.infer_inject(seed=seed, future_latents=z[:, :, 1:], **common)
                entries[f"Fr{delta:g}@{j}"] = self._chunk_logs(a_r)
            if j == 0 and inj["decode_dir"] is not None:
                inj["decode_dir"].mkdir(parents=True, exist_ok=True)
                np.savez_compressed(inj["decode_dir"] / f"{self._episode_id}-c{call_index:02d}.npz", dense_future=runner.decode_video(vid))
        out["entries"] = entries
        return out

    def _region_call(self, runner, call_index: int, prompt, image, proprio, images: dict) -> dict:
        """region-attrib (option ``region_attrib``): on a planned call and every seed offset, F_real(1) unmodified (``base``,
        with the attention capture), the static future (``Fr0``) and every ok condition of the plan, all through
        ``infer_inject``. A condition replaces its masked future tokens (both latent frames as the mask says) of F_real(1)
        by the static future (the current frame x 9, delta = 0), the delta = 0.5 future (``half``) or static + 0.5 (real -
        static) (``fade``)."""
        import base64

        from fastwam.casestudy import fovea as fv

        torch = self._torch
        plan, fr = self._ra_plan, self._ra_frames
        if call_index not in plan["calls"]:
            return {"used": False}
        ci = plan["calls"].index(call_index)
        t = 10 * call_index  # frame index of this call's observation: after 10 c executed steps
        out: dict = {"used": True, "current_frame_equal": bool(np.array_equal(fr["agentview"][t], images["agentview"])
                                                               and np.array_equal(fr["wrist"][t], images["wrist"]))}

        def model_frame(j: int):
            rgb = self._io.compose_model_image(fr["agentview"][j], fr["wrist"][j], processor=self.processor,
                                               concatenation=self.concatenation, width=self.input_w, height=self.input_h)
            return self._io.image_to_model_tensor(rgb, device=self.device, dtype=self.model.torch_dtype)

        def future(delta: float):
            clip = torch.cat([image] + [model_frame(t + int(round(4 * i * delta))) for i in range(1, 9)], dim=0)
            return runner.encode_clip(clip)[:, :, 1:]

        ok = plan["ok"][ci]
        z1, zs = future(1.0), future(0.0)
        src = {"static": zs,
               "half": future(0.5) if any(s == "half" and o for s, o in zip(plan["sources"], ok)) else None,
               "fade": (zs.float() + 0.5 * (z1.float() - zs.float())).to(z1.dtype)}
        nf, lh, lw = (int(x) for x in z1.shape[2:])
        ph, pw = int(runner.video.patch_size[1]), int(runner.video.patch_size[2])
        gh, gw = lh // ph, lw // pw
        if plan["masks"].shape[2:] != (nf, gh * gw):
            raise ValueError(f"region_attrib: plan masks {plan['masks'].shape[2:]} do not match the grid ({nf}, {gh * gw})")
        first = self.model._encode_input_image_latents_tensor(input_image=image, tiled=False)
        out["diff_z0"] = fv.token_norms(z1.float() - first.float(), gh, gw, ph, pw).cpu().tolist()
        out["diff_static"] = fv.token_norms(z1.float() - zs.float(), gh, gw, ph, pw).cpu().tolist()
        conds = [(name, torch.where(fv.token_latent_mask(plan["masks"][ci, k], gh, gw, ph, pw).to(z1.device),
                                    src[plan["sources"][k]], z1))
                 for k, name in enumerate(plan["names"]) if ok[k]]
        common = dict(prompt=prompt, input_image=image, proprio=proprio, action_horizon=self.action_horizon,
                      num_video_frames=self.num_video_frames, sigma_shift=self.sigma_shift, rand_device=self.rand_device)
        entries, attn = {}, {}
        for j in self.region["seeds"]:
            seed = self._call_seed(self._episode_key, call_index, j + self._repeat_offset)
            cap: dict = {}
            a_b, _ = runner.infer_inject(seed=seed, future_latents=z1, _attn_capture=cap, **common)
            entries[f"base@{j}"] = self._chunk_logs(a_b)
            av = cap["av"]
            attn[f"base@{j}"] = {"shape": list(av.shape[1:]),
                                 "av_mean_f32": base64.b64encode(av.mean(dim=0).numpy().astype(np.float32).tobytes()).decode(),
                                 "av_step0_f32": base64.b64encode(av[0].numpy().astype(np.float32).tobytes()).decode(),
                                 "f0_mean": cap["f0"].mean(dim=0).tolist()}
            a_0, _ = runner.infer_inject(seed=seed, future_latents=zs, **common)
            entries[f"Fr0@{j}"] = self._chunk_logs(a_0)
            for name, z in conds:
                a_c, _ = runner.infer_inject(seed=seed, future_latents=z, **common)
                entries[f"{name}@{j}"] = self._chunk_logs(a_c)
        out["entries"], out["attn"] = entries, attn
        return out

    def _hybrid_call(self, runner, call_index: int, prompt, image, proprio, seed: int, actions: np.ndarray):
        """Component study: on a selected call run the partner arm on the same input and seed (with the executed
        arm's layer choice) and return ``hybrid.mix`` of the two simulator chunks -> (actions, log)."""
        h = self.hybrid
        calls = h["calls"]
        if calls == "all":
            selected = True
        elif calls == "plan":
            selected = call_index in {int(x) for x in self._plan["hybrid_calls"]}
        elif "first" in calls:
            selected = call_index < int(calls["first"])
        else:
            selected = call_index % int(calls["every"]) == 0
        if not selected:
            return actions, {"selected": False}
        partner = self.hybrid_runner
        layers = getattr(runner, "_layers", None)
        if layers is not None and int(layers.get("act", -1)) >= 0:
            partner._layers = dict(layers)
        if h.get("partner_seed_offset") is not None:
            seed = self._call_seed(self._episode_key, call_index, h["partner_seed_offset"] + self._repeat_offset)
        a_s, d_s = partner.infer_action(
            prompt=prompt, input_image=image, proprio=proprio,
            action_horizon=self.action_horizon, num_video_frames=self.num_video_frames,
            seed=seed, sigma_shift=self.sigma_shift, rand_device=self.rand_device,
        )
        sim_s = np.asarray(
            self._io.postprocess_action(a_s.clone(), self.processor, binarize_gripper=self.binarize_gripper), dtype=np.float32
        )
        out, info = self._hybrid_mod.mix(actions, sim_s, h["mode"], h["eps_d"], h["eps_theta"], h["executed"], h["limit"])
        log = {"selected": True, **info, "partner_kept_index": d_s.get("kept_index"),
               **{f"partner_{k}": v for k, v in self._chunk_logs(a_s).items()}}
        return out, log

    def _runner_for_call(self, call_index: int):
        """Base runner, or the episode plan's override for this call."""
        plan = self._plan
        raw = None
        label = None
        calls = plan.get("calls") or {}
        if str(call_index) in calls:
            raw, label = calls[str(call_index)], f"call:{call_index}"
        elif plan.get("from_call") and call_index >= int(plan["from_call"]["call"]):
            raw, label = plan["from_call"]["casestudy"], f"from:{plan['from_call']['call']}"
        if raw is None:
            return self.runner, None
        from fastwam.casestudy import make_runner, parse_config

        cfg = parse_config(raw)
        key = cfg.config_hash()
        if key not in self._override_runners:
            self._override_runners[key] = make_runner(self.model, cfg, prompt_store=self.prompt_store)
        runner = self._override_runners[key]
        # Share the episode's frozen CF layer choice so a patched call scores like the base arm.
        if getattr(self.runner, "_layers", None) is not None:
            runner._layers = dict(self.runner._layers)
        return runner, {"label": label, "casestudy_hash": key}

    def _shadow_metrics(self, prompt, image, proprio, seed, action, actions) -> dict:
        """Shadow arm on the same input and seed; returns deviation metrics and its chunk."""
        import base64

        shadow_action, shadow_diag = self.shadow.infer_action(
            prompt=prompt, input_image=image, proprio=proprio,
            action_horizon=self.action_horizon, num_video_frames=self.num_video_frames,
            seed=seed, sigma_shift=self.sigma_shift, rand_device=self.rand_device,
        )
        shadow_sim = np.asarray(
            self._io.postprocess_action(shadow_action.clone(), self.processor, binarize_gripper=self.binarize_gripper),
            dtype=np.float32,
        )
        shadow_cont, shadow_sim, shadow_agc = self._agc(self.shadow.cfg, shadow_diag, self._continuous(shadow_action), shadow_sim)
        a = action.numpy()
        b = shadow_action.numpy()
        k = self.shadow_executed
        d = a - b
        return {
            # normalized action space ([-1, 1] per dim under min/max normalisation)
            "rms_executed": float(np.sqrt(np.mean(d[:k] ** 2))),
            "rms_full": float(np.sqrt(np.mean(d ** 2))),
            "max_abs_executed": float(np.abs(d[:k]).max()),
            # simulator units
            "gripper_disagree_executed": int(np.sum(np.sign(actions[:k, -1]) != np.sign(shadow_sim[:k, -1]))),
            "translation_diff_executed": float(np.linalg.norm(actions[:k, :3].sum(0) - shadow_sim[:k, :3].sum(0))),
            "rotation_diff_executed": float(np.linalg.norm(actions[:k, 3:6].sum(0) - shadow_sim[:k, 3:6].sum(0))),
            "shadow_sim_actions_f16": base64.b64encode(shadow_sim.astype(np.float16).tobytes()).decode(),
            "shadow_gate_score": shadow_diag.get("gate_score"),
            "shadow_kept_index": shadow_diag.get("kept_index"),
            **{f"shadow_{key}": value for key, value in {**self._chunk_logs(shadow_action, shadow_cont), **shadow_agc}.items()},
        }

    def _probe_metrics(self, prompt, image, proprio, call_index: int) -> dict:
        """Every probe arm at every probe seed offset on this observation: chunks and diagnostics."""
        import base64

        layers = getattr(self.runner, "_layers", None)
        out = {}
        for offset in self.probe_seed_offsets:
            seed = self._call_seed(self._episode_key, call_index, offset + self._repeat_offset)
            for name, runner in self.probes:
                if layers is not None and int(layers.get("act", -1)) >= 0:
                    runner._layers = dict(layers)
                real = self._real_kwargs(runner, image, call_index)
                if real is None:  # shrinkage-mechanism: real future frames run past the rollout's end
                    out[f"{name}@{offset}"] = {"seed": int(seed), "skipped": "frames"}
                    continue
                action, d = runner.infer_action(
                    prompt=prompt, input_image=image, proprio=proprio,
                    action_horizon=self.action_horizon, num_video_frames=self.num_video_frames,
                    seed=seed, sigma_shift=self.sigma_shift, rand_device=self.rand_device, **self._tube_kwargs(runner),
                    **self._mem_kwargs(runner, offset), **real, **self._qm_kwargs(runner),
                )
                cont, _, agc_log = self._agc(runner.cfg, d, self._continuous(action))
                entry = {"seed": int(seed), **self._chunk_logs(action, cont), **agc_log}
                for key in ("kept_index", "steer", "future_share_f32", "future_share_step", "motion_f32", "layer_act",
                            "steer_video", "video_frame0_f32", "motion_steps_f32", "paths", "cost", "modes",
                            "summary", "cache_kv_bytes", "recon", "recon_quality_f32", "recon_quality_shape",
                            "recon_checks", "av_scores_f16", "diff_scores_f16", "diff_sigma_v", "future_attn_f32",
                            "future_attn_shape", "dump_path", "gate_score", "fovea", "sink_index", "sink_share_step0",
                            "sink_recv_f16", "step0", "step0_x0_f32", "e9", "r2", "r3", "mem", "mem_rec", "side0",
                            "terminal_x0_f32", "draft", "reselect", "explore", "explore_plan", "x0d_f16", "zcur_f16",
                            "draft_mod", "late", "drafts_f32", "drafts_step", "drafts_y0_f32", "drafts_y1_f32", "zcur_f32",
                            "kval_f16", "kval_m", "kval_shape", "qm", "late_read", "late_z9_f32"):
                    if d.get(key) is not None:
                        entry[key] = d[key]
                if entry.get("qm") is not None:
                    self._qm_units(entry["qm"])
                for row in list((entry.get("r3") or {}).values()) + list((entry.get("explore") or {}).values()):  # R3 / explore pilots, simulator units
                    p = np.frombuffer(base64.b64decode(row["pilot_f32"]), dtype=np.float32).reshape(-1, 7)
                    row["pilot_cont_f32"] = base64.b64encode(
                        self._continuous(self._torch.from_numpy(p.copy())).astype(np.float32).tobytes()).decode()
                out[f"{name}@{offset}"] = entry
        return out

    # ---- shrinkage-mechanism: real future frames as drafts, and the metrics of every probe's imagination
    def _sm_model_frame(self, j: int):
        fr = self._sm_frames
        rgb = self._io.compose_model_image(fr["agentview"][j], fr["wrist"][j], processor=self.processor,
                                           concatenation=self.concatenation, width=self.input_w, height=self.input_h)
        return self._io.image_to_model_tensor(rgb, device=self.device, dtype=self.model.torch_dtype)

    def _real_latents(self, image, call_index: int, delta: float):
        """VAE latents [1, z, 3, h, w] of the current frame + the dense rollout's frames t + round(4 i delta) (i = 1..8), encoded
        as training encodes clips; None past the rollout's end (cached per call and delta)."""
        key = ("real", call_index, float(delta))
        if key not in self._sm_cache:
            fr, t = self._sm_frames, 10 * call_index
            idx = [t + int(round(4 * i * delta)) for i in range(1, 9)]
            if fr is None or max(idx) >= len(fr["agentview"]):
                self._sm_cache[key] = None
            else:
                clip = self._torch.cat([image] + [self._sm_model_frame(j) for j in idx], dim=0)
                self._sm_cache[key] = self.runner.encode_clip(clip)
        return self._sm_cache[key]

    def _real_kwargs(self, runner, image, call_index: int) -> dict | None:
        """``_real_future`` for a probe with draft_mod real (delta = draft_mod_scale); {} for other runners; None when the real
        future runs past the rollout's end (the probe is skipped on this call)."""
        prune = runner.cfg.prune
        if prune is None or prune.draft_mod != "real":
            return {}
        if self.shrink is None:
            raise ValueError("draft_mod real needs the adapter option shrink (frames_dir)")
        lat = self._real_latents(image, call_index, float(prune.draft_mod_scale))
        return None if lat is None else {"_real_future": lat[:, :, 1:]}

    def _shrink_metrics(self, probes: dict, image, images: dict, call_index: int) -> dict:
        """shrinkage-mechanism metrics of every probe that logged drafts, against the dense probe of the same seed offset (its
        step-0 / step-1 clean estimate, by the probe's drafts_step): VAE-decoded pixel projection, RAFT flow (magnitude ratio
        and direction cosine on the 20 % most-moving dense pixels) per camera for the near (pixel frame 4) and far (8) frame,
        the gripper-disk flow (end effector projected into the third-person view), and the dense flow against the real
        future (delta 1) as the pipeline's reference."""
        import base64

        torch = self._torch
        from einops import rearrange

        if self._raft is None:
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
            self._raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(self.device).eval()
        rn = self.runner
        pt, ph, pw = (int(x) for x in rn.video.patch_size)
        first = self.model._encode_input_image_latents_tensor(input_image=image, tiled=False)
        _, cz, _, lh, lw = first.shape

        def f32(b64: str) -> np.ndarray:
            return np.frombuffer(base64.b64decode(b64), dtype=np.float32)

        def decode(zf: np.ndarray) -> np.ndarray:
            z = torch.from_numpy(zf.copy()).to(self.device).reshape(2, (lh // ph) * (lw // pw), -1)
            fut = rearrange(z, "f (h w) (c a b e) -> c (f a) (h b) (w e)", h=lh // ph, a=pt, b=ph, e=pw)[None]
            lat = torch.cat([first.float(), fut.float()], dim=2).to(first.dtype)
            return rn.decode_video(lat)

        def flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            t1 = torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2).float().to(self.device) / 127.5 - 1.0
            t2 = torch.from_numpy(np.stack(b)).permute(0, 3, 1, 2).float().to(self.device) / 127.5 - 1.0
            with torch.no_grad():
                return self._raft(t1, t2)[-1].float().cpu().numpy()  # [N, 2, H, W]

        fr, t = self._sm_frames, 10 * call_index
        grip_xy = None
        if fr is not None and t < len(fr["eef"]):
            ph_ = fr["cam_agentview"] @ np.append(fr["eef"][t], 1.0)
            h0 = fr["agentview"].shape[1]
            grip_xy = ((h0 - 1 - ph_[0] / ph_[2]) * 224 / h0, (ph_[1] / ph_[2]) * 224 / h0)
        out: dict = {"grip_xy": None if grip_xy is None else [round(float(v), 2) for v in grip_xy]}
        if fr is not None:
            out["frames_current_equal"] = bool(np.array_equal(fr["agentview"][t], images["agentview"])
                                               and np.array_equal(fr["wrist"][t], images["wrist"]))
        yy, xx = np.mgrid[0:224, 0:224]
        disk = None if grip_xy is None else ((xx - grip_xy[0]) ** 2 + (yy - grip_xy[1]) ** 2) <= 12 ** 2
        refs: dict = {}
        for key, e in probes.items():
            name, j = key.split("@")
            if name == "dense" and e.get("drafts_y0_f32"):
                for st in (0, 1):
                    if e.get(f"drafts_y{st}_f32"):
                        refs[(j, st)] = decode(f32(e[f"drafts_y{st}_f32"]))
        res: dict = {}
        cams = {"agent": slice(0, 224), "wrist": slice(224, 448)}
        for key, e in probes.items():
            name, j = key.split("@")
            st = int(e.get("drafts_step", 0))
            if not e.get("drafts_f32") or (j, st) not in refs:
                continue
            dy, dd = refs[(j, st)], decode(f32(e["drafts_f32"]))
            iz = refs[(j, 0)][0]
            pairs_a, pairs_b, labels = [], [], []
            for fi, fname in ((4, "near"), (8, "far")):
                for cname, cs in cams.items():
                    for src, lab in ((dy, "y"), (dd, "d")):
                        pairs_a.append(iz[:, cs])
                        pairs_b.append(src[fi][:, cs])
                        labels.append((fname, cname, lab))
            fl = flow(pairs_a, pairs_b)
            fmap = {lab: fl[i] for i, lab in enumerate(labels)}
            rec: dict = {}
            for fi, fname in ((4, "near"), (8, "far")):
                for cname, cs in cams.items():
                    fy, fd = fmap[(fname, cname, "y")], fmap[(fname, cname, "d")]
                    my = np.linalg.norm(fy, axis=0)
                    m = my >= np.quantile(my, 0.8)
                    md = np.linalg.norm(fd, axis=0)
                    cos = (fy * fd).sum(axis=0) / (my * md + 1e-6)
                    a, b, z_ = dd[fi][:, cs].astype(np.float64), dy[fi][:, cs].astype(np.float64), iz[:, cs].astype(np.float64)
                    r = {"flow_ratio": float(md[m].mean() / max(my[m].mean(), 1e-6)), "flow_cos": float(cos[m].mean()),
                         "pix_proj": float(((a - z_) * (b - z_)).sum() / max(((b - z_) ** 2).sum(), 1e-6)),
                         "flow_mag_dense": float(my[m].mean())}
                    if cname == "agent" and disk is not None:
                        gy, gd = fy[:, disk].mean(axis=1), fd[:, disk].mean(axis=1)
                        ny, nd = float(np.linalg.norm(gy)), float(np.linalg.norm(gd))
                        r.update(grip_ratio=nd / max(ny, 1e-6), grip_angle=float(np.degrees(np.arccos(np.clip(
                            float(gy @ gd) / max(ny * nd, 1e-9), -1.0, 1.0)))), grip_dense=ny)
                    rec[f"{fname}_{cname}"] = r
            res[key] = rec
        # the pipeline's reference: the dense imagination's flow against the real future (delta 1), seed offset 0
        real_idx = [t + 16, t + 32]
        if fr is not None and ("0", 0) in refs and max(real_idx) < len(fr["agentview"]):
            ref = {}
            iz = refs[("0", 0)][0]
            for fi, ri, fname in ((4, real_idx[0], "near"), (8, real_idx[1], "far")):
                rgb = self._io.compose_model_image(fr["agentview"][ri], fr["wrist"][ri], processor=self.processor,
                                                   concatenation=self.concatenation, width=self.input_w, height=self.input_h)
                rgb = np.asarray(rgb)
                for cname, cs in cams.items():
                    fl = flow([iz[:, cs], iz[:, cs]], [refs[("0", 0)][fi][:, cs], rgb[:, cs]])
                    mr = np.linalg.norm(fl[1], axis=0)
                    m = mr >= np.quantile(mr, 0.8)
                    ref[f"{fname}_{cname}"] = {"dense_over_real": float(np.linalg.norm(fl[0], axis=0)[m].mean() / max(mr[m].mean(), 1e-6))}
            out["real_ref"] = ref
        lat1 = self._sm_cache.get(("real", call_index, 1.0))
        if lat1 is not None:
            z1 = rearrange(lat1[0, :, 1:].float(), "c (f a) (h b) (w e) -> f (h w) (c a b e)", a=pt, b=ph, e=pw)
            out["real1_f32"] = base64.b64encode(z1.reshape(-1, z1.shape[-1]).cpu().numpy().astype(np.float32).tobytes()).decode()
        out["probes"] = res
        return out

    def _mem_kwargs(self, runner, key) -> dict:
        """``_memory_key`` for a runner with ``prune.mem`` (step0-sparse-r2): the probe seed offset, or the executed arm."""
        prune = runner.cfg.prune
        return {"_memory_key": key} if prune is not None and prune.mem else {}

    def _tube_kwargs(self, runner) -> dict:
        """``_step0_tube`` for a runner with ``prune.step0: d`` (step0-sparse plan D), else nothing."""
        prune = runner.cfg.prune
        if prune is None or prune.step0 != "d":
            return {}
        if self._tube_camera is None:
            raise ValueError("prune.step0 d needs policy.options.step0_tube")
        return {"_step0_tube": self._tube(np.asarray(self._state, dtype=np.float64))}

    def _tube(self, state: np.ndarray):
        """step0-sparse plan D (DECISIONS B, D.2-D.4): pilot (normalized [T, 7]) -> (agentview cells per future latent frame,
        gripper switch, info). p_t = p_0 + M sum_{i<t} a_i[0:3] in the world frame, projected by the episode's agentview
        matrix (256 x 256, upright row / column, principal point at 128 so in pixel-edge units) onto the observation (the raw
        render flipped up-down and left-right: x = 256 - column, y = row), cell = floor(pixel * 7 / 256); frame 1 from
        t = 4..16, frame 2 from t = 20..32."""
        cam, m_scale, p0, res, n = self._tube_camera, self.step0_tube["M"], state[0:3], 256, 7

        def tube(pilot):
            cont = self._continuous(pilot.clone()).astype(np.float64)
            path = p0[None] + np.cumsum(cont[:, 0:3], axis=0) @ m_scale.T  # row t - 1 = p_t
            cells, outside = {}, 0
            for f, (a, b) in ((1, (4, 16)), (2, (20, 32))):
                pts = set()
                for t in range(a, b + 1):
                    ph = cam @ np.append(path[t - 1], 1.0)
                    row, col = ph[1] / ph[2], ph[0] / ph[2]
                    x, y = res - col, row
                    if 0 <= x < res and 0 <= y < res:
                        pts.add((int(y * n // res), int(x * n // res)))
                    else:
                        outside += 1
                cells[f] = sorted(pts)
            g = np.sign(cont[:, 6])
            switch = bool(np.any(g[1:] != g[:-1]))
            info = {"cells": {str(f): [list(c) for c in v] for f, v in cells.items()}, "gripper_switch": switch,
                    "outside": outside, "path_m": float(np.linalg.norm(path[-1] - p0))}
            return cells, switch, info

        return tube

    def _qm_kwargs(self, runner) -> dict:
        """quick-metrics (``casestudy.qm``): ``_qm_zero``, the normalized action of zero simulator translation per xyz dimension
        (the de-normalization is affine per dimension: a0 = -cont(0) / (cont(1) - cont(0))), for c(s)'s scaling of the executed
        displacement; {} for other runners."""
        if getattr(runner.cfg, "qm", None) is None:
            return {}
        if getattr(self, "_qm_zero", None) is None:
            torch = self._torch
            dim = int(runner.action.action_dim)
            c0 = self._continuous(torch.zeros(self.action_horizon, dim))[0, :3].astype(np.float64)
            c1 = self._continuous(torch.ones(self.action_horizon, dim))[0, :3].astype(np.float64)
            self._qm_zero = torch.tensor(-c0 / (c1 - c0), dtype=torch.float32, device=self.device)
        return {"_qm_zero": self._qm_zero}

    def _qm_units(self, qm: dict) -> None:
        """quick-metrics: the runner's normalized action chunks (``qm["actions"]``) -> per name the executed displacement ``D`` and
        rotation ``R`` (first ``n_exec`` actions summed, simulator units), in place; ``a0`` logged."""
        import base64

        n = int(qm.get("n_exec", 10))
        D, R = {}, {}
        for name, b64 in (qm.pop("actions", None) or {}).items():
            a = np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(-1, 7)
            c = self._continuous(self._torch.from_numpy(a.copy())).astype(np.float64)[:n]
            D[name], R[name] = [float(x) for x in c[:, 0:3].sum(0)], [float(x) for x in c[:, 3:6].sum(0)]
        qm["D"], qm["R"] = D, R
        if getattr(self, "_qm_zero", None) is not None:
            qm["a0"] = [float(x) for x in self._qm_zero.cpu().tolist()]

    def _continuous(self, action) -> np.ndarray:
        """De-normalized chunk [T, 7] before gripper binarisation (float32)."""
        return np.asarray(self._io.postprocess_action(action.clone(), self.processor, binarize_gripper=False), dtype=np.float32)

    def _agc(self, cfg, diag: dict, cont: np.ndarray, sim: np.ndarray | None = None):
        """Pilot logging and the arm's automatic gain (``cfg.agc``) on one chunk -> (cont, sim, log).

        Pops the runner's normalized pilot from ``diag`` and logs it de-normalized. Without ``agc`` the
        chunks come back as given; with it both are rescaled by ``agc.apply`` with the gains of the
        continuous chunk (dims 0-5 of ``cont`` and ``sim`` are identical), 1 / 1 under ``force_unit``."""
        import base64

        def b64(x) -> str:
            return base64.b64encode(np.asarray(x, dtype=np.float32).tobytes()).decode()

        log: dict = {}
        pilot = None
        raw = diag.pop("pilot_f32", None)
        if raw is not None:
            p = np.frombuffer(base64.b64decode(raw), dtype=np.float32).reshape(cont.shape)
            pilot = self._continuous(self._torch.from_numpy(p.copy()))
            log.update(pilot_cont_actions_f32=b64(pilot), pilot_sigma0=diag.pop("pilot_sigma0", None))
        agc = cfg.agc
        if agc is None:
            return cont, sim, log
        info = self._agc_mod.gains(agc, cont, pilot)
        g_t, g_r = (1.0, 1.0) if agc.force_unit else (info["g_t"], info["g_r"])
        out = self._agc_mod.apply(cont, g_t, g_r, agc.limit)
        clipped = sum(int(np.sum(np.abs(cont[:, sl].astype(np.float64) * g) > agc.limit))
                      for sl, g in ((self._agc_mod.TRANS, g_t), (self._agc_mod.ROT, g_r)) if g != 1.0)
        info.update(applied_t=g_t, applied_r=g_r, clipped=clipped)
        log.update(pre_agc_cont_actions_f32=b64(cont), agc=info)
        return out, None if sim is None else self._agc_mod.apply(sim, g_t, g_r, agc.limit), log

    def _chunk_logs(self, action, continuous=None) -> dict:
        """Full chunk as float32 base64 [T, 7]: normalized model output, and de-normalized
        before gripper binarisation (the executed chunk differs only in the gripper sign; with
        ``agc``, ``continuous`` is the rescaled chunk)."""
        import base64

        if continuous is None:
            continuous = self._continuous(action)
        return {
            "raw_actions_f32": base64.b64encode(np.asarray(action.numpy(), dtype=np.float32).tobytes()).decode(),
            "cont_actions_f32": base64.b64encode(np.asarray(continuous, dtype=np.float32).tobytes()).decode(),
        }

    def close(self) -> None:
        self.runner = None
        self.shadow = None
        self.model = None
        torch = getattr(self, "_torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_policy(config: dict, device: str):
    return FastWAMJointPolicy(config, device)
