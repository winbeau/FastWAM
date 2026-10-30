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
``probe_arms``           optional list of {name, casestudy}: open-loop probes (as on main). At every call (or only the
                         calls in ``probe_calls``) each probe arm runs on the same observation once per offset in
                         ``probe_seed_offsets`` (default [0]), sharing the executed arm's layer choice; chunks and diagnostics
                         go to ``diagnostics.probes["<name>@<offset>"]``; probe actions are never executed.
``seed_offset_per_repeat`` add the episode's repeat index to every seed offset (executed, shadow, probes), so repeats
                         of one episode are independent noise draws; repeat 0 keeps a single-repeat run's noise.
``episode_plans``        optional JSON {episode_id: {"calls": {"<k>": casestudy}, "from_call":
                         {"call": k, "casestudy": {...}}}}: per-call arm overrides used by the
                         attribution (patch at one call) and switch-point (arm from call k on)
                         experiments; every other call runs the base arm.
                         An episode plan may list ``shadow_calls`` (call indices): the shadow arm then
                         runs only on those calls (patch experiments that need one deviation).
``prompt_embeddings``    optional ``.pt`` from scripts/casestudy/precompute_prompts.py; when set the
                         text encoder is not loaded and unknown instructions are an error.
``hash_checkpoint``      default true.
``checkpoint_sha256``    optional expected digest; a mismatch is fatal.

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
        from fastwam.casestudy import call_seed, make_runner, parse_config
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from fastwam.utils.config_resolvers import register_default_resolvers

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch sees no GPU; refusing to fall back to CPU")
        self._torch = torch
        self._io = policy_io
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
            "probes": [{"name": n, "casestudy": r.cfg.describe()} for n, r in self.probes] or None,
            "probe_seed_offsets": self.probe_seed_offsets if self.probes else None,
            "seed_offset_per_repeat": self.seed_offset_per_repeat,
            "prompt_embeddings_sha256": self.prompt_store_sha256,
            "episode_plans_sha256": self.episode_plans_sha256,
            "episode_plans_count": len(self.episode_plans),
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
        self.predict_calls = 0
        for _name, runner in self.probes:
            runner.reset_episode()
        self._repeat_offset = int(episode.get("repeat") or 0) if self.seed_offset_per_repeat else 0
        # Seed key = (instruction, task, init, protocol seed); the repeat index is left out so
        # that replays of one episode (patch / switch-point experiments use repeats to index
        # variants) draw exactly the noise of the original rollout at every call index.
        import re as _re

        episode_id = str(episode.get("episode_id", ""))
        self._episode_id = episode_id
        self._episode_key = f"{episode.get('instruction', '')}|{_re.sub(r'-r[0-9]+', '', episode_id)}"
        self._plan = self.episode_plans.get(episode_id, {})

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
        if self.casestudy.seed_mode == "episode_call_hash":
            seed = self._call_seed(self._episode_key, call_index, self.casestudy.seed_offset + self._repeat_offset)
        else:
            seed = self.native_seed

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_started = time.perf_counter()
        runner, override = self._runner_for_call(call_index)
        action, diag = runner.infer_action(
            prompt=prompt, input_image=image, proprio=proprio,
            action_horizon=self.action_horizon, num_video_frames=self.num_video_frames,
            seed=seed, sigma_shift=self.sigma_shift, rand_device=self.rand_device,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_seconds = time.perf_counter() - model_started
        actions = self._io.postprocess_action(action.clone(), self.processor, binarize_gripper=self.binarize_gripper)
        actions = np.asarray(actions, dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("FastWAM produced non-finite actions")
        self.predict_calls += 1
        diag.update({"call_index": call_index, "model_seconds": round(model_seconds, 6)})
        diag.update(self._chunk_logs(action))
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
        return Prediction(
            actions=actions,
            timing={"predict_seconds": round(time.perf_counter() - started, 6), "model_seconds": round(model_seconds, 6)},
            diagnostics=diag,
        )

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
        }

    def _probe_metrics(self, prompt, image, proprio, call_index: int) -> dict:
        """Every probe arm at every probe seed offset on this observation: chunks and diagnostics (as on main)."""
        layers = getattr(self.runner, "_layers", None)
        out = {}
        for offset in self.probe_seed_offsets:
            seed = self._call_seed(self._episode_key, call_index, offset + self._repeat_offset)
            for name, runner in self.probes:
                if layers is not None and int(layers.get("act", -1)) >= 0:
                    runner._layers = dict(layers)
                action, d = runner.infer_action(
                    prompt=prompt, input_image=image, proprio=proprio,
                    action_horizon=self.action_horizon, num_video_frames=self.num_video_frames,
                    seed=seed, sigma_shift=self.sigma_shift, rand_device=self.rand_device,
                )
                entry = {"seed": int(seed), **self._chunk_logs(action)}
                for key in ("kept_index", "kept_count", "layer_act", "modes", "av_scores_f16", "gate_score", "future_token_steps",
                            "future_token_steps_dense", "reuse_step0_kv", "path", "fovea"):
                    if d.get(key) is not None:
                        entry[key] = d[key]
                out[f"{name}@{offset}"] = entry
        return out

    def _continuous(self, action) -> np.ndarray:
        """De-normalized chunk [T, 7] before gripper binarisation (float32)."""
        return np.asarray(self._io.postprocess_action(action.clone(), self.processor, binarize_gripper=False), dtype=np.float32)

    def _chunk_logs(self, action) -> dict:
        """Full chunk as float32 base64 [T, 7]: normalized model output, and de-normalized before gripper binarisation
        (the executed chunk differs only in the gripper sign)."""
        import base64

        return {
            "raw_actions_f32": base64.b64encode(np.asarray(action.numpy(), dtype=np.float32).tobytes()).decode(),
            "cont_actions_f32": base64.b64encode(self._continuous(action).tobytes()).decode(),
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
