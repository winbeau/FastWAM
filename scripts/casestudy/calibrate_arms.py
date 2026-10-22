"""Open-loop calibration of case-study arms on recorded LIBERO observations.

Loads FastWAM-Joint once through the action-eval adapter, then replays observations
saved by an action-eval run (``save_observations: true``) through every arm in
``--arms``. For each arm and observation it reports:

* per-call model latency (CUDA-synchronised, after ``--warmup`` discarded calls);
* deviation from the ``dense`` arm at the same seed, in the model's normalized action
  space (first ``--executed`` actions and the full chunk);
* computed future-token-steps relative to dense (from the arm's diagnostics).

It also checks that ``dense`` (frame-0 reuse) matches the native ``infer_action`` path
and that ``dense`` repeated on the same input is bitwise identical.

Run on the GPU server only, inside the FastWAM venv, with the action-eval SDK on
PYTHONPATH:

    python scripts/casestudy/calibrate_arms.py --arms scripts/casestudy/arms_fastwam_joint.yaml \
        --observations '<run>/episodes/*/attempts/0000/observations/*.npz' --out calib.json ...
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", ROOT / "evaluation" / "action_eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_obs(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {
            "agentview": z["image_agentview"],
            "wrist": z["image_wrist"],
            "state": z["state"].astype(np.float32),
            "instruction": str(z["instruction"]),
            "path": path,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default=str(ROOT / "configs" / "sim_libero.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-stats", required=True)
    ap.add_argument("--diffsynth-base", required=True)
    ap.add_argument("--arms", required=True, help="YAML: list of {name, casestudy}")
    ap.add_argument("--prompt-embeddings", default=None)
    ap.add_argument("--task", default="libero_joint_2cam224_1e-4")
    ap.add_argument("--hydra-override", action="append", default=[])
    ap.add_argument("--observations", required=True, help="glob of observation .npz files")
    ap.add_argument("--max-observations", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--executed", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from infer import FastWAMJointPolicy  # the adapter
    from fastwam.casestudy import call_seed, make_runner, parse_config, timed

    arms = yaml.safe_load(Path(args.arms).read_text())
    names = [a["name"] for a in arms]
    if "dense" not in names:
        raise SystemExit("--arms must contain an arm named 'dense'")
    paths = sorted(glob.glob(args.observations))[: args.max_observations]
    if not paths:
        raise SystemExit(f"no observations match {args.observations}")
    observations = [_load_obs(p) for p in paths]

    policy = FastWAMJointPolicy(
        {
            "model_root": str(ROOT),
            "model_config": args.model_config,
            "checkpoint": args.checkpoint,
            "dataset_stats_path": args.dataset_stats,
            "diffsynth_model_base_path": args.diffsynth_base,
            "hash_checkpoint": False,
            "task": args.task,
            "hydra_overrides": args.hydra_override,
            **({"prompt_embeddings": args.prompt_embeddings} if args.prompt_embeddings else {}),
        },
        "cuda:0",
    )
    io = policy._io

    def model_inputs(obs):
        rgb = io.compose_model_image(obs["agentview"], obs["wrist"], processor=policy.processor,
                                     concatenation=policy.concatenation, width=policy.input_w, height=policy.input_h)
        image = io.image_to_model_tensor(rgb, device="cuda:0", dtype=policy.model.torch_dtype)
        proprio = io.normalize_proprio(obs["state"], policy.processor)
        prompt = policy._prompt_template.format(task=obs["instruction"])
        return image, proprio, prompt

    def run(runner, obs, index):
        image, proprio, prompt = model_inputs(obs)
        seed = call_seed(f"calib|{obs['path']}", index, runner.cfg.seed_offset)
        (action, diag), seconds = timed(
            runner.infer_action, prompt=prompt, input_image=image, proprio=proprio,
            action_horizon=policy.action_horizon, num_video_frames=policy.num_video_frames,
            seed=seed, sigma_shift=policy.sigma_shift, rand_device=policy.rand_device,
        )
        return action.numpy(), diag, seconds

    results: dict[str, dict] = {}
    reference: dict[int, np.ndarray] = {}
    order = ["dense"] + [n for n in names if n != "dense"]
    for name in order:
        arm = next(a for a in arms if a["name"] == name)
        runner = make_runner(policy.model, parse_config(arm.get("casestudy")), prompt_store=policy.prompt_store)
        latencies, dev_exec, dev_full, token_frac, computed = [], [], [], [], []
        for _ in range(args.warmup):
            runner.reset_episode()
            run(runner, observations[0], 0)
        for i, obs in enumerate(observations):
            runner.reset_episode()
            action, diag, seconds = run(runner, obs, i)
            latencies.append(seconds)
            if name == "dense":
                reference[i] = action
            else:
                d = action - reference[i]
                dev_exec.append(float(np.linalg.norm(d[: args.executed]) / np.sqrt(d[: args.executed].size)))
                dev_full.append(float(np.linalg.norm(d) / np.sqrt(d.size)))
            if "future_token_steps" in diag:
                token_frac.append(diag["future_token_steps"] / diag["future_token_steps_dense"])
            computed.append(diag.get("computed_steps", diag.get("steps")))
        results[name] = {
            "casestudy": runner.cfg.describe(),
            "latency_median_s": statistics.median(latencies),
            "latency_mean_s": statistics.fmean(latencies),
            "rms_dev_executed_median": statistics.median(dev_exec) if dev_exec else 0.0,
            "rms_dev_full_median": statistics.median(dev_full) if dev_full else 0.0,
            "future_token_fraction_mean": statistics.fmean(token_frac) if token_frac else 1.0,
            "computed_steps_mean": statistics.fmean(computed),
        }
        print(f"[calib] {name}: {results[name]}", flush=True)
    dense_latency = results["dense"]["latency_median_s"]
    for name in results:
        results[name]["speedup_vs_dense"] = dense_latency / results[name]["latency_median_s"]

    # Equivalence and determinism of the dense path.
    dense = make_runner(policy.model, parse_config({"method": "dense"}), prompt_store=policy.prompt_store)
    native = make_runner(
        policy.model, parse_config({"method": "dense", "frame0_cache": False, "prompt_cache": False}),
        prompt_store=policy.prompt_store,
    )
    eq, det = [], []
    for i, obs in enumerate(observations[:10]):
        dense.reset_episode(); a1, _, _ = run(dense, obs, i)
        dense.reset_episode(); a2, _, _ = run(dense, obs, i)
        native.reset_episode(); a3, _, _ = run(native, obs, i)
        det.append(float(np.abs(a1 - a2).max()))
        eq.append(float(np.abs(a1 - a3).max()))
    native_latency = []
    for i, obs in enumerate(observations[:10]):
        native.reset_episode(); native_latency.append(run(native, obs, i)[2])
    summary = {
        "observations": len(observations),
        "arms": results,
        "dense_repeat_max_abs_diff": max(det),
        "dense_vs_native_max_abs_diff": max(eq),
        "native_latency_median_s": statistics.median(native_latency),
        "gpu": torch.cuda.get_device_name(0),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "arms"}, indent=2))


if __name__ == "__main__":
    main()
