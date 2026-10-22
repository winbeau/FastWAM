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

    # Interleaved measurement: every observation runs every arm back to back, so the
    # per-observation latency ratio to dense cancels drift in shared-GPU load.
    order = ["dense"] + [n for n in names if n != "dense"]
    runners = {}
    for name in order:
        arm = next(a for a in arms if a["name"] == name)
        runners[name] = make_runner(policy.model, parse_config(arm.get("casestudy")), prompt_store=policy.prompt_store)
    for _ in range(args.warmup):
        for runner in runners.values():
            runner.reset_episode()
            run(runner, observations[0], 0)
    rec = {n: {"lat": [], "ratio": [], "dev_exec": [], "dev_full": [], "tok": [], "steps": []} for n in order}
    for i, obs in enumerate(observations):
        ref, ref_lat = None, None
        for name in order:
            runner = runners[name]
            runner.reset_episode()
            action, diag, seconds = run(runner, obs, i)
            r = rec[name]
            r["lat"].append(seconds)
            if name == "dense":
                ref, ref_lat = action, seconds
            else:
                d = action - ref
                r["dev_exec"].append(float(np.sqrt(np.mean(d[: args.executed] ** 2))))
                r["dev_full"].append(float(np.sqrt(np.mean(d ** 2))))
            r["ratio"].append(ref_lat / seconds)
            if "future_token_steps" in diag:
                r["tok"].append(diag["future_token_steps"] / diag["future_token_steps_dense"])
            r["steps"].append(diag.get("computed_steps", diag.get("steps")))
        print(f"[calib] observation {i + 1}/{len(observations)}", flush=True)

    def pct(xs, q):
        return float(np.percentile(xs, q)) if xs else 0.0

    results = {}
    for name in order:
        r = rec[name]
        results[name] = {
            "casestudy": runners[name].cfg.describe(),
            "latency_median_s": statistics.median(r["lat"]),
            "speedup_vs_dense": statistics.median(r["ratio"]),
            "rms_dev_executed_median": pct(r["dev_exec"], 50),
            "rms_dev_executed_p90": pct(r["dev_exec"], 90),
            "rms_dev_executed_p95": pct(r["dev_exec"], 95),
            "rms_dev_full_median": pct(r["dev_full"], 50),
            "future_token_fraction_mean": statistics.fmean(r["tok"]) if r["tok"] else 1.0,
            "computed_steps_mean": statistics.fmean(r["steps"]),
        }
        print(f"[calib] {name}: speedup={results[name]['speedup_vs_dense']:.2f} dev_exec={results[name]['rms_dev_executed_median']:.4f}", flush=True)

    # Equivalence (optimized dense vs native infer_action) and determinism of the dense path.
    dense = runners["dense"]
    native = make_runner(
        policy.model, parse_config({"method": "dense", "frame0_cache": False, "prompt_cache": False}),
        prompt_store=policy.prompt_store,
    )
    eq_max, eq_rms, det, nat_ratio = [], [], [], []
    for i, obs in enumerate(observations[:10]):
        dense.reset_episode(); a1, _, s1 = run(dense, obs, i)
        dense.reset_episode(); a2, _, _ = run(dense, obs, i)
        native.reset_episode(); a3, _, s3 = run(native, obs, i)
        det.append(float(np.abs(a1 - a2).max()))
        eq_max.append(float(np.abs(a1 - a3).max()))
        eq_rms.append(float(np.sqrt(np.mean((a1[: args.executed] - a3[: args.executed]) ** 2))))
        nat_ratio.append(s3 / s1)
    summary = {
        "observations": len(observations),
        "timing": "interleaved per observation; speedup = median of per-observation dense/arm latency ratios",
        "arms": results,
        "dense_repeat_max_abs_diff": max(det),
        "dense_vs_native_max_abs_diff": max(eq_max),
        "dense_vs_native_rms_executed_median": statistics.median(eq_rms),
        "dense_speedup_vs_native": statistics.median(nat_ratio),
        "gpu": torch.cuda.get_device_name(0),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "arms"}, indent=2))


if __name__ == "__main__":
    main()
