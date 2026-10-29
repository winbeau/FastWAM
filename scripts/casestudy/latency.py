"""Call latency of dense, S0 and V2 (bf16 SDPA path) on one GPU, and V2's extra K/V memory.

Loads the model once through the action-eval adapter, with the options of a run's
``resolved.yaml``, and builds one runner per arm on that model: dense; A->V pruning at keep
20 / 25 / 50 % (S0); the same with the dropped tokens' K/V kept as keys for the video and action
rows (V2). After warm-up calls (which also fix each runner's layer choice), ``infer_action`` is
timed round-robin: one call of every arm per round, so load drift on a shared card hits every
arm alike. The observation is synthetic (random camera images, zero proprio, a real LIBERO
instruction); the cost of a call depends on the shapes, not on the content.

Memory: the peak allocated during one call of each arm, above what was allocated before it,
and the bytes of V2's cached dropped K/V (``diag["cache_kv_bytes"]``, all layers).

    CUDA_VISIBLE_DEVICES=N .venv/bin/python scripts/casestudy/latency.py \\
        --resolved $W/runs/<any FastWAM-Joint run>/resolved.yaml --calls 200 --warmup 10 --out latency.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

KEEPS = (0.2, 0.25, 0.5)
INSTRUCTION = "pick up the black bowl from table center and place it on the plate"


def arms() -> dict[str, dict]:
    out = {"dense": {"method": "dense", "num_inference_steps": 10, "record_attention": False}}
    for keep in KEEPS:
        base = {"selection": "av", "keep_ratio": keep, "warmup_steps": 1}
        tag = f"k{round(keep * 100)}"
        out[f"S0_{tag}"] = {"method": "prune", "num_inference_steps": 10, "record_attention": False, "prune": base}
        out[f"V2_{tag}"] = {"method": "prune", "num_inference_steps": 10, "record_attention": False,
                            "prune": {**base, "action_full_kv": True, "video_full_kv": True}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", required=True, help="resolved.yaml of a FastWAM-Joint run (model and options)")
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pol = yaml.safe_load(Path(args.resolved).read_text())["policy"]
    options = {"model_root": pol["repo_root"], "model_config": pol["model_config"], "checkpoint": pol["checkpoint"],
               **{k: v for k, v in pol["options"].items() if k != "casestudy_file"}, "casestudy": arms()["dense"]}
    spec = importlib.util.spec_from_file_location("fastwam_action_eval_infer", ROOT / "evaluation/action_eval/infer.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    policy = adapter.create_policy(options, "cuda:0")
    torch = policy._torch
    from fastwam.casestudy import make_runner, parse_config, timed

    runners = {name: make_runner(policy.model, parse_config(cfg), prompt_store=policy.prompt_store)
               for name, cfg in arms().items()}
    rng = np.random.default_rng(0)
    cams = {c: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for c in ("agentview", "wrist")}
    rgb = policy._io.compose_model_image(cams["agentview"], cams["wrist"], processor=policy.processor,
                                         concatenation=policy.concatenation, width=policy.input_w, height=policy.input_h)
    image = policy._io.image_to_model_tensor(rgb, device=policy.device, dtype=policy.model.torch_dtype)
    proprio = policy._io.normalize_proprio(np.zeros(8, dtype=np.float32), policy.processor)
    prompt = policy._prompt_template.format(task=INSTRUCTION)

    def call(runner, seed: int):
        return runner.infer_action(prompt=prompt, input_image=image, proprio=proprio, action_horizon=policy.action_horizon,
                                   num_video_frames=policy.num_video_frames, seed=seed, sigma_shift=policy.sigma_shift,
                                   rand_device=policy.rand_device)

    started = time.time()
    for i in range(args.warmup):
        for runner in runners.values():
            call(runner, 1_000_000 + i)
    secs: dict[str, list[float]] = {name: [] for name in runners}
    kept: dict[str, int | None] = {}
    for i in range(args.calls):
        for name, runner in runners.items():
            (_, diag), sec = timed(call, runner, i)
            secs[name].append(sec)
            kept[name] = diag.get("kept_count")
    memory = {}
    for name, runner in runners.items():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        _, diag = call(runner, 2_000_000)
        torch.cuda.synchronize()
        cache = diag.get("cache_kv_bytes")
        memory[name] = {"peak_above_start_bytes": int(torch.cuda.max_memory_allocated() - before),
                        "cache_kv_bytes": cache,
                        "cache_kv_bytes_per_layer": None if cache is None else int(cache // runner.num_layers)}
    report = {
        "gpu": torch.cuda.get_device_name(0), "calls": args.calls, "warmup": args.warmup,
        "order": "round-robin over arms", "observation": "synthetic (random images, zero proprio)",
        "instruction": INSTRUCTION, "arms": arms(), "wall_seconds": round(time.time() - started, 1),
        "latency_ms": {name: {"median": round(1e3 * float(np.median(v)), 2), "mean": round(1e3 * float(np.mean(v)), 2),
                              "p10": round(1e3 * float(np.percentile(v, 10)), 2), "p90": round(1e3 * float(np.percentile(v, 90)), 2),
                              "kept_count": kept[name]}
                       for name, v in secs.items()},
        "memory": memory,
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    for name, v in report["latency_ms"].items():
        m = memory[name]
        print(f"{name:8s} median {v['median']:8.2f} ms  mean {v['mean']:8.2f}  p10 {v['p10']:8.2f}  p90 {v['p90']:8.2f}  "
              f"kept {v['kept_count']}  peak +{m['peak_above_start_bytes'] / 2**20:.0f} MiB  "
              f"cache {None if m['cache_kv_bytes'] is None else round(m['cache_kv_bytes'] / 2**20, 2)} MiB")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
