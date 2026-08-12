#!/usr/bin/env python3
"""Prepare normalization and identity assets for a frozen Panthera dataset set."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from fastwam.datasets.lerobot.lerobot_v3_adapter import PantheraLeRobotV3Dataset
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.tasks import read_unique_tasks
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
from scripts.validate_panthera_dataset import sha256_file, validate_root


def shape_meta() -> dict[str, Any]:
    return {
        "images": [
            {"key": "overhead_rgb", "raw_shape": [3, 1, 1], "shape": [3, 1, 1]},
            {"key": "wrist_rgb", "raw_shape": [3, 1, 1], "shape": [3, 1, 1]},
        ],
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 7, "shape": 7}],
    }


def processor() -> FastWAMProcessor:
    return FastWAMProcessor(
        shape_meta=shape_meta(),
        num_obs_steps=5,
        num_output_cameras=2,
        action_output_dim=7,
        proprio_output_dim=7,
        delta_action_dim_mask={"default": [False] * 7},
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[],
        val_transforms=[],
    )


def validate_text_cache(path: Path, *, context_len: int) -> tuple[bool, str | None]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        return False, f"cannot load cache: {exc}"
    if not isinstance(payload, dict):
        return False, "cache payload must be a mapping"
    context = payload.get("context")
    mask = payload.get("mask")
    if not isinstance(context, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return False, "cache must contain tensor context and mask"
    if context.ndim != 2 or context.shape[0] != context_len:
        return False, f"context must have shape [{context_len}, D]"
    if mask.ndim != 1 or mask.shape[0] != context_len:
        return False, f"mask must have shape [{context_len}]"
    if not torch.isfinite(context).all():
        return False, "context contains NaN or Inf"
    return True, None


def expected_text_cache(
    dataset_dirs: list[str],
    cache_dir: Path | None,
    *,
    context_len: int = 128,
    encoder_id: str = "wan22ti2v5b",
) -> tuple[list[dict[str, Any]], int]:
    tasks, total_rows = read_unique_tasks(dataset_dirs)
    records = []
    for task in tasks:
        prompt = DEFAULT_PROMPT.format(task=task)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        filename = f"{prompt_sha}.t5_len{context_len}.{encoder_id}.pt"
        path = cache_dir / filename if cache_dir is not None else None
        present = bool(path and path.is_file())
        valid = False
        validation_error = None
        if present and path is not None:
            valid, validation_error = validate_text_cache(path, context_len=context_len)
        records.append(
            {
                "task": task,
                "prompt_sha256": prompt_sha,
                "filename": filename,
                "present": present,
                "valid": valid,
                "validation_error": validation_error,
                "sha256": sha256_file(path) if present and path is not None else None,
            }
        )
    return records, total_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-cache-dir", type=Path)
    parser.add_argument("--require-text-cache", action="store_true")
    parser.add_argument(
        "--compatibility-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "lerobot-compatibility.json",
    )
    args = parser.parse_args()

    dataset_dirs = [str(path.expanduser().resolve()) for path in args.dataset_dirs]
    preflight = [validate_root(Path(path), allow_failed_tasks=False) for path in dataset_dirs]
    preprocessor = processor()
    dataset = PantheraLeRobotV3Dataset(
        dataset_dirs,
        preprocessor.shape_meta,
        obs_size=5,
        action_size=4,
        val_set_proportion=0.0,
        is_training_set=True,
        video_backend="pyav",
    )
    stats = dataset.get_dataset_stats(preprocessor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "dataset_stats.json"
    save_dataset_stats_to_json(stats, stats_path)

    preprocessor.set_normalizer_from_stats(stats)
    first = dataset[0]
    physical = {
        "action": {"default": first["action"]["default"].clone()},
        "state": {"default": first["state"]["default"].clone()},
    }
    restored = preprocessor.normalizer.backward(preprocessor.normalizer.forward(deepcopy(physical)))
    torch.testing.assert_close(restored["action"]["default"], physical["action"]["default"])
    torch.testing.assert_close(restored["state"]["default"], physical["state"]["default"])

    text_records, total_task_rows = expected_text_cache(
        dataset_dirs,
        args.text_cache_dir.expanduser().resolve() if args.text_cache_dir else None,
    )
    text_ready = all(record["present"] and record["valid"] for record in text_records)
    if args.require_text_cache and not text_ready:
        invalid = [
            {
                "filename": record["filename"],
                "error": record["validation_error"] or "missing",
            }
            for record in text_records
            if not record["present"] or not record["valid"]
        ]
        raise ValueError(f"missing or invalid text embedding cache files: {invalid}")

    manifest = {
        "schema_version": 1,
        "dataset_preflight": preflight,
        "normalization": {
            "path": str(stats_path.resolve()),
            "sha256": sha256_file(stats_path),
            "mode": "global min/max",
            "absolute_action_round_trip_verified": True,
            "delta_action_dim_mask": [False] * 7,
        },
        "compatibility_manifest": {
            "path": str(args.compatibility_manifest.expanduser().resolve()),
            "sha256": sha256_file(args.compatibility_manifest),
        },
        "text_embeddings": {
            "cache_dir": str(args.text_cache_dir.expanduser().resolve()) if args.text_cache_dir else None,
            "task_rows": total_task_rows,
            "records": text_records,
            "ready": text_ready,
        },
        "training_ready": text_ready,
        "limitations": (
            [] if text_ready else ["real T5 text embeddings have not been generated for every task"]
        ),
    }
    manifest_path = args.output_dir / "asset-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
