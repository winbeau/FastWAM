#!/usr/bin/env python3
"""Fail-fast Panthera LeRobot-v3 package preflight for FastWAM training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastwam.datasets.lerobot.lerobot_v3_adapter import PantheraLeRobotV3Dataset

EXPECTED_AXES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)
EXPECTED_ACTION_SEMANTICS = "next_absolute_position_waypoint_q_t_plus_1_30hz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, *, exclude: Iterable[str] = ()) -> str:
    excluded = set(exclude)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_parquet_pattern(root: Path, pattern: str) -> list[dict[str, Any]]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched {pattern!r} below {root}")
    tables = [pq.read_table(path) for path in paths]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return table.to_pylist()


def shape_meta() -> dict[str, Any]:
    return {
        "images": [
            {"key": "overhead_rgb", "raw_shape": [3, 64, 64], "shape": [3, 64, 64]},
            {"key": "wrist_rgb", "raw_shape": [3, 64, 64], "shape": [3, 64, 64]},
        ],
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 7, "shape": 7}],
    }


def vector7(value: Any, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite float[7], got {array}")
    return array


def require_zero_map(report: dict[str, Any], field: str) -> None:
    values = report.get(field)
    if not isinstance(values, dict) or any(int(value) != 0 for value in values.values()):
        raise ValueError(f"{field} must contain only zero counts, got {values}")


def validate_root(root: Path, *, allow_failed_tasks: bool) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = load_json(root / "panthera-package-manifest.json")
    schema = load_json(root / "panthera-schema.json")
    episode_identity = load_json(root / "panthera-episode.json")
    info = load_json(root / "meta/info.json")
    expected_content_hash = manifest.get("dataset_content_sha256")
    observed_content_hash = sha256_tree(root, exclude={"panthera-package-manifest.json"})
    if observed_content_hash != expected_content_hash:
        raise ValueError(
            f"{root}: package content hash mismatch: expected={expected_content_hash}, "
            f"observed={observed_content_hash}"
        )
    if schema.get("action_semantics") != EXPECTED_ACTION_SEMANTICS:
        raise ValueError(f"{root}: action semantics mismatch")
    if tuple(schema.get("axes", ())) != EXPECTED_AXES:
        raise ValueError(f"{root}: axis order mismatch")
    if episode_identity.get("success") is not True and not allow_failed_tasks:
        raise ValueError(f"{root}: failed task episodes are excluded from training by default")
    if manifest.get("source_panthera_commit") != episode_identity.get("panthera_wam_commit"):
        raise ValueError(f"{root}: Panthera source commit mismatch")
    if manifest.get("source_calibration_sha256") != episode_identity.get("calibration_sha256"):
        raise ValueError(f"{root}: calibration identity mismatch")
    source_calibration = root / "aux/source/calibration.json"
    if sha256_file(source_calibration) != manifest.get("source_calibration_sha256"):
        raise ValueError(f"{root}: packaged calibration hash mismatch")

    sync_report = load_json(root / "aux/source/sync_report.json")
    timestamp_quality = load_json(root / "aux/source/timestamp_quality.json")
    if int(sync_report.get("timestamp_regressions", -1)) != 0:
        raise ValueError(f"{root}: source timestamp regression")
    for field in ("missing_frames", "duplicate_frames", "sequence_gaps", "ring_overflows"):
        require_zero_map(sync_report, field)
    if not math.isclose(float(timestamp_quality.get("coverage_fraction", 0.0)), 1.0):
        raise ValueError(f"{root}: timestamp metadata coverage is not 100%")

    rows = read_parquet_pattern(root, "data/chunk-*/file-*.parquet")
    rows.sort(key=lambda row: int(row["index"]))
    episodes = read_parquet_pattern(root, "meta/episodes/chunk-*/file-*.parquet")
    sidecar = [
        json.loads(line)
        for line in (root / "aux/timestamps.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(info["total_frames"]) or len(sidecar) != len(rows):
        raise ValueError(f"{root}: parquet/sidecar/info frame counts disagree")
    if int(manifest.get("frame_count", -1)) != len(rows):
        raise ValueError(f"{root}: manifest frame_count mismatch")

    for index, (row, record) in enumerate(zip(rows, sidecar, strict=True)):
        state = vector7(row["observation.state"], field=f"row[{index}].observation.state")
        action = vector7(row["action"], field=f"row[{index}].action")
        source_action = vector7(
            record.get("action_source_state_position"),
            field=f"sidecar[{index}].action_source_state_position",
        )
        np.testing.assert_allclose(action, source_action, atol=1e-6)
        if int(record.get("frame_index", -1)) != index or record.get("sync_ok") is not True:
            raise ValueError(f"{root}: invalid sidecar row {index}")
        if record.get("action_semantics") != EXPECTED_ACTION_SEMANTICS:
            raise ValueError(f"{root}: sidecar action semantics mismatch at row {index}")
        for camera_key in ("overhead_rgb", "wrist_rgb"):
            camera = record.get(camera_key)
            if not isinstance(camera, dict):
                raise ValueError(f"{root}: sidecar {camera_key} missing at row {index}")
            if not camera.get("timestamp_source") or not camera.get("timestamp_quality"):
                raise ValueError(f"{root}: timestamp source/quality missing at row {index}")
            receive = int(camera["host_receive_monotonic_ns"])
            publish = int(camera["host_publish_monotonic_ns"])
            if receive > publish:
                raise ValueError(f"{root}: camera publish precedes receive at row {index}")
        if not np.isfinite(state).all():
            raise ValueError(f"{root}: non-finite state at row {index}")

    for episode in episodes:
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        episode_rows = rows[start:stop]
        ticks = [int(row["panthera.tick_monotonic_ns"]) for row in episode_rows]
        state_sequences = [int(row["panthera.state_sequence"]) for row in episode_rows]
        if any(right <= left for left, right in zip(ticks, ticks[1:])):
            raise ValueError(f"{root}: non-monotonic canonical ticks")
        if any(right <= left for left, right in zip(state_sequences, state_sequences[1:])):
            raise ValueError(f"{root}: non-monotonic source state sequences")
        for local_index, row in enumerate(episode_rows[:-1]):
            action = vector7(
                row["action"],
                field=f"episode action[{start + local_index}]",
            )
            next_state = vector7(
                episode_rows[local_index + 1]["observation.state"],
                field=f"episode state[{start + local_index + 1}]",
            )
            np.testing.assert_allclose(action, next_state, atol=1e-6)

    dataset = PantheraLeRobotV3Dataset(
        [str(root)],
        shape_meta(),
        obs_size=5,
        action_size=4,
        val_set_proportion=0.0,
        is_training_set=True,
        video_backend="pyav",
    )
    probe_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    decoded_shapes = {}
    for index in probe_indices:
        sample = dataset[index]
        for key, frames in sample["images"].items():
            if frames.dtype is not torch.uint8 or not torch.isfinite(frames.float()).all():
                raise ValueError(f"{root}: invalid decoded {key} frames at sample {index}")
            decoded_shapes[key] = list(frames.shape)

    return {
        "root": str(root),
        "status": "passed",
        "episodes": len(episodes),
        "frames": len(rows),
        "dataset_content_sha256": observed_content_hash,
        "source_staging_sha256": manifest.get("source_staging_sha256"),
        "decoded_probe_indices": probe_indices,
        "decoded_shapes": decoded_shapes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dirs", nargs="+", type=Path)
    parser.add_argument("--allow-failed-tasks", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "status": "passed",
        "datasets": [
            validate_root(path, allow_failed_tasks=args.allow_failed_tasks) for path in args.dataset_dirs
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
