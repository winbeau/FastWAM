"""Versioned, content-digested teacher action target cache.

JSON is acceptable for the current structural fixture scope. It is not scalable for large
training corpora; a sharded binary format with the same lineage contract is required there.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

CACHE_VERSION = 2


@dataclass(frozen=True)
class TeacherCacheManifest:
    dataset_version: str
    dataset_lineage_sha256: str
    teacher_checkpoint_sha256: str
    stats_sha256: str
    schema_sha256: str
    inference_seeds: tuple[int, ...]
    inference_steps: int
    sampler_config_sha256: str
    ensemble_count: int
    variance_definition: str | None
    variance_space: str | None
    generation_code_sha256: str
    generation_config_sha256: str
    action_horizon: int = 32
    action_dim: int = 7
    cache_version: int = CACHE_VERSION


@dataclass(frozen=True)
class TeacherTarget:
    sample_id: str
    canonical_task: str
    normalized: np.ndarray
    physical: np.ndarray
    source_episode_id: str
    source_frame_index: int
    source_content_sha256: str
    filtering_status: str = "accepted"
    filtering_reason: str | None = None
    variance: np.ndarray | None = None


def stable_sample_id(
    dataset_lineage_sha256: str,
    episode_id: str,
    frame_index: int,
    source_content_sha256: str,
) -> str:
    for value, field in (
        (dataset_lineage_sha256, "dataset_lineage_sha256"),
        (source_content_sha256, "source_content_sha256"),
    ):
        _digest(value, field)
    if not episode_id or frame_index < 0:
        raise ValueError("episode_id must be nonempty and frame_index nonnegative")
    source = f"{dataset_lineage_sha256}\0{episode_id}\0{frame_index}\0{source_content_sha256}".encode()
    return hashlib.sha256(source).hexdigest()


def write_teacher_cache(
    path: str | Path,
    manifest: TeacherCacheManifest,
    targets: Iterable[TeacherTarget],
    *,
    physical_to_normalized: Callable[[np.ndarray], np.ndarray],
) -> None:
    _validate_manifest(manifest)
    records = []
    seen: set[str] = set()
    for target in targets:
        _validate_target_identity(target, manifest)
        if target.sample_id in seen:
            raise ValueError(f"duplicate sample_id: {target.sample_id}")
        seen.add(target.sample_id)
        normalized = _actions(target.normalized, manifest, "normalized")
        physical = _actions(target.physical, manifest, "physical")
        transformed = _actions(physical_to_normalized(physical.copy()), manifest, "transformed")
        if not np.allclose(normalized, transformed, rtol=1e-5, atol=1e-6):
            raise ValueError("teacher physical/normalized actions are inconsistent")
        variance = _variance(target.variance, manifest)
        records.append(
            {
                "sample_id": target.sample_id,
                "canonical_task": target.canonical_task,
                "normalized": normalized.tolist(),
                "physical": physical.tolist(),
                "variance": None if variance is None else variance.tolist(),
                "source_episode_id": target.source_episode_id,
                "source_frame_index": target.source_frame_index,
                "source_content_sha256": target.source_content_sha256,
                "filtering_status": target.filtering_status,
                "filtering_reason": target.filtering_reason,
            }
        )
    records.sort(key=lambda item: item["sample_id"])
    body = {"manifest": asdict(manifest), "targets": records}
    payload = {**body, "content_sha256": _content_digest(body)}
    _atomic_json(Path(path), payload)


def read_teacher_cache(
    path: str | Path,
    *,
    expected: TeacherCacheManifest,
    physical_to_normalized: Callable[[np.ndarray], np.ndarray],
) -> dict[str, TeacherTarget]:
    _validate_manifest(expected)
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest_data = dict(payload["manifest"])
        manifest_data["inference_seeds"] = tuple(manifest_data["inference_seeds"])
        actual = TeacherCacheManifest(**manifest_data)
        targets = payload["targets"]
        content_sha256 = payload["content_sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed teacher cache: {exc}") from exc
    _validate_manifest(actual)
    body = {"manifest": asdict(actual), "targets": targets}
    if content_sha256 != _content_digest(body):
        raise ValueError("teacher cache whole-payload content digest mismatch")
    if actual != expected:
        mismatches = [key for key in asdict(expected) if getattr(actual, key) != getattr(expected, key)]
        raise ValueError(f"teacher cache lineage mismatch: {', '.join(mismatches)}")
    result: dict[str, TeacherTarget] = {}
    for record in targets:
        try:
            target = TeacherTarget(
                sample_id=record["sample_id"],
                canonical_task=record["canonical_task"],
                normalized=_actions(record["normalized"], expected, "normalized"),
                physical=_actions(record["physical"], expected, "physical"),
                variance=_variance(record.get("variance"), expected),
                source_episode_id=record["source_episode_id"],
                source_frame_index=record["source_frame_index"],
                source_content_sha256=record["source_content_sha256"],
                filtering_status=record["filtering_status"],
                filtering_reason=record.get("filtering_reason"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed teacher target: {exc}") from exc
        _validate_target_identity(target, expected)
        if target.sample_id in result:
            raise ValueError(f"duplicate sample_id in teacher cache: {target.sample_id}")
        transformed = _actions(physical_to_normalized(target.physical.copy()), expected, "transformed")
        if not np.allclose(target.normalized, transformed, rtol=1e-5, atol=1e-6):
            raise ValueError("teacher physical/normalized actions are inconsistent")
        result[target.sample_id] = target
    return result


def _validate_target_identity(target: TeacherTarget, manifest: TeacherCacheManifest) -> None:
    if not target.canonical_task or not target.canonical_task.strip():
        raise ValueError("canonical_task must not be empty")
    _digest(target.source_content_sha256, "source_content_sha256")
    expected_id = stable_sample_id(
        manifest.dataset_lineage_sha256,
        target.source_episode_id,
        target.source_frame_index,
        target.source_content_sha256,
    )
    if target.sample_id != expected_id:
        raise ValueError("sample_id is not the stable SHA-256 for source lineage")
    if target.filtering_status not in {"accepted", "rejected"}:
        raise ValueError("filtering_status must be accepted or rejected")
    if target.filtering_status == "rejected" and not target.filtering_reason:
        raise ValueError("rejected targets require filtering_reason")
    if target.filtering_status == "accepted" and target.filtering_reason is not None:
        raise ValueError("accepted targets must not have filtering_reason")


def _actions(value, manifest: TeacherCacheManifest, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    expected = (manifest.action_horizon, manifest.action_dim)
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError(f"{field} teacher actions must be finite with shape {expected}")
    return array


def _variance(value, manifest: TeacherCacheManifest) -> np.ndarray | None:
    if value is None:
        if manifest.ensemble_count > 1:
            raise ValueError("ensemble teacher cache requires variance")
        return None
    if manifest.ensemble_count <= 1:
        raise ValueError("variance is only valid when ensemble_count > 1")
    result = _actions(value, manifest, "variance")
    if (result < 0).any():
        raise ValueError("teacher variance must be finite and nonnegative")
    return result


def _validate_manifest(manifest: TeacherCacheManifest) -> None:
    if manifest.cache_version != CACHE_VERSION:
        raise ValueError(f"unsupported teacher cache version {manifest.cache_version}")
    if not manifest.dataset_version or manifest.inference_steps <= 0 or manifest.ensemble_count <= 0:
        raise ValueError("dataset_version and positive inference settings are required")
    if len(manifest.inference_seeds) != manifest.ensemble_count or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in manifest.inference_seeds
    ):
        raise ValueError("inference_seeds must contain one integer per ensemble member")
    if len(set(manifest.inference_seeds)) != len(manifest.inference_seeds):
        raise ValueError("inference seeds must be unique")
    for field in (
        "dataset_lineage_sha256",
        "teacher_checkpoint_sha256",
        "stats_sha256",
        "schema_sha256",
        "sampler_config_sha256",
        "generation_code_sha256",
        "generation_config_sha256",
    ):
        _digest(getattr(manifest, field), field)
    if manifest.action_horizon <= 0 or manifest.action_dim <= 0:
        raise ValueError("action shape must be positive")
    if manifest.ensemble_count > 1:
        if not manifest.variance_definition or not manifest.variance_space:
            raise ValueError("ensemble variance definition and space are required")
    elif manifest.variance_definition is not None or manifest.variance_space is not None:
        raise ValueError("single-sample cache must not claim variance semantics")


def _digest(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _content_digest(body: dict) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
