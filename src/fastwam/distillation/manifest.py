"""Versioned detached deployment lineage for action-only students."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

SERVING_MANIFEST_VERSION = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ActionStudentServingManifest:
    """Detached deployment manifest.

    The checkpoint embeds the same contract with ``checkpoint_sha256=None``.  The detached
    copy is written only after the checkpoint exists and therefore binds its real file digest
    without introducing a circular self-hash.
    """

    checkpoint_sha256: str | None
    stats_sha256: str
    schema_sha256: str
    dataset_version: str
    dataset_lineage_sha256: str
    action_semantics: str
    action_hz: float
    axes: tuple[str, ...]
    camera_order: tuple[str, ...]
    image_sizes: tuple[tuple[int, int], ...]
    image_preprocessing: str
    text_encoder_id: str
    context_length: int
    prompt_template: str
    task_registry_sha256: str
    vision_encoder_id: str
    action_horizon: int = 32
    action_dim: int = 7
    state_dim: int = 7
    manifest_version: int = SERVING_MANIFEST_VERSION

    def validate(self, *, require_checkpoint: bool = False) -> None:
        if self.manifest_version != SERVING_MANIFEST_VERSION:
            raise ValueError(f"unsupported serving manifest version {self.manifest_version}")
        for field in ("stats_sha256", "schema_sha256", "dataset_lineage_sha256", "task_registry_sha256"):
            _digest(getattr(self, field), field)
        if require_checkpoint and self.checkpoint_sha256 is None:
            raise ValueError("detached deployment manifest requires checkpoint_sha256")
        if self.checkpoint_sha256 is not None:
            _digest(self.checkpoint_sha256, "checkpoint_sha256")
        for field in (
            "dataset_version", "action_semantics", "image_preprocessing", "text_encoder_id",
            "prompt_template", "vision_encoder_id",
        ):
            if not getattr(self, field):
                raise ValueError(f"{field} must not be empty")
        if self.action_hz <= 0 or self.context_length <= 0:
            raise ValueError("action_hz and context_length must be positive")
        if (self.action_horizon, self.action_dim, self.state_dim) != (32, 7, 7):
            raise ValueError("Panthera action student requires horizon/dim/state [32,7,7]")
        if len(self.axes) != 7 or len(set(self.axes)) != 7:
            raise ValueError("axes must contain seven unique entries in exact model order")
        if self.camera_order != ("overhead_rgb", "wrist_rgb"):
            raise ValueError("camera_order must be overhead_rgb then wrist_rgb")
        if len(self.image_sizes) != 2 or any(len(size) != 2 or min(size) <= 0 for size in self.image_sizes):
            raise ValueError("image_sizes must contain positive (height, width) per camera")

    def checkpoint_contract(self) -> "ActionStudentServingManifest":
        return replace(self, checkpoint_sha256=None)

    def bind_checkpoint(self, checkpoint_path: str | Path) -> "ActionStudentServingManifest":
        return replace(self, checkpoint_sha256=sha256_file(checkpoint_path))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def write(self, path: str | Path) -> None:
        self.validate(require_checkpoint=True)
        _atomic_write(Path(path), json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> "ActionStudentServingManifest":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid detached deployment manifest: {exc}") from exc
        result = cls.from_dict(value)
        result.validate(require_checkpoint=True)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionStudentServingManifest":
        data = dict(value)
        data["axes"] = tuple(data["axes"])
        data["camera_order"] = tuple(data["camera_order"])
        data["image_sizes"] = tuple(tuple(item) for item in data["image_sizes"])
        result = cls(**data)
        result.validate()
        return result

    def validate_assets(
        self,
        *,
        stats_path: str | Path,
        schema_path: str | Path,
        task_registry_path: str | Path,
    ) -> dict[str, str]:
        self.validate()
        for path, expected, label in (
            (stats_path, self.stats_sha256, "stats"),
            (schema_path, self.schema_sha256, "schema"),
            (task_registry_path, self.task_registry_sha256, "task registry"),
        ):
            try:
                actual = sha256_file(path)
            except OSError as exc:
                raise ValueError(f"serving {label} file unavailable: {exc}") from exc
            if actual != expected:
                raise ValueError(f"serving {label} file lineage mismatch")
        stats = _json_object(stats_path, "stats")
        for kind in ("state", "action"):
            try:
                entry = stats[kind]["default"]
                for key in ("global_min", "global_max"):
                    values = entry[key]
                    if len(values) != 7 or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                        raise ValueError
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"serving stats {kind} content mismatch") from None
        schema = _json_object(schema_path, "schema")
        if tuple(schema.get("axes", ())) != self.axes:
            raise ValueError("serving schema axes content mismatch")
        if tuple(schema.get("camera_order", ())) != self.camera_order:
            raise ValueError("serving schema camera order content mismatch")
        if schema.get("action_semantics") != self.action_semantics:
            raise ValueError("serving schema action semantics content mismatch")
        if float(schema.get("fps", -1)) != self.action_hz:
            raise ValueError("serving schema action frequency content mismatch")
        registry = _json_object(task_registry_path, "task registry")
        if not registry or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in registry.items()):
            raise ValueError("serving task registry must be a nonempty string-to-string JSON object")
        return registry

    def validate_files(
        self,
        *,
        checkpoint_path: str | Path,
        stats_path: str | Path,
        schema_path: str | Path,
        task_registry_path: str | Path,
    ) -> dict[str, str]:
        self.validate(require_checkpoint=True)
        try:
            actual = sha256_file(checkpoint_path)
        except OSError as exc:
            raise ValueError(f"serving checkpoint file unavailable: {exc}") from exc
        if actual != self.checkpoint_sha256:
            raise ValueError("serving checkpoint file lineage mismatch")
        return self.validate_assets(
            stats_path=stats_path, schema_path=schema_path, task_registry_path=task_registry_path
        )


def _json_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed serving {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"serving {label} JSON must be an object")
    return value


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(value: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value
