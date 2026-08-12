from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .base_lerobot_dataset import BaseLerobotDataset, sliding_window_with_replication
from .lerobot.datasets.video_utils import decode_video_frames

_EXPECTED_LEROBOT_VERSION = "0.4.4"
_EXPECTED_CODEBASE_VERSION = "v3.0"
_EXPECTED_SCHEMA_VERSION = "panthera-fastwam-v1"
_EXPECTED_ACTION_SEMANTICS = "next_absolute_position_waypoint_q_t_plus_1_30hz"
_EXPECTED_CAMERA_ORDER = ("overhead_rgb", "wrist_rgb")
_EXPECTED_AXES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)


@dataclass(frozen=True)
class _Episode:
    root_index: int
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    metadata: dict[str, Any]


@dataclass
class _DatasetRoot:
    path: Path
    info: dict[str, Any]
    schema: dict[str, Any]
    package_manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    rows_by_index: dict[int, dict[str, Any]]
    tasks: dict[int, str]
    episodes: list[_Episode]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_parquet_pattern(root: Path, pattern: str) -> pa.Table:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no parquet files matched {pattern!r} below {root}")
    tables = [pq.read_table(path) for path in paths]
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")


def _load_tasks(root: Path) -> dict[int, str]:
    table = pq.read_table(root / "meta/tasks.parquet")
    rows = table.to_pylist()
    tasks: dict[int, str] = {}
    for row in rows:
        task_index = int(row["task_index"])
        if "task" in row:
            task = row["task"]
        elif "__index_level_0__" in row:
            task = row["__index_level_0__"]
        else:
            strings = [value for key, value in row.items() if key != "task_index" and isinstance(value, str)]
            if len(strings) != 1:
                raise ValueError(f"cannot identify task text in meta/tasks.parquet row: {row}")
            task = strings[0]
        tasks[task_index] = str(task)
    return tasks


def _validate_identity(
    root: Path,
    info: dict[str, Any],
    schema: dict[str, Any],
    package_manifest: dict[str, Any],
    image_meta: list[dict[str, Any]],
    state_meta: list[dict[str, Any]],
    action_meta: list[dict[str, Any]],
) -> None:
    if info.get("codebase_version") != _EXPECTED_CODEBASE_VERSION:
        raise ValueError(
            f"{root}: codebase_version must be {_EXPECTED_CODEBASE_VERSION}, got {info.get('codebase_version')}"
        )
    if int(info.get("fps", -1)) != 30:
        raise ValueError(f"{root}: fps must be 30, got {info.get('fps')}")
    if package_manifest.get("lerobot_version") != _EXPECTED_LEROBOT_VERSION:
        raise ValueError(
            f"{root}: lerobot_version must be {_EXPECTED_LEROBOT_VERSION}, "
            f"got {package_manifest.get('lerobot_version')}"
        )
    if package_manifest.get("schema_version") != _EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"{root}: schema_version must be {_EXPECTED_SCHEMA_VERSION}, "
            f"got {package_manifest.get('schema_version')}"
        )
    if package_manifest.get("action_semantics") != _EXPECTED_ACTION_SEMANTICS:
        raise ValueError(f"{root}: unsupported action semantics")
    if tuple(schema.get("camera_order", ())) != _EXPECTED_CAMERA_ORDER:
        raise ValueError(f"{root}: camera order must be {_EXPECTED_CAMERA_ORDER}")
    if schema.get("color_space") != "RGB":
        raise ValueError(f"{root}: only RGB camera inputs are supported")
    if schema.get("depth_policy") != "sidecar_only_not_fastwam_rgb_input":
        raise ValueError(f"{root}: depth policy is not explicitly sidecar-only")
    if tuple(schema.get("axes", ())) != _EXPECTED_AXES:
        raise ValueError(f"{root}: axis order mismatch")

    if [meta["key"] for meta in image_meta] != list(_EXPECTED_CAMERA_ORDER):
        raise ValueError(
            f"FastWAM camera order must preserve overhead-left/wrist-right: {_EXPECTED_CAMERA_ORDER}"
        )
    if len(state_meta) != 1 or state_meta[0]["key"] != "default" or state_meta[0]["raw_shape"] != 7:
        raise ValueError("Panthera v3 adapter requires one default 7-D observation.state")
    if len(action_meta) != 1 or action_meta[0]["key"] != "default" or action_meta[0]["raw_shape"] != 7:
        raise ValueError("Panthera v3 adapter requires one default 7-D action")

    features = info.get("features", {})
    for camera in _EXPECTED_CAMERA_ORDER:
        key = f"observation.images.{camera}"
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(f"{root}: missing v3 video feature {key}")
    for key in ("observation.state", "action"):
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("shape") != [7]:
            raise ValueError(f"{root}: {key} must have shape [7]")
    unexpected_depth = [
        key
        for key, feature in features.items()
        if "depth" in key.lower() and isinstance(feature, dict) and feature.get("dtype") in {"video", "image"}
    ]
    if unexpected_depth:
        raise ValueError(
            f"{root}: depth must remain a sidecar, found model-visible features {unexpected_depth}"
        )


def _load_root(
    path: Path,
    root_index: int,
    image_meta: list[dict[str, Any]],
    state_meta: list[dict[str, Any]],
    action_meta: list[dict[str, Any]],
) -> _DatasetRoot:
    path = path.expanduser().resolve()
    info = _load_json(path / "meta/info.json")
    schema = _load_json(path / "panthera-schema.json")
    package_manifest = _load_json(path / "panthera-package-manifest.json")
    _validate_identity(path, info, schema, package_manifest, image_meta, state_meta, action_meta)

    rows = _read_parquet_pattern(path, "data/chunk-*/file-*.parquet").to_pylist()
    rows.sort(key=lambda row: int(row["index"]))
    rows_by_index = {int(row["index"]): row for row in rows}
    if len(rows_by_index) != len(rows):
        raise ValueError(f"{path}: duplicate global frame index")
    if len(rows) != int(info["total_frames"]):
        raise ValueError(f"{path}: info total_frames does not match parquet rows")

    episode_rows = _read_parquet_pattern(path, "meta/episodes/chunk-*/file-*.parquet").to_pylist()
    episode_rows.sort(key=lambda row: int(row["episode_index"]))
    episodes = [
        _Episode(
            root_index=root_index,
            episode_index=int(row["episode_index"]),
            length=int(row["length"]),
            dataset_from_index=int(row["dataset_from_index"]),
            dataset_to_index=int(row["dataset_to_index"]),
            metadata=row,
        )
        for row in episode_rows
    ]
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"{path}: info total_episodes does not match episode metadata")
    episode_indices = [episode.episode_index for episode in episodes]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError(f"{path}: duplicate episode index")
    covered_indices: set[int] = set()
    for episode in episodes:
        if episode.dataset_to_index - episode.dataset_from_index != episode.length:
            raise ValueError(f"{path}: invalid bounds for episode {episode.episode_index}")
        episode_indices_covered = set(
            range(episode.dataset_from_index, episode.dataset_to_index)
        )
        overlap = covered_indices & episode_indices_covered
        if overlap:
            raise ValueError(f"{path}: overlapping episode frame ranges at {sorted(overlap)}")
        covered_indices.update(episode_indices_covered)
        for index in episode_indices_covered:
            row = rows_by_index.get(index)
            if row is None or int(row["episode_index"]) != episode.episode_index:
                raise ValueError(f"{path}: missing or misassigned frame {index}")
    if covered_indices != set(rows_by_index):
        raise ValueError(f"{path}: episode metadata does not cover every frame exactly once")

    return _DatasetRoot(
        path=path,
        info=info,
        schema=schema,
        package_manifest=package_manifest,
        rows=rows,
        rows_by_index=rows_by_index,
        tasks=_load_tasks(path),
        episodes=episodes,
    )


class PantheraLeRobotV3Dataset(BaseLerobotDataset):
    """Strict local-only reader for the pinned Panthera LeRobotDataset v3 contract."""

    def __init__(
        self,
        dataset_dirs: list[str],
        shape_meta: dict[str, Any],
        action_size: int = 1,
        past_action_size: int = 0,
        obs_size: int = 1,
        past_obs_size: int = 0,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,
        global_sample_stride: int = 1,
        video_backend: str = "pyav",
        video_tolerance_s: float = 1e-4,
    ) -> None:
        if not dataset_dirs:
            raise ValueError("at least one Panthera v3 dataset directory is required")
        if past_action_size != 0 or past_obs_size != 0:
            raise ValueError("Panthera v3 adapter currently supports future-only windows")
        if action_size != obs_size - 1:
            raise ValueError("action_size must equal obs_size - 1")
        if global_sample_stride < 1:
            raise ValueError("global_sample_stride must be positive")

        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.obs_size = obs_size
        self.global_sample_stride = global_sample_stride
        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set
        self.video_backend = video_backend
        self.video_tolerance_s = video_tolerance_s
        self.processor = None
        self.return_images = True
        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        self._roots = [
            _load_root(Path(path), index, self.image_meta, self.state_meta, self.action_meta)
            for index, path in enumerate(dataset_dirs)
        ]
        all_episodes = [(root, episode) for root in self._roots for episode in root.episodes]
        rng = np.random.default_rng(seed)
        order = np.arange(len(all_episodes))
        rng.shuffle(order)
        split_index = int(len(order) * (1 - val_set_proportion))
        selected_indices = order[:split_index] if is_training_set else order[split_index:]
        if val_set_proportion < 1e-6:
            selected_indices = order
        self._episodes = [all_episodes[int(index)] for index in selected_indices]
        if not self._episodes:
            raise ValueError("episode split is empty; adjust val_set_proportion")
        self._samples = [
            (root, episode, frame_index)
            for root, episode in self._episodes
            for frame_index in range(episode.length)
        ]
        self.multi_dataset = SimpleNamespace(
            num_episodes=len(self._episodes),
            num_frames=len(self._samples),
        )

    def _set_return_images(self, flag: bool) -> None:
        self.return_images = flag

    def __len__(self) -> int:
        return len(self._samples)

    def _window(self, episode: _Episode, start: int, length: int) -> tuple[list[int], torch.Tensor]:
        raw = [start + offset * self.global_sample_stride for offset in range(length)]
        pad = torch.tensor([index >= episode.length for index in raw], dtype=torch.bool)
        local = [min(index, episode.length - 1) for index in raw]
        return local, pad

    @staticmethod
    def _row(root: _DatasetRoot, episode: _Episode, local_index: int) -> dict[str, Any]:
        return root.rows_by_index[episode.dataset_from_index + local_index]

    def _decode_images(
        self,
        root: _DatasetRoot,
        episode: _Episode,
        local_indices: list[int],
    ) -> dict[str, torch.Tensor]:
        images: dict[str, torch.Tensor] = {}
        video_template = str(root.info["video_path"])
        for meta in self.image_meta:
            camera = meta["key"]
            video_key = f"observation.images.{camera}"
            chunk_index = int(episode.metadata[f"videos/{video_key}/chunk_index"])
            file_index = int(episode.metadata[f"videos/{video_key}/file_index"])
            video_path = root.path / video_template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            start_timestamp = float(episode.metadata[f"videos/{video_key}/from_timestamp"])
            timestamps = [
                start_timestamp + float(self._row(root, episode, index)["timestamp"])
                for index in local_indices
            ]
            frames = decode_video_frames(
                video_path,
                timestamps,
                tolerance_s=self.video_tolerance_s,
                backend=self.video_backend,
            )
            frames = (frames.clamp(0, 1) * 255.0).round().to(torch.uint8)
            feature_shape = tuple(root.info["features"][video_key]["shape"])
            expected = (feature_shape[2], feature_shape[0], feature_shape[1])
            if tuple(frames.shape[1:]) != expected:
                raise ValueError(
                    f"decoded {video_key} shape {tuple(frames.shape[1:])} does not match metadata {expected}"
                )
            images[camera] = frames
        return images

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        root, episode, start = self._samples[idx]
        obs_indices, obs_pad = self._window(episode, start, self.obs_size)
        action_indices, action_pad = self._window(episode, start, self.action_size)
        obs_rows = [self._row(root, episode, index) for index in obs_indices]
        action_rows = [self._row(root, episode, index) for index in action_indices]
        first_row = obs_rows[0]
        task_index = int(first_row["task_index"])
        if task_index not in root.tasks:
            raise ValueError(f"task_index {task_index} is missing from meta/tasks.parquet")

        sample = {
            "idx": idx,
            "task": root.tasks[task_index],
            "action": {"default": torch.tensor([row["action"] for row in action_rows], dtype=torch.float32)},
            "state": {
                "default": torch.tensor([row["observation.state"] for row in obs_rows], dtype=torch.float32)
            },
            "images": self._decode_images(root, episode, obs_indices) if self.return_images else {},
            "action_is_pad": action_pad,
            "state_is_pad": obs_pad.clone(),
            "image_is_pad": obs_pad.clone(),
            "episode_index": episode.episode_index,
            "panthera.tick_monotonic_ns": int(first_row["panthera.tick_monotonic_ns"]),
            "panthera.state_sequence": int(first_row["panthera.state_sequence"]),
        }
        if self.processor is not None:
            sample = self.processor.preprocess(sample)
        return sample

    def _get_episode_data(self, episode_idx: int) -> dict[str, dict[str, torch.Tensor]]:
        root, episode = self._episodes[episode_idx]
        rows = [self._row(root, episode, index) for index in range(episode.length)]
        states = torch.tensor([row["observation.state"] for row in rows], dtype=torch.float32)
        actions = torch.tensor([row["action"] for row in rows], dtype=torch.float32)
        return {
            "state": {"default": states.unsqueeze(1)},
            "action": {"default": sliding_window_with_replication(actions, self.action_size)},
        }
