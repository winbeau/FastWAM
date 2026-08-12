from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torchvision.transforms import Resize

from fastwam.datasets.lerobot.lerobot_v3_adapter import PantheraLeRobotV3Dataset
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.transforms.image import ToTensor
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
from fastwam.utils import misc

FIXTURE = Path(__file__).parent / "fixtures/panthera_lerobot_v3_minimal"


def shape_meta() -> dict:
    return {
        "images": [
            {"key": "overhead_rgb", "raw_shape": [3, 64, 64], "shape": [3, 128, 128]},
            {"key": "wrist_rgb", "raw_shape": [3, 64, 64], "shape": [3, 128, 128]},
        ],
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 7, "shape": 7}],
    }


def make_processor(*, num_obs_steps: int = 5, image_size: int = 128) -> FastWAMProcessor:
    meta = shape_meta()
    for image in meta["images"]:
        image["shape"] = [3, image_size, image_size]
    resize = [ToTensor(), Resize(size=[image_size, image_size])]
    return FastWAMProcessor(
        shape_meta=meta,
        num_obs_steps=num_obs_steps,
        num_output_cameras=2,
        action_output_dim=7,
        proprio_output_dim=7,
        delta_action_dim_mask={"default": [False] * 7},
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=resize,
        val_transforms=resize,
    )


def write_text_cache(tmp_path: Path) -> Path:
    prompt = DEFAULT_PROMPT.format(task="Move the red block from the start area to the target area.")
    cache_dir = tmp_path / "text-cache"
    cache_dir.mkdir()
    cache_path = cache_dir / f"{hashlib.sha256(prompt.encode()).hexdigest()}.t5_len128.wan22ti2v5b.pt"
    torch.save(
        {"context": torch.zeros(128, 4096), "mask": torch.ones(128, dtype=torch.bool)},
        cache_path,
    )
    return cache_dir


def test_v3_fixture_reads_images_state_action_and_padding() -> None:
    dataset = PantheraLeRobotV3Dataset(
        [str(FIXTURE)],
        shape_meta(),
        obs_size=5,
        action_size=4,
        val_set_proportion=0.0,
        is_training_set=True,
        video_backend="pyav",
    )

    first = dataset[0]
    last = dataset[len(dataset) - 1]
    assert len(dataset) == 5
    assert first["images"]["overhead_rgb"].shape == (5, 3, 64, 64)
    assert first["images"]["wrist_rgb"].shape == (5, 3, 64, 64)
    assert first["images"]["overhead_rgb"].dtype == torch.uint8
    assert first["state"]["default"].shape == (5, 7)
    assert first["action"]["default"].shape == (4, 7)
    torch.testing.assert_close(first["action"]["default"][0], first["state"]["default"][1])
    assert first["image_is_pad"].tolist() == [False] * 5
    assert first["action_is_pad"].tolist() == [False] * 4
    assert last["image_is_pad"].tolist() == [False, True, True, True, True]
    assert last["action_is_pad"].tolist() == [False, True, True, True]


def test_robot_video_dataset_produces_fastwam_smoke_contract(tmp_path: Path) -> None:
    processor = make_processor()
    raw = PantheraLeRobotV3Dataset(
        [str(FIXTURE)],
        shape_meta(),
        obs_size=5,
        action_size=4,
        val_set_proportion=0.0,
        is_training_set=True,
        video_backend="pyav",
    )
    stats = raw.get_dataset_stats(processor)
    stats_path = tmp_path / "dataset_stats.json"
    save_dataset_stats_to_json(stats, stats_path)

    cache_dir = write_text_cache(tmp_path)

    misc.register_work_dir(tmp_path / "run")
    dataset = RobotVideoDataset(
        dataset_dirs=[str(FIXTURE)],
        dataset_format="panthera_v3",
        video_backend="pyav",
        shape_meta=shape_meta(),
        num_frames=5,
        video_size=[128, 256],
        processor=processor,
        text_embedding_cache_dir=str(cache_dir),
        context_len=128,
        pretrained_norm_stats=str(stats_path),
        val_set_proportion=0.0,
        is_training_set=True,
        action_video_freq_ratio=1,
        concat_multi_camera="horizontal",
        random_fallback_on_error=False,
    )
    sample = dataset[0]

    assert sample["video"].shape == (3, 5, 128, 256)
    assert sample["action"].shape == (4, 7)
    assert sample["proprio"].shape == (4, 7)
    assert sample["context"].shape == (128, 4096)
    assert sample["context_mask"].shape == (128,)
    assert sample["image_is_pad"].shape == (5,)
    assert sample["action_is_pad"].shape == (4,)
    assert sample["proprio_is_pad"].shape == (4,)
    assert torch.isfinite(sample["video"]).all()
    assert torch.isfinite(sample["action"]).all()
    assert torch.isfinite(sample["proprio"]).all()

    left = sample["video"][:, 0, :, :128]
    right = sample["video"][:, 0, :, 128:]
    assert left[0].mean() > right[0].mean()
    assert right[2].mean() > left[2].mean()


def test_absolute_action_normalization_round_trip() -> None:
    processor = make_processor()
    raw = PantheraLeRobotV3Dataset(
        [str(FIXTURE)],
        shape_meta(),
        obs_size=5,
        action_size=4,
        val_set_proportion=0.0,
        is_training_set=True,
        video_backend="pyav",
    )
    stats = raw.get_dataset_stats(processor)
    processor.set_normalizer_from_stats(stats)
    source = raw[0]
    batch = {
        "action": {"default": source["action"]["default"].clone()},
        "state": {"default": source["state"]["default"].clone()},
    }
    restored = processor.normalizer.backward(processor.normalizer.forward(deepcopy(batch)))
    torch.testing.assert_close(restored["action"]["default"], batch["action"]["default"])
    torch.testing.assert_close(restored["state"]["default"], batch["state"]["default"])
    assert processor.delta_action_dim_mask["default"].tolist() == [False] * 7


def test_production_geometry_contract_with_terminal_padding(tmp_path: Path) -> None:
    processor = make_processor(num_obs_steps=33, image_size=224)
    raw = PantheraLeRobotV3Dataset(
        [str(FIXTURE)],
        processor.shape_meta,
        obs_size=33,
        action_size=32,
        val_set_proportion=0.0,
        is_training_set=True,
        video_backend="pyav",
    )
    stats_path = tmp_path / "dataset_stats.json"
    save_dataset_stats_to_json(raw.get_dataset_stats(processor), stats_path)
    cache_dir = write_text_cache(tmp_path)
    misc.register_work_dir(tmp_path / "run")
    dataset = RobotVideoDataset(
        dataset_dirs=[str(FIXTURE)],
        dataset_format="panthera_v3",
        video_backend="pyav",
        shape_meta=processor.shape_meta,
        num_frames=33,
        video_size=[224, 448],
        processor=processor,
        text_embedding_cache_dir=str(cache_dir),
        context_len=128,
        pretrained_norm_stats=str(stats_path),
        val_set_proportion=0.0,
        is_training_set=True,
        action_video_freq_ratio=4,
        concat_multi_camera="horizontal",
        random_fallback_on_error=False,
    )
    sample = dataset[0]
    assert sample["video"].shape == (3, 9, 224, 448)
    assert sample["action"].shape == (32, 7)
    assert sample["proprio"].shape == (32, 7)
    assert sample["action_is_pad"].shape == (32,)
    assert sample["image_is_pad"].shape == (9,)
    assert sample["action_is_pad"].tolist() == [False] * 5 + [True] * 27
    assert sample["image_is_pad"].tolist() == [False, False] + [True] * 7


def test_v3_adapter_rejects_duplicate_episode_metadata(tmp_path: Path) -> None:
    corrupted = tmp_path / "dataset"
    shutil.copytree(FIXTURE, corrupted)
    episode_path = corrupted / "meta/episodes/chunk-000/file-000.parquet"
    row = pq.read_table(episode_path).to_pylist()[0]
    pq.write_table(pa.Table.from_pylist([row, row]), episode_path)
    info_path = corrupted / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = 2
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate episode index"):
        PantheraLeRobotV3Dataset(
            [str(corrupted)],
            shape_meta(),
            obs_size=5,
            action_size=4,
            val_set_proportion=0.0,
            is_training_set=True,
        )


def test_v3_adapter_rejects_camera_order_mismatch() -> None:
    bad_meta = shape_meta()
    bad_meta["images"] = list(reversed(bad_meta["images"]))
    with pytest.raises(ValueError, match="camera order"):
        PantheraLeRobotV3Dataset(
            [str(FIXTURE)],
            bad_meta,
            obs_size=5,
            action_size=4,
            val_set_proportion=0.0,
            is_training_set=True,
        )
