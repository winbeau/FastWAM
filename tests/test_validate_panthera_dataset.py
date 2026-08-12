from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.validate_panthera_dataset import validate_root

FIXTURE = Path(__file__).parent / "fixtures/panthera_lerobot_v3_minimal"


def test_fixture_passes_training_preflight() -> None:
    report = validate_root(FIXTURE, allow_failed_tasks=False)
    assert report["status"] == "passed"
    assert report["episodes"] == 1
    assert report["frames"] == 5
    assert report["decoded_probe_indices"] == [0, 2, 4]


def test_preflight_rejects_any_unmanifested_mutation(tmp_path: Path) -> None:
    corrupted = tmp_path / "dataset"
    shutil.copytree(FIXTURE, corrupted)
    episode_path = corrupted / "panthera-episode.json"
    episode_path.write_text(episode_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_root(corrupted, allow_failed_tasks=False)
