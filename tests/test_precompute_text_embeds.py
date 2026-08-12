from __future__ import annotations

from pathlib import Path

from fastwam.datasets.lerobot.tasks import read_unique_tasks

FIXTURE = Path(__file__).parent / "fixtures/panthera_lerobot_v3_minimal"


def test_v3_tasks_parquet_is_discovered_for_text_cache() -> None:
    tasks, total_rows = read_unique_tasks([str(FIXTURE)])
    assert total_rows == 1
    assert tasks == ["Move the red block from the start area to the target area."]
