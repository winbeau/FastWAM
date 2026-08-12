"""Task metadata readers shared by LeRobot-v2/v3 preprocessing paths."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


def read_unique_tasks(dataset_dirs: list[str]) -> tuple[list[str], int]:
    tasks: list[str] = []
    seen: set[str] = set()
    total_rows = 0
    for dataset_dir in dataset_dirs:
        meta_dir = Path(dataset_dir) / "meta"
        parquet_path = meta_dir / "tasks.parquet"
        jsonl_path = meta_dir / "tasks.jsonl"
        if parquet_path.exists():
            records = pq.read_table(parquet_path).to_pylist()
        elif jsonl_path.exists():
            records = []
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line_index, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"Task row is not an object at {jsonl_path}:{line_index}")
                    records.append(record)
        else:
            raise FileNotFoundError(f"Missing tasks file: expected {parquet_path} or {jsonl_path}")

        for record in records:
            if "task" in record:
                task = record["task"]
            elif "__index_level_0__" in record:
                task = record["__index_level_0__"]
            else:
                strings = [
                    value for key, value in record.items() if key != "task_index" and isinstance(value, str)
                ]
                if len(strings) != 1:
                    raise KeyError(f"Cannot identify task text in {parquet_path}: {record}")
                task = strings[0]
            task = str(task)
            total_rows += 1
            if task not in seen:
                seen.add(task)
                tasks.append(task)
    return tasks, total_rows
