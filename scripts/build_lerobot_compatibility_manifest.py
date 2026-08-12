from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

LEROBOT_VERSION = "0.4.4"
PYPI_JSON_SHA256 = "7d1d24e99f28300513288561476eb75d4070d1dd3ce7cabbf457395364408381"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, *, exclude: Iterable[str] = ()) -> str:
    excluded = set(exclude)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or relative.startswith(".venv/") or "/__pycache__/" in relative:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def sha256_paths(workspace: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(workspace).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def critical_requirements(payload: dict[str, Any]) -> list[str]:
    critical = (
        "accelerate",
        "av",
        "datasets",
        "huggingface-hub",
        "numpy",
        "rerun-sdk",
        "torch",
        "torchcodec",
        "torchvision",
        "transformers",
    )
    return [
        requirement
        for requirement in payload["info"].get("requires_dist") or []
        if requirement.lower().startswith(critical)
    ]


def build_manifest(workspace: Path, pypi_json: Path) -> dict[str, Any]:
    fastwam = workspace / "FastWAM"
    panthera = workspace / "Panthera-WAM"
    producer = panthera / "tools/lerobot-v3"
    staging_fixture = producer / "tests/fixtures/panthera_staging_minimal"
    consumer_fixture = fastwam / "tests/fixtures/panthera_lerobot_v3_minimal"
    package_manifest_path = consumer_fixture / "panthera-package-manifest.json"
    package_manifest = load_json(package_manifest_path)
    schema = load_json(consumer_fixture / "panthera-schema.json")
    pypi_payload = load_json(pypi_json)

    if pypi_payload["info"]["version"] != LEROBOT_VERSION:
        raise ValueError("PyPI metadata version mismatch")
    if sha256_file(pypi_json) != PYPI_JSON_SHA256:
        raise ValueError("PyPI metadata SHA-256 mismatch")
    producer_fixture_sha = sha256_tree(staging_fixture)
    if package_manifest["source_staging_sha256"] != producer_fixture_sha:
        raise ValueError("producer staging fixture hash does not match package manifest")
    dataset_content_sha = sha256_tree(
        consumer_fixture,
        exclude={"panthera-package-manifest.json"},
    )
    if package_manifest["dataset_content_sha256"] != dataset_content_sha:
        raise ValueError("consumer fixture content hash does not match package manifest")

    producer_sources = [
        producer / "pyproject.toml",
        producer / "uv.lock",
        producer / "src/panthera_lerobot_v3/schema.py",
        producer / "src/panthera_lerobot_v3/packager.py",
        producer / "src/panthera_lerobot_v3/fixture.py",
        producer / "src/panthera_lerobot_v3/cli.py",
    ]
    consumer_sources = [
        fastwam / "pyproject.toml",
        fastwam / "uv.lock",
        fastwam / "src/fastwam/datasets/lerobot/lerobot_v3_adapter.py",
        fastwam / "src/fastwam/datasets/lerobot/robot_video_dataset.py",
        fastwam / "src/fastwam/datasets/lerobot/processors/fastwam_processor.py",
        fastwam / "src/fastwam/datasets/lerobot/tasks.py",
        fastwam / "scripts/validate_panthera_dataset.py",
        fastwam / "configs/data/panthera_2cam_smoke.yaml",
        fastwam / "configs/data/panthera_2cam.yaml",
    ]
    collector_sources = [
        panthera / "proto/arm.proto",
        panthera / "proto/camera.proto",
        panthera / "armd/src/armd/hardware_loop.py",
        panthera / "armd/src/armd/state_tap.py",
        panthera / "armd/src/armd/camera/backend.py",
        panthera / "armd/src/armd/camera/service.py",
        *sorted((panthera / "armd/src/armd/collectord").glob("*.py")),
    ]

    return {
        "manifest_version": 1,
        "decision": {
            "exact_lerobot_version": LEROBOT_VERSION,
            "producer_environment": "isolated uv project, CPU-only, Linux x86_64",
            "consumer_environment": "FastWAM uv project, CUDA 12.8, direct frozen-v3 reader",
            "reason": (
                "LeRobot 0.4.x requires datasets>=4 and rerun-sdk/numpy>=2, while FastWAM and the "
                "Panthera hardware runtime retain incompatible dependency baselines. The producer is "
                "therefore isolated; the consumer reads the pinned v3 parquet/video contract without "
                "importing the official lerobot package."
            ),
        },
        "lerobot": {
            "package_version": LEROBOT_VERSION,
            "codebase_version": package_manifest["lerobot_codebase_version"],
            "pypi_json_sha256": PYPI_JSON_SHA256,
            "critical_requirements": critical_requirements(pypi_payload),
        },
        "locks": {
            "fastwam_uv_lock_sha256": sha256_file(fastwam / "uv.lock"),
            "producer_uv_lock_sha256": sha256_file(producer / "uv.lock"),
            "panthera_runtime_uv_lock_sha256": sha256_file(panthera / "uv.lock"),
        },
        "schema": {
            "schema_version": package_manifest["schema_version"],
            "schema_sha256": package_manifest["schema_sha256"],
            "action_semantics": package_manifest["action_semantics"],
            "camera_order": schema["camera_order"],
            "color_space": schema["color_space"],
            "depth_policy": schema["depth_policy"],
            "axes": schema["axes"],
            "staging_samples": "samples.parquet",
        },
        "producer": {
            "repo_commit": git_commit(panthera),
            "source_sha256": sha256_paths(workspace, producer_sources),
        },
        "consumer": {
            "repo_commit": git_commit(fastwam),
            "source_sha256": sha256_paths(workspace, consumer_sources),
        },
        "collector": {
            "repo_commit": git_commit(panthera),
            "source_sha256": sha256_paths(workspace, collector_sources),
            "state_rate_hz": 200,
            "canonical_rate_hz": 30,
            "loss_policy": "bounded rings with explicit DATA_LOSS and episode rejection",
        },
        "fixture": {
            "producer_staging_sha256": producer_fixture_sha,
            "consumer_dataset_sha256": sha256_tree(consumer_fixture),
            "consumer_dataset_content_sha256": dataset_content_sha,
            "frame_count": package_manifest["frame_count"],
            "source_panthera_commit": package_manifest["source_panthera_commit"],
            "source_calibration_sha256": package_manifest["source_calibration_sha256"],
            "producer_path": staging_fixture.relative_to(workspace).as_posix(),
            "consumer_path": consumer_fixture.relative_to(workspace).as_posix(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--pypi-json",
        type=Path,
        default=Path(f"/tmp/lerobot-pypi/lerobot-{LEROBOT_VERSION}.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "lerobot-compatibility.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args.workspace.resolve(), args.pypi_json.resolve())
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != serialized:
            raise SystemExit(f"compatibility manifest is stale: {args.output}")
        print(f"verified {args.output}")
        return
    args.output.write_text(serialized, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
