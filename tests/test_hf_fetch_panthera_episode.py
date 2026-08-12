from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.hf_fetch_panthera_episode import (
    HFEpisodeFetchError,
    download_bundle,
    extract_bundle,
    verify_bundle,
)


def _bundle(root: Path, *, episode_id: str = "episode-000001", kind: str = "episodes") -> Path:
    root.mkdir()
    source = root.parent / "source" / episode_id
    source.mkdir(parents=True)
    for name in (
        "COMPLETE",
        "episode.json",
        "samples.parquet",
        "sync_report.json",
        "timestamp_quality.json",
        "calibration.json",
    ):
        (source / name).write_bytes(name.encode())
    archive = root / f"{episode_id}.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(source, arcname=episode_id)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / f"{episode_id}.sha256").write_text(f"{digest}  {archive.name}\n")
    (root / f"{episode_id}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "layout": "panthera-hf-episode-v1",
                "kind": kind,
                "episode_id": episode_id,
                "archive": archive.name,
                "archive_bytes": archive.stat().st_size,
                "sha256": digest,
                "training_data": kind == "episodes",
            }
        )
    )
    return root


def test_verify_and_extract_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    manifest = verify_bundle(bundle, episode_id="episode-000001", kind="episodes")
    assert manifest["training_data"] is True

    output, extracted = extract_bundle(
        bundle,
        tmp_path / "output",
        episode_id="episode-000001",
        kind="episodes",
    )
    assert extracted == manifest
    assert (output / "COMPLETE").is_file()
    assert (output / "samples.parquet").is_file()


def test_verify_rejects_checksum_manifest_and_size_mismatch(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    archive = bundle / "episode-000001.tar"
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(HFEpisodeFetchError, match="checksum mismatch"):
        verify_bundle(bundle, episode_id="episode-000001", kind="episodes")

    bundle = _bundle(tmp_path / "second", episode_id="episode-000002")
    manifest_path = bundle / "episode-000002.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["archive_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(HFEpisodeFetchError, match="size"):
        verify_bundle(bundle, episode_id="episode-000002", kind="episodes")

    bundle = _bundle(tmp_path / "third", episode_id="episode-000003")
    checksum = bundle / "episode-000003.sha256"
    checksum.write_text(checksum.read_text().replace("episode-000003.tar", "/tmp/source.tar"))
    with pytest.raises(HFEpisodeFetchError, match="portable"):
        verify_bundle(bundle, episode_id="episode-000003", kind="episodes")


def test_download_bundle_pins_revision_and_exact_files(tmp_path: Path) -> None:
    log = tmp_path / "hf-log.json"
    fake_hf = tmp_path / "hf"
    fake_hf.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, shutil, sys\n"
        f"log = pathlib.Path({str(log)!r})\n"
        f"source = pathlib.Path({str(tmp_path / 'source-bundle')!r})\n"
        "args = sys.argv[1:]\n"
        "destination = pathlib.Path(args[args.index('--local-dir') + 1]) / 'staging' / 'smoke'\n"
        "destination.mkdir(parents=True, exist_ok=True)\n"
        "for path in source.iterdir(): shutil.copy2(path, destination / path.name)\n"
        "log.write_text(json.dumps(args))\n",
        encoding="utf-8",
    )
    fake_hf.chmod(0o755)
    _bundle(tmp_path / "source-bundle", episode_id="smoke-1", kind="smoke")

    bundle = download_bundle(
        tmp_path / "download",
        repo_id="winbeau/fastwam-lerobot",
        revision="d" * 40,
        episode_id="smoke-1",
        kind="smoke",
        hf_binary=str(fake_hf),
    )

    args = json.loads(log.read_text())
    assert args[:2] == ["download", "winbeau/fastwam-lerobot"]
    assert args[2:5] == [
        "staging/smoke/smoke-1.tar",
        "staging/smoke/smoke-1.sha256",
        "staging/smoke/smoke-1.json",
    ]
    assert args[args.index("--revision") + 1] == "d" * 40
    assert bundle.name == "smoke"


def test_extract_rejects_traversal_and_links(tmp_path: Path) -> None:
    for member, message, episode_id in (
        (tarfile.TarInfo("../escape"), "unsafe archive path", "traversal"),
        (tarfile.TarInfo("linked/bad"), "unsupported member", "linked"),
    ):
        bundle = tmp_path / episode_id
        bundle.mkdir()
        archive = bundle / f"{episode_id}.tar"
        if episode_id == "traversal":
            member.size = 1
            payload = io.BytesIO(b"x")
        else:
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/escape"
            payload = None
        with tarfile.open(archive, "w") as handle:
            handle.addfile(member, payload)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (bundle / f"{episode_id}.sha256").write_text(f"{digest}  {archive.name}\n")
        (bundle / f"{episode_id}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "layout": "panthera-hf-episode-v1",
                    "kind": "smoke",
                    "episode_id": episode_id,
                    "archive": archive.name,
                    "archive_bytes": archive.stat().st_size,
                    "sha256": digest,
                    "training_data": False,
                }
            )
        )
        with pytest.raises(HFEpisodeFetchError, match=message):
            extract_bundle(
                bundle,
                tmp_path / f"output-{episode_id}",
                episode_id=episode_id,
                kind="smoke",
            )
