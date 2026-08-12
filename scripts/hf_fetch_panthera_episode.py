#!/usr/bin/env python3
"""Download, verify, and safely extract one Pi collectord episode from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_REPO_ID = "winbeau/fastwam-lerobot"


class HFEpisodeFetchError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(
    path: Path,
    *,
    episode_id: str,
    kind: str,
    allow_legacy_manifest: bool = False,
) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HFEpisodeFetchError(f"invalid episode manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise HFEpisodeFetchError("episode manifest must be a JSON object")
    expected = {
        "schema_version": 1,
        "episode_id": episode_id,
        "kind": kind,
        "archive": f"{episode_id}.tar",
        "training_data": kind == "episodes",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise HFEpisodeFetchError(
                f"episode manifest {field} mismatch: {manifest.get(field)!r} != {value!r}"
            )
    layout = manifest.get("layout")
    legacy_manifest = layout is None
    if legacy_manifest and not allow_legacy_manifest:
        raise HFEpisodeFetchError(
            "episode manifest layout is missing; pass --allow-legacy-manifest only for a verified v0 bundle"
        )
    if not legacy_manifest and layout != "panthera-hf-episode-v1":
        raise HFEpisodeFetchError(
            f"episode manifest layout mismatch: {layout!r} != 'panthera-hf-episode-v1'"
        )
    digest = manifest.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise HFEpisodeFetchError("episode manifest sha256 is invalid")
    archive_bytes = manifest.get("archive_bytes")
    if archive_bytes is not None and (
        isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int) or archive_bytes <= 0
    ):
        raise HFEpisodeFetchError("episode manifest archive_bytes is invalid")
    if archive_bytes is None and not legacy_manifest:
        raise HFEpisodeFetchError("episode manifest archive_bytes is invalid")
    manifest = dict(manifest)
    manifest["legacy_manifest"] = legacy_manifest
    return manifest


def verify_bundle(
    bundle_dir: str | Path,
    *,
    episode_id: str,
    kind: str,
    allow_legacy_manifest: bool = False,
) -> dict[str, Any]:
    directory = Path(bundle_dir).expanduser().resolve()
    archive = directory / f"{episode_id}.tar"
    checksum = directory / f"{episode_id}.sha256"
    manifest_path = directory / f"{episode_id}.json"
    for path in (archive, checksum, manifest_path):
        if not path.is_file() or path.is_symlink():
            raise HFEpisodeFetchError(f"bundle file is unavailable: {path}")
    manifest = _load_manifest(
        manifest_path,
        episode_id=episode_id,
        kind=kind,
        allow_legacy_manifest=allow_legacy_manifest,
    )
    try:
        fields = checksum.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise HFEpisodeFetchError(f"cannot read checksum file: {checksum}") from exc
    if len(fields) != 2 or fields[1] != archive.name:
        raise HFEpisodeFetchError("checksum file must contain a portable archive filename")
    expected = manifest["sha256"]
    if fields[0] != expected:
        raise HFEpisodeFetchError("checksum file and manifest disagree")
    actual = sha256_file(archive)
    if actual != expected:
        raise HFEpisodeFetchError(f"archive checksum mismatch: {actual} != {expected}")
    archive_bytes = manifest.get("archive_bytes")
    if archive_bytes is not None and archive_bytes != archive.stat().st_size:
        raise HFEpisodeFetchError("archive size does not match the manifest")
    if archive_bytes is None:
        manifest = {**manifest, "archive_bytes": archive.stat().st_size}
    return manifest


def _validate_tar_members(archive: tarfile.TarFile, *, episode_id: str) -> None:
    root = Path(episode_id)
    members = archive.getmembers()
    if not members:
        raise HFEpisodeFetchError("episode archive is empty")
    names: set[str] = set()
    total_file_bytes = 0
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise HFEpisodeFetchError(f"unsafe archive path: {member.name}")
        if not path.parts or path.parts[0] != root.name:
            raise HFEpisodeFetchError(f"archive member escapes the episode root: {member.name}")
        normalized = path.as_posix()
        if normalized in names:
            raise HFEpisodeFetchError(f"archive contains a duplicate member: {member.name}")
        names.add(normalized)
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise HFEpisodeFetchError(f"archive contains an unsupported member: {member.name}")
        if not member.isfile() and not member.isdir():
            raise HFEpisodeFetchError(f"archive contains a non-regular member: {member.name}")
        if member.isfile():
            total_file_bytes += member.size
    archive_size = Path(archive.name).stat().st_size
    if total_file_bytes > archive_size:
        raise HFEpisodeFetchError("archive expands beyond its uncompressed size bound")


def extract_bundle(
    bundle_dir: str | Path,
    output_root: str | Path,
    *,
    episode_id: str,
    kind: str,
    allow_legacy_manifest: bool = False,
) -> tuple[Path, dict[str, Any]]:
    manifest = verify_bundle(
        bundle_dir,
        episode_id=episode_id,
        kind=kind,
        allow_legacy_manifest=allow_legacy_manifest,
    )
    directory = Path(bundle_dir).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / episode_id
    if final_path.exists():
        raise HFEpisodeFetchError(f"episode output already exists: {final_path}")
    archive_path = directory / manifest["archive"]
    temporary = Path(tempfile.mkdtemp(prefix=f".{episode_id}.tmp-", dir=output))
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            _validate_tar_members(archive, episode_id=episode_id)
            # Every member is a validated regular file/directory below one fixed root.
            # Python 3.10 has no tarfile extraction-filter API, so extract only after
            # the complete member list passes the checks above.
            archive.extractall(temporary)
        extracted = temporary / episode_id
        for name in (
            "COMPLETE",
            "episode.json",
            "samples.parquet",
            "sync_report.json",
            "timestamp_quality.json",
            "calibration.json",
        ):
            if not (extracted / name).is_file():
                raise HFEpisodeFetchError(f"extracted episode is missing {name}")
        extracted.rename(final_path)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return final_path, manifest


def download_bundle(
    destination: str | Path,
    *,
    repo_id: str,
    revision: str,
    episode_id: str,
    kind: str,
    hf_binary: str = "hf",
    allow_legacy_manifest: bool = False,
) -> Path:
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise HFEpisodeFetchError("--revision must be a 40-character lowercase commit SHA")
    resolved_hf = shutil.which(hf_binary)
    if resolved_hf is None:
        raise HFEpisodeFetchError(f"Hugging Face CLI is not available: {hf_binary}")
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    remote = f"staging/{kind}"
    filenames = [
        f"{remote}/{episode_id}.tar",
        f"{remote}/{episode_id}.sha256",
        f"{remote}/{episode_id}.json",
    ]
    command = [
        resolved_hf,
        "download",
        repo_id,
        *filenames,
        "--repo-type",
        "dataset",
        "--revision",
        revision,
        "--local-dir",
        str(destination),
        "--quiet",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        combined = "\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        ).strip()
        raise HFEpisodeFetchError(
            f"Hugging Face download failed: {combined[-4000:] or completed.returncode}"
        )
    bundle = destination / remote
    verify_bundle(
        bundle,
        episode_id=episode_id,
        kind=kind,
        allow_legacy_manifest=allow_legacy_manifest,
    )
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download, verify, and safely extract one Panthera episode from Hugging Face"
    )
    parser.add_argument("episode_id")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--kind", choices=("episodes", "smoke"), default="episodes")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("/root/autodl-tmp/fastwam-lerobot/downloads"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/fastwam-lerobot/staging"),
    )
    parser.add_argument("--hf-binary", default="hf")
    parser.add_argument(
        "--allow-legacy-manifest",
        action="store_true",
        help="Accept the verified pre-v1 manifest that lacks layout/archive_bytes",
    )
    parser.add_argument("--bundle-dir", type=Path, help="Verify/extract an existing local bundle")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.bundle_dir is None:
            args.download_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f"panthera-hf-{args.episode_id}-",
                dir=args.download_dir,
            ) as temporary:
                bundle = download_bundle(
                    temporary,
                    repo_id=args.repo_id,
                    revision=args.revision,
                    episode_id=args.episode_id,
                    kind=args.kind,
                    hf_binary=args.hf_binary,
                    allow_legacy_manifest=args.allow_legacy_manifest,
                )
                output, manifest = extract_bundle(
                    bundle,
                    args.output_root,
                    episode_id=args.episode_id,
                    kind=args.kind,
                    allow_legacy_manifest=args.allow_legacy_manifest,
                )
        else:
            output, manifest = extract_bundle(
                args.bundle_dir,
                args.output_root,
                episode_id=args.episode_id,
                kind=args.kind,
                allow_legacy_manifest=args.allow_legacy_manifest,
            )
    except HFEpisodeFetchError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "status": "complete",
                "repo_id": args.repo_id,
                "revision": args.revision,
                "episode": str(output),
                "sha256": manifest["sha256"],
                "training_data": manifest["training_data"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
