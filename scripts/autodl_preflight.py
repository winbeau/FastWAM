#!/usr/bin/env python3
"""Emit the bounded AutoDL preflight record used by P1."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> dict[str, Any]:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk(path: Path) -> dict[str, int | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def cgroup_cpu_quota() -> float | None:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        quota, period = cpu_max.read_text(encoding="utf-8").split()[:2]
        if quota != "max":
            return int(quota) / int(period)
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.is_file() and period_path.is_file():
        quota = int(quota_path.read_text(encoding="utf-8"))
        if quota > 0:
            return quota / int(period_path.read_text(encoding="utf-8"))
    return None


def writable(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix="fastwam-preflight-", delete=True):
            return True
    except OSError:
        return False


def main() -> int:
    output = Path(os.environ.get("FASTWAM_PREFLIGHT_OUTPUT", ROOT / "artifacts/autodl/preflight.json"))
    cache_paths = {
        name: Path(os.environ[name])
        for name in ("UV_CACHE_DIR", "HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME")
        if os.environ.get(name)
    }
    cpu_quota = cgroup_cpu_quota()
    affinity_cpus = len(os.sched_getaffinity(0))
    report: dict[str, Any] = {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "logical_cpus_host": os.cpu_count(),
        "logical_cpus_affinity": affinity_cpus,
        "logical_cpus_quota": cpu_quota,
        "logical_cpus_available": min(affinity_cpus, int(cpu_quota)) if cpu_quota else affinity_cpus,
        "uv": run("uv", "--version"),
        "nvcc": run("nvcc", "--version"),
        "nvidia_smi": run(
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ),
        "nvlink": run("nvidia-smi", "nvlink", "--status"),
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        },
        "tools": {name: shutil.which(name) for name in ("ffmpeg", "gcc", "git", "uv", "nvidia-smi", "nvcc")},
        "storage": {
            "project": disk(ROOT),
            "cache_paths": {
                name: {**disk(path), "writable": writable(path)} for name, path in cache_paths.items()
            },
        },
        "network_constraints": {
            "tailscale_binary": shutil.which("tailscale"),
            "tun_device_present": Path("/dev/net/tun").exists(),
            "closed_loop_eligible": False,
            "reason": "AutoDL has no approved route to Pi; P1 is offline training/data smoke only.",
        },
        "source": {
            "uv_lock_sha256": sha256(ROOT / "uv.lock"),
            "compatibility_manifest_sha256": sha256(ROOT / "lerobot-compatibility.json"),
        },
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        report["torch"]["device"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "free_memory_bytes": free_bytes,
            "driver_visible_total_memory_bytes": total_bytes,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
