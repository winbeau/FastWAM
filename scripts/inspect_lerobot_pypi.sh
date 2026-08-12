#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LEROBOT_PYPI_OUTPUT_DIR:-/tmp/lerobot-pypi}"
if (( $# > 0 )); then
  VERSIONS=("$@")
else
  VERSIONS=(0.4.0 0.4.1 0.4.2 0.4.3 0.4.4)
fi

command -v curl >/dev/null 2>&1 || {
  echo "error: curl is required" >&2
  exit 127
}
command -v python3 >/dev/null 2>&1 || {
  echo "error: python3 is required" >&2
  exit 127
}

mkdir -p "$OUTPUT_DIR"

for version in "${VERSIONS[@]}"; do
  url="https://pypi.org/pypi/lerobot/${version}/json"
  destination="${OUTPUT_DIR}/lerobot-${version}.json"
  echo "fetching ${url}"
  curl --fail --silent --show-error --location --retry 3 \
    "$url" \
    --output "${destination}.tmp"
  mv "${destination}.tmp" "$destination"
done

python3 - "$OUTPUT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
critical_packages = {
    "accelerate",
    "av",
    "datasets",
    "huggingface-hub",
    "numpy",
    "pyarrow",
    "rerun-sdk",
    "torch",
    "torchcodec",
    "torchvision",
    "transformers",
}

summary = []
for path in sorted(output_dir.glob("lerobot-*.json")):
    raw = path.read_bytes()
    payload = json.loads(raw)
    info = payload["info"]
    requirements = []
    for requirement in info.get("requires_dist") or []:
        package = requirement.split(";", 1)[0].split("[", 1)[0]
        for separator in ("<", ">", "=", "!", "~", " "):
            package = package.split(separator, 1)[0]
        if package.strip().lower() in critical_packages:
            requirements.append(requirement)

    entry = {
        "version": info["version"],
        "requires_python": info.get("requires_python"),
        "requirements": requirements,
        "source_json": str(path),
        "source_json_sha256": hashlib.sha256(raw).hexdigest(),
    }
    summary.append(entry)

    print(f"\n=== lerobot {entry['version']} ===")
    print(f"requires_python: {entry['requires_python']}")
    for requirement in requirements:
        print(f"  {requirement}")
    print(f"json_sha256: {entry['source_json_sha256']}")

summary_path = output_dir / "lerobot-compatibility-inputs.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"\nsummary: {summary_path}")
PY
