"""CPU-only inspection of a FastWAM-Joint LIBERO checkpoint and its dataset stats.

Prints the SHA-256, payload layout (top-level keys, training step, expert parameter
counts, presence of action/proprio weights) and the stats' action/state dimensions and
min/max ranges. Exits non-zero when the payload or stats do not look like a LIBERO
Joint release (7-dim actions, 8-dim proprio, both video and action experts present).
The full key/shape match against FastWAMJoint is done by the adapter at load time.

    python scripts/casestudy/inspect_checkpoint.py --checkpoint model.pt --stats dataset_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--expected-sha256", default=None)
    args = ap.parse_args()

    import torch

    ckpt = Path(args.checkpoint)
    report: dict = {"checkpoint": str(ckpt), "size_bytes": ckpt.stat().st_size, "sha256": sha256(ckpt)}
    if args.expected_sha256:
        report["sha256_matches_expected"] = report["sha256"] == args.expected_sha256

    payload = torch.load(str(ckpt), map_location="cpu")
    report["payload_keys"] = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
    problems: list[str] = []
    mot = payload.get("mot") if isinstance(payload, dict) else None
    if not isinstance(mot, dict):
        problems.append("no `mot` state dict")
    else:
        prefixes = Counter(k.split(".")[1] if k.startswith("mixtures.") else k.split(".")[0] for k in mot)
        params = Counter()
        for k, v in mot.items():
            params[k.split(".")[1] if k.startswith("mixtures.") else k.split(".")[0]] += v.numel()
        report["mot_tensors_by_expert"] = dict(prefixes)
        report["mot_params_by_expert"] = {k: int(v) for k, v in params.items()}
        report["mot_dtypes"] = dict(Counter(str(v.dtype) for v in mot.values()))
        for expert in ("video", "action"):
            if expert not in prefixes:
                problems.append(f"no `{expert}` expert in mot")
        heads = {k: tuple(v.shape) for k, v in mot.items() if "action" in k and ("head" in k or "action_encoder" in k)}
        report["action_head_shapes"] = heads
    if "proprio_encoder" not in payload:
        problems.append("no `proprio_encoder`")
    else:
        report["proprio_encoder_shapes"] = {k: tuple(v.shape) for k, v in payload["proprio_encoder"].items()}
    for key in ("step", "epoch", "global_step"):
        if key in payload:
            report[key] = payload[key]

    stats = json.loads(Path(args.stats).read_text())
    report["stats_sha256"] = sha256(Path(args.stats))
    summary = {}
    for group in ("action", "state"):
        for name, block in (stats.get(group) or {}).items():
            entry = {}
            for field in ("min", "max", "q01", "q99", "mean", "std", "global_min", "global_max"):
                if field in block:
                    arr = block[field]
                    flat = arr if not (arr and isinstance(arr[0], list)) else arr[-1]
                    entry[field] = {"len": len(flat), "values": [round(float(x), 4) for x in flat]}
            summary[f"{group}.{name}"] = entry
    report["stats"] = summary
    act = summary.get("action.default", {}).get("global_min", {}).get("len")
    st = summary.get("state.default", {}).get("global_min", {}).get("len")
    if act != 7:
        problems.append(f"action.default global_min has length {act}, expected 7")
    if st != 8:
        problems.append(f"state.default global_min has length {st}, expected 8")
    report["problems"] = problems
    print(json.dumps(report, indent=2, default=str))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
