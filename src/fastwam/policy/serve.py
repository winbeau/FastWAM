"""Runnable action-student HTTP policy server."""

from __future__ import annotations

import argparse
import pickle
import signal
import threading
from pathlib import Path
from types import FrameType

import torch

from fastwam.distillation import ActionStudentServingManifest, DirectBCStudent, TinyVisionEncoder

from .action_student import ActionStudentPolicyModel, ActionStudentPreprocessor
from .server import PolicyEngine, PolicyHTTPServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve one frozen FastWAM action student")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--task-registry", type=Path, required=True)
    parser.add_argument("--text-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-body-bytes", type=int, default=16 << 20)
    return parser


def build_engine(args: argparse.Namespace) -> PolicyEngine:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    manifest = ActionStudentServingManifest.read(args.deployment_manifest)
    if manifest.vision_encoder_id != TinyVisionEncoder.encoder_id:
        raise ValueError(
            "this entry point only knows the built-in tiny vision encoder; "
            f"unsupported encoder {manifest.vision_encoder_id!r}"
        )
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(f"malformed or truncated action student checkpoint: {exc}") from exc
    config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(config, dict) or not isinstance(config.get("vision_dim"), int):
        raise ValueError("checkpoint does not contain a valid DirectBC config")
    encoder = TinyVisionEncoder(config["vision_dim"])
    model = DirectBCStudent.load_checkpoint(
        args.checkpoint,
        vision_encoder=encoder,
        deployment_manifest_path=args.deployment_manifest,
        stats_path=args.stats,
        schema_path=args.schema,
        task_registry_path=args.task_registry,
        map_location="cpu",
    )
    preprocessor = ActionStudentPreprocessor(
        stats_path=args.stats,
        axes=manifest.axes,
        camera_order=manifest.camera_order,
        image_sizes=manifest.image_sizes,
    )
    adapter = ActionStudentPolicyModel(
        model=model,
        text_cache_dir=args.text_cache_dir,
        preprocessor=preprocessor,
        serving_manifest=manifest,
        checkpoint_path=args.checkpoint,
        deployment_manifest_path=args.deployment_manifest,
        schema_path=args.schema,
        task_registry_path=args.task_registry,
        device=args.device,
    )
    adapter.preflight_text_cache()
    return PolicyEngine(
        model=adapter,
        task_registry=dict(adapter.task_registry),
        checkpoint_sha256=manifest.checkpoint_sha256,
        stats_sha256=manifest.stats_sha256,
        schema_sha256=manifest.schema_sha256,
        action_hz=manifest.action_hz,
        max_waypoints=manifest.action_horizon,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.max_body_bytes <= 0:
        raise SystemExit("--max-body-bytes must be positive")
    server = PolicyHTTPServer((args.bind, args.port), build_engine(args), max_body_bytes=args.max_body_bytes)

    def stop(_signum: int, _frame: FrameType | None) -> None:
        # BaseServer.shutdown must run away from serve_forever's thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    print(f"FastWAM policy server listening on http://{args.bind}:{server.server_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
