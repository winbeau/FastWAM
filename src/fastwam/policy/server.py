"""Latest-only HTTP policy engine.

The Pi owns freshness: its monotonic deadline is echoed verbatim and is never
compared with the server's unrelated monotonic clock.
"""

from __future__ import annotations

import base64
import binascii
import json
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any, Protocol

import numpy as np
from PIL import Image


class PolicyRequestError(ValueError):
    pass


class PolicyBusyError(RuntimeError):
    pass


def _string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise PolicyRequestError(f"{field} must be a non-empty string")
    return value


def _integer(payload: dict[str, Any], field: str, *, positive: bool = True) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyRequestError(f"{field} must be an integer")
    if positive and value <= 0:
        raise PolicyRequestError(f"{field} must be positive")
    return value


def _decode_rgb(payload: dict[str, Any], field: str) -> Image.Image:
    encoded = _string(payload, field)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PolicyRequestError(f"{field} is not valid base64") from exc
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise PolicyRequestError(f"{field} is not a decodable image") from exc


def _sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    request_id: str
    session_id: str
    task_id: str
    canonical_prompt: str
    observation_sequence: int
    observation_sampled_monotonic_ns: int
    state_stream_instance_id: str
    deadline_pi_monotonic_ns: int
    state_position: np.ndarray
    overhead_rgb: Image.Image
    wrist_rgb: Image.Image
    overhead_capture_monotonic_ns: int
    wrist_capture_monotonic_ns: int

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        task_registry: dict[str, str],
    ) -> "PolicyRequest":
        if not isinstance(payload, dict):
            raise PolicyRequestError("request body must be a JSON object")
        task_id = _string(payload, "task_id")
        prompt = _string(payload, "canonical_prompt")
        expected_prompt = task_registry.get(task_id)
        if expected_prompt is None:
            raise PolicyRequestError(f"task_id {task_id!r} is not registered")
        if prompt != expected_prompt:
            raise PolicyRequestError("canonical_prompt does not match the registered task")
        try:
            state = np.asarray(payload.get("state_position"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise PolicyRequestError("state_position must contain seven finite values") from exc
        if state.shape != (7,) or not np.isfinite(state).all():
            raise PolicyRequestError("state_position must contain seven finite values")
        return cls(
            request_id=_string(payload, "request_id"),
            session_id=_string(payload, "session_id"),
            task_id=task_id,
            canonical_prompt=prompt,
            observation_sequence=_integer(payload, "observation_sequence"),
            observation_sampled_monotonic_ns=_integer(
                payload,
                "observation_sampled_monotonic_ns",
            ),
            state_stream_instance_id=_string(payload, "state_stream_instance_id"),
            deadline_pi_monotonic_ns=_integer(payload, "deadline_pi_monotonic_ns"),
            state_position=state,
            overhead_rgb=_decode_rgb(payload, "overhead_rgb_jpeg_base64"),
            wrist_rgb=_decode_rgb(payload, "wrist_rgb_jpeg_base64"),
            overhead_capture_monotonic_ns=_integer(
                payload,
                "overhead_capture_monotonic_ns",
            ),
            wrist_capture_monotonic_ns=_integer(payload, "wrist_capture_monotonic_ns"),
        )


@dataclass(frozen=True, slots=True)
class PolicyPrediction:
    waypoint_positions: np.ndarray
    step_offsets_ns: np.ndarray | None = None


class PolicyModel(Protocol):
    def infer(self, request: PolicyRequest) -> PolicyPrediction: ...


class PolicyEngine:
    def __init__(
        self,
        *,
        model: PolicyModel,
        task_registry: dict[str, str],
        checkpoint_sha256: str,
        stats_sha256: str,
        schema_sha256: str,
        action_hz: float = 30.0,
        max_waypoints: int = 64,
    ) -> None:
        if action_hz <= 0 or max_waypoints <= 0:
            raise ValueError("action_hz and max_waypoints must be positive")
        self.model = model
        self.task_registry = dict(task_registry)
        self.checkpoint_sha256 = _sha256(checkpoint_sha256, "checkpoint_sha256")
        self.stats_sha256 = _sha256(stats_sha256, "stats_sha256")
        self.schema_sha256 = _sha256(schema_sha256, "schema_sha256")
        self.action_hz = float(action_hz)
        self.max_waypoints = int(max_waypoints)
        self._inference_lock = threading.Lock()

    def infer_mapping(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._inference_lock.acquire(blocking=False):
            raise PolicyBusyError("policy server is busy; stale request was not queued")
        started_ns = time.perf_counter_ns()
        try:
            request = PolicyRequest.from_mapping(payload, task_registry=self.task_registry)
            prediction = self.model.infer(request)
            positions = np.asarray(prediction.waypoint_positions, dtype=np.float64)
            if positions.ndim != 2 or positions.shape[1] != 7:
                raise RuntimeError(f"model returned action shape {positions.shape}, expected [T,7]")
            if not 1 <= len(positions) <= self.max_waypoints:
                raise RuntimeError(
                    f"model returned {len(positions)} waypoints; expected 1..{self.max_waypoints}"
                )
            if not np.isfinite(positions).all():
                raise RuntimeError("model returned NaN or Inf actions")
            if prediction.step_offsets_ns is None:
                offsets = np.rint(
                    np.arange(1, len(positions) + 1, dtype=np.float64)
                    * (1_000_000_000 / self.action_hz)
                ).astype(np.int64)
            else:
                offsets = np.asarray(prediction.step_offsets_ns, dtype=np.int64)
            if offsets.shape != (len(positions),) or offsets[0] <= 0 or np.any(np.diff(offsets) <= 0):
                raise RuntimeError("model step offsets must be positive and strictly increasing")
            elapsed_ns = time.perf_counter_ns() - started_ns
            return {
                "request_id": request.request_id,
                "session_id": request.session_id,
                "observation_sequence": request.observation_sequence,
                "observation_sampled_monotonic_ns": request.observation_sampled_monotonic_ns,
                "state_stream_instance_id": request.state_stream_instance_id,
                "deadline_pi_monotonic_ns": request.deadline_pi_monotonic_ns,
                "waypoint_positions": positions.tolist(),
                "step_offsets_ns": offsets.tolist(),
                "checkpoint_sha256": self.checkpoint_sha256,
                "stats_sha256": self.stats_sha256,
                "schema_sha256": self.schema_sha256,
                "server_elapsed_ns": elapsed_ns,
            }
        finally:
            self._inference_lock.release()


class PolicyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, engine: PolicyEngine, *, max_body_bytes: int = 16 << 20):
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.engine = engine
        self.max_body_bytes = max_body_bytes
        super().__init__(server_address, PolicyHTTPRequestHandler)


class PolicyHTTPRequestHandler(BaseHTTPRequestHandler):
    server: PolicyHTTPServer

    def do_POST(self) -> None:
        if self.path != "/v1/infer":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
            return
        if length <= 0 or length > self.server.max_body_bytes:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body size rejected"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            response = self.server.engine.infer_mapping(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, PolicyRequestError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except PolicyBusyError as exc:
            self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(exc)})
            return
        except Exception as exc:  # model failures are reported, never retried implicitly
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self._json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args) -> None:
        return

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
