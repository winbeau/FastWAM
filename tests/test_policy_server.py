from __future__ import annotations

import base64
import threading
import time
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from fastwam.policy.server import (
    PolicyBusyError,
    PolicyEngine,
    PolicyPrediction,
    PolicyRequestError,
)

HASH = "f" * 64


def encoded_image() -> str:
    buffer = BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def payload():
    return {
        "request_id": "request-1",
        "session_id": "session-1",
        "task_id": "pick-red",
        "canonical_prompt": "pick up the red block",
        "observation_sequence": 42,
        "observation_sampled_monotonic_ns": 100,
        "state_stream_instance_id": "stream-1",
        "deadline_pi_monotonic_ns": 200,
        "state_position": [0.0] * 7,
        "overhead_rgb_jpeg_base64": encoded_image(),
        "wrist_rgb_jpeg_base64": encoded_image(),
        "overhead_capture_monotonic_ns": 90,
        "wrist_capture_monotonic_ns": 91,
    }


class Model:
    def infer(self, request):
        assert request.overhead_rgb.size == (4, 3)
        return PolicyPrediction(np.asarray([[0.01] * 7, [0.02] * 7]))


def engine(model=None):
    return PolicyEngine(
        model=Model() if model is None else model,
        task_registry={"pick-red": "pick up the red block"},
        checkpoint_sha256=HASH,
        stats_sha256=HASH,
        schema_sha256=HASH,
    )


def test_policy_engine_echoes_pi_identity_and_generates_30hz_offsets():
    response = engine().infer_mapping(payload())
    assert response["request_id"] == "request-1"
    assert response["observation_sequence"] == 42
    assert response["deadline_pi_monotonic_ns"] == 200
    assert response["step_offsets_ns"] == [33_333_333, 66_666_667]
    assert np.asarray(response["waypoint_positions"]).shape == (2, 7)
    assert response["server_elapsed_ns"] >= 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_id", "unknown", "not registered"),
        ("canonical_prompt", "wrong", "does not match"),
        ("state_position", [0.0] * 6, "seven finite"),
        ("state_position", {"bad": "value"}, "seven finite"),
        ("overhead_rgb_jpeg_base64", "bad", "valid base64"),
    ],
)
def test_policy_engine_rejects_unknown_or_malformed_requests(field, value, message):
    request = payload()
    request[field] = value
    with pytest.raises(PolicyRequestError, match=message):
        engine().infer_mapping(request)


def test_policy_engine_rejects_bad_model_output():
    class BadModel:
        def infer(self, request):
            return PolicyPrediction(np.asarray([[float("nan")] * 7]))

    with pytest.raises(RuntimeError, match="NaN or Inf"):
        engine(BadModel()).infer_mapping(payload())


def test_policy_engine_drops_concurrent_request_instead_of_queueing():
    started = threading.Event()
    release = threading.Event()

    class BlockingModel:
        def infer(self, request):
            started.set()
            release.wait(timeout=2)
            return PolicyPrediction(np.asarray([[0.0] * 7]))

    service = engine(BlockingModel())
    error = []

    def run_first():
        try:
            service.infer_mapping(payload())
        except Exception as exc:  # pragma: no cover - assertion below reports it
            error.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    assert started.wait(timeout=1)
    with pytest.raises(PolicyBusyError, match="not queued"):
        service.infer_mapping(payload())
    release.set()
    thread.join(timeout=2)
    assert not error
    assert not thread.is_alive()
