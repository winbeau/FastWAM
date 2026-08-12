from __future__ import annotations

import base64
import hashlib
import http.client
import json
import threading
import time
from argparse import Namespace
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.distillation.smoke import run as run_action_student_smoke
from fastwam.policy.serve import build_engine
from fastwam.policy.server import (
    PolicyBusyError,
    PolicyEngine,
    PolicyHTTPServer,
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
        task_registry={
            "pick-red": "pick up the red block",
            "place-blue": "place the blue block in the target area",
        },
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


def test_policy_engine_rejects_cross_task_prompt_mismatch():
    request = payload()
    request["task_id"] = "place-blue"
    with pytest.raises(PolicyRequestError, match="does not match"):
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


def test_action_student_server_entrypoint_preflights_and_infers(tmp_path):
    output = tmp_path / "serve-smoke.json"
    run_action_student_smoke(output, steps=1)
    prompt = DEFAULT_PROMPT.format(task="synthetic")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache = tmp_path / f"{digest}.t5_len5.synthetic-t5.pt"
    torch.save(
        {"context": torch.ones(5, 6), "mask": torch.ones(5, dtype=torch.bool)},
        cache,
    )
    args = Namespace(
        checkpoint=output.with_suffix(".student.pt"),
        deployment_manifest=output.with_suffix(".deployment.json"),
        stats=output.with_suffix(".stats.json"),
        schema=output.with_suffix(".schema.json"),
        task_registry=output.with_suffix(".tasks.json"),
        text_cache_dir=tmp_path,
        device="cpu",
    )
    service = build_engine(args)
    response = service.infer_mapping(
        {
            **payload(),
            "task_id": "synthetic",
            "canonical_prompt": "synthetic",
        }
    )
    assert np.asarray(response["waypoint_positions"]).shape == (32, 7)
    cache.unlink()
    with pytest.raises(ValueError, match="invalid cached text context"):
        build_engine(args)


def test_policy_http_server_health_ready_and_infer_round_trip():
    server = PolicyHTTPServer(("127.0.0.1", 0), engine(), max_body_bytes=1 << 20)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        for path in ("/healthz", "/readyz"):
            connection.request("GET", path)
            response = connection.getresponse()
            data = json.loads(response.read())
            assert response.status == 200
            assert data["ready"] is True
            assert data["checkpoint_sha256"] == HASH

        body = json.dumps(payload()).encode()
        connection.request(
            "POST",
            "/v1/infer",
            body=body,
            headers={"Content-Type": "text/plain", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 415

        connection.request(
            "POST",
            "/v1/infer",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        data = json.loads(response.read())
        assert response.status == 200
        assert data["observation_sequence"] == 42
        assert data["waypoint_positions"] == [[0.01] * 7, [0.02] * 7]
        connection.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
