from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.transforms.image import fastwam_validation_image_transforms
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.distillation import (
    ActionStudentServingManifest,
    DirectBCConfig,
    DirectBCStudent,
    TeacherCacheManifest,
    TeacherTarget,
    action_distillation_loss,
    benchmark_latency,
    canonical_sha256,
    offline_action_metrics,
    read_teacher_cache,
    sha256_file,
    stable_sample_id,
    write_teacher_cache,
)
from fastwam.policy import ActionStudentPolicyModel, ActionStudentPreprocessor
from fastwam.policy.action_student import IMAGE_PREPROCESSING, PANTHERA_AXES
from fastwam.policy.server import PolicyRequest

HASH = "a" * 64
FIXTURE_STATS = "artifacts/panthera-fixture/dataset_stats.json"
FIXTURE_SCHEMA = "tests/fixtures/panthera_lerobot_v3_minimal/panthera-schema.json"


class RecordingEncoder(nn.Module):
    output_dim = 4
    encoder_id = "recording-v1"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, value):
        return value.mean((2, 3))[:, :1].repeat(1, 4) * self.weight


def inputs(batch=2):
    return (
        torch.rand(batch, 3, 8, 8),
        torch.rand(batch, 3, 8, 8),
        torch.rand(batch, 7),
        torch.rand(batch, 5, 6),
        torch.tensor([[1, 1, 0, 0, 0]] * batch, dtype=torch.bool),
    )


def serving_manifest(**changes):
    data = dict(
        checkpoint_sha256=None,
        stats_sha256=HASH,
        schema_sha256=HASH,
        dataset_version="v1",
        dataset_lineage_sha256=HASH,
        action_semantics="next_absolute_position_waypoint_q_t_plus_1_30hz",
        action_hz=30.0,
        axes=PANTHERA_AXES,
        camera_order=("overhead_rgb", "wrist_rgb"),
        image_sizes=((2, 2), (2, 2)),
        image_preprocessing=IMAGE_PREPROCESSING,
        text_encoder_id="encoder",
        context_length=2,
        prompt_template=DEFAULT_PROMPT,
        task_registry_sha256=HASH,
        vision_encoder_id="recording-v1",
    )
    data.update(changes)
    return ActionStudentServingManifest(**data)


def deployment_assets(tmp_path: Path):
    stats = tmp_path / "stats.json"
    schema = tmp_path / "schema.json"
    tasks = tmp_path / "tasks.json"
    stats.write_bytes(Path(FIXTURE_STATS).read_bytes())
    schema.write_bytes(Path(FIXTURE_SCHEMA).read_bytes())
    tasks.write_text(json.dumps({"task-id": "task"}, sort_keys=True))
    manifest = serving_manifest(
        stats_sha256=sha256_file(stats),
        schema_sha256=sha256_file(schema),
        task_registry_sha256=sha256_file(tasks),
    )
    return stats, schema, tasks, manifest


def save_student(model, tmp_path, manifest=None):
    stats, schema, tasks, default_manifest = deployment_assets(tmp_path)
    manifest = default_manifest if manifest is None else manifest
    checkpoint = tmp_path / "student.pt"
    deployment = tmp_path / "student.deployment.json"
    bound = model.save_checkpoint(
        checkpoint,
        serving_manifest=manifest,
        deployment_manifest_path=deployment,
        stats_path=stats,
        schema_path=schema,
        task_registry_path=tasks,
    )
    return checkpoint, deployment, stats, schema, tasks, bound


def test_student_shapes_gradients_and_strict_checkpoint(tmp_path, monkeypatch):
    encoder = RecordingEncoder()
    model = DirectBCStudent(encoder, DirectBCConfig(vision_dim=4, text_dim=6, hidden_dim=12))
    original_inputs = inputs()
    output = model(*original_inputs)
    assert output.shape == (2, 32, 7)
    output.sum().backward()
    assert encoder.weight.grad is None
    assert any(parameter.grad is not None for parameter in model.head.parameters())

    path, deployment, stats, schema, tasks, manifest = save_student(model, tmp_path)
    assert manifest.checkpoint_sha256 == sha256_file(path)
    loaded = DirectBCStudent.load_checkpoint(
        path, vision_encoder=RecordingEncoder(), deployment_manifest_path=deployment,
        stats_path=stats, schema_path=schema, task_registry_path=tasks,
    )
    with torch.inference_mode():
        torch.testing.assert_close(model(*original_inputs), loaded(*original_inputs))
    schema.write_text("{}")
    with pytest.raises(ValueError, match="schema file lineage"):
        DirectBCStudent.load_checkpoint(
            path, vision_encoder=RecordingEncoder(), deployment_manifest_path=deployment,
            stats_path=stats, schema_path=schema, task_registry_path=tasks,
        )
    schema.write_bytes(Path(FIXTURE_SCHEMA).read_bytes())
    with pytest.raises(ValueError, match="encoder"):
        wrong = RecordingEncoder()
        wrong.encoder_id = "wrong"
        DirectBCStudent.load_checkpoint(
            path, vision_encoder=wrong, deployment_manifest_path=deployment,
            stats_path=stats, schema_path=schema, task_registry_path=tasks,
        )

    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(path.read_bytes()[:20])
    with pytest.raises(ValueError, match="malformed or truncated"):
        bad_manifest = manifest.bind_checkpoint(truncated)
        bad_manifest.write(tmp_path / "truncated.deployment.json")
        DirectBCStudent.load_checkpoint(
            truncated, vision_encoder=RecordingEncoder(),
            deployment_manifest_path=tmp_path / "truncated.deployment.json",
            stats_path=stats, schema_path=schema, task_registry_path=tasks,
        )

    original = path.read_bytes()
    monkeypatch.setattr(
        "fastwam.distillation.student.os.replace",
        lambda *_: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        model.save_checkpoint(
            path, serving_manifest=manifest, deployment_manifest_path=deployment,
            stats_path=stats, schema_path=schema, task_registry_path=tasks,
        )
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_loss_validation_masking_confidence_and_numerical_gradient():
    prediction = torch.zeros(1, 3, 1, dtype=torch.double, requires_grad=True)
    ground_truth = torch.tensor([[[1.0], [3.0], [100.0]]], dtype=torch.double)
    padding = torch.tensor([[False, False, True]])
    teacher = torch.ones_like(prediction)
    variance = torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.double)
    loss = action_distillation_loss(
        prediction,
        ground_truth,
        padding,
        teacher=teacher,
        teacher_variance=variance,
        teacher_weight=0.5,
        smoothness_weight=0.1,
    )
    assert loss.ground_truth.item() == pytest.approx(5.0)
    loss.total.backward()
    assert prediction.grad[0, 2].item() == 0
    assert torch.autograd.gradcheck(
        lambda value: action_distillation_loss(value, ground_truth, padding).total,
        (torch.randn(1, 3, 1, dtype=torch.double, requires_grad=True),),
    )
    with pytest.raises(ValueError, match="boolean"):
        action_distillation_loss(prediction.detach(), ground_truth, padding.float())
    with pytest.raises(ValueError, match="all-padded"):
        action_distillation_loss(prediction.detach(), ground_truth, torch.ones_like(padding))
    with pytest.raises(ValueError, match="teacher is required"):
        action_distillation_loss(prediction.detach(), ground_truth, padding, teacher_weight=1)
    with pytest.raises(ValueError, match="no valid supervision"):
        action_distillation_loss(
            prediction.detach(),
            ground_truth,
            padding,
            teacher=teacher.detach(),
            teacher_weight=1,
            teacher_valid=torch.tensor([False]),
        )
    with pytest.raises(ValueError, match="ground_truth_weight must be positive"):
        action_distillation_loss(
            prediction.detach(), ground_truth, padding, teacher=teacher.detach(),
            ground_truth_weight=0, teacher_weight=1,
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        action_distillation_loss(
            prediction.detach(),
            ground_truth,
            padding,
            teacher=teacher.detach(),
            teacher_variance=torch.full_like(teacher, -1),
            teacher_weight=1,
        )


def cache_manifest(**changes):
    data = dict(
        dataset_version="v1",
        dataset_lineage_sha256=HASH,
        teacher_checkpoint_sha256="b" * 64,
        stats_sha256=HASH,
        schema_sha256=HASH,
        inference_seeds=(1, 2),
        inference_steps=20,
        sampler_config_sha256=HASH,
        ensemble_count=2,
        variance_definition="population variance across ensemble",
        variance_space="normalized action",
        generation_code_sha256=HASH,
        generation_config_sha256=HASH,
    )
    data.update(changes)
    return TeacherCacheManifest(**data)


def target(sample_index=0, *, task="task"):
    physical = np.zeros((32, 7), np.float32)
    content = canonical_sha256({"frame": sample_index})
    return TeacherTarget(
        sample_id=stable_sample_id(HASH, "episode", sample_index, content),
        canonical_task=task,
        normalized=physical * 2,
        physical=physical,
        variance=np.zeros_like(physical),
        source_episode_id="episode",
        source_frame_index=sample_index,
        source_content_sha256=content,
    )


def test_teacher_cache_digest_lineage_duplicates_and_atomic_behavior(tmp_path, monkeypatch):
    transform = lambda value: value * 2
    path = tmp_path / "cache.json"
    write_teacher_cache(path, cache_manifest(), [target()], physical_to_normalized=transform)
    loaded = read_teacher_cache(path, expected=cache_manifest(), physical_to_normalized=transform)
    assert next(iter(loaded.values())).physical.shape == (32, 7)
    with pytest.raises(ValueError, match="duplicate"):
        write_teacher_cache(
            path, cache_manifest(), [target(), target()], physical_to_normalized=transform
        )
    with pytest.raises(ValueError, match="lineage"):
        read_teacher_cache(
            path,
            expected=cache_manifest(dataset_version="v2"),
            physical_to_normalized=transform,
        )
    payload = json.loads(path.read_text())
    payload["targets"][0]["canonical_task"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content digest"):
        read_teacher_cache(path, expected=cache_manifest(), physical_to_normalized=transform)

    write_teacher_cache(path, cache_manifest(), [target()], physical_to_normalized=transform)
    original = path.read_text()
    monkeypatch.setattr(
        "fastwam.distillation.cache.os.replace",
        lambda *_: (_ for _ in ()).throw(OSError("fail")),
    )
    with pytest.raises(OSError):
        write_teacher_cache(path, cache_manifest(), [target(1)], physical_to_normalized=transform)
    assert path.read_text() == original
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_teacher_cache_rejects_identity_variance_and_transform_mismatch(tmp_path):
    transform = lambda value: value * 2
    bad_id = target()
    bad_id = TeacherTarget(**{**bad_id.__dict__, "sample_id": "c" * 64})
    with pytest.raises(ValueError, match="stable SHA"):
        write_teacher_cache(tmp_path / "a", cache_manifest(), [bad_id], physical_to_normalized=transform)
    with pytest.raises(ValueError, match="inconsistent"):
        write_teacher_cache(
            tmp_path / "b",
            cache_manifest(),
            [TeacherTarget(**{**target().__dict__, "normalized": np.ones((32, 7))})],
            physical_to_normalized=transform,
        )
    single = cache_manifest(
        inference_seeds=(1,),
        ensemble_count=1,
        variance_definition=None,
        variance_space=None,
    )
    with pytest.raises(ValueError, match="only valid"):
        write_teacher_cache(tmp_path / "c", single, [target()], physical_to_normalized=transform)


def test_metrics_and_latency_are_padding_aware_and_labeled():
    prediction = torch.tensor([[[1.0, 2.0], [2.0, 4.0], [999.0, 999.0]]])
    target_value = torch.zeros_like(prediction)
    mask = torch.tensor([[False, False, True]])
    metrics = offline_action_metrics(prediction, target_value, mask)
    assert metrics["valid_steps"] == 2
    assert metrics["valid_elements"] == 4
    assert metrics["per_axis_mae"] == [1.5, 3.0]
    latency = benchmark_latency(
        lambda: prediction + 1,
        warmup=0,
        iterations=2,
        device="cpu",
        dtype="float32",
        scope="fixture",
        structural_only=True,
    )
    assert all(f"latency_ms_p{p}" in latency for p in (50, 90, 95, 99))
    assert latency["scope"] == "fixture" and latency["structural_only"] is True


def _processor(image_sizes=((2, 3), (2, 3))):
    shape_meta = {
        "images": [
            {"key": "overhead_rgb", "raw_shape": [3, 3, 4], "shape": [3, *image_sizes[0]]},
            {"key": "wrist_rgb", "raw_shape": [3, 3, 4], "shape": [3, *image_sizes[1]]},
        ],
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 7, "shape": 7}],
    }
    return FastWAMProcessor(
        shape_meta=shape_meta,
        num_obs_steps=1,
        num_output_cameras=2,
        action_output_dim=7,
        proprio_output_dim=7,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms={
            key: fastwam_validation_image_transforms(size)
            for key, size in zip(("overhead_rgb", "wrist_rgb"), image_sizes, strict=True)
        },
        val_transforms={
            key: fastwam_validation_image_transforms(size)
            for key, size in zip(("overhead_rgb", "wrist_rgb"), image_sizes, strict=True)
        },
    ).eval()


def test_preprocessor_equivalent_to_fastwam_normalizer_real_fixture():
    preprocessor = ActionStudentPreprocessor(
        stats_path=FIXTURE_STATS, image_sizes=((2, 3), (2, 3))
    )
    processor = _processor()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(FIXTURE_STATS))
    state = torch.tensor([0.02, 0.21, 0.292, -0.094, 0.05, -0.02, 0.44])
    action = torch.tensor([[0.03, 0.215, 0.288, -0.091, 0.05, -0.02, 0.46]])
    normalized = processor.normalizer.forward(
        {"state": {"default": state.clone()}, "action": {"default": action.clone()}}
    )
    torch.testing.assert_close(
        preprocessor.normalize_state(state), normalized["state"]["default"]
    )
    torch.testing.assert_close(
        preprocessor.normalize_action(action), normalized["action"]["default"]
    )
    torch.testing.assert_close(preprocessor.denormalize_action(normalized["action"]["default"]), action)
    # Fixture axes 5/6 are constant; exact FastWAM semantics map their observed value to zero.
    assert preprocessor.normalize_state(state)[4:6].tolist() == [0.0, 0.0]
    overhead_image = Image.fromarray(np.arange(36, dtype=np.uint8).reshape(3, 4, 3), "RGB")
    wrist_image = Image.fromarray(np.arange(36, dtype=np.uint8).reshape(3, 4, 3) + 100, "RGB")
    overhead, wrist = preprocessor.process_images(overhead_image, wrist_image)
    training = processor.preprocess({
        "task": "task", "image_is_pad": torch.tensor([False]),
        "images": {
            "overhead_rgb": torch.from_numpy(np.asarray(overhead_image).copy()).permute(2, 0, 1).unsqueeze(0),
            "wrist_rgb": torch.from_numpy(np.asarray(wrist_image).copy()).permute(2, 0, 1).unsqueeze(0),
        },
        "state": {"default": state.unsqueeze(0)}, "state_is_pad": torch.tensor([False]), "idx": 0,
    })
    assert overhead.shape == (3, 2, 3) and wrist.shape == (3, 2, 3)
    torch.testing.assert_close(overhead, training["pixel_values"][0, 0], rtol=0, atol=0)
    torch.testing.assert_close(wrist, training["pixel_values"][1, 0], rtol=0, atol=0)
    assert overhead.mean() < wrist.mean()


def test_policy_exact_text_filename_camera_order_and_manifest_validation(tmp_path):
    model = DirectBCStudent(RecordingEncoder(), DirectBCConfig(vision_dim=4, text_dim=6, hidden_dim=12))
    stats_path, schema_path, tasks_path, base_manifest = deployment_assets(tmp_path)
    base_manifest = ActionStudentServingManifest.from_dict({
        **base_manifest.to_dict(), "image_sizes": [[2, 2], [2, 2]]
    })
    checkpoint = tmp_path / "student.pt"
    deployment = tmp_path / "student.deployment.json"
    manifest = model.save_checkpoint(
        checkpoint, serving_manifest=base_manifest, deployment_manifest_path=deployment,
        stats_path=stats_path, schema_path=schema_path, task_registry_path=tasks_path,
    )
    model = DirectBCStudent.load_checkpoint(
        checkpoint, vision_encoder=RecordingEncoder(), deployment_manifest_path=deployment,
        stats_path=stats_path, schema_path=schema_path, task_registry_path=tasks_path,
    )
    preprocessor = ActionStudentPreprocessor(stats_path=stats_path, image_sizes=((2, 2), (2, 2)))
    task = "task"
    prompt = DEFAULT_PROMPT.format(task=task)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    torch.save(
        {"context": torch.ones(2, 6), "mask": torch.tensor([True, True])},
        tmp_path / f"{digest}.t5_len2.encoder.pt",
    )

    original_forward = model.forward
    def checked_forward(overhead, wrist, state, context, mask):
        assert overhead[:, 0].mean() < wrist[:, 0].mean()
        assert state.shape == (1, 7)
        return original_forward(overhead, wrist, state, context, mask)
    model.forward = checked_forward

    adapter = ActionStudentPolicyModel(
        model=model,
        text_cache_dir=tmp_path,
        preprocessor=preprocessor,
        serving_manifest=manifest,
        checkpoint_path=checkpoint,
        deployment_manifest_path=deployment,
        schema_path=schema_path,
        task_registry_path=tasks_path,
    )
    request = PolicyRequest(
        "r",
        "s",
        "task-id",
        task,
        1,
        1,
        "stream",
        2,
        np.array([0.02, 0.21, 0.292, -0.094, 0.05, -0.02, 0.44]),
        Image.new("RGB", (2, 2), (10, 0, 0)),
        Image.new("RGB", (2, 2), (20, 0, 0)),
        1,
        1,
    )
    prediction = adapter.infer(request)
    assert prediction.waypoint_positions.shape == (32, 7)
    wrong_task = PolicyRequest(
        "r-2",
        "s",
        "other-task-id",
        task,
        2,
        2,
        "stream",
        3,
        request.state_position,
        request.overhead_rgb,
        request.wrist_rgb,
        2,
        2,
    )
    with pytest.raises(ValueError, match="task id and canonical task"):
        adapter.infer(wrong_task)
    with pytest.raises(ValueError, match="checkpoint file lineage"):
        tampered = tmp_path / "tampered.pt"
        tampered.write_bytes(checkpoint.read_bytes() + b"x")
        ActionStudentPolicyModel(
            model=model, text_cache_dir=tmp_path, preprocessor=preprocessor,
            serving_manifest=manifest, checkpoint_path=tampered,
            deployment_manifest_path=deployment, schema_path=schema_path,
            task_registry_path=tasks_path,
        )


def test_detached_manifest_rejects_missing_hash_tampering_and_config_mismatches(tmp_path):
    model = DirectBCStudent(RecordingEncoder(), DirectBCConfig(vision_dim=4, text_dim=6, hidden_dim=12))
    checkpoint, deployment, stats, schema, tasks, manifest = save_student(model, tmp_path)
    payload = json.loads(deployment.read_text())
    payload["checkpoint_sha256"] = None
    deployment.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="requires checkpoint_sha256"):
        DirectBCStudent.load_checkpoint(
            checkpoint, vision_encoder=RecordingEncoder(), deployment_manifest_path=deployment,
            stats_path=stats, schema_path=schema, task_registry_path=tasks,
        )
    manifest.write(deployment)
    valid_stats = stats.read_bytes()
    stats.write_text("{}")
    bad_stats_manifest = ActionStudentServingManifest.from_dict({
        **manifest.to_dict(), "stats_sha256": sha256_file(stats)
    })
    bad_stats_manifest.write(deployment)
    with pytest.raises(ValueError, match="stats state content mismatch"):
        DirectBCStudent.load_checkpoint(
            checkpoint, vision_encoder=RecordingEncoder(), deployment_manifest_path=deployment,
            stats_path=stats, schema_path=schema, task_registry_path=tasks,
        )
    stats.write_bytes(valid_stats)
    manifest.write(deployment)
    tasks.write_text(json.dumps({"tampered": "task"}))
    with pytest.raises(ValueError, match="task registry file lineage"):
        DirectBCStudent.load_checkpoint(
            checkpoint, vision_encoder=RecordingEncoder(), deployment_manifest_path=deployment,
            stats_path=stats, schema_path=schema, task_registry_path=tasks,
        )
    tasks.write_text(json.dumps({"task-id": "task"}, sort_keys=True))

    checkpoint_payload = torch.load(checkpoint, weights_only=True)
    checkpoint_payload["config"]["action_horizon"] = 31
    torch.save(checkpoint_payload, checkpoint)
    bad = manifest.bind_checkpoint(checkpoint)
    bad.write(deployment)
    with pytest.raises(ValueError, match="geometry"):
        DirectBCStudent.load_checkpoint(
            checkpoint, vision_encoder=RecordingEncoder(), deployment_manifest_path=deployment,
            stats_path=stats, schema_path=schema, task_registry_path=tasks,
        )

    with pytest.raises(ValueError, match="horizon/dim/state"):
        model.save_checkpoint(
            tmp_path / "bad.pt",
            serving_manifest=serving_manifest(action_horizon=31),
            deployment_manifest_path=tmp_path / "bad.json",
            stats_path=stats, schema_path=schema, task_registry_path=tasks,
        )
