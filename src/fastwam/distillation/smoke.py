"""CPU-only structural/synthetic smoke; this is not real teacher/student training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.distillation import (
    ActionStudentServingManifest,
    DirectBCConfig,
    DirectBCStudent,
    TeacherCacheManifest,
    TeacherTarget,
    TinyVisionEncoder,
    action_distillation_loss,
    benchmark_latency,
    canonical_sha256,
    offline_action_metrics,
    read_teacher_cache,
    sha256_file,
    stable_sample_id,
    write_teacher_cache,
)
from fastwam.policy.action_student import IMAGE_PREPROCESSING, PANTHERA_AXES

HASH = "a" * 64


def _serving_manifest(stats: Path, schema: Path, tasks: Path) -> ActionStudentServingManifest:
    return ActionStudentServingManifest(
        checkpoint_sha256=None,
        stats_sha256=sha256_file(stats),
        schema_sha256=sha256_file(schema),
        dataset_version="synthetic-v1",
        dataset_lineage_sha256=HASH,
        action_semantics="synthetic_absolute_waypoint",
        action_hz=30.0,
        axes=PANTHERA_AXES,
        camera_order=("overhead_rgb", "wrist_rgb"),
        image_sizes=((12, 12), (12, 12)),
        image_preprocessing=IMAGE_PREPROCESSING,
        text_encoder_id="synthetic-t5",
        context_length=5,
        prompt_template=DEFAULT_PROMPT,
        task_registry_sha256=sha256_file(tasks),
        vision_encoder_id=TinyVisionEncoder.encoder_id,
    )


def _cache_manifest() -> TeacherCacheManifest:
    return TeacherCacheManifest(
        dataset_version="synthetic-v1",
        dataset_lineage_sha256=HASH,
        teacher_checkpoint_sha256="b" * 64,
        stats_sha256=HASH,
        schema_sha256=HASH,
        inference_seeds=(7,),
        inference_steps=1,
        sampler_config_sha256=HASH,
        ensemble_count=1,
        variance_definition=None,
        variance_space=None,
        generation_code_sha256=HASH,
        generation_config_sha256=HASH,
    )


def run(output: str | Path, *, steps: int = 4) -> dict:
    torch.manual_seed(7)
    config = DirectBCConfig(vision_dim=8, text_dim=6, hidden_dim=24)
    model = DirectBCStudent(TinyVisionEncoder(8), config)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
    overhead = torch.rand(3, 3, 12, 12)
    wrist = torch.rand(3, 3, 12, 12)
    state = torch.randn(3, 7)
    context = torch.randn(3, 5, 6)
    text_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool
    )
    ground_truth = torch.randn(3, 32, 7) * 0.1
    padding = torch.zeros(3, 32, dtype=torch.bool)
    padding[1, 25:] = True
    initial = None
    for _ in range(steps):
        prediction = model(overhead, wrist, state, context, text_mask)
        loss = action_distillation_loss(
            prediction, ground_truth, padding, smoothness_weight=0.01
        )
        initial = float(loss.total) if initial is None else initial
        optimizer.zero_grad()
        loss.total.backward()
        optimizer.step()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = output.with_suffix(".teacher-cache.json")
    manifest = _cache_manifest()
    targets = []
    for index in range(3):
        content_digest = canonical_sha256({"synthetic_frame": index})
        targets.append(
            TeacherTarget(
                sample_id=stable_sample_id(HASH, "episode-0", index, content_digest),
                canonical_task="synthetic",
                normalized=ground_truth[index].numpy(),
                physical=ground_truth[index].numpy(),
                source_episode_id="episode-0",
                source_frame_index=index,
                source_content_sha256=content_digest,
            )
        )
    identity = lambda value: value
    write_teacher_cache(cache_path, manifest, targets, physical_to_normalized=identity)
    loaded = read_teacher_cache(
        cache_path, expected=manifest, physical_to_normalized=identity
    )
    teacher = torch.from_numpy(
        np.stack([loaded[target.sample_id].normalized for target in targets])
    )
    loss = action_distillation_loss(
        model(overhead, wrist, state, context, text_mask),
        ground_truth,
        padding,
        teacher=teacher,
        teacher_weight=0.5,
    )
    optimizer.zero_grad()
    loss.total.backward()
    optimizer.step()

    checkpoint = output.with_suffix(".student.pt")
    deployment = output.with_suffix(".deployment.json")
    stats = output.with_suffix(".stats.json")
    schema = output.with_suffix(".schema.json")
    tasks = output.with_suffix(".tasks.json")
    stats.write_text(json.dumps({
        kind: {"default": {"global_min": [-1.0] * 7, "global_max": [1.0] * 7}}
        for kind in ("state", "action")
    }))
    schema.write_text(json.dumps({
        "axes": list(PANTHERA_AXES), "camera_order": ["overhead_rgb", "wrist_rgb"],
        "action_semantics": "synthetic_absolute_waypoint", "fps": 30,
    }))
    tasks.write_text(json.dumps({"synthetic": "synthetic"}))
    serving_manifest = _serving_manifest(stats, schema, tasks)
    serving_manifest = model.save_checkpoint(
        checkpoint, serving_manifest=serving_manifest, deployment_manifest_path=deployment,
        stats_path=stats, schema_path=schema, task_registry_path=tasks, metadata={"smoke": True},
    )
    reloaded = DirectBCStudent.load_checkpoint(
        checkpoint, vision_encoder=TinyVisionEncoder(8), deployment_manifest_path=deployment,
        stats_path=stats, schema_path=schema, task_registry_path=tasks,
    ).eval()
    with torch.inference_mode():
        final_prediction = reloaded(overhead, wrist, state, context, text_mask)
    artifact = {
        "artifact_type": "structural_synthetic_smoke",
        "real_teacher_student_training": False,
        "seed": 7,
        "steps": steps,
        "initial_loss": initial,
        "final_loss": float(action_distillation_loss(final_prediction, ground_truth, padding).total),
        "metrics": offline_action_metrics(final_prediction, ground_truth, padding),
        "latency": benchmark_latency(
            lambda: reloaded(overhead, wrist, state, context, text_mask),
            warmup=1,
            iterations=3,
            device="cpu",
            dtype="float32",
            scope="tiny_fixture_model_forward_batch3",
            structural_only=True,
        ),
        "teacher_cache_entries": len(loaded),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/action-student-smoke.json")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.output, steps=args.steps), sort_keys=True))


if __name__ == "__main__":
    main()
