# Action-only student/distillation scope

`fastwam.distillation` provides a two-camera DirectBC action student, strict padded
GT/teacher losses, padding-aware offline metrics, synchronized latency measurement, and a
versioned teacher-target cache. `ActionStudentPolicyModel` is deliberately independent of
Wan/MoT while using the same FastWAM global min/max `LinearNormalizer` scale/offset
semantics.

Production construction requires a detached `ActionStudentServingManifest`. The checkpoint
contains the same deployment contract with `checkpoint_sha256=null`; after the checkpoint is
atomically written, the detached manifest is generated with the actual checkpoint file SHA-256.
This avoids the impossible circular requirement for a file to contain its own final digest.
Save, load, and `ActionStudentPolicyModel` construction require and verify the checkpoint,
stats JSON, schema JSON, and task-registry JSON paths. Their bytes must match the detached
manifest, and schema/task-registry content plus DirectBC geometry and encoder identity are
validated before serving. Missing hashes, missing files, tampering, or mismatches fail closed.

Serving images use the exact deterministic training path: FastWAM `ToTensor` converts the raw
uint8 CHW batch to float32 `[0,1]`, followed by `torchvision.transforms.Resize(size=[H,W])`.
There is no separate PIL resize implementation.

The task registry is a JSON object mapping a stable `task_id` to canonical task text. Both
`PolicyEngine` and `ActionStudentPolicyModel` require the request's ID and text to match that
same entry. Text embeddings then use exactly:

```text
{sha256(DEFAULT_PROMPT.format(task=canonical_task_text))}.t5_len{context_len}.{encoder_id}.pt
```

The current teacher cache is canonical, whole-payload-digested JSON with atomic file and
directory fsync. JSON is retained only for structural fixtures and small experiments; it is not
scalable for a production corpus, which should use a sharded binary representation preserving
the same manifest and record validation contract.

Run the download-free CPU structural smoke with:

```bash
uv run --frozen python scripts/action_student_smoke.py \
  --output artifacts/action-student-smoke.json
```

The output is explicitly `structural_synthetic_smoke` with
`real_teacher_student_training: false`, and its tiny-iteration latency is labeled
`structural_only`. It validates fixture plumbing only. It is **not** evidence of real teacher
sampling, student training quality, production hardware latency, model quality, closed-loop
robot performance, or robot safety. Synthetic mode remains available only through explicitly
synthetic manifests/encoders; production serving must provide immutable real lineage assets.
