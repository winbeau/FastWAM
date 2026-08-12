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

## HTTP serving

`fastwam-policy-server` serves one detached action-student deployment. It defaults to
`127.0.0.1:8080`, validates the checkpoint, detached manifest, stats, schema, task registry,
model geometry, and vision encoder, then loads **every** registered text cache before binding
the socket. Startup fails closed when any asset or task embedding is missing or mismatched.
The built-in entry point currently accepts only `tiny-vision-fixture-v1`; production encoders
must add an explicit, reviewed constructor rather than silently falling back to the fixture
encoder.

```bash
uv run --frozen fastwam-policy-server \
  --checkpoint /srv/fastwam/action-student.pt \
  --deployment-manifest /srv/fastwam/action-student.deployment.json \
  --stats /srv/fastwam/dataset_stats.json \
  --schema /srv/fastwam/panthera-schema.json \
  --task-registry /srv/fastwam/task-registry.json \
  --text-cache-dir /srv/fastwam/text-cache \
  --device cuda
```

Read-only probes are `GET /healthz` and `GET /readyz`; inference is `POST /v1/infer`.
Requests are bounded, overlapping inference receives HTTP 429 rather than being queued, and
there is no retry or motion RPC in this process. A systemd template and environment example
are under `deploy/`; keep the loopback bind unless access is protected by an authenticated
tunnel or trusted private network.

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
