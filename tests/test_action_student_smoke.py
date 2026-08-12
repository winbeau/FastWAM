import json

from fastwam.distillation.smoke import run


def test_action_student_structural_smoke_end_to_end(tmp_path):
    output = tmp_path / "smoke.json"
    artifact = run(output, steps=3)
    assert artifact["artifact_type"] == "structural_synthetic_smoke"
    assert artifact["real_teacher_student_training"] is False
    assert artifact["teacher_cache_entries"] == 3
    assert artifact["latency"]["structural_only"] is True
    assert json.loads(output.read_text())["metrics"]["valid_steps"] == 89
