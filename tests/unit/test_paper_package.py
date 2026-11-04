import json
from pathlib import Path

from eval.paper_package import build_numbers, generate_package


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _case(variant: str, *, detected=(), expected=(), status="COMPLETED", allowed=True):
    return {
        "script_id": "3",
        "condition": "lecture",
        "variant_name": variant,
        "expected_violation_codes": list(expected),
        "detected_violation_codes": list(detected),
        "expected_blocked_actions": ["student_facing_feedback"] if expected else [],
        "review_decision": {"status": status},
        "action_decisions": {
            "student_facing_feedback": {"allowed": allowed},
            "persist_completed_result": {"allowed": allowed},
        },
        "model_call_attempt_count": 1 if allowed else 0,
        "model_call_failure_count": 0,
        "parse_failure_count": 0,
    }


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    (protocol_dir / "protocol_hash.txt").write_text("abc123\n")
    _write_json(
        run_dir / "manifest.json",
        {
            "dry_run": False,
            "protocol_hash": "abc123",
            "protocol_dir": str(protocol_dir),
            "source_run": "eval/runs/source",
            "script_ids": ["3"],
            "variant_names": ["negative_control", "controlled_fault"],
            "model": "gemini-test",
        },
    )
    _write_json(run_dir / "run_summary.json", {"case_count": 2, "dry_run": False})
    _append_jsonl(
        run_dir / "case_results.jsonl",
        [
            _case("negative_control", allowed=True),
            _case(
                "controlled_fault",
                detected=("review_status_dropped",),
                expected=("review_status_dropped",),
                status="PENDING_REVIEW",
                allowed=False,
            ),
        ],
    )
    _append_jsonl(
        run_dir / "stage_io.jsonl",
        [
            {"stage": "review_gate", "status": "succeeded", "model_call_attempted": False},
            {"stage": "feedback_proposal", "status": "succeeded", "model_call_attempted": True},
        ],
    )
    (run_dir / "prompts").mkdir()
    (run_dir / "prompts" / "3.prompt.txt").write_text("prompt text")
    (run_dir / "raw_model_outputs").mkdir()
    (run_dir / "raw_model_outputs" / "3.raw.txt").write_text("{}")
    _write_json(run_dir / "stage_io" / "3.feedback_proposal.json", {"proposal": "ok"})
    return run_dir


def test_build_numbers_recomputes_metrics_and_counts_artifacts(tmp_path):
    run_dir = _run_dir(tmp_path)

    numbers = build_numbers(run_dir, "cain2027")

    assert numbers["target"] == "cain2027"
    assert numbers["metrics"]["case_count"] == 2
    assert numbers["metrics"]["gate_containment_rate"] == 1.0
    assert numbers["counts_by_variant"] == {"controlled_fault": 1, "negative_control": 1}
    assert numbers["artifact_counts"]["prompt_files"] == 1
    assert numbers["artifact_counts"]["raw_model_output_files"] == 1
    assert numbers["protocol"]["manifest_hash_matches_recorded"] is True
    assert numbers["run_checks"]["all_cases_accounted_for"] is True


def test_generate_package_writes_tables_checklist_prompts_and_claim_boundaries(tmp_path):
    run_dir = _run_dir(tmp_path)
    out_dir = tmp_path / "package"

    generate_package(run_dir, "cain2027", out_dir)

    numbers = json.loads((out_dir / "numbers.json").read_text())
    assert numbers["metrics"]["contract_violation_detection_rate"] == 1.0
    assert (out_dir / "tables" / "contract_metrics.csv").exists()
    assert (out_dir / "tables" / "variant_outcomes.md").exists()
    assert (out_dir / "protocol" / "protocol_hash.txt").read_text() == "abc123\n"
    assert (out_dir / "prompts" / "3.prompt.txt").read_text() == "prompt text"
    assert "No qualified educator annotation" in (out_dir / "claim_boundary.md").read_text()
    assert "AI Use Statement" in (out_dir / "checklist.md").read_text()
