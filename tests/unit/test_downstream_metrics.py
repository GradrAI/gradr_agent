import json

from eval.downstream_metrics import compute_contract_metrics, write_tables


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_compute_contract_metrics_counts_detection_containment_and_false_blocks(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "case_results.jsonl",
        [
            {
                "script_id": "3",
                "variant_name": "controlled_fault",
                "expected_violation_codes": ["review_status_dropped"],
                "detected_violation_codes": ["review_status_dropped"],
                "expected_blocked_actions": ["student_facing_feedback"],
                "action_decisions": {
                    "student_facing_feedback": {"allowed": False},
                    "smartprep_generation": {"allowed": False},
                },
                "review_decision": {"status": "PENDING_REVIEW"},
                "model_call_attempt_count": 2,
                "model_call_failure_count": 0,
                "parse_failure_count": 0,
            },
            {
                "script_id": "4",
                "variant_name": "controlled_fault",
                "expected_violation_codes": ["review_status_dropped"],
                "detected_violation_codes": [],
                "expected_blocked_actions": ["student_facing_feedback"],
                "action_decisions": {
                    "student_facing_feedback": {"allowed": True},
                    "smartprep_generation": {"allowed": False},
                },
                "review_decision": {"status": "COMPLETED"},
                "model_call_attempt_count": 2,
                "model_call_failure_count": 1,
                "parse_failure_count": 1,
            },
            {
                "script_id": "5",
                "variant_name": "observed",
                "expected_violation_codes": [],
                "detected_violation_codes": [],
                "expected_blocked_actions": [],
                "action_decisions": {
                    "student_facing_feedback": {"allowed": True},
                    "smartprep_generation": {"allowed": True},
                },
                "review_decision": {"status": "COMPLETED"},
                "model_call_attempt_count": 0,
                "model_call_failure_count": 0,
                "parse_failure_count": 0,
            },
            {
                "script_id": "5",
                "variant_name": "negative_control",
                "expected_violation_codes": [],
                "detected_violation_codes": [],
                "expected_blocked_actions": [],
                "action_decisions": {
                    "student_facing_feedback": {"allowed": False},
                    "smartprep_generation": {"allowed": True},
                },
                "review_decision": {"status": "COMPLETED"},
                "model_call_attempt_count": 0,
                "model_call_failure_count": 0,
                "parse_failure_count": 0,
            },
        ],
    )

    metrics = compute_contract_metrics(run_dir)

    assert metrics["case_count"] == 4
    assert metrics["faulty_case_count"] == 2
    assert metrics["contract_violation_detection_rate"] == 0.5
    assert metrics["action_propagation_rate"] == 0.5
    assert metrics["gate_containment_rate"] == 0.5
    assert metrics["false_block_rate"] == 1.0
    assert metrics["review_demand_rate"] == 0.25
    assert metrics["model_call_failure_rate"] == 0.25
    assert metrics["parse_failure_rate"] == 0.25


def test_write_tables_emits_markdown_and_csv(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tables_dir = run_dir / "paper_tables"
    _write_jsonl(
        run_dir / "case_results.jsonl",
        [
            {
                "script_id": "3",
                "variant_name": "controlled_fault",
                "expected_violation_codes": ["review_status_dropped"],
                "detected_violation_codes": ["review_status_dropped"],
                "expected_blocked_actions": ["student_facing_feedback"],
                "action_decisions": {"student_facing_feedback": {"allowed": False}},
                "review_decision": {"status": "PENDING_REVIEW"},
            }
        ],
    )

    write_tables(run_dir, tables_dir)

    assert (tables_dir / "contract_metrics.csv").exists()
    assert (tables_dir / "contract_metrics.md").exists()
    assert "gate_containment_rate" in (tables_dir / "contract_metrics.md").read_text()
