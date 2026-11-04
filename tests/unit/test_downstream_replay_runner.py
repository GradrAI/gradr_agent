import json
import sys

from eval import downstream_replay_runner as runner


def _prediction(script_id="3"):
    return {
        "script_id": script_id,
        "lecture_mode": "summary",
        "ai_total_score": 45.0,
        "max_score": 60.0,
        "selected_question_ids": ["q1", "q2", "q3", "q4"],
        "raw_model_question_scores": {
            "q1": 18.0,
            "q2": 17.0,
            "q3": 8.5,
            "q4": 0.0,
            "q5": 0.0,
        },
        "normalized_question_scores": {
            "q1": 15.0,
            "q2": 15.0,
            "q3": 7.5,
            "q4": 0.0,
            "q5": 0.0,
        },
        "corrected_raw_question_max_scores": {
            "q1": 18.0,
            "q2": 17.0,
            "q3": 17.0,
            "q4": 17.0,
            "q5": 24.0,
        },
        "mean_confidence": 0.91,
        "min_confidence": 0.9,
        "low_confidence_count": 0,
        "validation_errors": [],
        "transcription_warnings": [],
        "extraction_transcription_warning_count": 0,
        "referee_status": "COMPLETED",
        "graded_questions": [
            {
                "question_id": "q1",
                "score": 18.0,
                "max_score": 18.0,
                "model_confidence": 0.9,
                "source_priority": "corrected_rubric",
                "justification": "visible answer evidence",
                "feedback": "review definitions",
            }
        ],
    }


def test_dry_run_writes_manifest_stage_io_metrics_and_no_model_calls(tmp_path):
    loaded_before = set(sys.modules)
    source_run = tmp_path / "source"
    source_run.mkdir()
    (source_run / "predictions.lecture.jsonl").write_text(json.dumps(_prediction()) + "\n")
    out_dir = tmp_path / "out"

    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--source-run",
            str(source_run),
            "--rubric-json",
            "eval/rubric_corrected_intro_python.json",
            "--out-dir",
            str(out_dir),
            "--script-ids",
            "3",
            "--conditions",
            "observed,controlled_fault",
            "--dry-run",
        ]
    )

    assert runner.run_experiment(args) == 0

    manifest = json.loads((out_dir / "manifest.json").read_text())
    summary = json.loads((out_dir / "run_summary.json").read_text())
    metrics = json.loads((out_dir / "metrics.json").read_text())
    rows = [json.loads(line) for line in (out_dir / "case_results.jsonl").read_text().splitlines()]
    stage_rows = [json.loads(line) for line in (out_dir / "stage_io.jsonl").read_text().splitlines()]

    assert manifest["dry_run"] is True
    assert manifest["variant_names"] == ["observed", "controlled_fault"]
    assert summary["case_count"] == 2
    assert metrics["case_count"] == 2
    assert {row["variant_name"] for row in rows} == {"observed", "controlled_fault"}
    assert any(row["stage"] == "review_gate" for row in stage_rows)
    assert all(row["model_call_attempted"] is False for row in stage_rows)
    assert (out_dir / "planned_actions.json").exists()
    loaded_after = set(sys.modules) - loaded_before
    assert "app.toolsets" not in loaded_after
    assert "app.agents.pbt_grading_pipeline" not in loaded_after
