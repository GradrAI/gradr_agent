import argparse
import csv
import json
from pathlib import Path

from eval.compute_metrics import compute_corrected_staged_metrics


def _write_score_sheet(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["serial", "q1", "q2", "q3", "q4", "q5", "sum", "total", "pages"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction(script_id: str, q_scores: dict[str, float], final_total: float) -> dict[str, object]:
    selected = ["q1", "q2", "q3", "q4"]
    normalized = {qid: (15.0 if qid in selected else 15.0) for qid in ("q1", "q2", "q3", "q4", "q5")}
    return {
        "script_id": script_id,
        "script_filename": f"{script_id}.pdf",
        "script_path": f"/tmp/{script_id}.pdf",
        "human_total_score": 60,
        "ai_total_score": final_total,
        "max_score": 60,
        "mean_confidence": 0.9,
        "low_confidence_count": 0,
        "referee_status": "COMPLETED",
        "selected_question_ids": selected,
        "unselected_question_ids": ["q5"],
        "raw_model_question_scores": q_scores,
        "model_reported_question_max_scores": {"q1": 18, "q2": 17, "q3": 17, "q4": 17, "q5": 24},
        "corrected_raw_question_max_scores": {"q1": 18, "q2": 17, "q3": 17, "q4": 17, "q5": 24},
        "bounded_raw_question_scores": q_scores,
        "normalized_question_scores": normalized,
        "raw_all_questions_total_score": sum(q_scores.values()),
        "raw_attempted_total_score": sum(q_scores.values()),
        "raw_selected_total_score": q_scores["q1"] + q_scores["q2"] + q_scores["q3"] + q_scores["q4"],
        "raw_model_subpart_scores": {qid: {} for qid in q_scores},
        "graded_questions": [
            {
                "question_id": qid,
                "attempted": True,
                "score": score,
                "max_score": {"q1": 18, "q2": 17, "q3": 17, "q4": 17, "q5": 24}[qid],
                "subpart_scores": [],
                "justification": f"raw {score}",
                "feedback": "feedback",
                "model_confidence": 0.9,
            }
            for qid, score in q_scores.items()
        ],
        "stage_io": {
            "grading_input_path": "stage_io/grading/no_lecture/inputs/1.json",
            "grading_output_path": "stage_io/grading/no_lecture/outputs/1.json",
        },
        "validation_errors": [],
        "transcription_warnings": [],
        "json_repair_used": False,
        "rubric_boundary_warning_count": 0,
        "extraction_transcription_warning_count": 0,
        "extraction_validation_error_count": 0,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _seed_stage_io(run_dir: Path) -> None:
    stages = [
        ("1", None, "extraction"),
        ("1", "no_lecture", "grading"),
        ("1", "lecture", "grading"),
        ("1", "no_lecture", "finalization"),
        ("1", "lecture", "finalization"),
        ("1", "no_lecture", "downstream_shadow"),
        ("1", "lecture", "downstream_shadow"),
    ]
    rows = []
    for script_id, condition, stage in stages:
        rows.append(
            {
                "script_id": script_id,
                "condition": condition,
                "stage": stage,
                "input_path": f"stage_io/{stage}/inputs/{script_id}.json",
                "output_path": f"stage_io/{stage}/outputs/{script_id}.json",
                "raw_output_path": None,
                "parsed_output_path": None,
                "prompt_path": None,
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": "2026-01-01T00:00:00Z",
                "duration_ms": 0,
                "status": "succeeded",
                "validation_errors": [],
                "warnings": [],
            }
        )
    (run_dir / "stage_io.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def _args(tmp_path: Path) -> argparse.Namespace:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    q_scores = {"q1": 18, "q2": 17, "q3": 17, "q4": 17, "q5": 24}
    _write_jsonl(run_dir / "predictions.no_lecture.jsonl", [_prediction("1", q_scores, 60)])
    _write_jsonl(run_dir / "predictions.lecture.jsonl", [_prediction("1", q_scores, 60)])
    (run_dir / "extractions.jsonl").write_text(json.dumps({"script_id": "1", "questions": [], "missing_or_blank_questions": []}) + "\n")
    _seed_stage_io(run_dir)
    original = tmp_path / "original.csv"
    grader2 = tmp_path / "grader2.csv"
    rows = [
        {"serial": serial, "q1": 18, "q2": 17, "q3": 17, "q4": 17, "q5": 24, "sum": 93, "total": 60, "pages": 1}
        for serial in range(1, 61)
    ]
    _write_score_sheet(original, rows)
    _write_score_sheet(grader2, rows)
    return argparse.Namespace(
        original_score_sheet=str(original),
        grader2_score_sheet=str(grader2),
        predictions_no_lecture=str(run_dir / "predictions.no_lecture.jsonl"),
        predictions_lecture=str(run_dir / "predictions.lecture.jsonl"),
        rubric_json="eval/rubric_corrected_intro_python.json",
        extractions=str(run_dir / "extractions.jsonl"),
        write_paper_tables=str(run_dir / "paper_tables"),
        output=str(run_dir / "metrics.corrected_staged.json"),
    )


def test_compute_corrected_staged_metrics_uses_mean_policy_and_raw_question_scores(tmp_path):
    report = compute_corrected_staged_metrics(_args(tmp_path))

    assert report["primary_reference"] == "human_mean_corrected_policy_total"
    primary_total = report["references"]["human_mean_corrected_policy_total"]["1"]["human_total_score"]
    assert primary_total == 60
    q5 = report["per_question_metrics"]["no_lecture"]["original"]["q5"]
    assert q5["mae_marks"] == 0


def test_corrected_metrics_separate_raw_and_final_over_60_and_write_trace(tmp_path):
    args = _args(tmp_path)
    report = compute_corrected_staged_metrics(args)
    run_dir = Path(args.output).parent

    assert report["safety_metrics"]["no_lecture"]["raw_all_questions_total_over_60_count"] == 1
    assert report["safety_metrics"]["no_lecture"]["final_ai_total_over_60_count"] == 0
    assert (run_dir / "paper_tables" / "corrected_raw_score_trace.csv").exists()
    assert (run_dir / "stage_io" / "metrics" / "inputs" / "corrected_staged.json").exists()
    assert (run_dir / "stage_io" / "metrics" / "outputs" / "corrected_staged.json").exists()
    index = json.loads((run_dir / "pipeline_trace_index.json").read_text())
    assert "extraction" in index["scripts"]["1"]["stages"]
    assert "grading" in index["scripts"]["1"]["conditions"]["no_lecture"]["stages"]
    assert "finalization" in index["scripts"]["1"]["conditions"]["lecture"]["stages"]
    assert "downstream_shadow" in index["scripts"]["1"]["conditions"]["lecture"]["stages"]
    assert "metrics" in index["global"]
