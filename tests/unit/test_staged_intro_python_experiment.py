import json
from pathlib import Path

from eval import staged_intro_python_experiment as runner
from eval.rubric_config import load_exam_rubric
from eval.total_score_utils import TotalScoreBook, TotalScoreLabel


def _write_blank_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as file:
        writer.write(file)


def _case(tmp_path: Path) -> runner.ExperimentCase:
    script = tmp_path / "3.pdf"
    _write_blank_pdf(script)
    return runner.ExperimentCase(
        script_id="3",
        script_filename="3.pdf",
        script_path=str(script),
        human_total_score=40,
        max_score=60,
        page_count=1,
        pdf_size_bytes=script.stat().st_size,
        pages_reported=1,
    )


def _rubric() -> runner.ExamRubric:
    return load_exam_rubric(Path("eval/rubric_corrected_intro_python.json"))


def _extraction(attempts=None):
    attempts = attempts or {qid: True for qid in runner.QUESTION_ORDER}
    return {
        "script_id": "3",
        "parsed_output_path": "extraction.json",
        "raw_model_response_path": "raw.txt",
        "repair_model_response_path": None,
        "questions": [
            {
                "question_id": qid,
                "attempted": attempts.get(qid, False),
                "answer_text": "answer" if attempts.get(qid, False) else "",
                "subparts": [],
            }
            for qid in runner.QUESTION_ORDER
        ],
        "transcription_warnings": [],
        "extraction_validation_errors": [],
    }


def _model_output(scores):
    rubric = _rubric()
    rows = []
    for qid, score in scores.items():
        question = rubric.questions[qid]
        first, second, third = question.parts
        rows.append(
            {
                "question_id": qid,
                "attempted": score > 0,
                "score": score,
                "max_score": question.raw_max_score,
                "subpart_scores": [
                    {"part_label": first.label, "score": score, "max_score": first.max_score, "evidence": "", "justification": ""},
                    {"part_label": second.label, "score": 0, "max_score": second.max_score, "evidence": "", "justification": ""},
                    {"part_label": third.label, "score": 0, "max_score": third.max_score, "evidence": "", "justification": ""},
                ],
                "rubric_alignment": [],
                "justification": f"score {score}",
                "model_confidence": 0.9,
                "feedback": "feedback",
                "source_priority": "corrected_rubric",
                "lecture_context_used": [],
            }
        )
    return {
        "student_identifier_seen": None,
        "graded_questions": rows,
        "ignored_annotations": [],
        "transcription_warnings": [],
        "rubric_boundary_warnings": [],
        "extraction_limitations": [],
    }


def test_finalize_extraction_requires_all_canonical_questions(tmp_path):
    case = _case(tmp_path)
    output = {
        "student_identifier_seen": None,
        "questions": [
            {"question_id": qid, "attempted": True, "answer_text": "x", "subparts": [], "page_numbers": [1]}
            for qid in ("q1", "q2", "q3", "q5")
        ],
        "missing_or_blank_questions": [],
        "ignored_annotations": [],
        "transcription_warnings": [],
    }

    finalized = runner.finalize_extraction(case, output, duration_ms=12)

    assert {question["question_id"] for question in finalized["questions"]} == {"q1", "q2", "q3", "q4", "q5"}
    q4 = next(question for question in finalized["questions"] if question["question_id"] == "q4")
    assert q4["attempted"] is False
    assert "extraction missing q4 row" in finalized["extraction_validation_errors"]


def test_finalize_staged_prediction_uses_extraction_attempt_mask(tmp_path):
    case = _case(tmp_path)
    extraction = _extraction({"q1": True, "q2": True, "q3": True, "q4": False, "q5": True})
    output = _model_output({"q1": 10, "q2": 10, "q3": 10, "q4": 5, "q5": 10})

    prediction = runner.finalize_staged_prediction(case, output, extraction, _rubric(), "summary", 1)

    assert prediction["raw_model_question_scores"]["q4"] == 5
    assert "graded q4 has positive score despite extraction attempted=false" in prediction["validation_errors"]
    assert prediction["referee_status"] == "PENDING_REVIEW"
    assert "q4" not in prediction["selected_question_ids"]


def test_finalize_staged_prediction_selects_best_four_equal_weighted(tmp_path):
    case = _case(tmp_path)
    output = _model_output({"q1": 8, "q2": 17, "q3": 17, "q4": 17, "q5": 24})

    prediction = runner.finalize_staged_prediction(case, output, _extraction(), _rubric(), "summary", 1)

    assert len(prediction["selected_question_ids"]) == 4
    assert prediction["ai_total_score"] <= 60
    assert prediction["raw_attempted_total_score"] == 83
    assert prediction["score_summary"].endswith("/60")


def test_staged_dry_run_writes_manifest_rubric_prompts_and_stage_io(tmp_path, monkeypatch):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for filename in ("question.pdf", "guide.pdf", "1.pdf", "2.pdf"):
        _write_blank_pdf(scripts_dir / filename)
    lecture = tmp_path / "lecture.pdf"
    _write_blank_pdf(lecture)
    score_book = TotalScoreBook(
        labels={
            "1": TotalScoreLabel("1", 1, 11.0, 60.0, 1),
            "2": TotalScoreLabel("2", 2, 6.0, 60.0, 1),
        },
        max_score=60.0,
        source_path=str(tmp_path / "scores.pdf"),
    )
    monkeypatch.setattr(runner, "parse_scores_pdf", lambda _path: score_book)
    monkeypatch.setattr(
        runner,
        "build_lecture_context",
        lambda materials, mode: {"mode": mode, "materials": [str(item) for item in materials], "concepts": [], "warnings": []},
    )

    out_dir = tmp_path / "out"
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--scripts-dir",
            str(scripts_dir),
            "--scores-pdf",
            str(tmp_path / "scores.pdf"),
            "--lecture-material",
            str(lecture),
            "--rubric-json",
            "eval/rubric_corrected_intro_python.json",
            "--script-ids",
            "1,2",
            "--out-dir",
            str(out_dir),
            "--dry-run",
            "--run-lecture-ablation",
            "--overwrite",
        ]
    )
    runner.validate_full_run_gate(args, parser)

    assert runner.run_experiment(args) == 0

    for name in (
        "manifest.json",
        "rubric.json",
        "lecture_context.json",
        "experiment_prompt.extraction.txt",
        "stage_io.jsonl",
        "pipeline_trace_index.json",
        "run_summary.json",
    ):
        assert (out_dir / name).exists(), name
    index = json.loads((out_dir / "pipeline_trace_index.json").read_text())
    assert "extraction" in index["scripts"]["1"]["stages"]
    assert "grading" in index["scripts"]["1"]["conditions"]["no_lecture"]["stages"]
    assert "finalization" in index["scripts"]["1"]["conditions"]["lecture"]["stages"]
