import json
import sys
from pathlib import Path

import pytest

from eval import intro_python_experiment as runner
from eval.total_score_utils import (
    TotalScoreBook,
    TotalScoreLabel,
    canonical_script_id,
    compute_total_score_metrics,
    parse_score_rows,
)

SCORE_TEXT = """
Course: Introduction to Programming Using Python
Number of students sample: 60
Maximum achievable score: 60
| Serial Number | Score | No. of Pages |
|---|---:|---:|
| 1 | 11 | 2 |
| 60 | 17 | 3 |
"""


def test_parse_score_rows():
    assert parse_score_rows(SCORE_TEXT) == {1: 11.0, 60: 17.0}


def test_parse_score_rows_rejects_duplicates():
    duplicate = SCORE_TEXT + "\n| 1 | 12 | 2 |\n"
    with pytest.raises(ValueError, match="Duplicate score rows"):
        parse_score_rows(duplicate)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1.pdf", "1"),
        ("001", "1"),
        ("answer-111502006.pdf", "111502006"),
    ],
)
def test_canonical_script_id(value, expected):
    assert canonical_script_id(value) == expected


def test_canonical_script_id_rejects_ambiguous_names():
    with pytest.raises(ValueError, match="Ambiguous script identifier"):
        canonical_script_id("script-1-retry-2.pdf")


def test_compute_total_score_metrics():
    labels = {
        "1": {"script_id": "1", "human_total_score": 11, "max_score": 60},
        "2": {"script_id": "2", "human_total_score": 6, "max_score": 60},
    }
    predictions = {
        "1": {"script_id": "1", "ai_total_score": 10, "mean_confidence": 0.9},
        "2": {
            "script_id": "2",
            "ai_total_score": 9,
            "mean_confidence": 0.6,
            "referee_status": "PENDING_REVIEW",
        },
    }

    metrics = compute_total_score_metrics(labels, predictions)

    assert metrics["paired_count"] == 2
    assert metrics["mae_marks"] == 2.0
    assert metrics["mean_signed_error_marks"] == 1.0


def test_prompt_contains_annotation_and_lecture_boundaries():
    prompt = runner.INTRO_PYTHON_EXPERIMENT_PROMPT

    for phrase in (
        "Do not write in either margin",
        "Student's Reg. No",
        "red-pen ticks",
        "circled scores",
        "teacher comments",
        "Never use visible teacher marks/scores as labels",
        "marking guide as the scoring authority",
        "Use lecture materials only",
        "Never award marks solely because lecture material",
    ):
        assert phrase in prompt


def test_full_run_gate_requires_explicit_reviewed_output_approval(capsys):
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--scripts-dir",
            "/tmp/scripts",
            "--scores-pdf",
            "/tmp/scores.pdf",
            "--out-dir",
            "/tmp/out",
            "--dry-run",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        runner.validate_full_run_gate(args, parser)

    assert exc.value.code == 2
    assert runner.FULL_RUN_ERROR in capsys.readouterr().err


def _write_blank_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as file:
        writer.write(file)


def test_dry_run_writes_manifest_and_lecture_context(tmp_path, monkeypatch):
    loaded_before = set(sys.modules)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for filename in ("question.pdf", "guide.pdf", "1.pdf", "2.pdf"):
        _write_blank_pdf(scripts_dir / filename)

    score_book = TotalScoreBook(
        labels={
            "1": TotalScoreLabel("1", 1, 11.0, 60.0, 2),
            "2": TotalScoreLabel("2", 2, 6.0, 60.0, 2),
        },
        max_score=60.0,
        source_path=str(tmp_path / "scores.pdf"),
    )
    monkeypatch.setattr(runner, "parse_scores_pdf", lambda _path: score_book)
    monkeypatch.setattr(
        runner,
        "build_lecture_context",
        lambda _materials, mode: {
            "mode": mode,
            "materials": [{"path": "lecture.pdf", "kind": "pdf", "pages_or_slides": 1}],
            "concepts": [{"concept": "algorithm"}],
            "warnings": [],
        },
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
            str(tmp_path / "lecture.pdf"),
            "--script-ids",
            "1,2",
            "--out-dir",
            str(out_dir),
            "--dry-run",
            "--overwrite",
        ]
    )
    runner.validate_full_run_gate(args, parser)

    assert runner.run_experiment(args) == 0

    manifest = json.loads((out_dir / "manifest.json").read_text())
    lecture_context = json.loads((out_dir / "lecture_context.json").read_text())
    summary = json.loads((out_dir / "run_summary.json").read_text())

    assert [case["script_id"] for case in manifest["cases"]] == ["1", "2"]
    assert lecture_context["concepts"] == [{"concept": "algorithm"}]
    assert summary["requested_count"] == 2
    assert summary["dry_run"] is True
    assert summary["succeeded_count"] == 0
    assert not (out_dir / "predictions.jsonl").exists()
    loaded_after = set(sys.modules) - loaded_before
    assert "app.toolsets" not in loaded_after
    assert "app.agents.pbt_grading_pipeline" not in loaded_after
