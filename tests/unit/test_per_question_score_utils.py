from pathlib import Path

import pytest

from eval.per_question_score_utils import (
    book_to_json,
    parse_per_question_scores_csv,
    question_attempt_set,
    question_positive_set,
)


def test_parse_original_csv_preserves_blanks_zero_and_warnings(tmp_path: Path):
    csv_path = tmp_path / "original.csv"
    csv_path.write_text(
        "Course: Introduction to Programming Using Python,,,,,,,,\n"
        "Maximum achievable score: 60,,,,,,,,\n"
        ",,,,,,,,\n"
        "Serial Number,Q1,Q2,Q3,Q4,Q5,SUM,Total Score,No. of Pages\n"
        "1,7,0,,,,7,8,2\n"
        "2,3,,,3,,6,6,2\n"
    )

    book = parse_per_question_scores_csv(csv_path, "original", expected_count=2)

    assert book.max_score == 60
    assert len(book.labels) == 2
    first = book.labels["1"]
    assert first.scores == {"q1": 7.0, "q2": 0.0, "q3": None, "q4": None, "q5": None}
    assert first.computed_sum == 7.0
    assert first.recorded_sum == 7.0
    assert first.recorded_total == 8.0
    assert first.pages_reported == 2
    assert question_attempt_set(first) == {"q1", "q2"}
    assert question_positive_set(first) == {"q1"}
    assert any("recorded total 8 != computed question sum 7" in warning for warning in book.warnings)


def test_parse_grader2_csv_with_direct_header(tmp_path: Path):
    csv_path = tmp_path / "grader2.csv"
    csv_path.write_text(
        "Student ID,Q1,Q2,Q3,Q4,Q5,Total\n"
        "1,5,3,,,,8\n"
        "2,2,,,1,,3\n"
    )

    book = parse_per_question_scores_csv(csv_path, "grader2", expected_count=2)

    assert book.max_score == 60
    assert book.labels["1"].recorded_total == 8.0
    assert book.labels["1"].recorded_sum is None
    assert book.labels["1"].pages_reported is None
    assert question_attempt_set(book.labels["2"]) == {"q1", "q4"}
    assert book.warnings == []


def test_parse_per_question_csv_rejects_duplicates(tmp_path: Path):
    csv_path = tmp_path / "duplicate.csv"
    csv_path.write_text(
        "Student ID,Q1,Q2,Q3,Q4,Q5,Total\n"
        "1,5,3,,,,8\n"
        "1,2,,,1,,3\n"
    )

    with pytest.raises(ValueError, match="Duplicate score rows"):
        parse_per_question_scores_csv(csv_path, "grader2", expected_count=1)


def test_parse_per_question_csv_rejects_missing_serials(tmp_path: Path):
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text(
        "Student ID,Q1,Q2,Q3,Q4,Q5,Total\n"
        "1,5,3,,,,8\n"
    )

    with pytest.raises(ValueError, match="missing serials"):
        parse_per_question_scores_csv(csv_path, "grader2", expected_count=2)


def test_parse_per_question_csv_rejects_nonnumeric_scores(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "Student ID,Q1,Q2,Q3,Q4,Q5,Total\n"
        "1,good,3,,,,8\n"
    )

    with pytest.raises(ValueError, match="Invalid numeric value"):
        parse_per_question_scores_csv(csv_path, "grader2", expected_count=1)


def test_book_to_json_includes_rows_and_warnings(tmp_path: Path):
    csv_path = tmp_path / "original.csv"
    csv_path.write_text(
        "Maximum achievable score: 60,,,,,,,,\n"
        "Serial Number,Q1,Q2,Q3,Q4,Q5,SUM,Total Score,No. of Pages\n"
        "1,7,4,,,,11,11,2\n"
    )

    book = parse_per_question_scores_csv(csv_path, "original", expected_count=1)
    data = book_to_json(book)

    assert data["grader_id"] == "original"
    assert data["row_count"] == 1
    assert data["labels"]["1"]["scores"]["q1"] == 7.0
    assert data["warnings"] == []
