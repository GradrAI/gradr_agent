"""Utilities for per-question human score-sheet parsing.

The Intro Python paper metrics use CSV exports of human score sheets. Blank cells
mean the grader left that question unmarked/unattempted; explicit numeric zero is
preserved as an attempted zero-mark cell.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

QUESTION_IDS = ("q1", "q2", "q3", "q4", "q5")
_MAX_SCORE_RE = re.compile(r"Maximum\s+achievable\s+score\s*:\s*(\d+(?:\.\d+)?)", re.I)


@dataclass(frozen=True)
class PerQuestionScoreRow:
    """Per-question human labels for one script."""

    script_id: str
    serial_number: int
    scores: dict[str, float | None]
    computed_sum: float
    recorded_sum: float | None
    recorded_total: float | None
    pages_reported: int | None


@dataclass(frozen=True)
class PerQuestionScoreBook:
    """Parsed per-question score sheet."""

    grader_id: str
    labels: dict[str, PerQuestionScoreRow]
    max_score: float
    source_path: str
    warnings: list[str]


def _number(value: str) -> float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def _format_number(value: float) -> str:
    parsed = float(value)
    return str(int(parsed)) if parsed.is_integer() else str(parsed)


def _normalize_header(value: str) -> str | None:
    compact = re.sub(r"[^a-z0-9]", "", value.strip().lower())
    if not compact:
        return None
    if compact in {"serial", "serialnumber", "studentid", "studentnumber", "id"}:
        return "serial"
    if compact in QUESTION_IDS:
        return compact
    if compact == "sum":
        return "recorded_sum"
    if compact in {"total", "totalscore", "score"}:
        return "recorded_total"
    if compact in {"noofpages", "numberofpages", "pages", "pagecount"}:
        return "pages_reported"
    return None


def _find_header_row(rows: list[list[str]]) -> tuple[int, list[str | None]]:
    for index, row in enumerate(rows):
        normalized = [_normalize_header(cell) for cell in row]
        present = {cell for cell in normalized if cell is not None}
        if "serial" in present and set(QUESTION_IDS).issubset(present):
            return index, normalized
    raise ValueError("Could not find per-question score-sheet header row")


def _metadata_max_score(rows: list[list[str]], explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    for row in rows:
        for cell in row:
            match = _MAX_SCORE_RE.search(cell)
            if match:
                return float(_number(match.group(1)))
    return 60.0


def _cell(row: list[str], index: int | None) -> str:
    if index is None or index >= len(row):
        return ""
    return row[index].strip()


def _parse_optional_float(value: str, field_name: str, serial: int | None) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(_number(stripped))
    except ValueError as exc:
        location = "header" if serial is None else f"serial {serial}"
        raise ValueError(f"Invalid numeric value for {field_name} at {location}: {value!r}") from exc


def _parse_optional_int(value: str, field_name: str, serial: int) -> int | None:
    parsed = _parse_optional_float(value, field_name, serial)
    if parsed is None:
        return None
    if float(parsed).is_integer():
        return int(parsed)
    raise ValueError(f"Invalid integer value for {field_name} at serial {serial}: {value!r}")


def _column_index(headers: list[str | None], name: str) -> int | None:
    try:
        return headers.index(name)
    except ValueError:
        return None


def _question_indexes(headers: list[str | None]) -> dict[str, int]:
    indexes: dict[str, int] = {}
    for question_id in QUESTION_IDS:
        try:
            indexes[question_id] = headers.index(question_id)
        except ValueError as exc:
            raise ValueError(f"Missing score-sheet column {question_id.upper()}") from exc
    return indexes


def _close_enough(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return True
    return abs(float(left) - float(right)) < 1e-9


def parse_per_question_scores_csv(
    path: Path,
    grader_id: str,
    expected_count: int = 60,
    max_score: float | None = None,
) -> PerQuestionScoreBook:
    """Parse a per-question human score-sheet CSV export."""

    source = Path(path)
    with source.open(newline="") as file:
        rows = list(csv.reader(file))

    if not rows:
        raise ValueError(f"Score-sheet CSV is empty: {source}")

    header_index, headers = _find_header_row(rows)
    score_max = _metadata_max_score(rows[:header_index], max_score)
    serial_index = _column_index(headers, "serial")
    if serial_index is None:
        raise ValueError("Missing serial/student id column")
    question_indexes = _question_indexes(headers)
    recorded_sum_index = _column_index(headers, "recorded_sum")
    recorded_total_index = _column_index(headers, "recorded_total")
    pages_index = _column_index(headers, "pages_reported")

    labels: dict[str, PerQuestionScoreRow] = {}
    warnings: list[str] = []
    duplicates: list[int] = []

    for raw_row in rows[header_index + 1 :]:
        if not any(cell.strip() for cell in raw_row):
            continue
        serial_text = _cell(raw_row, serial_index)
        if not serial_text:
            continue
        try:
            serial_float = float(_number(serial_text))
        except ValueError as exc:
            raise ValueError(f"Invalid serial/student id value: {serial_text!r}") from exc
        if not serial_float.is_integer():
            raise ValueError(f"Invalid non-integer serial/student id value: {serial_text!r}")
        serial = int(serial_float)
        if serial in {row.serial_number for row in labels.values()}:
            duplicates.append(serial)
            continue

        scores = {
            question_id: _parse_optional_float(
                _cell(raw_row, column_index), question_id.upper(), serial
            )
            for question_id, column_index in question_indexes.items()
        }
        computed_sum = float(sum(score for score in scores.values() if score is not None))
        recorded_sum = _parse_optional_float(
            _cell(raw_row, recorded_sum_index), "SUM", serial
        )
        recorded_total = _parse_optional_float(
            _cell(raw_row, recorded_total_index), "Total", serial
        )
        pages_reported = _parse_optional_int(
            _cell(raw_row, pages_index), "No. of Pages", serial
        )

        if not _close_enough(recorded_sum, computed_sum):
            warnings.append(
                "serial "
                f"{serial}: recorded SUM {_format_number(recorded_sum or 0)} "
                f"!= computed question sum {_format_number(computed_sum)}"
            )
        if not _close_enough(recorded_total, computed_sum):
            warnings.append(
                "serial "
                f"{serial}: recorded total {_format_number(recorded_total or 0)} "
                f"!= computed question sum {_format_number(computed_sum)}"
            )
        if recorded_total is not None and recorded_total > score_max:
            warnings.append(
                f"serial {serial}: recorded total {_format_number(recorded_total)} exceeds max {_format_number(score_max)}"
            )
        if computed_sum > score_max:
            warnings.append(
                f"serial {serial}: computed question sum {_format_number(computed_sum)} exceeds max {_format_number(score_max)}"
            )

        script_id = str(serial)
        labels[script_id] = PerQuestionScoreRow(
            script_id=script_id,
            serial_number=serial,
            scores=scores,
            computed_sum=computed_sum,
            recorded_sum=recorded_sum,
            recorded_total=recorded_total,
            pages_reported=pages_reported,
        )

    if duplicates:
        duplicate_text = ", ".join(str(item) for item in sorted(set(duplicates)))
        raise ValueError(f"Duplicate score rows for serial number(s): {duplicate_text}")

    expected_serials = set(range(1, expected_count + 1))
    actual_serials = {row.serial_number for row in labels.values()}
    missing = sorted(expected_serials - actual_serials)
    unexpected = sorted(actual_serials - expected_serials)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing serials: {missing}")
        if unexpected:
            parts.append(f"unexpected serials: {unexpected}")
        raise ValueError("Invalid per-question score table; " + "; ".join(parts))

    return PerQuestionScoreBook(
        grader_id=grader_id,
        labels=dict(sorted(labels.items(), key=lambda item: int(item[0]))),
        max_score=score_max,
        source_path=str(source),
        warnings=warnings,
    )


def question_attempt_set(row: PerQuestionScoreRow) -> set[str]:
    """Return question IDs whose cells are nonblank, including explicit zeros."""

    return {question_id for question_id, score in row.scores.items() if score is not None}


def question_positive_set(row: PerQuestionScoreRow) -> set[str]:
    """Return question IDs whose cells have positive scores."""

    return {question_id for question_id, score in row.scores.items() if score is not None and score > 0}


def row_to_json(row: PerQuestionScoreRow) -> dict[str, Any]:
    """Serialize a per-question row to JSON-compatible data."""

    return asdict(row)


def book_to_json(book: PerQuestionScoreBook) -> dict[str, Any]:
    """Serialize a score book to JSON-compatible data."""

    return {
        "grader_id": book.grader_id,
        "source_path": book.source_path,
        "max_score": book.max_score,
        "row_count": len(book.labels),
        "warnings": list(book.warnings),
        "labels": {key: row_to_json(value) for key, value in book.labels.items()},
    }


def write_per_question_score_book(path: Path, book: PerQuestionScoreBook) -> None:
    """Write a parsed score book as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(book_to_json(book), indent=2, sort_keys=True))
