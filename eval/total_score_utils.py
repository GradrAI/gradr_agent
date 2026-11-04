"""Utilities for total-score grading experiments.

The existing eval scaffold is question-level. These helpers support datasets where the
available human label is only the final script score.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

_ROW_WITH_PIPES_RE = re.compile(
    r"\|\s*(?P<serial>\d{1,4})\s*\|\s*(?P<score>-?\d+(?:\.\d+)?)\s*\|\s*(?P<pages>\d{1,4})\s*\|"
)
_ROW_WITH_SPACES_RE = re.compile(
    r"^\s*(?P<serial>\d{1,4})\s+(?P<score>-?\d+(?:\.\d+)?)\s+(?P<pages>\d{1,4})\s*$"
)
_MAX_SCORE_RE = re.compile(r"Maximum\s+achievable\s+score\s*:\s*(\d+(?:\.\d+)?)", re.I)
_NUMBER_RUN_RE = re.compile(r"\d+")
_NUMBER_LINE_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _is_number_token(value: str) -> bool:
    return _NUMBER_LINE_RE.match(value.strip()) is not None


def _parse_tokenized_rows(text: str) -> dict[int, tuple[float, int]]:
    """Parse PDFs that extract table cells as one numeric token per line."""

    tokens = [line.strip() for line in text.splitlines() if line.strip()]
    rows: dict[int, tuple[float, int]] = {}
    expected_serial = 1
    index = 0
    while index <= len(tokens) - 3:
        serial_token = tokens[index]
        score_token = tokens[index + 1]
        pages_token = tokens[index + 2]
        if (
            serial_token.isdigit()
            and int(serial_token) == expected_serial
            and _is_number_token(score_token)
            and pages_token.isdigit()
        ):
            rows[expected_serial] = (float(_number(score_token)), int(pages_token))
            expected_serial += 1
            index += 3
            continue
        index += 1
    return rows



@dataclass(frozen=True)
class TotalScoreLabel:
    """Human total-score label for one script."""

    script_id: str
    serial_number: int
    human_total_score: float
    max_score: float
    pages_reported: int | None = None


@dataclass(frozen=True)
class TotalScoreBook:
    """Parsed total-score book."""

    labels: dict[str, TotalScoreLabel]
    max_score: float
    source_path: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "max_score": self.max_score,
            "labels": {key: asdict(value) for key, value in self.labels.items()},
        }


def _number(value: str) -> float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or math.isnan(value):
        return None
    return round(float(value), digits)


def canonical_script_id(value: str) -> str:
    """Return the stable numeric script id embedded in a filename or identifier."""

    name = Path(str(value)).name
    stem = Path(name).stem if "." in name else name
    if stem.isdigit():
        return str(int(stem))

    runs = _NUMBER_RUN_RE.findall(stem)
    if len(runs) != 1:
        raise ValueError(f"Ambiguous script identifier: {value!r}")
    return str(int(runs[0]))


def _parse_table_rows(text: str) -> dict[int, tuple[float, int]]:
    rows: dict[int, tuple[float, int]] = {}
    duplicates: list[int] = []

    for line in text.splitlines():
        match = _ROW_WITH_PIPES_RE.search(line) or _ROW_WITH_SPACES_RE.search(line)
        if not match:
            continue

        serial = int(match.group("serial"))
        score = _number(match.group("score"))
        pages = int(match.group("pages"))
        if serial in rows:
            duplicates.append(serial)
            continue
        rows[serial] = (float(score), pages)

    if duplicates:
        duplicate_text = ", ".join(str(item) for item in sorted(set(duplicates)))
        raise ValueError(f"Duplicate score rows for serial number(s): {duplicate_text}")

    if not rows:
        return _parse_tokenized_rows(text)
    return rows



def parse_score_rows(text: str) -> dict[int, float]:
    """Parse score-table rows from extracted score-PDF text."""

    return {serial: score for serial, (score, _pages) in _parse_table_rows(text).items()}


def _parse_max_score(text: str) -> float:
    matches = [_number(match.group(1)) for match in _MAX_SCORE_RE.finditer(text)]
    if not matches:
        raise ValueError("Could not find 'Maximum achievable score' in score PDF text")

    first = float(matches[0])
    inconsistent = {float(item) for item in matches if float(item) != first}
    if inconsistent:
        raise ValueError(f"Inconsistent maximum score values in score PDF: {matches}")
    return first


def parse_scores_pdf(path: Path, expected_count: int = 60) -> TotalScoreBook:
    """Parse the provided total-score PDF into labels keyed by canonical script id."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to parse score PDFs") from exc

    source = Path(path)
    reader = PdfReader(str(source))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    max_score = _parse_max_score(text)
    table = _parse_table_rows(text)

    expected_serials = set(range(1, expected_count + 1))
    actual_serials = set(table)
    missing = sorted(expected_serials - actual_serials)
    unexpected = sorted(actual_serials - expected_serials)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing serials: {missing}")
        if unexpected:
            parts.append(f"unexpected serials: {unexpected}")
        raise ValueError("Invalid score table; " + "; ".join(parts))

    labels: dict[str, TotalScoreLabel] = {}
    for serial in sorted(table):
        score, pages = table[serial]
        if score < 0 or score > max_score:
            raise ValueError(
                f"Score for serial {serial} is outside 0..{max_score}: {score}"
            )
        script_id = str(serial)
        labels[script_id] = TotalScoreLabel(
            script_id=script_id,
            serial_number=serial,
            human_total_score=score,
            max_score=max_score,
            pages_reported=pages,
        )

    return TotalScoreBook(labels=labels, max_score=max_score, source_path=str(source))


def load_total_labels_file(path: Path) -> dict[str, dict[str, Any]]:
    """Load labels from a JSON file for future non-PDF total-score datasets."""

    data = json.loads(Path(path).read_text())
    max_score = float(data.get("max_score", 60)) if isinstance(data, dict) else 60.0
    raw_labels: Any
    if isinstance(data, dict) and "labels" in data:
        raw_labels = data["labels"]
    else:
        raw_labels = data

    labels: dict[str, dict[str, Any]] = {}
    if isinstance(raw_labels, dict):
        iterable = raw_labels.items()
    elif isinstance(raw_labels, list):
        iterable = ((None, item) for item in raw_labels)
    else:
        raise ValueError("total-labels JSON must be a list or object")

    for key, item in iterable:
        if not isinstance(item, dict):
            raise ValueError("Each total-label entry must be an object")
        raw_id = item.get("script_id") or item.get("script_filename") or key
        if raw_id is None:
            raise ValueError(f"Total-label entry missing script id: {item}")
        script_id = canonical_script_id(str(raw_id))
        score = item.get("human_total_score", item.get("score", item.get("human_score")))
        if score is None:
            raise ValueError(f"Total-label entry missing human score: {item}")
        entry_max = float(item.get("max_score", max_score))
        labels[script_id] = {
            "script_id": script_id,
            "human_total_score": float(score),
            "max_score": entry_max,
        }
    return labels


def labels_from_score_book(book: TotalScoreBook) -> dict[str, dict[str, Any]]:
    return {
        script_id: {
            "script_id": label.script_id,
            "serial_number": label.serial_number,
            "human_total_score": label.human_total_score,
            "max_score": label.max_score,
            "pages_reported": label.pages_reported,
        }
        for script_id, label in book.labels.items()
    }


def load_predictions_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    with source.open() as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            raw_id = item.get("script_id") or item.get("script_filename")
            if raw_id is None:
                raise ValueError(f"Prediction line {line_number} missing script_id")
            predictions[canonical_script_id(str(raw_id))] = item
    return predictions


def prediction_total(prediction: dict[str, Any]) -> float:
    value = prediction.get("ai_total_score", prediction.get("total_score"))
    if value is None:
        raise ValueError(f"Prediction missing ai_total_score: {prediction}")
    return float(value)


def prediction_confidence(prediction: dict[str, Any]) -> float | None:
    value = prediction.get("mean_confidence", prediction.get("confidence"))
    if value is None:
        return None
    return float(value)


def prediction_hitl(prediction: dict[str, Any]) -> bool:
    status = str(prediction.get("referee_status", "")).upper()
    if status == "PENDING_REVIEW":
        return True
    confidence = prediction_confidence(prediction)
    return confidence is not None and confidence < 0.70


def lecture_used(prediction: dict[str, Any]) -> bool:
    if prediction.get("lecture_context_used_count", 0):
        return True
    for question in prediction.get("graded_questions", []):
        if question.get("lecture_context_used"):
            return True
    return False


def rubric_boundary_warned(prediction: dict[str, Any]) -> bool:
    if prediction.get("rubric_boundary_warning_count", 0):
        return True
    warnings = prediction.get("rubric_boundary_warnings", [])
    return bool(warnings)


def _qwk_percent_bucket(
    human_scores: list[float], ai_scores: list[float], max_scores: list[float]
) -> float | None:
    if not human_scores:
        return None
    human_pct = [
        round(h / m * 100) for h, m in zip(human_scores, max_scores, strict=True)
    ]
    ai_pct = [round(a / m * 100) for a, m in zip(ai_scores, max_scores, strict=True)]
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return None
    value = float(cohen_kappa_score(human_pct, ai_pct, weights="quadratic"))
    if math.isnan(value):
        return None
    return value


def compute_total_score_metrics(
    labels: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    failed_count: int = 0,
) -> dict[str, Any]:
    """Compute total-score agreement metrics for paired labels/predictions."""

    human_scores: list[float] = []
    ai_scores: list[float] = []
    max_scores: list[float] = []
    confidences: list[float] = []
    hitl_flags: list[bool] = []
    lecture_flags: list[bool] = []
    boundary_flags: list[bool] = []

    for script_id, label in sorted(labels.items(), key=lambda item: int(item[0])):
        prediction = predictions.get(script_id)
        if prediction is None:
            continue
        human = float(label["human_total_score"])
        ai = prediction_total(prediction)
        max_score = float(label.get("max_score", prediction.get("max_score", 60)))
        human_scores.append(human)
        ai_scores.append(ai)
        max_scores.append(max_score)
        confidence = prediction_confidence(prediction)
        if confidence is not None:
            confidences.append(confidence)
        hitl_flags.append(prediction_hitl(prediction))
        lecture_flags.append(lecture_used(prediction))
        boundary_flags.append(rubric_boundary_warned(prediction))

    paired_count = len(human_scores)
    errors = [
        abs(ai - human) for human, ai in zip(human_scores, ai_scores, strict=True)
    ]
    signed_errors = [
        ai - human for human, ai in zip(human_scores, ai_scores, strict=True)
    ]

    metrics: dict[str, Any] = {
        "gold_count": len(labels),
        "prediction_count": len(predictions),
        "paired_count": paired_count,
        "failed_count": failed_count,
    }
    if paired_count == 0:
        metrics.update(
            {
                "mae_marks": None,
                "mae_percent": None,
                "mean_signed_error_marks": None,
                "mean_signed_error_percent": None,
                "exact_total_match_rate": None,
                "within_1_mark_rate": None,
                "within_3_marks_rate": None,
                "within_5_percent_rate": None,
                "qwk_percent_bucket": None,
                "mean_confidence": None,
                "hitl_trigger_rate": None,
                "lecture_context_use_rate": None,
                "rubric_boundary_warning_rate": None,
            }
        )
        return metrics

    percent_errors = [
        error / max_score * 100
        for error, max_score in zip(errors, max_scores, strict=True)
    ]
    signed_percent_errors = [
        error / max_score * 100
        for error, max_score in zip(signed_errors, max_scores, strict=True)
    ]

    metrics.update(
        {
            "mae_marks": _round(mean(errors)),
            "mae_percent": _round(mean(percent_errors)),
            "mean_signed_error_marks": _round(mean(signed_errors)),
            "mean_signed_error_percent": _round(mean(signed_percent_errors)),
            "exact_total_match_rate": _round(
                sum(1 for human, ai in zip(human_scores, ai_scores, strict=True) if human == ai)
                / paired_count
            ),
            "within_1_mark_rate": _round(sum(error <= 1 for error in errors) / paired_count),
            "within_3_marks_rate": _round(sum(error <= 3 for error in errors) / paired_count),
            "within_5_percent_rate": _round(
                sum(
                    (error / max_score * 100) <= 5
                    for error, max_score in zip(errors, max_scores, strict=True)
                )
                / paired_count
            ),
            "qwk_percent_bucket": _round(
                _qwk_percent_bucket(human_scores, ai_scores, max_scores)
            ),
            "mean_confidence": _round(mean(confidences)) if confidences else None,
            "hitl_trigger_rate": _round(sum(hitl_flags) / paired_count),
            "lecture_context_use_rate": _round(sum(lecture_flags) / paired_count),
            "rubric_boundary_warning_rate": _round(sum(boundary_flags) / paired_count),
        }
    )
    return metrics


def compute_lecture_ablation_metrics(
    labels: dict[str, dict[str, Any]],
    without_lecture: dict[str, dict[str, Any]],
    with_lecture: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare paired no-lecture and lecture-context predictions."""

    shared_ids = sorted(
        set(labels) & set(without_lecture) & set(with_lecture), key=lambda item: int(item)
    )
    deltas: list[float] = []
    confidence_deltas: list[float] = []
    hitl_without: list[bool] = []
    hitl_with: list[bool] = []
    improved = worsened = unchanged = 0

    for script_id in shared_ids:
        human = float(labels[script_id]["human_total_score"])
        error_without = abs(prediction_total(without_lecture[script_id]) - human)
        error_with = abs(prediction_total(with_lecture[script_id]) - human)
        delta = error_with - error_without
        deltas.append(delta)
        if delta < 0:
            improved += 1
        elif delta > 0:
            worsened += 1
        else:
            unchanged += 1

        conf_a = prediction_confidence(without_lecture[script_id])
        conf_b = prediction_confidence(with_lecture[script_id])
        if conf_a is not None and conf_b is not None:
            confidence_deltas.append(conf_b - conf_a)
        hitl_without.append(prediction_hitl(without_lecture[script_id]))
        hitl_with.append(prediction_hitl(with_lecture[script_id]))

    metrics_without = compute_total_score_metrics(labels, without_lecture)
    metrics_with = compute_total_score_metrics(labels, with_lecture)

    return {
        "paired_ablation_count": len(shared_ids),
        "mae_without_lecture": metrics_without.get("mae_marks"),
        "mae_with_lecture": metrics_with.get("mae_marks"),
        "delta_mae_marks": _round(mean(deltas)) if deltas else None,
        "improved_count": improved,
        "worsened_count": worsened,
        "unchanged_count": unchanged,
        "confidence_delta": _round(mean(confidence_deltas))
        if confidence_deltas
        else None,
        "hitl_trigger_delta": _round(
            (sum(hitl_with) / len(hitl_with)) - (sum(hitl_without) / len(hitl_without))
        )
        if hitl_with
        else None,
        "without_lecture": metrics_without,
        "with_lecture": metrics_with,
    }
