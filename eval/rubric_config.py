"""Corrected rubric configuration and best-question selection helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

QUESTION_ORDER = ("q1", "q2", "q3", "q4", "q5")
_PART_ORDER = ("a", "b", "c")
_SELECTION_POLICY = "best_4_equal_weighted"


@dataclass(frozen=True)
class RubricPart:
    label: str
    max_score: float


@dataclass(frozen=True)
class QuestionRubric:
    question_id: str
    parts: tuple[RubricPart, ...]
    raw_max_score: float


@dataclass(frozen=True)
class ExamRubric:
    exam: str
    total_max_score: float
    allowed_question_count: int
    selection_policy: str
    question_equal_max_score: float
    manual_corrections_apply_globally: bool
    questions: dict[str, QuestionRubric]
    source_path: str | None


@dataclass(frozen=True)
class QuestionScoreForSelection:
    question_id: str
    raw_model_score: float
    model_reported_max_score: float | None
    corrected_raw_max_score: float
    bounded_raw_score: float
    normalized_score: float
    attempted: bool


@dataclass(frozen=True)
class SelectionResult:
    selected_question_ids: list[str]
    unselected_question_ids: list[str]
    raw_model_question_scores: dict[str, float]
    model_reported_question_max_scores: dict[str, float | None]
    corrected_raw_question_max_scores: dict[str, float]
    bounded_raw_question_scores: dict[str, float]
    raw_all_questions_total_score: float
    raw_attempted_total_score: float
    raw_selected_total_score: float
    ai_total_score: float
    normalized_question_scores: dict[str, float]


def _float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from exc


def _round(value: float) -> float:
    return round(float(value), 4)


def normalize_question_id(raw: Any) -> str:
    """Normalize whole-question IDs to q1..q5; reject subpart IDs."""

    text = str(raw).strip().lower()
    compact = re.sub(r"[\s_-]+", "", text)
    if re.fullmatch(r"(?:q|question)?[1-5][a-z]", compact) or re.fullmatch(
        r"(?:q|question)?[1-5]\([a-z]\)", compact
    ):
        raise ValueError(f"Subpart id is not a question_id: {raw!r}")
    match = re.fullmatch(r"(?:q|question)?([1-5])", compact)
    if not match:
        raise ValueError(f"Unrecognized question_id: {raw!r}")
    return f"q{match.group(1)}"


def _question_from_payload(payload: dict[str, Any]) -> QuestionRubric:
    qid = normalize_question_id(payload.get("question_id"))
    raw_parts = payload.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError(f"{qid} must define non-empty parts")
    parts: list[RubricPart] = []
    seen: set[str] = set()
    for raw_part in raw_parts:
        if not isinstance(raw_part, dict):
            raise ValueError(f"{qid} part must be an object")
        label = str(raw_part.get("label", "")).strip().lower()
        if label not in _PART_ORDER:
            raise ValueError(f"{qid} has invalid part label: {label!r}")
        if label in seen:
            raise ValueError(f"{qid} has duplicate part label: {label}")
        seen.add(label)
        max_score = _float(raw_part.get("max_score"), f"{qid}{label}.max_score")
        if max_score < 0:
            raise ValueError(f"{qid}{label}.max_score is negative")
        parts.append(RubricPart(label=label, max_score=max_score))
    labels = tuple(part.label for part in parts)
    if labels != _PART_ORDER:
        raise ValueError(f"{qid} parts must be exactly a,b,c")
    expected = sum(part.max_score for part in parts)
    raw_max_score = _float(payload.get("raw_max_score"), f"{qid}.raw_max_score")
    if abs(raw_max_score - expected) > 1e-9:
        raise ValueError(
            f"{qid}.raw_max_score {raw_max_score:g} != sum(parts) {expected:g}"
        )
    return QuestionRubric(qid, tuple(parts), raw_max_score)


def load_exam_rubric(path: Path) -> ExamRubric:
    source = Path(path)
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Rubric JSON must be an object")
    selection_policy = str(payload.get("selection_policy", ""))
    if selection_policy != _SELECTION_POLICY:
        raise ValueError(f"selection_policy must be {_SELECTION_POLICY!r}")
    total_max = _float(payload.get("total_max_score"), "total_max_score")
    allowed = int(_float(payload.get("allowed_question_count"), "allowed_question_count"))
    if allowed <= 0:
        raise ValueError("allowed_question_count must be positive")
    question_equal = total_max / allowed
    if abs(question_equal - 15.0) > 1e-9:
        raise ValueError("question_equal_max_score must compute to 15.0")
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError("questions must be an array")
    questions = {_question_from_payload(item).question_id: _question_from_payload(item) for item in raw_questions}
    if tuple(sorted(questions)) != QUESTION_ORDER:
        raise ValueError("questions must be exactly q1..q5")
    return ExamRubric(
        exam=str(payload.get("exam", "")),
        total_max_score=total_max,
        allowed_question_count=allowed,
        selection_policy=selection_policy,
        question_equal_max_score=question_equal,
        manual_corrections_apply_globally=bool(
            payload.get("manual_corrections_apply_globally")
        ),
        questions=questions,
        source_path=str(source),
    )


def rubric_to_json(rubric: ExamRubric) -> dict[str, Any]:
    return {
        "exam": rubric.exam,
        "total_max_score": rubric.total_max_score,
        "allowed_question_count": rubric.allowed_question_count,
        "selection_policy": rubric.selection_policy,
        "question_equal_max_score": rubric.question_equal_max_score,
        "manual_corrections_apply_globally": rubric.manual_corrections_apply_globally,
        "source_path": rubric.source_path,
        "questions": {
            qid: {
                "question_id": q.question_id,
                "raw_max_score": q.raw_max_score,
                "parts": [asdict(part) for part in q.parts],
            }
            for qid, q in rubric.questions.items()
        },
    }


def rubric_prompt_section(rubric: ExamRubric) -> str:
    lines = [
        "Corrected rubric mark allocations (authoritative):",
        "The machine-readable corrected rubric overrides scanned, handwritten, or printed mark allocations when they conflict.",
        "Handwritten corrections apply globally. Use the scanned marking guide for model-answer content, not mark totals.",
        f"Final policy: answer any {rubric.allowed_question_count} questions; all questions carry equal marks; each selected question contributes up to {rubric.question_equal_max_score:g} marks toward {rubric.total_max_score:g}.",
        "Question | Part | Corrected raw part max",
        "--- | --- | ---",
    ]
    for qid in QUESTION_ORDER:
        question = rubric.questions[qid]
        for part in question.parts:
            lines.append(f"{qid} | {part.label} | {part.max_score:g}")
        lines.append(f"{qid} | total raw max | {question.raw_max_score:g}")
    return "\n".join(lines)


def question_score_for_selection(
    question_id: str,
    raw_model_score: float,
    model_reported_max_score: float | None,
    rubric: ExamRubric,
    attempted: bool,
) -> QuestionScoreForSelection:
    qid = normalize_question_id(question_id)
    corrected_raw_max = rubric.questions[qid].raw_max_score
    raw = float(raw_model_score)
    bounded = min(max(raw, 0.0), corrected_raw_max)
    normalized = (bounded / corrected_raw_max) * rubric.question_equal_max_score
    return QuestionScoreForSelection(
        question_id=qid,
        raw_model_score=raw,
        model_reported_max_score=model_reported_max_score,
        corrected_raw_max_score=corrected_raw_max,
        bounded_raw_score=_round(bounded),
        normalized_score=_round(normalized),
        attempted=attempted,
    )


def select_best_allowed_questions(
    scores: list[QuestionScoreForSelection], rubric: ExamRubric
) -> SelectionResult:
    score_by_id: dict[str, QuestionScoreForSelection] = {}
    for score in scores:
        qid = normalize_question_id(score.question_id)
        if qid in score_by_id:
            raise ValueError(f"duplicate question score for {qid}")
        score_by_id[qid] = score
    missing = [qid for qid in QUESTION_ORDER if qid not in score_by_id]
    if missing:
        raise ValueError(f"missing question scores: {missing}")

    attempted = [score_by_id[qid] for qid in QUESTION_ORDER if score_by_id[qid].attempted]
    ranked = sorted(
        attempted,
        key=lambda item: (-item.normalized_score, QUESTION_ORDER.index(item.question_id)),
    )
    selected = ranked[: rubric.allowed_question_count]
    selected_ids = [item.question_id for item in selected]
    unselected_ids = [qid for qid in QUESTION_ORDER if qid not in selected_ids]
    raw_scores = {qid: _round(score_by_id[qid].raw_model_score) for qid in QUESTION_ORDER}
    model_maxes = {
        qid: (
            _round(score_by_id[qid].model_reported_max_score)
            if score_by_id[qid].model_reported_max_score is not None
            else None
        )
        for qid in QUESTION_ORDER
    }
    corrected_maxes = {
        qid: _round(score_by_id[qid].corrected_raw_max_score) for qid in QUESTION_ORDER
    }
    bounded_scores = {qid: _round(score_by_id[qid].bounded_raw_score) for qid in QUESTION_ORDER}
    normalized_scores = {
        qid: _round(score_by_id[qid].normalized_score) for qid in QUESTION_ORDER
    }
    raw_all = sum(raw_scores.values())
    raw_attempted = sum(raw_scores[item.question_id] for item in attempted)
    raw_selected = sum(bounded_scores[qid] for qid in selected_ids)
    ai_total = min(sum(normalized_scores[qid] for qid in selected_ids), rubric.total_max_score)
    return SelectionResult(
        selected_question_ids=selected_ids,
        unselected_question_ids=unselected_ids,
        raw_model_question_scores=raw_scores,
        model_reported_question_max_scores=model_maxes,
        corrected_raw_question_max_scores=corrected_maxes,
        bounded_raw_question_scores=bounded_scores,
        raw_all_questions_total_score=_round(raw_all),
        raw_attempted_total_score=_round(raw_attempted),
        raw_selected_total_score=_round(raw_selected),
        ai_total_score=_round(ai_total),
        normalized_question_scores=normalized_scores,
    )


def selection_to_json(result: SelectionResult) -> dict[str, Any]:
    return asdict(result)
