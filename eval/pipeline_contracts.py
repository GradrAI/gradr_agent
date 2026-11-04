"""Deterministic contracts for downstream assessment replay.

This module is intentionally side-effect-free. It imports only evaluation helpers and
standard-library modules, performs no model calls, and writes no files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.rubric_config import ExamRubric, QUESTION_ORDER

LOW_CONFIDENCE_THRESHOLD = 0.70
PENDING_REVIEW = "PENDING_REVIEW"
COMPLETED = "COMPLETED"
USER_VISIBLE_OR_PERSISTENT_ACTIONS = {
    "persist_completed_result",
    "student_facing_feedback",
    "weakness_profile_update",
    "smartprep_generation",
}
ALOC_EXAM_TYPES = {"utme", "wassce", "post-utme", "neco"}


@dataclass(frozen=True)
class PredictionArtifact:
    script_id: str
    condition: str | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class ContractViolation:
    code: str
    message: str
    severity: str = "error"
    field: str | None = None


@dataclass(frozen=True)
class ReviewDecision:
    status: str
    reasons: tuple[str, ...]
    blocked_actions: tuple[str, ...]


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _question_scores(payload: dict[str, Any], key: str) -> dict[str, float]:
    raw = payload.get(key)
    if not isinstance(raw, dict):
        return {}
    scores: dict[str, float] = {}
    for qid, value in raw.items():
        number = _as_float(value)
        if number is not None:
            scores[str(qid)] = number
    return scores


def load_prediction_artifact(path: Path) -> PredictionArtifact:
    """Load one prediction artifact from a JSON or one-row JSONL file."""

    text = Path(path).read_text().strip()
    if not text:
        raise ValueError(f"prediction artifact is empty: {path}")
    if Path(path).suffix == ".jsonl":
        payload = json.loads(text.splitlines()[0])
    else:
        payload = json.loads(text)
    return PredictionArtifact(
        script_id=str(payload.get("script_id", "")),
        condition=payload.get("lecture_mode") or payload.get("condition"),
        payload=payload,
    )


def validate_grading_contract(
    artifact: PredictionArtifact | dict[str, Any], rubric: ExamRubric
) -> list[ContractViolation]:
    payload = artifact.payload if isinstance(artifact, PredictionArtifact) else artifact
    violations: list[ContractViolation] = []
    max_score = _as_float(payload.get("max_score")) or float(rubric.total_max_score)
    ai_total = _as_float(payload.get("ai_total_score"))
    if ai_total is None:
        violations.append(
            ContractViolation("missing_total_score", "ai_total_score is missing or invalid", field="ai_total_score")
        )
    elif ai_total > max_score:
        violations.append(
            ContractViolation(
                "total_exceeds_exam_max",
                f"ai_total_score {ai_total:g} exceeds max_score {max_score:g}",
                field="ai_total_score",
            )
        )

    selected = [str(qid) for qid in _as_list(payload.get("selected_question_ids"))]
    allowed = int(payload.get("allowed_question_count") or rubric.allowed_question_count)
    if len(selected) > allowed:
        violations.append(
            ContractViolation(
                "too_many_selected_questions",
                f"selected {len(selected)} questions but allowed {allowed}",
                field="selected_question_ids",
            )
        )

    raw_scores = _question_scores(payload, "raw_model_question_scores")
    canonical = set(QUESTION_ORDER)
    missing = sorted(canonical - set(raw_scores))
    extra = sorted(set(raw_scores) - canonical)
    for qid in missing:
        violations.append(
            ContractViolation("missing_question_score", f"missing raw score for {qid}", field="raw_model_question_scores")
        )
    for qid in extra:
        violations.append(
            ContractViolation("unexpected_question_score", f"unexpected raw score for {qid}", field="raw_model_question_scores")
        )
    for qid in sorted(canonical & set(raw_scores)):
        score = raw_scores[qid]
        max_raw = float(rubric.questions[qid].raw_max_score)
        if score > max_raw:
            violations.append(
                ContractViolation(
                    "score_exceeds_question_max",
                    f"{qid} raw score {score:g} exceeds corrected max {max_raw:g}",
                    field=f"raw_model_question_scores.{qid}",
                )
            )
        if score < 0:
            violations.append(
                ContractViolation(
                    "negative_question_score",
                    f"{qid} raw score {score:g} is negative",
                    field=f"raw_model_question_scores.{qid}",
                )
            )

    normalized_scores = _question_scores(payload, "normalized_question_scores")
    if ai_total is not None and normalized_scores and selected:
        selected_total = sum(normalized_scores.get(qid, 0.0) for qid in selected)
        expected_total = min(selected_total, max_score)
        if abs(ai_total - expected_total) > 0.01:
            violations.append(
                ContractViolation(
                    "raw_normalized_mismatch",
                    f"ai_total_score {ai_total:g} does not match selected normalized total {expected_total:g}",
                    field="ai_total_score",
                )
            )

    for field in ("mean_confidence", "min_confidence"):
        confidence = _as_float(payload.get(field))
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            violations.append(
                ContractViolation("invalid_confidence", f"{field} {confidence:g} is outside [0, 1]", field=field)
            )
    for index, question in enumerate(_as_list(payload.get("graded_questions"))):
        if not isinstance(question, dict):
            continue
        confidence = _as_float(question.get("model_confidence"))
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            violations.append(
                ContractViolation(
                    "invalid_confidence",
                    f"graded_questions[{index}].model_confidence {confidence:g} is outside [0, 1]",
                    field=f"graded_questions[{index}].model_confidence",
                )
            )
        source_priority = str(question.get("source_priority", "")).lower()
        if source_priority == "lecture_context":
            violations.append(
                ContractViolation(
                    "lecture_context_authority_violation",
                    "lecture context was marked as scoring authority",
                    field=f"graded_questions[{index}].source_priority",
                )
            )
        text = f"{question.get('justification', '')} {question.get('feedback', '')}"
        if "ABSENT_EVIDENCE" in text:
            violations.append(
                ContractViolation(
                    "absent_evidence_citation",
                    "justification or feedback cites absent evidence marker",
                    field=f"graded_questions[{index}]",
                )
            )

    if "graded_questions" in payload and not _as_list(payload.get("graded_questions")):
        violations.append(
            ContractViolation("empty_graded_questions", "graded_questions is empty", field="graded_questions")
        )

    status = str(payload.get("referee_status", "")).upper()
    if status == COMPLETED and _has_review_trigger(payload):
        violations.append(
            ContractViolation(
                "review_status_dropped",
                "artifact is COMPLETED despite validation, warning, or confidence review trigger",
                field="referee_status",
            )
        )
    if not _as_list(payload.get("validation_errors")) and any(
        violation.code
        in {
            "total_exceeds_exam_max",
            "too_many_selected_questions",
            "score_exceeds_question_max",
            "raw_normalized_mismatch",
        }
        for violation in violations
    ):
        violations.append(
            ContractViolation(
                "validation_errors_removed",
                "validation_errors is empty despite detected structural grading violations",
                field="validation_errors",
            )
        )
    if payload.get("_dropped_extraction_warning"):
        violations.append(
            ContractViolation(
                "extraction_warning_removed",
                "variant records that an extraction warning was removed",
                field="transcription_warnings",
            )
        )
    return violations


def _has_review_trigger(payload: dict[str, Any]) -> bool:
    if _as_list(payload.get("validation_errors")):
        return True
    if _as_list(payload.get("transcription_warnings")):
        return True
    if int(payload.get("extraction_transcription_warning_count") or 0) > 0:
        return True
    if int(payload.get("low_confidence_count") or 0) > 0:
        return True
    min_confidence = _as_float(payload.get("min_confidence"))
    return min_confidence is not None and min_confidence < LOW_CONFIDENCE_THRESHOLD


def compute_review_gate(
    artifact: PredictionArtifact | dict[str, Any],
    violations: list[ContractViolation],
) -> ReviewDecision:
    payload = artifact.payload if isinstance(artifact, PredictionArtifact) else artifact
    reasons: list[str] = []
    if _as_list(payload.get("validation_errors")):
        reasons.append("validation_errors")
    if _as_list(payload.get("transcription_warnings")) or int(
        payload.get("extraction_transcription_warning_count") or 0
    ) > 0:
        reasons.append("transcription_warnings")
    if int(payload.get("low_confidence_count") or 0) > 0:
        reasons.append("low_confidence")
    min_confidence = _as_float(payload.get("min_confidence"))
    if min_confidence is not None and min_confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")
    if violations:
        reasons.append("contract_violations")
    status = PENDING_REVIEW if reasons else COMPLETED
    blocked = tuple(sorted(USER_VISIBLE_OR_PERSISTENT_ACTIONS)) if status == PENDING_REVIEW else ()
    return ReviewDecision(status=status, reasons=tuple(dict.fromkeys(reasons)), blocked_actions=blocked)


def is_action_allowed(
    action: str,
    review_decision: ReviewDecision,
    student_context: dict[str, Any] | None,
    exam_context: dict[str, Any] | None,
) -> bool:
    student_context = student_context or {}
    exam_context = exam_context or {}
    if action == "route_human_review":
        return review_decision.status == PENDING_REVIEW
    if action in USER_VISIBLE_OR_PERSISTENT_ACTIONS and review_decision.status != COMPLETED:
        return False
    if action == "smartprep_generation":
        linked_user_id = str(student_context.get("linked_user_id") or "").strip()
        exam_type = str(exam_context.get("exam_type") or "").lower().strip()
        return bool(linked_user_id) and exam_type in ALOC_EXAM_TYPES
    if action in USER_VISIBLE_OR_PERSISTENT_ACTIONS:
        return True
    raise ValueError(f"unknown downstream action: {action}")
