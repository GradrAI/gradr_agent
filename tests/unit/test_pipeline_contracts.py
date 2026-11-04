from pathlib import Path

from eval.pipeline_contracts import (
    compute_review_gate,
    is_action_allowed,
    validate_grading_contract,
)
from eval.rubric_config import load_exam_rubric


def _rubric():
    return load_exam_rubric(Path("eval/rubric_corrected_intro_python.json"))


def _artifact(**overrides):
    base = {
        "script_id": "3",
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
    base.update(overrides)
    return base


def test_validate_grading_contract_detects_numeric_and_selection_violations():
    artifact = _artifact(
        ai_total_score=61.0,
        selected_question_ids=["q1", "q2", "q3", "q4", "q5"],
        raw_model_question_scores={
            "q1": 19.0,
            "q2": 17.0,
            "q3": 8.5,
            "q4": 0.0,
            "q5": 0.0,
        },
        mean_confidence=1.2,
    )

    codes = {violation.code for violation in validate_grading_contract(artifact, _rubric())}

    assert "total_exceeds_exam_max" in codes
    assert "too_many_selected_questions" in codes
    assert "score_exceeds_question_max" in codes
    assert "invalid_confidence" in codes


def test_review_gate_blocks_user_visible_actions_when_validation_errors_exist():
    artifact = _artifact(validation_errors=["graded total exceeds max"], referee_status="COMPLETED")
    violations = validate_grading_contract(artifact, _rubric())

    decision = compute_review_gate(artifact, violations)

    assert decision.status == "PENDING_REVIEW"
    assert "validation_errors" in decision.reasons
    assert not is_action_allowed("student_facing_feedback", decision, {}, {})
    assert not is_action_allowed("persist_completed_result", decision, {}, {})
    assert is_action_allowed("route_human_review", decision, {}, {})


def test_action_contract_requires_linked_user_and_supported_exam_for_smartprep():
    decision = compute_review_gate(_artifact(), [])

    assert is_action_allowed(
        "smartprep_generation",
        decision,
        {"linked_user_id": "user-1"},
        {"exam_type": "wassce"},
    )
    assert not is_action_allowed(
        "smartprep_generation",
        decision,
        {"linked_user_id": ""},
        {"exam_type": "wassce"},
    )
    assert not is_action_allowed(
        "smartprep_generation",
        decision,
        {"linked_user_id": "user-1"},
        {"exam_type": "ncee"},
    )
