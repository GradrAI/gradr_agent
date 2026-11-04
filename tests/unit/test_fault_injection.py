from pathlib import Path

from eval.fault_injection import inject_fault, make_negative_control, repair_fault
from eval.pipeline_contracts import compute_review_gate, validate_grading_contract
from eval.rubric_config import load_exam_rubric


def _rubric():
    return load_exam_rubric(Path("eval/rubric_corrected_intro_python.json"))


def _artifact():
    return {
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


def test_negative_control_preserves_payload_and_expected_no_faults():
    variant = make_negative_control(_artifact(), "drop_pending_review_status")

    assert variant.variant_name == "negative_control"
    assert variant.payload == _artifact()
    assert variant.before_hash == variant.after_hash
    assert variant.expected_violation_codes == ()
    assert variant.expected_blocked_actions == ()


def test_inject_drop_pending_review_status_creates_objective_fault():
    variant = inject_fault(_artifact(), "drop_pending_review_status")

    assert variant.variant_name == "controlled_fault"
    assert variant.fault_name == "drop_pending_review_status"
    assert "review_status_dropped" in variant.expected_violation_codes
    assert "student_facing_feedback" in variant.expected_blocked_actions
    codes = {violation.code for violation in validate_grading_contract(variant.payload, _rubric())}
    assert "review_status_dropped" in codes
    assert variant.before_hash != variant.after_hash


def test_repair_restores_pending_review_status_without_losing_review_gate():
    controlled = inject_fault(_artifact(), "drop_pending_review_status")

    repaired = repair_fault(controlled)

    assert repaired.variant_name == "repaired_counterfactual"
    assert repaired.payload["referee_status"] == "PENDING_REVIEW"
    codes = {violation.code for violation in validate_grading_contract(repaired.payload, _rubric())}
    assert "review_status_dropped" not in codes
    decision = compute_review_gate(repaired.payload, list(validate_grading_contract(repaired.payload, _rubric())))
    assert decision.status == "PENDING_REVIEW"
