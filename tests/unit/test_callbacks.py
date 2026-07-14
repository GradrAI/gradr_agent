import pytest
import json
from unittest.mock import MagicMock, patch

from app.callbacks import (
    preprocessing_after_callback,
    grading_after_callback,
    referee_after_callback,
    weakness_after_callback,
    deterministic_mcq_grading
)

class MockCallbackContext:
    def __init__(self, state=None):
        self.state = state or {}

def test_preprocessing_after_callback_skipped():
    context = MockCallbackContext(state={
        "preprocessing_context": json.dumps({"skipped": True, "message": "Result already exists"})
    })
    preprocessing_after_callback(context)
    assert context.state.get("skipped") is True

def test_preprocessing_after_callback_error():
    context = MockCallbackContext(state={
        "preprocessing_context": json.dumps({"error": "Missing exam"})
    })
    with pytest.raises(ValueError, match="CRITICAL: Preprocessing missing dependencies: Missing exam"):
        preprocessing_after_callback(context)

def test_preprocessing_after_callback_success():
    context = MockCallbackContext(state={
        "preprocessing_context": json.dumps({
            "questions": [{"id": 1}],
            "rubric": {"1": "A"},
            "max_score": 100,
            "linked_user_id": "user123"
        })
    })
    preprocessing_after_callback(context)
    assert context.state["questions"] == [{"id": 1}]
    assert context.state["rubric"] == {"1": "A"}
    assert context.state["max_score"] == 100
    assert context.state["linked_user_id"] == "user123"
    assert context.state["weakness_profile"] == []
    assert context.state["referee_status"] == "COMPLETED"

def test_grading_after_callback_success():
    context = MockCallbackContext(state={
        "graded_result": json.dumps({"graded_questions": [{"id": 1, "score": 5}]})
    })
    grading_after_callback(context)
    assert context.state["graded_questions"] == [{"id": 1, "score": 5}]

def test_grading_after_callback_empty():
    context = MockCallbackContext(state={
        "graded_result": json.dumps({"graded_questions": []})
    })
    with pytest.raises(ValueError, match="CRITICAL: GradingAgent produced zero graded questions."):
        grading_after_callback(context)

def test_grading_after_callback_no_output():
    context = MockCallbackContext(state={})
    with pytest.raises(ValueError, match="CRITICAL: GradingAgent produced no graded result."):
        grading_after_callback(context)

def test_referee_after_callback_success():
    context = MockCallbackContext(state={
        "referee_report": json.dumps({"status": "VERIFIED"})
    })
    referee_after_callback(context)
    assert context.state["referee_status"] == "VERIFIED"

def test_weakness_after_callback_success():
    context = MockCallbackContext(state={
        "weakness_profile_raw": json.dumps({"weakTopics": ["Math"]})
    })
    weakness_after_callback(context)
    assert context.state["weakness_profile"] == {"weakTopics": ["Math"]}

def test_weakness_after_callback_no_output():
    context = MockCallbackContext(state={})
    weakness_after_callback(context)
    assert context.state["weakness_profile"] == {"weakTopics": [], "classWeakTopics": []}

def test_deterministic_mcq_grading():
    context = MockCallbackContext(state={
        "attempt_context": json.dumps({
            "attempt": {
                "answers": {
                    "q1": "A",
                    "q2": "C"
                }
            },
            "exam": {
                "questions": [
                    {"_id": {"$oid": "q1"}, "correctOptionId": "A", "maxMarks": 2},
                    {"_id": {"$oid": "q2"}, "correctOptionId": "B", "maxMarks": 2}
                ]
            }
        })
    })
    
    result = deterministic_mcq_grading(context)
    
    # Assert
    assert result is not None
    assert len(result.parts) == 1
    assert "mcq_results" in result.parts[0].text
    
    # Verify the results string
    raw_results = context.state.get("mcq_results_raw")
    assert raw_results is not None
    
    results = json.loads(raw_results)["mcq_results"]
    assert len(results) == 2
    assert results[0]["questionId"] == "q1"
    assert results[0]["score"] == 2
    assert results[1]["questionId"] == "q2"
    assert results[1]["score"] == 0
