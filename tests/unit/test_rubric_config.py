from pathlib import Path

import pytest

from eval.rubric_config import (
    load_exam_rubric,
    normalize_question_id,
    question_score_for_selection,
    select_best_allowed_questions,
)


def test_load_exam_rubric_accepts_corrected_intro_python_config():
    rubric = load_exam_rubric(Path("eval/rubric_corrected_intro_python.json"))

    assert [rubric.questions[qid].raw_max_score for qid in ("q1", "q2", "q3", "q4", "q5")] == [18, 17, 17, 17, 24]
    assert rubric.allowed_question_count == 4
    assert rubric.question_equal_max_score == 15
    assert rubric.selection_policy == "best_4_equal_weighted"
    assert rubric.manual_corrections_apply_globally is True


def test_select_best_allowed_questions_normalizes_equal_weight_questions():
    rubric = load_exam_rubric(Path("eval/rubric_corrected_intro_python.json"))
    scores = [
        question_score_for_selection("q1", 18, 18, rubric, True),
        question_score_for_selection("q2", 17, 17, rubric, True),
        question_score_for_selection("q3", 8.5, 17, rubric, True),
        question_score_for_selection("q4", 17, 17, rubric, True),
        question_score_for_selection("q5", 24, 24, rubric, True),
    ]

    result = select_best_allowed_questions(scores, rubric)

    assert result.selected_question_ids == ["q1", "q2", "q4", "q5"]
    assert result.unselected_question_ids == ["q3"]
    assert result.normalized_question_scores["q3"] == 7.5
    assert result.ai_total_score == 60


def test_select_best_allowed_questions_preserves_raw_attempt_sum():
    rubric = load_exam_rubric(Path("eval/rubric_corrected_intro_python.json"))
    scores = [
        question_score_for_selection("q1", 18, 18, rubric, True),
        question_score_for_selection("q2", 17, 17, rubric, True),
        question_score_for_selection("q3", 17, 17, rubric, True),
        question_score_for_selection("q4", 17, 17, rubric, True),
        question_score_for_selection("q5", 24, 24, rubric, True),
    ]

    result = select_best_allowed_questions(scores, rubric)

    assert result.raw_all_questions_total_score == 93
    assert result.raw_attempted_total_score == 93
    assert result.raw_selected_total_score == 69
    assert result.ai_total_score == 60


@pytest.mark.parametrize("value", ["1a", "1(a)", "q1b"])
def test_normalize_question_id_rejects_subpart_ids(value):
    with pytest.raises(ValueError, match="Subpart id"):
        normalize_question_id(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", "q1"), ("Q1", "q1"), ("q1", "q1"), ("Question 1", "q1"), ("question_1", "q1")],
)
def test_normalize_question_id_accepts_whole_question_ids(value, expected):
    assert normalize_question_id(value) == expected
