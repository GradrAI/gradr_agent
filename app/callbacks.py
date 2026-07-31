import json
import logging
import typing

from google.adk.agents.callback_context import CallbackContext

from app.app_utils.common import _clean_json, _log_agent_complete

logger = logging.getLogger(__name__)


def preprocessing_after_callback(callback_context: CallbackContext) -> None:
    raw_context = callback_context.state.get("preprocessing_context")
    if not raw_context:
        return
    try:
        data = json.loads(_clean_json(raw_context))
        if data.get("skipped"):
            logger.info(
                f"Preprocessing skipped: {data.get('message', 'Result already exists')}"
            )
            callback_context.state["skipped"] = True
            return
        if data.get("error"):
            logger.error("Preprocessing aborted pipeline: %s", data.get("error"))
            raise ValueError(
                f"CRITICAL: Preprocessing missing dependencies: {data.get('error')}"
            )
        callback_context.state["questions"] = data.get("questions", [])
        callback_context.state["rubric"] = data.get("rubric", {})
        callback_context.state["historical_performance"] = data.get(
            "historical_performance", []
        )
        callback_context.state["linked_user_id"] = data.get("linked_user_id")
        callback_context.state["max_score"] = data.get("max_score")
        callback_context.state["exam_type"] = data.get("exam_type", "WAEC")
        # Defensive default state initialization for downstream agents
        callback_context.state["weakness_profile"] = []
        callback_context.state["referee_status"] = "COMPLETED"
        callback_context.state["practice_session_id"] = ""
        logger.info("PreprocessingAgent complete. Handing off to GradingAgent...")
        _log_agent_complete("PreprocessingAgent", "preprocessing_context")
    except Exception as e:
        logger.error("Error parsing preprocessing_context: %s", e, exc_info=True)
        raise e


def grading_after_callback(callback_context: CallbackContext) -> None:
    raw_res = callback_context.state.get("graded_result")
    if not raw_res:
        logger.error("GradingAgent returned no output. Aborting pipeline.")
        raise ValueError("CRITICAL: GradingAgent produced no graded result.")
    try:
        data = json.loads(_clean_json(raw_res))
        graded = data.get("graded_questions", [])
        if not graded:
            logger.error("GradingAgent returned empty graded_questions. Aborting pipeline.")
            raise ValueError("CRITICAL: GradingAgent produced zero graded questions.")
        callback_context.state["graded_questions"] = graded
        # Preserve per-question metrics for FinalAggregator persistence
        callback_context.state["confidences"] = [
            q.get("model_confidence", 0) for q in graded
        ]
        callback_context.state["rubric_alignments"] = [
            q.get("rubric_alignment", []) for q in graded
        ]
        logger.info("GradingAgent: Successfully evaluated answers against the rubric. Handing off to RefereeAgent...")
        _log_agent_complete("GradingAgent", "graded_result")
    except json.JSONDecodeError as e:
        logger.error("CRITICAL: Failed to parse grading result: %s", e, exc_info=True)
        raise ValueError(f"CRITICAL: GradingAgent output is not valid JSON: {e}") from e


def referee_after_callback(callback_context: CallbackContext) -> None:
    raw_rep = callback_context.state.get("referee_report")
    if not raw_rep:
        callback_context.state["referee_status"] = "COMPLETED"
        return
    try:
        data = json.loads(_clean_json(raw_rep))
        callback_context.state["referee_status"] = data.get("status", "COMPLETED")
        # Preserve referee metrics for FinalAggregator persistence
        callback_context.state["low_confidence_count"] = data.get("low_confidence_count", 0)
        callback_context.state["referee_corrections"] = data.get("corrected", [])
        if data.get("status") == "PENDING_REVIEW":
            logger.warning("[REFEREE AGENT]: Low confidence detected. Flagging for teacher review (HITL).")
        else:
            logger.info("RefereeAgent complete. Results verified with high confidence.")
        _log_agent_complete("RefereeAgent", "referee_report")
    except Exception as e:
        logger.error("Error parsing referee_report: %s", e, exc_info=True)
        callback_context.state["referee_status"] = "COMPLETED"


def final_after_callback(callback_context: CallbackContext) -> None:
    _log_agent_complete("FinalAggregator", "final_payload")


def weakness_after_callback(callback_context: CallbackContext) -> None:
    raw_profile = callback_context.state.get("weakness_profile_raw")
    if not raw_profile:
        callback_context.state["weakness_profile"] = {
            "weakTopics": [],
            "classWeakTopics": [],
        }
        return
    try:
        callback_context.state["weakness_profile"] = json.loads(
            _clean_json(raw_profile)
        )
        logger.info("WeaknessDetectionAgent: Identified weak topics for student.")
        _log_agent_complete("WeaknessDetectionAgent", "weakness_profile_raw")
    except Exception as e:
        logger.error("Error parsing weakness_profile_raw: %s", e, exc_info=True)
        callback_context.state["weakness_profile"] = {
            "weakTopics": [],
            "classWeakTopics": [],
        }


def smart_prep_after_callback(callback_context: CallbackContext) -> None:
    raw_res = callback_context.state.get("practice_sessions_created_raw")
    if not raw_res:
        callback_context.state["practice_sessions_created"] = []
        return
    try:
        data = json.loads(_clean_json(raw_res))
        callback_context.state["practice_sessions_created"] = data.get(
            "practice_sessions_created", []
        )
        logger.info("SmartPrepAgent: Auto-generated personalized practice session successfully. No teacher action required.")
        _log_agent_complete("SmartPrepAgent", "practice_sessions_created_raw")
    except Exception as e:
        logger.error("Error parsing practice_sessions_created: %s", e, exc_info=True)
        callback_context.state["practice_sessions_created"] = []


def generic_callback(output_key: str) -> typing.Callable[[CallbackContext], None]:
    def callback(callback_context: CallbackContext) -> None:
        raw = callback_context.state.get(f"{output_key}_raw")
        if not raw:
            callback_context.state[output_key] = {}
            return
        try:
            data = json.loads(_clean_json(raw))
            # Merge keys directly into state for downstream agents if needed
            if isinstance(data, dict):
                for k, v in data.items():
                    callback_context.state[k] = v
            callback_context.state[output_key] = data
            _log_agent_complete("Agent", f"{output_key}_raw")
        except Exception as e:
            logger.error("Error parsing %s_raw: %s", output_key, e, exc_info=True)
            callback_context.state[output_key] = {}

    return callback

def _get_message_payload(callback_context: CallbackContext) -> dict:
    # 1. Try to get message from state or session state
    msg = callback_context.state.get("message") or callback_context.session.state.get("message")
    
    # 2. If not found in state, try user_content
    if not msg and callback_context.user_content:
        user_content = callback_context.user_content
        if isinstance(user_content, str):
            msg = user_content
        elif isinstance(user_content, dict):
            return user_content
        elif hasattr(user_content, "parts") and user_content.parts:
            texts = [part.text for part in user_content.parts if hasattr(part, "text") and part.text]
            if texts:
                msg = "".join(texts)
                
    if not msg:
        return {}
        
    if isinstance(msg, dict):
        return msg
        
    if isinstance(msg, str):
        try:
            return json.loads(msg)
        except Exception:
            pass
            
    return {}


def deterministic_mcq_grading(callback_context: CallbackContext) -> typing.Optional[typing.Any]:
    """Deterministic MCQ grading — no LLM needed for string comparison."""
    attempt_context = callback_context.state.get("attempt_context")
    if not attempt_context:
        logger.warning("MCQ grading: no attempt_context in state, falling back to LLM")
        return None  # Fall through to LLM agent

    try:
        # Parse attempt_context if it's a string
        if isinstance(attempt_context, str):
            attempt_context = json.loads(_clean_json(attempt_context))

        attempt = attempt_context.get("attempt", {})
        exam = attempt_context.get("exam", {})

        answers = attempt.get("answers", {})
        questions = exam.get("questions", [])

        mcq_results = []
        for q in questions:
            q_id = str(q.get("_id", {}).get("$oid", q.get("_id", "")))
            correct = str(q.get("correctOptionId", q.get("correctAnswer", ""))).strip().lower()
            student_answer = str(answers.get(q_id, "")).strip().lower()
            max_score = q.get("maxMarks", q.get("marks", 1))

            is_correct = student_answer == correct and student_answer != ""
            score = max_score if is_correct else 0

            mcq_results.append({
                "questionId": q_id,
                "score": score,
                "maxScore": max_score,
                "explanation": f"{'Correct' if is_correct else 'Incorrect'}. The correct answer is {q.get('correctOptionId', q.get('correctAnswer', 'N/A'))}.",
                "feedback": "Well done!" if is_correct else "Review this topic.",
            })

        # Write results to state (same key the after_callback expects)
        result_json = json.dumps({"mcq_results": mcq_results})
        callback_context.state["mcq_results_raw"] = result_json

        total = sum(r["score"] for r in mcq_results)
        total_max = sum(r["maxScore"] for r in mcq_results)
        logger.info(
            "MCQ grading completed deterministically: %d/%d (no LLM call)",
            total, total_max,
        )

        # Return Content to skip the LLM agent
        from google.genai import types as genai_types
        return genai_types.Content(parts=[genai_types.Part(text=result_json)])

    except Exception as e:
        logger.error("Deterministic MCQ grading failed, falling back to LLM: %s", e, exc_info=True)
        return None  # Fall through to LLM agent as fallback

def skip_if_extract_only(callback_context: CallbackContext) -> typing.Optional[typing.Any]:
    payload = _get_message_payload(callback_context)
    if payload.get("task") == "extract_topics_only":
        callback_context.state["skipped"] = True
        logger.info("Skipping agent because task is extract_topics_only")
        from google.genai import types as genai_types
        return genai_types.Content(parts=[genai_types.Part(text='{"skipped": true}')])
    return None

def skip_if_generate_only(callback_context: CallbackContext) -> typing.Optional[typing.Any]:
    payload = _get_message_payload(callback_context)
    if payload.get("task") == "generate_questions_only":
        callback_context.state["skipped"] = True
        logger.info("Skipping agent because task is generate_questions_only")
        from google.genai import types as genai_types
        return genai_types.Content(parts=[genai_types.Part(text='{"skipped": true}')])
    return None


ALOC_EXAM_TYPES = {"utme", "wassce", "post-utme", "neco"}


def skip_smartprep_if_not_aloc(callback_context: CallbackContext) -> typing.Optional[typing.Any]:
    """Gate SmartPrep: only run for ALOC-compatible exam types."""
    exam_type = (callback_context.state.get("exam_type") or "").lower().strip()
    if exam_type not in ALOC_EXAM_TYPES:
        callback_context.state["smartprep_skipped"] = True
        callback_context.state["smartprep_skip_reason"] = (
            f"No question bank available for exam type '{exam_type or 'unknown'}'. "
            "Weakness profile was persisted for teacher review."
        )
        logger.info(
            "SmartPrepAgent skipped: exam_type '%s' is not ALOC-compatible. "
            "Weakness data persisted for teacher insights only.",
            exam_type,
        )
        from google.genai import types as genai_types
        return genai_types.Content(parts=[genai_types.Part(text='{"skipped": true, "reason": "non-ALOC exam type"}')])
    return None


def skip_smartprep_if_unlinked(callback_context: CallbackContext) -> typing.Optional[typing.Any]:
    """Gate SmartPrep: defer for students without a linked User account."""
    linked_user_id = callback_context.state.get("linked_user_id")
    if not linked_user_id:
        callback_context.state["smartprep_skipped"] = True
        callback_context.state["smartprep_skip_reason"] = (
            "Student has no linked user account. "
            "Practice session will be generated when the student is linked."
        )
        logger.info(
            "SmartPrepAgent deferred: no linked_user_id. "
            "Practice session creation deferred until student links their account."
        )
        from google.genai import types as genai_types
        return genai_types.Content(parts=[genai_types.Part(text='{"skipped": true, "reason": "unlinked student"}')])
    return None


def skip_smartprep_gate(callback_context: CallbackContext) -> typing.Optional[typing.Any]:
    """Combined gate: checks ALOC compatibility first, then linked status."""
    result = skip_smartprep_if_not_aloc(callback_context)
    if result is not None:
        return result
    return skip_smartprep_if_unlinked(callback_context)


# ---------------------------------------------------------------------------
# Pipeline-level timing
# ---------------------------------------------------------------------------
import time


def pipeline_timing_before_callback(
    callback_context: CallbackContext,
) -> typing.Optional[typing.Any]:
    """Record pipeline start time in state."""
    callback_context.state["pipeline_start_time"] = time.time()
    return None


def pipeline_timing_after_callback(callback_context: CallbackContext) -> None:
    """Compute pipeline duration and store in state for FinalAggregator."""
    start = callback_context.state.get("pipeline_start_time")
    if start is not None:
        callback_context.state["pipeline_duration_ms"] = int(
            (time.time() - start) * 1000
        )
