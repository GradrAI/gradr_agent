import json
import logging
import time
import typing

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

from app.app_utils.common import _clean_json, _log_agent_complete

logger = logging.getLogger(__name__)


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


def get_message_payload(callback_context: CallbackContext) -> dict:
    # 1. Try to get message from state or session state
    msg = callback_context.state.get("message") or callback_context.session.state.get(
        "message"
    )

    # 2. If not found in state, try user_content
    if not msg and callback_context.user_content:
        user_content = callback_context.user_content
        if isinstance(user_content, str):
            msg = user_content
        elif isinstance(user_content, dict):
            return user_content
        elif hasattr(user_content, "parts") and user_content.parts:
            texts = [
                part.text
                for part in user_content.parts
                if hasattr(part, "text") and part.text
            ]
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


def skip_if_extract_only(callback_context: CallbackContext) -> typing.Any | None:
    payload = get_message_payload(callback_context)
    if payload.get("task") == "extract_topics_only":
        callback_context.state["skipped"] = True
        logger.info("Skipping agent because task is extract_topics_only")
        return genai_types.Content(parts=[genai_types.Part(text='{"skipped": true}')])
    return None


def skip_if_generate_only(callback_context: CallbackContext) -> typing.Any | None:
    payload = get_message_payload(callback_context)
    if payload.get("task") == "generate_questions_only":
        callback_context.state["skipped"] = True
        logger.info("Skipping agent because task is generate_questions_only")
        return genai_types.Content(parts=[genai_types.Part(text='{"skipped": true}')])
    return None


# ---------------------------------------------------------------------------
# Pipeline-level timing
# ---------------------------------------------------------------------------
def pipeline_timing_before_callback(
    callback_context: CallbackContext,
) -> typing.Any | None:
    """Record pipeline start time in state."""
    callback_context.state["pipeline_start_time"] = time.time()
    return None


def pipeline_timing_after_callback(callback_context: CallbackContext) -> None:
    """Compute pipeline duration and store it in state."""
    start = callback_context.state.get("pipeline_start_time")
    if start is not None:
        callback_context.state["pipeline_duration_ms"] = int(
            (time.time() - start) * 1000
        )
