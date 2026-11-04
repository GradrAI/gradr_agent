"""Attach GCS files named in the JSON message to a model request.

The grading pipelines have no tools, so the backend sends file references as
``{"gcsUri": "gs://...", "mimeType": "..."}`` in the message payload and this
callback turns them into labelled URI parts on the model request.
"""

import typing

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from app.callbacks import get_message_payload


def _file_part(key: str, ref: typing.Any) -> types.Part:
    if not isinstance(ref, dict):
        raise ValueError(f"Invalid file reference for '{key}'")
    uri = ref.get("gcsUri")
    mime_type = ref.get("mimeType")
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError(f"Invalid file reference for '{key}'")
    if not isinstance(mime_type, str) or not mime_type:
        raise ValueError(f"Invalid file reference for '{key}'")
    return types.Part.from_uri(file_uri=uri, mime_type=mime_type)


def attach_payload_files(
    *keys: str,
) -> typing.Callable[[CallbackContext, LlmRequest], None]:
    """Build a before_model_callback attaching the payload files under ``keys``.

    Absent or null keys are skipped; malformed references raise ``ValueError``.
    """

    def callback(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        payload = get_message_payload(callback_context)
        parts: list[types.Part] = []
        for key in keys:
            ref = payload.get(key)
            if ref is None:
                continue
            file_part = _file_part(key, ref)
            parts.append(types.Part.from_text(text=f"[{key}]"))
            parts.append(file_part)
        if parts:
            llm_request.contents.append(types.Content(role="user", parts=parts))
        return None

    return callback
