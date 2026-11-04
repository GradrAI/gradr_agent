import json
from types import SimpleNamespace

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from app.file_parts import attach_payload_files


def _context(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        state={},
        session=SimpleNamespace(state={}),
        user_content=types.Content(
            role="user", parts=[types.Part.from_text(text=json.dumps(payload))]
        ),
    )


def test_attaches_script_and_question_paper_in_order() -> None:
    payload = {
        "script": {"gcsUri": "gs://grdr/script.pdf", "mimeType": "application/pdf"},
        "questionPaper": {"gcsUri": "gs://grdr/paper.png", "mimeType": "image/png"},
    }
    request = LlmRequest()

    result = attach_payload_files("script", "questionPaper")(_context(payload), request)

    assert result is None
    assert len(request.contents) == 1
    content = request.contents[0]
    assert content.role == "user"
    assert len(content.parts) == 4
    assert content.parts[0].text == "[script]"
    assert content.parts[1].file_data.file_uri == "gs://grdr/script.pdf"
    assert content.parts[1].file_data.mime_type == "application/pdf"
    assert content.parts[2].text == "[questionPaper]"
    assert content.parts[3].file_data.file_uri == "gs://grdr/paper.png"
    assert content.parts[3].file_data.mime_type == "image/png"


def test_https_uri_is_rejected() -> None:
    payload = {
        "script": {
            "gcsUri": "https://storage.googleapis.com/grdr/script.pdf",
            "mimeType": "application/pdf",
        }
    }

    with pytest.raises(ValueError, match="Invalid file reference for 'script'"):
        attach_payload_files("script")(_context(payload), LlmRequest())


def test_absent_key_leaves_request_unchanged() -> None:
    request = LlmRequest()

    attach_payload_files("markingGuide")(_context({"essays": []}), request)

    assert request.contents == []
