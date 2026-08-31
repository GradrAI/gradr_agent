"""Gemini model choices for deployed GradrAI agents.

Agent Runtime executes in a regional Agent Engine, while Gemini 3 Flash models
are available to this project through the global Vertex AI endpoint. When the
runtime is configured for Vertex AI, bare Gemini 3 model IDs are expanded to
fully qualified global model resource names so google-genai cannot prefix them
with the Agent Engine region.
"""

import os

_LIGHTWEIGHT_GEMINI_MODEL_ID = "gemini-3.1-flash-lite"
_ADVANCED_GEMINI_MODEL_ID = "gemini-3.5-flash-lite"


def _uses_vertexai() -> bool:
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"true", "1"}


def _model_project() -> str | None:
    return os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
        "GOOGLE_CLOUD_QUOTA_PROJECT"
    )


def _model_location() -> str:
    return os.environ.get("GRADR_GEMINI_MODEL_LOCATION", "global")


def _model_id(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def resolve_gemini_model(model_name: str) -> str:
    """Resolve Gemini 3 model IDs to the global Vertex model resource."""
    if model_name.startswith("projects/"):
        return model_name

    model_id = _model_id(model_name)
    project = _model_project()
    if _uses_vertexai() and project and model_id.startswith("gemini-3."):
        return (
            f"projects/{project}/locations/{_model_location()}"
            f"/publishers/google/models/{model_id}"
        )

    return model_name


LIGHTWEIGHT_GEMINI_MODEL = resolve_gemini_model(
    os.environ.get("GRADR_LIGHTWEIGHT_GEMINI_MODEL", _LIGHTWEIGHT_GEMINI_MODEL_ID)
)
ADVANCED_GEMINI_MODEL = resolve_gemini_model(
    os.environ.get("GRADR_ADVANCED_GEMINI_MODEL", _ADVANCED_GEMINI_MODEL_ID)
)
