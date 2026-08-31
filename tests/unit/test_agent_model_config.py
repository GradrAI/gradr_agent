import os

from google.adk.models.google_llm import Gemini

from app.agents.cbt_exam_pipeline import (
    question_generation_agent,
    topic_extraction_agent,
)
from app.agents.cbt_grading_pipeline import cbt_grading_pipeline
from app.agents.model_config import (
    ADVANCED_GEMINI_MODEL,
    LIGHTWEIGHT_GEMINI_MODEL,
    resolve_gemini_model,
)
from app.agents.pbt_grading_pipeline import pbt_grading_pipeline


def _walk_agents(agent):
    yield agent
    for child in getattr(agent, "sub_agents", []) or []:
        yield from _walk_agents(child)


def _gemini_model_id(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def _is_gemini_3_model(model_name: str) -> bool:
    return _gemini_model_id(model_name).startswith("gemini-3.")


def test_extract_topics_uses_gemini_3_lightweight_default_model() -> None:
    assert isinstance(topic_extraction_agent.model, Gemini)
    assert topic_extraction_agent.model.model == LIGHTWEIGHT_GEMINI_MODEL
    assert _gemini_model_id(LIGHTWEIGHT_GEMINI_MODEL) == "gemini-3.1-flash-lite"


def test_question_generation_uses_gemini_3_advanced_default_model() -> None:
    assert isinstance(question_generation_agent.model, Gemini)
    assert question_generation_agent.model.model == ADVANCED_GEMINI_MODEL
    assert _gemini_model_id(ADVANCED_GEMINI_MODEL) == "gemini-3.5-flash-lite"


def test_deployed_pipelines_default_to_gemini_3_models() -> None:
    models = {
        child.model.model
        for pipeline in (cbt_grading_pipeline, pbt_grading_pipeline)
        for child in _walk_agents(pipeline)
        if isinstance(getattr(child, "model", None), Gemini)
    }

    assert models == {LIGHTWEIGHT_GEMINI_MODEL, ADVANCED_GEMINI_MODEL}
    assert all(_is_gemini_3_model(model) for model in models)


def test_resolve_gemini_3_model_pins_global_vertex_resource(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "True")
    monkeypatch.setenv("GRADR_GEMINI_MODEL_PROJECT", "gradr-421618")
    monkeypatch.delenv("GRADR_GEMINI_MODEL_LOCATION", raising=False)

    assert resolve_gemini_model("gemini-3.1-flash-lite") == (
        "projects/gradr-421618/locations/global/publishers/google/models/"
        "gemini-3.1-flash-lite"
    )
    assert resolve_gemini_model("models/gemini-3.5-flash-lite") == (
        "projects/gradr-421618/locations/global/publishers/google/models/"
        "gemini-3.5-flash-lite"
    )
    assert resolve_gemini_model("publishers/google/models/gemini-3.5-flash-lite") == (
        "projects/gradr-421618/locations/global/publishers/google/models/"
        "gemini-3.5-flash-lite"
    )


def test_resolve_gemini_model_preserves_explicit_resource(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "True")
    monkeypatch.setenv("GRADR_GEMINI_MODEL_PROJECT", "gradr-421618")

    model = "projects/example/locations/global/publishers/google/models/gemini-3.1-flash-lite"
    assert resolve_gemini_model(model) == model


def test_agent_engine_app_configures_global_genai_env(monkeypatch) -> None:
    from app.agent_engine_app import configure_gemini_vertex_env

    monkeypatch.setenv("GRADR_GEMINI_MODEL_LOCATION", "global")
    monkeypatch.setenv("GRADR_GEMINI_MODEL_PROJECT", "gradr-421618")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    configure_gemini_vertex_env()

    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "global"
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "gradr-421618"
