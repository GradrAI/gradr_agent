from google.adk.models.google_llm import Gemini

from app.agents.cbt_exam_pipeline import (
    question_generation_agent,
    topic_extraction_agent,
)
from app.agents.cbt_grading_pipeline import cbt_grading_pipeline
from app.agents.model_config import ADVANCED_GEMINI_MODEL, LIGHTWEIGHT_GEMINI_MODEL
from app.agents.pbt_grading_pipeline import pbt_grading_pipeline


def _walk_agents(agent):
    yield agent
    for child in getattr(agent, "sub_agents", []) or []:
        yield from _walk_agents(child)


def test_extract_topics_uses_gemini_3_lightweight_default_model() -> None:
    assert isinstance(topic_extraction_agent.model, Gemini)
    assert topic_extraction_agent.model.model == LIGHTWEIGHT_GEMINI_MODEL
    assert LIGHTWEIGHT_GEMINI_MODEL == "gemini-3.1-flash-lite"


def test_question_generation_uses_gemini_3_advanced_default_model() -> None:
    assert isinstance(question_generation_agent.model, Gemini)
    assert question_generation_agent.model.model == ADVANCED_GEMINI_MODEL
    assert ADVANCED_GEMINI_MODEL == "gemini-3.5-flash-lite"


def test_deployed_pipelines_default_to_gemini_3_models() -> None:
    models = {
        child.model.model
        for pipeline in (cbt_grading_pipeline, pbt_grading_pipeline)
        for child in _walk_agents(pipeline)
        if isinstance(getattr(child, "model", None), Gemini)
    }

    assert models == {LIGHTWEIGHT_GEMINI_MODEL, ADVANCED_GEMINI_MODEL}
    assert all(model.startswith("gemini-3") for model in models)
