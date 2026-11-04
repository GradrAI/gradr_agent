import pytest

from app.agents.cbt_grading_pipeline import cbt_grading_pipeline
from app.agents.pbt_grading_pipeline import pbt_grading_pipeline


def test_cbt_grading_pipeline_sub_agents() -> None:
    assert [agent.name for agent in cbt_grading_pipeline.sub_agents] == [
        "EssayGradingAgent",
        "FeedbackNarrationAgent",
    ]


def test_pbt_grading_pipeline_sub_agents() -> None:
    assert [agent.name for agent in pbt_grading_pipeline.sub_agents] == [
        "ExtractionAgent",
        "PBTGradingAgent",
        "RefereeAgent",
        "FeedbackNarrationAgent",
    ]


@pytest.mark.parametrize("pipeline", [cbt_grading_pipeline, pbt_grading_pipeline])
def test_grading_sub_agents_are_tool_free_with_schemas(pipeline) -> None:
    for agent in pipeline.sub_agents:
        assert agent.tools == [], agent.name
        assert agent.output_schema is not None, agent.name
