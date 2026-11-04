from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini

from app.agents.model_config import ADVANCED_GEMINI_MODEL, retry_config
from app.agents.narration import create_feedback_narration_agent
from app.callbacks import (
    pipeline_timing_after_callback,
    pipeline_timing_before_callback,
)
from app.file_parts import attach_payload_files
from app.prompts import ESSAY_GRADING_PROMPT
from app.schemas import GradingOutput

essay_grading_agent = Agent(
    name="EssayGradingAgent",
    model=Gemini(model=ADVANCED_GEMINI_MODEL, retry_options=retry_config),
    instruction=ESSAY_GRADING_PROMPT,
    tools=[],
    output_schema=GradingOutput,
    output_key="grading",
    before_model_callback=attach_payload_files("markingGuide"),
)


class CBTPipelineAgent(SequentialAgent):
    pass


cbt_grading_pipeline: SequentialAgent = CBTPipelineAgent(
    name="CBTSubmissionGradingPipeline",
    sub_agents=[
        essay_grading_agent,
        create_feedback_narration_agent(),
    ],
    before_agent_callback=pipeline_timing_before_callback,
    after_agent_callback=pipeline_timing_after_callback,
)
