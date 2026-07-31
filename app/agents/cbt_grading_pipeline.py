import logging

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini

from app.agents.shared_agents import (
    create_smart_prep_agent,
    create_weakness_detection_agent,
)
from app.callbacks import (
    generic_callback,
    deterministic_mcq_grading,
    pipeline_timing_before_callback,
    pipeline_timing_after_callback,
)
from app.prompts import (
    ATTEMPT_RETRIEVAL_PROMPT,
    ESSAY_GRADING_PROMPT,
    FEEDBACK_NARRATION_PROMPT,
    MCQ_GRADING_PROMPT,
    RESULT_PERSISTENCE_PROMPT,
)
from app.toolsets import mongo_mcp_toolset, retry_config

logger = logging.getLogger(__name__)


attempt_retrieval_agent = Agent(
    name="AttemptRetrievalAgent",
    model=Gemini(model="gemini-3.1-flash-lite", retry_options=retry_config),
    instruction=ATTEMPT_RETRIEVAL_PROMPT,
    tools=[mongo_mcp_toolset],
    output_key="attempt_context_raw",
    after_agent_callback=generic_callback("attempt_context"),
)

mcq_grading_agent = Agent(
    name="MCQGradingAgent",
    model=Gemini(model="gemini-3.1-flash-lite", retry_options=retry_config),
    instruction=MCQ_GRADING_PROMPT,
    output_key="mcq_results_raw",
    before_agent_callback=deterministic_mcq_grading,
    after_agent_callback=generic_callback("mcq_results_group"),
)

essay_grading_agent = Agent(
    name="EssayGradingAgent",
    model=Gemini(model="gemini-3.5-flash-lite", retry_options=retry_config),
    instruction=ESSAY_GRADING_PROMPT,
    tools=[mongo_mcp_toolset],
    output_key="essay_results_raw",
    after_agent_callback=generic_callback("essay_results_group"),
)

feedback_narration_agent = Agent(
    name="FeedbackNarrationAgent",
    model=Gemini(model="gemini-3.1-flash-lite", retry_options=retry_config),
    instruction=FEEDBACK_NARRATION_PROMPT,
    output_key="grading_summary_raw",
    after_agent_callback=generic_callback("grading_summary"),
)

result_persistence_agent = Agent(
    name="ResultPersistenceAgent",
    model=Gemini(model="gemini-3.1-flash-lite", retry_options=retry_config),
    instruction=RESULT_PERSISTENCE_PROMPT,
    tools=[mongo_mcp_toolset],
    output_key="final_grading_payload_raw",
    after_agent_callback=generic_callback("final_grading_payload"),
)


class CBTPipelineAgent(SequentialAgent):
    pass


cbt_grading_pipeline: SequentialAgent = CBTPipelineAgent(
    name="CBTSubmissionGradingPipeline",
    sub_agents=[
        attempt_retrieval_agent,
        mcq_grading_agent,
        essay_grading_agent,
        feedback_narration_agent,
        create_weakness_detection_agent(),
        create_smart_prep_agent(),
        result_persistence_agent,
    ],
    before_agent_callback=pipeline_timing_before_callback,
    after_agent_callback=pipeline_timing_after_callback,
)
