# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool, google_search

from app.agents.shared_agents import (
    create_smart_prep_agent,
    create_weakness_detection_agent,
)
from app.callbacks import (
    final_after_callback,
    grading_after_callback,
    pipeline_timing_after_callback,
    pipeline_timing_before_callback,
    preprocessing_after_callback,
    referee_after_callback,
)
from app.prompts import (
    FINAL_AGGREGATOR_PROMPT,
    GRADER_PROMPT_BASE,
    ONLINE_ANSWERS_PROMPT,
    PREPROCESSING_PROMPT,
    REFEREE_PROMPT,
    SUMMARIZER_PROMPT,
)
from app.toolsets import (
    custom_mcp_toolset,
    gcs_mcp_toolset,
    mongo_mcp_toolset,
    retry_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------
preprocessing_agent = Agent(
    name="PreprocessingAgent",
    model=Gemini(model="gemini-3.5-flash-lite", retry_options=retry_config),
    instruction=PREPROCESSING_PROMPT,
    tools=[custom_mcp_toolset, mongo_mcp_toolset, gcs_mcp_toolset],
    output_key="preprocessing_context",
    after_agent_callback=preprocessing_after_callback,
)

online_answers_agent = Agent(
    name="OnlineAnswersAgent",
    model=Gemini(model="gemini-3.1-flash-lite", retry_options=retry_config),
    instruction=ONLINE_ANSWERS_PROMPT,
    tools=[google_search],
    output_key="online_answers",
)

summarizer_agent = Agent(
    name="SummarizerAgent",
    model=Gemini(model="gemini-3.1-flash-lite", retry_options=retry_config),
    instruction=SUMMARIZER_PROMPT,
    output_key="final_summary",
)

grading_agent = Agent(
    name="GradingAgent",
    model=Gemini(model="gemini-3.5-flash-lite", retry_options=retry_config),
    instruction=GRADER_PROMPT_BASE,
    tools=[AgentTool(online_answers_agent), AgentTool(summarizer_agent)],
    sub_agents=[online_answers_agent, summarizer_agent],
    output_key="graded_result",
    after_agent_callback=grading_after_callback,
)

referee_agent = Agent(
    name="RefereeAgent",
    model=Gemini(model="gemini-3.5-flash-lite", retry_options=retry_config),
    instruction=REFEREE_PROMPT,
    tools=[mongo_mcp_toolset],
    output_key="referee_report",
    after_agent_callback=referee_after_callback,
)

final_aggregator = Agent(
    name="FinalAggregator",
    model=Gemini(model="gemini-3.1-flash-lite", retry_options=retry_config),
    instruction=FINAL_AGGREGATOR_PROMPT,
    tools=[mongo_mcp_toolset],
    output_key="final_payload",
    after_agent_callback=final_after_callback,
)


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------
class PBTGradingPipelineAgent(SequentialAgent):
    """Subclass to ensure inspect.getmodule resolves to this file (ADK runner fix)."""

    pass


pbt_grading_pipeline: SequentialAgent = PBTGradingPipelineAgent(
    name="PBTGradingPipeline",
    sub_agents=[
        preprocessing_agent,
        grading_agent,
        referee_agent,
        create_weakness_detection_agent(),
        create_smart_prep_agent(),
        final_aggregator,
    ],
    before_agent_callback=pipeline_timing_before_callback,
    after_agent_callback=pipeline_timing_after_callback,
)
