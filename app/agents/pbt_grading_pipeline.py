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

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini

from app.agents.model_config import (
    ADVANCED_GEMINI_MODEL,
    LIGHTWEIGHT_GEMINI_MODEL,
    retry_config,
)
from app.agents.narration import create_feedback_narration_agent
from app.callbacks import (
    pipeline_timing_after_callback,
    pipeline_timing_before_callback,
)
from app.file_parts import attach_payload_files
from app.prompts import EXTRACTION_PROMPT, PBT_GRADING_PROMPT, REFEREE_PROMPT
from app.schemas import ExtractionOutput, GradingOutput, RefereeOutput

extraction_agent = Agent(
    name="ExtractionAgent",
    model=Gemini(model=ADVANCED_GEMINI_MODEL, retry_options=retry_config),
    instruction=EXTRACTION_PROMPT,
    tools=[],
    output_schema=ExtractionOutput,
    output_key="extraction",
    before_model_callback=attach_payload_files("script", "questionPaper"),
)

pbt_grading_agent = Agent(
    name="PBTGradingAgent",
    model=Gemini(model=ADVANCED_GEMINI_MODEL, retry_options=retry_config),
    instruction=PBT_GRADING_PROMPT,
    tools=[],
    output_schema=GradingOutput,
    output_key="grading",
    before_model_callback=attach_payload_files("markingGuide", "questionPaper"),
)

referee_agent = Agent(
    name="RefereeAgent",
    model=Gemini(model=LIGHTWEIGHT_GEMINI_MODEL, retry_options=retry_config),
    instruction=REFEREE_PROMPT,
    tools=[],
    output_schema=RefereeOutput,
    output_key="referee",
)


class PBTGradingPipelineAgent(SequentialAgent):
    """Subclass to ensure inspect.getmodule resolves to this file (ADK runner fix)."""

    pass


pbt_grading_pipeline: SequentialAgent = PBTGradingPipelineAgent(
    name="PBTGradingPipeline",
    sub_agents=[
        extraction_agent,
        pbt_grading_agent,
        referee_agent,
        create_feedback_narration_agent(),
    ],
    before_agent_callback=pipeline_timing_before_callback,
    after_agent_callback=pipeline_timing_after_callback,
)
