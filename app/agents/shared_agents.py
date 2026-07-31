import logging

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini

from app.callbacks import smart_prep_after_callback, weakness_after_callback, skip_smartprep_gate
from app.prompts import SMART_PREP_PROMPT, WEAKNESS_PROMPT
from app.toolsets import custom_mcp_toolset, mongo_mcp_toolset, retry_config

logger = logging.getLogger(__name__)


def create_weakness_detection_agent() -> Agent:
    return Agent(
        name="WeaknessDetectionAgent",
        model=Gemini(model="gemini-3.1-flash-lite", retry_options=retry_config),
        instruction=WEAKNESS_PROMPT,
        tools=[mongo_mcp_toolset],
        output_key="weakness_profile_raw",
        after_agent_callback=weakness_after_callback,
    )


def create_smart_prep_agent() -> Agent:
    return Agent(
        name="SmartPrepAgent",
        model=Gemini(model="gemini-3.1-flash-lite", retry_options=retry_config),
        instruction=SMART_PREP_PROMPT,
        tools=[custom_mcp_toolset, mongo_mcp_toolset],
        output_key="practice_sessions_created_raw",
        before_agent_callback=skip_smartprep_gate,
        after_agent_callback=smart_prep_after_callback,
    )
