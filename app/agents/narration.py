from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini

from app.agents.model_config import LIGHTWEIGHT_GEMINI_MODEL, retry_config
from app.prompts import FEEDBACK_NARRATION_PROMPT
from app.schemas import NarrationOutput


def create_feedback_narration_agent() -> Agent:
    """Build a fresh narration agent (an ADK agent can have only one parent)."""
    return Agent(
        name="FeedbackNarrationAgent",
        model=Gemini(model=LIGHTWEIGHT_GEMINI_MODEL, retry_options=retry_config),
        instruction=FEEDBACK_NARRATION_PROMPT,
        tools=[],
        output_schema=NarrationOutput,
        output_key="narration",
    )
