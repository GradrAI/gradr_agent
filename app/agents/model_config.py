"""Gemini model choices for deployed GradrAI agents.

Agent Runtime runs the agent in a regional runtime, while Gemini 3 Flash models
are available to this project through the global Vertex AI endpoint. Production
deploys must keep GOOGLE_CLOUD_LOCATION=global so these defaults resolve.
"""

import os

LIGHTWEIGHT_GEMINI_MODEL = os.environ.get(
    "GRADR_LIGHTWEIGHT_GEMINI_MODEL", "gemini-3.1-flash-lite"
)
ADVANCED_GEMINI_MODEL = os.environ.get(
    "GRADR_ADVANCED_GEMINI_MODEL", "gemini-3.5-flash-lite"
)
