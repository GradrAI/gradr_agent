"""Gemini model choices for deployed GradrAI agents.

Agent Engine runs the agent in a regional runtime, but Gemini model availability
can differ by region and service-account access. Defaults stay on broadly
available 2.5 GA models; override only after verifying the target runtime can
invoke the replacement model.
"""

import os

LIGHTWEIGHT_GEMINI_MODEL = os.environ.get(
    "GRADR_LIGHTWEIGHT_GEMINI_MODEL", "gemini-2.5-flash-lite"
)
ADVANCED_GEMINI_MODEL = os.environ.get(
    "GRADR_ADVANCED_GEMINI_MODEL", "gemini-2.5-flash"
)
