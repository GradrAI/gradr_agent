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

# mypy: disable-error-code="attr-defined,arg-type"
import logging
import os
import asyncio
from typing import Any

import vertexai
from google.adk.apps.app import App
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.cloud import logging as google_cloud_logging
from vertexai.agent_engines.templates.adk import AdkApp

from app.agents import (
    cbt_exam_generation_pipeline,
    cbt_grading_pipeline,
    pbt_grading_pipeline,
)
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback

logger = logging.getLogger(__name__)

pbt_grading_app = App(root_agent=pbt_grading_pipeline, name="pbt_grading_app")
cbt_grading_app = App(root_agent=cbt_grading_pipeline, name="cbt_grading_app")
cbt_exam_app = App(root_agent=cbt_exam_generation_pipeline, name="cbt_exam_app")


class AgentEngineApp(AdkApp):
    def set_up(self) -> None:
        """Initialize the agent engine app with logging and telemetry."""
        vertexai.init()
        setup_telemetry()
        super().set_up()
        logging.basicConfig(level=logging.INFO)
        logging_client = google_cloud_logging.Client()
        self.logger = logging_client.logger(__name__)
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        if location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = location

    def register_feedback(self, feedback: dict[str, Any]) -> None:
        """Collect and log feedback."""
        feedback_obj = Feedback.model_validate(feedback)
        self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

    async def query(
        self,
        *,
        message: str | dict[str, Any],
        user_id: str = "default_user",
        session_id: str | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> Any:
        """Unary query method for standard HTTP clients."""
        # Pipeline timeout: finite wall-clock cap for a complete grading run.
        pipeline_timeout = int(os.environ.get("PIPELINE_TIMEOUT_SECONDS", 900))
        events = []
        try:
            async with asyncio.timeout(pipeline_timeout):
                async for event in self.async_stream_query(
                    message=message,
                    user_id=user_id,
                    session_id=session_id,
                    run_config=run_config,
                ):
                    events.append(event)
        except TimeoutError:
            logger.error(
                "Pipeline timed out after %d seconds for session %s",
                pipeline_timeout, session_id,
            )
            return {
                "error": f"Pipeline timed out after {pipeline_timeout} seconds",
                "partial_events": len(events),
            }

        if not events:
            return {}

        # Accumulate all state deltas from the event stream
        accumulated_state_delta = {}
        for event in events:
            if isinstance(event, dict) and event.get("actions", {}).get("state_delta"):
                accumulated_state_delta.update(event["actions"]["state_delta"])

        final_event = dict(events[-1])
        if "actions" not in final_event:
            final_event["actions"] = {}
        else:
            final_event["actions"] = dict(final_event["actions"])
        final_event["actions"]["state_delta"] = accumulated_state_delta
        return final_event


    def register_operations(self) -> dict[str, list[str]]:
        """Registers the operations of the Agent."""
        operations = super().register_operations()
        operations[""] = [*operations.get("", []), "register_feedback"]
        operations["async"] = [*operations.get("async", []), "query"]
        return operations


logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")
artifact_service = (
    GcsArtifactService(bucket_name=logs_bucket_name)
    if logs_bucket_name
    else InMemoryArtifactService()
)

pbt_pipeline_engine = AgentEngineApp(
    app=pbt_grading_app,
    artifact_service_builder=lambda: artifact_service,
)

cbt_grading_engine = AgentEngineApp(
    app=cbt_grading_app,
    artifact_service_builder=lambda: artifact_service,
)

cbt_exam_engine = AgentEngineApp(
    app=cbt_exam_app,
    artifact_service_builder=lambda: artifact_service,
)
