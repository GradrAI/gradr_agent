# ==============================================================================
# Installation & Setup
# ==============================================================================

# Install dependencies using uv package manager
install:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/0.8.13/install.sh | sh; source $HOME/.local/bin/env; }
	uv sync

# ==============================================================================
# Playground Targets
# ==============================================================================

# Launch local dev playground
playground:
	@echo "==============================================================================="
	@echo "| 🚀 Starting your agent playground...                                        |"
	@echo "|                                                                             |"
	@echo "| 💡 Try asking: What's the weather in San Francisco?                         |"
	@echo "|                                                                             |"
	@echo "| 🔍 IMPORTANT: Select the agent you want to interact with from the dropdown! |"
	@echo "==============================================================================="
	PYTHONPATH=$(PWD) uv run adk web local_playground --port 8501 --reload_agents

# ==============================================================================
# Backend Deployment Targets
# ==============================================================================

# The MongoDB MCP subprocess is configured purely from this environment
# variable (see app/toolsets.py). It is NOT inherited by the deployed Agent
# Engine runtime, so it must be passed explicitly at deploy time — without it
# the MCP starts with an empty connection string and every MongoDB tool call
# in the grading pipelines silently fails.
MDB_MCP_CONNECTION_STRING ?= $(shell sed -n 's/^MDB_MCP_CONNECTION_STRING=//p' .env 2>/dev/null | tr -d "\"'" | head -1)

# Fail loudly rather than shipping an agent that cannot reach the database.
check-mdb:
	@test -n '$(MDB_MCP_CONNECTION_STRING)' || { \
		echo "ERROR: MDB_MCP_CONNECTION_STRING is empty."; \
		echo "Set it in gradr_agent/.env or pass MDB_MCP_CONNECTION_STRING=... to make."; \
		exit 1; }
	@echo "MongoDB MCP target: $$(echo '$(MDB_MCP_CONNECTION_STRING)' | sed 's|//[^@]*@|//***@|')"

# Generate requirements.txt
export-reqs:
	@(uv export --no-hashes --no-header --no-dev --no-emit-project --no-annotate > app/app_utils/.requirements.txt 2>/dev/null || \
	uv export --no-hashes --no-header --no-dev --no-emit-project > app/app_utils/.requirements.txt)

# Deploy PBT Grading Agent
# Recipe is prefixed with @ so make does not echo the connection string
# (which carries database credentials) into build logs.
deploy-pbt: check-mdb export-reqs
	@uv run -m app.app_utils.deploy \
		--source-packages=./app \
		--entrypoint-module=app.agent_engine_app \
		--entrypoint-object=pbt_pipeline_engine \
		--display-name="gradr-pbt-agent" \
		--set-env-vars='MDB_MCP_CONNECTION_STRING=$(MDB_MCP_CONNECTION_STRING)' \
		--requirements-file=app/app_utils/.requirements.txt

# Deploy CBT Grading Agent
deploy-cbt-grading: check-mdb export-reqs
	@uv run -m app.app_utils.deploy \
		--source-packages=./app \
		--entrypoint-module=app.agent_engine_app \
		--entrypoint-object=cbt_grading_engine \
		--display-name="gradr-cbt-grading-agent" \
		--set-env-vars='MDB_MCP_CONNECTION_STRING=$(MDB_MCP_CONNECTION_STRING)' \
		--requirements-file=app/app_utils/.requirements.txt

# Deploy CBT Exam Generation Agent
# Deliberately NOT given MDB_MCP_CONNECTION_STRING: this pipeline
# (TopicExtractionAgent -> QuestionGenerationAgent) touches no MongoDB, so it
# has no reason to hold database credentials.
deploy-cbt-exam: export-reqs
	uv run -m app.app_utils.deploy \
		--source-packages=./app \
		--entrypoint-module=app.agent_engine_app \
		--entrypoint-object=cbt_exam_engine \
		--display-name="gradr-cbt-exam-agent" \
		--requirements-file=app/app_utils/.requirements.txt

# Deploy all agents
deploy-all: deploy-pbt deploy-cbt-grading deploy-cbt-exam


# ==============================================================================
# Infrastructure Setup
# ==============================================================================

# Set up development environment resources using Terraform
setup-dev-env:
	PROJECT_ID=$$(gcloud config get-value project) && \
	(cd deployment/terraform/dev && terraform init && terraform apply --var-file vars/env.tfvars --var dev_project_id=$$PROJECT_ID --auto-approve)

# ==============================================================================
# Testing & Code Quality
# ==============================================================================

# Run unit and integration tests
test:
	uv sync --dev
	uv run pytest tests/unit && uv run pytest tests/integration

# Run code quality checks (codespell, ruff, mypy)
lint:
	uv sync --dev --extra lint
	uv run codespell
	uv run ruff check . --diff
	uv run ruff format . --check --diff
	uv run mypy .

# ==============================================================================
# Gemini Enterprise Integration
# ==============================================================================

# Register the deployed agent to Gemini Enterprise
# Usage: make register-gemini-enterprise (interactive - will prompt for required details)
# For non-interactive use, set env vars: ID or GEMINI_ENTERPRISE_APP_ID (full GE resource name)
# Optional env vars: GEMINI_DISPLAY_NAME, GEMINI_DESCRIPTION, GEMINI_TOOL_DESCRIPTION, AGENT_ENGINE_ID
register-gemini-enterprise:
	@uvx agent-starter-pack@0.21.0 register-gemini-enterprise

# ==============================================================================
# Evaluation
# ==============================================================================

# Run grading accuracy evaluation against gold-standard data
eval-accuracy:
	uv run python -m eval.compute_metrics --gold eval/gold_standard.json --output eval/results.json