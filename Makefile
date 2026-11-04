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

# Gemini 3 Flash models are available to this project via the global endpoint;
# Agent Runtime itself remains deployed in us-central1.
GEMINI_LOCATION ?= global
GOOGLE_GENAI_USE_VERTEXAI ?= True
PIPELINE_TIMEOUT_SECONDS ?= 900
GRADR_LIGHTWEIGHT_GEMINI_MODEL ?= gemini-3.1-flash-lite
GRADR_ADVANCED_GEMINI_MODEL ?= gemini-3.5-flash-lite
AGENT_RUNTIME_ENV = GOOGLE_CLOUD_LOCATION=$(GEMINI_LOCATION),GRADR_GEMINI_MODEL_LOCATION=$(GEMINI_LOCATION),GOOGLE_GENAI_USE_VERTEXAI=$(GOOGLE_GENAI_USE_VERTEXAI),PIPELINE_TIMEOUT_SECONDS=$(PIPELINE_TIMEOUT_SECONDS),GRADR_LIGHTWEIGHT_GEMINI_MODEL=$(GRADR_LIGHTWEIGHT_GEMINI_MODEL),GRADR_ADVANCED_GEMINI_MODEL=$(GRADR_ADVANCED_GEMINI_MODEL)

# Generate requirements.txt
export-reqs:
	@(uv export --no-hashes --no-header --no-dev --no-emit-project --no-annotate > app/app_utils/.requirements.txt 2>/dev/null || \
	uv export --no-hashes --no-header --no-dev --no-emit-project > app/app_utils/.requirements.txt)

# Deploy PBT Grading Agent
deploy-pbt: export-reqs
	uv run -m app.app_utils.deploy \
		--source-packages=./app \
		--entrypoint-module=app.agent_engine_app \
		--entrypoint-object=pbt_pipeline_engine \
		--display-name="gradr-pbt-agent" \
		--set-env-vars='$(AGENT_RUNTIME_ENV)' \
		--requirements-file=app/app_utils/.requirements.txt

# Deploy CBT Grading Agent
deploy-cbt-grading: export-reqs
	uv run -m app.app_utils.deploy \
		--source-packages=./app \
		--entrypoint-module=app.agent_engine_app \
		--entrypoint-object=cbt_grading_engine \
		--display-name="gradr-cbt-grading-agent" \
		--set-env-vars='$(AGENT_RUNTIME_ENV)' \
		--requirements-file=app/app_utils/.requirements.txt

# Deploy CBT Exam Generation Agent
deploy-cbt-exam: export-reqs
	uv run -m app.app_utils.deploy \
		--source-packages=./app \
		--entrypoint-module=app.agent_engine_app \
		--entrypoint-object=cbt_exam_engine \
		--display-name="gradr-cbt-exam-agent" \
		--set-env-vars='$(AGENT_RUNTIME_ENV)' \
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