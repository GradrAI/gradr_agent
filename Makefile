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

# Gemini 3 Flash models are available to this project via the global endpoint;
# Agent Runtime itself remains deployed in us-central1.
GRADR_LIGHTWEIGHT_GEMINI_MODEL ?= gemini-3.1-flash-lite
GRADR_ADVANCED_GEMINI_MODEL ?= gemini-3.5-flash-lite
AGENT_RUNTIME_ENV = GOOGLE_CLOUD_LOCATION=global,GOOGLE_GENAI_USE_VERTEXAI=True,PIPELINE_TIMEOUT_SECONDS=900,GRADR_LIGHTWEIGHT_GEMINI_MODEL=$(GRADR_LIGHTWEIGHT_GEMINI_MODEL),GRADR_ADVANCED_GEMINI_MODEL=$(GRADR_ADVANCED_GEMINI_MODEL)

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
		--set-env-vars='MDB_MCP_CONNECTION_STRING=$(MDB_MCP_CONNECTION_STRING),$(AGENT_RUNTIME_ENV)' \
		--requirements-file=app/app_utils/.requirements.txt

# Deploy CBT Grading Agent
deploy-cbt-grading: check-mdb export-reqs
	@uv run -m app.app_utils.deploy \
		--source-packages=./app \
		--entrypoint-module=app.agent_engine_app \
		--entrypoint-object=cbt_grading_engine \
		--display-name="gradr-cbt-grading-agent" \
		--set-env-vars='MDB_MCP_CONNECTION_STRING=$(MDB_MCP_CONNECTION_STRING),$(AGENT_RUNTIME_ENV)' \
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

INTRO_PYTHON_DIR ?= /Users/macbookair/Downloads/GRADRAI_Introduction_To_Programming_Using_Python
INTRO_PYTHON_SCORES ?= /Users/macbookair/Downloads/Introduction_to_Programming_Using_Python_Student_Scores.pdf
INTRO_PYTHON_OUT ?= eval/runs/intro-python-smoke
INTRO_PYTHON_SCRIPT_IDS ?= 1,2
INTRO_PYTHON_MODEL ?= gemini-3.5-flash-lite
INTRO_PYTHON_LECTURE_1 ?= /Users/macbookair/Downloads/Algorithm and flowcharts.pdf
INTRO_PYTHON_LECTURE_2 ?= /Users/macbookair/Downloads/Foundations_of_Programming_Upgraded (1).pptx
INTRO_PYTHON_LECTURE_3 ?= /Users/macbookair/Downloads/lesson1-introduction to programming.pdf
INTRO_PYTHON_RUBRIC ?= eval/rubric_corrected_intro_python.json
INTRO_PYTHON_STAGED_OUT ?= eval/runs/intro-python-staged-corrected
INTRO_PYTHON_ORIGINAL_SCORE_SHEET ?= eval/runs/intro-python-full-global-ablation-20260826T124102Z/paper_tables/per_question_labels.original.csv
INTRO_PYTHON_GRADER2_SCORE_SHEET ?= eval/runs/intro-python-full-global-ablation-20260826T124102Z/paper_tables/per_question_labels.grader2.csv

INTRO_PYTHON_LECTURE_ARGS = --lecture-material "$(INTRO_PYTHON_LECTURE_1)" \
	--lecture-material "$(INTRO_PYTHON_LECTURE_2)" \
	--lecture-material "$(INTRO_PYTHON_LECTURE_3)"

eval-intro-python-dry-run:
	PYTHONPATH=$(PWD) uv run --extra eval python -m eval.intro_python_experiment \
		--scripts-dir "$(INTRO_PYTHON_DIR)" \
		--scores-pdf "$(INTRO_PYTHON_SCORES)" \
		$(INTRO_PYTHON_LECTURE_ARGS) \
		--script-ids "$(INTRO_PYTHON_SCRIPT_IDS)" \
		--out-dir "$(INTRO_PYTHON_OUT)" \
		--model "$(INTRO_PYTHON_MODEL)" \
		--lecture-mode summary \
		--dry-run --overwrite

eval-intro-python-smoke:
	PYTHONPATH=$(PWD) uv run --extra eval python -m eval.intro_python_experiment \
		--scripts-dir "$(INTRO_PYTHON_DIR)" \
		--scores-pdf "$(INTRO_PYTHON_SCORES)" \
		$(INTRO_PYTHON_LECTURE_ARGS) \
		--script-ids "$(INTRO_PYTHON_SCRIPT_IDS)" \
		--out-dir "$(INTRO_PYTHON_OUT)" \
		--model "$(INTRO_PYTHON_MODEL)" \
		--lecture-mode summary \
		--overwrite

eval-intro-python-ablation:
	PYTHONPATH=$(PWD) uv run --extra eval python -m eval.intro_python_experiment \
		--scripts-dir "$(INTRO_PYTHON_DIR)" \
		--scores-pdf "$(INTRO_PYTHON_SCORES)" \
		$(INTRO_PYTHON_LECTURE_ARGS) \
		--script-ids "$(INTRO_PYTHON_SCRIPT_IDS)" \
		--out-dir "$(INTRO_PYTHON_OUT)" \
		--model "$(INTRO_PYTHON_MODEL)" \
		--lecture-mode summary \
		--run-lecture-ablation --overwrite

eval-intro-python-metrics:
	PYTHONPATH=$(PWD) uv run --extra eval python -m eval.compute_metrics \
		--scores-pdf "$(INTRO_PYTHON_SCORES)" \
		--predictions "$(INTRO_PYTHON_OUT)/predictions.jsonl" \
		--output "$(INTRO_PYTHON_OUT)/metrics-recomputed.json"

eval-intro-python-staged-dry-run:
	PYTHONPATH=$(PWD) GOOGLE_CLOUD_LOCATION=global uv run --extra eval python -m eval.staged_intro_python_experiment \
		--scripts-dir "$(INTRO_PYTHON_DIR)" \
		--scores-pdf "$(INTRO_PYTHON_SCORES)" \
		$(INTRO_PYTHON_LECTURE_ARGS) \
		--script-ids "$(INTRO_PYTHON_SCRIPT_IDS)" \
		--rubric-json "$(INTRO_PYTHON_RUBRIC)" \
		--out-dir "$(INTRO_PYTHON_STAGED_OUT)-smoke" \
		--model "$(INTRO_PYTHON_MODEL)" \
		--lecture-mode summary \
		--dry-run --overwrite

eval-intro-python-staged-smoke:
	PYTHONPATH=$(PWD) GOOGLE_CLOUD_PROJECT=gradr-421618 GOOGLE_CLOUD_LOCATION=global uv run --extra eval python -m eval.staged_intro_python_experiment \
		--scripts-dir "$(INTRO_PYTHON_DIR)" \
		--scores-pdf "$(INTRO_PYTHON_SCORES)" \
		$(INTRO_PYTHON_LECTURE_ARGS) \
		--script-ids "3,11,30,53" \
		--rubric-json "$(INTRO_PYTHON_RUBRIC)" \
		--out-dir "$(INTRO_PYTHON_STAGED_OUT)-smoke" \
		--model "$(INTRO_PYTHON_MODEL)" \
		--lecture-mode summary \
		--run-lecture-ablation --overwrite

eval-intro-python-staged-ablation:
	PYTHONPATH=$(PWD) GOOGLE_CLOUD_PROJECT=gradr-421618 GOOGLE_CLOUD_LOCATION=global uv run --extra eval python -m eval.staged_intro_python_experiment \
		--scripts-dir "$(INTRO_PYTHON_DIR)" \
		--scores-pdf "$(INTRO_PYTHON_SCORES)" \
		$(INTRO_PYTHON_LECTURE_ARGS) \
		--rubric-json "$(INTRO_PYTHON_RUBRIC)" \
		--out-dir "$(INTRO_PYTHON_STAGED_OUT)" \
		--model "$(INTRO_PYTHON_MODEL)" \
		--lecture-mode summary \
		--run-lecture-ablation --allow-full-run-after-review \
		--approval-note "Corrected rubric staged full rerun with saved extraction artifacts, best-four equal-weighted finalization, and lecture/no-lecture ablation." \
		--overwrite

eval-intro-python-staged-metrics:
	PYTHONPATH=$(PWD) uv run --extra eval python -m eval.compute_metrics \
		--corrected-staged \
		--predictions-no-lecture "$(INTRO_PYTHON_STAGED_OUT)/predictions.no_lecture.jsonl" \
		--predictions-lecture "$(INTRO_PYTHON_STAGED_OUT)/predictions.lecture.jsonl" \
		--extractions "$(INTRO_PYTHON_STAGED_OUT)/extractions.jsonl" \
		--rubric-json "$(INTRO_PYTHON_RUBRIC)" \
		--original-score-sheet "$(INTRO_PYTHON_ORIGINAL_SCORE_SHEET)" \
		--grader2-score-sheet "$(INTRO_PYTHON_GRADER2_SCORE_SHEET)" \
		--write-paper-tables "$(INTRO_PYTHON_STAGED_OUT)/paper_tables" \
		--output "$(INTRO_PYTHON_STAGED_OUT)/metrics.corrected_staged.json"