# Gradr-Agent

An AI-powered grading agent built with Google's Agent Development Kit (ADK) and optimized for automated evaluation of handwritten and typed student submissions. This project extends the base ReAct agent from [googleCloudPlatform/agent-starter-pack](https://github.com/GoogleCloudPlatform/agent-starter-pack) (v0.21.0) and integrates tightly with Google Cloud (Vertex AI, BigQuery, Cloud Storage) to deliver high-accuracy automated grading.

Gradr-Agent is part of the broader GradrAI system — a platform for automating grading of paper-based tests using advanced LLMs, prompt orchestration, evaluators and domain expert feedback.

## Table of Contents

- [System Architecture](#system-architecture)
- [Overview](#overview)
- [Agent Pipelines](#agent-pipelines)
- [Features](#features)
- [Project Structure](#project-structure)
- [Deployed Endpoints](#deployed-endpoints)
- [Local Setup (for Hackathon Judges)](#local-setup-for-hackathon-judges)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Evaluation & Quality Assurance](#evaluation--quality-assurance)
- [Monitoring & Observability](#monitoring--observability)

## System Architecture

The full GradrAI platform consists of three services. The frontend and backend are closed-source; the agent repo is provided for review.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         GradrAI Platform                                │
│                                                                         │
│  ┌───────────────┐       ┌───────────────────┐                          │
│  │   Frontend     │──────▶│     Backend        │                         │
│  │  (React/Vite)  │ REST  │  (Node.js/Express) │                         │
│  │  gradrai.com   │◀──────│  Cloud Run         │                         │
│  └───────────────┘       └────────┬──────────┘                          │
│                                   │                                      │
│                    Vertex AI Reasoning Engine API                        │
│                    (authenticated via Google Auth)                       │
│                                   │                                      │
│                                   ▼                                      │
│                   ┌───────────────────────────┐                          │
│                   │     Gradr-Agent (ADK)      │  ◀── This repo          │
│                   │  Deployed on Vertex AI     │                         │
│                   │  Reasoning Engine          │                         │
│                   │  Tool-free, JSON in/out    │                         │
│                   └─────────────┬─────────────┘                          │
│                                 │ gs:// file parts                       │
│                                 ▼                                        │
│                   ┌───────────────────────────┐                          │
│                   │ Gemini on Vertex AI        │                         │
│                   │ (reads scripts and guides  │                         │
│                   │  from Cloud Storage)       │                         │
│                   └───────────────────────────┘                          │
│                                                                         │
│  Supporting Infrastructure:                                             │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  ┌───────────────┐    │
│  │ BigQuery │  │ Cloud Trace  │  │ Cloud       │  │ Redis (BullMQ)│    │
│  │Telemetry │  │ (Waterfall)  │  │ Logging     │  │ Job Queues    │    │
│  └──────────┘  └──────────────┘  └─────────────┘  └───────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Request Flow (Paper-Based Test Grading)

1. **Educator** uploads question paper, marking guide, and student scripts via the **frontend** (`gradrai.com`).
2. **Backend** enqueues a grading job via BullMQ (Redis), loads everything the agent needs, and calls the **Vertex AI Reasoning Engine** `:query` API with one self-contained JSON message (see [Message Contracts](#message-contracts)).
3. **Gradr-Agent** runs a sequential pipeline of tool-free sub-agents (see [Agent Pipelines](#agent-pipelines)). Files referenced as `gs://` URIs in the message are attached to the model request; every stage returns schema-validated JSON into session state.
4. **Backend** reads the `extraction`, `grading`, `referee` and `narration` state keys, applies per-question bounds, "answer any N of M" selection, total caps and review routing, and persists the result to **MongoDB**.

> **Backend ↔ Agent integration**: The backend calls the deployed agent via the Vertex AI Reasoning Engine `:query` API, authenticated with Google service account credentials. `AgentEngineApp.query` returns the final event with all stages' `state_delta` merged.

## Overview

Gradr-Agent acts as the AI reasoning layer for automated grading. It processes marking guides, student answers, and grading rules to generate reliable scoring decisions. The agent uses:

- Gemini 3 Flash / Flash-Lite via the global Vertex AI endpoint (`app/agents/model_config.py`)
- Pydantic output schemas for every grading stage (`app/schemas.py`)
- No tools, database access or web search: the backend owns all I/O and the final score
- BigQuery telemetry for post-hoc evaluation
- OpenTelemetry GenAI instrumentation for distributed tracing

## Agent Pipelines

### PBT (Paper-Based Test) Grading Pipeline

A `SequentialAgent` (`PBTGradingPipeline`) of four tool-free agents:

| Step | Agent | Model | Output key | Role |
|------|-------|-------|------------|------|
| 1 | `ExtractionAgent` | advanced | `extraction` | Transcribes the attached `[script]` per question using the `[questionPaper]`; never grades |
| 2 | `PBTGradingAgent` | advanced | `grading` | Grades each extracted answer against the attached `[markingGuide]` only |
| 3 | `RefereeAgent` | lightweight | `referee` | Flags score/explanation contradictions and scores on unattempted questions; never changes scores |
| 4 | `FeedbackNarrationAgent` | lightweight | `narration` | Writes overall explanation and feedback without stating a total |

### CBT (Computer-Based Test) Grading Pipeline

A `SequentialAgent` (`CBTSubmissionGradingPipeline`) for the essay part of online exam submissions. Multiple-choice questions are scored deterministically by the backend.

| Step | Agent | Model | Output key | Role |
|------|-------|-------|------------|------|
| 1 | `EssayGradingAgent` | advanced | `grading` | Grades each essay against its model answer and rubric (or the attached `[markingGuide]`) |
| 2 | `FeedbackNarrationAgent` | lightweight | `narration` | Writes overall explanation and feedback from essay grades and the MCQ summary |

### CBT Exam Generation Pipeline

A `SequentialAgent` that generates exam questions from uploaded course materials:

| Step | Agent | Model | Role |
|------|-------|-------|------|
| 1 | `TopicExtractionAgent` | lightweight | Analyzes documents and extracts the top 5–10 academic topics with concentration weights |
| 2 | `QuestionGenerationAgent` | advanced | Generates curriculum-aligned MCQ, essay, or hybrid questions (each tagged with its extracted `topic`). Skipped if the request is extract-only |

### Message Contracts

The backend sends `query(message=JSON.stringify(payload))`. File references are `{"gcsUri": "gs://...", "mimeType": "..."}`.

```json
// CBTSubmissionGradingPipeline
{"contractVersion":2,"task":"grade_cbt_essays","attemptId":"…","examType":"GENERIC",
 "markingGuide":{"gcsUri":"gs://grdr/…pdf","mimeType":"application/pdf"} | null,
 "essays":[{"questionId":"Q3","question":"…","description":"…","maxMarks":10,"modelAnswer":"…"|null,"rubric":"…","topic":"…","studentAnswer":"…"}],
 "mcqSummary":[{"questionId":"Q1","topic":"…","correct":true,"score":2,"maxMarks":2}]}
// PBTGradingPipeline
{"contractVersion":2,"task":"grade_pbt_script","examType":"exam",
 "script":{"gcsUri":"gs://grdr/…","mimeType":"application/pdf"},
 "questionPaper":{…},"markingGuide":{…},
 "rubric":{"totalMaxScore":60,"selectionPolicy":{"mode":"best_n","answerCount":4},"questions":[{"questionId":"Q1","maxMarks":15}]}}
```

## Features

- Automated grading against marking guides, model answers and per-question rubrics
- Verbatim script transcription separated from grading
- Schema-validated JSON output for every stage
- Referee consistency checks feeding the backend's teacher-review routing (HITL)
- Cloud-first observability with BigQuery + OpenTelemetry
- Fully reproducible dev environment
- CI/CD-ready with Terraform + Cloud Build

## Project Structure

```
gradr-agent/
├── app/
│   ├── agents/
│   │   ├── pbt_grading_pipeline.py    # PBT sequential pipeline
│   │   ├── cbt_grading_pipeline.py    # CBT essay grading pipeline
│   │   ├── narration.py               # FeedbackNarrationAgent factory
│   │   ├── model_config.py            # Gemini model IDs and retry options
│   │   └── cbt_exam_pipeline.py       # CBT exam generation pipeline
│   ├── agent_engine_app.py            # AgentEngineApp (Vertex AI deployment wrapper)
│   ├── callbacks.py                   # Message payload parsing, skip gates, timing
│   ├── file_parts.py                  # Attaches gs:// payload files to model requests
│   ├── schemas.py                     # Pydantic output schemas
│   ├── prompts/                       # Agent prompts
│   └── app_utils/                     # Deploy scripts, telemetry, typing
├── .cloudbuild/                       # Cloud Build CI/CD pipelines
├── deployment/                        # Terraform IaC
├── local_playground/                  # ADK local dev playground config
├── notebooks/                         # Prototyping, evals, grading experiments
├── tests/                             # Unit + integration tests
├── Makefile                           # Dev automation
└── pyproject.toml                     # Python dependencies (managed by uv)
```

## Deployed Endpoints

### Agent (Vertex AI Reasoning Engine)

The agents are deployed to Vertex AI Reasoning Engine and invoked via the `:query` API. They are not publicly accessible — the backend authenticates using Google Cloud service account credentials.

| Agent | Reasoning Engine ID | API Endpoint |
|-------|-------------------|--------------|
| **PBT Grading Pipeline** | `3347285579636146176` | `https://us-central1-aiplatform.googleapis.com/v1beta1/projects/gradr-421618/locations/us-central1/reasoningEngines/3347285579636146176:query` |
| **CBT Grading Pipeline** | `7102161768956297216` | `https://us-central1-aiplatform.googleapis.com/v1beta1/projects/gradr-421618/locations/us-central1/reasoningEngines/7102161768956297216:query` |
| **CBT Exam Generation** | `1085942005021802496` | `https://us-central1-aiplatform.googleapis.com/v1beta1/projects/gradr-421618/locations/us-central1/reasoningEngines/1085942005021802496:query` |

### Backend (Cloud Run) — closed source

| Service | URL |
|---------|-----|
| **Gradr Backend** | `https://gradr-backend-943768265988.us-central1.run.app` |

> **How the backend calls the agent**: See the [agentRuntime.js gist](https://gist.github.com/Fiewor/29d17b6ad354cfdf63c25a01082064c3) for the full Node.js integration code that routes requests to the correct Reasoning Engine based on pipeline name.

## Local Setup (for Hackathon Judges)

These instructions reproduce the agent environment locally.

### Prerequisites

| Tool | Purpose | Link |
|------|---------|------|
| `uv` | Python package manager | [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| Google Cloud SDK | Access to GCP resources | [cloud.google.com/sdk](https://cloud.google.com/sdk/docs/install) |
| `make` | Automation (usually preinstalled) | [gnu.org/software/make](https://www.gnu.org/software/make/) |

### Clone the Repo

```bash
git clone <repo-url>
cd gradr-agent
```

### Install Dependencies

```bash
make install
```

### Environment Variables

Create a `.env` file (see `.env.example`) with the following:

```bash
GOOGLE_CLOUD_PROJECT=gradr-421618
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=True
GRADR_GEMINI_MODEL_LOCATION=global
```

You must also authenticate with GCP:

```bash
gcloud auth application-default login
gcloud config set project gradr-421618
```

### Run the Playground

This starts a local ADK playground UI with auto-reload at `http://localhost:8501`:

```bash
make playground
```

From the playground dropdown, select the agent pipeline you want to test (PBT Grading, CBT Grading, or CBT Exam Generation).

## Running Locally

### Run tests

```bash
make test
```

### Lint & type checks

```bash
make lint
```

## Deployment

### Deploy individual agents

```bash
make deploy-pbt           # PBT Grading Agent
make deploy-cbt-grading   # CBT Grading Agent
make deploy-cbt-exam      # CBT Exam Generation Agent
make deploy-all           # All agents
```

### Infrastructure (Terraform)

```bash
make setup-dev-env
```

See [deployment/](deployment/) for full Terraform configuration.

## Evaluation & Quality Assurance

Gradr-Agent uses a multi-layered evaluation strategy:

- Notebook-based eval suites
- Rubric consistency tests
- Automated comparison against gold-standard sample answers
- Confidence + chain-of-thought quality checks (via structured CoT)
- RefereeAgent cross-validation with HITL flagging

## Monitoring & Observability

The project integrates OpenTelemetry GenAI instrumentation with:

### Cloud Logging

- All agent operation logs including inter-agent handoff messages
- Viewable at: **GCP Console → Logging → Logs Explorer**

### BigQuery Telemetry

Fast SQL queries over model calls, tool traces, latency profiles, token usage, and error analysis:

```sql
SELECT * FROM `gradr-agent_telemetry.completions` LIMIT 10
```

### Cloud Trace

- Distributed waterfall traces for multi-step agent pipelines
- Viewable at: **GCP Console → Trace → Trace Explorer**
- Each trace shows the full timeline of a grading request across all agents
