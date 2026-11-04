"""Structured outputs of the tool-free grading pipelines.

Each model is the ``output_schema`` of one pipeline stage; the backend reads
the stored state keys (``extraction``, ``grading``, ``referee``,
``narration``) and owns all selection, totals and review routing.
"""

from typing import Literal

from pydantic import BaseModel


class GradedItem(BaseModel):
    questionId: str
    score: float
    maxScore: float
    topic: str = ""
    explanation: str = ""
    feedback: str = ""
    modelConfidence: float
    rubricAlignment: list[str] = []


class GradingOutput(BaseModel):
    items: list[GradedItem]


class NarrationOutput(BaseModel):
    overallExplanation: str
    overallFeedback: str


class ExtractedAnswer(BaseModel):
    questionId: str
    attempted: bool
    answerText: str = ""
    pageNumbers: list[int] = []
    transcriptionConfidence: float


class ExtractionOutput(BaseModel):
    questions: list[ExtractedAnswer]
    missingOrBlankQuestions: list[str] = []
    transcriptionWarnings: list[str] = []
    extractionValidationErrors: list[str] = []


class RefereeIssue(BaseModel):
    questionId: str
    issue: str


class RefereeOutput(BaseModel):
    status: Literal["ok", "issues"]
    issues: list[RefereeIssue] = []
