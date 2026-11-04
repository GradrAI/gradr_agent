"""Offline Intro Python total-score grading experiment runner.

This runner is deliberately isolated from production grading queues and database-backed
agents. It reads local PDFs, calls Gemini directly when not in dry-run mode, and writes
file artifacts for review before any full-dataset run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from xml.etree import ElementTree

from eval.total_score_utils import (
    TotalScoreBook,
    canonical_script_id,
    compute_lecture_ablation_metrics,
    compute_total_score_metrics,
    labels_from_score_book,
    parse_scores_pdf,
)

FULL_RUN_ERROR = "Full 60-script run requires explicit reviewed-output approval"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_PART_MB = 45.0
MAX_SCORE = 60.0
LOW_CONFIDENCE_THRESHOLD = 0.70
MODEL_CALL_MAX_ATTEMPTS = 5
MODEL_CALL_RETRY_DELAYS_SECONDS = (5, 15, 30, 60)
MODEL_CALL_TIMEOUT_MS = 900_000
logging.getLogger("pypdf").setLevel(logging.ERROR)


def _clean_json(raw: str) -> str:
    """Extract a JSON object or array from an LLM response without importing app."""
    stripped = raw.strip()
    start_curly = stripped.find("{")
    start_bracket = stripped.find("[")
    if start_curly != -1 and (start_bracket == -1 or start_curly < start_bracket):
        end = stripped.rfind("}")
        if end > start_curly:
            return stripped[start_curly : end + 1]
    if start_bracket != -1:
        end = stripped.rfind("]")
        if end > start_bracket:
            return stripped[start_bracket : end + 1]
    if stripped.startswith("```json"):
        stripped = stripped[7:]
    elif stripped.startswith("```"):
        stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()

EXPERIMENT_OUTPUT_SCHEMA = {
    "student_identifier_seen": "string or null",
    "graded_questions": [
        {
            "question_id": "string",
            "score": 0,
            "max_score": 0,
            "rubric_alignment": ["string"],
            "justification": "string",
            "model_confidence": 0.0,
            "feedback": "string",
            "source_priority": "marking_guide|question_paper|lecture_context|student_script",
            "lecture_context_used": [
                {
                    "concept": "string",
                    "source": "string",
                    "locator": "string",
                    "why_relevant": "string",
                }
            ],
        }
    ],
    "ignored_annotations": [
        {
            "kind": "margin_text|reg_number|teacher_mark|question_label|other",
            "description": "string",
        }
    ],
    "transcription_warnings": ["string"],
    "rubric_boundary_warnings": ["string"],
}

INTRO_PYTHON_EXPERIMENT_PROMPT = """
You are grading one Introduction to Programming Using Python answer script for an offline GradrAI evaluation.

Inputs attached in this request:
1. Official question paper.
2. Official marking guide/model answer.
3. Optional lecture/course context.
4. One student answer script.

Source priority and boundaries:
- Use the marking guide as the scoring authority.
- Use the official question paper to understand what was asked.
- Grade only the student's answer content against the marking guide.
- Use lecture materials only to interpret beginner-programming terminology, acceptable paraphrases, and feedback explanations.
- Never award marks solely because lecture material says a concept is important.
- Never override a marking-guide allocation using lecture material.
- Maximum total score is exactly 60.
- Student-supplied question numbers and headings may align answers only; they are not evidence of correctness.

Ignore boilerplate and non-answer content:
- Ignore `Do not write in either margin`.
- Ignore `Student's Reg. No`.
- Ignore registration numbers, page headers/footers, margin text, and printed administrative instructions.

Ignore teacher/examiner artifacts:
- Ignore red-pen ticks/checkmarks.
- Ignore X marks.
- Ignore circled scores.
- Ignore per-question scores, totals, teacher comments, and examiner annotations.
- Never use visible teacher marks/scores as labels or as evidence for scoring.

Return strict JSON only using this schema:
{schema}
""".strip()

COURSE_CONCEPTS = (
    {
        "concept": "programming",
        "keywords": ("programming", "instructions", "computer", "specific task"),
        "definition": "Programming is writing instructions for a computer to perform specific tasks.",
        "grading_use": "Accept equivalent definitions of programming when the rubric asks for meaning or purpose.",
    },
    {
        "concept": "program",
        "keywords": ("program refers", "set of commands", "perform a specific task"),
        "definition": "A program is a set of commands or instructions that tells a computer how to perform a task.",
        "grading_use": "Use only to interpret program-definition answers; do not create extra marking points.",
    },
    {
        "concept": "algorithm",
        "keywords": ("algorithm", "step-by-step", "finite", "unambiguous"),
        "definition": "An algorithm is a clear, step-by-step, finite method for solving a problem.",
        "grading_use": "Accept accurate paraphrases mentioning ordered steps, input/output, and termination when the rubric allows.",
    },
    {
        "concept": "flowchart",
        "keywords": ("flowchart", "start/end", "decision", "input/output", "process"),
        "definition": "A flowchart is a visual representation of a process or algorithm using symbols and logical flow.",
        "grading_use": "Use to interpret flowchart symbol and purpose answers while preserving rubric point allocation.",
    },
    {
        "concept": "pseudocode",
        "keywords": ("pseudocode", "human-readable", "logical order", "decision points"),
        "definition": "Pseudocode is a high-level human-readable description of an algorithm.",
        "grading_use": "Accept clear natural-language algorithm descriptions if the marking guide permits pseudocode phrasing.",
    },
    {
        "concept": "syntax and semantics",
        "keywords": ("syntax", "semantics", "meaning", "rules"),
        "definition": "Syntax is the form/rules of instructions; semantics is their meaning or behavior.",
        "grading_use": "Use for interpreting answers distinguishing valid code form from program meaning.",
    },
    {
        "concept": "programming language levels",
        "keywords": ("low level", "high-level", "machine language", "assembly language"),
        "definition": "Low-level languages are close to machine code; high-level languages are more human-readable and portable.",
        "grading_use": "Use for language-classification answers without overriding the official marking guide.",
    },
    {
        "concept": "compiled and interpreted languages",
        "keywords": ("compiled", "interpreted", "line-by-line", "runtime"),
        "definition": "Compiled languages are translated before execution; interpreted languages commonly execute line by line at runtime.",
        "grading_use": "Use to interpret compiled/interpreted comparisons when present in the question paper or guide.",
    },
    {
        "concept": "variables constants and data types",
        "keywords": ("variable", "constant", "integer", "float", "string", "boolean", "list", "dictionary"),
        "definition": "Variables store changeable values, constants store fixed values, and data types classify values such as integers, floats, strings, booleans, lists, and dictionaries.",
        "grading_use": "Use for beginner Python terminology and feedback where the rubric already asks for these concepts.",
    },
)


@dataclass(frozen=True)
class ExperimentCase:
    script_id: str
    script_filename: str
    script_path: str
    human_total_score: float
    max_score: float
    page_count: int
    pdf_size_bytes: int
    pages_reported: int | None = None

@dataclass(frozen=True)
class ModelCallResult:
    model_output: dict[str, Any]
    raw_text: str
    repair_text: str | None = None
    request_attempts: int = 1
    repair_attempts: int = 0


class PdfPartTooLargeError(RuntimeError):
    """Raised when a PDF or split PDF page exceeds the configured part limit."""

    def __init__(self, path: Path, size_bytes: int, limit_bytes: int) -> None:
        super().__init__(
            f"PDF part too large: {path} is {size_bytes} bytes; limit is {limit_bytes} bytes"
        )
        self.path = path
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


def reset_output_files(out_dir: Path, overwrite: bool) -> None:
    tracked = [
        "predictions.jsonl",
        "predictions.no_lecture.jsonl",
        "predictions.lecture.jsonl",
        "failures.jsonl",
        "pairs.csv",
        "metrics.json",
        "metrics-recomputed.json",
        "lecture_ablation_metrics.json",
        "run_events.jsonl",
        "run_summary.json",
        "experiment_prompt.none.txt",
        "experiment_prompt.summary.txt",
        "experiment_prompt.full.txt",
    ]
    existing = [out_dir / name for name in tracked if (out_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output directory already contains experiment artifacts ({names}); pass --overwrite"
        )
    if overwrite:
        for path in existing:
            path.unlink()
        for name in ("raw_model_outputs", "tmp_parts"):
            path = out_dir / name
            if path.exists():
                shutil.rmtree(path)


def parse_script_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    ids = [canonical_script_id(item.strip()) for item in raw.split(",") if item.strip()]
    if not ids:
        raise ValueError("--script-ids must contain at least one numeric script id")
    seen: set[str] = set()
    duplicates: list[str] = []
    for script_id in ids:
        if script_id in seen:
            duplicates.append(script_id)
        seen.add(script_id)
    if duplicates:
        raise ValueError(f"Duplicate script ids: {sorted(set(duplicates))}")
    return ids


def validate_full_run_gate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.script_ids and args.allow_full_run_after_review:
        parser.error("--allow-full-run-after-review cannot be combined with --script-ids")
    if args.allow_full_run_after_review and not args.approval_note:
        parser.error("--approval-note is required with --allow-full-run-after-review")
    if not args.script_ids and not args.allow_full_run_after_review:
        parser.error(FULL_RUN_ERROR)


def _pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to inspect PDFs") from exc
    return len(PdfReader(str(path)).pages)


def build_manifest(
    scripts_dir: Path,
    label_book: TotalScoreBook,
    question_pdf: Path,
    guide_pdf: Path,
    script_ids: list[str] | None,
) -> list[ExperimentCase]:
    labels = label_book.labels
    selected_ids = script_ids or sorted(labels, key=int)
    cases: list[ExperimentCase] = []

    for script_id in selected_ids:
        if script_id not in labels:
            raise ValueError(f"No total-score label for script id {script_id}")
        script_path = scripts_dir / f"{script_id}.pdf"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing script PDF: {script_path}")
        label = labels[script_id]
        cases.append(
            ExperimentCase(
                script_id=script_id,
                script_filename=script_path.name,
                script_path=str(script_path),
                human_total_score=label.human_total_score,
                max_score=label.max_score,
                page_count=_pdf_page_count(script_path),
                pdf_size_bytes=script_path.stat().st_size,
                pages_reported=label.pages_reported,
            )
        )

    for required in (question_pdf, guide_pdf):
        if not required.exists():
            raise FileNotFoundError(f"Missing required PDF: {required}")
    return cases


def _extract_pdf_segments(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to extract PDF lecture text") from exc

    warnings: list[str] = []
    reader = PdfReader(str(path))
    segments: list[dict[str, Any]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            segments.append({"locator": f"page {index}", "text": text})
        else:
            warnings.append(f"{path}: no extractable text on page {index}")
    return segments, warnings


def _slide_number(name: str) -> int:
    digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
    return int(digits) if digits else 0


def _extract_pptx_segments(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    segments: list[dict[str, Any]] = []
    namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}

    with zipfile.ZipFile(path) as archive:
        slide_names = sorted(
            (
                name
                for name in archive.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ),
            key=_slide_number,
        )
        for name in slide_names:
            xml = archive.read(name)
            root = ElementTree.fromstring(xml)
            texts = [node.text for node in root.findall(".//a:t", namespace) if node.text]
            if texts:
                segments.append(
                    {"locator": f"slide {_slide_number(name)}", "text": "\n".join(texts)}
                )
            else:
                warnings.append(f"{path}: no extractable text in {Path(name).name}")
    return segments, warnings


def extract_document_segments(path: Path) -> tuple[list[dict[str, Any]], list[str], str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        segments, warnings = _extract_pdf_segments(path)
        return segments, warnings, "pdf"
    if suffix == ".pptx":
        segments, warnings = _extract_pptx_segments(path)
        return segments, warnings, "pptx"
    return [], [f"{path}: unsupported lecture material type {suffix}"], suffix.lstrip(".")


def _first_title(path: Path, segments: list[dict[str, Any]]) -> str:
    for segment in segments:
        for line in str(segment["text"]).splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
    return path.stem


def _locate_concept(
    concept: dict[str, Any], materials: list[dict[str, Any]]
) -> dict[str, str] | None:
    keywords = tuple(str(keyword).lower() for keyword in concept["keywords"])
    for material in materials:
        for segment in material.get("segments", []):
            text = str(segment.get("text", ""))
            lower = text.lower()
            if any(keyword in lower for keyword in keywords):
                return {
                    "source": material["path"],
                    "locator": str(segment.get("locator", "unknown")),
                    "excerpt": " ".join(text.split())[:240],
                }
    return None


def build_lecture_context(material_paths: list[Path], mode: str) -> dict[str, Any]:
    if mode == "none" or not material_paths:
        return {"mode": mode, "materials": [], "concepts": [], "warnings": []}

    materials: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in material_paths:
        if not path.exists():
            warnings.append(f"Lecture material missing: {path}")
            materials.append(
                {
                    "path": str(path),
                    "title": path.stem,
                    "kind": path.suffix.lower().lstrip("."),
                    "pages_or_slides": 0,
                    "segments": [],
                    "_segments_for_index": [],
                }
            )
            continue
        try:
            segments, segment_warnings, kind = extract_document_segments(path)
        except Exception as exc:
            segments, segment_warnings, kind = (
                [],
                [f"{path}: extraction failed: {exc}"],
                path.suffix.lower().lstrip("."),
            )
        warnings.extend(segment_warnings)
        materials.append(
            {
                "path": str(path),
                "title": _first_title(path, segments),
                "kind": kind,
                "pages_or_slides": len(segments),
                "segments": segments if mode == "full" else [],
                "_segments_for_index": segments,
            }
        )

    searchable_materials = [
        {**material, "segments": material.pop("_segments_for_index")}
        for material in materials
    ]
    concepts: list[dict[str, Any]] = []
    for concept in COURSE_CONCEPTS:
        located = _locate_concept(concept, searchable_materials)
        if located is None:
            continue
        concepts.append(
            {
                "concept": concept["concept"],
                "definition": concept["definition"],
                "source": located["source"],
                "locator": located["locator"],
                "excerpt": located["excerpt"],
                "grading_use": concept["grading_use"],
            }
        )

    for material in materials:
        material.pop("_segments_for_index", None)
    return {"mode": mode, "materials": materials, "concepts": concepts, "warnings": warnings}


def lecture_prompt_text(lecture_context: dict[str, Any], mode: str) -> str:
    if mode == "none":
        return "Lecture context mode: none. No lecture material is available for this run."
    if mode == "full":
        payload = lecture_context
    else:
        payload = {
            "mode": lecture_context.get("mode"),
            "materials": lecture_context.get("materials", []),
            "concepts": lecture_context.get("concepts", []),
            "warnings": lecture_context.get("warnings", []),
        }
    return "Lecture/course context, bounded by the prompt source-priority rules:\n" + json.dumps(
        payload, indent=2, sort_keys=True
    )


def prompt_for_lecture_context(lecture_context: dict[str, Any], mode: str) -> str:
    schema = json.dumps(EXPERIMENT_OUTPUT_SCHEMA, indent=2)
    return (
        INTRO_PYTHON_EXPERIMENT_PROMPT.format(schema=schema)
        + "\n\n"
        + lecture_prompt_text(lecture_context, mode)
    )


def prepare_pdf_parts(pdf_path: Path, run_tmp_dir: Path, max_part_mb: float) -> list[Any]:
    from google.genai import types
    from pypdf import PdfReader, PdfWriter

    limit_bytes = int(max_part_mb * 1024 * 1024)
    size = pdf_path.stat().st_size
    if size <= limit_bytes:
        return [types.Part.from_bytes(data=pdf_path.read_bytes(), mime_type="application/pdf")]

    reader = PdfReader(str(pdf_path))
    split_dir = run_tmp_dir / pdf_path.stem
    split_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Any] = []
    for index, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        page_path = split_dir / f"page-{index:03d}.pdf"
        with page_path.open("wb") as file:
            writer.write(file)
        page_size = page_path.stat().st_size
        if page_size > limit_bytes:
            raise PdfPartTooLargeError(page_path, page_size, limit_bytes)
        parts.append(types.Part.from_bytes(data=page_path.read_bytes(), mime_type="application/pdf"))
    return parts


def parse_model_json(raw: str) -> dict[str, Any]:
    parsed = json.loads(_clean_json(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object")
    return parsed

def create_gemini_client(project: str, location: str) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(timeout=MODEL_CALL_TIMEOUT_MS),
    )


def _is_retryable_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    non_retryable_markers = (
        "not_found",
        "not found",
        "does not have access",
        "invalid model",
        "permission denied",
    )
    if any(marker in message for marker in non_retryable_markers):
        return False
    retryable_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "connection",
        "deadline",
        "failed to resolve",
        "internal server error",
        "max retries exceeded",
        "nameresolutionerror",
        "resource exhausted",
        "service unavailable",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "unavailable",
    )
    return any(marker in message for marker in retryable_markers)


def generate_content_with_retries(
    client: Any,
    model: str,
    contents: list[Any],
    config: Any,
) -> tuple[Any, int]:
    for attempt in range(1, MODEL_CALL_MAX_ATTEMPTS + 1):
        try:
            return (
                client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                ),
                attempt,
            )
        except Exception as exc:
            if attempt == MODEL_CALL_MAX_ATTEMPTS or not _is_retryable_model_error(exc):
                raise
            delay = MODEL_CALL_RETRY_DELAYS_SECONDS[
                min(attempt - 1, len(MODEL_CALL_RETRY_DELAYS_SECONDS) - 1)
            ]
            time.sleep(delay)
    raise RuntimeError("unreachable retry loop exit")


def call_gemini_for_case(
    client: Any,
    case: ExperimentCase,
    question_parts: list[Any],
    guide_parts: list[Any],
    script_parts: list[Any],
    prompt: str,
    model: str,
) -> ModelCallResult:
    from google.genai import types

    contents = [
        types.Part.from_text(text=f"Official question paper for script {case.script_id}:"),
        *question_parts,
        types.Part.from_text(text="Official marking guide/model answer:"),
        *guide_parts,
        types.Part.from_text(text=f"Student answer script {case.script_id}:"),
        *script_parts,
        types.Part.from_text(text=prompt),
    ]
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0,
    )
    response, request_attempts = generate_content_with_retries(
        client=client,
        model=model,
        contents=contents,
        config=config,
    )
    raw_text = response.text or ""
    try:
        return ModelCallResult(
            model_output=parse_model_json(raw_text),
            raw_text=raw_text,
            request_attempts=request_attempts,
        )
    except Exception:
        repair_prompt = (
            "Repair the following response into strict JSON only using this exact schema. "
            "Do not change the grading content unless required to make valid JSON.\n\n"
            f"Schema:\n{json.dumps(EXPERIMENT_OUTPUT_SCHEMA, indent=2)}\n\n"
            f"Invalid response:\n{raw_text}"
        )
        repair_response, repair_attempts = generate_content_with_retries(
            client=client,
            model=model,
            contents=[types.Part.from_text(text=repair_prompt)],
            config=config,
        )
        repair_text = repair_response.text or ""
        return ModelCallResult(
            model_output=parse_model_json(repair_text),
            raw_text=raw_text,
            repair_text=repair_text,
            request_attempts=request_attempts,
            repair_attempts=repair_attempts,
        )


def _float_field(item: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = item.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finalize_prediction(
    case: ExperimentCase,
    model_output: dict[str, Any],
    lecture_mode: str,
    duration_ms: int,
) -> dict[str, Any]:
    graded_questions = model_output.get("graded_questions", [])
    validation_errors: list[str] = []
    if not isinstance(graded_questions, list) or not graded_questions:
        validation_errors.append("graded_questions must be a non-empty array")
        graded_questions = []

    total = 0.0
    question_max_total = 0.0
    confidences: list[float] = []
    lecture_use_count = 0
    for index, question in enumerate(graded_questions):
        if not isinstance(question, dict):
            validation_errors.append(f"graded_questions[{index}] must be an object")
            continue
        score = _float_field(question, "score")
        max_score = _float_field(question, "max_score")
        confidence = _float_field(question, "model_confidence", default=-1.0)
        if score < 0:
            validation_errors.append(f"graded_questions[{index}].score is negative")
        if max_score < 0:
            validation_errors.append(f"graded_questions[{index}].max_score is negative")
        if max_score and score > max_score:
            validation_errors.append(f"graded_questions[{index}].score exceeds max_score")
        if 0 <= confidence <= 1:
            confidences.append(confidence)
        else:
            validation_errors.append(
                f"graded_questions[{index}].model_confidence outside 0..1"
            )
        if question.get("lecture_context_used"):
            lecture_use_count += 1
        total += score
        question_max_total += max_score

    if total > case.max_score:
        validation_errors.append(f"ai_total_score exceeds max_score {case.max_score}")

    ignored_annotations = model_output.get("ignored_annotations", [])
    transcription_warnings = model_output.get("transcription_warnings", [])
    rubric_boundary_warnings = model_output.get("rubric_boundary_warnings", [])
    if not isinstance(ignored_annotations, list):
        ignored_annotations = []
        validation_errors.append("ignored_annotations must be an array")
    if not isinstance(transcription_warnings, list):
        transcription_warnings = []
        validation_errors.append("transcription_warnings must be an array")
    if not isinstance(rubric_boundary_warnings, list):
        rubric_boundary_warnings = []
        validation_errors.append("rubric_boundary_warnings must be an array")

    low_confidence_count = sum(
        confidence < LOW_CONFIDENCE_THRESHOLD for confidence in confidences
    )
    referee_status = (
        "PENDING_REVIEW"
        if low_confidence_count > 0 or validation_errors or transcription_warnings
        else "COMPLETED"
    )
    ai_total_score = round(total, 4)
    mean_confidence = round(mean(confidences), 4) if confidences else None
    min_confidence = round(min(confidences), 4) if confidences else None

    return {
        "script_id": case.script_id,
        "script_filename": case.script_filename,
        "script_path": case.script_path,
        "human_total_score": case.human_total_score,
        "ai_total_score": ai_total_score,
        "max_score": case.max_score,
        "score_summary": f"{ai_total_score:g}/{case.max_score:g}",
        "absolute_error": round(abs(ai_total_score - case.human_total_score), 4),
        "signed_error": round(ai_total_score - case.human_total_score, 4),
        "sum_question_max_score": round(question_max_total, 4),
        "mean_confidence": mean_confidence,
        "min_confidence": min_confidence,
        "low_confidence_count": low_confidence_count,
        "referee_status": referee_status,
        "lecture_mode": lecture_mode,
        "lecture_context_used_count": lecture_use_count,
        "rubric_boundary_warning_count": len(rubric_boundary_warnings),
        "duration_ms": duration_ms,
        "validation_errors": validation_errors,
        "student_identifier_seen": model_output.get("student_identifier_seen"),
        "graded_questions": graded_questions,
        "ignored_annotations": ignored_annotations,
        "transcription_warnings": transcription_warnings,
        "rubric_boundary_warnings": rubric_boundary_warnings,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if path.exists():
            path.unlink()
        return
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def write_pairs_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    fieldnames = [
        "script_id",
        "script_filename",
        "human_total_score",
        "ai_total_score",
        "max_score",
        "absolute_error",
        "signed_error",
        "mean_confidence",
        "referee_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow({field: prediction.get(field) for field in fieldnames})


def failure_record(case: ExperimentCase, error_code: str, error: str) -> dict[str, Any]:
    return {
        "script_id": case.script_id,
        "script_filename": case.script_filename,
        "error_code": error_code,
        "error": error,
        "timestamp": utc_now(),
    }


def run_prediction_set(
    cases: list[ExperimentCase],
    labels: dict[str, dict[str, Any]],
    question_pdf: Path,
    guide_pdf: Path,
    lecture_context: dict[str, Any],
    lecture_mode: str,
    out_dir: Path,
    output_name: str,
    model: str,
    project: str,
    location: str,
    max_part_mb: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tmp_dir = out_dir / "tmp_parts" / output_name.replace(".", "_")
    question_parts = prepare_pdf_parts(question_pdf, tmp_dir / "question", max_part_mb)
    guide_parts = prepare_pdf_parts(guide_pdf, tmp_dir / "guide", max_part_mb)
    prompt = prompt_for_lecture_context(lecture_context, lecture_mode)
    prompt_path = out_dir / f"experiment_prompt.{lecture_mode}.txt"
    prompt_path.write_text(prompt)
    events_path = out_dir / "run_events.jsonl"
    raw_output_dir = out_dir / "raw_model_outputs" / Path(output_name).stem.replace(".", "_")
    client = create_gemini_client(project, location)

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        started_at = utc_now()
        start = time.monotonic()
        append_jsonl(
            events_path,
            {
                "event": "script_started",
                "lecture_mode": lecture_mode,
                "output_name": output_name,
                "prompt_path": str(prompt_path),
                "script_filename": case.script_filename,
                "script_id": case.script_id,
                "timestamp": started_at,
            },
        )
        try:
            script_parts = prepare_pdf_parts(
                Path(case.script_path),
                tmp_dir / case.script_id,
                max_part_mb,
            )
            call_result = call_gemini_for_case(
                client=client,
                case=case,
                question_parts=question_parts,
                guide_parts=guide_parts,
                script_parts=script_parts,
                prompt=prompt,
                model=model,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            raw_path = raw_output_dir / f"{case.script_id}.raw.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(call_result.raw_text)
            repair_path = None
            if call_result.repair_text is not None:
                repair_path = raw_output_dir / f"{case.script_id}.repair.txt"
                repair_path.write_text(call_result.repair_text)

            prediction = finalize_prediction(
                case=case,
                model_output=call_result.model_output,
                lecture_mode=lecture_mode,
                duration_ms=duration_ms,
            )
            prediction["raw_model_response_path"] = str(raw_path)
            prediction["repair_model_response_path"] = str(repair_path) if repair_path else None
            prediction["json_repair_used"] = call_result.repair_text is not None
            prediction["prompt_path"] = str(prompt_path)
            prediction["model_request_attempts"] = call_result.request_attempts
            prediction["model_repair_attempts"] = call_result.repair_attempts
            append_jsonl(out_dir / output_name, prediction)
            predictions.append(prediction)
            append_jsonl(
                events_path,
                {
                    "absolute_error": prediction["absolute_error"],
                    "ai_total_score": prediction["ai_total_score"],
                    "duration_ms": duration_ms,
                    "event": "script_succeeded",
                    "human_total_score": prediction["human_total_score"],
                    "json_repair_used": prediction["json_repair_used"],
                    "model_repair_attempts": prediction["model_repair_attempts"],
                    "model_request_attempts": prediction["model_request_attempts"],
                    "lecture_mode": lecture_mode,
                    "low_confidence_count": prediction["low_confidence_count"],
                    "mean_confidence": prediction["mean_confidence"],
                    "output_name": output_name,
                    "raw_model_response_path": str(raw_path),
                    "referee_status": prediction["referee_status"],
                    "script_filename": case.script_filename,
                    "script_id": case.script_id,
                    "timestamp": utc_now(),
                    "validation_error_count": len(prediction["validation_errors"]),
                },
            )
        except PdfPartTooLargeError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            failure = failure_record(case, "pdf_part_too_large", str(exc))
            failure.update(
                {
                    "duration_ms": duration_ms,
                    "lecture_mode": lecture_mode,
                    "output_name": output_name,
                }
            )
            failures.append(failure)
            append_jsonl(
                events_path,
                {
                    "duration_ms": duration_ms,
                    "error": str(exc),
                    "error_code": "pdf_part_too_large",
                    "event": "script_failed",
                    "lecture_mode": lecture_mode,
                    "output_name": output_name,
                    "script_filename": case.script_filename,
                    "script_id": case.script_id,
                    "timestamp": utc_now(),
                },
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            failure = failure_record(case, "grading_failed", str(exc))
            failure.update(
                {
                    "duration_ms": duration_ms,
                    "lecture_mode": lecture_mode,
                    "output_name": output_name,
                }
            )
            failures.append(failure)
            append_jsonl(
                events_path,
                {
                    "duration_ms": duration_ms,
                    "error": str(exc),
                    "error_code": "grading_failed",
                    "event": "script_failed",
                    "lecture_mode": lecture_mode,
                    "output_name": output_name,
                    "script_filename": case.script_filename,
                    "script_id": case.script_id,
                    "timestamp": utc_now(),
                },
            )
    prediction_map = {prediction["script_id"]: prediction for prediction in predictions}
    metrics = compute_total_score_metrics(labels, prediction_map, failed_count=len(failures))
    write_jsonl(out_dir / output_name, predictions)
    return predictions, failures, metrics


def write_manifest(
    out_dir: Path,
    cases: list[ExperimentCase],
    label_book: TotalScoreBook,
    question_pdf: Path,
    guide_pdf: Path,
    lecture_materials: list[Path],
) -> None:
    write_json(
        out_dir / "manifest.json",
        {
            "question_pdf": str(question_pdf),
            "guide_pdf": str(guide_pdf),
            "lecture_materials": [str(path) for path in lecture_materials],
            "selected_case_count": len(cases),
            "cases": [asdict(case) for case in cases],
            "score_book": label_book.to_json_dict(),
        },
    )


def write_run_summary(
    out_dir: Path,
    args: argparse.Namespace,
    cases: list[ExperimentCase],
    predictions: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    started_at: str,
    ended_at: str,
) -> None:
    write_json(
        out_dir / "run_summary.json",
        {
            "started_at": started_at,
            "ended_at": ended_at,
            "requested_count": len(cases),
            "succeeded_count": 0 if args.dry_run else len(predictions),
            "failed_count": len(failures),
            "paired_count": 0 if args.dry_run else len(predictions),
            "dry_run": args.dry_run,
            "lecture_mode": args.lecture_mode,
            "run_lecture_ablation": args.run_lecture_ablation,
            "model": args.model,
            "project": args.project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            "location": args.location,
            "allow_full_run_after_review": args.allow_full_run_after_review,
            "approval_note": args.approval_note,
            "paths": {
                "out_dir": str(out_dir),
                "manifest": str(out_dir / "manifest.json"),
                "lecture_context": str(out_dir / "lecture_context.json"),
                "predictions": str(out_dir / "predictions.jsonl"),
                "predictions_no_lecture": str(out_dir / "predictions.no_lecture.jsonl"),
                "predictions_lecture": str(out_dir / "predictions.lecture.jsonl"),
                "failures": str(out_dir / "failures.jsonl"),
                "pairs": str(out_dir / "pairs.csv"),
                "metrics": str(out_dir / "metrics.json"),
                "metrics_recomputed": str(out_dir / "metrics-recomputed.json"),
                "lecture_ablation_metrics": str(out_dir / "lecture_ablation_metrics.json"),
                "run_events": str(out_dir / "run_events.jsonl"),
                "raw_model_outputs": str(out_dir / "raw_model_outputs"),
                "experiment_prompt_none": str(out_dir / "experiment_prompt.none.txt"),
                "experiment_prompt_summary": str(out_dir / "experiment_prompt.summary.txt"),
                "experiment_prompt_full": str(out_dir / "experiment_prompt.full.txt"),
            },
        },
    )


def run_experiment(args: argparse.Namespace) -> int:
    started_at = utc_now()
    scripts_dir = Path(args.scripts_dir)
    scores_pdf = Path(args.scores_pdf)
    question_pdf = Path(args.question_pdf) if args.question_pdf else scripts_dir / "question.pdf"
    guide_pdf = Path(args.guide_pdf) if args.guide_pdf else scripts_dir / "guide.pdf"
    out_dir = Path(args.out_dir)
    lecture_materials = [Path(item) for item in args.lecture_material]

    out_dir.mkdir(parents=True, exist_ok=True)
    reset_output_files(out_dir, args.overwrite)

    label_book = parse_scores_pdf(scores_pdf)
    labels = labels_from_score_book(label_book)
    script_ids = parse_script_ids(args.script_ids)
    cases = build_manifest(scripts_dir, label_book, question_pdf, guide_pdf, script_ids)
    lecture_context = build_lecture_context(lecture_materials, args.lecture_mode)

    write_manifest(out_dir, cases, label_book, question_pdf, guide_pdf, lecture_materials)
    write_json(out_dir / "lecture_context.json", lecture_context)

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    metrics: dict[str, Any]

    if args.dry_run:
        metrics = compute_total_score_metrics(labels, {})
        write_pairs_csv(out_dir / "pairs.csv", [])
    elif args.run_lecture_ablation:
        no_lecture_context = build_lecture_context([], "none")
        no_lecture_predictions, no_lecture_failures, no_lecture_metrics = run_prediction_set(
            cases=cases,
            labels=labels,
            question_pdf=question_pdf,
            guide_pdf=guide_pdf,
            lecture_context=no_lecture_context,
            lecture_mode="none",
            out_dir=out_dir,
            output_name="predictions.no_lecture.jsonl",
            model=args.model,
            project=args.project,
            location=args.location,
            max_part_mb=args.max_part_mb,
        )
        lecture_predictions, lecture_failures, lecture_metrics = run_prediction_set(
            cases=cases,
            labels=labels,
            question_pdf=question_pdf,
            guide_pdf=guide_pdf,
            lecture_context=lecture_context,
            lecture_mode=args.lecture_mode,
            out_dir=out_dir,
            output_name="predictions.lecture.jsonl",
            model=args.model,
            project=args.project,
            location=args.location,
            max_part_mb=args.max_part_mb,
        )
        predictions = lecture_predictions
        failures = no_lecture_failures + lecture_failures
        write_jsonl(out_dir / "predictions.jsonl", predictions)
        write_pairs_csv(out_dir / "pairs.csv", predictions)
        metrics = lecture_metrics
        ablation = compute_lecture_ablation_metrics(
            labels=labels,
            without_lecture={item["script_id"]: item for item in no_lecture_predictions},
            with_lecture={item["script_id"]: item for item in lecture_predictions},
        )
        ablation["no_lecture_metrics"] = no_lecture_metrics
        ablation["lecture_metrics"] = lecture_metrics
        write_json(out_dir / "lecture_ablation_metrics.json", ablation)
    else:
        predictions, failures, metrics = run_prediction_set(
            cases=cases,
            labels=labels,
            question_pdf=question_pdf,
            guide_pdf=guide_pdf,
            lecture_context=lecture_context,
            lecture_mode=args.lecture_mode,
            out_dir=out_dir,
            output_name="predictions.jsonl",
            model=args.model,
            project=args.project,
            location=args.location,
            max_part_mb=args.max_part_mb,
        )
        write_pairs_csv(out_dir / "pairs.csv", predictions)

    write_jsonl(out_dir / "failures.jsonl", failures)
    write_json(out_dir / "metrics.json", metrics)
    write_run_summary(out_dir, args, cases, predictions, failures, started_at, utc_now())

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "requested_count": len(cases),
                "succeeded_count": 0 if args.dry_run else len(predictions),
                "failed_count": len(failures),
                "dry_run": args.dry_run,
                "lecture_mode": args.lecture_mode,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline Intro Python total-score grading experiment."
    )
    parser.add_argument("--scripts-dir", required=True)
    parser.add_argument("--scores-pdf", required=True)
    parser.add_argument("--question-pdf", default=None)
    parser.add_argument("--guide-pdf", default=None)
    parser.add_argument("--lecture-material", action="append", default=[])
    parser.add_argument("--script-ids", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument(
        "--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    )
    parser.add_argument("--max-part-mb", type=float, default=DEFAULT_MAX_PART_MB)
    parser.add_argument("--lecture-mode", choices=("none", "summary", "full"), default="summary")
    parser.add_argument("--run-lecture-ablation", action="store_true")
    parser.add_argument("--allow-full-run-after-review", action="store_true")
    parser.add_argument("--approval-note", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_full_run_gate(args, parser)
    try:
        raise SystemExit(run_experiment(args))
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
