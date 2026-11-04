"""Staged corrected-rubric Intro Python grading experiment runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from eval.intro_python_experiment import (
    DEFAULT_MAX_PART_MB,
    DEFAULT_MODEL,
    FULL_RUN_ERROR,
    LOW_CONFIDENCE_THRESHOLD,
    EXPERIMENT_OUTPUT_SCHEMA,
    ExperimentCase,
    ModelCallResult,
    PdfPartTooLargeError,
    append_jsonl,
    build_lecture_context,
    build_manifest,
    create_gemini_client,
    failure_record,
    generate_content_with_retries,
    parse_model_json,
    parse_script_ids,
    prepare_pdf_parts,
    utc_now,
    validate_full_run_gate,
    write_json,
    write_jsonl,
    write_pairs_csv,
)
from eval.rubric_config import (
    ExamRubric,
    QUESTION_ORDER,
    QuestionScoreForSelection,
    load_exam_rubric,
    normalize_question_id,
    question_score_for_selection,
    rubric_prompt_section,
    rubric_to_json,
    select_best_allowed_questions,
)
from eval.total_score_utils import (
    compute_lecture_ablation_metrics,
    compute_total_score_metrics,
    labels_from_score_book,
    parse_scores_pdf,
)

EXTRACTION_OUTPUT_SCHEMA = {
    "script_id": "string",
    "student_identifier_seen": "string or null",
    "pages_observed": [
        {
            "page_number": 1,
            "page_present": True,
            "content_summary": "string",
            "non_answer_artifacts": ["string"],
        }
    ],
    "questions": [
        {
            "question_id": "q1",
            "attempted": True,
            "page_numbers": [1],
            "answer_text": "string",
            "subparts": [
                {
                    "part_label": "a|b|c",
                    "answer_text": "string",
                    "page_numbers": [1],
                    "uncertain": False,
                }
            ],
            "evidence_notes": "string",
            "transcription_confidence": 0.0,
        }
    ],
    "missing_or_blank_questions": ["q4"],
    "ignored_annotations": [{"kind": "string", "description": "string"}],
    "transcription_warnings": ["string"],
}

GRADE_OUTPUT_SCHEMA = {
    "student_identifier_seen": "string or null",
    "graded_questions": [
        {
            "question_id": "q1",
            "attempted": True,
            "score": 0,
            "max_score": 0,
            "subpart_scores": [
                {
                    "part_label": "a|b|c",
                    "score": 0,
                    "max_score": 0,
                    "evidence": "string",
                    "justification": "string",
                }
            ],
            "rubric_alignment": ["string"],
            "justification": "string",
            "model_confidence": 0.0,
            "feedback": "string",
            "source_priority": "corrected_rubric|marking_guide|question_paper|lecture_context|student_extraction",
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
    "ignored_annotations": [{"kind": "string", "description": "string"}],
    "transcription_warnings": ["string"],
    "rubric_boundary_warnings": ["string"],
    "extraction_limitations": ["string"],
}

EXTRACTION_PROMPT = """
You are performing extraction/OCR only for an offline GradrAI evaluation.

Inputs attached in this request:
1. Official question paper.
2. One student answer script.

Do not grade. Do not use a marking guide. Do not infer missing work. Do not use lecture material or human labels.
Transcribe only visible student answer content. Separate student writing from teacher/examiner artifacts: red-pen ticks, X marks, circled scores, written totals, per-question scores, teacher comments, registration numbers, page headers/footers, and boilerplate.

Return all five canonical question rows q1..q5. For unseen or blank questions set attempted=false, answer_text="", subparts=[], and list the question in missing_or_blank_questions. Mark uncertain handwriting with uncertain=true at subpart level and add a transcription warning.

Return strict JSON only using this schema:
{schema}
""".strip()

GRADE_FROM_EXTRACTION_PROMPT = """
You are grading one Introduction to Programming Using Python answer script for a staged offline GradrAI evaluation.

Inputs attached in this request:
1. Official question paper.
2. Official scanned marking guide/model answer.
3. Saved extraction JSON from the extraction/OCR stage.
4. Corrected machine-readable rubric allocations.
5. Optional lecture/course context.

The saved extraction JSON is the student-answer source of truth for this grading stage. Do not use hidden visual OCR from the original student script. If the extraction lacks evidence for a question, do not invent evidence.

Source priority and boundaries:
- The corrected machine-readable rubric is the scoring authority and overrides printed, handwritten, or scanned mark allocations when they conflict.
- Use the scanned marking guide/model answer for expected answer content, not mark totals.
- Use the official question paper to understand what was asked.
- Use lecture materials only to interpret beginner-programming terminology, acceptable paraphrases, and feedback explanations.
- Never award marks solely because lecture material says a concept is important.
- Never override corrected rubric allocations using lecture material.
- Grade every attempted question and provide subpart scores for every attempted question.
- For unattempted questions included in the output, set attempted=false and score=0.

{rubric_section}

Lecture/course context, bounded by the source-priority rules:
{lecture_context}

Return strict JSON only using this schema:
{schema}
""".strip()


@dataclass(frozen=True)
class StageIORecord:
    script_id: str | None
    condition: str | None
    stage: str
    input_path: str
    output_path: str
    raw_output_path: str | None
    parsed_output_path: str | None
    prompt_path: str | None
    started_at: str
    ended_at: str
    duration_ms: int
    status: str
    validation_errors: list[str]
    warnings: list[str]


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_artifact(
    path: Path, kind: str, mime_type: str | None = None, page_count: int | None = None
) -> dict[str, Any]:
    source = Path(path)
    return {
        "path": str(source),
        "kind": kind,
        "mime_type": mime_type,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "page_count": page_count,
    }


def json_artifact(path: Path, kind: str) -> dict[str, Any]:
    source = Path(path)
    keys: list[str] = []
    try:
        parsed = json.loads(source.read_text())
        if isinstance(parsed, dict):
            keys = sorted(str(key) for key in parsed)
    except Exception:
        keys = []
    return {
        "path": str(source),
        "kind": kind,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "top_level_keys": keys,
    }


def _stage_path(
    out_dir: Path,
    stage: str,
    io_kind: str,
    condition: str | None,
    script_id: str | None,
) -> Path:
    filename = f"{script_id}.json" if script_id is not None else "corrected_staged.json"
    if condition:
        return out_dir / "stage_io" / stage / condition / io_kind / filename
    return out_dir / "stage_io" / stage / io_kind / filename


def write_stage_input(
    out_dir: Path,
    stage: str,
    condition: str | None,
    script_id: str | None,
    payload: dict[str, Any],
) -> Path:
    path = _stage_path(out_dir, stage, "inputs", condition, script_id)
    write_json(path, payload)
    return path


def write_stage_output(
    out_dir: Path,
    stage: str,
    condition: str | None,
    script_id: str | None,
    payload: dict[str, Any],
) -> Path:
    path = _stage_path(out_dir, stage, "outputs", condition, script_id)
    write_json(path, payload)
    return path


def append_stage_io_record(out_dir: Path, record: StageIORecord) -> None:
    append_jsonl(out_dir / "stage_io.jsonl", asdict(record))


def write_pipeline_trace_index(out_dir: Path) -> None:
    records = []
    path = out_dir / "stage_io.jsonl"
    if path.exists():
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    scripts: dict[str, Any] = {}
    globals_: dict[str, Any] = {}
    for record in records:
        script_id = record.get("script_id")
        condition = record.get("condition")
        stage = record.get("stage")
        if script_id is None:
            globals_[stage] = record
            continue
        script_trace = scripts.setdefault(script_id, {"stages": {}, "conditions": {}})
        if condition is None:
            script_trace["stages"][stage] = record
        else:
            condition_trace = script_trace["conditions"].setdefault(
                condition, {"stages": {}}
            )
            condition_trace["stages"][stage] = record
    write_json(out_dir / "pipeline_trace_index.json", {"scripts": scripts, "global": globals_})


def _prompt_hash(path: Path) -> str:
    return sha256_file(path) if path.exists() else ""


def _make_repair_prompt(schema: dict[str, Any], raw_text: str) -> str:
    return (
        "Repair the following response into strict JSON only using this exact schema. "
        "Do not change the content unless required to make valid JSON.\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Invalid response:\n{raw_text}"
    )


def _generate_json_with_repair(
    client: Any,
    model: str,
    contents: list[Any],
    schema: dict[str, Any],
) -> ModelCallResult:
    from google.genai import types

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
        repair_response, repair_attempts = generate_content_with_retries(
            client=client,
            model=model,
            contents=[types.Part.from_text(text=_make_repair_prompt(schema, raw_text))],
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


def extraction_prompt_text() -> str:
    return EXTRACTION_PROMPT.format(schema=json.dumps(EXTRACTION_OUTPUT_SCHEMA, indent=2))


def grade_prompt_text(rubric: ExamRubric, lecture_context: dict[str, Any], mode: str) -> str:
    if mode == "none":
        lecture_payload = "Lecture context mode: none. No lecture material is available for this run."
    else:
        payload = (
            lecture_context
            if mode == "full"
            else {
                "mode": lecture_context.get("mode"),
                "materials": lecture_context.get("materials", []),
                "concepts": lecture_context.get("concepts", []),
                "warnings": lecture_context.get("warnings", []),
            }
        )
        lecture_payload = json.dumps(payload, indent=2, sort_keys=True)
    return GRADE_FROM_EXTRACTION_PROMPT.format(
        rubric_section=rubric_prompt_section(rubric),
        lecture_context=lecture_payload,
        schema=json.dumps(GRADE_OUTPUT_SCHEMA, indent=2),
    )


def call_gemini_for_extraction(
    client: Any,
    case: ExperimentCase,
    question_parts: list[Any],
    script_parts: list[Any],
    prompt: str,
    model: str,
) -> ModelCallResult:
    from google.genai import types

    contents = [
        types.Part.from_text(text=f"Official question paper for script {case.script_id}:"),
        *question_parts,
        types.Part.from_text(text=f"Student answer script {case.script_id}:"),
        *script_parts,
        types.Part.from_text(text=prompt),
    ]
    return _generate_json_with_repair(client, model, contents, EXTRACTION_OUTPUT_SCHEMA)


def call_gemini_for_staged_grading(
    client: Any,
    case: ExperimentCase,
    extraction: dict[str, Any],
    question_parts: list[Any],
    guide_parts: list[Any],
    prompt: str,
    model: str,
) -> ModelCallResult:
    from google.genai import types

    contents = [
        types.Part.from_text(text=f"Official question paper for script {case.script_id}:"),
        *question_parts,
        types.Part.from_text(text="Official scanned marking guide/model answer:"),
        *guide_parts,
        types.Part.from_text(
            text="Saved extraction JSON from extraction/OCR stage:\n"
            + json.dumps(extraction, indent=2, sort_keys=True)
        ),
        types.Part.from_text(text=prompt),
    ]
    return _generate_json_with_repair(client, model, contents, GRADE_OUTPUT_SCHEMA)


def _array(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_page_numbers(value: Any) -> list[int]:
    numbers = []
    for item in _array(value):
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            continue
    return numbers


def finalize_extraction(
    case: ExperimentCase, model_output: dict[str, Any], duration_ms: int
) -> dict[str, Any]:
    errors: list[str] = []
    raw_questions = _array(model_output.get("questions"))
    normalized_questions: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for question in raw_questions:
        if not isinstance(question, dict):
            errors.append("extraction question row is not an object")
            continue
        try:
            qid = normalize_question_id(question.get("question_id"))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if qid in normalized_questions:
            duplicates.append(qid)
            continue
        normalized = dict(question)
        normalized["question_id"] = qid
        normalized["attempted"] = bool(normalized.get("attempted"))
        normalized["page_numbers"] = _normalize_page_numbers(normalized.get("page_numbers"))
        normalized["answer_text"] = str(normalized.get("answer_text", ""))
        normalized["subparts"] = _array(normalized.get("subparts"))
        normalized["evidence_notes"] = str(normalized.get("evidence_notes", ""))
        normalized["transcription_confidence"] = _safe_float(
            normalized.get("transcription_confidence"), 0.0
        )
        normalized_questions[qid] = normalized
    for duplicate in sorted(set(duplicates)):
        errors.append(f"duplicate extraction question_id {duplicate}")
    for qid in QUESTION_ORDER:
        if qid not in normalized_questions:
            errors.append(f"extraction missing {qid} row")
            normalized_questions[qid] = {
                "question_id": qid,
                "attempted": False,
                "page_numbers": [],
                "answer_text": "",
                "subparts": [],
                "evidence_notes": "missing extraction row inserted by finalizer",
                "transcription_confidence": 0.0,
            }
    missing = sorted(
        set(str(item) for item in _array(model_output.get("missing_or_blank_questions")))
        | {qid for qid, row in normalized_questions.items() if not row.get("attempted")}
    )
    return {
        "script_id": case.script_id,
        "script_filename": case.script_filename,
        "script_path": case.script_path,
        "page_count": case.page_count,
        "pages_reported": case.pages_reported,
        "student_identifier_seen": model_output.get("student_identifier_seen"),
        "pages_observed": _array(model_output.get("pages_observed")),
        "questions": [normalized_questions[qid] for qid in QUESTION_ORDER],
        "missing_or_blank_questions": missing,
        "ignored_annotations": _array(model_output.get("ignored_annotations")),
        "transcription_warnings": _array(model_output.get("transcription_warnings")),
        "extraction_validation_errors": errors,
        "duration_ms": duration_ms,
        "raw_model_response_path": None,
        "repair_model_response_path": None,
        "json_repair_used": False,
        "model_request_attempts": 0,
        "model_repair_attempts": 0,
    }


def _extraction_attempt_map(extraction: dict[str, Any]) -> dict[str, bool]:
    attempts = {qid: False for qid in QUESTION_ORDER}
    for question in _array(extraction.get("questions")):
        if not isinstance(question, dict):
            continue
        try:
            qid = normalize_question_id(question.get("question_id"))
        except ValueError:
            continue
        attempts[qid] = bool(question.get("attempted"))
    return attempts


def _part_max_map(rubric: ExamRubric, qid: str) -> dict[str, float]:
    return {part.label: part.max_score for part in rubric.questions[qid].parts}


def raw_model_score_snapshot(model_output: dict[str, Any]) -> dict[str, Any]:
    raw_scores = {qid: 0.0 for qid in QUESTION_ORDER}
    model_maxes: dict[str, float | None] = {qid: None for qid in QUESTION_ORDER}
    subparts: dict[str, dict[str, Any]] = {qid: {} for qid in QUESTION_ORDER}
    errors: list[str] = []
    for index, question in enumerate(_array(model_output.get("graded_questions"))):
        if not isinstance(question, dict):
            errors.append(f"graded_questions[{index}] is not an object")
            continue
        try:
            qid = normalize_question_id(question.get("question_id"))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        raw_scores[qid] = _round(_safe_float(question.get("score"), 0.0)) or 0.0
        model_maxes[qid] = _round(_safe_float_or_none(question.get("max_score")))
        subparts[qid] = {
            str(part.get("part_label", "")).strip().lower(): {
                "score": _safe_float(part.get("score"), 0.0),
                "max_score": _safe_float_or_none(part.get("max_score")),
                "evidence": part.get("evidence"),
                "justification": part.get("justification"),
            }
            for part in _array(question.get("subpart_scores"))
            if isinstance(part, dict)
        }
    return {
        "raw_model_question_scores": raw_scores,
        "model_reported_question_max_scores": model_maxes,
        "raw_model_subpart_scores": subparts,
        "raw_all_questions_total_score": _round(sum(raw_scores.values())),
        "normalization_errors": errors,
    }


def _validate_and_score_questions(
    model_output: dict[str, Any], extraction: dict[str, Any], rubric: ExamRubric
) -> tuple[list[QuestionScoreForSelection], list[str], list[float], int, dict[str, Any]]:
    validation_errors: list[str] = []
    confidences: list[float] = []
    lecture_use_count = 0
    attempts = _extraction_attempt_map(extraction)
    snapshot = raw_model_score_snapshot(model_output)
    validation_errors.extend(snapshot["normalization_errors"])
    seen: set[str] = set()
    rows_by_qid: dict[str, dict[str, Any]] = {}
    for index, question in enumerate(_array(model_output.get("graded_questions"))):
        if not isinstance(question, dict):
            continue
        try:
            qid = normalize_question_id(question.get("question_id"))
        except ValueError:
            continue
        if qid in seen:
            validation_errors.append(f"duplicate graded question_id {qid}")
            continue
        seen.add(qid)
        rows_by_qid[qid] = question
        score = _safe_float(question.get("score"), 0.0)
        confidence = _safe_float(question.get("model_confidence"), -1.0)
        if score < 0:
            validation_errors.append(f"graded {qid}.score is negative")
        if 0 <= confidence <= 1:
            confidences.append(confidence)
        else:
            validation_errors.append(f"graded {qid}.model_confidence outside 0..1")
        if question.get("lecture_context_used"):
            lecture_use_count += 1
        corrected_question_max = rubric.questions[qid].raw_max_score
        if score > corrected_question_max:
            validation_errors.append(
                f"graded_questions[{index}].score exceeds corrected rubric max_score for {qid}"
            )
        if bool(question.get("attempted")) and not attempts[qid] and score > 0:
            validation_errors.append(f"graded {qid} has positive score despite extraction attempted=false")
        part_maxes = _part_max_map(rubric, qid)
        subparts = _array(question.get("subpart_scores"))
        if subparts:
            subpart_sum = 0.0
            for part in subparts:
                if not isinstance(part, dict):
                    validation_errors.append(f"graded {qid} subpart row is not an object")
                    continue
                label = str(part.get("part_label", "")).strip().lower()
                if label not in part_maxes:
                    validation_errors.append(f"graded {qid} has invalid subpart {label!r}")
                    continue
                part_score = _safe_float(part.get("score"), 0.0)
                reported_max = _safe_float_or_none(part.get("max_score"))
                if reported_max is None or abs(reported_max - part_maxes[label]) > 1e-9:
                    validation_errors.append(f"graded {qid}{label}.max_score does not match corrected rubric")
                if part_score < 0:
                    validation_errors.append(f"graded {qid}{label}.score is negative")
                if part_score > part_maxes[label]:
                    validation_errors.append(f"graded {qid}{label}.score exceeds corrected rubric max")
                subpart_sum += part_score
            if abs(subpart_sum - score) > 0.01:
                validation_errors.append(f"graded {qid}.score does not equal subpart score sum")
    for qid in QUESTION_ORDER:
        if qid not in rows_by_qid and attempts[qid]:
            validation_errors.append(f"extracted attempted {qid} missing from graded_questions")

    selection_inputs: list[QuestionScoreForSelection] = []
    for qid in QUESTION_ORDER:
        selection_inputs.append(
            question_score_for_selection(
                qid,
                snapshot["raw_model_question_scores"].get(qid, 0.0),
                snapshot["model_reported_question_max_scores"].get(qid),
                rubric,
                attempts[qid],
            )
        )
    return selection_inputs, validation_errors, confidences, lecture_use_count, snapshot


def finalize_staged_prediction(
    case: ExperimentCase,
    model_output: dict[str, Any],
    extraction: dict[str, Any],
    rubric: ExamRubric,
    lecture_mode: str,
    duration_ms: int,
) -> dict[str, Any]:
    graded_questions = _array(model_output.get("graded_questions"))
    validation_errors: list[str] = []
    if not graded_questions:
        validation_errors.append("graded_questions must be a non-empty array")
    selection_inputs, question_errors, confidences, lecture_use_count, snapshot = _validate_and_score_questions(
        model_output, extraction, rubric
    )
    validation_errors.extend(question_errors)
    selection = select_best_allowed_questions(selection_inputs, rubric)

    ignored_annotations = _array(model_output.get("ignored_annotations"))
    transcription_warnings = _array(model_output.get("transcription_warnings"))
    rubric_boundary_warnings = _array(model_output.get("rubric_boundary_warnings"))
    extraction_validation_errors = _array(extraction.get("extraction_validation_errors"))
    extraction_transcription_warnings = _array(extraction.get("transcription_warnings"))
    validation_errors.extend(f"extraction: {error}" for error in extraction_validation_errors)
    low_confidence_count = sum(confidence < LOW_CONFIDENCE_THRESHOLD for confidence in confidences)
    referee_status = (
        "PENDING_REVIEW"
        if low_confidence_count > 0
        or validation_errors
        or transcription_warnings
        or extraction_transcription_warnings
        or rubric_boundary_warnings
        else "COMPLETED"
    )
    ai_total = selection.ai_total_score
    extraction_attempts = _extraction_attempt_map(extraction)
    prediction = {
        "script_id": case.script_id,
        "script_filename": case.script_filename,
        "script_path": case.script_path,
        "human_total_score": case.human_total_score,
        "ai_total_score": ai_total,
        "max_score": rubric.total_max_score,
        "score_summary": f"{ai_total:g}/{rubric.total_max_score:g}",
        "absolute_error": _round(abs(ai_total - case.human_total_score)),
        "signed_error": _round(ai_total - case.human_total_score),
        "sum_question_max_score": _round(
            sum(rubric.questions[qid].raw_max_score for qid in QUESTION_ORDER)
        ),
        "mean_confidence": _round(mean(confidences)) if confidences else None,
        "min_confidence": _round(min(confidences)) if confidences else None,
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
        "extraction_limitations": _array(model_output.get("extraction_limitations")),
        "extraction_transcription_warning_count": len(extraction_transcription_warnings),
        "extraction_validation_error_count": len(extraction_validation_errors),
        "extracted_attempted_questions": [
            qid for qid in QUESTION_ORDER if extraction_attempts.get(qid, False)
        ],
        "extracted_unattempted_questions": [
            qid for qid in QUESTION_ORDER if not extraction_attempts.get(qid, False)
        ],
        "missing_or_blank_questions": [
            qid for qid in QUESTION_ORDER if not extraction_attempts.get(qid, False)
        ],
        "raw_model_subpart_scores": snapshot["raw_model_subpart_scores"],
        **asdict(selection),
        "question_equal_max_score": rubric.question_equal_max_score,
        "allowed_question_count": rubric.allowed_question_count,
        "selection_policy": rubric.selection_policy,
        "extraction_path": extraction.get("parsed_output_path"),
        "extraction_raw_model_response_path": extraction.get("raw_model_response_path"),
        "extraction_repair_model_response_path": extraction.get("repair_model_response_path"),
    }
    return prediction


def build_downstream_shadow_payload(
    prediction: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_payload = {
        "script_id": prediction["script_id"],
        "condition": prediction.get("lecture_mode"),
        "finalized_prediction": prediction,
        "extraction_path": prediction.get("extraction_path"),
    }
    pending = str(prediction.get("referee_status", "")).upper() == "PENDING_REVIEW"
    blocked = (
        [
            "persist_completed_result",
            "student_facing_feedback",
            "weakness_profile_update",
            "smartprep_generation",
        ]
        if pending
        else []
    )
    output_payload = {
        "script_id": prediction["script_id"],
        "condition": prediction.get("lecture_mode"),
        "would_persist_result": not pending,
        "would_route_review": pending,
        "would_generate_student_feedback": not pending,
        "would_feed_weakness_detection": not pending,
        "would_feed_smartprep": not pending,
        "blocked_actions": blocked,
    }
    return input_payload, output_payload


def reset_staged_output_files(out_dir: Path, overwrite: bool) -> None:
    tracked = [
        "manifest.json",
        "rubric.json",
        "lecture_context.json",
        "extractions.jsonl",
        "stage_io.jsonl",
        "pipeline_trace_index.json",
        "predictions.no_lecture.jsonl",
        "predictions.lecture.jsonl",
        "predictions.jsonl",
        "pairs.csv",
        "failures.jsonl",
        "metrics.json",
        "lecture_ablation_metrics.json",
        "run_events.jsonl",
        "run_summary.json",
        "experiment_prompt.extraction.txt",
        "experiment_prompt.none.txt",
        "experiment_prompt.summary.txt",
        "experiment_prompt.full.txt",
    ]
    existing = [out_dir / name for name in tracked if (out_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Output directory already contains staged artifacts ({names}); pass --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
        for name in ("raw_model_outputs", "stage_io", "tmp_parts", "paper_tables"):
            path = out_dir / name
            if path.exists():
                shutil.rmtree(path)


def _write_rubric_copy(out_dir: Path, rubric: ExamRubric) -> Path:
    path = out_dir / "rubric.json"
    write_json(path, rubric_to_json(rubric))
    return path


def _base_manifest(
    out_dir: Path,
    cases: list[ExperimentCase],
    label_book: Any,
    question_pdf: Path,
    guide_pdf: Path,
    lecture_materials: list[Path],
    rubric: ExamRubric,
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
            "rubric": rubric_to_json(rubric),
        },
    )


def _record_stage(
    out_dir: Path,
    script_id: str | None,
    condition: str | None,
    stage: str,
    input_path: Path,
    output_path: Path,
    raw_output_path: Path | None,
    parsed_output_path: Path | None,
    prompt_path: Path | None,
    started_at: str,
    duration_ms: int,
    status: str,
    validation_errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> None:
    append_stage_io_record(
        out_dir,
        StageIORecord(
            script_id=script_id,
            condition=condition,
            stage=stage,
            input_path=str(input_path),
            output_path=str(output_path),
            raw_output_path=str(raw_output_path) if raw_output_path else None,
            parsed_output_path=str(parsed_output_path) if parsed_output_path else None,
            prompt_path=str(prompt_path) if prompt_path else None,
            started_at=started_at,
            ended_at=utc_now(),
            duration_ms=duration_ms,
            status=status,
            validation_errors=validation_errors or [],
            warnings=warnings or [],
        ),
    )


def _extraction_input_payload(
    case: ExperimentCase,
    question_pdf: Path,
    prompt_path: Path,
    model: str,
    project: str | None,
    location: str,
) -> dict[str, Any]:
    return {
        "stage": "extraction",
        "script_id": case.script_id,
        "model": model,
        "project": project,
        "location": location,
        "temperature": 0,
        "prompt_path": str(prompt_path),
        "prompt_sha256": _prompt_hash(prompt_path),
        "request_content_sequence": [
            file_artifact(question_pdf, "official_question_paper", "application/pdf"),
            file_artifact(Path(case.script_path), "student_script", "application/pdf", case.page_count),
            file_artifact(prompt_path, "extraction_prompt", "text/plain"),
        ],
        "excluded_inputs": {
            "marking_guide": True,
            "corrected_rubric": True,
            "lecture_context": True,
            "human_labels": True,
        },
    }


def _grading_input_payload(
    case: ExperimentCase,
    condition: str,
    question_pdf: Path,
    guide_pdf: Path,
    rubric_path: Path,
    extraction_path: Path,
    lecture_context_path: Path | None,
    prompt_path: Path,
    model: str,
    project: str | None,
    location: str,
) -> dict[str, Any]:
    payload = {
        "stage": "grading",
        "condition": condition,
        "script_id": case.script_id,
        "model": model,
        "project": project,
        "location": location,
        "temperature": 0,
        "prompt_path": str(prompt_path),
        "prompt_sha256": _prompt_hash(prompt_path),
        "question_paper": file_artifact(question_pdf, "official_question_paper", "application/pdf"),
        "marking_guide": file_artifact(guide_pdf, "scanned_marking_guide", "application/pdf"),
        "corrected_rubric": json_artifact(rubric_path, "corrected_rubric"),
        "saved_extraction": json_artifact(extraction_path, "saved_extraction"),
        "student_script_pdf_included": False,
    }
    if lecture_context_path:
        payload["lecture_context"] = json_artifact(lecture_context_path, "lecture_context")
    return payload


def _manual_findings() -> dict[str, Any]:
    return {
        "scope": "targeted tutor/source inspection supplied before corrected staged rerun",
        "findings": [
            {
                "script_id": "30",
                "finding": "There was no Question 4; both the original grader and second lecturer scored the script above the nominal maximum; this is a marking-guide allocation/total-policy issue, not a simple human or AI grading error.",
            },
            {
                "script_id": "53",
                "finding": "There is really only one student page; partial/incomplete handling should be treated as a correct one-page/lazy-student case, not an OCR truncation failure.",
            },
            {
                "script_id": "11",
                "finding": "There is no Question 1; the AI high award can be justified by strong student performance, but the model failed to account for manually corrected mark allocations written poorly in pen over printed scores.",
            },
            {
                "script_id": "3",
                "question_id": "q5",
                "part_label": "c",
                "finding": "The student answer is for a function returning a square and a loop printing 1-5; the prior model rationale wrongly described an even/odd check program. If the numeric score remains justified, classify this as rationale/localization error rather than score error.",
            },
        ],
    }


def run_extraction_stage(
    cases: list[ExperimentCase],
    question_pdf: Path,
    out_dir: Path,
    model: str,
    project: str | None,
    location: str,
    max_part_mb: float,
    dry_run: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    prompt = extraction_prompt_text()
    prompt_path = out_dir / "experiment_prompt.extraction.txt"
    prompt_path.write_text(prompt)
    tmp_dir = out_dir / "tmp_parts" / "extractions"
    question_parts = [] if dry_run else prepare_pdf_parts(question_pdf, tmp_dir / "question", max_part_mb)
    client = None if dry_run else create_gemini_client(project, location)
    raw_dir = out_dir / "raw_model_outputs" / "extractions"
    extractions: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for case in cases:
        started_at = utc_now()
        start = time.monotonic()
        input_path = write_stage_input(
            out_dir,
            "extraction",
            None,
            case.script_id,
            _extraction_input_payload(case, question_pdf, prompt_path, model, project, location),
        )
        append_jsonl(
            out_dir / "run_events.jsonl",
            {"event": "extraction_started", "script_id": case.script_id, "timestamp": started_at},
        )
        try:
            if dry_run:
                extraction = finalize_extraction(case, {"questions": []}, 0)
                raw_path = None
                repair_path = None
                parsed_path = raw_dir / f"{case.script_id}.extracted.json"
            else:
                script_parts = prepare_pdf_parts(Path(case.script_path), tmp_dir / case.script_id, max_part_mb)
                result = call_gemini_for_extraction(
                    client=client,
                    case=case,
                    question_parts=question_parts,
                    script_parts=script_parts,
                    prompt=prompt,
                    model=model,
                )
                raw_path = raw_dir / f"{case.script_id}.raw.txt"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(result.raw_text)
                repair_path = None
                if result.repair_text is not None:
                    repair_path = raw_dir / f"{case.script_id}.repair.txt"
                    repair_path.write_text(result.repair_text)
                duration_ms = int((time.monotonic() - start) * 1000)
                extraction = finalize_extraction(case, result.model_output, duration_ms)
                extraction.update(
                    {
                        "raw_model_response_path": str(raw_path),
                        "repair_model_response_path": str(repair_path) if repair_path else None,
                        "json_repair_used": result.repair_text is not None,
                        "model_request_attempts": result.request_attempts,
                        "model_repair_attempts": result.repair_attempts,
                    }
                )
                parsed_path = raw_dir / f"{case.script_id}.extracted.json"
            extraction["parsed_output_path"] = str(parsed_path)
            write_json(parsed_path, extraction)
            output_path = write_stage_output(
                out_dir,
                "extraction",
                None,
                case.script_id,
                {
                    "raw_response": str(raw_path) if raw_path else None,
                    "repair_response": str(repair_path) if repair_path else None,
                    "parsed_extraction": str(parsed_path),
                    "finalized_extraction": extraction,
                    "extraction_validation_errors": extraction["extraction_validation_errors"],
                    "transcription_warnings": extraction["transcription_warnings"],
                },
            )
            append_jsonl(out_dir / "extractions.jsonl", extraction)
            extractions[case.script_id] = extraction
            duration_ms = int((time.monotonic() - start) * 1000)
            _record_stage(
                out_dir,
                case.script_id,
                None,
                "extraction",
                input_path,
                output_path,
                raw_path,
                parsed_path,
                prompt_path,
                started_at,
                duration_ms,
                "succeeded" if not extraction["extraction_validation_errors"] else "succeeded_with_warnings",
                extraction["extraction_validation_errors"],
                extraction["transcription_warnings"],
            )
            append_jsonl(
                out_dir / "run_events.jsonl",
                {
                    "event": "extraction_succeeded",
                    "script_id": case.script_id,
                    "duration_ms": duration_ms,
                    "timestamp": utc_now(),
                    "validation_error_count": len(extraction["extraction_validation_errors"]),
                },
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            failure = failure_record(case, "extraction_failed", str(exc))
            failures.append(failure)
            output_path = write_stage_output(
                out_dir,
                "extraction",
                None,
                case.script_id,
                {"error": str(exc)},
            )
            _record_stage(
                out_dir,
                case.script_id,
                None,
                "extraction",
                input_path,
                output_path,
                None,
                None,
                prompt_path,
                started_at,
                duration_ms,
                "failed",
                [str(exc)],
                [],
            )
            append_jsonl(
                out_dir / "run_events.jsonl",
                {
                    "event": "extraction_failed",
                    "script_id": case.script_id,
                    "duration_ms": duration_ms,
                    "error": str(exc),
                    "timestamp": utc_now(),
                },
            )
    return extractions, failures


def load_existing_extractions(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    loaded = {str(row["script_id"]): row for row in rows}
    if len(loaded) != len(rows):
        raise ValueError("Duplicate script_id in extractions")
    return loaded


def run_staged_prediction_set(
    cases: list[ExperimentCase],
    labels: dict[str, dict[str, Any]],
    question_pdf: Path,
    guide_pdf: Path,
    rubric: ExamRubric,
    rubric_path: Path,
    extractions_by_script: dict[str, dict[str, Any]],
    lecture_context: dict[str, Any],
    lecture_context_path: Path,
    lecture_mode: str,
    out_dir: Path,
    output_name: str,
    model: str,
    project: str | None,
    location: str,
    max_part_mb: float,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    condition = "no_lecture" if lecture_mode == "none" else "lecture"
    prompt = grade_prompt_text(rubric, lecture_context, lecture_mode)
    prompt_path = out_dir / f"experiment_prompt.{lecture_mode}.txt"
    prompt_path.write_text(prompt)
    tmp_dir = out_dir / "tmp_parts" / output_name.replace(".", "_")
    question_parts = [] if dry_run else prepare_pdf_parts(question_pdf, tmp_dir / "question", max_part_mb)
    guide_parts = [] if dry_run else prepare_pdf_parts(guide_pdf, tmp_dir / "guide", max_part_mb)
    client = None if dry_run else create_gemini_client(project, location)
    raw_dir = out_dir / "raw_model_outputs" / Path(output_name).stem.replace(".", "_")
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        if case.script_id not in extractions_by_script:
            failure = failure_record(case, "missing_extraction", "No extraction row for script")
            failures.append(failure)
            continue
        extraction = extractions_by_script[case.script_id]
        extraction_path = Path(extraction["parsed_output_path"])
        started_at = utc_now()
        start = time.monotonic()
        input_path = write_stage_input(
            out_dir,
            "grading",
            condition,
            case.script_id,
            _grading_input_payload(
                case,
                condition,
                question_pdf,
                guide_pdf,
                rubric_path,
                extraction_path,
                lecture_context_path if lecture_mode != "none" else None,
                prompt_path,
                model,
                project,
                location,
            ),
        )
        append_jsonl(
            out_dir / "run_events.jsonl",
            {
                "event": "grading_started",
                "script_id": case.script_id,
                "condition": condition,
                "lecture_mode": lecture_mode,
                "output_name": output_name,
                "timestamp": started_at,
            },
        )
        try:
            if dry_run:
                model_output = {"graded_questions": []}
                raw_path = None
                repair_path = None
                result = None
            else:
                result = call_gemini_for_staged_grading(
                    client=client,
                    case=case,
                    extraction=extraction,
                    question_parts=question_parts,
                    guide_parts=guide_parts,
                    prompt=prompt,
                    model=model,
                )
                model_output = result.model_output
                raw_path = raw_dir / f"{case.script_id}.raw.txt"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(result.raw_text)
                repair_path = None
                if result.repair_text is not None:
                    repair_path = raw_dir / f"{case.script_id}.repair.txt"
                    repair_path.write_text(result.repair_text)
            parsed_path = raw_dir / f"{case.script_id}.parsed.json"
            write_json(parsed_path, model_output)
            snapshot = raw_model_score_snapshot(model_output)
            grading_output_path = write_stage_output(
                out_dir,
                "grading",
                condition,
                case.script_id,
                {
                    "raw_response": str(raw_path) if raw_path else None,
                    "repair_response": str(repair_path) if repair_path else None,
                    "parsed_model_json": str(parsed_path),
                    "model_output": model_output,
                    "raw_model_score_snapshot": snapshot,
                    "model_request_attempts": result.request_attempts if result else 0,
                    "model_repair_attempts": result.repair_attempts if result else 0,
                },
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            _record_stage(
                out_dir,
                case.script_id,
                condition,
                "grading",
                input_path,
                grading_output_path,
                raw_path,
                parsed_path,
                prompt_path,
                started_at,
                duration_ms,
                "succeeded",
                snapshot.get("normalization_errors", []),
                _array(model_output.get("transcription_warnings")) + _array(model_output.get("rubric_boundary_warnings")),
            )

            final_started = utc_now()
            final_start = time.monotonic()
            final_input_path = write_stage_input(
                out_dir,
                "finalization",
                condition,
                case.script_id,
                {
                    "stage": "finalization",
                    "condition": condition,
                    "script_id": case.script_id,
                    "parsed_grading_json": json_artifact(parsed_path, "parsed_grading_json"),
                    "raw_model_score_snapshot": snapshot,
                    "extraction_output": json_artifact(extraction_path, "extraction_output"),
                    "corrected_rubric": json_artifact(rubric_path, "corrected_rubric"),
                    "human_total_label": labels[case.script_id],
                    "deterministic_constants": {
                        "LOW_CONFIDENCE_THRESHOLD": LOW_CONFIDENCE_THRESHOLD,
                        "selection_policy": rubric.selection_policy,
                        "allowed_question_count": rubric.allowed_question_count,
                        "question_equal_max_score": rubric.question_equal_max_score,
                    },
                },
            )
            prediction = finalize_staged_prediction(
                case, model_output, extraction, rubric, lecture_mode, duration_ms
            )
            prediction.update(
                {
                    "raw_model_response_path": str(raw_path) if raw_path else None,
                    "repair_model_response_path": str(repair_path) if repair_path else None,
                    "json_repair_used": bool(result and result.repair_text is not None),
                    "prompt_path": str(prompt_path),
                    "model_request_attempts": result.request_attempts if result else 0,
                    "model_repair_attempts": result.repair_attempts if result else 0,
                }
            )
            final_output_path = write_stage_output(
                out_dir,
                "finalization",
                condition,
                case.script_id,
                prediction,
            )
            shadow_input, shadow_output = build_downstream_shadow_payload(prediction)
            shadow_input_path = write_stage_input(
                out_dir, "downstream_shadow", condition, case.script_id, shadow_input
            )
            shadow_output_path = write_stage_output(
                out_dir, "downstream_shadow", condition, case.script_id, shadow_output
            )
            prediction["stage_io"] = {
                "grading_input_path": str(input_path),
                "grading_output_path": str(grading_output_path),
                "finalization_input_path": str(final_input_path),
                "finalization_output_path": str(final_output_path),
                "downstream_shadow_input_path": str(shadow_input_path),
                "downstream_shadow_output_path": str(shadow_output_path),
            }
            write_json(final_output_path, prediction)
            shadow_input["finalized_prediction"] = prediction
            write_json(shadow_input_path, shadow_input)
            append_jsonl(out_dir / output_name, prediction)
            predictions.append(prediction)
            final_duration_ms = int((time.monotonic() - final_start) * 1000)
            _record_stage(
                out_dir,
                case.script_id,
                condition,
                "finalization",
                final_input_path,
                final_output_path,
                None,
                final_output_path,
                None,
                final_started,
                final_duration_ms,
                "succeeded" if not prediction["validation_errors"] else "succeeded_with_warnings",
                prediction["validation_errors"],
                prediction["transcription_warnings"] + prediction["rubric_boundary_warnings"],
            )
            _record_stage(
                out_dir,
                case.script_id,
                condition,
                "downstream_shadow",
                shadow_input_path,
                shadow_output_path,
                None,
                shadow_output_path,
                None,
                utc_now(),
                0,
                "succeeded",
                [],
                [],
            )
            append_jsonl(
                out_dir / "run_events.jsonl",
                {
                    "absolute_error": prediction["absolute_error"],
                    "ai_total_score": prediction["ai_total_score"],
                    "raw_all_questions_total_score": prediction["raw_all_questions_total_score"],
                    "raw_attempted_total_score": prediction["raw_attempted_total_score"],
                    "duration_ms": duration_ms,
                    "event": "grading_succeeded",
                    "human_total_score": prediction["human_total_score"],
                    "json_repair_used": prediction["json_repair_used"],
                    "lecture_mode": lecture_mode,
                    "condition": condition,
                    "low_confidence_count": prediction["low_confidence_count"],
                    "mean_confidence": prediction["mean_confidence"],
                    "output_name": output_name,
                    "raw_model_response_path": str(raw_path) if raw_path else None,
                    "referee_status": prediction["referee_status"],
                    "script_filename": case.script_filename,
                    "script_id": case.script_id,
                    "timestamp": utc_now(),
                    "validation_error_count": len(prediction["validation_errors"]),
                },
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            failure = failure_record(case, "grading_failed", str(exc))
            failure.update({"duration_ms": duration_ms, "lecture_mode": lecture_mode, "output_name": output_name})
            failures.append(failure)
            output_path = write_stage_output(out_dir, "grading", condition, case.script_id, {"error": str(exc)})
            _record_stage(
                out_dir,
                case.script_id,
                condition,
                "grading",
                input_path,
                output_path,
                None,
                None,
                prompt_path,
                started_at,
                duration_ms,
                "failed",
                [str(exc)],
                [],
            )
            append_jsonl(
                out_dir / "run_events.jsonl",
                {
                    "duration_ms": duration_ms,
                    "error": str(exc),
                    "error_code": "grading_failed",
                    "event": "grading_failed",
                    "lecture_mode": lecture_mode,
                    "condition": condition,
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


def write_staged_run_summary(
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
            "extraction_model": args.extraction_model or args.model,
            "project": args.project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            "location": args.location,
            "allow_full_run_after_review": args.allow_full_run_after_review,
            "approval_note": args.approval_note,
            "paths": {
                "out_dir": str(out_dir),
                "manifest": str(out_dir / "manifest.json"),
                "rubric_json": str(out_dir / "rubric.json"),
                "lecture_context": str(out_dir / "lecture_context.json"),
                "extractions": str(out_dir / "extractions.jsonl"),
                "predictions": str(out_dir / "predictions.jsonl"),
                "predictions_no_lecture": str(out_dir / "predictions.no_lecture.jsonl"),
                "predictions_lecture": str(out_dir / "predictions.lecture.jsonl"),
                "failures": str(out_dir / "failures.jsonl"),
                "pairs": str(out_dir / "pairs.csv"),
                "metrics": str(out_dir / "metrics.json"),
                "lecture_ablation_metrics": str(out_dir / "lecture_ablation_metrics.json"),
                "run_events": str(out_dir / "run_events.jsonl"),
                "raw_model_outputs": str(out_dir / "raw_model_outputs"),
                "raw_extraction_outputs": str(out_dir / "raw_model_outputs" / "extractions"),
                "experiment_prompt_extraction": str(out_dir / "experiment_prompt.extraction.txt"),
                "experiment_prompt_none": str(out_dir / "experiment_prompt.none.txt"),
                "experiment_prompt_summary": str(out_dir / "experiment_prompt.summary.txt"),
                "paper_tables": str(out_dir / "paper_tables"),
            },
            "stage_io": {
                "stage_io_jsonl": str(out_dir / "stage_io.jsonl"),
                "pipeline_trace_index": str(out_dir / "pipeline_trace_index.json"),
                "stage_io_dir": str(out_dir / "stage_io"),
            },
        },
    )


def _dry_run_stage_io(
    out_dir: Path,
    cases: list[ExperimentCase],
    question_pdf: Path,
    guide_pdf: Path,
    rubric_path: Path,
    lecture_context_path: Path,
    rubric: ExamRubric,
    args: argparse.Namespace,
) -> None:
    extraction_prompt = out_dir / "experiment_prompt.extraction.txt"
    extraction_prompt.write_text(extraction_prompt_text())
    for mode, condition in (("none", "no_lecture"), (args.lecture_mode, "lecture")):
        prompt_path = out_dir / f"experiment_prompt.{mode}.txt"
        prompt_path.write_text(grade_prompt_text(rubric, {}, mode))
    for case in cases:
        started = utc_now()
        input_path = write_stage_input(
            out_dir,
            "extraction",
            None,
            case.script_id,
            _extraction_input_payload(case, question_pdf, extraction_prompt, args.extraction_model or args.model, args.project, args.location),
        )
        output_path = write_stage_output(out_dir, "extraction", None, case.script_id, {"dry_run": True})
        _record_stage(out_dir, case.script_id, None, "extraction", input_path, output_path, None, output_path, extraction_prompt, started, 0, "dry_run")
        for mode, condition in (("none", "no_lecture"), (args.lecture_mode, "lecture")) if args.run_lecture_ablation else ((args.lecture_mode, "lecture"),):
            prompt_path = out_dir / f"experiment_prompt.{mode}.txt"
            fake_extraction_path = output_path
            grading_input = write_stage_input(
                out_dir,
                "grading",
                condition,
                case.script_id,
                _grading_input_payload(case, condition, question_pdf, guide_pdf, rubric_path, fake_extraction_path, lecture_context_path if mode != "none" else None, prompt_path, args.model, args.project, args.location),
            )
            grading_output = write_stage_output(out_dir, "grading", condition, case.script_id, {"dry_run": True})
            _record_stage(out_dir, case.script_id, condition, "grading", grading_input, grading_output, None, grading_output, prompt_path, started, 0, "dry_run")
            final_input = write_stage_input(out_dir, "finalization", condition, case.script_id, {"dry_run": True})
            final_output = write_stage_output(out_dir, "finalization", condition, case.script_id, {"dry_run": True})
            _record_stage(out_dir, case.script_id, condition, "finalization", final_input, final_output, None, final_output, None, started, 0, "dry_run")
            shadow_input = write_stage_input(out_dir, "downstream_shadow", condition, case.script_id, {"dry_run": True})
            shadow_output = write_stage_output(out_dir, "downstream_shadow", condition, case.script_id, {"dry_run": True})
            _record_stage(out_dir, case.script_id, condition, "downstream_shadow", shadow_input, shadow_output, None, shadow_output, None, started, 0, "dry_run")


def run_experiment(args: argparse.Namespace) -> int:
    started_at = utc_now()
    scripts_dir = Path(args.scripts_dir)
    scores_pdf = Path(args.scores_pdf)
    question_pdf = Path(args.question_pdf) if args.question_pdf else scripts_dir / "question.pdf"
    guide_pdf = Path(args.guide_pdf) if args.guide_pdf else scripts_dir / "guide.pdf"
    out_dir = Path(args.out_dir)
    lecture_materials = [Path(item) for item in args.lecture_material]
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_staged_output_files(out_dir, args.overwrite)
    rubric = load_exam_rubric(Path(args.rubric_json))
    rubric_path = _write_rubric_copy(out_dir, rubric)
    label_book = parse_scores_pdf(scores_pdf)
    labels = labels_from_score_book(label_book)
    script_ids = parse_script_ids(args.script_ids)
    cases = build_manifest(scripts_dir, label_book, question_pdf, guide_pdf, script_ids)
    lecture_context = build_lecture_context(lecture_materials, args.lecture_mode)
    lecture_context_path = out_dir / "lecture_context.json"
    write_json(lecture_context_path, lecture_context)
    _base_manifest(out_dir, cases, label_book, question_pdf, guide_pdf, lecture_materials, rubric)

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    metrics: dict[str, Any]
    if args.dry_run:
        _dry_run_stage_io(out_dir, cases, question_pdf, guide_pdf, rubric_path, lecture_context_path, rubric, args)
        metrics = compute_total_score_metrics(labels, {})
        write_pairs_csv(out_dir / "pairs.csv", [])
    else:
        if args.use_existing_extractions:
            extractions_by_script = load_existing_extractions(Path(args.use_existing_extractions))
            extraction_failures: list[dict[str, Any]] = []
        else:
            extractions_by_script, extraction_failures = run_extraction_stage(
                cases,
                question_pdf,
                out_dir,
                args.extraction_model or args.model,
                args.project,
                args.location,
                args.max_part_mb,
                args.dry_run,
            )
        failures.extend(extraction_failures)
        if args.run_lecture_ablation:
            no_lecture_context = build_lecture_context([], "none")
            no_preds, no_failures, no_metrics = run_staged_prediction_set(
                cases,
                labels,
                question_pdf,
                guide_pdf,
                rubric,
                rubric_path,
                extractions_by_script,
                no_lecture_context,
                lecture_context_path,
                "none",
                out_dir,
                "predictions.no_lecture.jsonl",
                args.model,
                args.project,
                args.location,
                args.max_part_mb,
                False,
            )
            le_preds, le_failures, le_metrics = run_staged_prediction_set(
                cases,
                labels,
                question_pdf,
                guide_pdf,
                rubric,
                rubric_path,
                extractions_by_script,
                lecture_context,
                lecture_context_path,
                args.lecture_mode,
                out_dir,
                "predictions.lecture.jsonl",
                args.model,
                args.project,
                args.location,
                args.max_part_mb,
                False,
            )
            predictions = le_preds
            failures.extend(no_failures + le_failures)
            write_jsonl(out_dir / "predictions.jsonl", predictions)
            write_pairs_csv(out_dir / "pairs.csv", predictions)
            metrics = le_metrics
            ablation = compute_lecture_ablation_metrics(
                labels,
                {item["script_id"]: item for item in no_preds},
                {item["script_id"]: item for item in le_preds},
            )
            ablation["no_lecture_metrics"] = no_metrics
            ablation["lecture_metrics"] = le_metrics
            write_json(out_dir / "lecture_ablation_metrics.json", ablation)
        else:
            predictions, grade_failures, metrics = run_staged_prediction_set(
                cases,
                labels,
                question_pdf,
                guide_pdf,
                rubric,
                rubric_path,
                extractions_by_script,
                lecture_context,
                lecture_context_path,
                args.lecture_mode,
                out_dir,
                "predictions.jsonl",
                args.model,
                args.project,
                args.location,
                args.max_part_mb,
                False,
            )
            failures.extend(grade_failures)
            write_pairs_csv(out_dir / "pairs.csv", predictions)
        write_json(out_dir / "manual_findings.supplied.json", _manual_findings())

    write_jsonl(out_dir / "failures.jsonl", failures)
    write_json(out_dir / "metrics.json", metrics)
    write_pipeline_trace_index(out_dir)
    write_staged_run_summary(out_dir, args, cases, predictions, failures, started_at, utc_now())
    print(json.dumps({"out_dir": str(out_dir), "requested_count": len(cases), "succeeded_count": 0 if args.dry_run else len(predictions), "failed_count": len(failures), "dry_run": args.dry_run, "lecture_mode": args.lecture_mode}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the staged corrected-rubric Intro Python experiment.")
    parser.add_argument("--scripts-dir", required=True)
    parser.add_argument("--scores-pdf", required=True)
    parser.add_argument("--rubric-json", required=True)
    parser.add_argument("--question-pdf", default=None)
    parser.add_argument("--guide-pdf", default=None)
    parser.add_argument("--lecture-material", action="append", default=[])
    parser.add_argument("--script-ids", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--extraction-model", default=None)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    parser.add_argument("--max-part-mb", type=float, default=DEFAULT_MAX_PART_MB)
    parser.add_argument("--lecture-mode", choices=("none", "summary", "full"), default="summary")
    parser.add_argument("--run-lecture-ablation", action="store_true")
    parser.add_argument("--use-existing-extractions", default=None)
    parser.add_argument("--allow-full-run-after-review", action="store_true")
    parser.add_argument("--approval-note", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_full_run_gate(args, parser)
        raise SystemExit(run_experiment(args))
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
