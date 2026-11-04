"""Deterministic artifact variants for downstream contract replay."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_DIR = Path(__file__).with_name("downstream_contract_protocol")
FAULT_LIBRARY_PATH = PROTOCOL_DIR / "fault_library.json"


@dataclass(frozen=True)
class ArtifactVariant:
    variant_name: str
    fault_name: str | None
    payload: dict[str, Any]
    expected_violation_codes: tuple[str, ...]
    expected_blocked_actions: tuple[str, ...]
    repair_name: str | None
    before_hash: str
    after_hash: str


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_fault_library(path: Path = FAULT_LIBRARY_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _fault_spec(fault_name: str) -> dict[str, Any]:
    library = load_fault_library()
    for item in library.get("faults", []):
        if item.get("name") == fault_name:
            return item
    raise ValueError(f"unknown fault: {fault_name}")


def make_negative_control(
    artifact: dict[str, Any], fault_name: str = "drop_pending_review_status"
) -> ArtifactVariant:
    payload = copy.deepcopy(artifact)
    digest = stable_hash(payload)
    return ArtifactVariant(
        variant_name="negative_control",
        fault_name=fault_name,
        payload=payload,
        expected_violation_codes=(),
        expected_blocked_actions=(),
        repair_name=None,
        before_hash=digest,
        after_hash=digest,
    )


def inject_fault(artifact: dict[str, Any], fault_name: str) -> ArtifactVariant:
    spec = _fault_spec(fault_name)
    payload = copy.deepcopy(artifact)
    before_hash = stable_hash(payload)
    _apply_fault(payload, fault_name)
    return ArtifactVariant(
        variant_name="controlled_fault",
        fault_name=fault_name,
        payload=payload,
        expected_violation_codes=tuple(spec.get("expected_violation_codes", [])),
        expected_blocked_actions=tuple(spec.get("expected_blocked_actions", [])),
        repair_name=spec.get("repair"),
        before_hash=before_hash,
        after_hash=stable_hash(payload),
    )


def repair_fault(variant: ArtifactVariant) -> ArtifactVariant:
    payload = copy.deepcopy(variant.payload)
    before_hash = stable_hash(payload)
    repair_name = variant.repair_name
    if repair_name is None:
        raise ValueError(f"variant has no repair: {variant.fault_name}")
    _apply_repair(payload, repair_name)
    return ArtifactVariant(
        variant_name="repaired_counterfactual",
        fault_name=variant.fault_name,
        payload=payload,
        expected_violation_codes=(),
        expected_blocked_actions=(),
        repair_name=repair_name,
        before_hash=before_hash,
        after_hash=stable_hash(payload),
    )


def _first_question_id(payload: dict[str, Any]) -> str:
    scores = payload.get("raw_model_question_scores")
    if isinstance(scores, dict) and scores:
        return sorted(str(key) for key in scores)[0]
    return "q1"


def _apply_fault(payload: dict[str, Any], fault_name: str) -> None:
    if fault_name == "score_exceeds_question_max":
        qid = _first_question_id(payload)
        maxes = payload.get("corrected_raw_question_max_scores") or {}
        max_score = float(maxes.get(qid, payload.get("max_score", 60)))
        payload.setdefault("raw_model_question_scores", {})[qid] = max_score + 1.0
        return
    if fault_name == "total_exceeds_exam_max":
        payload["ai_total_score"] = float(payload.get("max_score", 60)) + 1.0
        return
    if fault_name == "raw_normalized_mismatch":
        payload["ai_total_score"] = float(payload.get("ai_total_score", 0)) + 5.0
        return
    if fault_name == "too_many_selected_questions":
        selected = [str(qid) for qid in payload.get("selected_question_ids", [])]
        for qid in ("q1", "q2", "q3", "q4", "q5"):
            if qid not in selected:
                selected.append(qid)
        payload["selected_question_ids"] = selected
        return
    if fault_name == "question_id_mismatch":
        scores = payload.setdefault("raw_model_question_scores", {})
        if "q1" in scores:
            scores["q99"] = scores.pop("q1")
        else:
            scores["q99"] = 1.0
        return
    if fault_name == "invalid_confidence":
        payload["mean_confidence"] = 1.2
        return
    if fault_name == "empty_graded_questions":
        payload["graded_questions"] = []
        return
    if fault_name == "drop_pending_review_status":
        payload.setdefault("validation_errors", ["controlled pending-review marker"])
        if not payload["validation_errors"]:
            payload["validation_errors"] = ["controlled pending-review marker"]
        payload["referee_status"] = "COMPLETED"
        return
    if fault_name == "drop_validation_errors":
        payload["ai_total_score"] = float(payload.get("max_score", 60)) + 1.0
        payload["validation_errors"] = []
        return
    if fault_name == "unlinked_student_practice_request":
        payload["_student_context"] = {"linked_user_id": ""}
        return
    if fault_name == "unsupported_exam_type_practice_request":
        payload["_exam_context"] = {"exam_type": "ncee"}
        return
    if fault_name == "drop_extraction_warning":
        payload["_dropped_extraction_warning"] = True
        payload["transcription_warnings"] = []
        payload["extraction_transcription_warning_count"] = 0
        return
    if fault_name == "lecture_context_as_authority":
        questions = payload.setdefault("graded_questions", [{}])
        if not questions:
            questions.append({})
        questions[0]["source_priority"] = "lecture_context"
        return
    if fault_name == "absent_evidence_citation":
        questions = payload.setdefault("graded_questions", [{}])
        if not questions:
            questions.append({})
        questions[0]["justification"] = f"{questions[0].get('justification', '')} ABSENT_EVIDENCE"
        return
    raise ValueError(f"unhandled fault: {fault_name}")


def _apply_repair(payload: dict[str, Any], repair_name: str) -> None:
    if repair_name == "clip_question_score_to_max":
        maxes = payload.get("corrected_raw_question_max_scores") or {}
        scores = payload.get("raw_model_question_scores") or {}
        for qid, max_score in maxes.items():
            if qid in scores:
                scores[qid] = min(float(scores[qid]), float(max_score))
        return
    if repair_name == "cap_total_score":
        payload["ai_total_score"] = min(float(payload.get("ai_total_score", 0)), float(payload.get("max_score", 60)))
        return
    if repair_name == "recompute_total_from_questions":
        selected = [str(qid) for qid in payload.get("selected_question_ids", [])]
        normalized = payload.get("normalized_question_scores") or {}
        payload["ai_total_score"] = min(
            sum(float(normalized.get(qid, 0)) for qid in selected),
            float(payload.get("max_score", 60)),
        )
        return
    if repair_name == "trim_selected_questions_to_allowed_count":
        allowed = int(payload.get("allowed_question_count") or 4)
        payload["selected_question_ids"] = list(payload.get("selected_question_ids", []))[:allowed]
        return
    if repair_name == "restore_canonical_question_ids":
        scores = payload.get("raw_model_question_scores") or {}
        if "q99" in scores:
            scores["q1"] = scores.pop("q99")
        return
    if repair_name == "clip_confidence_to_unit_interval":
        for key in ("mean_confidence", "min_confidence"):
            if key in payload:
                payload[key] = min(max(float(payload[key]), 0.0), 1.0)
        return
    if repair_name == "restore_graded_questions":
        payload["graded_questions"] = payload.get("_original_graded_questions") or [
            {"question_id": "q1", "score": 0, "max_score": 18, "model_confidence": 0.7}
        ]
        return
    if repair_name == "restore_pending_review_status":
        payload["referee_status"] = "PENDING_REVIEW"
        return
    if repair_name == "restore_validation_errors":
        payload["validation_errors"] = payload.get("validation_errors") or ["restored validation error"]
        return
    if repair_name in {"block_practice_until_linked", "block_unsupported_exam_type"}:
        payload["referee_status"] = "PENDING_REVIEW"
        return
    if repair_name == "restore_extraction_warning":
        payload.pop("_dropped_extraction_warning", None)
        payload["transcription_warnings"] = payload.get("transcription_warnings") or ["restored extraction warning"]
        payload["extraction_transcription_warning_count"] = max(
            int(payload.get("extraction_transcription_warning_count") or 0), 1
        )
        payload["referee_status"] = "PENDING_REVIEW"
        return
    if repair_name == "restore_corrected_rubric_priority":
        for question in payload.get("graded_questions") or []:
            if question.get("source_priority") == "lecture_context":
                question["source_priority"] = "corrected_rubric"
        return
    if repair_name == "remove_absent_evidence_citation":
        for question in payload.get("graded_questions") or []:
            if "justification" in question:
                question["justification"] = str(question["justification"]).replace("ABSENT_EVIDENCE", "").strip()
        return
    raise ValueError(f"unhandled repair: {repair_name}")
