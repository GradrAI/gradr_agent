"""Isolated downstream contract-replay runner for staged grading artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.downstream_metrics import compute_contract_metrics, write_tables
from eval.fault_injection import ArtifactVariant, inject_fault, make_negative_control, repair_fault, stable_hash
from eval.pipeline_contracts import (
    USER_VISIBLE_OR_PERSISTENT_ACTIONS,
    compute_review_gate,
    is_action_allowed,
    validate_grading_contract,
)
from eval.rubric_config import load_exam_rubric

ACTIONS = (
    "persist_completed_result",
    "student_facing_feedback",
    "weakness_profile_update",
    "smartprep_generation",
    "route_human_review",
)
MODEL_BACKED_ACTIONS = {
    "student_facing_feedback",
    "weakness_profile_update",
    "smartprep_generation",
    "persist_completed_result",
}
PROTOCOL_HASH_FILES = (
    "protocol.md",
    "fault_library.json",
    "action_contracts.json",
    "analysis_plan.md",
)
DEFAULT_PROTOCOL_DIR = Path(__file__).with_name("downstream_contract_protocol")
DEFAULT_FAULT_NAME = "drop_pending_review_status"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def protocol_hash(protocol_dir: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in PROTOCOL_HASH_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((protocol_dir / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def reset_output_dir(out_dir: Path, overwrite: bool) -> None:
    tracked = [
        "manifest.json",
        "case_results.jsonl",
        "stage_io.jsonl",
        "metrics.json",
        "planned_actions.json",
        "run_summary.json",
    ]
    existing = [out_dir / name for name in tracked if (out_dir / name).exists()]
    existing_dirs = [out_dir / name for name in ("stage_io", "raw_model_outputs", "prompts", "paper_tables") if (out_dir / name).exists()]
    if (existing or existing_dirs) and not overwrite:
        names = ", ".join(path.name for path in [*existing, *existing_dirs])
        raise FileExistsError(f"Output directory already contains downstream artifacts ({names}); pass --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()
        for path in existing_dirs:
            shutil.rmtree(path)


def load_predictions(source_run: Path) -> list[dict[str, Any]]:
    split_candidates = [
        ("no_lecture", source_run / "predictions.no_lecture.jsonl"),
        ("lecture", source_run / "predictions.lecture.jsonl"),
    ]
    available_split = [(condition, path) for condition, path in split_candidates if path.exists()]
    candidates = available_split or [("single", source_run / "predictions.jsonl")]
    rows: list[dict[str, Any]] = []
    for condition, path in candidates:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("condition", condition)
            row.setdefault("lecture_condition", condition)
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No prediction JSONL files found in {source_run}")
    return rows


def _observed_variant(payload: dict[str, Any]) -> ArtifactVariant:
    digest = stable_hash(payload)
    return ArtifactVariant(
        variant_name="observed",
        fault_name=None,
        payload=json.loads(json.dumps(payload)),
        expected_violation_codes=(),
        expected_blocked_actions=(),
        repair_name=None,
        before_hash=digest,
        after_hash=digest,
    )


def build_variants(
    payload: dict[str, Any], variant_names: list[str], fault_name: str
) -> list[ArtifactVariant]:
    variants: list[ArtifactVariant] = []
    controlled: ArtifactVariant | None = None
    for variant_name in variant_names:
        if variant_name == "observed":
            variants.append(_observed_variant(payload))
        elif variant_name == "negative_control":
            variants.append(make_negative_control(payload, fault_name))
        elif variant_name == "controlled_fault":
            controlled = inject_fault(payload, fault_name)
            variants.append(controlled)
        elif variant_name == "repaired_counterfactual":
            if controlled is None:
                controlled = inject_fault(payload, fault_name)
            variants.append(repair_fault(controlled))
        else:
            raise ValueError(f"Unknown condition/variant: {variant_name}")
    return variants


def _safe_name(*parts: str | None) -> str:
    joined = "__".join(str(part or "none") for part in parts)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in joined)


def _stage_paths(out_dir: Path, stage: str, row_id: str) -> tuple[Path, Path]:
    base = out_dir / "stage_io" / stage
    return base / f"{row_id}.input.json", base / f"{row_id}.output.json"


def _record_stage(
    out_dir: Path,
    script_id: str,
    condition: str,
    variant_name: str,
    stage: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
    status: str,
    started_at: str,
    duration_ms: int,
    model_call_attempted: bool = False,
    raw_output_path: Path | None = None,
    prompt_path: Path | None = None,
) -> None:
    row_id = _safe_name(script_id, condition, variant_name)
    input_path, output_path = _stage_paths(out_dir, stage, row_id)
    write_json(input_path, input_payload)
    write_json(output_path, output_payload)
    append_jsonl(
        out_dir / "stage_io.jsonl",
        {
            "script_id": script_id,
            "condition": condition,
            "variant_name": variant_name,
            "stage": stage,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "raw_output_path": str(raw_output_path) if raw_output_path else None,
            "prompt_path": str(prompt_path) if prompt_path else None,
            "status": status,
            "started_at": started_at,
            "ended_at": utc_now(),
            "duration_ms": duration_ms,
            "model_call_attempted": model_call_attempted,
        },
    )


def _student_context(payload: dict[str, Any]) -> dict[str, Any]:
    context = payload.get("_student_context") if isinstance(payload.get("_student_context"), dict) else {}
    return {"student_id": payload.get("script_id", "anonymous-script"), "linked_user_id": "anonymous-linked-user", **context}


def _exam_context(payload: dict[str, Any]) -> dict[str, Any]:
    context = payload.get("_exam_context") if isinstance(payload.get("_exam_context"), dict) else {}
    return {"exam_type": "wassce", "max_score": payload.get("max_score", 60.0), **context}


def _proposal_prompt(payload: dict[str, Any], allowed_actions: list[str]) -> str:
    compact_payload = {
        "script_id": payload.get("script_id"),
        "ai_total_score": payload.get("ai_total_score"),
        "max_score": payload.get("max_score"),
        "referee_status": payload.get("referee_status"),
        "validation_errors": payload.get("validation_errors", []),
        "transcription_warnings": payload.get("transcription_warnings", []),
        "graded_questions": payload.get("graded_questions", []),
        "allowed_actions": allowed_actions,
    }
    return (
        "You are running an isolated research replay of downstream assessment actions. "
        "Do not claim to persist records, contact students, call databases, or call external tools. "
        "Return strict JSON only with keys feedback_proposal, weakness_proposal, "
        "practice_proposal, and result_persistence_proposal. For actions not listed in "
        "allowed_actions, set the corresponding proposal to null.\n\n"
        f"Input artifact:\n{json.dumps(compact_payload, indent=2, sort_keys=True)}"
    )


def _parse_json(raw: str) -> dict[str, Any]:
    stripped = raw.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def call_model_for_proposals(
    payload: dict[str, Any],
    allowed_actions: list[str],
    model: str,
    project: str | None,
    location: str,
) -> tuple[dict[str, Any], str, int]:
    from eval.intro_python_experiment import create_gemini_client, generate_content_with_retries
    from google.genai import types

    client = create_gemini_client(project, location)
    prompt = _proposal_prompt(payload, allowed_actions)
    config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
    response, attempts = generate_content_with_retries(
        client=client,
        model=model,
        contents=[types.Part.from_text(text=prompt)],
        config=config,
    )
    raw_text = response.text or ""
    return _parse_json(raw_text), raw_text, attempts


def _dry_proposals(payload: dict[str, Any], allowed_actions: list[str]) -> dict[str, Any]:
    return {
        "feedback_proposal": (
            {"dry_run": True, "script_id": payload.get("script_id"), "feedback": "local dry-run feedback proposal"}
            if "student_facing_feedback" in allowed_actions
            else None
        ),
        "weakness_proposal": (
            {"dry_run": True, "weakTopics": ["programming fundamentals"]}
            if "weakness_profile_update" in allowed_actions
            else None
        ),
        "practice_proposal": (
            {"dry_run": True, "questions": []}
            if "smartprep_generation" in allowed_actions
            else None
        ),
        "result_persistence_proposal": (
            {"dry_run": True, "score": f"{payload.get('ai_total_score')}/{payload.get('max_score')}"}
            if "persist_completed_result" in allowed_actions
            else None
        ),
    }


def _proposal_for_action(bundle: dict[str, Any], action: str) -> Any:
    key_by_action = {
        "student_facing_feedback": "feedback_proposal",
        "weakness_profile_update": "weakness_proposal",
        "smartprep_generation": "practice_proposal",
        "persist_completed_result": "result_persistence_proposal",
    }
    return bundle.get(key_by_action[action])


def process_variant(
    out_dir: Path,
    variant: ArtifactVariant,
    rubric: Any,
    dry_run: bool,
    model: str,
    project: str | None,
    location: str,
) -> dict[str, Any]:
    payload = variant.payload
    script_id = str(payload.get("script_id"))
    condition = str(payload.get("lecture_condition") or payload.get("condition") or payload.get("lecture_mode") or "unknown")
    violations = validate_grading_contract(payload, rubric)
    decision = compute_review_gate(payload, violations)
    student_context = _student_context(payload)
    exam_context = _exam_context(payload)
    action_decisions: dict[str, dict[str, Any]] = {}
    for action in ACTIONS:
        allowed = is_action_allowed(action, decision, student_context, exam_context)
        action_decisions[action] = {
            "allowed": allowed,
            "expected_blocked": action in variant.expected_blocked_actions,
            "reason": "allowed" if allowed else "blocked_by_contract_or_context",
        }

    started = utc_now()
    start = time.monotonic()
    review_output = {"review_decision": asdict(decision), "violations": [asdict(item) for item in violations]}
    _record_stage(
        out_dir,
        script_id,
        condition,
        variant.variant_name,
        "review_gate",
        {"artifact_hash": stable_hash(payload)},
        review_output,
        "succeeded",
        started,
        int((time.monotonic() - start) * 1000),
    )

    allowed_model_actions = [
        action for action in MODEL_BACKED_ACTIONS if action_decisions[action]["allowed"]
    ]
    model_attempts = 0
    model_failures = 0
    parse_failures = 0
    raw_output_path: Path | None = None
    prompt_path: Path | None = None
    bundle: dict[str, Any]
    if dry_run:
        bundle = _dry_proposals(payload, allowed_model_actions)
    elif allowed_model_actions:
        prompt = _proposal_prompt(payload, allowed_model_actions)
        row_id = _safe_name(script_id, condition, variant.variant_name)
        prompt_path = out_dir / "prompts" / f"{row_id}.downstream_proposal_bundle.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt)
        started = utc_now()
        start = time.monotonic()
        try:
            bundle, raw_text, model_attempts = call_model_for_proposals(
                payload, allowed_model_actions, model, project, location
            )
            raw_output_path = out_dir / "raw_model_outputs" / "downstream_proposal_bundle" / f"{row_id}.raw.txt"
            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_output_path.write_text(raw_text)
            status = "succeeded"
        except Exception as exc:
            bundle = {
                "feedback_proposal": None,
                "weakness_proposal": None,
                "practice_proposal": None,
                "result_persistence_proposal": None,
                "error": str(exc),
            }
            model_failures = 1
            parse_failures = 1
            status = "failed"
        _record_stage(
            out_dir,
            script_id,
            condition,
            variant.variant_name,
            "downstream_proposal_bundle",
            {"allowed_actions": allowed_model_actions, "artifact_hash": stable_hash(payload)},
            bundle,
            status,
            started,
            int((time.monotonic() - start) * 1000),
            model_call_attempted=True,
            raw_output_path=raw_output_path,
            prompt_path=prompt_path,
        )
    else:
        bundle = _dry_proposals(payload, [])

    proposals: dict[str, Any] = {}
    for action in MODEL_BACKED_ACTIONS:
        stage = {
            "student_facing_feedback": "feedback_proposal",
            "weakness_profile_update": "weakness_proposal",
            "smartprep_generation": "practice_proposal",
            "persist_completed_result": "result_persistence_proposal",
        }[action]
        started = utc_now()
        start = time.monotonic()
        proposal = _proposal_for_action(bundle, action) if action_decisions[action]["allowed"] else None
        output = {
            "action": action,
            "allowed": action_decisions[action]["allowed"],
            "proposal": proposal,
        }
        proposals[action] = proposal
        _record_stage(
            out_dir,
            script_id,
            condition,
            variant.variant_name,
            stage,
            {"action": action, "artifact_hash": stable_hash(payload)},
            output,
            "succeeded" if action_decisions[action]["allowed"] else "blocked",
            started,
            int((time.monotonic() - start) * 1000),
        )

    return {
        "script_id": script_id,
        "condition": condition,
        "variant_name": variant.variant_name,
        "fault_name": variant.fault_name,
        "before_hash": variant.before_hash,
        "after_hash": variant.after_hash,
        "expected_violation_codes": list(variant.expected_violation_codes),
        "detected_violation_codes": [item.code for item in violations],
        "expected_blocked_actions": list(variant.expected_blocked_actions),
        "review_decision": asdict(decision),
        "action_decisions": action_decisions,
        "proposals": proposals,
        "model_call_attempt_count": model_attempts,
        "model_call_failure_count": model_failures,
        "parse_failure_count": parse_failures,
    }


def run_experiment(args: argparse.Namespace) -> int:
    source_run = Path(args.source_run)
    out_dir = Path(args.out_dir)
    protocol_dir = Path(args.protocol_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_output_dir(out_dir, args.overwrite)
    current_protocol_hash = protocol_hash(protocol_dir)
    if args.all_scripts and args.require_protocol_hash != current_protocol_hash:
        raise ValueError("confirmatory run requires matching --require-protocol-hash")

    variant_names = parse_csv(args.conditions) or ["observed"]
    script_ids = set(parse_csv(args.script_ids))
    rubric = load_exam_rubric(Path(args.rubric_json))
    predictions = load_predictions(source_run)
    selected = [
        row
        for row in predictions
        if not script_ids or str(row.get("script_id")) in script_ids
    ]
    if not selected:
        raise ValueError("no predictions matched requested script ids")

    manifest = {
        "source_run": str(source_run),
        "rubric_json": str(args.rubric_json),
        "protocol_dir": str(protocol_dir),
        "protocol_hash": current_protocol_hash,
        "script_ids": sorted({str(row.get("script_id")) for row in selected}, key=lambda item: int(item) if item.isdigit() else item),
        "prediction_count": len(selected),
        "variant_names": variant_names,
        "fault_name": args.fault_name,
        "dry_run": args.dry_run,
        "model": args.model,
        "location": args.location,
        "started_at": utc_now(),
    }
    write_json(out_dir / "manifest.json", manifest)

    case_count = 0
    planned_actions: list[dict[str, Any]] = []
    for prediction in selected:
        variants = build_variants(prediction, variant_names, args.fault_name)
        for variant in variants:
            result = process_variant(
                out_dir,
                variant,
                rubric,
                args.dry_run,
                args.model,
                args.project,
                args.location,
            )
            append_jsonl(out_dir / "case_results.jsonl", result)
            case_count += 1
            for action, decision in result["action_decisions"].items():
                planned_actions.append(
                    {
                        "script_id": result["script_id"],
                        "condition": result["condition"],
                        "variant_name": result["variant_name"],
                        "action": action,
                        "allowed": decision["allowed"],
                    }
                )

    write_json(out_dir / "planned_actions.json", planned_actions)
    metrics = compute_contract_metrics(out_dir)
    write_json(out_dir / "metrics.json", metrics)
    write_tables(out_dir, out_dir / "paper_tables")
    write_json(
        out_dir / "run_summary.json",
        {
            "case_count": case_count,
            "prediction_count": len(selected),
            "variant_count": len(variant_names),
            "dry_run": args.dry_run,
            "protocol_hash": current_protocol_hash,
            "completed_at": utc_now(),
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "case_count": case_count, "dry_run": args.dry_run}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay downstream contract propagation from staged grading artifacts.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--rubric-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--script-ids", default=None)
    parser.add_argument("--all-scripts", action="store_true")
    parser.add_argument("--conditions", default="observed,negative_control,controlled_fault,repaired_counterfactual")
    parser.add_argument("--fault-name", default=DEFAULT_FAULT_NAME)
    parser.add_argument("--protocol-dir", default=str(DEFAULT_PROTOCOL_DIR))
    parser.add_argument("--require-protocol-hash", default=None)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.script_ids and args.all_scripts:
        parser.error("--script-ids cannot be combined with --all-scripts")
    if not args.script_ids and not args.all_scripts:
        parser.error("Pass --script-ids for a bounded run or --all-scripts for confirmatory execution")
    try:
        raise SystemExit(run_experiment(args))
    except FileExistsError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
