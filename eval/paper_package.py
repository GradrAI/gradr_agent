"""Paper-package generator for downstream contract replay runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.downstream_metrics import compute_contract_metrics, write_tables

SUPPORTED_TARGETS = {"cain2027"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def schema_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, Any] = {}
    for row in rows:
        for key, value in row.items():
            fields[key].add(_type_name(value))
            examples.setdefault(key, value)
    return {
        "row_count": len(rows),
        "fields": {
            key: {
                "types": sorted(types),
                "example_shape": _shape(examples[key]),
            }
            for key, types in sorted(fields.items())
        },
    }


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _type_name(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        if not value:
            return []
        return [_shape(value[0])]
    return _type_name(value)


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def _expected_blocked_action_allowed(row: dict[str, Any]) -> bool:
    actions = row.get("action_decisions") or {}
    for action in row.get("expected_blocked_actions") or []:
        if (actions.get(action) or {}).get("allowed"):
            return True
    return False


def _user_action_blocked(row: dict[str, Any]) -> bool:
    actions = row.get("action_decisions") or {}
    user_actions = {
        "persist_completed_result",
        "student_facing_feedback",
        "weakness_profile_update",
        "smartprep_generation",
    }
    return any(action in user_actions and not decision.get("allowed") for action, decision in actions.items())


def variant_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("variant_name", ""))].append(row)
    outcomes: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(by_variant.items()):
        outcomes.append(
            {
                "variant_name": variant,
                "case_count": len(variant_rows),
                "expected_violation_cases": sum(bool(row.get("expected_violation_codes")) for row in variant_rows),
                "detected_violation_cases": sum(bool(row.get("detected_violation_codes")) for row in variant_rows),
                "propagated_expected_blocked_action_cases": sum(
                    _expected_blocked_action_allowed(row) for row in variant_rows
                ),
                "user_action_blocked_cases": sum(_user_action_blocked(row) for row in variant_rows),
                "pending_review_cases": sum(
                    (row.get("review_decision") or {}).get("status") == "PENDING_REVIEW"
                    for row in variant_rows
                ),
                "model_call_attempt_count": sum(int(row.get("model_call_attempt_count") or 0) for row in variant_rows),
                "model_call_failure_count": sum(int(row.get("model_call_failure_count") or 0) for row in variant_rows),
                "parse_failure_count": sum(int(row.get("parse_failure_count") or 0) for row in variant_rows),
            }
        )
    return outcomes


def stage_outcomes(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        buckets[str(row.get("stage", ""))].append(row)
    outcomes: list[dict[str, Any]] = []
    for stage, rows in sorted(buckets.items()):
        statuses = Counter(str(row.get("status", "")) for row in rows)
        outcomes.append(
            {
                "stage": stage,
                "record_count": len(rows),
                "model_call_attempted_count": sum(bool(row.get("model_call_attempted")) for row in rows),
                "succeeded_count": statuses.get("succeeded", 0),
                "failed_count": statuses.get("failed", 0),
                "blocked_count": statuses.get("blocked", 0),
                "skipped_count": statuses.get("skipped", 0),
            }
        )
    return outcomes


def artifact_counts(run_dir: Path) -> dict[str, int]:
    prompt_files = list((run_dir / "prompts").glob("**/*")) if (run_dir / "prompts").exists() else []
    raw_files = list((run_dir / "raw_model_outputs").glob("**/*")) if (run_dir / "raw_model_outputs").exists() else []
    parsed_files = list((run_dir / "stage_io").glob("**/*.json")) if (run_dir / "stage_io").exists() else []
    return {
        "prompt_files": sum(path.is_file() for path in prompt_files),
        "raw_model_output_files": sum(path.is_file() for path in raw_files),
        "parsed_proposal_files": sum(path.is_file() for path in parsed_files),
    }


def prompt_hashes(run_dir: Path) -> list[dict[str, Any]]:
    prompt_dir = run_dir / "prompts"
    if not prompt_dir.exists():
        return []
    hashes: list[dict[str, Any]] = []
    for path in sorted(prompt_dir.glob("**/*")):
        if path.is_file():
            hashes.append(
                {
                    "path": str(path.relative_to(prompt_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return hashes


def protocol_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    protocol_dir_raw = manifest.get("protocol_dir")
    protocol_dir = Path(protocol_dir_raw) if protocol_dir_raw else None
    recorded = None
    if protocol_dir and (protocol_dir / "protocol_hash.txt").exists():
        recorded = (protocol_dir / "protocol_hash.txt").read_text().splitlines()[0].strip()
    manifest_hash = manifest.get("protocol_hash")
    return {
        "manifest_hash": manifest_hash,
        "recorded_hash": recorded,
        "manifest_hash_matches_recorded": None if recorded is None else manifest_hash == recorded,
        "protocol_dir": protocol_dir_raw,
    }


def build_numbers(run_dir: Path, target: str) -> dict[str, Any]:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"Unsupported target: {target}")
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    run_summary = read_json(run_dir / "run_summary.json") if (run_dir / "run_summary.json").exists() else {}
    case_rows = read_jsonl(run_dir / "case_results.jsonl")
    stage_rows = read_jsonl(run_dir / "stage_io.jsonl")
    metrics = compute_contract_metrics(run_dir)
    recorded_metrics = read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else None
    expected_case_count = run_summary.get("case_count", manifest.get("prediction_count", 0) * len(manifest.get("variant_names", [])))
    production_side_effect_records = [
        row
        for row in stage_rows
        if row.get("production_write_attempted") or "production" in str(row.get("stage", "")).lower()
    ]
    return {
        "generated_at": utc_now(),
        "target": target,
        "run_dir": str(run_dir),
        "manifest": manifest,
        "run_summary": run_summary,
        "metrics": metrics,
        "recorded_metrics_match_recomputed": None if recorded_metrics is None else recorded_metrics == metrics,
        "counts_by_variant": count_by(case_rows, "variant_name"),
        "counts_by_condition": count_by(case_rows, "condition"),
        "review_status_counts": dict(
            sorted(
                Counter(str((row.get("review_decision") or {}).get("status", "")) for row in case_rows).items()
            )
        ),
        "variant_outcomes": variant_outcomes(case_rows),
        "stage_outcomes": stage_outcomes(stage_rows),
        "artifact_counts": artifact_counts(run_dir),
        "prompt_hashes": prompt_hashes(run_dir),
        "protocol": protocol_summary(manifest),
        "run_checks": {
            "not_dry_run": manifest.get("dry_run") is False,
            "all_cases_accounted_for": expected_case_count == len(case_rows),
            "stage_records_present": bool(stage_rows),
            "metrics_recomputed_from_case_results": True,
            "no_production_side_effect_records": not production_side_effect_records,
            "production_side_effect_record_count": len(production_side_effect_records),
        },
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No rows.\n")
        return
    columns = sorted({key for row in rows for key in row})
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_additional_tables(numbers: dict[str, Any], tables_dir: Path) -> None:
    _write_rows_csv(tables_dir / "variant_outcomes.csv", numbers["variant_outcomes"])
    _write_rows_md(tables_dir / "variant_outcomes.md", numbers["variant_outcomes"])
    _write_rows_csv(tables_dir / "stage_outcomes.csv", numbers["stage_outcomes"])
    _write_rows_md(tables_dir / "stage_outcomes.md", numbers["stage_outcomes"])


def copy_protocol(manifest: dict[str, Any], out_dir: Path) -> None:
    target = out_dir / "protocol"
    target.mkdir(parents=True, exist_ok=True)
    protocol_dir_raw = manifest.get("protocol_dir")
    if protocol_dir_raw:
        protocol_dir = Path(protocol_dir_raw)
        if protocol_dir.exists():
            for path in sorted(protocol_dir.iterdir()):
                if path.is_file():
                    shutil.copy2(path, target / path.name)
            return
    (target / "protocol_hash.txt").write_text(str(manifest.get("protocol_hash", "")) + "\n")


def copy_prompts(run_dir: Path, out_dir: Path) -> None:
    prompt_dir = run_dir / "prompts"
    target = out_dir / "prompts"
    target.mkdir(parents=True, exist_ok=True)
    if not prompt_dir.exists():
        return
    for path in sorted(prompt_dir.glob("**/*")):
        if path.is_file():
            relative = path.relative_to(prompt_dir)
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / relative)


def write_schemas(run_dir: Path, out_dir: Path) -> None:
    schema_dir = out_dir / "schemas"
    write_json(schema_dir / "case_results.schema.json", schema_for_rows(read_jsonl(run_dir / "case_results.jsonl")))
    write_json(schema_dir / "stage_io.schema.json", schema_for_rows(read_jsonl(run_dir / "stage_io.jsonl")))
    manifest = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    write_json(schema_dir / "manifest.schema.json", schema_for_rows([manifest] if manifest else []))


def write_claim_boundary(path: Path, numbers: dict[str, Any]) -> None:
    metrics = numbers["metrics"]
    lines = [
        "# Claim boundary",
        "",
        "## Supported by this package",
        "",
        f"- Case count: {metrics['case_count']} isolated replay cases.",
        f"- Contract-violation detection rate: {metrics['contract_violation_detection_rate']}.",
        f"- Gate containment rate on controlled faults: {metrics['gate_containment_rate']}.",
        f"- Action-propagation rate on controlled faults: {metrics['action_propagation_rate']}.",
        f"- False-block rate on negative controls: {metrics['false_block_rate']}.",
        f"- Model-call failures: {metrics['model_call_failure_count']} of {metrics['model_call_attempt_count']} attempts.",
        "- Isolated replay/proposal stages, not live production execution.",
        "",
        "## Not supported by this package",
        "",
        "- No qualified educator annotation is available; do not claim teacher-verified feedback quality, pedagogical correctness, learning benefit, fairness, or student harm reduction.",
        "- Do not claim production safety from isolated replay alone.",
        "- Do not generalize beyond this course/exam corpus, fault library, model setting, and frozen protocol without additional evidence.",
        "",
        "## Required manuscript phrasing",
        "",
        "Use mechanically checked contract compliance, action propagation, review-gate behavior, cost/latency, and reproducible audit artifacts as the evidence boundary.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_checklist(path: Path, numbers: dict[str, Any]) -> None:
    checks = numbers["run_checks"]
    lines = [
        "# CAIN 2027 package checklist",
        "",
        "## Venue constraints",
        "",
        "- Target: CAIN 2027 Research Track.",
        "- Page limit to verify before submission: full papers 10 pages plus 2 reference pages; short papers 5 pages plus 2 reference pages.",
        "- Deadline recorded in planning register: 2026-10-30.",
        "- Double-blind/anonymization: prepare anonymous manuscript unless the portal states otherwise.",
        "",
        "## Experiment gates",
        "",
        f"- [{'x' if checks['not_dry_run'] else ' '}] Package is based on a non-dry confirmatory run.",
        f"- [{'x' if checks['all_cases_accounted_for'] else ' '}] All planned cases have case-result records.",
        f"- [{'x' if checks['stage_records_present'] else ' '}] Stage records are present.",
        f"- [{'x' if checks['no_production_side_effect_records'] else ' '}] No production side-effect records are present.",
        f"- [{'x' if numbers['protocol']['manifest_hash_matches_recorded'] else ' '}] Manifest protocol hash matches recorded protocol hash.",
        "",
        "## Disclosure text required in manuscript",
        "",
        "- AI Use Statement: disclose AI assistance in coding, drafting, and editing as required by the venue.",
        "- Overlap disclosure: disclose relationship to prior accepted/rejected grading-pipeline manuscripts and state that this paper evaluates a distinct controlled contract-propagation experiment.",
        "- Limitations: include missing educator annotations, one course/exam, synthetic controlled faults, isolated replay instead of production execution, and model/time specificity.",
        "- Artifact statement: state which prompts, schemas, protocol hash, metrics, and generated tables are included.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_ai_use_statement(path: Path) -> None:
    path.write_text(
        "# AI Use Statement draft\n\n"
        "AI tools were used to assist with experiment-runner implementation, debugging, result packaging, and manuscript drafting/editing. "
        "All executable claims in the manuscript should be checked against the frozen protocol, recorded run artifacts, and generated numbers.json before submission.\n"
    )


def generate_package(run_dir: Path, target: str, out_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    numbers = build_numbers(run_dir, target)
    write_json(out_dir / "numbers.json", numbers)
    write_tables(run_dir, out_dir / "tables")
    write_additional_tables(numbers, out_dir / "tables")
    copy_protocol(numbers["manifest"], out_dir)
    copy_prompts(run_dir, out_dir)
    write_json(out_dir / "prompts" / "prompt_hashes.json", numbers["prompt_hashes"])
    write_schemas(run_dir, out_dir)
    write_claim_boundary(out_dir / "claim_boundary.md", numbers)
    write_checklist(out_dir / "checklist.md", numbers)
    write_ai_use_statement(out_dir / "ai_use_statement.md")
    shutil.copy2(run_dir / "manifest.json", out_dir / "manifest.json")
    if (run_dir / "run_summary.json").exists():
        shutil.copy2(run_dir / "run_summary.json", out_dir / "run_summary.json")
    return numbers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a paper package from a downstream contract replay run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target", default="cain2027", choices=sorted(SUPPORTED_TARGETS))
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    numbers = generate_package(Path(args.run_dir), args.target, Path(args.out_dir))
    print(json.dumps({"out_dir": args.out_dir, "case_count": numbers["metrics"]["case_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
