"""Metrics for downstream contract replay runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

USER_VISIBLE_OR_PERSISTENT_ACTIONS = {
    "persist_completed_result",
    "student_facing_feedback",
    "weakness_profile_update",
    "smartprep_generation",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _action_allowed(row: dict[str, Any], action: str) -> bool:
    decision = (row.get("action_decisions") or {}).get(action) or {}
    return bool(decision.get("allowed"))


def _any_expected_blocked_action_allowed(row: dict[str, Any]) -> bool:
    return any(_action_allowed(row, action) for action in row.get("expected_blocked_actions") or [])


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("script_id", "")), str(row.get("condition", "")))


def _negative_control_blocks_observed_action(row: dict[str, Any], observed: dict[str, Any] | None) -> bool:
    if observed is None:
        return False
    negative_actions = row.get("action_decisions") or {}
    observed_actions = observed.get("action_decisions") or {}
    for action in USER_VISIBLE_OR_PERSISTENT_ACTIONS:
        if (observed_actions.get(action) or {}).get("allowed") and not (negative_actions.get(action) or {}).get("allowed"):
            return True
    return False


def compute_contract_metrics(run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(Path(run_dir) / "case_results.jsonl")
    case_count = len(rows)
    expected_violation_rows = [row for row in rows if row.get("expected_violation_codes")]
    detected_expected = [
        row
        for row in expected_violation_rows
        if set(row.get("expected_violation_codes") or [])
        & set(row.get("detected_violation_codes") or [])
    ]
    faulty_rows = [row for row in rows if row.get("variant_name") == "controlled_fault"]
    propagated = [row for row in faulty_rows if _any_expected_blocked_action_allowed(row)]
    contained = [row for row in faulty_rows if not _any_expected_blocked_action_allowed(row)]
    observed_by_key = {
        _row_key(row): row for row in rows if row.get("variant_name") == "observed"
    }
    negative_rows = [row for row in rows if row.get("variant_name") == "negative_control"]
    false_blocked = [
        row
        for row in negative_rows
        if _negative_control_blocks_observed_action(row, observed_by_key.get(_row_key(row)))
    ]
    review_rows = [row for row in rows if (row.get("review_decision") or {}).get("status") == "PENDING_REVIEW"]
    model_attempts = sum(int(row.get("model_call_attempt_count") or 0) for row in rows)
    model_failures = sum(int(row.get("model_call_failure_count") or 0) for row in rows)
    parse_failures = sum(int(row.get("parse_failure_count") or 0) for row in rows)

    by_action: dict[str, dict[str, int]] = {}
    for row in rows:
        for action, decision in (row.get("action_decisions") or {}).items():
            bucket = by_action.setdefault(action, {"allowed": 0, "blocked": 0})
            if decision.get("allowed"):
                bucket["allowed"] += 1
            else:
                bucket["blocked"] += 1

    return {
        "case_count": case_count,
        "expected_violation_case_count": len(expected_violation_rows),
        "faulty_case_count": len(faulty_rows),
        "negative_control_case_count": len(negative_rows),
        "contract_violation_detection_rate": _rate(len(detected_expected), len(expected_violation_rows)),
        "action_propagation_rate": _rate(len(propagated), len(faulty_rows)),
        "gate_containment_rate": _rate(len(contained), len(faulty_rows)),
        "false_block_rate": _rate(len(false_blocked), len(negative_rows)),
        "review_demand_rate": _rate(len(review_rows), case_count),
        "model_call_attempt_count": model_attempts,
        "model_call_failure_count": model_failures,
        "parse_failure_count": parse_failures,
        "model_call_failure_rate": _rate(model_failures, model_attempts),
        "parse_failure_rate": _rate(parse_failures, model_attempts),
        "by_action": by_action,
    }


def write_tables(run_dir: Path, tables_dir: Path) -> None:
    metrics = compute_contract_metrics(run_dir)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    scalar_items = [
        (key, value)
        for key, value in metrics.items()
        if not isinstance(value, dict)
    ]
    with (tables_dir / "contract_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in scalar_items:
            writer.writerow([key, "" if value is None else value])
    markdown_lines = ["| metric | value |", "|---|---:|"]
    for key, value in scalar_items:
        markdown_lines.append(f"| {key} | {'' if value is None else value} |")
    (tables_dir / "contract_metrics.md").write_text("\n".join(markdown_lines) + "\n")

    with (tables_dir / "action_decisions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["action", "allowed", "blocked"])
        for action, counts in sorted(metrics["by_action"].items()):
            writer.writerow([action, counts["allowed"], counts["blocked"]])
