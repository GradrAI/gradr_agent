"""Grading accuracy evaluation: compute QWK, MAE, and calibration metrics.

Usage:
    uv run python -m eval.compute_metrics --gold eval/gold_standard.json --output eval/results.json
    uv run python -m eval.compute_metrics --gold eval/gold_standard.json --mongo-uri "mongodb://..." --output eval/results.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from eval.per_question_score_utils import (
    QUESTION_IDS,
    PerQuestionScoreBook,
    parse_per_question_scores_csv,
    question_attempt_set,
)
from eval.total_score_utils import (
    compute_lecture_ablation_metrics,
    compute_total_score_metrics,
    labels_from_score_book,
    load_predictions_jsonl,
    load_total_labels_file,
    parse_scores_pdf,
    prediction_confidence,
    prediction_hitl,
    prediction_total,
)
from eval.rubric_config import (
    QUESTION_ORDER,
    ExamRubric,
    load_exam_rubric,
    question_score_for_selection,
    select_best_allowed_questions,
)

try:
    from sklearn.metrics import cohen_kappa_score, mean_absolute_error
except ImportError:
    print(
        "ERROR: scikit-learn is required. Install it with:\n"
        "  uv add scikit-learn\n"
        "or:\n"
        "  pip install scikit-learn",
        file=sys.stderr,
    )
    sys.exit(1)



def load_gold_standard(path: str) -> list[dict]:
    """Load gold-standard grading records from JSON."""
    with open(path) as f:
        entries = json.load(f)
    if not isinstance(entries, list) or len(entries) == 0:
        print(f"ERROR: {path} must be a non-empty JSON array.", file=sys.stderr)
        sys.exit(1)
    required = {"eval_id", "max_score", "human_score"}
    for i, entry in enumerate(entries):
        missing = required - set(entry.keys())
        if missing:
            print(
                f"ERROR: entry {i} ({entry.get('eval_id', '?')}) missing fields: {missing}",
                file=sys.stderr,
            )
            sys.exit(1)
    return entries


def load_ai_scores_from_mongo(
    mongo_uri: str, entries: list[dict]
) -> dict[str, dict]:
    """Load AI grading results from MongoDB, keyed by eval_id.

    Matches gold-standard entries to MongoDB results by question_id within
    recent results that have gradingMeta populated.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        print(
            "ERROR: pymongo is required for --mongo-uri. Install with:\n"
            "  uv add pymongo",
            file=sys.stderr,
        )
        sys.exit(1)

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    results_col = db["results"]

    ai_scores: dict[str, dict] = {}
    for entry in entries:
        qid = entry.get("question_id")
        if not qid:
            continue
        # Find the most recent result containing this question with gradingMeta
        doc = results_col.find_one(
            {
                "gradingMeta": {"$exists": True},
                "results.questionId": qid,
            },
            sort=[("createdAt", -1)],
        )
        if not doc:
            continue

        # Extract per-question AI score
        q_index = None
        for idx, r in enumerate(doc.get("results", [])):
            if r.get("questionId") == qid:
                q_index = idx
                break
        if q_index is None:
            continue

        meta = doc.get("gradingMeta", {})
        confidences = meta.get("confidences", [])
        q_result = doc["results"][q_index]

        # Parse score from "X/Y" format
        raw_score = q_result.get("score", "0/0")
        if isinstance(raw_score, str) and "/" in raw_score:
            ai_score = float(raw_score.split("/")[0])
        else:
            ai_score = float(raw_score)

        ai_scores[entry["eval_id"]] = {
            "ai_score": ai_score,
            "confidence": confidences[q_index] if q_index < len(confidences) else None,
            "mean_confidence": meta.get("meanConfidence"),
        }

    client.close()
    return ai_scores


def compute_qwk(human_scores: list[float], ai_scores: list[float]) -> float:
    """Compute Quadratic Weighted Kappa between human and AI scores."""
    return float(cohen_kappa_score(human_scores, ai_scores, weights="quadratic"))


def compute_mae(human_scores: list[float], ai_scores: list[float]) -> float:
    """Compute Mean Absolute Error."""
    return float(mean_absolute_error(human_scores, ai_scores))


def compute_exact_match_rate(
    human_scores: list[float], ai_scores: list[float]
) -> float:
    """Fraction of scores where AI exactly matches human."""
    if not human_scores:
        return 0.0
    matches = sum(1 for h, a in zip(human_scores, ai_scores, strict=True) if h == a)
    return matches / len(human_scores)


def compute_hitl_trigger_rate(confidences: list[float], threshold: float = 0.70) -> float:
    """Fraction of results where mean confidence < threshold."""
    if not confidences:
        return 0.0
    triggered = sum(1 for c in confidences if c < threshold)
    return triggered / len(confidences)


def compute_confidence_calibration(
    confidences: list[float],
    accuracies: list[bool],
    n_bins: int = 10,
) -> list[dict]:
    """Bin predictions by confidence decile, compute actual accuracy per bin.

    Returns a list of dicts with bin_start, bin_end, count, accuracy.
    """
    bins: dict[int, list[bool]] = defaultdict(list)
    for conf, acc in zip(confidences, accuracies, strict=False):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append(acc)

    calibration = []
    for i in range(n_bins):
        items = bins.get(i, [])
        calibration.append({
            "bin_start": round(i / n_bins, 2),
            "bin_end": round((i + 1) / n_bins, 2),
            "count": len(items),
            "accuracy": round(sum(items) / len(items), 4) if items else None,
        })
    return calibration


def print_table(metrics: dict) -> None:
    """Print a formatted metrics summary table."""
    print("\n" + "=" * 60)
    print("  Grading Accuracy Evaluation Results")
    print("=" * 60)

    paired = metrics.get("paired_count", 0)
    print(f"\n  Paired entries (human + AI):  {paired}")
    print(f"  Gold-standard entries:       {metrics.get('gold_count', 0)}")

    if paired > 0:
        print(f"\n  {'Metric':<30} {'Value':>10}")
        print(f"  {'-' * 30} {'-' * 10}")
        print(f"  {'QWK (Quadratic Weighted Kappa)':<30} {metrics['qwk']:>10.4f}")
        print(f"  {'Mean Absolute Error':<30} {metrics['mae']:>10.4f}")
        print(f"  {'Exact Match Rate':<30} {metrics['exact_match_rate']:>10.4f}")

    if metrics.get("hitl_trigger_rate") is not None:
        print(f"  {'HITL Trigger Rate (<0.70)':<30} {metrics['hitl_trigger_rate']:>10.4f}")

    cal = metrics.get("confidence_calibration", [])
    if cal and any(b["count"] > 0 for b in cal):
        print("\n  Confidence Calibration (Reliability Diagram Data)")
        print(f"  {'Bin':<12} {'Count':>6} {'Accuracy':>10}")
        print(f"  {'-' * 12} {'-' * 6} {'-' * 10}")
        for b in cal:
            acc_str = f"{b['accuracy']:.4f}" if b["accuracy"] is not None else "   N/A"
            print(f"  [{b['bin_start']:.1f}, {b['bin_end']:.1f}){'':<3} {b['count']:>6} {acc_str:>10}")

    if paired == 0:
        print("\n  No AI scores available for comparison.")
        print("  Provide --mongo-uri to load AI results, or add 'ai_score'")
        print("  fields to gold_standard.json for offline evaluation.")

    print("\n" + "=" * 60 + "\n")


def print_total_table(metrics: dict) -> None:
    """Print a formatted total-score metrics summary table."""
    print("\n" + "=" * 60)
    print("  Total-Score Grading Evaluation Results")
    print("=" * 60)
    print(f"\n  Gold labels:                {metrics.get('gold_count', 0)}")
    print(f"  Predictions:                {metrics.get('prediction_count', 0)}")
    print(f"  Paired scripts:             {metrics.get('paired_count', 0)}")
    print(f"  Failed scripts:             {metrics.get('failed_count', 0)}")

    if metrics.get("paired_count", 0) > 0:
        print(f"\n  {'Metric':<30} {'Value':>10}")
        print(f"  {'-' * 30} {'-' * 10}")
        for key in (
            "mae_marks",
            "mae_percent",
            "mean_signed_error_marks",
            "mean_signed_error_percent",
            "exact_total_match_rate",
            "within_1_mark_rate",
            "within_3_marks_rate",
            "within_5_percent_rate",
            "qwk_percent_bucket",
            "mean_confidence",
            "hitl_trigger_rate",
        ):
            value = metrics.get(key)
            rendered = "N/A" if value is None else f"{value:.4f}"
            print(f"  {key:<30} {rendered:>10}")

    print("\n" + "=" * 60 + "\n")


def count_sibling_failures(predictions_path: Path) -> int:
    failures_path = predictions_path.parent / "failures.jsonl"
    if not failures_path.exists():
        return 0
    return sum(1 for line in failures_path.read_text().splitlines() if line.strip())


def load_total_labels(args: argparse.Namespace) -> dict[str, dict]:
    if args.total_labels:
        return load_total_labels_file(Path(args.total_labels))
    if args.scores_pdf:
        return labels_from_score_book(parse_scores_pdf(Path(args.scores_pdf)))
    raise ValueError("--scores-pdf or --total-labels is required for total-score metrics")


def write_output(path: str, metrics: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results written to {path}")


def compute_question_level_metrics(args: argparse.Namespace) -> dict:
    if not args.gold:
        raise ValueError("--gold is required for question-level metrics")

    entries = load_gold_standard(args.gold)
    print(f"Loaded {len(entries)} gold-standard entries.")

    # Collect AI scores — either from MongoDB or inline in gold_standard.json
    ai_data: dict[str, dict] = {}

    if args.mongo_uri:
        print("Connecting to MongoDB to load AI scores...")
        ai_data = load_ai_scores_from_mongo(args.mongo_uri, entries)
        print(f"Matched {len(ai_data)} entries with AI results.")
    else:
        # Allow offline evaluation with ai_score fields in gold_standard.json
        for entry in entries:
            if "ai_score" in entry:
                ai_data[entry["eval_id"]] = {
                    "ai_score": entry["ai_score"],
                    "confidence": entry.get("ai_confidence"),
                    "mean_confidence": entry.get("ai_confidence"),
                }

    # Build paired arrays
    human_scores: list[float] = []
    ai_scores: list[float] = []
    confidences: list[float] = []
    exact_matches: list[bool] = []

    for entry in entries:
        eid = entry["eval_id"]
        if eid not in ai_data:
            continue
        h = float(entry["human_score"])
        a = float(ai_data[eid]["ai_score"])
        human_scores.append(h)
        ai_scores.append(a)
        exact_matches.append(h == a)

        conf = ai_data[eid].get("confidence") or ai_data[eid].get("mean_confidence")
        if conf is not None:
            confidences.append(float(conf))

    metrics: dict = {
        "gold_count": len(entries),
        "paired_count": len(human_scores),
    }

    if human_scores:
        metrics["qwk"] = compute_qwk(human_scores, ai_scores)
        metrics["mae"] = compute_mae(human_scores, ai_scores)
        metrics["exact_match_rate"] = compute_exact_match_rate(human_scores, ai_scores)
    else:
        metrics["qwk"] = None
        metrics["mae"] = None
        metrics["exact_match_rate"] = None

    if confidences:
        metrics["hitl_trigger_rate"] = compute_hitl_trigger_rate(confidences)
        metrics["confidence_calibration"] = compute_confidence_calibration(
            confidences, exact_matches[: len(confidences)]
        )
    else:
        metrics["hitl_trigger_rate"] = None
        metrics["confidence_calibration"] = []

    return metrics

def _round_metric(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return round(parsed, digits)


def _rate(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return _round_metric(count / total)


def _score_status(score: float | None) -> str:
    if score is None:
        return "blank"
    if score == 0:
        return "attempted_zero"
    return "scored"


def _format_question_set(question_ids: set[str]) -> str:
    return ",".join(sorted(question_ids))


def _score_agreement_metrics(
    reference_scores: list[float], comparison_scores: list[float]
) -> dict[str, Any]:
    paired_count = len(reference_scores)
    if paired_count == 0:
        return {
            "paired_count": 0,
            "mae_marks": None,
            "rmse_marks": None,
            "median_absolute_error_marks": None,
            "mean_signed_error_marks": None,
            "exact_match_rate": None,
            "within_1_mark_rate": None,
            "within_2_marks_rate": None,
            "within_3_marks_rate": None,
        }

    errors = [
        abs(comparison - reference)
        for reference, comparison in zip(reference_scores, comparison_scores, strict=True)
    ]
    signed_errors = [
        comparison - reference
        for reference, comparison in zip(reference_scores, comparison_scores, strict=True)
    ]
    return {
        "paired_count": paired_count,
        "mae_marks": _round_metric(mean(errors)),
        "rmse_marks": _round_metric(math.sqrt(mean([error * error for error in errors]))),
        "median_absolute_error_marks": _round_metric(median(errors)),
        "mean_signed_error_marks": _round_metric(mean(signed_errors)),
        "exact_match_rate": _rate(
            sum(
                reference == comparison
                for reference, comparison in zip(
                    reference_scores, comparison_scores, strict=True
                )
            ),
            paired_count,
        ),
        "within_1_mark_rate": _rate(sum(error <= 1 for error in errors), paired_count),
        "within_2_marks_rate": _rate(sum(error <= 2 for error in errors), paired_count),
        "within_3_marks_rate": _rate(sum(error <= 3 for error in errors), paired_count),
    }


def _sum_total_mismatches(book: PerQuestionScoreBook) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row in book.labels.values():
        if row.recorded_total is None:
            continue
        reference_total = row.recorded_sum
        reference_name = "recorded_sum"
        if reference_total is None:
            reference_total = row.computed_sum
            reference_name = "computed_sum"
        difference = row.recorded_total - reference_total
        if abs(difference) < 1e-9:
            continue
        mismatches.append(
            {
                "script_id": row.script_id,
                "serial_number": row.serial_number,
                "reference_field": reference_name,
                "reference_total": reference_total,
                "recorded_total": row.recorded_total,
                "difference": _round_metric(difference),
            }
        )
    return mismatches


def _over_max_rows(book: PerQuestionScoreBook) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in book.labels.values():
        checked_values = {
            "computed_sum": row.computed_sum,
            "recorded_sum": row.recorded_sum,
            "recorded_total": row.recorded_total,
        }
        exceeded = {
            field: value
            for field, value in checked_values.items()
            if value is not None and value > book.max_score
        }
        if exceeded:
            rows.append(
                {
                    "script_id": row.script_id,
                    "serial_number": row.serial_number,
                    "max_score": book.max_score,
                    "exceeded": exceeded,
                }
            )
    return rows


def _book_audit_summary(book: PerQuestionScoreBook) -> dict[str, Any]:
    sum_total_mismatches = _sum_total_mismatches(book)
    over_max = _over_max_rows(book)
    return {
        "grader_id": book.grader_id,
        "source_path": book.source_path,
        "row_count": len(book.labels),
        "max_score": book.max_score,
        "warning_count": len(book.warnings),
        "warnings": book.warnings,
        "sum_total_mismatch_count": len(sum_total_mismatches),
        "sum_total_mismatches": sum_total_mismatches,
        "over_max_count": len(over_max),
        "over_max_rows": over_max,
    }


def _human_total_agreement(
    original: PerQuestionScoreBook, grader2: PerQuestionScoreBook
) -> dict[str, Any]:
    shared_ids = sorted(set(original.labels) & set(grader2.labels), key=int)
    original_recorded: list[float] = []
    original_sum: list[float] = []
    grader2_total_for_recorded: list[float] = []
    grader2_total_for_sum: list[float] = []

    for script_id in shared_ids:
        original_row = original.labels[script_id]
        grader2_row = grader2.labels[script_id]
        if original_row.recorded_total is not None and grader2_row.recorded_total is not None:
            original_recorded.append(original_row.recorded_total)
            grader2_total_for_recorded.append(grader2_row.recorded_total)
        if grader2_row.recorded_total is not None:
            original_sum.append(original_row.computed_sum)
            grader2_total_for_sum.append(grader2_row.recorded_total)

    return {
        "original_recorded_total_vs_grader2_total": _score_agreement_metrics(
            original_recorded, grader2_total_for_recorded
        ),
        "original_computed_sum_vs_grader2_total": _score_agreement_metrics(
            original_sum, grader2_total_for_sum
        ),
    }


def _human_cell_agreement(
    original: PerQuestionScoreBook, grader2: PerQuestionScoreBook
) -> dict[str, Any]:
    by_question: dict[str, dict[str, list[float]]] = {
        question_id: {"original": [], "grader2": []} for question_id in QUESTION_IDS
    }
    original_scores: list[float] = []
    grader2_scores: list[float] = []

    for script_id in sorted(set(original.labels) & set(grader2.labels), key=int):
        original_row = original.labels[script_id]
        grader2_row = grader2.labels[script_id]
        for question_id in QUESTION_IDS:
            original_score = original_row.scores[question_id]
            grader2_score = grader2_row.scores[question_id]
            if original_score is None or grader2_score is None:
                continue
            original_scores.append(original_score)
            grader2_scores.append(grader2_score)
            by_question[question_id]["original"].append(original_score)
            by_question[question_id]["grader2"].append(grader2_score)

    metrics = _score_agreement_metrics(original_scores, grader2_scores)
    metrics["by_question"] = {
        question_id: _score_agreement_metrics(
            scores["original"], scores["grader2"]
        )
        for question_id, scores in by_question.items()
    }
    return metrics


def _attempted_set_agreement(
    original: PerQuestionScoreBook, grader2: PerQuestionScoreBook
) -> dict[str, Any]:
    shared_ids = sorted(set(original.labels) & set(grader2.labels), key=int)
    exact_matches = 0
    blank_nonblank_conflicts = 0
    zero_vs_blank_conflicts = 0
    nonzero_vs_blank_conflicts = 0
    mismatches: list[dict[str, Any]] = []
    one_for_one_swap_script_ids: list[str] = []

    for script_id in shared_ids:
        original_row = original.labels[script_id]
        grader2_row = grader2.labels[script_id]
        original_set = question_attempt_set(original_row)
        grader2_set = question_attempt_set(grader2_row)
        only_original = original_set - grader2_set
        only_grader2 = grader2_set - original_set
        one_for_one = len(only_original) == 1 and len(only_grader2) == 1

        if original_set == grader2_set:
            exact_matches += 1
        else:
            if one_for_one:
                one_for_one_swap_script_ids.append(script_id)
            mismatches.append(
                {
                    "script_id": script_id,
                    "original_attempted": sorted(original_set),
                    "grader2_attempted": sorted(grader2_set),
                    "only_original": sorted(only_original),
                    "only_grader2": sorted(only_grader2),
                    "one_for_one_swap": one_for_one,
                }
            )

        for question_id in QUESTION_IDS:
            original_score = original_row.scores[question_id]
            grader2_score = grader2_row.scores[question_id]
            if (original_score is None) == (grader2_score is None):
                continue
            blank_nonblank_conflicts += 1
            nonblank = grader2_score if original_score is None else original_score
            if nonblank == 0:
                zero_vs_blank_conflicts += 1
            else:
                nonzero_vs_blank_conflicts += 1

    return {
        "paired_script_count": len(shared_ids),
        "exact_attempted_set_match_count": exact_matches,
        "exact_attempted_set_match_rate": _rate(exact_matches, len(shared_ids)),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "blank_nonblank_conflicts": blank_nonblank_conflicts,
        "zero_vs_blank_conflicts": zero_vs_blank_conflicts,
        "nonzero_vs_blank_conflicts": nonzero_vs_blank_conflicts,
        "one_for_one_swap_count": len(one_for_one_swap_script_ids),
        "one_for_one_swap_script_ids": one_for_one_swap_script_ids,
    }


def _human_disagreements(
    original: PerQuestionScoreBook,
    grader2: PerQuestionScoreBook,
    attempted_set_agreement: dict[str, Any],
) -> list[dict[str, Any]]:
    disagreements: list[dict[str, Any]] = []
    swap_ids = set(attempted_set_agreement["one_for_one_swap_script_ids"])

    for script_id in sorted(set(original.labels) & set(grader2.labels), key=int):
        original_row = original.labels[script_id]
        grader2_row = grader2.labels[script_id]
        for question_id in QUESTION_IDS:
            original_score = original_row.scores[question_id]
            grader2_score = grader2_row.scores[question_id]
            if original_score is None and grader2_score is None:
                continue
            conflict_type: str | None = None
            absolute_difference: float | None = None
            if original_score is not None and grader2_score is not None:
                if abs(original_score - grader2_score) < 1e-9:
                    continue
                conflict_type = "same_question_score_gap"
                absolute_difference = abs(grader2_score - original_score)
            else:
                nonblank = grader2_score if original_score is None else original_score
                conflict_type = "zero_vs_blank" if nonblank == 0 else "nonzero_vs_blank"

            disagreements.append(
                {
                    "script_id": script_id,
                    "question_id": question_id,
                    "original_score": original_score,
                    "original_status": _score_status(original_score),
                    "grader2_score": grader2_score,
                    "grader2_status": _score_status(grader2_score),
                    "absolute_difference": _round_metric(absolute_difference),
                    "conflict_type": conflict_type,
                    "part_of_question_set_swap": script_id in swap_ids,
                }
            )

        if script_id in swap_ids:
            original_set = question_attempt_set(original_row)
            grader2_set = question_attempt_set(grader2_row)
            disagreements.append(
                {
                    "script_id": script_id,
                    "question_id": "attempt_set",
                    "original_score": _format_question_set(original_set),
                    "original_status": "attempted_set",
                    "grader2_score": _format_question_set(grader2_set),
                    "grader2_status": "attempted_set",
                    "absolute_difference": None,
                    "conflict_type": "question_set_swap",
                    "part_of_question_set_swap": True,
                }
            )

    for mismatch in _sum_total_mismatches(original):
        disagreements.append(
            {
                "script_id": mismatch["script_id"],
                "question_id": "total",
                "original_score": mismatch["reference_total"],
                "original_status": mismatch["reference_field"],
                "grader2_score": mismatch["recorded_total"],
                "grader2_status": "original_recorded_total",
                "absolute_difference": abs(mismatch["difference"]),
                "conflict_type": "original_sum_total_mismatch",
                "part_of_question_set_swap": mismatch["script_id"] in swap_ids,
            }
        )

    for row in _over_max_rows(grader2):
        disagreements.append(
            {
                "script_id": row["script_id"],
                "question_id": "total",
                "original_score": None,
                "original_status": "not_applicable",
                "grader2_score": row["exceeded"].get("recorded_total")
                or row["exceeded"].get("computed_sum"),
                "grader2_status": "over_max_total",
                "absolute_difference": None,
                "conflict_type": "over_max_total",
                "part_of_question_set_swap": row["script_id"] in swap_ids,
            }
        )

    return disagreements


def _build_human_human_report(
    original: PerQuestionScoreBook, grader2: PerQuestionScoreBook
) -> dict[str, Any]:
    attempted = _attempted_set_agreement(original, grader2)
    return {
        "paper_claim": "human_label_variability_is_a_safety_context_not_clean_ground_truth",
        "primary_reference_policy": "strict_consensus_cells_for_per_question_scores; original_computed_sum_and_human_mean_total_for_total_sensitivity",
        "original": _book_audit_summary(original),
        "grader2": _book_audit_summary(grader2),
        "human_total_agreement": _human_total_agreement(original, grader2),
        "human_cell_agreement": _human_cell_agreement(original, grader2),
        "attempted_set_agreement": attempted,
        "disagreements": _human_disagreements(original, grader2, attempted),
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_score_book_csv(path: Path, book: PerQuestionScoreBook) -> None:
    rows = []
    for row in book.labels.values():
        rows.append(
            {
                "script_id": row.script_id,
                "serial_number": row.serial_number,
                **{question_id: row.scores[question_id] for question_id in QUESTION_IDS},
                "computed_sum": row.computed_sum,
                "recorded_sum": row.recorded_sum,
                "recorded_total": row.recorded_total,
                "pages_reported": row.pages_reported,
            }
        )
    _write_csv(
        path,
        [
            "script_id",
            "serial_number",
            *QUESTION_IDS,
            "computed_sum",
            "recorded_sum",
            "recorded_total",
            "pages_reported",
        ],
        rows,
    )


def _write_human_paper_tables(
    output_dir: str | None,
    report: dict[str, Any],
    original: PerQuestionScoreBook,
    grader2: PerQuestionScoreBook,
) -> None:
    if not output_dir:
        return
    table_dir = Path(output_dir)
    summary = report["human_total_agreement"]
    rows = [
        {
            "metric": "original_recorded_total_vs_grader2_total_mae",
            "value": summary["original_recorded_total_vs_grader2_total"]["mae_marks"],
        },
        {
            "metric": "original_computed_sum_vs_grader2_total_mae",
            "value": summary["original_computed_sum_vs_grader2_total"]["mae_marks"],
        },
        {
            "metric": "both_human_scored_cells",
            "value": report["human_cell_agreement"]["paired_count"],
        },
        {
            "metric": "per_question_cell_mae",
            "value": report["human_cell_agreement"]["mae_marks"],
        },
        {
            "metric": "exact_per_question_cell_match_rate",
            "value": report["human_cell_agreement"]["exact_match_rate"],
        },
        {
            "metric": "blank_nonblank_conflicts",
            "value": report["attempted_set_agreement"]["blank_nonblank_conflicts"],
        },
        {
            "metric": "likely_one_for_one_question_set_swaps",
            "value": report["attempted_set_agreement"]["one_for_one_swap_count"],
        },
        {
            "metric": "original_sum_total_mismatches",
            "value": report["original"]["sum_total_mismatch_count"],
        },
        {
            "metric": "grader2_over_max_rows",
            "value": report["grader2"]["over_max_count"],
        },
    ]
    _write_csv(table_dir / "human_human_summary.csv", ["metric", "value"], rows)
    _write_csv(
        table_dir / "human_human_disagreements.csv",
        [
            "script_id",
            "question_id",
            "original_score",
            "original_status",
            "grader2_score",
            "grader2_status",
            "absolute_difference",
            "conflict_type",
            "part_of_question_set_swap",
        ],
        report["disagreements"],
    )
    _write_score_book_csv(table_dir / "per_question_labels.original.csv", original)
    _write_score_book_csv(table_dir / "per_question_labels.grader2.csv", grader2)


def compute_human_human_report(args: argparse.Namespace) -> dict[str, Any]:
    if not args.original_score_sheet or not args.grader2_score_sheet:
        raise ValueError("--original-score-sheet and --grader2-score-sheet are required")
    original = parse_per_question_scores_csv(
        Path(args.original_score_sheet), "original"
    )
    grader2 = parse_per_question_scores_csv(Path(args.grader2_score_sheet), "grader2")
    report = _build_human_human_report(original, grader2)
    _write_human_paper_tables(args.write_paper_tables, report, original, grader2)
    return report


def _normalize_question_id(raw: Any) -> str:
    text = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    match = re.fullmatch(r"(?:q|question_?)?([1-5])", text)
    if not match:
        raise ValueError(f"Unrecognized question_id: {raw!r}")
    return f"q{match.group(1)}"


def _prediction_question_scores(
    prediction: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    scores: dict[str, float] = {}
    errors: list[str] = []
    questions = prediction.get("graded_questions", [])
    if not isinstance(questions, list):
        return scores, ["graded_questions is not an array"]

    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            errors.append(f"graded_questions[{index}] is not an object")
            continue
        try:
            question_id = _normalize_question_id(question.get("question_id"))
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if question_id in scores:
            errors.append(f"duplicate normalized question_id {question_id}")
            continue
        try:
            scores[question_id] = float(question.get("score", 0))
        except (TypeError, ValueError):
            errors.append(f"invalid score for {question_id}: {question.get('score')!r}")
    return scores, errors


def _total_reference_labels(
    original: PerQuestionScoreBook, grader2: PerQuestionScoreBook
) -> dict[str, dict[str, dict[str, Any]]]:
    shared_ids = sorted(set(original.labels) & set(grader2.labels), key=int)
    references: dict[str, dict[str, dict[str, Any]]] = {
        "original_recorded_total": {},
        "original_computed_sum": {},
        "grader2_total": {},
        "human_mean_total": {},
    }
    for script_id in shared_ids:
        original_row = original.labels[script_id]
        grader2_row = grader2.labels[script_id]
        if original_row.recorded_total is not None:
            references["original_recorded_total"][script_id] = {
                "script_id": script_id,
                "human_total_score": original_row.recorded_total,
                "max_score": original.max_score,
            }
        references["original_computed_sum"][script_id] = {
            "script_id": script_id,
            "human_total_score": original_row.computed_sum,
            "max_score": original.max_score,
        }
        if grader2_row.recorded_total is not None:
            references["grader2_total"][script_id] = {
                "script_id": script_id,
                "human_total_score": grader2_row.recorded_total,
                "max_score": grader2.max_score,
            }
            references["human_mean_total"][script_id] = {
                "script_id": script_id,
                "human_total_score": mean(
                    [original_row.computed_sum, grader2_row.recorded_total]
                ),
                "max_score": original.max_score,
            }
    return references


def _reference_cells(
    original: PerQuestionScoreBook,
    grader2: PerQuestionScoreBook,
    policy: str,
) -> dict[tuple[str, str], float]:
    cells: dict[tuple[str, str], float] = {}
    for script_id in sorted(set(original.labels) & set(grader2.labels), key=int):
        original_row = original.labels[script_id]
        grader2_row = grader2.labels[script_id]
        for question_id in QUESTION_IDS:
            original_score = original_row.scores[question_id]
            grader2_score = grader2_row.scores[question_id]
            if policy == "individual_human_cells":
                available = [
                    score
                    for score in (original_score, grader2_score)
                    if score is not None
                ]
                if available:
                    cells[(script_id, question_id)] = mean(available)
            elif policy == "grader2_total":
                if grader2_score is not None:
                    cells[(script_id, question_id)] = grader2_score
            elif policy == "original_computed_sum":
                if original_score is not None:
                    cells[(script_id, question_id)] = original_score
            else:
                if original_score is not None and grader2_score is not None:
                    cells[(script_id, question_id)] = mean(
                        [original_score, grader2_score]
                    )
    return cells


def _per_question_ai_metrics(
    predictions: dict[str, dict[str, Any]],
    reference_cells: dict[tuple[str, str], float],
) -> dict[str, Any]:
    reference_scores: list[float] = []
    ai_scores: list[float] = []
    by_question: dict[str, dict[str, list[float]]] = {
        question_id: {"reference": [], "ai": []} for question_id in QUESTION_IDS
    }
    normalization_errors: list[dict[str, Any]] = []

    normalized_predictions: dict[str, dict[str, float]] = {}
    for script_id, prediction in predictions.items():
        scores, errors = _prediction_question_scores(prediction)
        if errors:
            normalization_errors.append({"script_id": script_id, "errors": errors})
        normalized_predictions[script_id] = scores

    for (script_id, question_id), reference_score in sorted(
        reference_cells.items(), key=lambda item: (int(item[0][0]), item[0][1])
    ):
        score_map = normalized_predictions.get(script_id, {})
        if question_id not in score_map:
            continue
        ai_score = score_map[question_id]
        reference_scores.append(reference_score)
        ai_scores.append(ai_score)
        by_question[question_id]["reference"].append(reference_score)
        by_question[question_id]["ai"].append(ai_score)

    aggregate = _score_agreement_metrics(reference_scores, ai_scores)
    aggregate["reference_cell_count"] = len(reference_cells)
    aggregate["normalization_error_count"] = len(normalization_errors)
    aggregate["normalization_errors"] = normalization_errors
    aggregate["by_question"] = {
        question_id: _score_agreement_metrics(scores["reference"], scores["ai"])
        for question_id, scores in by_question.items()
    }
    return aggregate


def _prediction_safety_metrics(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    over_max_scripts: list[str] = []
    validation_error_scripts: list[str] = []
    pending_review_scripts: list[str] = []
    transcription_warning_scripts: list[str] = []
    high_confidence_invalid_scripts: list[str] = []
    json_repair_count = 0

    for script_id, prediction in sorted(predictions.items(), key=lambda item: int(item[0])):
        max_score = float(prediction.get("max_score", 60))
        ai_total = prediction_total(prediction)
        validation_errors = prediction.get("validation_errors", [])
        transcription_warnings = prediction.get("transcription_warnings", [])
        if ai_total > max_score:
            over_max_scripts.append(script_id)
        if validation_errors:
            validation_error_scripts.append(script_id)
        if str(prediction.get("referee_status", "")).upper() == "PENDING_REVIEW":
            pending_review_scripts.append(script_id)
        if transcription_warnings:
            transcription_warning_scripts.append(script_id)
        if prediction.get("json_repair_used"):
            json_repair_count += 1
        confidence = prediction_confidence(prediction)
        if validation_errors and confidence is not None and confidence >= 0.70:
            high_confidence_invalid_scripts.append(script_id)

    total = len(predictions)
    return {
        "prediction_count": total,
        "over_max_total_count": len(over_max_scripts),
        "over_max_total_rate": _rate(len(over_max_scripts), total),
        "over_max_total_scripts": over_max_scripts,
        "validation_error_count": len(validation_error_scripts),
        "validation_error_rate": _rate(len(validation_error_scripts), total),
        "validation_error_scripts": validation_error_scripts,
        "pending_review_count": len(pending_review_scripts),
        "pending_review_rate": _rate(len(pending_review_scripts), total),
        "pending_review_scripts": pending_review_scripts,
        "transcription_warning_count": len(transcription_warning_scripts),
        "transcription_warning_rate": _rate(len(transcription_warning_scripts), total),
        "transcription_warning_scripts": transcription_warning_scripts,
        "json_repair_count": json_repair_count,
        "high_confidence_invalid_count": len(high_confidence_invalid_scripts),
        "high_confidence_invalid_scripts": high_confidence_invalid_scripts,
    }


def _heuristic_attempt_detection(
    predictions: dict[str, dict[str, Any]],
    human_book: PerQuestionScoreBook,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    exact = 0
    paired = 0
    swaps: list[str] = []
    for script_id, human_row in human_book.labels.items():
        prediction = predictions.get(script_id)
        if prediction is None:
            continue
        paired += 1
        score_map, errors = _prediction_question_scores(prediction)
        if errors:
            rows.append(
                {
                    "script_id": script_id,
                    "human_attempted": _format_question_set(question_attempt_set(human_row)),
                    "ai_positive": "",
                    "only_human": "",
                    "only_ai": "",
                    "one_for_one_swap": False,
                    "normalization_errors": "; ".join(errors),
                }
            )
            continue
        human_set = question_attempt_set(human_row)
        ai_positive = {
            question_id for question_id, score in score_map.items() if score > 0
        }
        only_human = human_set - ai_positive
        only_ai = ai_positive - human_set
        one_for_one = len(only_human) == 1 and len(only_ai) == 1
        if human_set == ai_positive:
            exact += 1
        else:
            if one_for_one:
                swaps.append(script_id)
            rows.append(
                {
                    "script_id": script_id,
                    "human_attempted": _format_question_set(human_set),
                    "ai_positive": _format_question_set(ai_positive),
                    "only_human": _format_question_set(only_human),
                    "only_ai": _format_question_set(only_ai),
                    "one_for_one_swap": one_for_one,
                    "normalization_errors": "",
                }
            )
    return {
        "heuristic_attempt_detection": True,
        "paired_count": paired,
        "exact_match_count": exact,
        "exact_match_rate": _rate(exact, paired),
        "mismatch_count": len([row for row in rows if not row["normalization_errors"]]),
        "one_for_one_swap_script_ids": swaps,
        "flags": rows,
    }


def _legacy_condition_metrics(
    predictions: dict[str, dict[str, Any]],
    original: PerQuestionScoreBook,
    grader2: PerQuestionScoreBook,
    reference_policy: str,
) -> dict[str, Any]:
    total_sensitivity = {
        name: compute_total_score_metrics(labels, predictions)
        for name, labels in _total_reference_labels(original, grader2).items()
    }
    reference_cells = _reference_cells(original, grader2, reference_policy)
    return {
        "total_sensitivity": total_sensitivity,
        "per_question": _per_question_ai_metrics(predictions, reference_cells),
        "safety": _prediction_safety_metrics(predictions),
        "heuristic_attempt_detection": {
            "original": _heuristic_attempt_detection(predictions, original),
            "grader2": _heuristic_attempt_detection(predictions, grader2),
        },
    }


def _legacy_ablation_delta(
    no_lecture: dict[str, Any], lecture: dict[str, Any]
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for reference_name, lecture_metrics in lecture["total_sensitivity"].items():
        no_metrics = no_lecture["total_sensitivity"].get(reference_name, {})
        deltas[reference_name] = {
            "delta_mae_marks": _round_metric(
                lecture_metrics.get("mae_marks") - no_metrics.get("mae_marks")
            )
            if lecture_metrics.get("mae_marks") is not None
            and no_metrics.get("mae_marks") is not None
            else None,
            "delta_qwk_percent_bucket": _round_metric(
                lecture_metrics.get("qwk_percent_bucket")
                - no_metrics.get("qwk_percent_bucket")
            )
            if lecture_metrics.get("qwk_percent_bucket") is not None
            and no_metrics.get("qwk_percent_bucket") is not None
            else None,
            "delta_within_3_marks_rate": _round_metric(
                lecture_metrics.get("within_3_marks_rate")
                - no_metrics.get("within_3_marks_rate")
            )
            if lecture_metrics.get("within_3_marks_rate") is not None
            and no_metrics.get("within_3_marks_rate") is not None
            else None,
            "delta_hitl_trigger_rate": _round_metric(
                lecture_metrics.get("hitl_trigger_rate")
                - no_metrics.get("hitl_trigger_rate")
            )
            if lecture_metrics.get("hitl_trigger_rate") is not None
            and no_metrics.get("hitl_trigger_rate") is not None
            else None,
        }
    deltas["safety"] = {
        "delta_over_max_total_count": lecture["safety"]["over_max_total_count"]
        - no_lecture["safety"]["over_max_total_count"],
        "delta_high_confidence_invalid_count": lecture["safety"][
            "high_confidence_invalid_count"
        ]
        - no_lecture["safety"]["high_confidence_invalid_count"],
    }
    return deltas


def _load_legacy_prediction_sets(args: argparse.Namespace) -> dict[str, dict[str, dict[str, Any]]]:
    prediction_sets: dict[str, dict[str, dict[str, Any]]] = {}
    if args.predictions:
        prediction_sets["predictions"] = load_predictions_jsonl(Path(args.predictions))
    if args.predictions_no_lecture:
        prediction_sets["no_lecture"] = load_predictions_jsonl(
            Path(args.predictions_no_lecture)
        )
    if args.predictions_lecture:
        prediction_sets["lecture"] = load_predictions_jsonl(Path(args.predictions_lecture))
    if not prediction_sets:
        raise ValueError(
            "--predictions, --predictions-no-lecture, or --predictions-lecture is required"
        )
    return prediction_sets


def _write_legacy_paper_tables(
    output_dir: str | None,
    report: dict[str, Any],
    prediction_sets: dict[str, dict[str, dict[str, Any]]],
    original: PerQuestionScoreBook,
    grader2: PerQuestionScoreBook,
) -> None:
    if not output_dir:
        return
    table_dir = Path(output_dir)

    total_rows: list[dict[str, Any]] = []
    by_question_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    heuristic_rows: list[dict[str, Any]] = []

    for condition_name in prediction_sets:
        condition = report[condition_name]
        for reference_name, metrics in condition["total_sensitivity"].items():
            total_rows.append(
                {
                    "condition": condition_name,
                    "reference": reference_name,
                    "paired_count": metrics.get("paired_count"),
                    "mae_marks": metrics.get("mae_marks"),
                    "mean_signed_error_marks": metrics.get("mean_signed_error_marks"),
                    "exact_total_match_rate": metrics.get("exact_total_match_rate"),
                    "within_3_marks_rate": metrics.get("within_3_marks_rate"),
                    "qwk_percent_bucket": metrics.get("qwk_percent_bucket"),
                    "hitl_trigger_rate": metrics.get("hitl_trigger_rate"),
                }
            )
        for question_id, metrics in condition["per_question"]["by_question"].items():
            by_question_rows.append(
                {
                    "condition": condition_name,
                    "question_id": question_id,
                    "paired_count": metrics.get("paired_count"),
                    "mae_marks": metrics.get("mae_marks"),
                    "rmse_marks": metrics.get("rmse_marks"),
                    "mean_signed_error_marks": metrics.get("mean_signed_error_marks"),
                    "exact_match_rate": metrics.get("exact_match_rate"),
                    "within_1_mark_rate": metrics.get("within_1_mark_rate"),
                    "within_2_marks_rate": metrics.get("within_2_marks_rate"),
                    "within_3_marks_rate": metrics.get("within_3_marks_rate"),
                }
            )
        for script_id, prediction in sorted(
            prediction_sets[condition_name].items(), key=lambda item: int(item[0])
        ):
            safety_rows.append(
                {
                    "condition": condition_name,
                    "script_id": script_id,
                    "ai_total_score": prediction_total(prediction),
                    "max_score": prediction.get("max_score", 60),
                    "mean_confidence": prediction_confidence(prediction),
                    "referee_status": prediction.get("referee_status"),
                    "hitl_flag": prediction_hitl(prediction),
                    "validation_error_count": len(
                        prediction.get("validation_errors", [])
                    ),
                    "transcription_warning_count": len(
                        prediction.get("transcription_warnings", [])
                    ),
                }
            )
        for reference_name in ("original", "grader2"):
            for flag in condition["heuristic_attempt_detection"][reference_name]["flags"]:
                heuristic_rows.append(
                    {
                        "condition": condition_name,
                        "reference": reference_name,
                        **flag,
                    }
                )

    ablation_rows: list[dict[str, Any]] = []
    if {"no_lecture", "lecture"}.issubset(prediction_sets):
        human_mean_labels = _total_reference_labels(original, grader2)["human_mean_total"]
        shared_ids = sorted(
            set(human_mean_labels)
            & set(prediction_sets["no_lecture"])
            & set(prediction_sets["lecture"]),
            key=int,
        )
        for script_id in shared_ids:
            human = human_mean_labels[script_id]["human_total_score"]
            no_total = prediction_total(prediction_sets["no_lecture"][script_id])
            lecture_total = prediction_total(prediction_sets["lecture"][script_id])
            ablation_rows.append(
                {
                    "script_id": script_id,
                    "human_mean_total": human,
                    "no_lecture_ai_total": no_total,
                    "lecture_ai_total": lecture_total,
                    "no_lecture_abs_error": abs(no_total - human),
                    "lecture_abs_error": abs(lecture_total - human),
                    "delta_abs_error_lecture_minus_no": abs(lecture_total - human)
                    - abs(no_total - human),
                }
            )

    _write_csv(
        table_dir / "ai_human_total_sensitivity.csv",
        [
            "condition",
            "reference",
            "paired_count",
            "mae_marks",
            "mean_signed_error_marks",
            "exact_total_match_rate",
            "within_3_marks_rate",
            "qwk_percent_bucket",
            "hitl_trigger_rate",
        ],
        total_rows,
    )
    _write_csv(
        table_dir / "ai_human_by_question.csv",
        [
            "condition",
            "question_id",
            "paired_count",
            "mae_marks",
            "rmse_marks",
            "mean_signed_error_marks",
            "exact_match_rate",
            "within_1_mark_rate",
            "within_2_marks_rate",
            "within_3_marks_rate",
        ],
        by_question_rows,
    )
    _write_csv(
        table_dir / "lecture_ablation_by_script.csv",
        [
            "script_id",
            "human_mean_total",
            "no_lecture_ai_total",
            "lecture_ai_total",
            "no_lecture_abs_error",
            "lecture_abs_error",
            "delta_abs_error_lecture_minus_no",
        ],
        ablation_rows,
    )
    _write_csv(
        table_dir / "safety_gate_outcomes.csv",
        [
            "condition",
            "script_id",
            "ai_total_score",
            "max_score",
            "mean_confidence",
            "referee_status",
            "hitl_flag",
            "validation_error_count",
            "transcription_warning_count",
        ],
        safety_rows,
    )
    _write_csv(
        table_dir / "heuristic_question_mapping_flags.csv",
        [
            "condition",
            "reference",
            "script_id",
            "human_attempted",
            "ai_positive",
            "only_human",
            "only_ai",
            "one_for_one_swap",
            "normalization_errors",
        ],
        heuristic_rows,
    )


def compute_legacy_per_question_metrics(args: argparse.Namespace) -> dict[str, Any]:
    if not args.original_score_sheet or not args.grader2_score_sheet:
        raise ValueError("--original-score-sheet and --grader2-score-sheet are required")

    original = parse_per_question_scores_csv(
        Path(args.original_score_sheet), "original"
    )
    grader2 = parse_per_question_scores_csv(Path(args.grader2_score_sheet), "grader2")
    prediction_sets = _load_legacy_prediction_sets(args)
    human_report = _build_human_human_report(original, grader2)

    report: dict[str, Any] = {
        "mode": "legacy_per_question_metrics",
        "attemptedness_supported": False,
        "reference_policy": args.reference_policy,
        "human_human_summary": {
            "original": human_report["original"],
            "grader2": human_report["grader2"],
            "human_total_agreement": human_report["human_total_agreement"],
            "human_cell_agreement": human_report["human_cell_agreement"],
            "attempted_set_agreement": human_report["attempted_set_agreement"],
        },
        "paper_claims_supported": [
            "human-human grading variability is measurable and paper-relevant",
            "deterministic validation catches over-maximum AI totals",
            "lecture-summary context changes aggregate agreement and review routing",
            "legacy per-question score agreement can be reported against unadjudicated human references",
        ],
        "paper_claims_not_supported": [
            "attempted-mask correctness",
            "absolute per-question correctness",
            "calibrated confidence",
            "autonomous high-stakes grading readiness",
        ],
    }

    for condition_name, predictions in prediction_sets.items():
        report[condition_name] = _legacy_condition_metrics(
            predictions, original, grader2, args.reference_policy
        )

    if {"no_lecture", "lecture"}.issubset(report):
        report["ablation_delta"] = _legacy_ablation_delta(
            report["no_lecture"], report["lecture"]
        )
    else:
        report["ablation_delta"] = None

    _write_human_paper_tables(args.write_paper_tables, human_report, original, grader2)
    _write_legacy_paper_tables(
        args.write_paper_tables, report, prediction_sets, original, grader2
    )
    return report



def _load_prediction_rows(path: str) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {str(row["script_id"]): row for row in rows}


def _selection_for_human_row(row_scores: dict[str, float | None], rubric: ExamRubric) -> dict[str, Any]:
    scores = [
        question_score_for_selection(
            qid,
            0.0 if row_scores.get(qid) is None else float(row_scores[qid]),
            rubric.questions[qid].raw_max_score,
            rubric,
            row_scores.get(qid) is not None,
        )
        for qid in QUESTION_ORDER
    ]
    result = select_best_allowed_questions(scores, rubric)
    return {
        "total": result.ai_total_score,
        "selected_question_ids": result.selected_question_ids,
        "normalized_question_scores": result.normalized_question_scores,
        "raw_attempted_total_score": result.raw_attempted_total_score,
    }


def _corrected_human_references(
    original: PerQuestionScoreBook, grader2: PerQuestionScoreBook, rubric: ExamRubric
) -> dict[str, dict[str, dict[str, Any]]]:
    references: dict[str, dict[str, dict[str, Any]]] = {
        "original_corrected_policy_total": {},
        "grader2_corrected_policy_total": {},
        "human_mean_corrected_policy_total": {},
        "original_recorded_total": {},
        "original_computed_sum": {},
        "grader2_recorded_total": {},
    }
    for script_id in sorted(set(original.labels) | set(grader2.labels), key=lambda x: int(x) if x.isdigit() else x):
        original_row = original.labels.get(script_id)
        grader2_row = grader2.labels.get(script_id)
        original_selection = (
            _selection_for_human_row(original_row.scores, rubric) if original_row else None
        )
        grader2_selection = (
            _selection_for_human_row(grader2_row.scores, rubric) if grader2_row else None
        )
        if original_selection is not None:
            references["original_corrected_policy_total"][script_id] = {
                "script_id": script_id,
                "human_total_score": original_selection["total"],
                "selected_question_ids": original_selection["selected_question_ids"],
                "source": original.source_path,
            }
            references["original_computed_sum"][script_id] = {
                "script_id": script_id,
                "human_total_score": original_row.computed_sum,
                "source": original.source_path,
            }
            if original_row.recorded_total is not None:
                references["original_recorded_total"][script_id] = {
                    "script_id": script_id,
                    "human_total_score": original_row.recorded_total,
                    "source": original.source_path,
                }
        if grader2_selection is not None:
            references["grader2_corrected_policy_total"][script_id] = {
                "script_id": script_id,
                "human_total_score": grader2_selection["total"],
                "selected_question_ids": grader2_selection["selected_question_ids"],
                "source": grader2.source_path,
            }
            if grader2_row.recorded_total is not None:
                references["grader2_recorded_total"][script_id] = {
                    "script_id": script_id,
                    "human_total_score": grader2_row.recorded_total,
                    "source": grader2.source_path,
                }
        if original_selection is not None and grader2_selection is not None:
            references["human_mean_corrected_policy_total"][script_id] = {
                "script_id": script_id,
                "human_total_score": _round_metric(
                    (original_selection["total"] + grader2_selection["total"]) / 2
                ),
                "original_total": original_selection["total"],
                "grader2_total": grader2_selection["total"],
                "original_selected_question_ids": original_selection["selected_question_ids"],
                "grader2_selected_question_ids": grader2_selection["selected_question_ids"],
            }
    return references


def _prediction_raw_question_scores(prediction: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for question in prediction.get("graded_questions", []) or []:
        if not isinstance(question, dict):
            continue
        try:
            qid = _normalize_question_id(question.get("question_id"))
        except ValueError:
            continue
        try:
            scores[qid] = float(question.get("score", 0.0))
        except (TypeError, ValueError):
            continue
    return scores


def _condition_total_rows(
    condition: str,
    predictions: dict[str, dict[str, Any]],
    references: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_by_reference: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for reference_name, labels in references.items():
        if not labels:
            continue
        metrics = compute_total_score_metrics(labels, predictions)
        metrics_by_reference[reference_name] = metrics
        rows.append(
            {
                "condition": condition,
                "reference": reference_name,
                "paired_count": metrics.get("paired_count"),
                "mae_marks": metrics.get("mae_marks"),
                "mean_signed_error_marks": metrics.get("mean_signed_error_marks"),
                "exact_match_rate": metrics.get("exact_match_rate"),
                "within_1_mark_rate": metrics.get("within_1_mark_rate"),
                "within_3_mark_rate": metrics.get("within_3_mark_rate"),
                "within_5_percent_rate": metrics.get("within_5_percent_rate"),
                "qwk_percent_buckets": metrics.get("qwk_percent_buckets"),
                "mean_confidence": metrics.get("mean_confidence"),
                "hitl_trigger_rate": metrics.get("hitl_trigger_rate"),
            }
        )
    return metrics_by_reference, rows


def _corrected_per_question_rows(
    condition: str,
    predictions: dict[str, dict[str, Any]],
    original: PerQuestionScoreBook,
    grader2: PerQuestionScoreBook,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_builders: dict[str, Any] = {
        "original": lambda sid, qid: original.labels.get(sid).scores.get(qid) if sid in original.labels else None,
        "grader2": lambda sid, qid: grader2.labels.get(sid).scores.get(qid) if sid in grader2.labels else None,
        "strict_consensus_cells": lambda sid, qid: (
            original.labels[sid].scores[qid]
            if sid in original.labels
            and sid in grader2.labels
            and original.labels[sid].scores[qid] is not None
            and grader2.labels[sid].scores[qid] is not None
            and abs(original.labels[sid].scores[qid] - grader2.labels[sid].scores[qid]) < 1e-9
            else None
        ),
        "human_mean_cells": lambda sid, qid: (
            (original.labels[sid].scores[qid] + grader2.labels[sid].scores[qid]) / 2
            if sid in original.labels
            and sid in grader2.labels
            and original.labels[sid].scores[qid] is not None
            and grader2.labels[sid].scores[qid] is not None
            else None
        ),
    }
    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    raw_by_prediction = {
        script_id: _prediction_raw_question_scores(prediction)
        for script_id, prediction in predictions.items()
    }
    for reference_name, value_for in reference_builders.items():
        nested[reference_name] = {}
        for qid in QUESTION_ORDER:
            human_values: list[float] = []
            ai_values: list[float] = []
            for script_id, ai_scores in raw_by_prediction.items():
                if qid not in ai_scores:
                    continue
                value = value_for(script_id, qid)
                if value is None:
                    continue
                human_values.append(float(value))
                ai_values.append(float(ai_scores[qid]))
            metrics = _score_agreement_metrics(human_values, ai_values)
            nested[reference_name][qid] = metrics
            rows.append(
                {
                    "condition": condition,
                    "question_id": qid,
                    "reference_policy": reference_name,
                    "paired_count": metrics["paired_count"],
                    "mae": metrics["mae_marks"],
                    "rmse": metrics["rmse_marks"],
                    "mean_signed_error": metrics["mean_signed_error_marks"],
                    "exact_match_rate": metrics["exact_match_rate"],
                    "within_1_rate": metrics["within_1_mark_rate"],
                    "within_2_rate": metrics["within_2_marks_rate"],
                    "within_3_rate": metrics["within_3_marks_rate"],
                }
            )
    return nested, rows


def _raw_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": _round_metric(mean(values)),
        "median": _round_metric(median(values)),
        "min": _round_metric(min(values)),
        "max": _round_metric(max(values)),
    }


def _corrected_safety_and_trace_rows(
    condition: str, predictions: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    safety_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    aggregate = defaultdict(int)
    for script_id, prediction in predictions.items():
        validation_errors = prediction.get("validation_errors", []) or []
        selected = set(prediction.get("selected_question_ids", []) or [])
        raw_all = float(prediction.get("raw_all_questions_total_score", 0) or 0)
        raw_attempted = float(prediction.get("raw_attempted_total_score", 0) or 0)
        raw_selected = float(prediction.get("raw_selected_total_score", 0) or 0)
        ai_total = float(prediction.get("ai_total_score", 0) or 0)
        if raw_all > 60:
            aggregate["raw_all_questions_total_over_60_count"] += 1
        if raw_attempted > 60:
            aggregate["raw_attempted_total_over_60_count"] += 1
        if raw_selected > 60:
            aggregate["raw_selected_total_over_60_count"] += 1
        if ai_total > 60:
            aggregate["final_ai_total_over_60_count"] += 1
        if prediction.get("referee_status") == "PENDING_REVIEW":
            aggregate["pending_review_count"] += 1
        if prediction.get("low_confidence_count", 0):
            aggregate["low_confidence_count"] += 1
        if prediction.get("json_repair_used"):
            aggregate["json_repair_count"] += 1
        aggregate["rubric_boundary_warning_count"] += int(prediction.get("rubric_boundary_warning_count", 0) or 0)
        for error in validation_errors:
            if "score exceeds corrected rubric max" in error:
                if ".score exceeds" in error and "graded_questions" in error:
                    aggregate["per_question_corrected_max_violation_count"] += 1
                else:
                    aggregate["subpart_corrected_max_violation_count"] += 1
            if "positive score despite extraction attempted=false" in error:
                aggregate["positive_score_on_extracted_unattempted_count"] += 1
            if "does not equal subpart score sum" in error:
                aggregate["score_justification_contradiction_count"] += 1
        if validation_errors and (prediction.get("mean_confidence") or 0) >= 0.7:
            aggregate["high_confidence_invalid_count"] += 1
        safety_rows.append(
            {
                "condition": condition,
                "script_id": script_id,
                "ai_total_score": ai_total,
                "raw_all_questions_total_score": raw_all,
                "raw_attempted_total_score": raw_attempted,
                "raw_selected_total_score": raw_selected,
                "selected_question_ids": ",".join(prediction.get("selected_question_ids", []) or []),
                "unselected_question_ids": ",".join(prediction.get("unselected_question_ids", []) or []),
                "referee_status": prediction.get("referee_status"),
                "validation_error_count": len(validation_errors),
                "extraction_warning_count": prediction.get("extraction_transcription_warning_count", 0),
                "extraction_validation_error_count": prediction.get("extraction_validation_error_count", 0),
                "mean_confidence": prediction.get("mean_confidence"),
            }
        )
        graded_by_qid = {
            question.get("question_id"): question
            for question in prediction.get("graded_questions", []) or []
            if isinstance(question, dict)
        }
        for qid in QUESTION_ORDER:
            question = graded_by_qid.get(qid, {})
            trace_rows.append(
                {
                    "condition": condition,
                    "script_id": script_id,
                    "question_id": qid,
                    "raw_model_score": (prediction.get("raw_model_question_scores") or {}).get(qid),
                    "model_reported_max_score": (prediction.get("model_reported_question_max_scores") or {}).get(qid),
                    "corrected_raw_max_score": (prediction.get("corrected_raw_question_max_scores") or {}).get(qid),
                    "bounded_raw_score": (prediction.get("bounded_raw_question_scores") or {}).get(qid),
                    "normalized_score": (prediction.get("normalized_question_scores") or {}).get(qid),
                    "attempted_from_extraction": qid in (prediction.get("extracted_attempted_questions") or []),
                    "selected_for_total": qid in selected,
                    "subpart_scores_json": json.dumps((prediction.get("raw_model_subpart_scores") or {}).get(qid, {}), sort_keys=True),
                    "justification": question.get("justification"),
                    "feedback": question.get("feedback"),
                }
            )
    for key in (
        "raw_all_questions_total_over_60_count",
        "raw_attempted_total_over_60_count",
        "raw_selected_total_over_60_count",
        "final_ai_total_over_60_count",
        "per_question_corrected_max_violation_count",
        "subpart_corrected_max_violation_count",
        "positive_score_on_extracted_unattempted_count",
        "score_justification_contradiction_count",
        "pending_review_count",
        "low_confidence_count",
        "high_confidence_invalid_count",
        "json_repair_count",
        "rubric_boundary_warning_count",
    ):
        aggregate.setdefault(key, 0)
    return dict(aggregate), safety_rows, trace_rows


def _selection_rows(
    condition: str,
    predictions: dict[str, dict[str, Any]],
    references: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    aggregate = defaultdict(int)
    for reference_name in (
        "original_corrected_policy_total",
        "grader2_corrected_policy_total",
        "human_mean_corrected_policy_total",
    ):
        labels = references.get(reference_name, {})
        for script_id, prediction in predictions.items():
            human = labels.get(script_id)
            if not human:
                continue
            ai_set = set(prediction.get("selected_question_ids", []) or [])
            if reference_name == "human_mean_corrected_policy_total":
                human_set = set(human.get("original_selected_question_ids", [])) | set(
                    human.get("grader2_selected_question_ids", [])
                )
            else:
                human_set = set(human.get("selected_question_ids", []))
            exact = ai_set == human_set
            aggregate[f"{reference_name}_paired_count"] += 1
            aggregate[f"{reference_name}_exact_match_count"] += int(exact)
            rows.append(
                {
                    "condition": condition,
                    "script_id": script_id,
                    "reference": reference_name,
                    "ai_selected_question_ids": ",".join(sorted(ai_set)),
                    "human_selected_question_ids": ",".join(sorted(human_set)),
                    "raw_all_question_total": prediction.get("raw_all_questions_total_score"),
                    "raw_attempted_total": prediction.get("raw_attempted_total_score"),
                    "raw_selected_total": prediction.get("raw_selected_total_score"),
                    "final_normalized_total": prediction.get("ai_total_score"),
                    "exact_match": exact,
                    "ai_minus_human": ",".join(sorted(ai_set - human_set)),
                    "human_minus_ai": ",".join(sorted(human_set - ai_set)),
                }
            )
    return dict(aggregate), rows


def _extraction_rows(extractions_path: str | None) -> list[dict[str, Any]]:
    if not extractions_path:
        return []
    path = Path(extractions_path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        extraction = json.loads(line)
        attempted = [
            question.get("question_id")
            for question in extraction.get("questions", []) or []
            if question.get("attempted")
        ]
        rows.append(
            {
                "script_id": extraction.get("script_id"),
                "page_count": extraction.get("page_count"),
                "pages_reported": extraction.get("pages_reported"),
                "extracted_attempted_questions": ",".join(attempted),
                "missing_blank_questions": ",".join(extraction.get("missing_or_blank_questions", []) or []),
                "transcription_warnings": json.dumps(extraction.get("transcription_warnings", []) or []),
                "validation_error_count": len(extraction.get("extraction_validation_errors", []) or []),
            }
        )
    return rows


def _corrected_ablation_rows(
    no_lecture: dict[str, dict[str, Any]],
    lecture: dict[str, dict[str, Any]],
    primary: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for script_id in sorted(set(no_lecture) & set(lecture) & set(primary), key=lambda x: int(x) if x.isdigit() else x):
        no_pred = no_lecture[script_id]
        le_pred = lecture[script_id]
        ref = float(primary[script_id]["human_total_score"])
        rows.append(
            {
                "script_id": script_id,
                "primary_reference_total": ref,
                "no_lecture_ai_total": no_pred.get("ai_total_score"),
                "lecture_ai_total": le_pred.get("ai_total_score"),
                "no_lecture_raw_all_questions_total": no_pred.get("raw_all_questions_total_score"),
                "lecture_raw_all_questions_total": le_pred.get("raw_all_questions_total_score"),
                "no_lecture_raw_attempted_total": no_pred.get("raw_attempted_total_score"),
                "lecture_raw_attempted_total": le_pred.get("raw_attempted_total_score"),
                "no_lecture_raw_selected_total": no_pred.get("raw_selected_total_score"),
                "lecture_raw_selected_total": le_pred.get("raw_selected_total_score"),
                "no_lecture_absolute_error": abs(float(no_pred.get("ai_total_score", 0)) - ref),
                "lecture_absolute_error": abs(float(le_pred.get("ai_total_score", 0)) - ref),
                "no_lecture_signed_error": float(no_pred.get("ai_total_score", 0)) - ref,
                "lecture_signed_error": float(le_pred.get("ai_total_score", 0)) - ref,
                "lecture_minus_no_lecture": float(le_pred.get("ai_total_score", 0)) - float(no_pred.get("ai_total_score", 0)),
            }
        )
    return rows


def _write_corrected_metrics_stage_io(
    run_dir: Path,
    consumed_files: list[Path],
    output_json: Path,
    paper_tables: list[Path],
) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    metrics_input = run_dir / "stage_io" / "metrics" / "inputs" / "corrected_staged.json"
    metrics_output = run_dir / "stage_io" / "metrics" / "outputs" / "corrected_staged.json"
    metrics_input.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    input_payload = {
        "stage": "metrics",
        "consumed_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            }
            for path in consumed_files
        ],
    }
    metrics_input.write_text(json.dumps(input_payload, indent=2, sort_keys=True))
    output_payload = {
        "stage": "metrics",
        "metrics_corrected_staged": {
            "path": str(output_json),
            "size_bytes": output_json.stat().st_size if output_json.exists() else None,
            "sha256": hashlib.sha256(output_json.read_bytes()).hexdigest() if output_json.exists() else None,
        },
        "paper_tables": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            }
            for path in paper_tables
            if path.exists()
        ],
    }
    metrics_output.write_text(json.dumps(output_payload, indent=2, sort_keys=True))
    record = {
        "script_id": None,
        "condition": None,
        "stage": "metrics",
        "input_path": str(metrics_input),
        "output_path": str(metrics_output),
        "raw_output_path": None,
        "parsed_output_path": str(output_json),
        "prompt_path": None,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": 0,
        "status": "succeeded",
        "validation_errors": [],
        "warnings": [],
    }
    stage_jsonl = run_dir / "stage_io.jsonl"
    records = []
    if stage_jsonl.exists():
        records = [
            json.loads(line)
            for line in stage_jsonl.read_text().splitlines()
            if line.strip()
        ]
        records = [
            item
            for item in records
            if not (item.get("script_id") is None and item.get("stage") == "metrics")
        ]
    records.append(record)
    stage_jsonl.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records)
    )
    scripts: dict[str, Any] = {}
    globals_: dict[str, Any] = {}
    for item in records:
        script_id = item.get("script_id")
        condition = item.get("condition")
        stage = item.get("stage")
        if script_id is None:
            globals_[stage] = item
            continue
        script_trace = scripts.setdefault(script_id, {"stages": {}, "conditions": {}})
        if condition is None:
            script_trace["stages"][stage] = item
        else:
            script_trace["conditions"].setdefault(condition, {"stages": {}})["stages"][stage] = item
    (run_dir / "pipeline_trace_index.json").write_text(
        json.dumps({"scripts": scripts, "global": globals_}, indent=2, sort_keys=True)
    )


def _write_extraction_review_queue(
    table_dir: Path,
    no_lecture: dict[str, dict[str, Any]],
    lecture: dict[str, dict[str, Any]],
    original: PerQuestionScoreBook,
    grader2: PerQuestionScoreBook,
    references: dict[str, dict[str, dict[str, Any]]],
) -> None:
    primary = references["human_mean_corrected_policy_total"]
    rows: list[dict[str, Any]] = []
    targeted = {"3", "11", "30", "53"}
    for condition, predictions in (("no_lecture", no_lecture), ("lecture", lecture)):
        other = lecture if condition == "no_lecture" else no_lecture
        for script_id, prediction in predictions.items():
            primary_total = primary.get(script_id, {}).get("human_total_score")
            high_error = (
                primary_total is not None
                and abs(float(prediction.get("ai_total_score", 0)) - float(primary_total)) >= 10
            )
            selected_diff = set(prediction.get("selected_question_ids", []) or []) != set(
                primary.get(script_id, {}).get("original_selected_question_ids", [])
            )
            condition_diff = script_id in other and set(
                prediction.get("selected_question_ids", []) or []
            ) != set(other[script_id].get("selected_question_ids", []) or [])
            include_script = (
                script_id in targeted
                or prediction.get("validation_errors")
                or prediction.get("transcription_warnings")
                or prediction.get("extraction_transcription_warning_count", 0)
                or high_error
                or selected_diff
                or condition_diff
            )
            if not include_script:
                continue
            questions = {
                q.get("question_id"): q
                for q in prediction.get("graded_questions", []) or []
                if isinstance(q, dict)
            }
            for qid in QUESTION_ORDER:
                question = questions.get(qid, {})
                rows.append(
                    {
                        "script_id": script_id,
                        "question_id": qid,
                        "condition": condition,
                        "source_pdf_path": prediction.get("script_path"),
                        "extraction_json_path": prediction.get("extraction_path"),
                        "raw_extraction_response_path": prediction.get("extraction_raw_model_response_path"),
                        "raw_grading_response_path": prediction.get("raw_model_response_path"),
                        "extracted_attempted": qid not in (prediction.get("missing_or_blank_questions") or []),
                        "extracted_answer_text": "",
                        "extracted_subparts_json": "",
                        "ai_score": (prediction.get("raw_model_question_scores") or {}).get(qid),
                        "ai_normalized_score": (prediction.get("normalized_question_scores") or {}).get(qid),
                        "ai_selected_for_total": qid in (prediction.get("selected_question_ids") or []),
                        "ai_justification": question.get("justification"),
                        "ai_feedback": question.get("feedback"),
                        "original_q_score": original.labels.get(script_id).scores.get(qid) if script_id in original.labels else None,
                        "grader2_q_score": grader2.labels.get(script_id).scores.get(qid) if script_id in grader2.labels else None,
                        "primary_reference_total": primary_total,
                        "validation_errors": json.dumps(prediction.get("validation_errors", []) or []),
                        "transcription_warnings": json.dumps(
                            (prediction.get("transcription_warnings", []) or [])
                            + [str(prediction.get("extraction_transcription_warning_count", 0))]
                        ),
                        "reviewer_pages_checked": "",
                        "reviewer_student_answer_visible": "",
                        "reviewer_extraction_matches_source": "",
                        "reviewer_marking_correction_visible": "",
                        "reviewer_score_reasonable_under_corrected_rubric": "",
                        "reviewer_rationale_matches_question": "",
                        "reviewer_propagation_risk": "",
                        "reviewer_notes": "",
                    }
                )
    _write_csv(
        table_dir.parent / "extraction_review_queue.csv",
        [
            "script_id",
            "question_id",
            "condition",
            "source_pdf_path",
            "extraction_json_path",
            "raw_extraction_response_path",
            "raw_grading_response_path",
            "extracted_attempted",
            "extracted_answer_text",
            "extracted_subparts_json",
            "ai_score",
            "ai_normalized_score",
            "ai_selected_for_total",
            "ai_justification",
            "ai_feedback",
            "original_q_score",
            "grader2_q_score",
            "primary_reference_total",
            "validation_errors",
            "transcription_warnings",
            "reviewer_pages_checked",
            "reviewer_student_answer_visible",
            "reviewer_extraction_matches_source",
            "reviewer_marking_correction_visible",
            "reviewer_score_reasonable_under_corrected_rubric",
            "reviewer_rationale_matches_question",
            "reviewer_propagation_risk",
            "reviewer_notes",
        ],
        rows,
    )


def compute_corrected_staged_metrics(args: argparse.Namespace) -> dict[str, Any]:
    if not args.original_score_sheet or not args.grader2_score_sheet:
        raise ValueError("--original-score-sheet and --grader2-score-sheet are required")
    if not args.predictions_no_lecture or not args.predictions_lecture:
        raise ValueError("--predictions-no-lecture and --predictions-lecture are required")
    if not args.rubric_json:
        raise ValueError("--rubric-json is required")
    rubric = load_exam_rubric(Path(args.rubric_json))
    original = parse_per_question_scores_csv(Path(args.original_score_sheet), "original")
    grader2 = parse_per_question_scores_csv(Path(args.grader2_score_sheet), "grader2")
    no_lecture = _load_prediction_rows(args.predictions_no_lecture)
    lecture = _load_prediction_rows(args.predictions_lecture)
    references = _corrected_human_references(original, grader2, rubric)
    total_no, total_rows_no = _condition_total_rows("no_lecture", no_lecture, references)
    total_le, total_rows_le = _condition_total_rows("lecture", lecture, references)
    per_no, per_rows_no = _corrected_per_question_rows("no_lecture", no_lecture, original, grader2)
    per_le, per_rows_le = _corrected_per_question_rows("lecture", lecture, original, grader2)
    safety_no, safety_rows_no, trace_rows_no = _corrected_safety_and_trace_rows("no_lecture", no_lecture)
    safety_le, safety_rows_le, trace_rows_le = _corrected_safety_and_trace_rows("lecture", lecture)
    selection_no, selection_rows_no = _selection_rows("no_lecture", no_lecture, references)
    selection_le, selection_rows_le = _selection_rows("lecture", lecture, references)
    primary = references["human_mean_corrected_policy_total"]
    ablation_metrics = compute_lecture_ablation_metrics(primary, no_lecture, lecture)
    ablation_rows = _corrected_ablation_rows(no_lecture, lecture, primary)
    raw_score_metrics = {
        condition: {
            key: _raw_distribution([float(pred.get(key, 0) or 0) for pred in predictions.values()])
            for key in (
                "raw_all_questions_total_score",
                "raw_attempted_total_score",
                "raw_selected_total_score",
                "ai_total_score",
            )
        }
        for condition, predictions in (("no_lecture", no_lecture), ("lecture", lecture))
    }
    for condition, predictions in (("no_lecture", no_lecture), ("lecture", lecture)):
        raw_score_metrics[condition]["raw_all_differs_from_final_count"] = sum(
            1
            for pred in predictions.values()
            if float(pred.get("raw_all_questions_total_score", 0) or 0)
            != float(pred.get("ai_total_score", 0) or 0)
        )
    report = {
        "mode": "corrected_staged",
        "primary_reference": "human_mean_corrected_policy_total",
        "rubric": {
            "source_path": str(args.rubric_json),
            "selection_policy": rubric.selection_policy,
            "allowed_question_count": rubric.allowed_question_count,
            "question_equal_max_score": rubric.question_equal_max_score,
        },
        "references": references,
        "total_metrics": {"no_lecture": total_no, "lecture": total_le},
        "per_question_metrics": {"no_lecture": per_no, "lecture": per_le},
        "raw_score_metrics": raw_score_metrics,
        "selection_metrics": {"no_lecture": selection_no, "lecture": selection_le},
        "safety_metrics": {"no_lecture": safety_no, "lecture": safety_le},
        "lecture_ablation": ablation_metrics,
        "counts": {
            "no_lecture_predictions": len(no_lecture),
            "lecture_predictions": len(lecture),
            "primary_reference_labels": len(primary),
        },
    }
    if args.write_paper_tables:
        table_dir = Path(args.write_paper_tables)
        _write_csv(
            table_dir / "corrected_total_metrics.csv",
            [
                "condition",
                "reference",
                "paired_count",
                "mae_marks",
                "mean_signed_error_marks",
                "exact_match_rate",
                "within_1_mark_rate",
                "within_3_mark_rate",
                "within_5_percent_rate",
                "qwk_percent_buckets",
                "mean_confidence",
                "hitl_trigger_rate",
            ],
            total_rows_no + total_rows_le,
        )
        _write_csv(
            table_dir / "corrected_per_question_metrics.csv",
            [
                "condition",
                "question_id",
                "reference_policy",
                "paired_count",
                "mae",
                "rmse",
                "mean_signed_error",
                "exact_match_rate",
                "within_1_rate",
                "within_2_rate",
                "within_3_rate",
            ],
            per_rows_no + per_rows_le,
        )
        _write_csv(
            table_dir / "corrected_safety_gates.csv",
            [
                "condition",
                "script_id",
                "ai_total_score",
                "raw_all_questions_total_score",
                "raw_attempted_total_score",
                "raw_selected_total_score",
                "selected_question_ids",
                "unselected_question_ids",
                "referee_status",
                "validation_error_count",
                "extraction_warning_count",
                "extraction_validation_error_count",
                "mean_confidence",
            ],
            safety_rows_no + safety_rows_le,
        )
        _write_csv(
            table_dir / "corrected_raw_score_trace.csv",
            [
                "condition",
                "script_id",
                "question_id",
                "raw_model_score",
                "model_reported_max_score",
                "corrected_raw_max_score",
                "bounded_raw_score",
                "normalized_score",
                "attempted_from_extraction",
                "selected_for_total",
                "subpart_scores_json",
                "justification",
                "feedback",
            ],
            trace_rows_no + trace_rows_le,
        )
        _write_csv(
            table_dir / "corrected_selection_by_script.csv",
            [
                "condition",
                "script_id",
                "reference",
                "ai_selected_question_ids",
                "human_selected_question_ids",
                "raw_all_question_total",
                "raw_attempted_total",
                "raw_selected_total",
                "final_normalized_total",
                "exact_match",
                "ai_minus_human",
                "human_minus_ai",
            ],
            selection_rows_no + selection_rows_le,
        )
        _write_csv(
            table_dir / "corrected_extraction_summary.csv",
            [
                "script_id",
                "page_count",
                "pages_reported",
                "extracted_attempted_questions",
                "missing_blank_questions",
                "transcription_warnings",
                "validation_error_count",
            ],
            _extraction_rows(args.extractions),
        )
        _write_csv(
            table_dir / "corrected_ablation_by_script.csv",
            [
                "script_id",
                "primary_reference_total",
                "no_lecture_ai_total",
                "lecture_ai_total",
                "no_lecture_raw_all_questions_total",
                "lecture_raw_all_questions_total",
                "no_lecture_raw_attempted_total",
                "lecture_raw_attempted_total",
                "no_lecture_raw_selected_total",
                "lecture_raw_selected_total",
                "no_lecture_absolute_error",
                "lecture_absolute_error",
                "no_lecture_signed_error",
                "lecture_signed_error",
                "lecture_minus_no_lecture",
            ],
            ablation_rows,
        )
        _write_extraction_review_queue(table_dir, no_lecture, lecture, original, grader2, references)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        consumed = [
            Path(args.predictions_no_lecture),
            Path(args.predictions_lecture),
            Path(args.rubric_json),
            Path(args.original_score_sheet),
            Path(args.grader2_score_sheet),
        ]
        if args.extractions:
            consumed.append(Path(args.extractions))
        paper_tables = list(table_dir.glob("corrected_*.csv")) + [table_dir.parent / "extraction_review_queue.csv"]
        _write_corrected_metrics_stage_io(table_dir.parent, consumed, output_path, paper_tables)
    return report

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute grading accuracy metrics (question-level or total-score)."
    )
    parser.add_argument(
        "--gold",
        default=None,
        help="Path to gold_standard.json for question-level metrics",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write results JSON",
    )
    parser.add_argument(
        "--mongo-uri",
        default=None,
        help="MongoDB connection URI (reads from results collection)",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Path to experiment predictions.jsonl for total-score metrics",
    )
    parser.add_argument(
        "--total-labels",
        default=None,
        help="Path to total-score labels JSON",
    )
    parser.add_argument(
        "--scores-pdf",
        default=None,
        help="Path to score PDF for total-score labels",
    )
    parser.add_argument(
        "--ablation-a",
        default=None,
        help="Path to no-lecture predictions JSONL for lecture ablation metrics",
    )
    parser.add_argument(
        "--ablation-b",
        default=None,
        help="Path to lecture-context predictions JSONL for lecture ablation metrics",
    )
    parser.add_argument(
        "--original-score-sheet",
        default=None,
        help="Path to original lecturer per-question score-sheet CSV",
    )
    parser.add_argument(
        "--grader2-score-sheet",
        default=None,
        help="Path to second lecturer per-question score-sheet CSV",
    )
    parser.add_argument(
        "--per-question-human-report",
        action="store_true",
        help="Compute human-human per-question reliability and score-sheet audit metrics",
    )
    parser.add_argument(
        "--per-question",
        action="store_true",
        help="Compute legacy AI-human per-question metrics from prediction JSONL files",
    )
    parser.add_argument(
        "--corrected-staged",
        action="store_true",
        help="Compute corrected staged metrics with best-four equal-weighted finalization artifacts",
    )
    parser.add_argument(
        "--predictions-no-lecture",
        default=None,
        help="Path to no-lecture prediction JSONL for legacy per-question metrics",
    )
    parser.add_argument(
        "--predictions-lecture",
        default=None,
        help="Path to lecture-summary prediction JSONL for legacy per-question metrics",
    )
    parser.add_argument(
        "--reference-policy",
        choices=(
            "original_computed_sum",
            "grader2_total",
            "strict_consensus_cells",
            "individual_human_cells",
        ),
        default="strict_consensus_cells",
        help="Human reference policy for legacy per-question metrics",
    )
    parser.add_argument(
        "--rubric-json",
        default=None,
        help="Corrected rubric JSON path for corrected staged metrics",
    )
    parser.add_argument(
        "--extractions",
        default=None,
        help="Path to staged extractions.jsonl for corrected staged extraction metrics",
    )
    parser.add_argument(
        "--write-paper-tables",
        default=None,
        help="Directory for CSV tables used by the paper draft",
    )
    args = parser.parse_args()

    total_mode = any(
        [args.predictions, args.total_labels, args.scores_pdf, args.ablation_a, args.ablation_b]
    )

    try:
        if args.corrected_staged:
            metrics = compute_corrected_staged_metrics(args)
        elif args.per_question_human_report:
            metrics = compute_human_human_report(args)
        elif args.per_question:
            metrics = compute_legacy_per_question_metrics(args)
        elif total_mode:
            labels = load_total_labels(args)
            if args.ablation_a or args.ablation_b:
                if not args.ablation_a or not args.ablation_b:
                    parser.error("--ablation-a and --ablation-b must be provided together")
                metrics = compute_lecture_ablation_metrics(
                    labels,
                    load_predictions_jsonl(Path(args.ablation_a)),
                    load_predictions_jsonl(Path(args.ablation_b)),
                )
                print_total_table(metrics.get("with_lecture", metrics))
            else:
                if not args.predictions:
                    parser.error("--predictions is required for total-score metrics")
                predictions_path = Path(args.predictions)
                metrics = compute_total_score_metrics(
                    labels,
                    load_predictions_jsonl(predictions_path),
                    failed_count=count_sibling_failures(predictions_path),
                )
                print_total_table(metrics)
        else:
            metrics = compute_question_level_metrics(args)
            print_table(metrics)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    write_output(args.output, metrics)


if __name__ == "__main__":
    main()
