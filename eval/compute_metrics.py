"""Grading accuracy evaluation: compute QWK, MAE, and calibration metrics.

Usage:
    uv run python -m eval.compute_metrics --gold eval/gold_standard.json --output eval/results.json
    uv run python -m eval.compute_metrics --gold eval/gold_standard.json --mongo-uri "mongodb://..." --output eval/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

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
    matches = sum(1 for h, a in zip(human_scores, ai_scores) if h == a)
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
    for conf, acc in zip(confidences, accuracies):
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
        print(f"\n  Confidence Calibration (Reliability Diagram Data)")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute grading accuracy metrics (QWK, MAE, calibration)."
    )
    parser.add_argument(
        "--gold",
        required=True,
        help="Path to gold_standard.json",
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
    args = parser.parse_args()

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

    # Print summary
    print_table(metrics)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
