# Grading Accuracy Evaluation

Compute inter-rater reliability (QWK), accuracy, and calibration metrics between GradrAI's automated grading and human expert scores.

## Collecting Gold-Standard Data

1. Select ~50 student papers across exam types (WASSCE, JAMB, NECO) and subjects.
2. Have **2 or more** qualified teachers independently grade each paper using the official rubric.
3. Record scores in `gold_standard.json` — one entry per question per paper.
4. Use the consensus score (or average) as `human_score`. Track grader identity with `human_grader_id` for inter-rater analysis.

### gold_standard.json Schema

Each entry requires:

| Field | Type | Description |
|---|---|---|
| `eval_id` | string | Unique identifier (e.g. `gs-001`) |
| `exam_type` | string | `WASSCE`, `JAMB`, `NECO`, etc. |
| `subject` | string | Subject name |
| `question_id` | string | Question identifier |
| `student_answer_gcs_uri` | string | GCS URI of the student's answer script |
| `rubric_text` | string | Rubric/marking scheme for this question |
| `max_score` | number | Maximum possible score |
| `human_score` | number | Expert-assigned score |
| `human_grader_id` | string | Identifier for the human grader |

Optional fields for offline evaluation (no MongoDB needed):

| Field | Type | Description |
|---|---|---|
| `ai_score` | number | AI-assigned score (if known) |
| `ai_confidence` | number | AI confidence for this question (0-1) |

## Running the Evaluation

```bash
# From the gradr_agent/ directory:

# Offline mode (requires ai_score fields in gold_standard.json):
make eval-accuracy

# With MongoDB (reads AI results from production data):
uv run python -m eval.compute_metrics \
  --gold eval/gold_standard.json \
  --mongo-uri "mongodb+srv://..." \
  --output eval/results.json
```

## Interpreting QWK Scores

Quadratic Weighted Kappa (QWK) measures agreement between two raters, penalizing larger disagreements more heavily.

| QWK Range | Interpretation |
|---|---|
| < 0.40 | Poor agreement — AI grading unreliable |
| 0.40 - 0.60 | Moderate agreement — useful with human review |
| 0.60 - 0.80 | Substantial agreement — production-ready with HITL |
| > 0.80 | Excellent agreement — near human-level consistency |

### Other Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between AI and human scores. Lower is better.
- **Exact Match Rate**: Fraction of questions where AI score equals human score exactly.
- **HITL Trigger Rate**: Fraction of results where AI confidence < 0.70, triggering human review via RefereeAgent.
- **Confidence Calibration**: Reliability diagram data — when the AI says it's 80% confident, is it correct ~80% of the time?
