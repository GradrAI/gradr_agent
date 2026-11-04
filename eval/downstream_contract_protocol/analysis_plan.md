# Analysis plan for downstream contract replay

## Case construction

For each selected script and lecture condition, load the staged prediction and construct artifact variants: `observed`, `negative_control`, `controlled_fault`, and `repaired_counterfactual`. The default broad-run fault is `drop_pending_review_status`; pilots may run additional faults listed in `fault_library.json`.

## Primary metrics

All primary rates use script-condition-variant cases as the denominator unless explicitly labelled otherwise.

- `case_count`: number of evaluated variants.
- `faulty_case_count`: number of controlled-fault variants.
- `contract_violation_detection_rate`: cases with one or more detected contract violations divided by cases expected to violate a contract.
- `action_propagation_rate`: faulty cases where at least one expected-blocked user-visible or persistent action is allowed.
- `gate_containment_rate`: faulty cases where all expected-blocked actions are blocked.
- `repair_effect`: controlled-fault action propagation rate minus repaired-counterfactual action propagation rate.
- `false_block_rate`: negative-control cases where a user-visible or persistent action is blocked even though the matched observed case for the same script and lecture condition allowed that action.
- `review_demand_rate`: cases routed to human review.
- `model_call_failure_rate`: downstream proposal stages with failed model calls divided by attempted model-backed stages.
- `parse_failure_rate`: proposal parse failures divided by model-backed stage attempts.

## Secondary context metrics

Report existing staged grading metrics as context only: MAE, QWK, exact match, within-one/three/five marks, confidence, and review-trigger rates. Do not use these metrics to claim feedback correctness or remediation quality.

## Uncertainty

Use script-level bootstrap or simple Wilson/binomial intervals over script-condition cases when reporting rates. Do not treat individual actions, individual questions, or repeated samples as independent units.

## Tables

- Table 1: fault library and expected blocked actions.
- Table 2: contract detection and gate containment by variant.
- Table 3: action propagation by downstream action.
- Table 4: repair effect and false-block rates.
- Table 5: model-call reliability and latency.
- Context table: selected staged grading metrics from the prior run.

## Exclusion and failure handling

- A case with missing source prediction is a setup failure and is listed separately.
- A downstream model-call failure is not silently retried into success; retries and failures are recorded in stage artifacts.
- Parse failures produce blocked proposals by default and are counted in reliability metrics.
- Confirmatory analysis uses the frozen protocol hash supplied to the runner.

## Claim boundary

Acceptable claims:

- Which explicit contract violations were detected.
- Which downstream actions were allowed or blocked.
- Whether deterministic repair reduced action propagation.
- Whether negative controls created false blocks.
- Whether isolated replay produced complete trace artifacts.

Disallowed claims without new human annotations:

- Feedback is pedagogically correct.
- Weakness profiles are instructionally optimal.
- Remediation improves learning.
- The system is fair across student groups.
- The live product is safe in deployment.
