# Downstream contract propagation protocol

## Study question

When grading artifacts from the staged Intro Python evaluation are passed into downstream assessment stages, which explicit contract violations propagate into proposed user-visible or persistent actions, and which deterministic gates block them?

## Claim boundary

This protocol measures software-workflow behavior: contract violations, review-gate decisions, allowed or blocked downstream action proposals, and reproducible artifact traces. It does not measure teacher-verified grading correctness, feedback quality, fairness, learning benefit, or deployed student impact.

## Source artifacts

Primary source run: `eval/runs/intro-python-staged-corrected`.

Inputs used by this protocol:

- `predictions.lecture.jsonl`
- `predictions.no_lecture.jsonl`
- `extractions.jsonl`
- `rubric.json` or `eval/rubric_corrected_intro_python.json`
- `metrics.corrected_staged.json` as contextual background only
- `stage_io.jsonl` and `pipeline_trace_index.json` as trace metadata

## Unit of analysis

The independent unit is a script-level artifact under one lecture condition. Per-action and per-violation counts may be reported descriptively, but uncertainty and comparisons are computed over script-level cases, not individual actions or individual questions.

## Conditions

Each selected script-condition artifact is expanded into variants:

1. `observed`: saved upstream artifact without intentional changes.
2. `negative_control`: a contract-preserving copy with benign metadata normalization only.
3. `controlled_fault`: one objectively defined violation injected from `fault_library.json`.
4. `repaired_counterfactual`: deterministic repair or blocking state for the corresponding controlled fault.

The default controlled fault for broad runs is `drop_pending_review_status`, because it directly tests whether review state is preserved before user-visible or persistent actions.

## Downstream stages

All downstream stages are isolated. They never write to MongoDB, GCS, ALOC, queues, production records, or student-facing channels.

- `review_gate`: deterministic review decision from artifact state and contract violations.
- `feedback_proposal`: local proposal for student-facing feedback, blocked when review status requires human review.
- `weakness_proposal`: local proposal for weak-topic update, blocked when review status requires human review.
- `practice_proposal`: local proposal for remediation/practice session, blocked when review status, linked-user, or exam-type gates fail.
- `result_persistence_proposal`: local proposed result document, blocked or marked `PENDING_REVIEW` when validation or review gates require it.

Dry-run mode emits deterministic placeholder proposals and must not call a model.

## Primary endpoints

- Contract-violation detection rate.
- Action-propagation rate: faulty variants whose user-visible or persistent action is allowed despite an expected block.
- Gate containment rate: faulty variants blocked from action propagation.
- Repair effect: difference between controlled-fault and repaired-counterfactual propagation.
- False-block rate on negative controls.
- Review-demand rate.
- Model-call failure rate and parse-failure rate.
- Latency and raw-output availability.

## Protocol freeze

The protocol directory hash is computed before confirmatory execution over these files:

- `protocol.md`
- `fault_library.json`
- `action_contracts.json`
- `analysis_plan.md`

Confirmatory runs must pass `--require-protocol-hash` with the recorded hash. Result-dependent changes to prompts, metrics, fault definitions, or action contracts after this hash are not permitted for the confirmatory analysis.
