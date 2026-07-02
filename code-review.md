# Code Review: V7b Repair Instrumentation

## Scope

Review of the V7b change in `lingbot-va-riskrouter-repairinstrument-20260702`.

The intended single variable is repair instrumentation only: request repaired action candidates from the teacher verifier, reverify them, log diagnostic pass/fail fields, but keep the executed policy unchanged by falling back to the original teacher path.

## Findings

No blocking findings.

## Root Cause Found

V7 did not actually enable repair in `RiskRouterClientPolicy`: the launcher passed `--repair_enable`, but `eval_polict_client_openpi.py` did not forward `repair_enable` or `repair_lambda` into the policy constructor. This explains `draft_repair_accept=0` without evidence of `teacher_repair_reject`.

## Risk Assessment

Risk level: medium.

The new instrumentation adds extra teacher verify calls on rejected draft chunks, so latency may increase. This is acceptable for V7b because the goal is diagnosis, not speed. It should not change action success/failure decisions when `--repair_instrument_only` is used.

## Verification

- `python3 -m py_compile evaluation/robotwin/specverify_client_policy.py evaluation/robotwin/eval_polict_client_openpi.py scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`
- `python3 -m pytest -q tests/test_specverify_client_policy.py tests/test_specverify.py`

Result: `15 passed`.

## Proceed Decision

Allowed to proceed to low10 TN=10 V7b evaluation with:

- `--repair-enable`
- `--repair-instrument-only`
- same V6/V7 fixed controls

Do not enable repaired-action execution until V7b shows nonzero `repair_accept` candidates and no runtime instability.
