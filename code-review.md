# V2 Code Review

## Diff Reviewed

Command: `git diff`

Changed files:

1. `evaluation/robotwin/specverify_client_policy.py`
2. `evaluation/robotwin/eval_polict_client_openpi.py`
3. `scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`

## Changes

### `RiskRouterClientPolicy`

- Added `phase_threshold_scale` to the constructor.
- If `phase_switch=True`, low-risk bypass is disabled; the chunk must go through verifier.
- During verifier call, `threshold` becomes `base_threshold * phase_threshold_scale` when phase switch is present.
- Verifier metadata now records `phase_switch`, `phase_mode`, `base_threshold`, and `effective_threshold`.

### `eval_polict_client_openpi.py`

- Passes `--specverify_phase_threshold_scale` into `RiskRouterClientPolicy`.

### Launcher script

- Points `CODE` to the V2 repo root.
- Default `--risk-phase-weight` changed from `0.25` to `0.0`, so phase no longer directly increases `risk_score`.
- Summary text now says phase switch tightens verify rather than contributing to risk.
- Metrics aggregation now counts phase-tightened teacher verify rejections.

## Review Findings

No blocking issues found in the intended single-variable diff.

Residual risk:

- A phase-switch chunk with very low motion now pays verifier cost instead of free draft. This is intended because gripper switch is a contact-phase boundary.
- If phase-tightened verify rejects too often, teacher rate may remain high through `teacher_verify_reject` rather than `teacher_router_high`. That will be visible in `specverify_counts`.

## Required Smoke Tests

- Python compile of modified files.
- Fake-client test where phase switch action has low continuous risk: V2 should call verifier with tightened threshold and should not call teacher action when verifier accepts.
