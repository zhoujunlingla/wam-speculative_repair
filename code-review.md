# V3 Code Review

## Diff Reviewed

Command: `git diff`

Changed files:

1. `evaluation/robotwin/specverify_client_policy.py`
2. `scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`
3. `design.md`
4. `code-review.md`

## Changes

### `RiskRouterClientPolicy`

- Removed the direct `risk_score >= risk_high -> teacher_router_high` execution branch.
- Added `high_risk = risk_score >= risk_high` and lets high-risk chunks continue to the verifier path.
- Verifier metadata now records `risk_zone = high|medium`.
- Low-risk bypass still applies only when `risk_score < risk_low` and no phase switch.
- Phase switch behavior from V2 is unchanged: it tightens the verification threshold.

### Launcher summary

- `CODE` points to the V3 repo.
- Metrics now count `high_verify`, `high_verify_accept`, and `high_verify_reject`.

## Review Findings

No blocking issues in the intended single-variable diff.

Residual risks:

- High-risk accepted drafts may harm success if verifier threshold is too loose for contact tasks.
- High-risk rejected drafts still pay verifier + teacher, so latency could worsen if rejection is common.

Both risks are measurable via `high_verify_accept/reject`, source latencies, and low10 success.

## Required Smoke Tests

- Python compile modified files.
- Fake-client high-risk action should call verifier first and avoid teacher action when accepted.
