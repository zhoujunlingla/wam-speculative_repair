# Code Review: V5 Lazy Partial-Prefix Guard

## Change Reviewed

Repository: `/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-riskrouter-lazyprefix-20260701`

Single change: extend the existing partial-prefix rejection guard from `stale_reference` to both `stale_reference` and `lazy_reference` teacher cache modes.

## Findings

No blocking findings.

Risk level: low. This is a narrow correctness guard. It intentionally reduces some speculative acceptance opportunities, but only for partial chunks under non-sync teacher cache modes. Full-chunk draft execution and verifier accept/reject behavior are unchanged.

## Evidence

V4 lazy-reference crash evidence showed both shards accepted a 16-step partial prefix, then queued a 1-frame pending teacher cache update. When teacher fallback later attempted lazy catch-up, LingBot teacher KV-cache update received `action_model_input=None` and crashed before any valid trial.

This patch prevents that incompatible path by forcing partial verifier accept to become a teacher fallback. That keeps teacher cache updates aligned with complete chunk/frame boundaries.

## Tests Run

- `lazy partial-prefix smoke passed`: fake draft + fake verifier confirmed `accepted_prefix=16` in `lazy_reference` becomes `teacher_verify_reject`, while `accepted_prefix=32` remains `draft_verify_accept`.
- `python3 -m pytest -q tests/test_specverify.py tests/test_specverify_client_policy.py`
  - Result: `8 passed in 1.91s`

## Decision

Allowed to proceed to V5 low10 TN=10 relaunch on allowed GPUs. Continue to monitor whether teacher rate increases too much from rejecting partial prefixes; if so, the next single-variable iteration should target a cache-safe partial execution mechanism rather than re-enabling partial prefixes blindly.

## Additional Launcher Path Review

Finding: the copied launcher still pointed `CODE` at the older `lingbot-va-riskrouter-highverify-20260701` root. This would silently run the wrong code and invalidate V5 results.

Fix: update the launcher `CODE` constant to `/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-riskrouter-lazyprefix-20260701`.

Verification: reran `python3 -m pytest -q tests/test_specverify.py tests/test_specverify_client_policy.py`; result `8 passed in 1.56s`.

Decision: allowed to proceed.
