# V6 World-Latent Flow Consistency Verifier

## Goal

Validate whether LingBot-VA's video/world latent flow can improve speculative action acceptance beyond action-only verification.

## Source

Copied from V5 `lingbot-va-riskrouter-lazyprefix-20260701` commit `209a180`.

## Problem Evidence

Action-only RiskRouter variants either overused teacher fallback or crashed when trying to make non-sync teacher cache modes too aggressive. More importantly, action-space agreement can miss world-state errors in contact-rich tasks: two action chunks may be numerically close while inducing different object/contact outcomes.

## Hypothesis

If a draft chunk is trustworthy, its one-step/few-step predicted video latent should also be consistent with the teacher's video flow field. Adding a teacher video-latent flow consistency gate after action verification should reject action chunks whose imagined world transition is not teacher-consistent.

## Single New Feature

Add optional `world_verify` gate:

1. Draft server returns both normalized action latent and predicted video latent.
2. Teacher first runs the existing action-flow verifier.
3. If action verifier accepts the full chunk, teacher runs a sparse video latent flow verifier on the draft video latent.
4. The draft action is accepted only if both action and world-latent flow checks pass.

The first validation uses `teacher_cache_mode=sync` to isolate verifier quality from lazy/stale cache correctness issues.

## Fixed Controls

- Draft: LingBot v1/a2 direct.
- Teacher/verifier: LingBot v2/a4 direct.
- Action verify tau: `[150, 300]`.
- Action threshold: `0.18`.
- Risk thresholds: `risk_low=0.25`, `risk_high=0.55`.
- Phase switch tightens action threshold by `0.5`.

## New Controls

- `--world-verify-enable`.
- `--world-verify-threshold`: default `0.35` normalized L1 latent residual.
- `--world-verify-tau`: default `[150, 300]`.

## Gate

Small validation should show no runtime crash and produce world verifier metrics. Continue only if:

- world verifier rejects some action-passing chunks instead of being constant pass/fail,
- low10 success does not collapse relative to action-only variants,
- teacher source rate remains interpretable.

If world gate is too strict, next single variable is threshold calibration or logging-only mode. If it is useful but slow, next variable is running world verify only for high-risk chunks.

## V6b Threshold Calibration: P95 World Score

The first V6 calibration run showed the world-latent verifier was functional but too strict when using raw max patch distance: `world_n=16`, `pass=1`, `reject=15`; median `world_distance_max ~= 0.56` but median `world_distance_p95 ~= 0.24`. V6b keeps the same threshold (`0.35`) and changes only the aggregation criterion from max patch distance to p95 patch distance. This follows video-generation caching practice where percentile/top-k scores are more stable than a single worst patch.
