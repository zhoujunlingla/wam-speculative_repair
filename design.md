# V3 RiskRouter High-Risk Verify Design

## Source

This repo was copied from the V2 phase-verify baseline:

- Source: `/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-riskrouter-phaseverify-20260701`
- Source commit: `acfaf1f`
- New repo: `/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-riskrouter-highverify-20260701`

## Evidence From V2

V2 changed phase handling from hard teacher routing to tightened verification. Early evidence:

- easier shard improved speed structure: put_object_cabinet shard had teacher rate `23.8%` and first trial `1/1` success.
- hard shard still failed speed gate: hanging_mug shard had teacher rate `76.2%` before any completed trial.

This shows phase handling was not the only source of teacher overuse. The remaining culprit is the hard branch:

`risk_score >= risk_high -> teacher_router_high`

## Hypothesis

High action risk should trigger stronger verification, not immediate teacher execution. This is closer to speculative sampling: draft proposes, teacher verifies, teacher executes only when verification rejects.

## Single New Variable

Only high-risk handling changes:

- V2: high-risk chunks execute teacher directly.
- V3: high-risk chunks call the action-flow verifier first. If accepted, execute draft prefix; if rejected, fallback to teacher.

No new threshold is introduced for high-risk chunks. They use the same base verifier threshold. Phase switches still tighten the threshold.

## Fixed Controls

- Draft: LingBot v1/a2 direct
- Teacher/verifier: LingBot v2/a4 direct
- Teacher cache: `stale_reference`
- `risk_low=0.25`, `risk_high=0.55`
- `risk_verify_mode=medium`
- `specverify_threshold=0.18`
- `phase_threshold_scale=0.5`
- `risk_phase_weight=0.0`

## Gate

Run low10 TN=10. V3 is useful if:

- teacher action-source rate drops below V2/V1 and toward `<25%-30%`,
- success stays on track toward `>=67/100`,
- high_verify_accept is meaningfully larger than high_verify_reject,
- cache latency remains around V1/V2 `~0.52s`.

## Next Decision

If high-risk verifier accepts often but success drops, the verifier is too permissive. Next single variable: high-risk-specific tighter threshold.
If high-risk verifier rejects often, verifier cost rises but teacher rate remains high. Next single variable: repair/correction before teacher fallback.

## V5 Cache-Correctness Guard: Lazy Partial Prefix Rejection

Source evidence from V4 lazy-reference run `20260701_051809`: both shards accepted a 16-step partial draft prefix, then queued a 1-frame pending teacher cache update. When a later teacher action needed cache catch-up, LingBot's teacher KV update path received `action_model_input=None` and crashed before any valid trial.

Single new variable for V5: apply the same partial-prefix rejection used by `stale_reference` to `lazy_reference`. Under non-sync teacher cache modes, verified draft execution is now limited to either `0` or the full action chunk. This preserves ordered teacher cache compatibility while still allowing full-chunk draft execution and high-risk verification.

No verifier thresholds, risk weights, teacher model, draft model, tau set, or task split are changed.
