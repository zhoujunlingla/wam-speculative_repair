# V2 RiskRouter Phase-Verify Design

## Source

This repo was copied from the clean V1 stale-cache baseline:

- Source: `/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-riskrouter-stalecache-20260701`
- Source commit: `d6f646a`
- New repo: `/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-riskrouter-phaseverify-20260701`

## V1 Evidence

V1 proved that `teacher_cache_mode=stale_reference` reduced cache-update cost from about `1.05-1.10s` to about `0.52s`, but it failed the router gate:

- low10a early: hanging_mug `0/7`, teacher rate `54.4%`
- low10b early: `6/12`, teacher rate `48.2%`
- gate target: teacher action-source rate `<25%-30%`, draft majority `>70%`

The main remaining bottleneck is that gripper/phase changes push the risk score above `risk_high`, sending too many rounds directly to the teacher.

## Hypothesis

Phase changes should not force teacher execution. In contact-rich tasks, phase changes are important, but a hard teacher fallback makes the system too slow and still did not rescue hanging_mug. A better first test is:

- keep V1 stale teacher cache unchanged,
- remove phase contribution from the cheap risk score by default,
- when a phase switch occurs, force medium-risk verification with a stricter action-flow threshold.

This turns phase from a teacher-routing signal into a verifier-tightening signal.

## Single New Variable

Only phase handling changes:

- V1: `phase_switch -> risk_score += phase_weight -> may trigger teacher_router_high`
- V2: `phase_switch -> verify_threshold = base_threshold * phase_threshold_scale`

All other main controls stay the same:

- draft: LingBot v1/a2 direct
- teacher/verifier: LingBot v2/a4 direct
- `teacher_cache_mode=stale_reference`
- `risk_low=0.25`, `risk_high=0.55`
- `risk_verify_mode=medium`
- tau `{150,300}`

## Gate

Run low10 TN=10. V2 is useful only if:

- success improves toward or above `67/100`, and
- teacher action-source rate falls materially below V1, ideally `<35%`, and
- draft_low_risk + draft_verify_accept rises toward `>70%`, and
- cache/update latency remains around V1 `~0.52s` rather than V0 `~1.05s`.

## Expected Failure Mode

If success drops while teacher rate falls, phase hard teacher was masking draft errors. Next single-variable candidate would be adaptive teacher correction only after verify rejection, not reintroducing periodic full refresh.
