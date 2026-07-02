# Code Review: V8 SVDR Video-Guided Draft Repair

## Scope

Review of the V8 change in `lingbot-va-svdr-videorepair-20260702`.

The intended single variable is training-free SVDR: when action verification rejects a draft, use FlashWAM draft future video latent motion/region risk to weight the teacher endpoint residual repair, then run the existing action-only verifier again before teacher fallback.

## Findings

No blocking findings.

## Diff Summary

- Added `video_motion_risk` and `video_guided_blend` in `evaluation/robotwin/specverify_client_policy.py`.
- Added SVDR policy flags and forwarding in `evaluation/robotwin/eval_polict_client_openpi.py`.
- Updated the low10 launcher/summary script to record `draft_svdr_repair_accept` and `teacher_svdr_repair_reject`.
- Added unit tests that verify high-motion latent frames receive stronger repair and that SVDR requests a full teacher endpoint repair before applying video-guided per-step blending.

## Risk Assessment

Risk level: medium.

- Runtime risk: SVDR requires the draft server to return `video_latent` when `return_video_latent=True`; the first run must be a smoke test before full low10.
- Policy risk: the repaired action is now executable, not instrumentation-only. The existing action verifier remains the gate, but this can change closed-loop behavior.
- Experiment-validity risk: keep `world_verify_enable` off for V8, otherwise SVDR would be mixed with world-latent verifier changes.

## Verification

- `python3 -m py_compile evaluation/robotwin/specverify_client_policy.py evaluation/robotwin/eval_polict_client_openpi.py scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`
- `python3 -m pytest -q tests/test_specverify_client_policy.py tests/test_specverify.py`

Result: `17 passed`.

## Proceed Decision

Allowed to proceed to an SVDR smoke evaluation, then low10 only if the smoke confirms video latent return and no reset/cache mismatch.

- `--repair-enable`
- `--svdr-repair-enable`
- `--world-verify-enable` disabled
- same V6/V7 action verifier and risk-router controls

Do not tune SVDR thresholds until the first smoke/low10 run shows whether `draft_svdr_repair_accept` is nonzero.
