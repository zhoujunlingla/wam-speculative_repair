# Progress

## 2026-07-12 - Clean Realtime-VLA-FLASH Migration

- Base: `robbyant/lingbot-va@7c6ffa9`.
- Draft: FlashWAM official step3000 v1/a2.
- Teacher: LingBot-VA posttrain v2/a4.
- Implemented first-full routing, last-full teacher reference, queued real
  cache replay, shared-noise K-point endpoint verification, 16-action prefix
  boundaries, gripper hard fallback, periodic refresh, and L=0 same-observation
  replan.
- Excluded all repair, RiskRouter, world verifier, and Verify++ paths.
- Local state-machine checks: 11 passed.
- Next: remote verifier tests, real-model smoke, then matched four-task TN=10.
- First remote collection exposed a flash-attention/Torch ABI import failure
  before model construction. Torch attention is the required backend, so the
  optional flash-attention import was made lazy-safe and tests were retried.
- Remote focused suite: `14 passed`.
- Real A800 smoke loaded both models, reached `Render Well`, completed the
  initial teacher full/cache anchor, accepted draft flash rounds, performed a
  teacher-gripper `L=0` same-observation replan, replayed pending cache updates,
  and reached periodic refresh without a cache or shape exception.

### Prior Failure Carried Forward

V31 achieved only `41/80 = 51.25%` at delta 0.15 versus the historical draft
reference `25/40 = 62.50%`. It was not a faithful FLASH implementation: the
first round executed draft after a shadow prime, low-risk chunks bypassed
verification, teacher cache was synchronized every round, phase handling only
tightened a threshold, and the verifier used MAE including gripper channels.
Do not compare the new implementation to V31 as if only the threshold changed.
