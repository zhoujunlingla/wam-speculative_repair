# Code Review: Realtime-VLA-FLASH Migration

## Scope

Clean LingBot-VA commit `7c6ffa9` plus a new one-server speculative policy,
official-style endpoint verifier, RoboTwin replan protocol, model configs, and
four-task launcher. No repair or risk-routing code is included.

## Findings

No blocking static finding remains before the real-model smoke.

Fixed during review:

1. `prime_only` originally fell through to full draft generation. It now only
   encodes the initial observation needed by the draft's first real cache
   update.
2. `PF=0` originally forced a full round every time. It now disables periodic
   refresh.
3. Teacher-reconstructed gripper transitions were not hard fallbacks. Any
   verifier timestep that reconstructs a switch now returns a zero prefix and
   schedules teacher full inference.
4. Probe defaults were changed from historical WAM `{150,300}` to the official
   Realtime-VLA levels `{50,100}`, equivalent to sigma `{0.05,0.10}` for the
   action scheduler with shift one.
5. Clean LingBot imported flash-attention even for `attn_mode=torch`. The
   optional import now degrades to `None`; selecting torch remains unchanged,
   while selecting an unavailable flash-attention backend still fails at use.
6. The first launcher used a raw TCP readiness probe, which produced a false
   WebSocket handshake error. Readiness now follows the server's explicit log
   marker and does not touch the socket.
7. The first TN=10 run copied every block's live KV cache for each parallel
   verifier call. Cache size grew with long episodes until three tasks OOMed
   near 79 GB. Verification now microbatches K probes and uses an explicit
   read-only attention path that never writes or duplicates persistent cache
   tensors. Episode reset also releases unused CUDA allocator blocks. The
   launcher can put rendering on a separate GPU.
8. The first memory-fix smoke exposed the expected mixed-CFG shape: video CFG
   initializes two cache rows while action guidance one produces one query
   row. The read-only path now selects row zero exactly as the removed compact
   cache did; guidance-enabled action verification selects rows `(0, 1)`.

## Risk Assessment

Residual quality risk is medium; runtime risk is low after the repaired
long-episode smoke and 30 completed post-fix trials.

- The new read-only attention path is limited to action verification and must
  be checked for numerical equivalence and exact cache preservation in the
  configured torch-attention model.
- LingBot temporal cache updates are queued after draft actions and replayed
  before full inference. Shape/order errors must fail the smoke before any
  benchmark is launched.
- Prefixes are quantized to 16 actions because that is the model's temporal
  observation boundary; this is the sole structural deviation from pi0.
- A combined server/client reached the A800 limit in long episodes. Formal
  completion uses separate server and render GPUs even after removing the
  temporary KV copy, and records both assignments.

## Checks

- `py_compile` passed for policy, client, server entrypoint, launcher, and
  configs.
- Eleven fake-model state-machine tests passed with a direct function runner.
- Remote post-fix suite: `15 passed`, including exact equality of every KV
  cache tensor before and after read-only attention.

## Decision

Allowed to proceed. Remote tests passed `15/15`, the 1500-step long smoke and
30 post-fix trials completed without OOM/cache errors, and the merged quality
gate passed at `22/40` versus draft `21/40` and teacher `23/40`. Do not claim
uniform per-task improvement: `hanging_mug` remained `0/10`, and teacher action
use is still high at `40.81%`.
# Speed Iteration V1 Review

## Scope

- Raise the experiment refresh ceiling from PF=2 to PF=10 through the existing
  launcher argument.
- Add an opt-out for reconstructed teacher-latent gripper hard fallback while
  retaining its diagnostic log and the decoded draft gripper boundary.
- Preserve the original defaults for baseline reproduction.

## Findings

No blocking correctness finding in the diff.

- **Low risk:** `teacher_gripper_fallback=True` remains the default, so existing
  commands and the reproduced Realtime-VLA-FLASH behavior are unchanged.
- **Low risk:** diagnostic-only mode changes policy behavior only when the
  continuous endpoint verifier accepted the chunk and a reconstructed latent
  gripper crossed zero. The decoded action still passes through
  `first_gripper_switch()` and retains the hard phase fallback.
- **Low risk:** cache replay, frame indices, action slicing, K, tau, delta, and
  same-observation replan are untouched.
- **Experiment risk:** disabling the latent gripper fallback may admit a bad
  discrete transition. The four-task matched run must therefore meet the
  existing quality gate before this behavior is retained.

## Verification

- `python3 -m compileall`: passed.
- `git diff --check`: passed.
- Remote `pytest -q tests/test_realtime_flash_policy.py tests/test_specverify.py`:
  15 passed.
- The full read-only-cache test requires the serving environment with
  `diffusers`; it was not runnable from the host's default Python. The change
  does not touch the attention/cache implementation.

## Decision

Allowed to proceed to one real smoke with `--pf-interval 10` and
`--no-teacher-gripper-fallback`. A matched four-task evaluation is allowed only
if the smoke has no cache, reset, render, or action-shape failure.

## Calibration Telemetry Review

No blocking finding. The change serializes values already returned by the
teacher verifier and writes them only through the existing policy metrics log.
It does not feed telemetry back into prefix acceptance or runtime state.

- Risk: each flash record gains 64 distance scalars for K=2/F=2/N=16. This is
  small for RoboTwin evaluation and bounded by the fixed action horizon.
- `np.asarray(...).tolist()` makes both NumPy arrays and Python lists JSON-safe.
- Missing telemetry remains represented as `null`, preserving compatibility
  with fake or alternate verifier adapters.

Verification: compileall and diff check passed; remote policy/specverify suite
passed `16/16`, including exact preservation of the accepted prefix.

## Diagnostic Gripper Prefix Fix Review

The first V1 run proved that the original opt-out was incomplete: the server
returned `accepted_prefix=0` together with
`accepted_prefix_before_gripper=32`. The policy now selects the latter only
when `teacher_gripper_fallback=False`; force mode remains unchanged.

- The continuous endpoint result is still quantized by the existing helper.
- The decoded action still passes through `first_gripper_switch()` afterward,
  so a real executable gripper boundary remains a hard fallback.
- The force-mode and diagnostic-mode tests use the same server-shaped response
  and prove opposite expected decisions.

Remote policy/specverify suite passed `16/16`. The invalid V1 partial run is
marked STOP and must not be reported as benchmark evidence.

## V2 Cumulative Flow-Error Refresh Review

No blocking correctness finding remains after review.

- Medium experiment risk: cumulative endpoint discrepancy is a stale-cache
  proxy, not a semantic task-progress certificate. Low residual false accepts
  remain possible, so the matched quality gate is mandatory.
- A telemetry bias was fixed during review: cached/conditioned historical
  frames would inflate gripper agreement. Agreement now uses only future,
  unconditioned frames.
- Existing behavior is unchanged when `flow_budget_threshold=0`. The diff does
  not add repair, adaptive probes, or a new gripper routing decision.

Verification:

- Remote policy/specverify suite: `17 passed`.
- Local `py_compile` and `git diff --check`: passed.

Decision: allowed to proceed to one RoboTwin smoke. Formal evaluation requires
nonzero budget telemetry, correct reset after a budget-triggered full round,
and no cache/reset/render error.
