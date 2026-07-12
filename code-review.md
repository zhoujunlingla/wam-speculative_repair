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

## Previous-Gripper State Parity Fix Review

The audit found a blocking migration gap: the reference Realtime-VLA-FLASH
tracks the last executed gripper phase, while the LingBot policy only inspected
adjacent steps inside each new chunk. This made step-zero phase changes
invisible. The partial V2 run is therefore invalid benchmark evidence.

The fix stages the final gripper value from the exact returned action prefix and
commits it only after the matching cache-update acknowledgement. Replans and
unexecuted tails cannot update the committed phase. The committed decoded phase
is converted to the teacher latent sign convention and sent through the
already-supported `previous_gripper` request field; the decoded gate uses the
same committed state.

No blocking finding remains. Reset, first-full, full/partial prefix, and cache
ordering follow the existing state machine. Remote policy/specverify tests pass
`18/18`; local compile and diff checks pass.

Decision: allowed to proceed to a real step-zero switch smoke. Consensus-based
gripper acceptance remains disabled.

## V2b Repeated-Budget Teacher Burst Review

No blocking correctness finding remains. The first budget crossing retains the
existing one-round refresh. From the configured crossing count onward, the
counter schedules exactly two consecutive full rounds including the triggering
round. Cache acknowledgement still occurs between them, so the second teacher
action conditions on the first teacher action rather than a stale snapshot.

The refresh count and burst state reset per episode, while ordinary full rounds
reset only the accumulated discrepancy. Defaults are zero, preserving all prior
commands. Endpoint acceptance, gripper routing, repair, and video paths are
unchanged.

Remote policy/specverify suite passes `19/19`; local compile and diff checks
pass. Allowed to run a two-task (`hanging_mug`, `open_microwave`) TN=5
comparison while the non-burst V2 run finishes.

## Gripper Full Window Review

No blocking finding. The change exposes the existing Realtime-VLA-FLASH phase
window parameter without altering its default. Window one still schedules only
the already-existing next teacher round; window two adds exactly one contiguous
teacher round after it. The counter advances only on a returned full action and
survives the required cache acknowledgement.

The new test covers a 16-step safe prefix followed by two teacher rounds.
Remote policy/specverify suite passes `20/20`; local compile and diff checks
pass. Allowed to run `hanging_mug` TN=5 with window two. No consensus acceptance
or repair is enabled.

## Cross-Tau Gripper Consensus Review

No blocking correctness finding remains after review.

- The server is the sole owner of gripper acceptance in consensus mode; the
  client no longer applies a second decoded-action truncation.
- Consensus cannot enlarge the continuous endpoint prefix. It only finds the
  first step where any teacher probe's discrete gripper phase differs from the
  draft and floors that bound to the existing frame boundary.
- A transition reproduced by both tau probes is executable. `K=1` is rejected
  because it is not cross-tau consensus.
- The mode is opt-in and defaults off, preserving all completed baselines.
- Existing `gripper_force_teacher` remains telemetry in consensus mode and no
  longer overrides the consensus prefix.
- A consensus-accepted decoded transition is logged as
  `decoded_gripper_switch_step`; it is not mislabeled as a fallback.
- A rejected suffix after a nonzero consensus prefix schedules the configured
  teacher phase window. This closes a smoke-discovered gap where the next round
  incorrectly resumed drafting immediately after a 16-step prefix.

Risks: exact phase equality can still be conservative around a one-step switch
boundary, and phase agreement does not certify contact success. This mode must
first pass a small matched `hanging_mug`/`place_can_basket` evaluation; it is not
yet evidence for removing teacher phase windows globally.

Verification:

- Remote policy/specverify suite: `25 passed`.
- Local `py_compile` and `git diff --check`: passed.

Decision: allowed to proceed to a two-task smoke with video-motion routing kept
disabled. Raw video-motion statistics remain shadow-only because their observed
AUC is approximately random (`0.52`-`0.58`).

## Progress Evidence Update Review

Documentation-only update. It records completed artifacts and explicitly marks
PF10-only routing, cumulative flow budget, and raw video-motion gating as failed
or unsupported paths. Numbers were read from completed run summaries; no policy
behavior or experiment command changed. Allowed to commit without an additional
runtime test beyond the already-passing `25` focused tests.

## Bounded Flow-Budget Burst Review

No blocking finding. A positive `flow_budget_burst_limit` caps only the number
of burst escalations per episode; every budget crossing still receives its
ordinary teacher refresh. Limit zero preserves the previous unlimited behavior,
and reset clears the burst-use counter. The focused state-machine test covers
three crossings and proves that only the second crossing receives one extra
teacher round when the limit is one.

Decision: allowed to test `.18`, burst-after-two, two rounds, limit one on
`hanging_mug`. It remains an empirical recovery policy, not a distribution-
preserving speculative-sampling guarantee.

Verification: remote focused suite `26 passed`; local compile and diff checks
passed.

## Delayed Video Prediction Error Shadow Review

No blocking correctness finding remains.

- Prediction and observation are compared only inside the draft server, using
  the same VAE normalization and camera packing.
- The policy marks comparison only for an actually executed draft prefix.
  Teacher actions, replans, resets, and abandoned drafts clear or ignore the
  pending prediction, preventing invalid labels.
- Frame count is inherited from the existing cache update, so 16/32 executed
  actions compare one/two latent frames respectively.
- The implementation adds no model forward, encode, decode, or latent network
  transfer. One two-frame GPU tensor is retained until the next cache update;
  only scalar reductions enter JSONL telemetry.
- Frame-id mismatch and missing prediction are hard errors rather than silently
  producing misleading research data.
- The signal is shadow-only and cannot influence acceptance or teacher use.

Risks: temporal alignment still requires a real RoboTwin smoke, and latent
error is not calibrated across tasks/cameras. It must not route until completed
episode traces support a conformal threshold at a matched fallback rate.

Verification: local compile/diff checks passed; remote focused suite
`28 passed`.

Decision: allowed to run one real shadow smoke after an experiment GPU becomes
free. No routing change is approved.

## TCP Server Readiness Review

No blocking finding. The previous launcher waited for one exact log message;
real delayed-shadow servers were already listening while their clients remained
blocked because that message was absent. The shared readiness helper now checks
both server liveness and localhost TCP connectivity. It uses only the standard
library, has a bounded one-second connect timeout, and preserves the overall
launcher timeout. Tests cover a live listener and an exited child process.

Decision: allowed to restart only the delayed-shadow runs that produced no
client/trial. Existing benchmark processes are untouched.

Verification: remote focused suite including launcher readiness tests
`30 passed`; local compile and diff checks passed.

## Draft Video Motion Shadow Review

No blocking finding. The statistic reduces the video latent already produced by
the draft; it adds no DiT/VAE forward, decode, model, or dependency. Only five
Python floats cross the local adapter boundary. Motion is non-negative channel
RMS, and top-k count is bounded to at least one patch.

The initial chunk compares a conditioned frame with a future frame, while later
chunks compare two predicted future frames. `frame_st_id` is already logged, so
calibration must stratify or exclude the initial case before routing. The signal
is shadow-only and cannot change current actions.

Remote policy/specverify suite passes `21/21`; local compile and diff checks
pass. Allowed to run matched shadow collection; not allowed to gate teacher use
until separation is demonstrated.
