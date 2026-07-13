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

## Persistent Delayed-Error Recovery Review

No blocking correctness finding remains after an independent state-machine
review.

- Error persistence is updated only when acknowledging an actually executed
  draft cache update. Replans, rejected tails, and teacher actions cannot add a
  sample.
- A two-round recovery includes an already-scheduled flow refresh as its first
  round; the counters overlap rather than stacking a third teacher round.
- Every full teacher action and episode reset clears the streak, so evidence
  cannot leak across a new teacher anchor or episode.
- Threshold zero disables routing and preserves the prior behavior. All three
  launcher parameters reach the policy and are saved in the run summary.
- Tests cover persistent triggering, low-error streak reset, and overlap with
  an already-scheduled flow refresh.

Residual risks: overlapping trigger telemetry reports only the highest-priority
`full_reason`, although the preceding cache-update record retains the delayed
trigger; the real server is trusted to provide finite `latent_nrmse`. The first
pilot must remain small because the threshold was selected from only 12 shadow
episodes.

Verification: remote policy/launcher/specverify suite `33 passed`; local
`py_compile` and `git diff --check` passed. The optional read-only-cache test
cannot collect under the system Python because `diffusers` is absent; that path
was unchanged and had passed in its configured model environment previously.

Decision: allowed to proceed to a four-task TN=3 pilot with delayed threshold
0.55, two consecutive high-error updates, two teacher recovery rounds, and a
0.18 single flow refresh. Repair and adaptive K remain disabled.

## Pre-Execution Video-Motion Gate Review

No blocking correctness finding remains after independent review.

- The gate runs after draft video/action generation but before teacher action
  verification. A gated draft is never executed and never receives a cache
  acknowledgement.
- It does not stage gripper state, add a teacher replay update, increment flash
  age, or charge the flow budget. The existing same-observation replan protocol
  runs teacher full after replaying only previously executed draft updates.
- The real draft server clears the gated pending video prediction on the full
  teacher action's cache acknowledgement without comparing it, so no false
  delayed-error label crosses the new teacher anchor.
- Threshold zero preserves K=2 verification. CLI forwarding and run-summary
  recording are complete.

Residual risks: the fake model does not directly model the GPU pending-video
tensor, and callers must continue using the existing `infer_with_replan`
protocol. These are covered by the unchanged real-server clear path and client
wrapper, but must be observed in the real smoke.

Verification: remote policy/launcher/specverify suite `35 passed`; local
`py_compile` and `git diff --check` passed.

Decision: allowed to run a four-task TN=3 pilot with motion threshold 1.2,
flow-budget disabled, delayed recovery disabled, and delayed-error telemetry,
K=2 endpoint verification, PF=20, and gripper consensus retained.

## Hybrid Motion/Flow Configuration Review

Configuration-only change; no source behavior changed. It composes the tested
pre-execution motion gate with the existing tested 0.18 single flow refresh.
Delayed recovery and flow bursts remain disabled, so the two teacher triggers
cannot create an unreviewed multi-round counter interaction. Existing K=2
verification, PF20, and gripper consensus are unchanged.

Decision: allowed to run the same four-task TN=3 pilot. Compare both success
and teacher action-source against motion-only (`8/12`, 14.57%) and delayed-
route (`8/12`, 28.77%).

## Motion-Conditioned Flow Budget Review

The first independent review found one P1 transactional bug: invalid or missing
motion telemetry was checked after `pending_cache_source="flash"` and flash age
had been committed, so an exception could leave the policy waiting for a cache
acknowledgement for an action never returned. Motion telemetry validation was
moved before every policy-state mutation. A regression test now proves that a
NaN raises without changing round id, pending source, or flash age, and that the
same request can be retried after valid telemetry is restored.

After the fix, no blocking finding remains:

- low motion retains and accumulates the raw verifier charge;
- motion above the ceiling clears historical budget and charges zero;
- the immediate motion gate still runs first and teacher full resets all budget;
- ceiling zero preserves the previous cumulative-budget behavior;
- raw/effective charge and reset decisions are logged separately;
- CLI forwarding and run-summary recording are complete.

Verification: remote focused suite `37 passed`; local `py_compile` and
`git diff --check` passed.

Decision: allowed to run four-task TN=3 with immediate motion threshold 1.2,
flow threshold 0.4, low-motion ceiling 0.5, no delayed recovery/burst, PF20,
K=2 endpoint verification, and gripper consensus.

## Motion-Score E0-E3 Review

No blocking correctness finding remains for audit or shadow telemetry.

- Pairing is tied to the policy state machine: only an executed positive draft
  prefix can open a pair, and only its same-round cache acknowledgement can
  close it. Frame-count alignment prevents a delayed label from being assigned
  to the wrong action horizon.
- The analysis never randomly splits rounds. It reports task-wise rankings,
  task-balanced matched-budget metrics, and thresholds calibrated with one
  whole task held out.
- Existing `global_mean`, `median`, and top-patch fields are unchanged. The new
  RoboTwin score is emitted only as telemetry and has no policy caller, CLI
  flag, threshold, or action-side effect.
- The T-shaped split matches the encoder layout: wrist cameras occupy the top
  third split across width and the head camera occupies the lower two thirds.
  Each region is normalized by its own latent RMS before robust median and
  entropy statistics are computed.

Residual risks: region-wise RMS normalization can amplify a nearly empty
camera region, and `max(region)` may be sensitive to one noisy camera. These
are acceptable only in shadow mode; E3 must reject the score if its matched
precision, AUPRC, or task-held-out behavior is worse than `global_mean`.
Current historical logs do not contain motion-v2, so they cannot be used to
claim an improvement.

Verification: `git diff --check` and Python compilation passed locally. The
A800 focused suite passed `41/41`, covering audit pairing, frame mismatch,
matched trigger budgets, regional motion telemetry, verifier behavior, and
policy state transitions. Replaying the frozen logs produced 276/276 aligned
pairs and 20 explicit unexecuted exclusions.

Decision: E0/E1 evidence may be recorded; motion-v2 may proceed to E3 shadow
collection only. It is not allowed to route or alter teacher usage. E2 must
compare motion off versus `global_mean >= 1.2` on one matched low10 x 20
manifest before any causal benefit is claimed.

## Repair Counterfactual Shadow Review

Scope: shadow-only bounded endpoint repair for pure continuous zero-prefix
rejections. It must not alter the executed action, fallback decision, primary
verifier RNG, or cache state.

No blocking correctness finding remains after finite-value validation was
added. The mean endpoint can overfit the primary probe, so it is evaluated only
with an independent Gaussian holdout from a dedicated RNG. Continuous endpoint
agreement cannot certify contact; motion, gripper, phase, and refresh fallbacks
are therefore ineligible, and repair remains non-executable.

Verification: local `py_compile` and `git diff --check` passed. The A800 focused
suite passed `39/39`, covering continuous-prefix-only modification, exact
gripper/suffix preservation, RMS clipping, independent noise, unchanged
replan, and absence of pending cache state. A real-model smoke reached
`Render Well` without OOM or cache mutation.

Decision: allowed to run a four-task TN=5 counterfactual audit. Do not enable
execution unless independent holdout prefix recovery has useful episode-level
predictive value.

## Motion-Conditioned Selective Verification Review

Scope: replace the high-video-motion hard Teacher fallback with the existing
two-probe action verifier. A verified high-motion chunk may execute at most one
16-action observation interval; a zero prefix still replans the same
observation with Teacher. Normal-motion verification, initial Teacher anchor,
PF=20, gripper consensus, cache replay, and repair behavior are unchanged.

The independent review initially found three contract gaps, all fixed before
benchmark launch:

1. Selective mode now requires exactly two tau probes; it cannot silently run
   an uncalibrated K=1 or K>2 policy.
2. The high-motion cap is fixed to `action_per_frame`, rather than exposed as
   another experiment hyperparameter.
3. Telemetry separates the raw continuous verifier prefix, the gripper-adjusted
   prefix before the motion cap, and the final executed prefix. Run summaries
   aggregate K counts, high-motion proposals, accepts, rejects, and cap events.

The proposed low-motion K=1 optimization is explicitly rejected. Frozen logs
contain 1306 low-motion rounds that passed tau 50 without a decoded gripper
switch; 49 (3.75%) were rejected by tau 100 or cross-tau phase consensus, and
`hanging_mug` missed 18/130 (13.85%). Reducing K would therefore trade away
task quality instead of safely reducing Teacher use.

Verification:

- Remote focused suite: `44 passed`.
- Local Python compilation and `git diff --check`: passed.
- Real-model smoke reached `Render Well`, completed cache updates, and entered
  repeated K=2 action verification without shape, reset, or cache-state errors.
  Formal success/latency evidence still comes from the matched low10 x 20 run.

Risk is medium and experimental rather than structural. Motion magnitude does
not certify correctness; the verifier remains the sole accept/reject signal,
and the 16-step cap limits high-motion open-loop exposure. Promotion requires
success statistically compatible with the 73.0% hard-gate baseline, Teacher
action-source below 21.63%, and lower model-forward latency.

Decision: allowed to proceed to one matched low10 x 20 run with repair disabled.

## Motion/verifier-margin telemetry review

Scope: diagnostic-only logging and offline pairing support for a possible
region-motion plus verifier-margin cap-release rule. No live routing rule is
added.

The first review found one blocking parser edge case: an empty distance tensor
could divide by zero rather than remain visible as malformed telemetry. The
parser now validates a non-empty list, finite positive threshold, finite
distances, and at least 32 action entries per tau before deriving the tail
margin. Missing, malformed, and short telemetry use separate audit counters and
never remove an otherwise valid motion/delayed-error pair.

No blocking finding remains:

- the policy adds one scalar log field and does not alter inference control
  flow, RNG, cache state, or response data;
- legacy logs use a threshold only when the caller explicitly supplies
  `--verify-threshold`;
- the derived tail maximum is the maximum over actions 16:32 across every tau,
  matching the verifier's `[K, 2, 16]` temporal order;
- the analysis keeps task/run/episode identity and does not introduce a random
  split or an online motion-v2 threshold;
- missing or malformed verifier data cannot be misclassified as a strong
  margin.

Verification: an isolated copy on A800 passed `40/40` focused pytest cases.
Local `py_compile` and `git diff --check` passed. Partial replay of 426 aligned
B1 pairs completed with 427 valid margin records and one still-pending cache
acknowledgement.

Residual risk: delayed error after a 16-action cap cannot establish safety of
the unexecuted second 16 actions. Offline margin analysis may nominate a rule,
but a matched closed-loop low10 x 20 run remains mandatory.

Decision: telemetry and offline analysis are approved. Motion-v2 remains
shadow-only and no cap-release policy is approved yet.
# Motion Gate V3 Shadow Telemetry Review (2026-07-14)

## Findings

No blocking correctness finding after review.

- Risk: low. The server adds only reductions over the video latent already
  produced by draft inference. It does not change action tensors, RNG, cache
  contents, verifier inputs, or policy decisions.
- The saliency value is deliberately a channel-variance proxy. It is not
  treated as an object, contact, or physical-importance label.
- History innovation is computed offline only after an executed draft receives
  its aligned cache acknowledgement. Reset, replan, Teacher execution, missing
  delayed labels, and frame-alignment failures break the chain.
- Legacy logs remain valid: missing V3 saliency is counted explicitly and does
  not remove an otherwise valid delayed-error pair.

## Verification

- Isolated A800 review root:
  `/mnt/afs/intern/manlichen/ivan/zhoujunl/tmp/motion_v3_shadow_review`
- `/usr/bin/python -m pytest -q tests/test_specverify.py tests/test_analyze_motion_score.py`
  -> `15 passed`
- Local `python3 -m py_compile` passed for implementation and tests.
- `git diff --check` passed.

Decision: allowed to proceed as shadow telemetry only. Online routing remains
locked until whole-task holdout calibration and a matched Low10 x 20 run pass.

## Model-Only Latency Profiling Review (2026-07-14)

### Findings

No blocking correctness finding remains after review.

- The profiler is opt-in. With `profile_model_time` disabled, no CUDA event,
  synchronization, response telemetry, or routing behavior is added.
- Timings cover only model-side VAE encode, video/action DiT generation,
  teacher action verification, and video/action cache-transformer forwards.
  RPC, client, rendering, environment stepping, JSON, and queue time are not
  included in `model_timing_ms`.
- Speculative aggregation retains failed draft/replan cost, teacher cache
  replay, repair holdout verification, initial draft priming, and teacher
  generation. Role-prefixed keys prevent draft, verifier, and cache work from
  being conflated.
- The initial Teacher response reports 16 executed low-level actions rather
  than the generated 32 because the RoboTwin client skips the conditioned
  first frame. Later full chunks report 32. This closes the original action-Hz
  overcount.
- VAE camera inputs are transferred and encoded sequentially. The temporary
  high-camera CUDA tensor is explicitly released before wrist-camera transfer,
  preserving the default peak-memory behavior.

### Residual Risk

- CUDA-event timing and action Hz still require an exclusive-GPU real-model
  smoke. Shared-GPU results are diagnostic only and cannot support a benchmark
  claim.
- The direct-only adapter is covered by the same response schema but not by a
  dedicated unit test; the exclusive-GPU smoke must verify first-round 16 and
  later-round 32 executed-action accounting.

### Verification

- Local `git diff --check`: passed.
- Local `python3 -m py_compile`: passed for implementation and focused tests.
- Isolated A800 review root:
  `/mnt/afs/intern/manlichen/ivan/zhoujunl/tmp/model_profile_review`.
- `/usr/bin/python -m pytest -q tests/test_realtime_flash_policy.py tests/test_run_realtime_flash_task.py tests/test_specverify.py tests/test_analyze_motion_score.py`
  -> `54 passed in 1.35s`.

Decision: allowed to proceed to an exclusive-GPU speed smoke. It is not yet
approved as formal v1/a2, v2/a4, or speculative throughput evidence.

## Motion Gate V3 Budget Evidence Review (2026-07-14)

### Findings

No code finding. This documentation change records a read-only replay of the
completed hard-gate low10 x 20 logs: 305 gripper disagreements split into 224
failures before action 16 and 81 at or after action 16.

Risk is low because no runtime behavior, threshold, model call, or experiment
command changes. The projected Teacher savings are explicitly fixed-denominator
estimates and cannot be reported as an achieved online rate. The design keeps
motion from overriding a continuous zero prefix and treats `tau=75` only as a
conditional phase tie-break.

### Verification

- Inspected `git diff -- design.md`.
- Replayed all ten `specverify_*.jsonl` files under
  `20260713_130446_rtflash_gripfallback_low10_tn20`: 305 total, 224 early,
  81 late. Every early row is a zero-prefix replan and every late row executes
  a 16-action draft prefix.

Decision: allowed to proceed to shadow design only. No online gripper policy
is approved from this documentation evidence alone.

## Conditional Gripper Tie-Break Shadow Review (2026-07-14)

### Findings

No blocking correctness finding remains after review.

- The feature is opt-in and requires existing K=2 gripper consensus.  With the
  flag disabled, the server does not return phase tensors and no extra Teacher
  forward is issued.
- The `tau=75` request reuses the exact primary action latent, frame id,
  previous gripper state, and Gaussian noise.  Its only changed verifier input
  is the flow timestep and `gripper_consensus=False`, which is required for a
  single probe.
- The 2-of-3 vote is performed per gripper channel and per action step over the
  first 16 actions.  Counterfactual rescue additionally requires both the
  primary and tie-break continuous prefixes to cover all 16 actions.
- Shadow telemetry is merged into model-only profiling, but the live accepted
  prefix, replan reason, Teacher scheduling, cache state, and RNG state are not
  changed.

### Residual Risk

- This probe measures cross-timestep phase stability, not true contact or
  task success.  It may be promoted only after complete task-held-out shadow
  coverage and matched closed-loop validation.
- Shadow runs include the extra conditional forward and cannot be used as the
  final speculative speed number.  Formal speed must be measured with the
  promoted policy or with the shadow cost reported separately.

### Verification

- Local `python3 -m py_compile` and `git diff --check`: passed.
- Isolated A800 review root:
  `/mnt/afs/intern/manlichen/ivan/zhoujunl/tmp/gripper_tiebreak_shadow_review`.
- `/usr/bin/python -m pytest -q tests/test_realtime_flash_policy.py tests/test_run_realtime_flash_task.py tests/test_specverify.py tests/test_analyze_motion_score.py`
  -> `58 passed in 1.43s`.

Decision: allowed for shadow telemetry only.  Online tie-break execution is not
approved.

## Model-only profiler evidence review (2026-07-14)

### Findings

No runtime code changed in this review.  The exclusive-GPU smoke completed and
showed `draft=36.79`, `Teacher=36.05`, and `speculative=13.36` model-only
action Hz.  The speculative result contradicts any claim that reducing only
full-Teacher action rounds is sufficient for draft-like speed: unconditional
K=2 verification, duplicate VAE/cache work, and extra short-prefix rounds must
also be addressed.

The TN=1 trajectories executed different action counts, so their aggregate Hz
is diagnostic rather than a final matched throughput claim.  Component timing
and forward counts are authoritative for locating the bottleneck; final speed
still requires a fixed-workload or completed matched benchmark.

### Decision

Motion Gate V3 may proceed to telemetry-only shadow collection.  Online routing
is not approved until the incomplete MCSV-B1 benchmark is closed and the
shadow gates in `design.md` pass.
