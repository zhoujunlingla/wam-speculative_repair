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
# Adaptive-K Paired Shadow Review (2026-07-16)

## Decision

Allowed to proceed to manifest and TN=1 off/shadow smoke only. Live adaptive-K
is not implemented and is not approved by this review.

## Findings

- **P1 fixed:** RoboTwin override keys were initially lost when the client
  reloaded `demo_clean.yml`. `scene_manifest_in/out`, `manifest_only`, and
  `paired_rng` are now explicitly copied into the task args.
- **P1 fixed:** the paired comparator initially accepted empty/unhashed traces.
  It now rejects empty traces, missing cache hashes, missing executed-action
  hashes, and traces without action decisions.
- **P1 fixed:** the first cache audit hashed only mutation requests. Audit mode
  now also reads active draft and teacher KV cache tensors and hashes per-layer
  size, ID, prediction mask, sum, squared sum, and absolute sum. The request
  chain remains as an independent logical-cache check.
- **P1 fixed:** fail-certificate comparison initially ignored second-probe
  gripper routing. A fail match now also requires identical reconstructed
  switch presence and gripper-consensus failure index.
- **P2 fixed:** ordinary runs would have acquired a fixed verifier seed. The
  runner now passes `--seed` only when explicitly requested; paired episode
  streams are controlled by the manifest seed on reset.
- **P2 fixed:** existing nonempty policy traces are rejected instead of
  appended, preventing contaminated paired results.

## Residual Risk

- The K1 pass condition is an empirical candidate certificate, not a theorem:
  tau-50 below 0.05 does not mathematically bound tau-100 below 0.15. Zero
  conflict on the frozen manifest is required by the requested gate, but a
  later live implementation must retain deterministic full-K audits and must
  not claim distributional equivalence outside the audited scenes.
- The KV checksum is a compact numerical fingerprint rather than a bytewise
  copy of every cache tensor. It reads actual cache state and is suitable for
  detecting this verifier-mutation regression without transferring the full
  multi-layer cache to CPU.
- Manifest replay intentionally restores the recorded prompt instead of
  rerunning expert planning. Scene actor/articulation state, RoboTwin commit,
  tracked diff, episode metadata digest, and manifest file digest are all
  validated fail closed.

## Checks

- `python3 -m py_compile` passed for all changed runtime, script, and test files.
- `git diff --check` passed.
- Focused policy tests for episode RNG replay and shadow forwarding passed via
  direct Python invocation.
- Synthetic paired trace and adaptive summary checks passed.
- Local `pytest` is unavailable; the full focused pytest suite must pass in the
  A800 runtime before any RoboTwin smoke.

## Adaptive-K Host-Only Corrective Review (2026-07-16)

### Findings

- **P1 fixed:** shadow-only teacher-server tensor work changed later cache and
  action hashes despite leaving the immediate K=2 decision unchanged. Adaptive
  candidate logic now runs in the policy after the unchanged K=2 response and
  performs only NumPy/host operations. The obsolete adaptive branch and its
  request parameters were deleted from the teacher server, making accidental
  shadow CUDA execution impossible through this API.
- **P1 fixed:** the previous fail certificate compared an internal failure
  index instead of the final policy decision. Fail candidates are removed; the
  remaining pass candidate predicts the exact source, full action hash, prefix,
  and fallback reason.
- **P1 fixed:** candidate matching previously happened before gripper fallback
  fields were written. Matching now uses the completed response, so a gripper
  fallback cannot be reported as a pass match.
- **P1 fixed:** malformed certificate rows could be ignored by the comparator.
  Every declared certificate must carry a strict boolean match.
- **P2 fixed:** zero-conflict shadow runs with zero coverage could pass. The
  comparator now also requires at least one declared shadow certificate.
- **P1 fixed after independent review:** incoming action requests could retain
  stale adaptive fields. The policy now strips both fields before constructing
  the teacher request, in addition to the server-side branch deletion.
- **P1 fixed after independent review:** malformed host telemetry could raise
  only in shadow mode. Candidate parsing now abstains on nonintegral K values,
  nonfinite/nonintegral prefixes, malformed arrays, reordered tau values,
  threshold mismatch, or missing shared-noise semantics.
- **P1 fixed after independent review:** the comparator accepted unknown/fail
  kinds and orphan match values. Its only legal schemas are now `(None, None)`
  and `("pass", bool)`, and any off-side certificate invalidates the run.
- **P2 fixed after independent review:** a nominal K=2 response did not prove it
  was the configured probe pair. Candidates now require exact K counters, tau
  values and order, verifier threshold, and shared-noise flag.

### Residual Risk

- The pass condition remains an empirical candidate, not a proof about the
  entire flow trajectory. Live mode is prohibited until both the repeated TN=1
  and formal paired TN=20 traces are exactly equivalent with zero conflicts.
- Host-only telemetry must remain derived from the unchanged K=2 response. No
  adaptive flag may be forwarded into the teacher server during shadow runs.
- The compact cache fingerprint is not bytewise, but off and shadow execute the
  same audit operations and it already detected the original isolation failure.

### Checks and Decision

- Local `python3 -m py_compile` and `git diff --check` pass.
- Direct policy checks cover request stripping, strict candidate parsing, and
  final policy-log match/conflict behavior; a synthetic comparator smoke covers
  identical off/shadow decisions with nonzero pass coverage.
- Local pytest is unavailable and must be run in the A800 environment.
- Decision: code may be pushed for remote tests and the same-manifest TN=1
  rerun only. Formal TN=20 and live mode are not yet approved.

## Deterministic Audit Profile Review (2026-07-16)

### Evidence and Scope

The same current commit, GPU2, manifest, and seeds produced an off/off split at
the identical cache/action indices as off/shadow. The deterministic profile is
therefore an experiment-validity repair, not an adaptive-K behavior change.

### Review

- The profile is activated only when the existing `--equivalence-audit` flag is
  set. Ordinary training, evaluation, future live routing, and latency runs do
  not receive `CUBLAS_WORKSPACE_CONFIG` or math-SDPA forcing.
- The cuBLAS environment is set by the parent runner before the server process
  imports torch. Torch deterministic algorithms, cuDNN determinism, TF32
  disablement, and SDPA backend selection are then applied before model load.
- Parent-shell cuBLAS settings are removed from every child environment; only
  the audited server receives the fixed value. Direct audit-server launches
  without that value fail closed before model construction.
- The client process is unchanged, so the profile cannot alter RoboTwin physics
  or rendering. Off/off must still pass; otherwise exact cross-process closed-
  loop validation is not feasible and request replay is required.

### Risks and Gate

- Math SDPA is slower and may use more memory. These numbers are invalid for
  the requested live 5% speed gate.
- A deterministic-algorithm error is fail-closed and blocks the experiment; no
  fallback to a nondeterministic backend is allowed.
- Decision: allow remote focused tests and TN=1 off/off only. Off/shadow, TN=20,
  live implementation, and latency claims remain blocked.
