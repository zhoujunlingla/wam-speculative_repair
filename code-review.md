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

## Motion-Baseline Gripper Phase-Snap Repair Review

### Scope

Add one opt-in repair path to the frozen Motion-on SpecVerify policy. The path
is entered only when the continuous endpoint verifier accepts the whole chunk
but cross-tau gripper consensus rejects it. A third same-noise tau probe forms a
2-of-3 phase vote; at most two discrete phase cells may be snapped. The result
must then pass the unchanged K=2 verifier with an independent Gaussian probe.

### Findings

No blocking static finding remains.

- The existing behavior is unchanged unless both `--gripper-consensus` and
  `--gripper-repair` are enabled.
- The repair cannot alter continuous action channels, conditioned history, or
  the unverified suffix. It rejects pulses and any candidate with more than one
  phase transition per gripper channel.
- The construction probe deliberately reuses primary noise, but execution is
  gated by a dedicated RNG and a fresh K=2 holdout. This avoids the prior
  same-question repair/reverify bias.
- A successful repair executes only one 16-action temporal unit and returns the
  teacher-verifier's decoded stitched candidate. Cache acknowledgement remains
  on the existing draft path and therefore corresponds to the action actually
  returned to the client.
- Failed or malformed repair probes fail closed into the pre-existing teacher
  fallback. The per-episode attempt budget bounds the added teacher forwards.

Residual experiment risk is medium. Agreement among flow probes certifies only
the discrete action phase, not successful contact. The low10x20 run must report
repair-attempt count, independent holdout acceptance, accepted-repair episode
success, teacher source rate, and latency; raw repair acceptance is not a
quality claim.

### Verification

- Local `py_compile`: passed before sync.
- `git diff --check`: passed.
- Remote policy/launcher/verifier suite: `44 passed`.
- Tests cover phase-only edits, pulse rejection, conditioned-frame offset,
  independent holdout noise, successful 16-step execution, and the opt-in
  configuration guard.

### Decision

Allowed to proceed to a real-model smoke. Do not launch the formal low10x20
comparison until the smoke records a repair attempt, preserves cache ordering,
and completes without reset, render, shape, or non-finite-value errors.

## Zero-Prefix Endpoint Repair And Holdout Hardening Review

### Findings First

The review found one blocking inherited acceptance bug and three experiment
validity issues. All are fixed in the reviewed diff.

1. In consensus mode, `--no-teacher-gripper-fallback` could make the policy
   read `accepted_prefix_before_gripper` and bypass the server's final gripper
   consensus prefix. Both primary acceptance and repair holdout now use the
   final server prefix whenever consensus is enabled. Repair holdouts always
   prefer final `accepted_prefix`, independent of the legacy fallback switch.
2. Shadow probes shared the executable repair RNG. A dedicated `shadow_rng`
   now guarantees that enabling counterfactual logging cannot change later
   repair candidates or actions.
3. The gripper helper implements a 2-of-3 vote and therefore requires exactly
   two primary probes plus one tiebreak probe. Invalid K is now rejected at
   configuration time instead of silently disabling repair.
4. Gripper repair budget is consumed only after the primary continuous prefix
   covers the full horizon. Telemetry now records eligibility, attempts,
   action-only repair forwards, holdout prefix, correction norm, and execution.

The zero-prefix path reuses the teacher flow endpoint already produced by the
primary verifier. It changes only the first aligned 16-action block of the 14
continuous latent channels, caps per-step RMS, freezes gripper/suffix values,
and executes only after a fresh K=2 holdout recovers at least 16 actions.
Malformed/non-finite candidates fail closed to the existing teacher replan.

Residual risk is medium: endpoint projection can improve verifier agreement
without improving contact semantics. The formal result must therefore report
accepted-repair episode success, not merely repair acceptance.

### Verification

- Local `py_compile`: passed.
- `git diff --check`: passed.
- Remote policy/launcher/verifier suite: `50 passed`.
- Added regression coverage for final-prefix gripper rejection under
  `--no-teacher-gripper-fallback`, K contract, shadow RNG isolation, accepted
  zero-prefix execution, failed holdout replan, trust-region bound, and cache
  source state.

### Decision

Allowed to proceed to real-model smoke with the frozen Motion-on controls. A
formal low10x20 run is allowed only after smoke evidence contains at least one
eligible repair or proves the path remains safely dormant; no benchmark claim
may combine pre-fix and post-fix consensus behavior.

## ACP Low10x20 Orchestration Review

No blocking finding remains. The new shell runner only shards the fixed ten
tasks and calls the existing one-task launcher. It pins the completed Motion-on
control values and adds the reviewed gripper/zero-prefix repair flags; model
configs, checkpoints, client protocol, seed, task config, and render environment
remain owned by existing code.

Safety checks: fresh run/result roots are mandatory, each GPU owns one server
and client at a unique port, failures propagate through the final exit status,
and unrelated processes are never inspected or terminated by the runner.
Summary merging reads only completed per-task JSON artifacts and includes both
success and repair/source telemetry.

Verification: local and remote `bash -n` passed; local `py_compile` and
`git diff --check` passed; remote combined suite passed `51/51`.

Decision: allowed to submit a one-task TN=1 pool smoke. The formal 8xA800
low10x20 job may start only after the smoke reaches rendering and returns a
valid summary with no cache, reset, action-shape, or non-finite error.

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

## Flow-Consistent Repair Final Review

### Scope

Add a complete repair layer to the frozen Motion-on SpecVerify policy:
teacher-flow midpoint/RK2 repair for pure continuous zero-prefix rejection,
bounded discrete phase-snap repair for gripper consensus, independent K=2
holdout verification, exact 16/32 prefix execution, task-outcome accounting,
and a guarded low10 x 20 launcher.

### Findings Resolved

1. The first review found that finite midpoint inputs could still produce a
   non-finite full-interval RK2 endpoint. The server now checks the midpoint,
   midpoint velocity, and final RK2 endpoint separately. Any non-finite value
   returns `flow_repair_eligible=false` with a rejection reason and preserves
   the original Teacher fallback; an invalid latent is never sent into the next
   model forward.
2. Gripper repair originally charged K holdout forwards before knowing whether
   the tie-break probe could construct a candidate. The probe now returns the
   actual construction and holdout counts on every early-exit path. Summary
   acceptance is `executed / holdout_evaluated`, not
   `executed / attempted`.
3. Gripper holdout semantics now match the design: an independently verified
   prefix of 16 executes 16, a prefix of 32 executes 32, and zero falls back to
   the Teacher. A partial phase rejection schedules the existing Teacher phase
   window rather than clearing it.
4. Independent repair holdouts explicitly set `flow_repair=false`; a caller
   cannot accidentally nest construction inside the unchanged verifier. A
   regression test passes an inherited true value and verifies it is cleared.
5. Repair telemetry separates primary verifier, candidate construction, and
   holdout action-only forwards. Model-only reporting sums action/replan and
   cache model time, reports executed low-level actions, and derives both
   action-path and end-to-end model Hz.
6. Episode repair success is joined only when reset ids and visualization
   episode ids are an exact match. The formal merge requires all ten tasks,
   exactly TN=20 each, zero client errors, valid artifacts, and identical
   policy parameters; incomplete artifacts are written with
   `complete=false` and the launcher exits nonzero.
7. The formal launcher rejects any `TEST_NUM` other than 20, any checkout whose
   `HEAD` differs from `CODE_COMMIT`, and any dirty tracked or untracked file.
   Provenance is therefore checked before GPU processes or result roots exist.
8. The formal launcher requires an explicit reviewed `CODE_COMMIT`, preventing
   a dirty worktree from being mislabeled as its old HEAD. The experiment is
   explicitly labeled as the combined
   `motion-on+gripper-phase-snap+zero-prefix-flow-rk2` policy, so it is not
   falsely attributed to FCR alone.

### Residual Risk

Research risk remains medium. Flow-consistent action repair can improve
Teacher-manifold agreement without proving contact success, and discrete phase
agreement is not a force/contact sensor. Promotion therefore depends on final
episode success and Teacher usage, not repair acceptance alone. The dedicated
repair tau=150 costs two construction forwards; measured model Hz must include
them.

### Verification

- Local `py_compile`, `bash -n`, and `git diff --check`: passed.
- Formal-launch negative gates reject TN other than 20, a mismatched commit, and
  a dirty checkout before creating run roots: passed.
- A800 focused suite: `59 passed` in `1.51s`.
- Tests cover RK2 endpoint semantics, continuous-axis caps, gripper/suffix
  preservation, independent holdout RNG, exact 16/32 execution, gripper early
  exit accounting, non-finite fail-closed policy behavior, episode alignment,
  repaired-episode success, and model-time aggregation.
- A first real-model smoke reached `Render Well`, executed a three-probe
  gripper repair, and continued cache updates without OOM or state mutation.
  It was started before the final accounting-only fixes and is diagnostic, not
  benchmark evidence. A final reviewed-commit smoke remains required before
  low10 x 20.
- The unchanged full read-only-cache suite still requires the configured model
  environment; default host Python cannot collect it because of the existing
  `diffusers`/`bitsandbytes`/`triton` dependency conflict.

### Decision

Allowed to create the reviewed commit and run one final real-model smoke from
that exact code. Formal low10 x 20 is allowed only after the final smoke has a
valid artifact, no cache/reset/render error, and at least one repair path is
observed or a dedicated server-level FCR probe succeeds.

## 2026-07-15 FlowGuard Near-Miss Shadow Review

### Findings

No blocking correctness finding remains after review.

1. **Fixed, high experiment-validity risk:** the initial implementation used
   the complete K=2 prefix when labeling the first probe. Classification now
   derives its continuous prefix only from `distances[0]`; a regression test
   proves that a K1 pass is still labeled as such when K2 rejects it.
2. **Fixed, high safety risk:** a continuous K1 pass could originally hide a
   gripper-phase disagreement. The classifier now consumes only first-probe
   gripper agreement/switch telemetry and routes such rounds to `ambiguous`.
3. **Low runtime risk:** FlowGuard is shadow-only. It uses a dedicated RNG,
   never replaces the action, never changes cache/frame state, and is mutually
   exclusive with every executable repair path.
4. **Low comparison risk:** Probe A constructs the candidate. Probe B is
   independent of A, while the original and repaired candidates share the
   exact same B noise. This removes the same-exam circularity found in earlier
   repair runs.
5. **Medium research risk:** endpoint residual improvement is only evidence of
   local Teacher-flow agreement. It is not evidence of task success. The
   summary therefore labels avoided full rollouts as a projected upper bound
   and separately reports improved/equal/worsened Probe-B prefixes.
6. **Medium compute risk:** every eligible near miss costs four extra
   action-only forwards for the paired K=2 Probe B. A repair is not promotable
   unless projected full-Teacher savings exceed this cost in measured model
   time.

### Verification

- Local `py_compile`: passed.
- Local `git diff --check`: passed.
- A800 focused suite:
  `pytest -q tests/test_realtime_flash_policy.py tests/test_run_realtime_flash_task.py tests/test_specverify.py`:
  `65 passed in 2.89s`.
- Tests cover first-probe-only classification, K1/K2 disagreement telemetry,
  phase-risk ambiguity, bounded decaying repair windows, independent A/B
  noise, paired-B noise equality, unchanged replan behavior, and summary
  aggregation.

### Decision

Allowed to run a Motion-on matched shadow audit. Executable FlowGuard repair
is blocked until valid paired-B rescue is at least 20%, no candidate worsens
the original paired-B prefix, and the projected savings remain positive after
counting all extra action-only forwards. WCAS V0 and FlowGuard must be reported
separately: WCAS can reduce verifier NFE without changing `R_full`/`R_anyT`;
FlowGuard can reduce `R_full` only after a later closed-loop execution test.
