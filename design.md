# Realtime-VLA-FLASH Migration For LingBot-VA

## Goal

Start from clean `robbyant/lingbot-va` commit `7c6ffa9` and reproduce the
complete Realtime-VLA-FLASH inference state machine with:

- draft: FlashWAM official step3000, video 1 / action 2;
- teacher: LingBot-VA posttrain, video 2 / action 4;
- one policy server containing both models;
- first-round full inference, K-point endpoint verification, longest safe
  prefix, gripper-aware fallback, and periodic full refresh.

The first experiment contains no repair, risk router, world verifier, or
Verify++ signal. The verifier is useful only if matched four-task success is
better than direct draft and close to direct teacher.

## Runtime State Machine

1. Reset both model caches and all speculative state.
2. Run teacher full path when there is no reference cache, a fallback is
   pending, or the periodic refresh interval has expired.
3. Otherwise generate one draft chunk from the current observation.
4. Verify the draft with the teacher action branch at all configured flow
   timesteps in one batch, using one shared Gaussian probe.
5. Execute the longest prefix that passes every timestep. A zero prefix
   executes nothing and requests the same observation again; that request runs
   the teacher full path.
6. Treat gripper switches as hard phase boundaries: truncate before the first
   switch and force the next round to use the teacher.

## WAM Cache Adaptation

Pi0 reuses a static last-full VLM KV snapshot. LingBot has an ordered streaming
video/action cache, so skipped real updates cannot be discarded.

- Draft consumes every real observation/action cache update.
- The cache update following a teacher full action is applied immediately to
  the teacher and becomes the new reference anchor.
- Cache updates following accepted draft actions are queued, leaving the
  teacher verifier on the last-full reference.
- Before the next teacher full action, queued updates are replayed in order.
- Action-only verification must not mutate cache contents or `frame_st_id`.

### A800 memory-safe verification

Copying every live KV tensor into a K-batched temporary cache grows with long
episodes and exceeds 80 GB when the policy server and RoboTwin renderer share a
GPU. The A800 path therefore evaluates the same K probes serially against the
teacher's existing cache. A read-only attention path selects the same CFG cache
rows as the action branch and concatenates live cached keys/values with the
probe query without allocating cache slots, so all cache tensors and
`frame_st_id` stay unchanged. Shared noise, probe levels, distance reduction,
and prefix acceptance are identical; only K-way execution is microbatched. The
launcher may place rendering clients on a separate GPU without changing the
episode or policy configuration.

## Verifier

For shared noise `eps` and each sigma `s_k`:

```text
z_k = (1 - s_k) * A_draft + s_k * eps
v_k = teacher_action_flow(z_k, s_k | last_full_cache, current_state)
A_hat_k = z_k - s_k * v_k
```

The step distance is normalized L2 over the 14 continuous dual-arm pose and
quaternion channels. Gripper channels 28 and 29 are excluded and checked by
the phase gate. A step passes only if all K distances are below `delta`.

The official initial probe levels `t={0.05,0.10}` map to LingBot scheduler
timesteps `{50,100}` when the action shift is one.

LingBot actions are tied to temporal latent frames. The verifier computes a
raw per-action prefix, then floors it to the nearest 16-action boundary:
`{0, 16, 32}`.

## Evaluation Contract

Run the same task/episode manifest for direct draft, direct teacher, and the
speculative policy on:

- `hanging_mug`
- `turn_switch`
- `place_can_basket`
- `open_microwave`

Each task has 10 completed trials. Record per-task and aggregate success,
draft/full action-source rates, verifier prefix histogram, fallback reasons,
model-only action latency, cache latency, and errors.

The first go gate is:

- speculative success is higher than matched direct draft;
- speculative success is within two successes of matched direct teacher over
  40 trials;
- no cache mutation during verification and no reset/render/runtime error.

Threshold and periodic-refresh tuning may begin only after the implementation
passes unit tests and a real episode smoke. Repair remains locked until this
gate passes.

Launcher readiness is determined by a successful localhost TCP connection and
a live server process. Log text is diagnostic only; model/server logging may be
buffered or configured differently and cannot be the synchronization primitive.

## Speed Iteration V1: Remove Unproductive Full Paths

The matched four-task run passed the quality gate at `22/40`, versus draft
`21/40` and teacher `23/40`, but used teacher actions in `40.81%` of policy
rounds. The full-path reasons show two dominant costs that are not endpoint
verification failures:

- `periodic`: 204 calls under `PF=2`;
- `teacher_gripper_switch`: 102 calls caused by any reconstructed latent
  gripper crossing zero at either verifier timestep.

Only 14 calls came from a zero endpoint prefix. The first speed iteration
therefore keeps the verified-prefix algorithm, `K=2`, tau values, threshold,
cache replay, decoded-action gripper boundary, and same-observation fallback
unchanged, while testing two evidence-backed policy changes together:

1. increase the hard refresh ceiling from `PF=2` to `PF=10`;
2. retain reconstructed teacher gripper switches as diagnostics rather than a
   hard fallback, while keeping the decoded draft gripper phase boundary as a
   hard fallback.

The first V1 launch exposed a call-boundary bug: the teacher server had already
zeroed `accepted_prefix` before returning `gripper_force_teacher`, so ignoring
only the boolean in the policy did not make the signal diagnostic-only. In
diagnostic mode the policy must consume `accepted_prefix_before_gripper`, which
is the continuous endpoint-verifier result, and then independently apply the
decoded draft gripper boundary. The invalid partial run is marked STOP and is
not benchmark evidence.

This iteration does not add repair, risk prediction, adaptive K, world latent
signals, or endpoint correction. It may proceed only if unit tests preserve
the original default behavior and a real smoke has no cache/reset/runtime
error. Its matched four-task gate is success at least `22/40`, teacher
action-source at most `15%`, and no material increase in zero-prefix fallback.
If it passes, adaptive K and flow-error-budget refresh may be evaluated as the
next speed improvement.

## Quality Iteration V2: Cumulative Flow-Error Refresh

V1 was stopped after `14/26` trials: increasing PF from 2 to 10 reduced the
teacher action-source rate to about 29%, but `hanging_mug=0/5` and
`open_microwave=0/3` showed that isolated refreshes every ten accepted draft
rounds were too sparse. Per-episode telemetry also showed that failures can
have small instantaneous endpoint residuals while running for many rounds.

V2 replaces the productive part of periodic refresh with a cumulative flow
discrepancy budget. For every executed draft prefix, compute the mean over the
executed steps of the maximum normalized endpoint distance across verifier
timesteps:

```text
q_r = mean_{j < L_r} max_k d_{k,j}
B_{r+1} = B_r + q_r
```

When `B` reaches a configured threshold, the next action round uses the full
teacher and resets `B` to zero. A large PF ceiling may remain as a safety net,
but it is not the primary refresh schedule. This is the smallest WAM analogue
of bounded-difference adaptive diffusion: repeated low residuals are allowed,
but their accumulated stale-cache risk is not treated as zero.

V2 does not change gripper routing yet. It only logs, for each verifier probe,
the reconstructed gripper switch index and whether the probe-wise discrete
phase sequences agree with the draft. That diagnostic is required before a
later cross-tau gripper-consensus gate can replace the current blanket decoded
gripper fallback.

The V2 four-task gate is success at least `22/40`, no new runtime/cache error,
and fewer teacher actions than the PF2 V0 (`40.81%`). Repair, adaptive K, and
world-latent signals remain locked.

### Required phase-state parity fix

The first V2 launch exposed a missing piece of the Realtime-VLA-FLASH state
machine: the reference implementation advances `last_gripper` using the action
steps actually executed by the client, then includes that phase when checking
the next chunk. The LingBot migration only compared adjacent steps inside the
new chunk. It therefore could not detect a switch at action step zero.

The policy must stage the last decoded gripper values when returning an action,
commit them only after the corresponding cache-update acknowledgement, and
pass their discrete phase to the teacher verifier. Both latent and decoded
phase checks must compare the first proposed step against that committed phase.
Reset clears it; unexecuted tails and rejected/replanned drafts never update it.

The partial V2 run before this fix is marked invalid. No consensus-based phase
acceptance may be enabled until this state parity fix passes unit tests and a
real smoke.

### V2b: Escalate repeated budget refreshes

Valid episode telemetry shows that one flow-budget refresh is often sufficient
for a successful episode, while stuck episodes repeatedly cross the same budget
two to seven times. Isolated teacher chunks do not reliably return those
trajectories to the teacher manifold. V2b therefore leaves the first budget
refresh unchanged, but on the second and later crossings executes a configurable
two-round teacher burst. The burst is counted in full action rounds, survives
the intervening cache acknowledgement, and resets only at episode reset.

This is a quality recovery mechanism for repeatedly detected drift, not a new
risk score. It does not alter endpoint acceptance, gripper fallback, repair, or
video signals. The first test is restricted to `hanging_mug` and
`open_microwave`; it must improve completed-trial success before replacing V2.

Repeated bursting on every later crossing raises teacher use without rescuing
long failed episodes. A bounded variant may cap bursts per episode: the first
crossing remains a single teacher round, one later crossing may schedule a
two-round recovery burst, and subsequent crossings return to single-round
refresh. A zero limit preserves the existing unlimited behavior.

### Phase window parity

Realtime-VLA-FLASH exposes `gripper_full_window`; the initial LingBot migration
implemented only its default one-round behavior. `hanging_mug` remains poor even
when repeated flow-budget refreshes are added, indicating that refresh after
drift is too late for a precision contact transition. The next matched hanging
test therefore changes only `gripper_full_window` from one to two: a detected
decoded or teacher gripper boundary schedules two consecutive teacher action
rounds. The default remains one and reproduces the existing path.

## WAM Shadow Signal: Draft Video Motion Concentration

The two-round phase window can recover `hanging_mug`, but early trials use the
teacher for roughly 70% of action rounds. Before routing on video, expose a
zero-extra-forward statistic from the future video latent that FlashWAM already
generates. For adjacent latent frames, compute the per-patch channel RMS motion,
then log its global mean, top-10% mean, top/median ratio, and top-10%
concentration. No latent is copied to the client and no image is decoded.

This first version is shadow-only. It must demonstrate separation between
precision-contact gripper rounds and ordinary gripper rounds before it may
select `gripper_full_window=2`. It does not change action acceptance, budget,
teacher use, or repair.

## WAM Shadow Signal: Delayed Video Prediction Error

Raw motion measures how much the imagined video changes, not whether that
future is correct. The next WAM-specific shadow signal compares an executed
draft's future video latent with the real observation latent produced by the
draft server's next mandatory cache update:

```text
r_t = distance(predicted_video_latent[t], encode(real_observation[t]))
```

Both tensors stay in the same draft VAE normalization space. A 16-action
prefix compares one predicted frame and a 32-action prefix compares two. A
rejected draft, replan, or teacher action must clear/ignore the pending
prediction. The cache update already performs the VAE encode, so the shadow
adds no DiT, VAE encode, decode, or network transfer; it stores at most one
two-frame latent and returns scalar reductions only.

The first version logs RMSE, normalized RMSE, cosine distance, per-frame RMSE,
and top-patch residual. It cannot route teacher use until completed episodes
show that it predicts failure better than raw motion at a matched fallback
rate.

### Delayed-error recovery pilot

The four-task shadow run completed 12 episodes. Episode-level latent NRMSE
median/max both reached failure AUC 0.857 and maximum cosine distance reached
0.914, while raw intra-video motion had previously remained near chance. A
single large residual is not sufficient: successful `place_can_basket`
episodes contain isolated spikes. The first routing experiment therefore uses
only a persistent error condition:

```text
if latent_nrmse > 0.55 for two consecutive executed draft cache updates:
    run the teacher for two consecutive action rounds
```

Any low-error update resets the streak. A teacher action or episode reset also
resets it, so stale evidence cannot cross a new teacher anchor. If a flow-budget
refresh is already scheduled, that teacher round counts as the first round of
the two-round recovery window rather than stacking a third round. The feature
is disabled when its threshold is zero and must remain separately configurable
from the flow budget.

This pilot does not alter endpoint verification, gripper consensus, draft
actions, cache tensors, or repair. It is evaluated with a less aggressive
0.18 flow budget so the new signal replaces blind refreshes instead of merely
adding teacher calls. The go gate is better `hanging_mug` success than the
0/3 shadow result, no regression on the other three tasks, and lower teacher
action-source rate than 36.36%.

The pilot reached `8/12` and reduced teacher use to 28.77%, but all six delayed
triggers occurred in failed episodes and none rescued the episode. Delayed
error is therefore retained as a supervision/calibration label, not promoted
as a post-hoc recovery controller.

### Pre-execution video-motion risk gate

Pairing each executed draft with its next delayed-error label produced 268
samples. The draft future-video global motion available before execution
predicted `latent_nrmse > 0.55` with AUC 0.867, versus 0.73-0.76 for endpoint
residual features. A conservative global-motion threshold of 1.2 selected
18/268 draft rounds (6.7%), and 15/18 selected rounds produced high delayed
error. This relationship also held within `hanging_mug`, `place_can_basket`,
and `open_microwave`; `turn_switch` had one false-positive selected round.

The next pilot uses this statistic before teacher action verification:

```text
draft action + future video latent
if global_video_motion >= 1.2:
    discard the unexecuted draft and replan the same observation with teacher
else:
    run the unchanged K=2 action verifier
```

This is an immediate safety gate, not repair: no draft action or cache update
is committed before the teacher replan. It skips the now-unnecessary action
verifier on gated rounds. The first pilot disables flow-budget and delayed-
error routing while retaining delayed-error shadow telemetry, cross-tau
gripper consensus, and the PF=20 ceiling. This tests whether a sparse WAM-
specific pre-execution gate can replace blind teacher refreshes. The gate is
disabled by threshold zero.

The motion-only pilot reached `8/12` with 14.57% teacher actions. It improved
`hanging_mug` from 0/3 to 2/3 and preserved 3/3 `place_can_basket`, but
`open_microwave` fell from 3/3 under the 0.18 flow-refresh policy to 1/3.
Global motion and cumulative endpoint drift detect complementary failures.
The next configuration therefore combines the unchanged motion threshold 1.2
with a 0.18 single-round flow refresh. Delayed recovery and flow bursts remain
off. This is a configuration composition of two already-tested signals; it
requires no code change.

That ungated composition regressed to `6/12` and raised teacher use to 22.73%.
Twenty-three blind flow refreshes interfered with successful high-motion
trajectories without improving `open_microwave`. Flow discrepancy must only
accumulate inside a persistent low-motion regime.

Offline replay of the motion-only run resets the flow budget whenever global
motion exceeds 0.5. With threshold 0.4, it predicts zero refreshes in both
successful `hanging_mug` episodes, one in the failed hanging episode, two to
four per `open_microwave` episode, and zero in all `place_can_basket` and
`turn_switch` episodes. The next policy therefore uses:

```text
if global_video_motion > 0.5:
    flow_budget = 0
else:
    flow_budget += executed_action_flow_residual
    if flow_budget >= 0.4:
        schedule one teacher refresh
```

The immediate motion gate at 1.2 remains unchanged. The low-motion ceiling is
disabled at zero and does not change the existing flow-budget default.

The low-motion pilot was early-stopped at `2/6`; it could no longer beat the
motion-only `8/12` even under a perfect remaining outcome. It is not promoted.
Because TN=3 task outcomes have varied substantially across identical controls,
the next decision uses four tasks x 10 trials with the motion-only policy. No
additional router or verifier change is stacked into that run.

## Cross-Tau Gripper Consensus

The original migration applies two independent phase fallbacks: the server
rejects any teacher reconstruction containing a gripper transition, then the
client truncates any decoded draft transition again. This rejects transitions
even when the draft and every teacher probe agree on the discrete phase.

An opt-in consensus mode makes the server the single owner of gripper
acceptance. Within the continuous endpoint prefix, every draft gripper phase
must equal the reconstructed phase at every configured tau. The first phase
disagreement bounds the prefix and is floored to the normal 16-action frame
boundary. A phase transition that all probes reproduce is allowed. In this
mode the client logs decoded transitions but does not apply a second fallback.
When disagreement leaves a nonzero safe prefix, the client executes that
prefix and schedules the configured teacher phase window for the next round;
it must not resume drafting immediately from the rejected suffix boundary.

The mode requires at least two tau probes, remains disabled by default, and
cannot enlarge the continuous endpoint prefix. It is tested before any video
signal is allowed to route teacher phase windows.

### Verifier calibration telemetry

The V1 smoke showed that removing periodic and reconstructed-gripper full paths
does not by itself bound teacher use: contact-heavy rounds can still produce a
continuous endpoint prefix of zero. Before changing `delta` or `K`, every flash
round must log the existing verifier distances, per-tau prefixes, and tau values.
This is diagnostic-only and must not change acceptance. The resulting traces
support exact offline replay of candidate thresholds and identify whether the
first or second tau is responsible for each rejected prefix.

## Verification Plan

- Unit-test shared-noise K verification and min-over-K prefix acceptance.
- Unit-test continuous-channel RMS and gripper exclusion.
- Unit-test first-full, periodic refresh, L=0 replan, and gripper fallback.
- Unit-test teacher anchor/pending-update transitions.
- Assert verifier calls preserve teacher cache state and `frame_st_id`.
- Run a long-episode smoke with server and renderer memory recorded; no
  per-probe temporary KV cache may remain allocated.
- Run direct draft and teacher smokes before speculative evaluation.

## Adaptive-K Paired Validation

This branch is cut directly from the frozen Motion-on commit `46f0c38`. Its
only policy experiment is an adaptive-K certificate around the existing
`tau={50,100}` verifier. It does not carry repair, alter the acceptance
threshold, change gripper routing, change motion routing, or address false
accepts.

The experiment has three gated phases:

1. Build and freeze a manifest containing 20 valid `hanging_mug` scenes and
   20 valid `open_microwave` scenes. Each row records the accepted RoboTwin
   seed, episode metadata digest, and selected instruction. Replay must fail
   closed if any recorded scene or instruction cannot be reproduced.
2. Run adaptive-K `off` and `shadow` from the same commit and manifest with
   the same client, verifier, NumPy, and Torch random streams. Shadow always
   executes the original two probes. Every policy round records the executed
   action hash, accepted prefix, fallback reason, and cache-state fingerprint.
   The pair may proceed only when all four fields match for every round and
   `certificate_conflict == 0`.
3. Enable `live` only after phase 2 passes. Live may skip the second teacher
   probe only when the first-probe certificate proves that the unchanged K=2
   result is already decided. If end-to-end model latency improves by less
   than 5%, adaptive-K is deleted rather than maintained.

The manifest is the source of episode identity. A failed setup, changed
instruction, or digest mismatch is an experiment error; the client must not
silently increment the seed. Audit fingerprints are debug instrumentation and
are excluded from live latency measurement.

Success-rate recovery is explicitly out of scope. Adaptive-K reduces verifier
compute but cannot make a false-accepted draft correct. Any later work on
success belongs to a separate false-accept verifier branch.
