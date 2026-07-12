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

This iteration does not add repair, risk prediction, adaptive K, world latent
signals, or endpoint correction. It may proceed only if unit tests preserve
the original default behavior and a real smoke has no cache/reset/runtime
error. Its matched four-task gate is success at least `22/40`, teacher
action-source at most `15%`, and no material increase in zero-prefix fallback.
If it passes, adaptive K and flow-error-budget refresh may be evaluated as the
next speed improvement.

## Verification Plan

- Unit-test shared-noise K verification and min-over-K prefix acceptance.
- Unit-test continuous-channel RMS and gripper exclusion.
- Unit-test first-full, periodic refresh, L=0 replan, and gripper fallback.
- Unit-test teacher anchor/pending-update transitions.
- Assert verifier calls preserve teacher cache state and `frame_st_id`.
- Run a long-episode smoke with server and renderer memory recorded; no
  per-probe temporary KV cache may remain allocated.
- Run direct draft and teacher smokes before speculative evaluation.
