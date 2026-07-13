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
- First formal run root:
  `20260712_115916_rtflash_full_k2d015_pf2_4task_tn10_g2467`.
  `turn_switch` completed at `9/10`; `hanging_mug` stopped at `1/2`,
  `place_can_basket` at `4/5`, and `open_microwave` at `1/1`. The incomplete
  tasks all hit the same OOM in `_readonly_verify_cache` after the temporary
  K-batched KV copy grew with episode length. These partial scores are not a
  valid four-task result.
- Memory repair removes the temporary cache, microbatches the unchanged K
  probes through a cache-preserving attention path, clears unused allocator
  blocks on reset, and allows clients to render on a separate GPU. Next gate:
  one long smoke, then rerun the three tasks from 10 completed trials rather
  than merging partial episodes. Post-fix remote suite: `15 passed`, including
  exact equality of all cache tensors across read-only attention.
- First post-fix smoke stopped before a valid episode because the teacher's
  video-CFG cache has batch two while action guidance one has batch one. The
  removed compact cache selected positive row zero; the read-only path now
  carries the same explicit CFG-row mapping. No experiment score was produced
  from this failed smoke.
- The corrected long `open_microwave` smoke ran all 1500 environment steps and
  48 policy rounds without OOM or cache error. Server and renderer were split
  across GPUs 7 and 0.
- Final four-task TN=10 result: `22/40 = 55.0%`, versus the step3000 draft
  reference `21/40 = 52.5%` and LingBot v2/a4 teacher reference
  `23/40 = 57.5%`. Per task: `hanging_mug 0/10`, `turn_switch 9/10`,
  `place_can_basket 9/10`, `open_microwave 4/10`. The predefined quality gate
  passed: +1 success over draft and one success below teacher.
- Aggregate teacher action-source was `362/887 = 40.81%`; draft prefixes were
  503 full 32-step and 22 partial 16-step. Full reasons were 204 periodic, 102
  teacher gripper switch, 14 zero prefix, 40 initial, and 2 draft gripper
  switch. Quality is established, but reducing teacher use remains future work.
- Final artifact:
  `/mnt/afs/intern/manlichen/ivan/zhoujunl/result/Wam_Speed_up/20260712_rtflash_full_clean4task_tn10_k2d015_pf2_final`.

### Prior Failure Carried Forward

V31 achieved only `41/80 = 51.25%` at delta 0.15 versus the historical draft
reference `25/40 = 62.50%`. It was not a faithful FLASH implementation: the
first round executed draft after a shadow prime, low-risk chunks bypassed
verification, teacher cache was synchronized every round, phase handling only
tightened a threshold, and the verifier used MAE including gripper channels.
Do not compare the new implementation to V31 as if only the threshold changed.

## 2026-07-13 - Speed/Quality Iteration Evidence

- PF10 plus diagnostic-only teacher gripper checks was stopped at `14/26`:
  `hanging_mug 0/5` and `open_microwave 0/3` showed that sparse periodic
  refresh alone loses contact-stage quality.
- The corrected cumulative-flow-budget run completed at `20/40 = 50.0%` with
  about `31.7%` teacher actions. It was worse than both the matched draft
  (`21/40`) and the faithful PF2 migration (`22/40`), so cumulative residual
  budget is not retained as the primary router.
- Escalating repeated budget crossings to two teacher rounds improved the first
  five `open_microwave` trials from `2/5` to `5/5`, but did not rescue
  `hanging_mug`. This is task-dependent recovery, not a general verifier.
- A two-round hard gripper window reached `hanging_mug 3/5 = 60%`, but required
  `62.5%` teacher actions. It is useful evidence that contact transitions need
  contiguous teacher control, but its blanket cost violates the speed target.
- Draft future-video latent motion shadow completed at `6/12`. Its
  `global_mean`, `top_relative`, and `top_concentration` signals overlap heavily
  between phase-window and ordinary rounds (AUC about `0.52`-`0.58`). Raw motion
  statistics must remain telemetry and must not gate teacher use.
- Offline replay of 682 K=2 verifier calls found that when the tau-50 maximum
  residual is below `0.05`, skipping tau-100 would cover about `65%` of calls
  and changed zero observed `{0,16,32}` prefixes. This supports a later audited
  adaptive-K path after phase routing passes its quality gate.
- Cross-tau gripper consensus was added so a discrete transition is executable
  only when every teacher probe reproduces the draft phase. The first smoke
  found and fixed a state-machine gap: a 16-step consensus prefix must schedule
  teacher takeover rather than resume drafting at the rejected suffix.

Current experiments compare two forms on four matched tasks: consensus with
the old flow budget, and consensus with no flow budget plus a two-round teacher
window only on phase disagreement. Repair remains locked until one form matches
teacher quality with materially lower teacher use.

### Delayed-error shadow result

- The completed four-task TN=3 shadow run scored `7/12 = 58.3%`:
  `hanging_mug 0/3`, `turn_switch 2/3`, `place_can_basket 3/3`, and
  `open_microwave 2/3`.
- It used teacher actions in `80/220 = 36.36%` of action rounds. Draft/teacher
  action latency p50 was `0.390s/1.190s`; draft/teacher cache latency p50 was
  `0.374s/0.757s`.
- Delayed draft-video error separated the seven successful and five failed
  episodes better than raw motion: latent-NRMSE median/max failure AUC was
  `0.857`, and maximum cosine-distance AUC was `0.914`. This is promising but
  still task-confounded and too small for a universal threshold claim.
- Isolated high errors occur in successful episodes. Two consecutive NRMSE
  values above `0.55`, with persistence reset at every teacher anchor, produced
  no trigger in the seven successful episodes and triggered in all three failed
  `hanging_mug` episodes. The next controlled pilot combines this trigger with
  a less aggressive `0.18` flow budget.
- A bounded `.18` flow-budget pilot was early-stopped at `hanging_mug 0/2`;
  it showed no recovery advantage and is not promoted on its own.

### Verifier compute replay

- Replaying 2,268 K=2 calls showed that neither tau probe ever produced a
  zero raw prefix, so first-failure early stopping has no useful opportunity at
  the current threshold and is not implemented.
- An audited adaptive-K guard remains promising: when tau-50 max residual is
  below `0.05` and neither the draft nor tau-50 reconstruction contains a
  gripper switch, K=1 would cover `55.6%` of eligible calls with zero observed
  tau-100 prefix or phase changes. This is the next speed change after the
  delayed-error routing pilot passes its quality gate.

### Persistent delayed recovery pilot

- The pilot scored `8/12 = 66.7%`: `hanging_mug 0/3`, `turn_switch 3/3`,
  `place_can_basket 2/3`, and `open_microwave 3/3`.
- Teacher action-source fell from 36.36% in the shadow policy to 28.77%, but
  all six delayed triggers occurred in episodes that ultimately failed and no
  trigger rescued an episode. The mechanism is rejected as a recovery policy.
- Pairing pre-execution telemetry with the following delayed latent error found
  that draft future-video global motion predicts `latent_nrmse > 0.55` at AUC
  0.867. A threshold of 1.2 selects 6.7% of draft rounds at 83.3% precision.
  The next pilot gates these drafts before execution and disables blind flow-
  budget refresh; delayed error remains shadow-only.

### Pre-execution motion gate pilot

- The motion-only policy scored `8/12 = 66.7%`: `hanging_mug 2/3`,
  `turn_switch 2/3`, `place_can_basket 3/3`, and `open_microwave 1/3`.
- Teacher action-source was `36/247 = 14.57%`, meeting the sub-15% routing
  target. Draft/teacher action latency p50 was `0.422s/1.123s`.
- Six high-motion drafts were rejected before execution. The task-level shift
  supports complementary failure modes: immediate video motion protects
  `hanging_mug`, while `open_microwave` needs low-motion cumulative refresh.
- The next no-code configuration combines motion threshold 1.2 with a 0.18
  single flow refresh. Delayed recovery and repeated flow bursts remain off.

### Ungated hybrid rejection

- Combining motion threshold 1.2 with an unconditional 0.18 flow budget
  regressed to `6/12 = 50.0%` and raised teacher action-source to 22.73%.
- Its 23 flow-budget refreshes did not improve `open_microwave` and reduced
  `hanging_mug`/`turn_switch` relative to motion-only. More teacher calls are
  not monotonically safer when refresh timing changes the closed-loop policy.
- Offline replay supports a regime-conditioned budget instead: reset whenever
  global motion exceeds 0.5 and use threshold 0.4 during sustained low motion.
