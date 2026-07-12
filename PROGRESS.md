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
