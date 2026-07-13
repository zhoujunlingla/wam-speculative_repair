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

### Low-motion budget early stop

- The conditioned budget was stopped after `2/6` valid outcomes
  (`hanging_mug 1/2`, `place_can_basket 1/3`, `open_microwave 0/1`). Even six
  remaining successes could only tie the promoted motion-only `8/12`, so the
  branch had no possible improvement outcome.
- The feature remains default-off for future analysis, but it is not promoted.
  Four-task TN=10 now evaluates the motion-only policy before any further
  routing or adaptive-K change.

### Motion-score E0/E1 audit

- The frozen pairing rule now includes only a `draft_flash` action with a
  positive executed prefix whose same-round `flash_cache_update` contains a
  delayed-error label and exactly `ceil(prefix / 16)` acknowledged frames.
  Replans and unexecuted proposals are excluded.
- Replaying the two source runs produces 276 valid pairs, not 268: 276 executed
  proposals all have aligned acknowledgements, 20 unexecuted/replan proposals
  are excluded, and no executed proposal is missing its cache update. The old
  268 count is not reproducible and is retired as a bookkeeping error.
- With `latent_nrmse > 0.55` as the frozen label, there are 55 positives. At
  the existing 18/276 (6.52%) trigger budget, `global_mean` reaches AUPRC
  0.632, precision 15/18 = 83.3%, recall 27.3%; `median` reaches AUPRC 0.656,
  precision 14/18 = 77.8%, recall 25.5%.
- Task-balanced selection reduces both scores to 10/18 = 55.6% precision, and
  leave-one-task-out thresholds produce different trigger rates across tasks.
  The apparent aggregate separation is therefore task-confounded. The current
  `global_mean >= 1.2` remains the causal E2 baseline, but the offline audit
  alone does not prove that motion routing improves success.
- Equal-weight macro task AUPRC is 0.495 for `global_mean` and 0.509 for
  `median`. This small ranking gain does not offset median's lower matched-rate
  precision and is not enough to replace the deployed baseline.
- `motion-v2` is now shadow-only telemetry. It splits the RoboTwin T-shaped
  latent into head/left-wrist/right-wrist regions and scores the largest
  region-wise product of normalized median patch motion and spatial entropy.
  It cannot affect routing until complete low10 telemetry beats global mean at
  the same trigger budget.

### Motion/verifier-margin audit instrumentation

- The running MCSV-B1 benchmark is unchanged. Offline audit support now retains
  the logged verifier threshold, per-tau prefix, distance tensor, pre-cap
  prefix, and gripper-consensus status for each action-aligned delayed-error
  pair. It derives the maximum K=2 residual over actions 16:32 and its margin
  from `delta`; malformed or missing telemetry remains an explicit audit count.
- Legacy logs require an explicit `--verify-threshold` argument. The analysis
  never guesses a threshold that was not recorded in the log.
- A partial audit over 426 aligned B1 pairs found eight high-global-motion
  proposals whose continuous/gripper prefix was 32 before the motion cap. None
  had `tail_max <= 0.05`; their tail residuals ranged from 0.0847 to 0.1190.
  This is incomplete, task-imbalanced evidence, but it already shows that a
  margin-only cap release at 0.05 has zero expected coverage. Region-aware
  telemetry is still required before any cap-release policy is considered.
- The local policy change only logs `verify_threshold`; it does not change
  routing, model calls, RNG, cache state, or the accepted prefix. The active
  remote B1 process was not modified or restarted.

### Motion Gate V3 shadow telemetry

- Added a saliency-weighted regional motion statistic using channel variance
  from the draft future video latent. It adds no VAE, DiT, decode, or network
  transfer and remains absent from online routing.
- Offline audit derives history innovation only across consecutive executed
  draft proposals with aligned cache acknowledgements. Teacher actions,
  replans, resets, malformed labels, and frame mismatches break the chain.
- The active MCSV-B1 run remains unchanged. This telemetry is intended for the
  next shadow collection after B1 completes; it cannot retroactively appear in
  the current logs.
- Isolated A800 verification passed `15/15` focused tests. Local Python
  compilation and `git diff --check` also passed.

Verification: isolated A800 review copy passed `40/40` focused tests; local
Python compilation and `git diff --check` passed. Formal routing conclusions
remain locked until B1 low10 x 20 completes.

### Model-only profiling for gate and repair comparisons

The existing policy `elapsed_sec` is retained as wall-time telemetry but is no
longer treated as model-only speed. An opt-in profiler now measures completed
CUDA work for VAE encoding, video/action generation DiT forwards, Teacher
action-only verification, and video/action KV-cache transformer forwards.
Speculative logs preserve failed draft/verify cost, Teacher cache replay, and
executed low-level action counts so summaries can report actions per model
second without RPC, rendering, or RoboTwin stepping.

The profiler is disabled by default and therefore does not alter the active
MCSV-B1 run. Validation in the isolated remote review copy
`/mnt/afs/intern/manlichen/ivan/zhoujunl/tmp/model_profile_review` passed:

```text
/usr/bin/python -m pytest -q \
  tests/test_realtime_flash_policy.py \
  tests/test_run_realtime_flash_task.py \
  tests/test_specverify.py \
  tests/test_analyze_motion_score.py
54 passed
```

An exclusive-GPU speed-only smoke is still required before using the metric in
a benchmark claim.

### Exclusive model-only profiler and selective-motion diagnostic (2026-07-14)

- The exclusive GPU4 `turn_switch` TN=1 profiler completed for all three
  modes.  Model-only throughput, including VAE, video/action DiT, verification,
  and KV-cache transformer work, was `36.79 action Hz` for draft v1/a2,
  `36.05 action Hz` for Teacher v2/a4, and `13.36 action Hz` for the current
  speculative policy.  These are smoke measurements, not success estimates.
- The speculative path spent `5987.9 ms` for 80 executed actions.  Duplicate
  draft/Teacher VAE and cache work plus six Teacher action-verifier forwards
  dominated the gap.  Lowering only the Teacher action-source rate cannot make
  this implementation approach draft speed while every draft round still pays
  K=2 verification and short prefixes increase the number of rounds.
- The running MCSV-B1 selective-motion benchmark is not complete and remains
  excluded from final claims.  Its completed tasks are `turn_switch 12/20`
  versus hard-gate `14/20`, and `open_microwave 11/20` versus `9/20`.
  The valid recovery shard for `hanging_mug` is currently `2/17`, versus the
  hard-gate result `8/20`.
- In the current hanging shard, 31 high-motion proposals were shortened from
  a verified 32-action prefix to 16 actions.  The shorter closed-loop horizon
  was followed by more gripper-consensus and zero-prefix events.  Therefore
  `high motion -> always cap 16` is not approved for promotion even if it lowers
  Teacher action-source.
- GPT-5.5 technical review approved only the next shadow collection.  It did
  not approve online early-gripper rescue, K=1, or unconditional 32-action cap
  release.  Same-noise tau=75 is treated only as a cross-timestep phase
  stability probe, and fixed-denominator Teacher savings remain estimates.

Next: collect Motion Gate V3 regional/saliency/history telemetry and tau=75
gripper tie-break telemetry without changing routing, RNG, caches, or Teacher
scheduling.  Online routing requires whole-task held-out improvement and an
estimated early-conflict rescue lower bound of at least `35/224`.

### Invalid V3 shadow launch: phase payload shape (2026-07-14)

The first V3 shadow launch produced no valid completed trial. Its first
conditional gripper probe failed because `VA_Server.verify_action_chunk`
serialized internal `[K,2,F,N,1]` phase tensors unchanged, while the policy
contract is `[K,2,F,N]`. The run is invalid benchmark evidence. The root fix
removes the latent-only singleton at the server response boundary; the policy
keeps strict validation so the same interface regression cannot pass silently.

### Selective-motion B1 stopped (2026-07-14)

The valid B1 `hanging_mug` shard completed `2/20`, versus `8/20` for the
hard-motion baseline. Completed B1 tasks were `turn_switch 12/20` and
`open_microwave 11/20`; they do not offset the contact-task regression. The
32-to-16 high-motion cap also changes the horizon used by delayed-error labels,
so B1 is unsuitable for fitting Motion Gate V3. The remaining B1 queue is
stopped and retained as negative evidence. The next collection leaves motion
fully shadow-only and preserves the action verifier's prefix.

### Motion V4 late-gripper deferral code gate (2026-07-14)

- Added an opt-in policy that executes only the consensus-safe prefix when a
  gripper disagreement begins at or after action 16, then reobserves without
  scheduling a Teacher phase window.
- Code review caught and fixed a consensus bypass where disabling the legacy
  Teacher gripper fallback could restore `accepted_prefix_before_gripper=32`
  after the server had capped the prefix at 16.
- The feature now requires gripper consensus at configuration time and also
  caps the executed prefix at the quantized failure boundary.
- Corrected `728/3366 = 21.63%` to Teacher-full rounds per action-producing
  round; action-step rate remains a separate summary metric.

Verification: local diff/compile passed; isolated A800 focused tests passed
`65/65`. No active shadow process was changed. Live evaluation remains gated on
the complete uncensored Motion V3 analysis.

### Adaptive-K second-probe audit (2026-07-14)

- Added a separate audit over every standard `tau={50,100}` K=2 proposal,
  including zero-prefix replans. The label is whether tau100 quantizes to a
  shorter prefix or introduces a gripper disagreement absent at tau50; delayed
  video NRMSE and episode success are not used as labels.
- Code review fixed three optimistic failure modes before using the result:
  proposal-level independence, folds with no risky training examples, and
  treating gripper agreement as equivalent to no phase transition. Safety is
  now clustered by episode with an exact one-sided Clopper-Pearson bound, empty
  evidence abstains, and the K=1 candidate requires no draft/tau50 phase switch.
- A partial read-only audit over the completed/active uncensored shards found
  `1082` K=2 proposals. Only `185` passed the strict tau50 certificate, and
  tau100 restricted one of those. The frozen `global_mean` leave-one-task-out
  rule selected `105/185 = 56.8%` with zero observed proposal misses, but those
  covered only 34 episodes; the exact episode-level upper bound is still
  `8.43%`. This is useful compute-saving signal but is not enough evidence to
  enable adaptive K.
- Region-aware scores did not improve this partial adaptive-K target:
  `motion_v2` covered `48.1%` and `motion_v3` `42.2%`, both with weaker average
  precision than `global_mean`. They remain exploratory shadow telemetry.

Verification: isolated A800 review copy passed `70/70` focused tests; local
Python compilation and `git diff --check` passed. No online routing or active
experiment was changed.
