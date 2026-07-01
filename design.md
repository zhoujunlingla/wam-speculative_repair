# V6 World-Latent Flow Consistency Verifier

## Goal

Validate whether LingBot-VA's video/world latent flow can improve speculative action acceptance beyond action-only verification.

## Source

Copied from V5 `lingbot-va-riskrouter-lazyprefix-20260701` commit `209a180`.

## Problem Evidence

Action-only RiskRouter variants either overused teacher fallback or crashed when trying to make non-sync teacher cache modes too aggressive. More importantly, action-space agreement can miss world-state errors in contact-rich tasks: two action chunks may be numerically close while inducing different object/contact outcomes.

## Hypothesis

If a draft chunk is trustworthy, its one-step/few-step predicted video latent should also be consistent with the teacher's video flow field. Adding a teacher video-latent flow consistency gate after action verification should reject action chunks whose imagined world transition is not teacher-consistent.

## Single New Feature

Add optional `world_verify` gate:

1. Draft server returns both normalized action latent and predicted video latent.
2. Teacher first runs the existing action-flow verifier.
3. If action verifier accepts the full chunk, teacher runs a sparse video latent flow verifier on the draft video latent.
4. The draft action is accepted only if both action and world-latent flow checks pass.

The first validation uses `teacher_cache_mode=sync` to isolate verifier quality from lazy/stale cache correctness issues.

## Fixed Controls

- Draft: LingBot v1/a2 direct.
- Teacher/verifier: LingBot v2/a4 direct.
- Action verify tau: `[150, 300]`.
- Action threshold: `0.18`.
- Risk thresholds: `risk_low=0.25`, `risk_high=0.55`.
- Phase switch tightens action threshold by `0.5`.

## New Controls

- `--world-verify-enable`.
- `--world-verify-threshold`: default `0.35` normalized L1 latent residual.
- `--world-verify-tau`: default `[150, 300]`.

## Gate

Small validation should show no runtime crash and produce world verifier metrics. Continue only if:

- world verifier rejects some action-passing chunks instead of being constant pass/fail,
- low10 success does not collapse relative to action-only variants,
- teacher source rate remains interpretable.

If world gate is too strict, next single variable is threshold calibration or logging-only mode. If it is useful but slow, next variable is running world verify only for high-risk chunks.

## V6b Threshold Calibration: P95 World Score

The first V6 calibration run showed the world-latent verifier was functional but too strict when using raw max patch distance: `world_n=16`, `pass=1`, `reject=15`; median `world_distance_max ~= 0.56` but median `world_distance_p95 ~= 0.24`. V6b keeps the same threshold (`0.35`) and changes only the aggregation criterion from max patch distance to p95 patch distance. This follows video-generation caching practice where percentile/top-k scores are more stable than a single worst patch.

## V6c Reset/Initial Prefix Guard

The p95 verifier reached `hanging_mug` trial 3/5 and then crashed after a new trial reset. The first post-reset draft passed action/world verification with `accepted_prefix=16`. In RoboTwin evaluation the first action frame is conditioned and skipped (`start_idx=1`), so accepting only the first frame executes zero real actions and produces an empty `key_frame_list`. The teacher verifier had not run a full `_infer` in that fresh trial, so its `init_latent` was still unset; the following `compute_kv_cache` could not build a valid latent/action cache and failed with `AttributeError: 'NoneType' object has no attribute 'shape'`.

Fix: in `RiskRouterClientPolicy`, when `frame_st_id == 0`, reject verified prefixes that cover only the conditioned first frame (`accepted_prefix <= action_per_frame`) and fall back to teacher. This preserves normal full-prefix draft execution and only blocks the non-executable first-frame prefix.

## V6d Teacher Initial Prime

The V6c guard fixed the empty first-frame prefix, but the next run showed a deeper reset-state issue. If the first post-reset action chunk is accepted from draft, the teacher never runs a full `_infer(frame_st_id=0)`. Action/world verifier calls use the transformer flow fields but do not initialize `self.init_latent` or the streaming VAE causal cache in the same way as normal teacher generation. The next sync `compute_kv_cache` can then call `streaming_vae.encode_chunk` on post-action key frames with an unprimed temporal cache, causing a Wan VAE shortcut mismatch:

`RuntimeError: The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 2`

Fix: after draft produces the initial action candidate, `RiskRouterClientPolicy` now forces the first executable chunk of every trial through the teacher as `teacher_initial_prime`, before low-risk acceptance or action/world verification. This is intentionally conservative and only affects `frame_st_id == 0`; later chunks still use the RiskRouter/world-latent verifier path. The draft server still receives the initial inference request, so draft-side state remains aligned for later cache updates.

## V7 Action Local Repair Before Teacher Fallback

### Goal

Reduce unnecessary teacher fallbacks while preserving or improving V6 world-latent success on RoboTwin low10.

### Hypothesis

The action verifier already computes teacher flow reconstructions of the draft action endpoint at several tau values. If a draft is mildly rejected, the mean teacher reconstruction can be used as a cheap local correction before falling back to full teacher generation. This keeps the verifier decoupled from generation step count and uses no new model or training.

### Single New Feature

Optional action repair in RiskRouter:

1. Draft produces action/action latent and optional video latent as in V6.
2. Teacher action verifier can return a repaired normalized action latent and executable action from its endpoint reconstructions.
3. If the original action verify rejects, RiskRouter reverifies the repaired action once.
4. If repaired action passes action verify and the existing world-latent verify, execute the repaired draft prefix.
5. If repair still fails, fall back to teacher exactly as before.

### Fixed Controls

- Draft/teacher configs stay `v1/a2` and `v2/a4`.
- Teacher cache mode stays `sync` for the first V7 validation.
- Existing action threshold, tau set, risk thresholds, world verifier threshold, and world tau set are unchanged.
- No video-risk, repair-network, new training, cache strategy, or PF change is introduced.

### New Controls

- `--repair_enable`: request teacher endpoint repair and allow one reverify attempt.
- `--repair_lambda`: blend factor from draft endpoint toward teacher reconstructed endpoint.

### Gate

Compare against V6 `20260701_0758_worldlatent_initialprime_sync_low10_tn10_g012`:

- success must improve beyond `72/100`,
