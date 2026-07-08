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

## V7b Repair Instrumentation Only

### Goal

Determine whether action local repair has viable opportunities before allowing it to change closed-loop behavior.

### Problem Evidence

The V7 low10 run was stopped early at `18/30 = 60%`. More importantly, its metrics showed `draft_repair_accept = 0`, so the added repair capability had no observable positive effect. A root-cause check found an implementation gap: the launcher passed `--repair_enable`, but `eval_polict_client_openpi.py` did not forward `repair_enable` or `repair_lambda` into `RiskRouterClientPolicy`, so the policy never requested repair from the teacher verifier.

### Single New Feature

V7b fixes the missing repair argument forwarding and adds `repair_instrument_only`.

When enabled:

1. The teacher verifier returns a repaired action candidate when the original draft fails action verification.
2. The policy re-verifies the repaired candidate and optionally checks the existing world-latent verifier.
3. The policy logs `repair_attempt`, `repair_action_pass`, `repair_world_pass`, `repair_fail_action`, `repair_fail_world`, and `repair_accept`.
4. The policy still falls back to the original teacher path. It never executes the repaired action.

### Fixed Controls

V7b keeps V6/V7 controls unchanged: draft `v1/a2`, teacher `v2/a4`, `teacher_cache_mode=sync`, action threshold `0.18`, risk thresholds `0.25/0.55`, world threshold `0.35`, and tau `[150, 300]`.

### Gate

Do not enable repair execution unless V7b shows nontrivial `repair_accept` candidates on low10 without introducing runtime issues. If `repair_attempt` remains near zero, the repair branch is not reachable and the next change should target verifier/reject routing rather than repair quality.

## V8 SVDR: Video-Guided Draft Repair

### Goal

Use the WAM-specific future video latent without adding training or a new learned module. The verifier should still trust Realtime-VLA-style action endpoint consistency, but rejected draft chunks get one chance to repair before falling back to the teacher.

### Hypothesis

FlashWAM's cached future video latent contains a short-horizon prediction of where the scene is changing. Contact-rich or discrete-transition phases should show concentrated latent motion in a few regions/frames. That motion can guide how strongly each action step should move toward the teacher action verifier endpoint residual:

`A_repair[h] = A_draft[h] + lambda_video[h] * (A_teacher_endpoint[h] - A_draft[h])`

If the repair passes the same action-only verifier, the system can execute more draft-derived actions and reduce teacher fallback without replacing the verifier with an untrusted video score.

### Single New Feature

SVDR adds a training-free video-guided repair branch:

1. Draft returns action, action latent, and future video latent.
2. Existing teacher action verifier rejects the draft and returns the endpoint reconstruction repair candidate.
3. `video_motion_risk` computes frame-level risk from adjacent draft future video latent differences using top-k regional motion, median-relative motion, and concentration.
4. `video_guided_blend` maps that risk to per-frame/per-action-step repair strengths between `svdr_lambda_min` and `svdr_lambda_max`.
5. The repaired action is verified again by the existing action-only verifier.
6. If it passes, execute `draft_svdr_repair_accept`; otherwise fall back as `teacher_svdr_repair_reject`.

No world-latent verifier, no learned IDM, no teacher video repair, no training.

### Fixed Controls

- Draft/teacher configs remain `v1/a2` and `v2/a4`.
- Teacher cache mode remains `sync` for the first validation.
- Action verify tau stays `[150, 300]`.
- Action threshold stays `0.18`.
- Risk thresholds stay `0.25/0.55`.
- `world_verify_enable` stays off for SVDR validation so the second check is action-only.

### New Controls

- `--svdr_repair_enable`
- `--svdr_lambda_min`, default `0.05`
- `--svdr_lambda_max`, default `0.90`
- `--svdr_motion_ref`, default `3.0`
- `--svdr_topk_frac`, default `0.10`
- `--svdr_temperature`, default `1.0`

### Gate

Run smoke first to confirm the draft server returns `video_latent` under `return_video_latent=True`. Then low10 TN=10 can proceed if:

- no reset/cache shape mismatch,
- nonzero `draft_svdr_repair_accept`,
- teacher fallback rate decreases relative to V6/V7,
- success is not worse than V6 worldlatent baseline,
- latency is not worse than action repair without video guidance.

## V9 On-Policy Step2000 Draft Swap

### Goal

Test whether the same SVDR verifier/repair policy improves when the draft model is the on-policy step2000 video1/action2 checkpoint instead of the raw LingBot v1/a2 direct draft.

### Single New Variable

Only the draft server checkpoint changes:

- Draft config: `robotwin_onpolicy_v1a2_draft`.
- Draft model path: `/mnt/afs/intern/manlichen/ivan/zhoujunl/models/FlashWAM_eval_onpolicy_v1a2/step_2000_target_student`.
- Draft inference budget: video=1, action=2.

Fixed controls remain unchanged: teacher config `robotwin_lingbot_v2a4_teacher`, teacher inference budget video=2/action=4, low10 clean TN=10, SVDR action-only reverify, no world-latent gate, and no new training.

### Gate

Compare against the previous SVDR v1/a2 draft run using the same low10 TN=10 protocol. Proceed only if the on-policy draft improves success or reduces teacher fallback without new cache/reset errors.



## V10 Single-Server Speculative Policy Smoke

### Goal

Match the Realtime-VLA FLASH deployment shape more closely: expose one policy server to the RobotWin client while keeping draft and teacher inside that server process.

### Single New Feature

Add `wan_va/wan_va_single_spec_server.py`, a thin wrapper that instantiates local draft and teacher `VA_Server` objects and passes them into the existing `RiskRouterClientPolicy` through local client adapters.

### Not Changing

- No new verifier logic.
- No threshold change.
- No new training.
- No low10 evaluation for this gate.
- No separate draft/teacher websocket servers in this smoke path.

### Smoke Gate

Pass if the single server starts on one A800 GPU and can serve a minimal RobotWin client request path without import/config errors. If two models in one process OOM, the next minimal variant is a one-policy-server/two-GPU runtime, not another verifier change.

## V11 Formal Single-Server Low10 Launcher

### Goal

Run the V10 single-server runtime through the durable low10 launcher so RobotWin clients use the same one-port deployment shape as Realtime-VLA FLASH style serving.

### Single New Feature

Add `--single-server` to `scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`.

When enabled, the launcher starts `wan_va/wan_va_single_spec_server.py` on the draft GPU. The RobotWin client connects to that single port without `--specverify_enable`; draft/teacher routing happens inside the server process.

### Fixed Controls

- Draft config stays `robotwin_onpolicy_v1a2_draft`.
- Teacher config stays `robotwin_lingbot_v2a4_teacher`.
- SVDR/RiskRouter thresholds and flags are passed through unchanged.
- Two-server mode remains available and unchanged for fallback comparison.

### Verification Plan

1. `py_compile` launcher and single-server server.
2. `--help` confirms the launcher exposes `--single-server`.
3. Start low10 TN=10 with one single server and one RobotWin client GPU.
4. Check first task reaches normal RobotWin progress and server logs metrics to `specverify_metrics.jsonl`.

### Added

2026-07-05T10:38:38

## Stage A0 All-Draft Sanity Prime Bypass

The previous all-draft sanity run was invalid because `RiskRouterClientPolicy` still forced `teacher_initial_prime` at `frame_st_id == 0`. Add an explicit `--disable-teacher-initial-prime` switch for sanity-only runs. Default behavior remains unchanged, so normal action verification, SVDR, and world-verifier experiments still keep the conservative teacher initial prime.

Gate: with `risk_verify_mode=off`, `phase_mode=ignore`, and `--disable-teacher-initial-prime`, router success should approach the direct onpolicy step2000 v1/a2 baseline before enabling action-only verification.

## V10b Stage A0 Single-Model Sanity Modes

### Goal

Make Stage A0 test the router boundary without loading an unused model. In a draft-only sanity run, the teacher must not be constructed; in a teacher-only sanity run, the draft must not be constructed. Only full speculative evaluation should load both models.

### Change

`wan_va_single_spec_server.py` now supports `--server-mode`:

- `draft_teacher`: default, unchanged full RiskRouter path with draft and teacher.
- `draft_only`: loads only `--draft-config-name` and forwards requests directly to that VA server.
- `teacher_only`: loads only `--teacher-config-name` and forwards requests directly to that VA server.

`scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py` forwards this as `--single-server-mode`.

### Gate

Use `--single-server --single-server-mode draft_only` for Stage A0. Its result should match the new-code direct draft baseline before any action verifier, SVDR repair, or world verifier experiment is trusted.

## V12 Realtime-VLA Action Verify Explicit Config Launcher

### Goal

Make the two-server Realtime-VLA-style action verifier use the current explicit model configs instead of a generic env-injected checkpoint config. This keeps draft and teacher experiment identity unambiguous.

### Reference

Cloned dexmal/realtime-vla-flash to /mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/realtime-vla-flash at da6ceccad603695a8a3d6fa14dd410c3aadb536f. The relevant path is scripts/spec/spec_serve_policy.py: draft action -> teacher verify at t_list -> radius prefix acceptance -> gripper/phase fallback -> optional periodic full refresh.

### Single New Change

Update scripts/cci_specverify_rtvla_low10_tn10.py to run from the current SVDR code root and start the draft server with robotwin_onpolicy_v1a2_draft by default. The teacher remains robotwin_lingbot_v2a4_teacher. The verifier math is unchanged and still uses wan_va/wan_va_server.py verify_action_chunk.

### Not Changing

No new verifier implementation, no threshold change, no cache-policy change, no SVDR/world verifier change, and no experiment launch in this edit.

### Verification Plan

Run py_compile on the launcher and existing specverify modules, then run the existing lightweight specverify tests.

## V13 Success-First SVDR Repair Gate

### Goal

Make speculative inference beat the direct draft on low10 before optimizing teacher usage further. The target is low10 TN=10 success above 75%.

### Problem Evidence

V12/V8-style runs showed many failed chunks were logged as `draft_verify_accept`, often with full-prefix acceptance. `SVDR` only ran after action verifier rejection, so it never repaired false-positive accepts. The first post-reset teacher prime also executed the teacher chunk, which can hurt draft-heavy contact tasks such as `hanging_mug`.

### Minimal Change

1. Change initial teacher prime to shadow prime: run teacher once to initialize caches, but do not execute its action.
2. If an accepted draft is risky (`phase_switch`, risk score >= 0.35, or verifier margin near threshold), force it through the existing SVDR repair + action reverify path instead of executing it directly.

No new model, no new loss, no world verifier change.

## V13b Reject-Only SVDR Repair

### Goal

Fix V13 over-repair. SVDR repair should only run when the original action verifier rejects the draft chunk (`accepted_prefix <= 0`), i.e. the path that would otherwise fall back to teacher.

### Change

Remove the risky-accept override that converted accepted draft prefixes into repair attempts. Accepted draft prefixes now execute normally. Rejected prefixes still use the existing SVDR repair + action reverify path before teacher fallback.

### Gate

`forced_repair_after_accept` must disappear from new metrics. `draft_svdr_repair_accept` should only appear after verifier rejection, and success should be compared against V13 and direct draft.

## V14 DPS-Style Bounded Residual Repair

### Goal

Replace high-gain video-lambda SVDR repair with a safer inference-time repair inspired by DPS / manifold-constrained guidance: small bounded correction toward the teacher verifier endpoint, followed by the same action verifier.

### Change

1. Keep Realtime-VLA-FLASH-style action verifier unchanged.
2. Request teacher endpoint repair whenever either `repair_enable` or `svdr_repair_enable` is enabled.
3. In the SVDR branch, use video latent motion only as logging metadata.
4. Replace `video_guided_blend` execution with `bounded_residual_blend`:
   `A_repair = A_draft + alpha * clip(A_endpoint - A_draft)`, where `alpha = repair_lambda` and per-step continuous-action L2 clip is `0.30`.
5. Reverify repaired latent with the same action verifier. Execute repaired action only if it passes; otherwise fallback teacher.

### Not Changing

No new training, no candidate search, no world/video teacher rollout, no verifier threshold change.

### Gate

New metrics should show `svdr.repair_method=bounded_residual`. Compare success, teacher rate, and `draft_svdr_repair_accept` against V13b and direct draft.

## V14b WanVAE Temporal Cache Retry

A bounded-repair shard repeatedly crashed on `turn_switch` when a teacher fallback hit WanVAE streaming encode with only two temporal frames for a kernel-size-3 conv. The minimal guard is in the shared `VA_Server.infer` full-inference path: if and only if that exact WanVAE temporal-cache RuntimeError is raised, reset the server cache/prompt and retry the same observation once. This keeps verifier/repair policy unchanged and avoids hiding unrelated errors.

## V15 Action-Only Verify Hyperparameter Tuning

Goal: finish stage 1 before repair. Keep world verifier and repair disabled, tune only the Realtime-VLA-FLASH action verifier hyperparameters: selected flow timesteps tau and endpoint distance threshold. The launcher now exposes `--specverify-tau` and passes it to both single-server and two-server paths. No verifier math or policy logic changes.


## V15b Variable-K Action Verify Cache Chunking

V15 showed K=3 action verifier timesteps fail before evaluation with transformer KV cache batch mismatch: verifier input batch was 3 while LingBot teacher caches are allocated with CFG batch size 2. The action verifier now keeps the existing fast path when K matches the cache batch and otherwise evaluates tau timesteps in cache-sized chunks, padding only the final partial chunk and discarding padded outputs. Verifier math, thresholds, repair, and world verifier remain unchanged.
