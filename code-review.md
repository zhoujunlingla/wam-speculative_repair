# Code Review: V6 World-Latent Flow Consistency Verifier

## Change Reviewed

Repository: `/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-riskrouter-worldlatent-20260701`

Single new feature: optional world-latent flow verification after an action-flow verifier accepts a draft chunk.

## Findings

No blocking findings.

Risk level: medium. The server-side video verifier reuses the teacher video branch and full-resolution flow scheduler, so it should be semantically aligned with LingBot-VA. However, it adds extra teacher forwards after action verification and the initial `world_verify_threshold=0.35` is uncalibrated. First run should be interpreted as signal calibration, not final speed result.

## Notes

- Default behavior is preserved: `world_verify_enable=False` unless explicitly set.
- The first validation should use `teacher_cache_mode=sync` to isolate verifier quality from the previously observed lazy/stale VAE cache synchronization failures.
- The verifier checks draft video latent consistency under teacher video flow; it does not yet condition video rollout on draft action. This is the minimal inference-only version of the world-latent idea.

## Tests Run

- `python3 -m py_compile wan_va/wan_va_server.py`
- `python3 -m py_compile evaluation/robotwin/specverify_client_policy.py`
- `python3 -m py_compile evaluation/robotwin/eval_polict_client_openpi.py`
- `python3 -m py_compile scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`
- `python3 -m pytest -q tests/test_specverify.py tests/test_specverify_client_policy.py`
  - Result: `9 passed in 1.14s`

## Decision

Allowed to proceed to small RoboTwin clean validation on GPUs 0/1/2. Use sync teacher cache for the first validation. Stop early if the world verifier is constant reject/pass or if any server-side shape/cache error recurs.

## V6b Review: P95 World Score Calibration

Change: world verifier pass criterion changed from raw max latent-patch distance to p95 latent-patch distance. The threshold remains `0.35`.

Reason: first calibration run showed the interface works but max distance is dominated by single-patch outliers (`world_n=16`, `pass=1`, `reject=15`; median max about `0.56`, median p95 about `0.24`). P95 keeps a high-region criterion while avoiding a single noisy patch deciding the whole chunk.

Risk: medium-low. This makes the world gate less strict; it may accept chunks that max-distance would reject. The action verifier still runs first, and world max/p95 remain logged for later analysis.

Tests: `python3 -m py_compile wan_va/wan_va_server.py`; `python3 -m pytest -q tests/test_specverify.py tests/test_specverify_client_policy.py` -> `9 passed in 1.25s`.

Decision: allowed to rerun the same TN5 sync-cache calibration.

## V6c Review: Quantile Dtype Fix

Finding: V6b failed immediately because `torch.quantile` does not accept half/bfloat16 tensors. The world verifier distance tensor follows model dtype, so the p95 score must cast to float before quantile.

Fix: compute `torch.quantile(valid_dist.float().flatten(), 0.95)`.

Tests: `python3 -m py_compile wan_va/wan_va_server.py`; `python3 -m pytest -q tests/test_specverify.py tests/test_specverify_client_policy.py` -> `9 passed in 1.35s`.

Decision: allowed to rerun the same TN5 sync-cache calibration.
# Code Review: V6c Initial Partial Prefix Guard

## Scope

- Repository: `/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-riskrouter-worldlatent-20260701`
- Change: reject `RiskRouterClientPolicy` verified prefixes that only include the first conditioned frame at episode start.
- Motivation: V6c crashed after reset because a 16-step initial prefix executes zero real RobotWin steps, creates an empty cache update, and leaves the teacher verifier without `init_latent`.

## Findings

- No blocking findings.
- Risk level: low-to-medium. The change is deliberately narrow: it only applies when `frame_st_id == 0` and `0 < accepted_prefix <= action_per_frame`. It does not alter action verifier distances, world-latent verifier scores, or non-initial prefix behavior.

## Tests Run

- `python3 -m pytest -q tests/test_specverify_client_policy.py::test_risk_router_rejects_initial_partial_prefix_before_cache_update tests/test_specverify_client_policy.py tests/test_specverify.py`
  - Result: `11 passed`

## Proceed Decision

Allowed to rerun the `hanging_mug`/low10 smoke. The next run should confirm no `action_model_input=None` / empty-cache crash after trial reset.
