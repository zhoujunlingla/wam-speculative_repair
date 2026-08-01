# V16 Code Review

## Findings Fixed

- P0: verifier tau rows were incorrectly mapped onto positive/negative CFG cache rows. Each tau now runs a complete CFG batch and uses the same action-guidance reduction as normal LingBot inference.
- P1: original verify and repair reverify sampled different Gaussian probes. One `verify_seed` is now shared by both calls.
- P1: the repair-reject fallback test was empty. It now executes the policy and asserts two verifies followed by full teacher fallback.
- P1: execute-mode repair attempts were absent from summaries and reject records dropped `repair_verify`. Logs and aggregation now preserve/count accept and reject outcomes.
- P1: the launcher hardcoded the old code root and a shared `ckpt_setting`. It now resolves its own repository and derives an isolated setting from the run tag.

## Residual Risks

- Each verifier tau now costs one full CFG forward. This is correct but slower than the invalid K-as-CFG batching and must be measured.
- `lazy_reference` keeps verification on the last teacher reference and replays ordered cache updates before full teacher fallback. Fallback latency may be bursty.
- Initial shadow-prime behavior is unchanged because it is required by the existing WanVAE cache path; both comparison runs use the same behavior.

## Checks

- `python3 -m pytest -q tests/test_specverify.py tests/test_specverify_client_policy.py`: 21 passed.
- `python3 -m py_compile` on changed Python files: passed.
- `bash -n scripts/acp_specverify_low10_4gpu.sh`: passed.
- `git diff --check`: passed.
- Two independent post-change reviewers: GO, no blocking P0/P1.

## Decision

GO. The real-model repair smoke reached `Render Well`, completed a 400-step
RoboTwin episode, and logged initial verify, seeded repair reverify, and teacher
fallback records without verifier/cache errors. Launch paired no-repair and
reject-only-repair low10 TN=20 jobs on separate 4xA800 ACP pods.
