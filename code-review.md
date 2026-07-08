# Code Review: V9 On-Policy Step2000 Draft Swap

## Scope

Review of the V9 change in `lingbot-va-svdr-videorepair-20260702`.

User request: keep the SVDR/RiskRouter policy unchanged, switch the draft model to the on-policy step2000 video1/action2 checkpoint, keep teacher as LingBot-VA video2/action4, and prepare for low10 x 10 evaluation without launching the experiment.

## Findings

No blocking findings.

## Diff Summary

- Added `wan_va/configs/va_robotwin_onpolicy_v1a2_draft_cfg.py`.
- Registered `robotwin_onpolicy_v1a2_draft` in `wan_va/configs/__init__.py`.
- Updated the low10 launcher default `--draft-config-name` from `robotwin_lingbot_v1a2_draft` to `robotwin_onpolicy_v1a2_draft`.
- Updated launcher experiment text and `ckpt_setting` to identify the on-policy step2000 draft.
- Relaxed the RiskRouter docstring from LingBot-specific draft wording to generic v1/a2 draft wording.
- Appended V9 design notes to `design.md`.

## Risk Assessment

Risk level: low to medium.

- The code change is configuration-only for the model swap; SVDR verifier, repair logic, thresholds, teacher config, and cache mode are unchanged.
- Runtime risk remains that the on-policy checkpoint must be LingBot-compatible and return `action_latent` / `video_latent` under the existing server flags. The model directory has the expected `text_encoder`, `transformer`, and `vae` layout.
- Experiment validity risk is low: the single new variable is the draft checkpoint. The teacher remains `robotwin_lingbot_v2a4_teacher` with video=2/action=4.

## Verification

Commands run:

```bash
python3 -m py_compile   wan_va/configs/va_robotwin_onpolicy_v1a2_draft_cfg.py   wan_va/configs/__init__.py   scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py   evaluation/robotwin/specverify_client_policy.py
```

```bash
DIFFUSERS_DISABLE_BITSANDBYTES=1 PYTHONPATH=/mnt/afs/intern/manlichen/ivan/zhoujunl/env/torch29_clean_pkgs:/mnt/afs/intern/manlichen/ivan/zhoujunl/env/python_pkgs python3 - <<'CHECK'
from wan_va.configs import VA_CONFIGS
cfg = VA_CONFIGS['robotwin_onpolicy_v1a2_draft']
print(cfg.wan22_pretrained_model_name_or_path)
print(cfg.num_inference_steps, cfg.action_num_inference_steps, cfg.video_exec_step)
print(VA_CONFIGS['robotwin_lingbot_v2a4_teacher'].num_inference_steps, VA_CONFIGS['robotwin_lingbot_v2a4_teacher'].action_num_inference_steps)
CHECK
```

Result: draft path is `/mnt/afs/intern/manlichen/ivan/zhoujunl/models/FlashWAM_eval_onpolicy_v1a2/step_2000_target_student`; draft steps are `1 2 -1`; teacher steps are `2 4`.

```bash
DIFFUSERS_DISABLE_BITSANDBYTES=1 PYTHONPATH=/mnt/afs/intern/manlichen/ivan/zhoujunl/env/torch29_clean_pkgs:/mnt/afs/intern/manlichen/ivan/zhoujunl/env/python_pkgs pytest -q tests/test_specverify_client_policy.py tests/test_specverify.py
```

Result: `17 passed in 8.39s`.

## Proceed Decision

Allowed to proceed to low10 x 10 evaluation when requested. No experiment was launched in this change, per user instruction.


# Code Review: V16 Verify++ and Step-Mask Repair

## Scope

Review of the Verify++ / step-mask repair implementation on branch `wam-speculative_repair`.

Changed files:

- `wan_va/specverify.py`
- `wan_va/wan_va_server.py`
- `evaluation/robotwin/specverify_client_policy.py`
- `evaluation/robotwin/eval_polict_client_openpi.py`
- `wan_va/wan_va_single_spec_server.py`
- `scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`
- `wan_va/configs/__init__.py`
- `wan_va/configs/va_robotwin_flashwam_official_step3000_v1a2_draft_cfg.py`
- `tests/test_specverify_client_policy.py`

## Findings

No blocking findings remain.

Fixed during review:

- Step-mask repair originally defaulted `repair_mask_dilate_radius=1`, which could repair verified-pass neighbor steps. This violated the requirement that pass steps stay unchanged. Default is now `0`; dilation remains an explicit ablation knob.

## Diff Summary

- Added `scheduler_step_to_timestep_batched()` for two-step shortcut verification.
- Added optional Verify++ signals in `VA_Server.verify_action_chunk()`: cross-tau endpoint consistency, shortcut consistency, and dynamics gate.
- Added JSON-safe per-step verifier outputs: `pass_by_step`, `step_score`, `endpoint_dist_max`, `cross_tau_score`, `shortcut_score`, `dynamics_score`.
- Added step-mask repair helpers and wired them before repair re-verification.
- Added CLI pass-through for Verify++ and step-mask repair flags.
- Added FlashWAM official step3000 v1/a2 draft config and made it the default draft for the launcher/single-server path.

## Reviews

- Server verifier sub-agent: no blocking bugs; old endpoint verifier is preserved when `verify_plus=False`; no cache mutation beyond existing action-only verify path.
- Client/launcher sub-agent: found the dilation default bug above; after fixing to default `0`, no remaining blocking issue identified.

## Verification

Commands run:

```bash
python3 -m py_compile wan_va/specverify.py wan_va/wan_va_server.py wan_va/wan_va_single_spec_server.py evaluation/robotwin/specverify_client_policy.py evaluation/robotwin/eval_polict_client_openpi.py scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py tests/test_specverify_client_policy.py
```

```bash
python3 - <<'PY'
import importlib.util
from pathlib import Path
p = Path("tests/test_specverify_client_policy.py").resolve()
spec = importlib.util.spec_from_file_location("test_specverify_client_policy", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in ["test_step_masked_repair_preserves_verified_steps", "test_step_repair_mask_dilation_expands_neighbors"]:
    getattr(mod, name)()
print("step-mask checks passed")
PY
```

Both passed locally and on `a800-cci-8855` after syncing.

## Proceed Decision

Allowed to push. No evaluation was launched in this change.


# Code Review: V10 Single-Server Speculative Policy Smoke

## Scope

Review of `wan_va/wan_va_single_spec_server.py`.

## Findings

No blocking findings before smoke.

## Diff Summary

- Added a single-process websocket policy server.
- Reused existing `RiskRouterClientPolicy` by adapting local `VA_Server` instances to the same `.infer(obs)` client interface.
- Kept SVDR/RiskRouter behavior unchanged.

## Risks

- Medium runtime risk: loading draft and teacher in one process on one GPU may OOM. This is exactly what the smoke test checks.
- Low correctness risk: routing logic is reused rather than reimplemented.
- Low artifact risk: default save root is `/tmp/wanva_single_spec_server`, avoiding AFS result writes during smoke.

## Verification

- `python3 -m py_compile wan_va/wan_va_single_spec_server.py` passed.
- Runtime smoke started `wan_va_single_spec_server.py` on GPU4 with draft `robotwin_onpolicy_v1a2_draft` and teacher `robotwin_lingbot_v2a4_teacher`.
- `/healthz` returned OK after both models loaded; GPU4 memory was about 61.5 GB, so one-process/two-model loading is feasible on A800 80GB.
- A real RobotWin `hanging_mug` TN=1 client connected to the single server on one port and reached step 817/900.
- Server metrics logged 61 requests: `teacher_initial_prime=1`, `compute_kv_cache=30`, `draft_low_risk=3`, `draft_verify_accept=25`, `teacher_svdr_repair_reject=2`. No server traceback, OOM, `action_model_input=None`, or tensor shape mismatch was observed.
- The smoke did not produce a final `res.json`; treat it as API/runtime-path evidence only, not success-rate evidence.

## Proceed Decision

Allowed to proceed from architecture smoke. Next change should either wire this single-server path into the standard launcher for durable runs, or keep using the old two-server launcher for success-rate experiments until a full single-server episode writer is needed. Do not claim low10 performance from this smoke.

# Code Review: V11 Formal Single-Server Low10 Launcher

## Scope

Review of wiring `wan_va/wan_va_single_spec_server.py` into `scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py` via `--single-server`.

## Findings

No blocking findings.

## Diff Summary

- Added `start_single_spec_server()` to launch the single-process draft+teacher server.
- Added `--single-server` launcher flag.
- In single-server mode the RobotWin client now connects to one `--port` and does not receive client-side `--specverify_enable` flags.
- RiskRouter, phase, world-verify, action-repair, and SVDR parameters are forwarded to the single server so the formal run matches the launcher CLI.
- Two-server mode remains intact for comparison.

## Risk Assessment

Risk level: medium runtime risk, low code risk.

The change moves routing from client-side policy into the one server process. The smoke already showed both models fit on one A800 and the request path emits metrics. The formal low10 risk is RobotWin episode completion, not import or shape compatibility.

## Verification

Commands run:

```bash
python3 -m py_compile scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py wan_va/wan_va_single_spec_server.py
```

```bash
python3 scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py --help
```

Both passed. `--help` exposes `--single-server` and the SVDR/RiskRouter controls.

## Proceed Decision

Allowed to launch low10 TN=10 in single-server mode. Proceeding does not claim success until `summary_latest.json`, per-task results, and `specverify_metrics.jsonl` are produced.

Recorded at 2026-07-05T10:38:38.

# Code Review: Stage A0 Single-Model Sanity Modes

## Scope

Changed:

- `wan_va/wan_va_single_spec_server.py`
- `scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`

## Findings

No blocking findings.

## Risk

Low. Default behavior remains `draft_teacher`, so existing full speculative runs keep loading both models. New `draft_only` and `teacher_only` modes bypass `RiskRouterClientPolicy` and forward requests directly to one `VA_Server`, which is exactly the Stage A0 sanity requirement.

## Checks Run

```bash
python3 -m py_compile wan_va/wan_va_single_spec_server.py scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py
python3 scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py --help | grep -n "single-server-mode\|single-server"
git diff --check -- wan_va/wan_va_single_spec_server.py scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py design.md code-review.md
```

Passed.

## Proceed Decision

Allowed to proceed to Stage A0 paired sanity with `--single-server-mode draft_only`. Do not proceed to verifier/SVDR until this matches direct draft under the same client episode/instruction protocol.


# Code Review: V12 Realtime-VLA Explicit Config Launcher

## Scope

Review of mapping dexmal/realtime-vla-flash action verification into the current LingBot-VA draft/teacher setup without changing verifier math.

## Findings

No blocking findings.

## Diff Summary

- Cloned realtime-vla-flash as a read-only reference at /mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/realtime-vla-flash, commit da6ceccad603695a8a3d6fa14dd410c3aadb536f.
- Updated scripts/cci_specverify_rtvla_low10_tn10.py to use the current SVDR code root instead of the old lingbot-va-specverify root.
- Replaced the hardcoded draft config robotwin_flashwam with explicit --draft-config-name, default robotwin_onpolicy_v1a2_draft.
- Kept teacher default robotwin_lingbot_v2a4_teacher.
- Left wan_va/wan_va_server.py verify_action_chunk unchanged because it already implements the Realtime-VLA radius-prefix endpoint verification pattern.
- Set --wait-screen default to None so the launcher does not block on an old historical screen by default.

## Risk Assessment

Risk level: low. The change is launcher/config identity only. Runtime risk is the usual two-server memory and RoboTwin episode variability; verifier math, thresholds, phase fallback, and cache policy are unchanged.

## Verification

- python3 -m py_compile scripts/cci_specverify_rtvla_low10_tn10.py wan_va/specverify.py wan_va/wan_va_server.py evaluation/robotwin/specverify_client_policy.py wan_va/wan_va_single_spec_server.py passed.
- pytest -q tests/test_specverify.py tests/test_specverify_client_policy.py passed: 17 passed in 8.62s.
- scripts/cci_specverify_rtvla_low10_tn10.py --help exposes --draft-config-name, --teacher-config-name, --specverify-pf, --phase-mode, and --wait-screen.

## Proceed Decision

Allowed to use this launcher for two-server Realtime-VLA action-only verification after the direct explicit-config baselines are accepted. Do not treat robotwin_eval_ckpt runs as formal baselines.


## 2026-07-07 Launcher Torch Extension Cache Review

Change reviewed: `scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py` now honors an externally supplied `TORCH_EXTENSIONS_DIR` before falling back to the shared cache. This is needed because prior torch2.9 RoboTwin evals failed from stale shared curobo extension ABI artifacts.

Findings: no blocker. The change is scoped to launcher environment construction and does not alter model configs, verifier thresholds, routing policy, or metrics. Existing behavior is preserved when no override is supplied.

Risks: per-run extension directories may add one-time compile latency on first launch, but avoid cross-run ABI contamination.

Checks run: `python3 -m py_compile scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py wan_va/wan_va_single_spec_server.py evaluation/robotwin/specverify_client_policy.py`.

Proceed decision: allowed to launch speculative sampling experiments with per-run `TORCH_EXTENSIONS_DIR`.

## 2026-07-07 V13 Success-First SVDR Repair Gate

Findings: no blocking issues in the diff. The change reuses existing teacher verifier repair outputs and `video_guided_blend`; it does not add a new model path. Main risk is higher teacher verifier/fallback cost because accepted-but-risky chunks now repair/reverify. This is intentional for the success-first gate.

Checks run:
- `python3 -m py_compile evaluation/robotwin/specverify_client_policy.py wan_va/wan_va_single_spec_server.py scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`

Proceed: yes, launch low10 TN=10 validation.

## 2026-07-08 V13b Reject-Only SVDR Repair Review

Finding: V13 forced repair on accepted-but-risky prefixes, which violated the intended policy and overrode valid verifier accepts. This likely explains excessive repair activity and poor success despite low teacher rate.

Change reviewed: deleted the `accepted_prefix > 0` forced-repair block from `evaluation/robotwin/specverify_client_policy.py`. Existing `accepted_prefix <= 0` repair-before-teacher path is unchanged.

Risk: low. This is a deletion of the offending path. It may increase direct draft execution and lower repair counts; teacher fallback behavior for true rejects is unchanged.

Checks run: `python3 -m py_compile evaluation/robotwin/specverify_client_policy.py wan_va/wan_va_single_spec_server.py scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`; grep confirmed `forced_repair_after_accept` is no longer present in the policy code.

Proceed decision: allowed to relaunch V13b after py_compile and grep checks pass.

## 2026-07-08 V14 DPS-Style Bounded Residual Repair Review

Findings: no blocker after updating tests. The old SVDR test expected video-latent motion to produce non-uniform repair weights; that behavior is intentionally replaced by bounded residual repair. Video motion remains logged under `svdr.video_motion` but no longer controls correction magnitude.

Change reviewed:
1. Added `bounded_residual_blend` in `evaluation/robotwin/specverify_client_policy.py`.
2. Changed repair request to `want_repair = repair_enable or svdr_repair_enable`.
3. Replaced SVDR execution path with DPS-style small endpoint residual correction using `repair_lambda` as alpha and per-step L2 clip 0.30.
4. Updated tests to assert shadow-prime behavior and bounded residual repair metadata.

Risk: medium. The correction is safer than the previous high-gain video lambda, but fixed clip 0.30 may be too conservative or too strong for some tasks. This is acceptable for the first validation because the repaired action is still reverified before execution.

Checks run:
- `python3 -m py_compile evaluation/robotwin/specverify_client_policy.py tests/test_specverify_client_policy.py wan_va/wan_va_single_spec_server.py scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`
- `pytest -q tests/test_specverify.py tests/test_specverify_client_policy.py` -> 17 passed.

Proceed decision: allowed to launch V14 low10 shard validation. Use `--repair-lambda 0.25` for the first bounded-guidance run.

## V14b Code Review - WanVAE Temporal Cache Retry

Findings: no blocking issue in the minimal guard. It catches only the exact WanVAE temporal-cache conv3d error, resets cache through existing `_reset(prompt)`, and retries once. Risk: mid-episode fallback loses old teacher temporal cache, but that is preferable to crashing and only occurs on the known invalid cache state.

Checks: `python3 -m py_compile wan_va/wan_va_server.py`; direct helper asserts for matching/non-matching RuntimeError messages.

Decision: allowed to relaunch the failed A shard continuation on free GPUs.

## V15 Code Review - Action Verify Tau CLI

Findings: no blocking issue. The change only exposes existing `RiskRouterClientPolicy.tau_timesteps` through `scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py` as `--specverify-tau`; defaults remain `[150, 300]`. Repair/world verifier paths are unchanged and disabled for V15 experiments.

Checks: `python3 -m py_compile scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py wan_va/wan_va_single_spec_server.py evaluation/robotwin/specverify_client_policy.py`; help output contains `--specverify-tau`.

Decision: allowed to launch action-only tuning runs.


## V15b Code Review - Variable-K Action Verify Cache Chunking

Findings: no blocking issues found in the V15b diff. Root cause was action verifier batching tau timesteps directly into the transformer batch dimension while the LingBot teacher KV cache is allocated with CFG batch size 2. K=2 matched by accident; K=3 failed before evaluation. The fix keeps the existing fast path when verifier K matches cache batch size and chunks/pads only mismatched K values.

Risk: medium-low. K=2 behavior should remain unchanged. K>2 now costs extra verifier forwards for the padded chunk, so latency must be measured in V15b. The fix does not change verifier math, thresholds, repair, or world-verifier logic.

Checks run:
- `python3 -m py_compile wan_va/wan_va_server.py wan_va/wan_va_single_spec_server.py scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py evaluation/robotwin/specverify_client_policy.py`
- `python3 tests/test_specverify_client_policy.py`
- `python3 -m unittest tests.test_specverify_client_policy -v` was attempted but the repository's `tests/` directory is not an importable package; direct execution is the valid lightweight check.

Allowed to proceed: yes, launch a V15b K=3 smoke/low10 retry when a server+client GPU pair is free.
