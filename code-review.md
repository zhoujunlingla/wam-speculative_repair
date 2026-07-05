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

