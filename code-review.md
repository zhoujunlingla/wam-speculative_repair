# Code Review

## Change
- Added `--eval_video_log False` and `--render_freq 0` to the shared RoboTwin client command in `scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py`.

## Why
- ACP specverify shards failed before any trial with `Render Error`.
- The known-good ACP renderfix launcher uses these exact client flags.
- This keeps verifier/model/cache logic unchanged and only aligns evaluation startup with the working render path.

## Findings
- Risk: low. The flags disable video logging/render frequency in the client and match the successful reference run.
- Regression risk: low. Direct local smoke had already reached `Render Well`; this change only affects client args.
- Scope: one shared launcher callsite, so two-server and single-server specverify evals stay consistent.

## Checks
- `python3 -m py_compile scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py` passed.
- `git diff` reviewed; only the two renderfix client flags were added.

## Decision
Proceed to ACP 1-GPU smoke before relaunching 8-GPU low10.

## 2026-07-09 ACP Renderfix Environment Review

### Change
1. Reused the known-good ACP renderfix Vulkan/EGL stack in `base_env()`: `nvidia_icd_abs_egl.json`, SAPIEN `10_nvidia.json`, and SAPIEN bundled `libvulkan.so.1.3.224`.
2. Added `CODE` and `ROBOTWIN_ROOT` to the launcher `PYTHONPATH`, matching the successful reference ACP evaluator.
3. Kept the earlier client render flags `--eval_video_log False --render_freq 0`.

### Why
The same SpecVerify config reaches `Render Well` on the development machine, while ACP fails immediately with `Render Error`. The successful ACP reference job uses the EGL renderfix/codeenv path above, so the root cause is launcher environment drift rather than verifier math.

### Findings
No verifier, threshold, cache, model, task, or teacher-rate parameter changed. The fix is limited to the shared launcher environment; both single-server and two-server paths inherit it through `base_env()`.

### Checks
- `python3 -m py_compile scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py` passed.
- `git diff --check -- scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py code-review.md` passed.
- `git diff` reviewed; only ACP render/codeenv alignment and existing client render flags are present.

### Decision
Allowed to run ACP 1-GPU smoke. Do not relaunch 8-GPU low10 until the ACP smoke reaches `Render Well` and starts a real trial.

## 2026-07-09 ACP Smoke Follow-up: Remove Unsupported Client Flags

### Change
Removed `--eval_video_log False --render_freq 0` from the SpecVerify client command.

### Why
After the ACP renderfix environment patch, the smoke reached `Render Well`, proving the render stack is fixed. The job then failed because this repository's `eval_polict_client_openpi.py` does not define those two client arguments. They were useful in the reference launcher but are not reusable here.

### Findings
This is still startup-only. Verifier parameters, draft/teacher configs, cache policy, tau, and threshold are unchanged.

### Checks
- `python3 -m py_compile scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py` passed.
- `git diff --check -- scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py code-review.md` passed.

### Decision
Allowed to rerun ACP 1-GPU smoke. Expected next gate: `Render Well` followed by a real trial instead of argparse failure.
