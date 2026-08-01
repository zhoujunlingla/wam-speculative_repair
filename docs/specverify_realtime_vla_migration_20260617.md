# Realtime-VLA-Style Action Verify Migration - 2026-06-17

## Scope

Implement the first code slice for speculative verification without touching the
original clean checkout:

- original directory left untouched:
  `/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va`
- new experimental code directory:
  `/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-specverify`

This slice adds only verifier utilities and an optional action-only verifier API.
It does not change the default `VA_Server.infer()` path and does not yet change
the RobotWin client execution loop.

## Code Decisions

1. Use an independent full-resolution verifier scheduler.

   The normal `action_scheduler` is mutated by `set_timesteps(action_steps)`.
   The verifier needs exact tau levels such as 150 and 300, so
   `make_verify_scheduler()` always calls `set_timesteps(1000, training=True)`.

2. Preserve the teacher CFG cache contract for every verifier timestep.

   The teacher cache has one row per CFG condition. Each tau therefore runs as
   its own complete CFG batch; tau rows must never be mapped onto CFG cache rows.

3. Do not use scheduler vector APIs blindly for K-batch verification.

   `FlowMatchScheduler.add_noise(..., t_dim=2)` broadcasts timestep values along
   the frame dimension, not the verifier batch dimension. The new
   `scheduler_add_noise_batched()` applies one tau per batch row.

   `FlowMatchScheduler.step()` accepts only one timestep, so
   `scheduler_step_to_final_batched()` loops over K rows after the batched model
   forward.

4. Respect the chunk-0 action condition boundary.

   At `frame_st_id == 0`, LingBot-VA forces action frame 0 to a clean zero
   condition. The verifier excludes that frame from distance scoring.

5. Prefix execution should remain at observation-frame boundaries.

   Prefixes are quantized to `{0, 16, 32}` for RobotWin
   (`action_per_frame=16`, `frame_chunk_size=2`) to avoid half-frame replanning
   before the environment has a fresh observation.

## Added Files

Remote experimental checkout:

```text
/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-specverify/wan_va/specverify.py
/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-specverify/tests/test_specverify.py
```

Modified in the experimental checkout only:

```text
/mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-specverify/wan_va/wan_va_server.py
```

## Verification

Lightweight unit runner, because cci system Python has Torch but no pytest:

```bash
cd /mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-specverify
python3 - <<'PY'
import importlib.util
from pathlib import Path
path = Path('tests/test_specverify.py')
spec = importlib.util.spec_from_file_location('test_specverify', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in sorted(n for n in dir(mod) if n.startswith('test_')):
    getattr(mod, name)()
print('all test_specverify functions passed')
PY
```

Result:

```text
all test_specverify functions passed
```

Syntax check:

```bash
python3 -m py_compile wan_va/specverify.py wan_va/wan_va_server.py tests/test_specverify.py
```

Result: passed.

## Offline Teacher-Self Sanity Result

Run root:

```text
/mnt/afs/intern/manlichen/ivan/zhoujunl/experiments/Wam_Speed_up/20260617_specverify_teacher_self_sanity
```

Command summary:

```bash
CUDA_VISIBLE_DEVICES=2 python3 scripts/offline_teacher_self_sanity.py \
  --model-path /mnt/afs/intern/manlichen/ivan/zhoujunl/models/lingbot-va-posttrain-robotwin \
  --run-root /mnt/afs/intern/manlichen/ivan/zhoujunl/experiments/Wam_Speed_up/20260617_specverify_teacher_self_sanity \
  --gpu 2 \
  --threshold 0.15 \
  --tau 150 300 \
  --enable-offload
```

Result:

| Check | Value |
|---|---:|
| Verdict | PASS |
| Tau timesteps | 150, 300 |
| Threshold | 0.15 |
| Valid action positions | 32 |
| Pass count | 32 |
| Pass rate | 100.00% |
| Accepted prefix | 32 |
| Distance mean | 0.002022 |
| Distance max | 0.003128 |
| Distance p95 | 0.002834 |
| Elapsed | 61.65 s |

Interpretation:

The teacher-generated normalized action chunk passes the teacher action-flow
verifier by a large margin. This supports the engineering alignment of the
verifier scheduler, K-batch action packing, normalized action space, and
pre-action-cache verification point for this one offline example. It is not a
RobotWin success claim.

## Next Step

The next allowed code step is the closed-loop client/runtime bridge:

1. Add frame-boundary prefix execution only: `{0, 16, 32}`.
2. Add Realtime-VLA-style phase-aware fallback on gripper switch.
3. Add `PF=2` periodic full-path refresh.
4. Log accepted prefix, fallback reason, verifier distance, and cache age.

Do not add WAM video-risk routing or repair-before-fallback in the same run.

## Closed-Loop Client Bridge - 2026-06-17 18:09 CST

Implemented the first closed-loop bridge with only the Realtime-VLA first
version components:

1. Prefix execution at observation-frame boundaries: `{0, 16, 32}`.
2. Phase-aware fallback from gripper switch detection.
3. `PF=2` periodic full-path refresh.

Important LingBot-specific adaptation:

- `compute_kv_cache` is mirrored to both draft and teacher servers after every
  executed prefix. PF controls whether the next action is generated by the
  teacher full path; it does not leave the teacher KV cache stale. This keeps
  verifier and fallback frame indices aligned with LingBot's autoregressive
  video-action cache.

Added/modified files:

```text
evaluation/robotwin/specverify_client_policy.py
evaluation/robotwin/eval_polict_client_openpi.py
wan_va/configs/va_robotwin_flashwam_cfg.py
wan_va/configs/va_robotwin_lingbot_posttrain_cfg.py
scripts/cci_specverify_rtvla_low10_tn10.py
```

Verification:

```bash
cd /mnt/afs/intern/manlichen/ivan/zhoujunl/Wam_Speed_up/lingbot-va-specverify
python3 - <<'PY'
import importlib.util
from pathlib import Path
for path in [Path('tests/test_specverify.py'), Path('tests/test_specverify_client_policy.py')]:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(n for n in dir(mod) if n.startswith('test_')):
        getattr(mod, name)()
print('all lightweight tests passed')
PY

python3 -m py_compile \
  evaluation/robotwin/eval_polict_client_openpi.py \
  evaluation/robotwin/specverify_client_policy.py \
  wan_va/wan_va_server.py \
  scripts/cci_specverify_rtvla_low10_tn10.py
```

Result: passed on cci.

Launch status:

```text
Screen: specverify_rtvla_low10_tn10_20260617
Run root: /mnt/afs/intern/manlichen/ivan/zhoujunl/experiments/Wam_Speed_up/20260617_specverify_rtvla_low10_tn10
Result root: /mnt/afs/intern/manlichen/ivan/zhoujunl/result/Wam_Speed_up/20260617_specverify_rtvla_low10_tn10
State: queued, waiting for screen lingbotva_top10_chunks_20260617 to finish before using GPU0/1/2
```

Experiment content:

```text
FlashWAM-RoboTwin v1/a1 draft + LingBot-VA posttrain RobotWin v25/a50 teacher
action-flow verifier/fallback, tau={150,300}, delta=0.15, PF=2, clean low10,
10 trials per task.
```

The run will write `summary_latest.json`, `summary.md`,
`per_task_success.csv`, `command.sh`, `queued_at.txt`, `start_time.txt`, and
client/server logs under the result root.
