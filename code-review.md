# Code Review: Realtime-VLA-FLASH Migration

## Scope

Clean LingBot-VA commit `7c6ffa9` plus a new one-server speculative policy,
official-style endpoint verifier, RoboTwin replan protocol, model configs, and
four-task launcher. No repair or risk-routing code is included.

## Findings

No blocking static finding remains before the real-model smoke.

Fixed during review:

1. `prime_only` originally fell through to full draft generation. It now only
   encodes the initial observation needed by the draft's first real cache
   update.
2. `PF=0` originally forced a full round every time. It now disables periodic
   refresh.
3. Teacher-reconstructed gripper transitions were not hard fallbacks. Any
   verifier timestep that reconstructs a switch now returns a zero prefix and
   schedules teacher full inference.
4. Probe defaults were changed from historical WAM `{150,300}` to the official
   Realtime-VLA levels `{50,100}`, equivalent to sigma `{0.05,0.10}` for the
   action scheduler with shift one.
5. Clean LingBot imported flash-attention even for `attn_mode=torch`. The
   optional import now degrades to `None`; selecting torch remains unchanged,
   while selecting an unavailable flash-attention backend still fails at use.
6. The first launcher used a raw TCP readiness probe, which produced a false
   WebSocket handshake error. Readiness now follows the server's explicit log
   marker and does not touch the socket.
7. The first TN=10 run copied every block's live KV cache for each parallel
   verifier call. Cache size grew with long episodes until three tasks OOMed
   near 79 GB. Verification now microbatches K probes and uses an explicit
   read-only attention path that never writes or duplicates persistent cache
   tensors. Episode reset also releases unused CUDA allocator blocks. The
   launcher can put rendering on a separate GPU.
8. The first memory-fix smoke exposed the expected mixed-CFG shape: video CFG
   initializes two cache rows while action guidance one produces one query
   row. The read-only path now selects row zero exactly as the removed compact
   cache did; guidance-enabled action verification selects rows `(0, 1)`.

## Risk Assessment

Residual quality risk is medium; runtime risk is low after the repaired
long-episode smoke and 30 completed post-fix trials.

- The new read-only attention path is limited to action verification and must
  be checked for numerical equivalence and exact cache preservation in the
  configured torch-attention model.
- LingBot temporal cache updates are queued after draft actions and replayed
  before full inference. Shape/order errors must fail the smoke before any
  benchmark is launched.
- Prefixes are quantized to 16 actions because that is the model's temporal
  observation boundary; this is the sole structural deviation from pi0.
- A combined server/client reached the A800 limit in long episodes. Formal
  completion uses separate server and render GPUs even after removing the
  temporary KV copy, and records both assignments.

## Checks

- `py_compile` passed for policy, client, server entrypoint, launcher, and
  configs.
- Eleven fake-model state-machine tests passed with a direct function runner.
- Remote post-fix suite: `15 passed`, including exact equality of every KV
  cache tensor before and after read-only attention.

## Decision

Allowed to proceed. Remote tests passed `15/15`, the 1500-step long smoke and
30 post-fix trials completed without OOM/cache errors, and the merged quality
gate passed at `22/40` versus draft `21/40` and teacher `23/40`. Do not claim
uniform per-task improvement: `hanging_mug` remained `0/10`, and teacher action
use is still high at `40.81%`.
