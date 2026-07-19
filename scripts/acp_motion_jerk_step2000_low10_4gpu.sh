#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/afs/intern/manlichen/ivan/zhoujunl
CODE=${CODE:-$ROOT/Wam_Speed_up/wam-worldflow-progressivek-20260717}
RUN_TAG=${RUN_TAG:?set RUN_TAG}
RUN_ROOT=${RUN_ROOT:-$ROOT/experiments/Wam_Speed_up/$RUN_TAG}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/result/Wam_Speed_up/$RUN_TAG}
TEST_NUM=${TEST_NUM:-20}
GPU_COUNT=${GPU_COUNT:-4}
MOTION_THRESHOLD=${MOTION_THRESHOLD:-1.2}
MOTION_JERK_THRESHOLD=${MOTION_JERK_THRESHOLD:-1.3}
STRICT_VERIFY_THRESHOLD=${STRICT_VERIFY_THRESHOLD:-0.10}
STRICT_PREFIX=${STRICT_PREFIX:-16}
CODE_COMMIT=${CODE_COMMIT:?set CODE_COMMIT to the reviewed implementation commit}
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python}
TASK_TIMEOUT_SEC=${TASK_TIMEOUT_SEC:-21600}
BOOT_DIR=${BOOT_DIR:-$ROOT/result/Wam_Speed_up/acp_boot_logs}
POLICY_NAME=official-step2000+motion-jerk-stratified+fixed-k2
COMPARABILITY_NOTE="formal Official step2000 fixed-K2 Motion-Jerk Low10x20 evaluation"
DRAFT_CONFIG=robotwin_flashwam_step2000_v1a2
TEACHER_CONFIG=robotwin_lingbot_v2a4
DRAFT_MODEL=$ROOT/models/FlashWAM_eval_official_v1a2/step_2000_target_student
TEACHER_MODEL=$ROOT/models/lingbot-va-posttrain-robotwin

mkdir -p "$BOOT_DIR"
exec > >(tee -a "$BOOT_DIR/${RUN_TAG}.log") 2>&1
echo "[$(date -Is)] Motion-on fixed-K2 ACP preflight"
echo "run_tag=$RUN_TAG code_commit=$CODE_COMMIT"
nvidia-smi -L

if [[ "$GPU_COUNT" != "4" || "$TEST_NUM" != "20" ]]; then
  echo "Formal evaluation requires GPU_COUNT=4 and TEST_NUM=20" >&2
  exit 2
fi
visible_gpu_count=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
if [[ "$visible_gpu_count" != "$GPU_COUNT" ]]; then
  echo "Formal evaluation requires exactly $GPU_COUNT CUDA-visible GPUs; got $visible_gpu_count" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required for unattended task evaluation" >&2
  exit 2
fi
git config --global --add safe.directory "$CODE"
ACTUAL_COMMIT=$(git -c safe.directory="$CODE" -C "$CODE" rev-parse HEAD)
if [[ "$ACTUAL_COMMIT" != "$CODE_COMMIT" ]]; then
  echo "CODE_COMMIT=$CODE_COMMIT does not match checkout $ACTUAL_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git -c safe.directory="$CODE" -C "$CODE" status --porcelain --untracked-files=all)" ]]; then
  echo "Formal evaluation requires a clean code checkout" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" || -e "$RESULT_ROOT" ]]; then
  echo "Refusing to reuse an experiment root" >&2
  exit 2
fi
require_transformer_weights() {
  local transformer=$1
  "$PYTHON_BIN" - "$transformer" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
single = root / "diffusion_pytorch_model.safetensors"
if single.is_file() and single.stat().st_size > 0:
    raise SystemExit(0)
index = root / "diffusion_pytorch_model.safetensors.index.json"
if not index.is_file() or index.stat().st_size == 0:
    raise SystemExit(f"missing transformer weights under {root}")
payload = json.loads(index.read_text())
shards = sorted(set(payload.get("weight_map", {}).values()))
if not shards:
    raise SystemExit(f"empty transformer weight map: {index}")
for name in shards:
    path = (root / name).resolve()
    if path.parent != root or not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or invalid transformer shard: {name}")
PY
}

for model in "$DRAFT_MODEL" "$TEACHER_MODEL"; do
  for component in vae tokenizer text_encoder transformer; do
    if [[ ! -e "$model/$component" ]]; then
      echo "Missing model component: $model/$component" >&2
      exit 2
    fi
  done
  for file in \
    "$model/vae/config.json" \
    "$model/tokenizer/tokenizer_config.json" \
    "$model/text_encoder/config.json" \
    "$model/transformer/config.json"; do
    if [[ ! -s "$file" ]]; then
      echo "Missing or empty model artifact: $file" >&2
      exit 2
    fi
  done
  require_transformer_weights "$model/transformer"
done

mkdir -p "$RUN_ROOT/logs" "$RESULT_ROOT/logs" "$RESULT_ROOT/shards"
export ROOT CODE RUN_ROOT RESULT_ROOT TEST_NUM GPU_COUNT MOTION_THRESHOLD
export MOTION_JERK_THRESHOLD STRICT_VERIFY_THRESHOLD STRICT_PREFIX
export CODE_COMMIT POLICY_NAME
export DRAFT_CONFIG TEACHER_CONFIG PYTHONUNBUFFERED=1
export DRAFT_MODEL TEACHER_MODEL COMPARABILITY_NOTE
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
export DIFFUSERS_DISABLE_BITSANDBYTES=1
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}
export LD_LIBRARY_PATH="$ROOT/env/nvidia-550.90.07-jammy/extract/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.1/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-$ROOT/experiments/Wam_Speed_up/20260709_acp_eval_entrypoints/nvidia_icd_abs_egl.json}
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-$ROOT/env/python_pkgs/sapien/vulkan_library/10_nvidia.json}
export SAPIEN_VULKAN_LIBRARY_PATH=${SAPIEN_VULKAN_LIBRARY_PATH:-$ROOT/env/python_pkgs/sapien/vulkan_library/libvulkan.so.1.3.224}

{
  echo "run_tag=$RUN_TAG"
  echo "code=$CODE"
  echo "code_commit=$CODE_COMMIT"
  echo "test_num=$TEST_NUM"
  echo "gpu_count=$GPU_COUNT"
  echo "policy=$POLICY_NAME"
  echo "draft_config=$DRAFT_CONFIG"
  echo "teacher_config=$TEACHER_CONFIG"
  echo "draft_model=$DRAFT_MODEL"
  echo "teacher_model=$TEACHER_MODEL"
  echo "draft_transformer_realpath=$(readlink -f "$DRAFT_MODEL/transformer")"
  echo "teacher_transformer_realpath=$(readlink -f "$TEACHER_MODEL/transformer")"
  echo "comparability_note=$COMPARABILITY_NOTE"
  echo "task_timeout_sec=$TASK_TIMEOUT_SEC"
  echo "threshold=0.15"
  echo "tau=50,100"
  echo "pf_interval=20"
  echo "video_motion_gate_threshold=$MOTION_THRESHOLD"
  echo "video_motion_jerk_gate_threshold=$MOTION_JERK_THRESHOLD"
  echo "video_motion_strict_verify_threshold=$STRICT_VERIFY_THRESHOLD"
  echo "video_motion_strict_prefix=$STRICT_PREFIX"
  echo "adaptive_k=off"
  echo "start=$(date -Is)"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
} | tee "$RUN_ROOT/logs/meta.txt"
cp "$RUN_ROOT/logs/meta.txt" "$RESULT_ROOT/logs/meta.txt"
cp "$CODE/design.md" "$CODE/code-review.md" "$RESULT_ROOT/"
cp "$0" "$RESULT_ROOT/command.sh"

# Run one server/client pair per GPU; two shards carry two tasks to cover low10.
if [[ "$GPU_COUNT" == "4" ]]; then
  task_shards=(
    "hanging_mug press_stapler dump_bin_bigbin"
    "turn_switch put_object_cabinet pick_diverse_bottles"
    "place_can_basket stack_bowls_three"
    "open_microwave put_bottles_dustbin"
  )
else
  task_shards=(
    "hanging_mug"
    "turn_switch"
    "place_can_basket"
    "open_microwave"
    "press_stapler"
    "put_object_cabinet"
    "stack_bowls_three dump_bin_bigbin"
    "put_bottles_dustbin pick_diverse_bottles"
  )
fi

pids=()
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
  shard_run="$RUN_ROOT/shards/g$gpu"
  shard_result="$RESULT_ROOT/shards/g$gpu"
  mkdir -p "$shard_run" "$shard_result"
  (
    task_index=0
    for task in ${task_shards[$gpu]}; do
      port=$((64000 + gpu * 20 + task_index))
      cd "$CODE"
      timeout --signal=TERM --kill-after=60s "$TASK_TIMEOUT_SEC" \
        "$PYTHON_BIN" scripts/run_realtime_flash_task.py \
        --mode spec --task "$task" --gpu "$gpu" --client-gpu "$gpu" \
        --port "$port" --test-num "$TEST_NUM" \
        --run-root "$shard_run" --result-root "$shard_result" \
        --draft-config-name "$DRAFT_CONFIG" \
        --teacher-config-name "$TEACHER_CONFIG" \
        --threshold 0.15 --tau 50 100 --pf-interval 20 \
        --video-motion-gate-threshold "$MOTION_THRESHOLD" \
        --video-motion-jerk-gate-threshold "$MOTION_JERK_THRESHOLD" \
        --video-motion-strict-verify-threshold "$STRICT_VERIFY_THRESHOLD" \
        --video-motion-strict-prefix "$STRICT_PREFIX" \
        --gripper-consensus --teacher-gripper-fallback \
        --paired-rng --policy-seed 0 \
        --profile-verify-latency --server-timeout 900
      task_index=$((task_index + 1))
    done
  ) > "$RUN_ROOT/logs/shard_g$gpu.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
export SHARD_STATUS=$status

"$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
result_root = Path(os.environ["RESULT_ROOT"])
summaries = [
    json.loads(path.read_text())
    for path in sorted((result_root / "shards").glob("g*/summary_*.json"))
]
expected_tasks = [
    "hanging_mug", "turn_switch", "place_can_basket", "open_microwave",
    "press_stapler", "put_object_cabinet", "stack_bowls_three",
    "put_bottles_dustbin", "dump_bin_bigbin", "pick_diverse_bottles",
]
test_num = int(os.environ["TEST_NUM"])
rows = [item["metric"] for item in summaries]
success = sum(row["succ_num"] for row in rows)
total = sum(row["total_num"] for row in rows)
source_counts = {}
verify_calls = effective_forwards = requested_forwards = 0
adaptive_live_calls = adaptive_live_k1_accepts = 0
world_flow = {
    "available_k2": 0,
    "unavailable_k1": 0,
    "missing": 0,
    "midpoint_numeric": 0,
    "half_gap_numeric": 0,
}
draft_evidence = {
    "action_dynamics_calls": 0,
    "video_motion_calls": 0,
    "delayed_video_error_calls": 0,
    "jerk_numeric": 0,
    "video_global_mean_numeric": 0,
}
for item in summaries:
    policy = item["policy"]
    for key, value in policy["counts"].items():
        source_counts[key] = source_counts.get(key, 0) + int(value)
    profile = policy.get("verify_profile", {})
    verify_calls += int(profile.get("calls", 0))
    effective_forwards += int(profile.get("effective_forwards", 0))
    requested_forwards += int(profile.get("requested_forwards", 0))
    adaptive_live = policy.get("adaptive_k_live", {})
    adaptive_live_calls += int(adaptive_live.get("calls", 0))
    adaptive_live_k1_accepts += int(adaptive_live.get("k1_accepts", 0))
    evidence = policy.get("world_flow", {})
    for key in ("available_k2", "unavailable_k1", "missing"):
        world_flow[key] += int(evidence.get(key, 0))
    world_flow["midpoint_numeric"] += int(
        evidence.get("endpoint_midpoint_distance_mean", {}).get("count", 0)
    )
    world_flow["half_gap_numeric"] += int(
        evidence.get("cross_tau_half_gap_mean", {}).get("count", 0)
    )
    draft = policy.get("draft_evidence", {})
    for key in (
        "action_dynamics_calls", "video_motion_calls", "delayed_video_error_calls"
    ):
        draft_evidence[key] += int(draft.get(key, 0))
    draft_evidence["jerk_numeric"] += int(
        draft.get("jerk_rms", {}).get("count", 0)
    )
    draft_evidence["video_global_mean_numeric"] += int(
        draft.get("video_global_mean", {}).get("count", 0)
    )

errors = []
task_names = [item.get("task") for item in summaries]
if sorted(task_names) != sorted(expected_tasks):
    errors.append(f"task set mismatch: {task_names}")
for item in summaries:
    task = item.get("task")
    if item.get("client_rc") != 0:
        errors.append(f"{task}: client_rc={item.get('client_rc')}")
    if item.get("metric", {}).get("total_num") != test_num:
        errors.append(
            f"{task}: total={item.get('metric', {}).get('total_num')}"
        )
    if item.get("draft_config_name") != os.environ["DRAFT_CONFIG"]:
        errors.append(f"{task}: wrong draft config")
    if item.get("teacher_config_name") != os.environ["TEACHER_CONFIG"]:
        errors.append(f"{task}: wrong teacher config")
    frozen = {
        "mode": "spec",
        "threshold": 0.15,
        "tau": [50.0, 100.0],
        "pf_interval": 20,
        "flow_budget_threshold": 0.0,
        "flow_budget_burst_after": 0,
        "flow_budget_burst_rounds": 0,
        "flow_budget_burst_limit": 0,
        "flow_budget_motion_ceiling": 0.0,
        "delayed_error_threshold": 0.0,
        "delayed_error_consecutive": 2,
        "delayed_error_teacher_rounds": 2,
        "video_motion_gate_threshold": float(os.environ["MOTION_THRESHOLD"]),
        "video_motion_jerk_gate_threshold": float(
            os.environ["MOTION_JERK_THRESHOLD"]
        ),
        "video_motion_strict_verify_threshold": float(
            os.environ["STRICT_VERIFY_THRESHOLD"]
        ),
        "video_motion_strict_prefix": int(os.environ["STRICT_PREFIX"]),
        "gripper_full_window": 1,
        "gripper_consensus": True,
        "teacher_gripper_fallback": True,
        "adaptive_k_shadow": False,
        "adaptive_k_live": False,
        "adaptive_k_distance_threshold": 0.05,
        "profile_verify_latency": True,
        "equivalence_audit": False,
        "deterministic_audit": False,
        "paired_rng": True,
    }
    for key, expected in frozen.items():
        if item.get(key) != expected:
            errors.append(f"{task}: {key}={item.get(key)!r}, expected {expected!r}")
if world_flow["missing"]:
    errors.append(f"world-flow telemetry missing on {world_flow['missing']} verify calls")
if verify_calls <= 0:
    errors.append("no verifier calls were recorded")
if world_flow["available_k2"] + world_flow["unavailable_k1"] != verify_calls:
    errors.append("world-flow telemetry coverage does not equal verifier calls")
if world_flow["available_k2"] <= 0:
    errors.append("no completed K2 call produced cross-tau evidence")
if world_flow["midpoint_numeric"] != world_flow["available_k2"]:
    errors.append("K2 midpoint evidence is incomplete")
if world_flow["half_gap_numeric"] != world_flow["available_k2"]:
    errors.append("K2 cross-tau gap evidence is incomplete")
if adaptive_live_calls != 0:
    errors.append(
        f"fixed-K2 unexpectedly recorded {adaptive_live_calls} Progressive-K calls"
    )
if draft_evidence["action_dynamics_calls"] <= 0:
    errors.append("no continuous-action dynamics telemetry was recorded")
if draft_evidence["video_motion_calls"] <= 0:
    errors.append("no future-video motion telemetry was recorded")
if draft_evidence["jerk_numeric"] != draft_evidence["action_dynamics_calls"]:
    errors.append("continuous-action dynamics telemetry is incomplete or non-finite")
if draft_evidence["video_global_mean_numeric"] != draft_evidence["video_motion_calls"]:
    errors.append("future-video motion telemetry is incomplete or non-finite")
if draft_evidence["action_dynamics_calls"] != draft_evidence["video_motion_calls"]:
    errors.append("draft action/video telemetry call counts do not match")
expected_draft_rows = source_counts.get("draft_flash", 0) + source_counts.get(
    "replan", 0
)
if draft_evidence["jerk_numeric"] != expected_draft_rows:
    errors.append(
        f"action evidence {draft_evidence['jerk_numeric']} != draft/replan rows "
        f"{expected_draft_rows}"
    )
if draft_evidence["video_global_mean_numeric"] != expected_draft_rows:
    errors.append(
        f"video evidence {draft_evidence['video_global_mean_numeric']} != "
        f"draft/replan rows {expected_draft_rows}"
    )
if int(os.environ["SHARD_STATUS"]) != 0:
    errors.append(f"one or more shards failed: status={os.environ['SHARD_STATUS']}")

action_rounds = source_counts.get("teacher_full", 0) + source_counts.get(
    "draft_flash", 0
)
complete = not errors and total == len(expected_tasks) * test_num
summary = {
    "run_root": str(run_root),
    "result_root": str(result_root),
    "success": success,
    "total": total,
    "gpu_count": int(os.environ["GPU_COUNT"]),
    "success_rate": success / total if total else None,
    "complete": complete,
    "errors": errors,
    "policy_name": os.environ["POLICY_NAME"],
    "code_commit": os.environ["CODE_COMMIT"],
    "draft_config": os.environ["DRAFT_CONFIG"],
    "teacher_config": os.environ["TEACHER_CONFIG"],
    "draft_model": os.environ["DRAFT_MODEL"],
    "teacher_model": os.environ["TEACHER_MODEL"],
    "comparability_note": os.environ["COMPARABILITY_NOTE"],
    "source_counts": source_counts,
    "teacher_action_rate": (
        source_counts.get("teacher_full", 0) / action_rounds
        if action_rounds else None
    ),
    "verify_calls": verify_calls,
    "effective_verify_forwards": effective_forwards,
    "requested_verify_forwards": requested_forwards,
    "verifier_forward_reduction": (
        1 - effective_forwards / requested_forwards
        if requested_forwards else None
    ),
    "world_flow_coverage": world_flow,
    "draft_evidence_coverage": draft_evidence,
    "adaptive_k_live_calls": adaptive_live_calls,
    "adaptive_k_live_k1_accepts": adaptive_live_k1_accepts,
    "rows": rows,
}
for root in (run_root, result_root):
    (root / "summary_merged.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    with (root / "per_task_success_merged.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "succ_num", "total_num", "succ_rate", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)
if not complete:
    raise SystemExit(2)
PY

exit "$status"
