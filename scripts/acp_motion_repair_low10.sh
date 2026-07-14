#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/afs/intern/manlichen/ivan/zhoujunl}
CODE=${CODE:-$ROOT/Wam_Speed_up/lingbot-va-motion-repair-20260714}
RUN_TAG=${RUN_TAG:?set RUN_TAG}
RUN_ROOT=${RUN_ROOT:-$ROOT/experiments/Wam_Speed_up/$RUN_TAG}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/result/Wam_Speed_up/$RUN_TAG}
TEST_NUM=${TEST_NUM:-20}
GPU_COUNT=${GPU_COUNT:-8}
CODE_COMMIT=${CODE_COMMIT:?set CODE_COMMIT to the reviewed implementation commit}
POLICY_NAME=motion-on+gripper-phase-snap+zero-prefix-flow-rk2

if [[ "$GPU_COUNT" != "8" ]]; then
  echo "This matched low10 runner requires GPU_COUNT=8" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" || -e "$RESULT_ROOT" ]]; then
  echo "Refusing to reuse an experiment root" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$RESULT_ROOT/logs" "$RESULT_ROOT/shards"
export ROOT CODE RUN_ROOT RESULT_ROOT TEST_NUM CODE_COMMIT POLICY_NAME
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false

{
  echo "run_tag=$RUN_TAG"
  echo "code=$CODE"
  echo "code_commit=$CODE_COMMIT"
  echo "test_num=$TEST_NUM"
  echo "policy=$POLICY_NAME"
  echo "start=$(date -Is)"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
} | tee "$RUN_ROOT/logs/meta.txt"
cp "$RUN_ROOT/logs/meta.txt" "$RESULT_ROOT/logs/meta.txt"

task_shards=(
  "hanging_mug"
  "turn_switch"
  "place_can_basket"
  "open_microwave"
  "press_stapler"
  "put_object_cabinet"
  "stack_bowls_three put_bottles_dustbin"
  "dump_bin_bigbin pick_diverse_bottles"
)

pids=()
for gpu in $(seq 0 7); do
  shard_run="$RUN_ROOT/shards/g$gpu"
  shard_result="$RESULT_ROOT/shards/g$gpu"
  mkdir -p "$shard_run" "$shard_result"
  (
    task_index=0
    for task in ${task_shards[$gpu]}; do
      port=$((64000 + gpu * 20 + task_index))
      cd "$CODE"
      /usr/bin/python scripts/run_realtime_flash_task.py \
        --mode spec --task "$task" --gpu "$gpu" --client-gpu "$gpu" \
        --port "$port" --test-num "$TEST_NUM" \
        --run-root "$shard_run" --result-root "$shard_result" \
        --threshold 0.15 --tau 50 100 --pf-interval 20 \
        --video-motion-gate-threshold 1.2 \
        --gripper-consensus --teacher-gripper-fallback \
        --gripper-repair --gripper-repair-tau 75 \
        --flow-consistent-repair --flow-repair-tau 150 --repair-max-per-episode 4 \
        --flow-repair-max-axis-delta 0.2 --repair-max-step-rms 0.15 \
        --repair-prefix-len 16 --server-timeout 900
      task_index=$((task_index + 1))
    done
  ) > "$RUN_ROOT/logs/shard_g$gpu.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done

/usr/bin/python - <<'PY'
import csv
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
result_root = Path(os.environ["RESULT_ROOT"])
summaries = []
for path in sorted((result_root / "shards").glob("g*/summary_*.json")):
    summaries.append(json.loads(path.read_text()))

rows = [item["metric"] for item in summaries]
success = sum(row["succ_num"] for row in rows)
total = sum(row["total_num"] for row in rows)
source_counts = {}
repair = {"eligible": 0, "attempted": 0, "executed": 0,
          "primary_verify_forwards": 0, "construction_forwards": 0,
          "holdout_forwards": 0, "holdout_evaluated": 0,
          "action_only_forwards": 0, "kinds": {},
          "episodes_executed": 0, "episodes_success": 0,
          "episode_success_rate": None, "episode_kinds": {}}
latency = {}
model_time = {"action_sec": 0.0, "cache_sec": 0.0, "total_sec": 0.0,
              "executed_actions": 0}
for item in summaries:
    policy = item["policy"]
    for key, value in policy["counts"].items():
        source_counts[key] = source_counts.get(key, 0) + int(value)
    for key in ("eligible", "attempted", "executed",
                "primary_verify_forwards", "construction_forwards",
                "holdout_forwards", "holdout_evaluated",
                "action_only_forwards",
                "episodes_executed", "episodes_success"):
        repair[key] += int(policy.get("repair", {}).get(key, 0))
    for key, value in policy.get("repair", {}).get("kinds", {}).items():
        repair["kinds"][key] = repair["kinds"].get(key, 0) + int(value)
    for key, value in policy.get("repair", {}).get("episode_kinds", {}).items():
        stats = repair["episode_kinds"].setdefault(
            key, {"episodes": 0, "success": 0})
        stats["episodes"] += int(value.get("episodes", 0))
        stats["success"] += int(value.get("success", 0))
    latency[item["task"]] = policy.get("latency", {})
    for key in ("action_sec", "cache_sec", "total_sec"):
        model_time[key] += float(policy.get("model_time", {}).get(key, 0.0))
    model_time["executed_actions"] += int(
        policy.get("model_time", {}).get("executed_actions", 0))

action_rounds = source_counts.get("teacher_full", 0) + source_counts.get("draft_flash", 0)
if repair["episodes_executed"]:
    repair["episode_success_rate"] = (
        repair["episodes_success"] / repair["episodes_executed"])
model_time["action_path_hz"] = (
    model_time["executed_actions"] / model_time["action_sec"]
    if model_time["action_sec"] else None)
model_time["end_to_end_model_hz"] = (
    model_time["executed_actions"] / model_time["total_sec"]
    if model_time["total_sec"] else None)

expected_tasks = [
    "hanging_mug", "turn_switch", "place_can_basket", "open_microwave",
    "press_stapler", "put_object_cabinet", "stack_bowls_three",
    "put_bottles_dustbin", "dump_bin_bigbin", "pick_diverse_bottles",
]
test_num = int(os.environ["TEST_NUM"])
errors = []
task_names = [item.get("task") for item in summaries]
if sorted(task_names) != sorted(expected_tasks):
    errors.append(f"task set mismatch: {task_names}")
for item in summaries:
    if item.get("client_rc") != 0:
        errors.append(f"{item.get('task')}: client_rc={item.get('client_rc')}")
    if item.get("metric", {}).get("total_num") != test_num:
        errors.append(
            f"{item.get('task')}: total={item.get('metric', {}).get('total_num')}"
        )
    if not item.get("artifact_valid", False):
        errors.append(
            f"{item.get('task')}: artifact={item.get('artifact_errors', [])}"
        )

config_keys = (
    "mode", "threshold", "tau", "pf_interval", "video_motion_gate_threshold",
    "gripper_consensus", "gripper_repair", "gripper_repair_tau",
    "flow_consistent_repair", "flow_repair_tau",
    "flow_repair_max_axis_delta", "repair_max_per_episode",
    "repair_max_step_rms", "repair_prefix_len", "teacher_gripper_fallback",
)
experiment_config = {
    key: summaries[0].get(key) for key in config_keys
} if summaries else {}
for item in summaries[1:]:
    mismatched = [
        key for key in config_keys if item.get(key) != experiment_config.get(key)
    ]
    if mismatched:
        errors.append(f"{item.get('task')}: config mismatch {mismatched}")

complete = not errors and total == len(expected_tasks) * test_num
summary = {
    "run_root": str(run_root),
    "result_root": str(result_root),
    "success": success,
    "total": total,
    "success_rate": success / total if total else None,
    "complete": complete,
    "errors": errors,
    "policy_name": os.environ["POLICY_NAME"],
    "code_commit": os.environ["CODE_COMMIT"],
    "test_num": test_num,
    "experiment_config": experiment_config,
    "source_counts": source_counts,
    "teacher_action_rate": (
        source_counts.get("teacher_full", 0) / action_rounds if action_rounds else None
    ),
    "repair": repair,
    "repair_holdout_accept_rate": (
        repair["executed"] / repair["holdout_evaluated"]
        if repair["holdout_evaluated"] else None
    ),
    "model_time": model_time,
    "per_task_latency": latency,
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
