#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/afs/intern/manlichen/ivan/zhoujunl}
CODE=${CODE:-$ROOT/Wam_Speed_up/lingbot-va-motion-repair-20260714}
RUN_TAG=${RUN_TAG:?set RUN_TAG}
RUN_ROOT=${RUN_ROOT:-$ROOT/experiments/Wam_Speed_up/$RUN_TAG}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/result/Wam_Speed_up/$RUN_TAG}
TEST_NUM=${TEST_NUM:-20}
GPU_COUNT=${GPU_COUNT:-8}

if [[ "$GPU_COUNT" != "8" ]]; then
  echo "This matched low10 runner requires GPU_COUNT=8" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" || -e "$RESULT_ROOT" ]]; then
  echo "Refusing to reuse an experiment root" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$RESULT_ROOT/logs" "$RESULT_ROOT/shards"
export ROOT CODE RUN_ROOT RESULT_ROOT PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false

{
  echo "run_tag=$RUN_TAG"
  echo "code=$CODE"
  echo "code_commit=${CODE_COMMIT:-$(git -C "$CODE" rev-parse HEAD 2>/dev/null || echo synced-worktree)}"
  echo "test_num=$TEST_NUM"
  echo "policy=motion-on+gripper-phase-snap+zero-prefix-endpoint"
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
        --zero-prefix-repair --repair-max-per-episode 2 \
        --repair-strength 0.5 --repair-max-step-rms 0.15 \
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
          "action_only_forwards": 0, "kinds": {}}
latency = {}
for item in summaries:
    policy = item["policy"]
    for key, value in policy["counts"].items():
        source_counts[key] = source_counts.get(key, 0) + int(value)
    for key in ("eligible", "attempted", "executed", "action_only_forwards"):
        repair[key] += int(policy.get("repair", {}).get(key, 0))
    for key, value in policy.get("repair", {}).get("kinds", {}).items():
        repair["kinds"][key] = repair["kinds"].get(key, 0) + int(value)
    latency[item["task"]] = policy.get("latency", {})

action_rounds = source_counts.get("teacher_full", 0) + source_counts.get("draft_flash", 0)
summary = {
    "run_root": str(run_root),
    "result_root": str(result_root),
    "success": success,
    "total": total,
    "success_rate": success / total if total else None,
    "source_counts": source_counts,
    "teacher_action_rate": (
        source_counts.get("teacher_full", 0) / action_rounds if action_rounds else None
    ),
    "repair": repair,
    "repair_holdout_accept_rate": (
        repair["executed"] / repair["attempted"] if repair["attempted"] else None
    ),
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
PY

exit "$status"
