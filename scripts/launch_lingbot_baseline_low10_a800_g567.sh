#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/afs/intern/manlichen/ivan/zhoujunl"
CODE="$ROOT/Wam_Speed_up/lingbot-va-specverify"
STAMP="$(date +%H%M%S)"
RUN_ID="${RUN_ID:-20260621_lingbot_pure_low10_tn10_a800_g567_${STAMP}}"
EXP_ROOT="$ROOT/experiments/Wam_Speed_up/$RUN_ID"
RESULT_ROOT="$ROOT/result/Wam_Speed_up/$RUN_ID"
mkdir -p "$EXP_ROOT/logs" "$RESULT_ROOT/logs"

cat > "$RESULT_ROOT/run_manifest.json" <<JSON
{
  "run_id": "$RUN_ID",
  "start_time": "$(date '+%F %T')",
  "experiment_content": "Pure LingBot-VA posttrain v25/a50 baseline on RoboTwin clean low10 TN10; SpecVerify disabled.",
  "gpu_policy": "Use currently clean A800 GPU5/6/7. Speed profiling must be run separately on an exclusive GPU.",
  "shards": [
    {"name": "shard0", "server_gpu": 5, "client_gpu": 5, "tasks": ["hanging_mug", "open_microwave", "stack_bowls_three", "pick_diverse_bottles"]},
    {"name": "shard1", "server_gpu": 6, "client_gpu": 6, "tasks": ["turn_switch", "press_stapler", "put_bottles_dustbin"]},
    {"name": "shard2", "server_gpu": 7, "client_gpu": 7, "tasks": ["place_can_basket", "put_object_cabinet", "dump_bin_bigbin"]}
  ]
}
JSON
cp "$RESULT_ROOT/run_manifest.json" "$EXP_ROOT/run_manifest.json"

launch_shard() {
  local shard="$1"
  local gpu="$2"
  local port="$3"
  local master_port="$4"
  shift 4
  local tasks=("$@")
  local shard_exp="$EXP_ROOT/$shard"
  local shard_result="$RESULT_ROOT/$shard"
  mkdir -p "$shard_exp" "$shard_result"
  echo "[$(date '+%F %T')] launch $shard gpu=$gpu tasks=${tasks[*]}" | tee -a "$RESULT_ROOT/logs/scheduler.log"
  (
    cd "$CODE"
    python3 scripts/cci_lingbot_baseline_low10_tn10.py \
      --run-root "$shard_exp" \
      --result-root "$shard_result" \
      --server-gpu "$gpu" \
      --client-gpu "$gpu" \
      --port "$port" \
      --master-port "$master_port" \
      --server-warmup-sec 300 \
      --test-num 10 \
      --tasks "${tasks[@]}"
  ) > "$RESULT_ROOT/logs/${shard}.log" 2>&1
}

launch_shard shard0 5 37200 38200 hanging_mug open_microwave stack_bowls_three pick_diverse_bottles &
echo "$!" > "$RESULT_ROOT/logs/shard0.pid"
launch_shard shard1 6 37300 38300 turn_switch press_stapler put_bottles_dustbin &
echo "$!" > "$RESULT_ROOT/logs/shard1.pid"
launch_shard shard2 7 37400 38400 place_can_basket put_object_cabinet dump_bin_bigbin &
echo "$!" > "$RESULT_ROOT/logs/shard2.pid"

set +e
wait
status=$?
set -e

python3 - "$RESULT_ROOT" "$EXP_ROOT" "$status" <<'PY'
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

result_root = Path(sys.argv[1])
exp_root = Path(sys.argv[2])
status = int(sys.argv[3])
ref = {
    "hanging_mug": 0.40,
    "turn_switch": 0.44,
    "place_can_basket": 0.81,
    "open_microwave": 0.82,
    "press_stapler": 0.85,
    "put_object_cabinet": 0.85,
    "stack_bowls_three": 0.86,
    "put_bottles_dustbin": 0.87,
    "dump_bin_bigbin": 0.89,
    "pick_diverse_bottles": 0.89,
}
ordered = list(ref)
rows_by_task = {}
for shard in sorted(p for p in result_root.iterdir() if p.is_dir() and p.name.startswith("shard")):
    summary_path = shard / "summary_latest.json"
    if not summary_path.exists():
        continue
    data = json.loads(summary_path.read_text())
    for row in data.get("rows") or []:
        if int(row.get("total_num", 0)) > 0:
            rows_by_task[row["task"]] = row

rows = []
for task in ordered:
    rows.append(rows_by_task.get(task) or {
        "task": task,
        "succ_num": 0,
        "total_num": 0,
        "succ_rate": 0.0,
        "teacher_clean_rate": ref[task],
        "delta_vs_teacher": -ref[task],
        "status": "missing",
    })

success = sum(int(r["succ_num"]) for r in rows)
total = sum(int(r["total_num"]) for r in rows)
rate = success / total if total else 0.0
ref_mean = sum(ref.values()) / len(ref)
summary = {
    "run_root": str(exp_root),
    "result_root": str(result_root),
    "end_time": datetime.now().isoformat(timespec="seconds"),
    "exit_status": status,
    "experiment_content": "Pure LingBot-VA posttrain v25/a50 baseline on RoboTwin clean low10 TN10; SpecVerify disabled.",
    "success": success,
    "total": total,
    "success_rate": rate,
    "teacher_low10_mean_rate": ref_mean,
    "delta_vs_teacher_low10_mean": rate - ref_mean,
    "rows": rows,
}
for root in (result_root, exp_root):
    (root / "summary_latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (root / "per_task_success.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "succ_num", "total_num", "succ_rate", "teacher_clean_rate", "delta_vs_teacher", "status"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Pure LingBot-VA Low10 TN10 A800 G567",
        "",
        f"- End: {summary['end_time']}",
        f"- Exit status: {status}",
        f"- Success: {success}/{total} = {rate * 100:.2f}%",
        f"- Reference low10 mean: {ref_mean * 100:.2f}%",
        f"- Delta: {(rate - ref_mean) * 100:+.2f} pp",
        "",
        "| Task | Success | Total | Rate | Reference | Delta | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['task']} | {r['succ_num']} | {r['total_num']} | "
            f"{float(r['succ_rate']) * 100:.2f}% | {float(r['teacher_clean_rate']) * 100:.2f}% | "
            f"{float(r['delta_vs_teacher']) * 100:+.2f} pp | {r['status']} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "[$(date '+%F %T')] complete status=$status result=$RESULT_ROOT" | tee -a "$RESULT_ROOT/logs/scheduler.log"
exit "$status"
