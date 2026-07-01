#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/afs/intern/manlichen/ivan/zhoujunl"
CODE="$ROOT/Wam_Speed_up/lingbot-va-specverify"
STAMP="$(date +%H%M%S)"
RUN_ID="${RUN_ID:-20260620_specverify_rtvla_low10_tn10_a800_avoid_gpu4_${STAMP}}"
EXP_ROOT="$ROOT/experiments/Wam_Speed_up/$RUN_ID"
RESULT_ROOT="$ROOT/result/Wam_Speed_up/$RUN_ID"
mkdir -p "$EXP_ROOT/logs" "$RESULT_ROOT/logs"

cat > "$RESULT_ROOT/run_manifest.json" <<JSON
{
  "run_id": "$RUN_ID",
  "start_time": "$(date '+%F %T')",
  "experiment_content": "Realtime-VLA first-version SpecVerify on LingBot-VA: FlashWAM v1/a1 draft + LingBot v25/a50 teacher verifier/fallback, low10 clean TN10, GPU4 avoided due unrelated memory use.",
  "gpu_policy": "Avoid GPU4. Do not kill or preempt unrelated processes.",
  "shards": [
    {"name": "shard0", "server_gpu": 0, "client_gpu": 1, "tasks": ["hanging_mug", "press_stapler", "dump_bin_bigbin"]},
    {"name": "shard1", "server_gpu": 2, "client_gpu": 3, "tasks": ["turn_switch", "put_object_cabinet", "pick_diverse_bottles"]},
    {"name": "shard2", "server_gpu": 5, "client_gpu": 7, "tasks": ["place_can_basket", "stack_bowls_three"]},
    {"name": "shard3", "server_gpu": 6, "client_gpu": 6, "tasks": ["open_microwave", "put_bottles_dustbin"]}
  ]
}
JSON
cp "$RESULT_ROOT/run_manifest.json" "$EXP_ROOT/run_manifest.json"

launch_shard() {
  local shard="$1"
  local server_gpu="$2"
  local client_gpu="$3"
  local port_base="$4"
  local master_base="$5"
  shift 5
  local tasks=("$@")
  local shard_exp="$EXP_ROOT/$shard"
  local shard_result="$RESULT_ROOT/$shard"
  mkdir -p "$shard_exp" "$shard_result"
  echo "[$(date '+%F %T')] launch $shard server_gpu=$server_gpu client_gpu=$client_gpu tasks=${tasks[*]}" | tee -a "$RESULT_ROOT/logs/scheduler.log"
  (
    cd "$CODE"
    python3 scripts/cci_specverify_rtvla_low10_tn10.py \
      --run-root "$shard_exp" \
      --result-root "$shard_result" \
      --draft-gpu "$server_gpu" \
      --teacher-gpu "$server_gpu" \
      --client-gpu "$client_gpu" \
      --port-base "$port_base" \
      --master-port-base "$master_base" \
      --server-warmup-sec 300 \
      --test-num 10 \
      --wait-screen __no_wait_screen__ \
      --tasks "${tasks[@]}"
  ) > "$RESULT_ROOT/logs/${shard}.log" 2>&1
}

launch_shard shard0 0 1 36600 37600 hanging_mug press_stapler dump_bin_bigbin &
echo "$!" > "$RESULT_ROOT/logs/shard0.pid"
launch_shard shard1 2 3 36700 37700 turn_switch put_object_cabinet pick_diverse_bottles &
echo "$!" > "$RESULT_ROOT/logs/shard1.pid"
launch_shard shard2 5 7 36800 37800 place_can_basket stack_bowls_three &
echo "$!" > "$RESULT_ROOT/logs/shard2.pid"
launch_shard shard3 6 6 36900 37900 open_microwave put_bottles_dustbin &
echo "$!" > "$RESULT_ROOT/logs/shard3.pid"

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
teacher = {
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
rows_by_task = {}
counts = {
    "draft_accept": 0,
    "teacher_full": 0,
    "teacher_fallback": 0,
    "phase_fallback": 0,
    "verify_reject": 0,
}
for shard in sorted(p for p in result_root.iterdir() if p.is_dir() and p.name.startswith("shard")):
    summary_path = shard / "summary_latest.json"
    if not summary_path.exists():
        continue
    data = json.loads(summary_path.read_text())
    for k, v in (data.get("specverify_counts") or {}).items():
        counts[k] = counts.get(k, 0) + int(v)
    for row in data.get("rows") or []:
        if int(row.get("total_num", 0)) > 0:
            rows_by_task[row["task"]] = row

rows = []
for task, ref in teacher.items():
    row = rows_by_task.get(task) or {
        "task": task,
        "succ_num": 0,
        "total_num": 0,
        "succ_rate": 0.0,
        "teacher_clean_rate": ref,
        "delta_vs_teacher": -ref,
        "status": "missing",
    }
    rows.append(row)

success = sum(int(r["succ_num"]) for r in rows)
total = sum(int(r["total_num"]) for r in rows)
rate = success / total if total else 0.0
teacher_mean = sum(teacher.values()) / len(teacher)
summary = {
    "run_root": str(exp_root),
    "result_root": str(result_root),
    "end_time": datetime.now().isoformat(timespec="seconds"),
    "exit_status": status,
    "experiment_content": "Realtime-VLA first-version SpecVerify on LingBot-VA, low10 clean TN10, GPU4 avoided.",
    "success": success,
    "total": total,
    "success_rate": rate,
    "teacher_low10_mean_rate": teacher_mean,
    "delta_vs_teacher_low10_mean": rate - teacher_mean,
    "specverify_counts": counts,
    "rows": rows,
}
for root in (result_root, exp_root):
    (root / "summary_latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (root / "per_task_success.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "succ_num", "total_num", "succ_rate", "teacher_clean_rate", "delta_vs_teacher", "status"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# SpecVerify RTVLA Low10 TN10 A800 Avoid GPU4",
        "",
        f"- End: {summary['end_time']}",
        f"- Exit status: {status}",
        f"- Success: {success}/{total} = {rate * 100:.2f}%",
        f"- Teacher low10 mean: {teacher_mean * 100:.2f}%",
        f"- Delta: {(rate - teacher_mean) * 100:+.2f} pp",
        f"- SpecVerify counts: `{counts}`",
        "",
        "| Task | Success | Total | Rate | Teacher Clean | Delta | Status |",
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
