#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/afs/intern/manlichen/ivan/zhoujunl}
RUN_TAG=${RUN_TAG:?set RUN_TAG}
RUN_ROOT=${RUN_ROOT:-$ROOT/experiments/Wam_Speed_up/$RUN_TAG}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/result/Wam_Speed_up/$RUN_TAG}
CODE=${CODE:-$ROOT/Wam_Speed_up/wam-speculative_repair-v16-20260710}
LAUNCHER=$CODE/scripts/cci_riskrouter_lingbot_v1a2_v2a4_low10_tn10.py
DRAFT_MODEL=${DRAFT_MODEL:-$ROOT/models/FlashWAM_eval_multiscale_v1a2/step_2000_target_student_20260706_resume1500_s3000}
DRAFT_CONFIG=${DRAFT_CONFIG:-robotwin_eval_ckpt}
TEACHER_CONFIG=${TEACHER_CONFIG:-robotwin_lingbot_v2a4_teacher}
TEST_NUM=${TEST_NUM:-20}
REPAIR_ENABLE=${REPAIR_ENABLE:-0}
REPAIR_LAMBDA=${REPAIR_LAMBDA:-0.25}
TEACHER_CACHE_MODE=${TEACHER_CACHE_MODE:-lazy_reference}
SERVER_WARMUP_SEC=${SERVER_WARMUP_SEC:-240}

mkdir -p "$RUN_ROOT/logs" "$RESULT_ROOT/logs" "$RESULT_ROOT/shards"
git -C "$CODE" diff --quiet
git -C "$CODE" diff --cached --quiet
test -z "$(git -C "$CODE" ls-files --others --exclude-standard)"
export ROOT RUN_ROOT RESULT_ROOT TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export LD_LIBRARY_PATH="$ROOT/env/nvidia-550.90.07-jammy/extract/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.1/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-$ROOT/experiments/Wam_Speed_up/20260709_acp_eval_entrypoints/nvidia_icd_abs_egl.json}
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-$ROOT/env/python_pkgs/sapien/vulkan_library/10_nvidia.json}
export SAPIEN_VULKAN_LIBRARY_PATH=${SAPIEN_VULKAN_LIBRARY_PATH:-$ROOT/env/python_pkgs/sapien/vulkan_library/libvulkan.so.1.3.224}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}
export EVAL_MODEL_PATH=$DRAFT_MODEL EVAL_VIDEO_STEPS=1 EVAL_ACTION_STEPS=2
export TORCH_EXTENSIONS_DIR=$RUN_ROOT/torch_extensions

repair_args=()
if [[ "$REPAIR_ENABLE" == "1" ]]; then
  repair_args=(--repair-enable --repair-lambda "$REPAIR_LAMBDA" --repair-step-mask-enable)
fi

{
  echo "run_tag=$RUN_TAG"
  echo "code=$CODE"
  echo "code_commit=$(git -C "$CODE" rev-parse HEAD)"
  echo "draft_model=$DRAFT_MODEL"
  echo "teacher_config=$TEACHER_CONFIG"
  echo "test_num=$TEST_NUM"
  echo "repair_enable=$REPAIR_ENABLE"
  echo "repair_lambda=$REPAIR_LAMBDA"
  echo "teacher_cache_mode=$TEACHER_CACHE_MODE"
  echo "start=$(date -Is)"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
} | tee "$RUN_ROOT/logs/meta.txt"
cp "$RUN_ROOT/logs/meta.txt" "$RESULT_ROOT/logs/meta.txt"

task_shards=(
  "hanging_mug open_microwave"
  "turn_switch put_bottles_dustbin pick_diverse_bottles"
  "place_can_basket stack_bowls_three press_stapler"
  "put_object_cabinet dump_bin_bigbin"
)

pids=()
for gpu in 0 1 2 3; do
  shard_run=$RUN_ROOT/shards/g$gpu
  shard_result=$RESULT_ROOT/shards/g$gpu
  mkdir -p "$shard_run/logs" "$shard_result/logs"
  (
    cd "$CODE"
    python3 "$LAUNCHER" \
      --single-server --single-server-mode draft_teacher \
      --draft-config-name "$DRAFT_CONFIG" --teacher-config-name "$TEACHER_CONFIG" \
      --draft-gpu "$gpu" --teacher-gpu "$gpu" --client-gpu "$gpu" \
      --port-base "$((65000 + gpu * 10))" --master-port-base "$((65200 + gpu * 10))" \
      --server-warmup-sec "$SERVER_WARMUP_SEC" --test-num "$TEST_NUM" \
      --run-root "$shard_run" --result-root "$shard_result" \
      --teacher-cache-mode "$TEACHER_CACHE_MODE" --risk-verify-mode medium \
      --risk-low 1.01 --risk-high 0.0 \
      --specverify-threshold 0.26 --specverify-tau 150 300 \
      --phase-mode tighten --phase-threshold-scale 1.0 \
      --verify-plus-enable --verify-alpha-cross-tau 0.10 \
      --verify-shortcut-enable --verify-alpha-shortcut 0.10 \
      --verify-dynamics-gate "${repair_args[@]}" --wait-screen "" \
      --tasks ${task_shards[$gpu]}
  ) > "$RUN_ROOT/logs/shard_g$gpu.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done

python3 - <<'PY'
import csv, json, os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
result_root = Path(os.environ["RESULT_ROOT"])
rows, counts, integrity, shard_latency = [], {}, {"log_exists": True, "parsed_rows": 0, "invalid_rows": 0}, {}
for shard in sorted((result_root / "shards").glob("g*")):
    path = shard / "summary_latest.json"
    if not path.exists():
        integrity["log_exists"] = False
        continue
    data = json.loads(path.read_text())
    rows.extend(r for r in data.get("rows", []) if int(r.get("total_num", 0)) > 0)
    for key, value in data.get("specverify_counts", {}).items():
        counts[key] = counts.get(key, 0) + int(value)
    item = data.get("metrics_integrity", {})
    integrity["log_exists"] = integrity["log_exists"] and bool(item.get("log_exists", False))
    integrity["parsed_rows"] += int(item.get("parsed_rows", 0))
    integrity["invalid_rows"] += int(item.get("invalid_rows", 0))
    shard_latency[shard.name] = data.get("latency_summary", {})

success = sum(int(row.get("succ_num", 0)) for row in rows)
total = sum(int(row.get("total_num", 0)) for row in rows)
action_rounds = counts.get("draft_accept", 0) + counts.get("teacher_full", 0) + counts.get("teacher_fallback", 0)
teacher_total = counts.get("teacher_full", 0) + counts.get("teacher_fallback", 0)
repair_attempts = counts.get("repair_attempt", 0)
summary = {
    "run_root": str(run_root), "result_root": str(result_root),
    "success": success, "total": total, "success_rate": success / total if total else 0.0,
    "specverify_counts": counts,
    "specverify_rates": {
        "action_rounds": action_rounds,
        "draft_accept_rate": counts.get("draft_accept", 0) / action_rounds if action_rounds else None,
        "teacher_total_rate": teacher_total / action_rounds if action_rounds else None,
        "repair_accept_rate": counts.get("repair_accept", 0) / repair_attempts if repair_attempts else None,
    },
    "metrics_integrity": integrity, "shard_latency": shard_latency, "rows": rows,
}
for root in (run_root, result_root):
    (root / "summary_merged.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (root / "per_task_success_merged.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "succ_num", "total_num", "succ_rate", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
PY

exit "$status"
