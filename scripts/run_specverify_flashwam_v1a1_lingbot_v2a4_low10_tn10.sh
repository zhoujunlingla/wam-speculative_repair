#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/afs/intern/manlichen/ivan/zhoujunl
CODE=$ROOT/Wam_Speed_up/lingbot-va-specverify
TS=${1:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=$ROOT/experiments/Wam_Speed_up/${TS}_specverify_flashwam_v1a1_lingbot_v2a4_low10_tn10
RESULT_ROOT=$ROOT/result/Wam_Speed_up/${TS}_specverify_flashwam_v1a1_lingbot_v2a4_low10_tn10
LOG=$RUN_ROOT/launcher_stdout.log

mkdir -p "$RUN_ROOT" "$RESULT_ROOT"
cd "$CODE"

python3 scripts/cci_specverify_rtvla_low10_tn10.py \
  --run-root "$RUN_ROOT" \
  --result-root "$RESULT_ROOT" \
  --draft-gpu "${DRAFT_GPU:-4}" \
  --teacher-gpu "${TEACHER_GPU:-5}" \
  --client-gpu "${CLIENT_GPU:-6}" \
  --port-base "${PORT_BASE:-33380}" \
  --master-port-base "${MASTER_PORT_BASE:-33480}" \
  --server-warmup-sec "${SERVER_WARMUP_SEC:-300}" \
  --test-num "${TEST_NUM:-10}" \
  --teacher-config-name "${TEACHER_CONFIG_NAME:-robotwin_lingbot_v2a4_teacher}" \
  --teacher-cache-mode "${TEACHER_CACHE_MODE:-lazy_reference}" \
  --wait-screen "${WAIT_SCREEN:-}" \
  2>&1 | tee "$LOG"
