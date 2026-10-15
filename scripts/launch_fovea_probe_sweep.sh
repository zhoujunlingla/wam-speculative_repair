#!/bin/bash
# Resolution-sensitivity sweep for LingBot-VA on RoboTwin (training-free gate
# before foveated future prediction). Start a LingBot-VA server first, e.g.
#   bash evaluation/robotwin/launch_server.sh
# then run:
#   bash scripts/launch_fovea_probe_sweep.sh <save_root> <port> "<task1 task2 ...>" [test_num] [seed]
# Every condition walks the same seed sequence, so episodes are paired by seed.
# Summarize with:
#   python scripts/analyze_fovea_probe.py <save_root>/fovea_logs/*/*.jsonl
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

save_root=${1:-./results/fovea_probe}
PORT=${2:-29056}
tasks=${3:-"hanging_mug place_object_basket stack_blocks_two"}
test_num=${4:-100}
seed=${5:-0}

# view phase factor ; factor 1 is the logged no-degradation baseline
conditions=(
  "all all 1"
  "wrist fine 4"
  "wrist coarse 4"
  "head fine 4"
  "head coarse 4"
  "all all 4"
  "wrist fine 2"
  "head fine 2"
)
rule=${FOVEA_PROBE_RULE:-either}
speed_thresh=${FOVEA_PROBE_SPEED_THRESH:-0.003}

for task_name in ${tasks}; do
  for cond in "${conditions[@]}"; do
    read -r view phase factor <<< "${cond}"
    tag="${view}-${phase}-x${factor}-${rule}"
    log_dir="${save_root}/fovea_logs/${task_name}"
    mkdir -p "${log_dir}"
    PYTHONWARNINGS=ignore::UserWarning \
    python -m evaluation.robotwin.eval_polict_client_openpi --config policy/ACT/deploy_policy.yml \
      --port "${PORT}" \
      --save_root "${save_root}/runs/${tag}" \
      --video_guidance_scale 5 \
      --action_guidance_scale 1 \
      --test_num "${test_num}" \
      --fovea_probe_view "${view}" \
      --fovea_probe_phase "${phase}" \
      --fovea_probe_factor "${factor}" \
      --fovea_probe_rule "${rule}" \
      --fovea_probe_speed_thresh "${speed_thresh}" \
      --fovea_probe_log "${log_dir}/${tag}.jsonl" \
      --overrides \
      --task_name "${task_name}" \
      --task_config demo_clean \
      --train_config_name 0 \
      --model_name 0 \
      --ckpt_setting 0 \
      --seed "${seed}" \
      --policy_name ACT \
      --save_root "${save_root}/runs/${tag}"
  done
done
