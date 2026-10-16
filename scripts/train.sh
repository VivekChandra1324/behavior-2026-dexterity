#!/bin/bash
# Train pi0.5 on the BEHAVIOR-1K 2026 challenge data with our settings.
#
#   bash scripts/train.sh <openpi_dir>                 # settings from configs/train.env
#   TRAIN_ENV=my.env bash scripts/train.sh <openpi_dir>
#
# Starts fresh if <CKPT_DIR>/pi05_b1k/<EXP_NAME> holds no checkpoint, otherwise RESUMES from the newest one -- so
# the same command continues a run after a crash, a reboot or a job time limit.
# Multi-node: see docs/TRAINING_GUIDE.md ("More than one machine"); this script runs one process per machine.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
OPENPI=$(cd "${1:?usage: train.sh <openpi_dir>}" && pwd)
source "${TRAIN_ENV:-$HERE/configs/train.env}"

ASSET_DIR=$ASSETS_DIR/pi05_b1k/$B1K_REPO_ID
[ -f "$ASSET_DIR/norm_stats.json" ] || { echo "missing $ASSET_DIR/norm_stats.json -- run scripts/prepare_assets.sh first (README step 4)"; exit 1; }
if [ "$B1K_STAGE_AUX" = "1" ]; then
  export B1K_STAGE_TABLE=${B1K_STAGE_TABLE:-$ASSET_DIR/stage_table.json}
  [ -f "$B1K_STAGE_TABLE" ] || { echo "B1K_STAGE_AUX=1 but no $B1K_STAGE_TABLE -- run scripts/prepare_assets.sh (README step 4)"; exit 1; }
fi
NGPU=$(nvidia-smi -L | wc -l)
(( BATCH_SIZE % NGPU == 0 )) || { echo "BATCH_SIZE $BATCH_SIZE is not divisible by the $NGPU visible GPUs"; exit 1; }

# warmup-stable-decay schedule: constant peak LR until STABLE_END, then linear decay to the end
STABLE_END=$(( TOTAL_STEPS * (100 - COOLDOWN_PCT) / 100 ))
export B1K_WSD="$STABLE_END,$TOTAL_STEPS"
(( WARMUP_STEPS <= STABLE_END && STABLE_END < TOTAL_STEPS )) || {
  echo "schedule impossible: need WARMUP_STEPS ($WARMUP_STEPS) <= stable end ($STABLE_END) < TOTAL_STEPS ($TOTAL_STEPS);"
  echo "raise TOTAL_STEPS, lower WARMUP_STEPS or COOLDOWN_PCT"; exit 1; }

# resume if a complete checkpoint exists (a step directory with params/), otherwise start a fresh run
RUN_DIR=$CKPT_DIR/pi05_b1k/$EXP_NAME
if find "$RUN_DIR" -mindepth 2 -maxdepth 2 -type d -name params 2>/dev/null | grep -qE '/[0-9]+/params$'; then
  MODE=--resume
else
  MODE=--overwrite
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true"   # +2.5% throughput, measured
export WANDB_MODE=${WANDB_MODE:-offline}                                             # 'online' to stream to W&B

echo "$(date) | $NGPU GPUs | batch $BATCH_SIZE ($(( BATCH_SIZE / NGPU ))/GPU) | FSDP $FSDP_DEVICES | steps $TOTAL_STEPS | WSD $B1K_WSD | $MODE"
cd "$OPENPI"
exec uv run python scripts/b1k/train_b1k.py pi05_b1k \
  --exp_name="$EXP_NAME" $MODE \
  --batch_size="$BATCH_SIZE" --num_train_steps="$TOTAL_STEPS" --save_interval="$SAVE_INTERVAL" \
  --fsdp_devices="$FSDP_DEVICES" --num_workers="$NUM_WORKERS" \
  --assets_base_dir="$ASSETS_DIR" --checkpoint_base_dir="$CKPT_DIR" \
  --lr-schedule.warmup-steps="$WARMUP_STEPS" --lr-schedule.peak-lr="$PEAK_LR" "${@:2}"
