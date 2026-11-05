#!/bin/bash
# Serve a trained checkpoint to the BEHAVIOR-1K evaluator, with the inference settings of our submission.
#
#   bash scripts/serve.sh <openpi_dir> <checkpoint_dir> [port]
#
# <openpi_dir>      set up by scripts/setup_openpi.sh (README step 1)
# <checkpoint_dir>  holds params/ and assets/ -- your own <CKPT_DIR>/pi05_b1k/<EXP_NAME>/<step>, or ours:
#                   hf download VivekChandra/pi05-behavior-2026-dexterity --local-dir $WORK/pi05-dexterity
#                   -> $WORK/pi05-dexterity/checkpoint
#
# Same command as the entrypoint of the published Docker image (ghcr.io/vivekchandra1324/b1k-pi05-policy:135199):
# one server for all 100 tasks (the prompt follows each request's task_id), the official WebSocket policy protocol,
# GET /healthz for health checks. Inference settings: replan every 20 steps from a 26-step chunk compressed to 20
# steps (--action_horizon 26 --compress_from 26 --compress_to 20) and the gripper re-open rule
# (serving/gripper_table.json, rebuilt by scripts/derive_gripper_table.py). Optional: B1K_TASK=<task_name> pins one task.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
OPENPI=$(cd "${1:?usage: serve.sh <openpi_dir> <checkpoint_dir> [port]}" && pwd)
CKPT=$(cd "${2:?usage: serve.sh <openpi_dir> <checkpoint_dir> [port]}" && pwd)
PORT=${3:-8000}
[ -d "$CKPT/params" ] || { echo "no params/ in $CKPT -- pass the directory that holds params/ and assets/"; exit 1; }
source "$HERE/serving/serve.env"
# as in the image: allocate GPU memory on demand (~8.5 GB) so the simulator can share the GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

cd "$OPENPI"
exec uv run python scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task "b1k/${B1K_TASK:-auto}" \
  --repo_id behavior-1k/2026-challenge-rgb224 --port "$PORT" \
  --action_horizon 26 --compress_from 26 --compress_to 20 --gripper_table "$HERE/serving/gripper_table.json" \
  --policy.config pi05_b1k --policy.dir "$CKPT"
