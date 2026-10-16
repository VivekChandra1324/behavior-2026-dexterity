#!/bin/bash
# Step 4: normalisation statistics + stage table, written exactly where scripts/train.sh looks for them
# (<ASSETS_DIR>/pi05_b1k/<B1K_REPO_ID>/), using the same configuration file.
#
#   bash scripts/prepare_assets.sh $WORK/openpi-b1k            # ~1 h 40 min (statistics over the full dataset)
#   MAX_FRAMES=200000 bash scripts/prepare_assets.sh ...       # quicker approximation
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
OPENPI=$(cd "${1:?usage: prepare_assets.sh <openpi_dir>}" && pwd)
source "${TRAIN_ENV:-$HERE/configs/train.env}"
OUT=$ASSETS_DIR/pi05_b1k/$B1K_REPO_ID
mkdir -p "$OUT"
cd "$OPENPI"

echo "== normalisation statistics -> $OUT/norm_stats.json"
uv run python "$HERE/scripts/compute_norm_stats.py" \
    --dataset-root "$B1K_DATASET_ROOT" --repo-id "$B1K_REPO_ID" --assets-base-dir "$ASSETS_DIR" --no-video \
    ${MAX_FRAMES:+--max-frames $MAX_FRAMES}

echo "== stage table -> $OUT/stage_table.json"
uv run python "$HERE/scripts/compute_stage_table.py" --root "$B1K_DATASET_ROOT" --out "$OUT/stage_table.json"

ls -la "$OUT"
