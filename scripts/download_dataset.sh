#!/bin/bash
# Download the official 2026 BEHAVIOR Challenge demonstrations (LeRobot v3 format) from Hugging Face.
#
#   bash scripts/download_dataset.sh /data/2026-challenge-demos            # RGB + low-dim (what training needs)
#   INCLUDE_DEPTH=1 bash scripts/download_dataset.sh /data/2026-challenge-demos   # also the depth videos
#
# Size: ~1 TB without depth, ~3.3 TB with it. Depth is NOT used by this pipeline (and neither baseline reads it).
# Resumable: re-run the same command to fill any gaps. Needs `pip install "huggingface_hub[cli,hf_xet]"` and, if
# the dataset requires it, `hf auth login`.
set -euo pipefail
ROOT=${1:?usage: download_dataset.sh <dataset_root>}
REPO=behavior-1k/2026-challenge-demos
mkdir -p "$ROOT"

# 1. metadata first (small): info.json, episodes, tasks, stats
hf download $REPO --repo-type dataset --local-dir "$ROOT" --include "meta/**" --include "README.md" --max-workers 4

# 2. low-dimensional data (state, actions, timestamps): 200 parquet files
hf download $REPO --repo-type dataset --local-dir "$ROOT" --include "data/**" --max-workers 16

# 3. RGB videos of the three cameras (head + two wrists)
hf download $REPO --repo-type dataset --local-dir "$ROOT" --include "videos/observation.rgb.*/**" --max-workers 16

# 4. optional: depth videos (~2.1 TB)
if [ "${INCLUDE_DEPTH:-0}" = "1" ]; then
  hf download $REPO --repo-type dataset --local-dir "$ROOT" --include "videos/observation.depth*/**" --max-workers 16
fi

echo "data chunks:  $(ls -d "$ROOT"/data/chunk-* 2>/dev/null | wc -l)  (expect 100)"
echo "video dirs:   $(ls -d "$ROOT"/videos/*/chunk-* 2>/dev/null | wc -l)  (expect 300 RGB-only, 600 with depth)"
du -sh "$ROOT"
