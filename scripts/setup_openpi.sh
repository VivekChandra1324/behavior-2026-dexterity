#!/bin/bash
# Set up the training code: the BEHAVIOR fork of openpi at the exact commit we trained on, plus our overlay.
#
#   bash scripts/setup_openpi.sh [target_dir]        (default: ./openpi-b1k)
#
# Afterwards:  cd <target_dir> && GIT_LFS_SKIP_SMUDGE=1 uv sync   (installs JAX/CUDA, the lerobot fork, ...)
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
DST=${1:-openpi-b1k}
BASE=$(cat "$HERE/openpi/BASE_COMMIT")

if [ ! -d "$DST/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q -b behavior https://github.com/wensi-ai/openpi.git "$DST"
fi
git -C "$DST" fetch -q origin "$BASE" 2>/dev/null || true
git -C "$DST" checkout -q "$BASE"
git -C "$DST" checkout -q -- .                      # start from a clean tree at the base commit
cp -r "$HERE/openpi/overlay/." "$DST/"              # our modified files (see openpi/CHANGES.md)
echo "openpi at $BASE + overlay -> $DST"
git -C "$DST" status --short
echo
echo "Next: cd $DST && GIT_LFS_SKIP_SMUDGE=1 uv sync"
