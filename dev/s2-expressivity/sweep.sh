#!/usr/bin/env bash
# The expressivity arms at N seeds, through a Fluksio engine on this machine.
#
# Arms cheapest first; A last, because its deep circuits are unrolled and their
# compile cost dominates (200 s per run at n_layers=2 against 6800 s at 16).
# Everything the study reads or writes sits in its own folder, so this needs no mount.
#
#     dev/s2-expressivity/sweep.sh [seeds] [jobs]
set -u
cd "$(dirname "$0")/../.."
SEEDS=${1:-10}
JOBS=${2:-10}
DATA=1787760161002-8bde9189   # the `generate` run this study's ./data came from

export JAX_PLATFORMS=cpu
export FLUKSIO_URL=${FLUKSIO_URL:-http://127.0.0.1:8765}
export FLUKSIO_TOKEN=${FLUKSIO_TOKEN:-$(python3 -c "import json;print(json.load(open('.fluksio/client.json'))['token'])")}
# No OMP_NUM_THREADS here on purpose: fluksio's `fair_share_env` caps the thread
# vars per worker, and only for vars the operator has not already set.

for arm in c b baseline a; do
  echo "=== arm $arm starting $(date -u +%H:%M:%S) ==="
  uv run python dev/s2-expressivity/run.py --arm "$arm" --seeds "$SEEDS" --jobs "$JOBS" \
    --fluksio "$DATA" 2>&1 | grep -v "NVIDIA GPU"
  echo "=== arm $arm done $(date -u +%H:%M:%S) ==="
done
echo "=== ALL ARMS DONE $(date -u +%H:%M:%S) ==="
