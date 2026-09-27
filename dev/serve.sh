#!/usr/bin/env bash
# The engine. Run from the repo root, so its store is the default ./.fluksio and
# there is no data-dir to pass. Concurrency comes from the flags: the data-dir `env`
# file does not take effect for them (see NOTEPAD.md). DEVICES is the JAX CPU
# device count per worker, over which jaqsi splits each circuit batch; keep
# RUNS x DEVICES near the core count, and RUNS within memory (D115).
set -u
cd "$(dirname "$0")/.."
exec env JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES="${DEVICES:-1}" uv run fluksio serve --port "${PORT:-8765}" \
  --max-runs "${RUNS:-10}" --max-cascades "${RUNS:-10}" --max-workers "${RUNS:-10}"
