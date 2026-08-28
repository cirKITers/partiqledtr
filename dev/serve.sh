#!/usr/bin/env bash
# The engine. Run from the repo root, so its store is the default ./.fluksio and
# there is no data-dir to pass. Concurrency comes from the flags: the data-dir `env`
# file does not take effect for them (see NOTEPAD.md).
set -u
cd "$(dirname "$0")/.."
exec env JAX_PLATFORMS=cpu uv run fluksio serve --port "${PORT:-8765}" \
  --max-runs "${RUNS:-10}" --max-cascades "${RUNS:-10}" --max-workers "${RUNS:-10}"
