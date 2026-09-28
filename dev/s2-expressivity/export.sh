#!/usr/bin/env bash
# The engine's own view of the study, via `fluksio export` (needs fluksio >= 0.1.5
# for dotted paths into json outputs).
#
# Both land in this study's own `figures/`, next to what `figures.py` draws:
#
#   runs.csv    one row per run: inputs as `param.*` columns, the final numbers as
#               `metric.*` columns selected by dotted path, and `code_digest` --
#               the engine's hash of the code it ran
#   curves.csv  tidy (run, name, step, ts, value): the per-epoch series
#
# Metric names are flow-qualified: `train.train_loss`, not `train_loss`.
# `fluksio export metrics --flow train --status ok --list` names what is available.
#
# This is the whole arm-comparison table. `dev/s2-expressivity/figures.py` still reads
# `results/arm_*.json` because those also hold the pre-engine in-process runs;
# once every cell has been re-run through the engine, this file is the source.
#
# SINCE excludes runs that predate the purity-subset fix, whose recorded
# observable was measured on a biased subset and is not comparable. Runs before it
# are still in the engine and still legitimate as history; they are simply not this
# table's rows.
set -u
cd "$(dirname "$0")/../.."
SINCE=${SINCE:-2026-08-27T10:00}
export FLUKSIO_URL=${FLUKSIO_URL:-http://127.0.0.1:8765}
export FLUKSIO_TOKEN=${FLUKSIO_TOKEN:-$(python3 -c "import json;print(json.load(open('.fluksio/client.json'))['token'])")}
mkdir -p dev/s2-expressivity/figures

# `frontend` is what this study's 280 runs recorded; the axis was renamed to
# `preconditioner` afterwards, and the engine keeps each run's own name.
PARAMS=model,ansatz,enc_weights,enc_reupload,n_layers,preconditioner,frontend,encoding,angle_map,seed,epochs
METRICS=final_metrics.train_loss,final_metrics.n_params,final_metrics.g_purity_initial
METRICS=$METRICS,final_metrics.val_g_purity,final_metrics.config.ansatz
METRICS=$METRICS,test_metrics.known.accuracy,test_metrics.known.perfect
METRICS=$METRICS,test_metrics.known.valid_tree_strict,test_metrics.unknown.accuracy
METRICS=$METRICS,dla_report.dim_g,dla_report.n_diag_words

uv run fluksio export runs --flow train --status ok --since "$SINCE" \
  --params "$PARAMS" --metrics "$METRICS" --format csv -o dev/s2-expressivity/figures/runs.csv
uv run fluksio export metrics --flow train --status ok --since "$SINCE" \
  --name train.train_loss,train.val_loss,train.val_accuracy,train.val_perfect,train.g_purity \
  --format csv -o dev/s2-expressivity/figures/curves.csv
wc -l dev/s2-expressivity/figures/runs.csv dev/s2-expressivity/figures/curves.csv
