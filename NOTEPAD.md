# Notepad

Small intermediate bugs/features land here (see README note).

## qml-essentials

- **`Model(circuit_type=...)` is annotated for an instance but wants a class.** The
  hint is `Union[str, Circuit]`, while `Model.__init__` does `self.pqc =
  circuit_type()` -- so passing a `Circuit` *subclass*, which is what the
  `Ansaetze` members are and what a project-local ansatz has to be, is correct at
  runtime and a type error under `ty`. `Union[str, type[Circuit]]` would fix it.
  Suppressed at the one call site in `models/qfm.py`.


## phasespace-jax


## Fluksio

Everything reported here is fixed as of **0.1.6**. Kept as a record of what was
tested and how, in case any of it regresses.

**Fixed, each verified by re-running the thing that failed.**

- ~~Concurrent nodes oversubscribe the CPU and starve the engine.~~ `fair_share_env`
  caps thread vars per worker; `flavors` and node-declared `resources` go further.
- ~~`Client()`'s 30 s read timeout is fatal.~~ 120 s, `retries=3` on idempotent GETs.
- ~~Concurrent runs stay at 4.~~ `serve --max-runs/--max-cascades/--max-workers`;
  `--max-runs 10` gives ten immediately.
- ~~`events()` cannot filter by run.~~ `run=` exists.
- ~~No way to get data out except reading the sqlite file.~~ `fluksio export`, with
  both shapes: `runs` (one row per run, inputs flattened to `param.*`, run id in
  every row, `--since`) and `metrics` (tidy `run, name, step, ts, value`, `--stride`).
  `experiments/export.sh` builds the whole arm-comparison table from two commands.
- ~~`export runs --metrics` cannot reach into a json output.~~ Dotted paths work:
  `--metrics final_metrics.train_loss,test_metrics.known.perfect` gives real columns.
- ~~"nothing matched" said nothing about what would match.~~ Points at `--list`.
- ~~A node's cache fingerprint does not cover the modules it calls into~~, and
  ~~a run's stamp does not identify the code it ran.~~ One cause, fixed together by
  `sdk/imports.py`: it walks the imports out from a node's module and stops at
  anything installed or standard, so the digest names the project's own code and
  nothing else. Tested in all three directions on `fit`:

  | change | digest | `sync` |
  | --- | --- | --- |
  | nothing, three times | `57f1414b` | `unchanged`, `unchanged`, `unchanged` |
  | `train_model`, which `fit` calls into | -> `ed404c67` | `updated (flow, fit, evaluate) — published` |
  | `experiments/figures.py`, which no node reaches | `57f1414b` | `unchanged` |

  Necessary *and* sufficient, and idempotent. `reached(fit)` returns the 17
  `partiqledtr.*` modules and nothing from `experiments/` or site-packages.
  `cache=False` on `fit` (D93) can come off once this has a release behind it.
- ~~The version string stayed put across patches.~~ 0.1.4 -> 0.1.5 -> 0.1.6.

**One operational note, not a bug.** The client is upgraded by `uv sync` while the
engine keeps running the version it started with, so a fix can look absent when it
is only unloaded -- twice here, and the second time it looked like a *regression*
(`sync` reporting `unchanged` for a real change) that vanished on restarting the
engine. `serve` printing its version at startup makes it diagnosable, and
`fluksio status` naming the engine's version next to the client's would make it
obvious.

Smaller, still open: `fluksio runs` prints every input inline, so a flow taking a
large `json` input (here `dataset_meta`, a few kB) is unreadable in a terminal.
Truncating, or printing only the inputs that differ from the flow's defaults --
which `export runs --params` already computes -- would fix it.
