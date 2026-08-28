# s2 — Expressivity: can the quantum arm be made to fit the task?

ROADMAP phase 4b. **280 training runs**, every cell at 10 seeds, 40 epochs, on the
30k-event dataset. Three independent arms, each holding everything else at the
phase-4 configuration so only its own axis moves:

| arm | axis | question |
| --- | --- | --- |
| A | `n_layers` ∈ {2, 4, 8, 16} | is depth the bottleneck? |
| B | `enc_weights` × `enc_reupload` | is the spectrum the bottleneck? |
| C | `ansatz` ∈ 4 arms | is the bond structure the bottleneck? |

Trainable frequencies stay off in all three, so the axes remain separable.

## What it found

None of the three closes the gap. The classical GNN reaches 0.949 accuracy and
0.731 Perfect-LCAG on known topologies; the best quantum cell reaches 0.453 and
0.087. Depth buys +0.063 accuracy for 8× the frequency support; every dissociated
encoding scores at or below the Hamming baseline.

The result that reframes what is left: **the readout is outside the algebra.** The
unflattening variance law needs the observable inside the DLA, and single-qubit Z is
not in the DLA of any XY arm — so the front end's effect on the encoded state has no
channel to the loss. That is ROADMAP phase 4c item 1, and it comes before phase 5.

One arm did land: `XY_Ring` is floor-free *and* partition-respecting, and at equal
parameter count gives +27% Perfect-LCAG and +36% valid trees over the arm it
replaces.

`docs/FINDINGS.md` is the claims; `docs/RESEARCH.md` §10 the measurement history.

## How to re-run it

```sh
dev/serve.sh                                     # the engine, on ./.fluksio
dev/s2-expressivity/sweep.sh 10 10               # every arm, in order, at 10 seeds
python dev/s2-expressivity/run.py --report       # the per-arm tables
python dev/s2-expressivity/figures.py            # fig 1-3 and the csv behind fig 2
dev/s2-expressivity/export.sh                    # the engine's own view, as csv
```

`run.py` also runs in process (`--arm c`, no `--fluksio`), which is the sandbox path
rather than a fork of the flow — it calls the same `train_model` the `fit` node calls
(`docs/DECISIONS.md` D104). Everything reads and writes `results/`, `data/` and
`figures/` at the repo root, all gitignored and all reproducible from `generate`.
