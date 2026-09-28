# s2 — Quantum-arm expressivity

## Question

Do greater depth, a richer encoding spectrum, or a different bond structure
close the gap between the quantum arm and the classical GNN?

## Method

The study comprises 280 training runs on a 30,000-event dataset: 10 seeds per
cell and 40 epochs. Each arm changes one axis from the baseline (`XY_Brickwork`,
Hamming weights, two layers); trainable frequencies remain disabled.

| arm | variable | test |
| --- | --- | --- |
| A | `n_layers` ∈ {2, 4, 8, 16} | depth |
| B | `enc_weights` × `enc_reupload` | encoding spectrum |
| C | four `ansatz` choices | bond structure |

## Findings

- None of the three changes closes the gap on known topologies. The classical
  GNN reaches 0.949 accuracy and 0.731 Perfect-LCAG; the best quantum cell
  reaches 0.453 and 0.087.
- Eightfold greater frequency support through depth adds 0.063 accuracy. Every
  dissociated encoding scores at or below the Hamming baseline. Single-qubit Z
  also lies outside the DLA of every XY arm, preventing the preconditioner's
  effect on the encoded state from reaching the loss through this readout.
- `XY_Ring` is floor-free and partition-respecting. At equal parameter count,
  it improves Perfect-LCAG by 27% and valid trees by 36% over the arm it
  replaces. Study s3 tests an in-algebra readout.

## Reproduce

```sh
dev/serve.sh
dev/s2-expressivity/sweep.sh 10 10
python dev/s2-expressivity/run.py --report
python dev/s2-expressivity/figures.py
dev/s2-expressivity/export.sh
```

The study's `data/`, `results/`, `figures/`, and `logs/` directories are
reproducible and gitignored. Without `--fluksio`, `run.py --arm c` runs the same
training code in process.
