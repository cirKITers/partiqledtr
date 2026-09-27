# s4 — scaling: the graph trichotomy at n = 6 (ROADMAP phase 6)

Does the unflattening classification acquire range once the register grows, and
does a provably hard floor-free arm train at moderate `n`? Three graph arms over
the same intra-particle chains, at three qubits per particle
(`pi = (0 3)(1 4)(2 5)`), certified before training (`tests/test_analysis.py`):

| arm | extra bonds | dim_g/4095 | d_Z | role |
| --- | --- | --- | --- | --- |
| `XY_Cycle` | rungs 03, 25 | 60 | 0 | floor-free tractable (poly) |
| `XY_Ladder` | rungs 03, 14, 25 | 510 | 0 | **floor-free hard** (degree-3 bipartite, encoded-universal) |
| `XY_OddChord` | + chords 02, 35 | 1020 | 30 | floored hard control |

crossed with `preconditioner` in {none, mlp}, at the phase-4c configuration
(`n_channels=4`, `lr_qfm=1e-2`, rest 1e-3, 40 epochs), on the s2 export of
generate run `1787760161002-8bde9189` — the same data as s2/s3, so the register
and the chart are the only things that moved.

**The chart.** The third per-particle angle is gated, not tuned: the three
`pair_polar_*` candidates were priced in g-purity on real kinematics before any
training (`run.py --encodings`, `results/encoding.json`, 2026-08-31):

| chart | cycle P/mu_n | ladder P/mu_n |
| --- | --- | --- |
| **`pair_polar_boost`** (atan2(\|p\|, m)) | **1.87** | **1.43** |
| `pair_polar_theta` (atan2(p_T, p_z)) | 1.56 | 1.30 |
| `pair_polar_mass` (atan2(m, E)) | 1.01 | 1.05 |

`pair_polar_boost` wins — highest above the prior mean on both floor-free arms,
0% of edges below the acceptance threshold. Every chart sits at or above `mu_n`:
the FSPs are not relativistic enough to cluster even the mass chart, so this
dataset offers no clustered three-angle arm.

**Scope (user decision 2026-08-31).** The clustered `legacy` axis is dropped, so
the annihilation/rescue prediction (ROADMAP phase 6, prediction 2) is deferred.
What the grid tests: prediction 1 (certificate range — the variance scale spans
~1.2e-1 to ~7e-3 across the arms), prediction 3 (the hard ladder trains at
`n = 6`), prediction 4 (the floored control shows no purity-loss coupling — on
this favourable chart the floor-free arms are expected to show none either,
mirroring RESEARCH §15's `pair_polar` rows). Smoke-first: 6 cells x 3 seeds, in
process; topped up to 5 seeds as versioned engine runs (user, 2026-09-27): seeds
3-4 run through the engine, seeds 0-2 are imported from the in-process records
(D114). The engine runs two cells at a time, each over eight CPU devices (D115).

```
RUNS=2 DEVICES=8 MAX_RSS=2048 ../serve.sh  # the engine (D115)
python run.py --encodings                  # the chart gate (no training)
python run.py --gate                       # one worst-case cell, cost projection
python run.py --import-inprocess           # the in-process seeds 0-2, into the engine
python run.py --fluksio                    # the grid, 6 cells x 5 seeds, versioned
python run.py --report                     # tables + correlations
python summary.py                          # results/summary.csv + figures/summary.png
```

Results land as one json per block in `results/`; the trace keys, the
first-difference correlation analysis and the summary figure match s3, so the
two studies read the same way.

## Findings (smoke grid, 3 seeds, 2026-09-02)

The register pays on every cell (+0.04-0.07 known accuracy over the `n = 4`
`pair_polar` rows of s3), the hard ladder trains at its capped variance, the
certificate range doubled and stays anti-correlated with accuracy, and the
purity-loss coupling is silent on the favourable chart. `RESEARCH.md` §16.

## Findings (5 seeds, versioned, 2026-09-27)

§16 holds: every cell mean moved by at most 0.005 (known accuracy, none/mlp:
cycle 0.552/0.562, ladder 0.562/0.578, odd-chord 0.577/**0.589**). Two of its
directions are now results. Accuracy rises with `dim_g` (Spearman 0.81 none,
0.74 mlp over 15 runs each, p <= 0.002), capacity-confounded as before. The
preconditioner adds +0.013 paired by seed (11/15 positive, p = 0.002; ladder
alone 5/5), but it contracts purity toward `mu_n` and the coupling stays
absent (14/15 positive r(dP,dL)), so the gain is not the unflattening rescue.
Validation loss bottoms out near epoch 10 in every cell; accuracies are read at
epoch 40, as in s3. Full reading, the §16 sign-count correction and the s5
lifts on the new baselines: `RESEARCH.md` §20.
