# s4 — Scaling the graph trichotomy to n = 6

## Question

Does a larger register make the unflattening classification more distinct, and
can a provably hard floor-free arm train at moderate size?

## Method

Three graph arms share intra-particle chains with three qubits per particle
(`pi = (0 3)(1 4)(2 5)`). Each is certified before training and tested with
`none` and `mlp` preconditioners at five seeds and 40 epochs. Runs use four
channels, `lr_qfm=1e-2`, other learning rates of `1e-3`, and the s2/s3 dataset.

| arm | extra bonds | `dim_g` / 4095 | `d_Z` | role |
| --- | --- | --- | --- | --- |
| `XY_Cycle` | rungs 03, 25 | 60 | 0 | tractable floor-free arm |
| `XY_Ladder` | rungs 03, 14, 25 | 510 | 0 | hard floor-free arm |
| `XY_OddChord` | ladder + chords 02, 35 | 1020 | 30 | floored control |

The input chart is `pair_polar_boost`, selected by comparing the g-purity of
three candidate third angles on real kinematics before training.

## Findings

- `pair_polar_boost` reaches 1.87 `mu_n` on cycle and 1.43 `mu_n` on ladder,
  with no edges below the acceptance threshold. None of the three-angle charts
  clusters this dataset, so the clustered-input rescue prediction remains
  untested here.
- The larger register improves known-topology accuracy by 0.04–0.07 over s3's
  `n = 4` `pair_polar` cells, and the hard ladder trains at its capped variance.
  The certificate range doubles, with variance scales from ~1.2e-1 to ~7e-3.
  Accuracy rises with `dim_g` (Spearman 0.81 for none, 0.74 for mlp; 15 runs
  each, `p <= 0.002`), although capacity is a confound.
- Mean known-topology accuracy for none/mlp is 0.552/0.562 on cycle,
  0.562/0.578 on ladder, and 0.577/0.589 on odd-chord. The preconditioner
  adds 0.013 accuracy paired by seed (11/15 positive, p = 0.002; ladder 5/5),
  but contracts purity toward `mu_n` without purity-loss coupling (14/15
  positive `r(dP,dL)`). This gain does not appear to be unflattening rescue.
  Validation loss reaches its minimum near epoch 10; accuracy is measured at
  epoch 40.

## Reproduce

Requires the s2 dataset export. From the repository root:

```sh
RUNS=2 DEVICES=8 MAX_RSS=2048 dev/serve.sh
python dev/s4-scaling/run.py --encodings
python dev/s4-scaling/run.py --fluksio
python dev/s4-scaling/run.py --report
python dev/s4-scaling/summary.py
```

Results are stored in the gitignored `results/` and `figures/` directories.
