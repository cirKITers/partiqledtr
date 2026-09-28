# s3 — The readout channel

## Question

Does an in-algebra readout connect input g-purity to the training loss, and does
that connection improve task performance?

## Method

Replace per-qubit Z with `<XX_b> + <YY_b>` on each coupling bond. Compare
`none` and `mlp` preconditioners across one- and four-channel models, clustered
`legacy` and favourable `pair_polar` inputs, and floor-free and floored XY
arms. Runs use the s2 dataset splits, five seeds, and 40 epochs. Within-run
Spearman and Pearson correlations of first differences in g-purity and
validation loss measure the channel without a shared time trend.

## Findings

- On clustered `XY_Ring` with one channel, Pearson `r(dP,dL)` is −0.67 ± 0.18
  with all five `mlp` seeds negative (circular-shift null `p <= 0.026` per
  seed). The fixed-distribution control has no purity change. Coupling is
  concentrated in the first ~10 epochs, but known-topology accuracy remains
  nearly unchanged (0.425 vs 0.428).
- Four channels raise known-topology accuracy from 0.428 to 0.498 ± 0.004
  without a preconditioner, exceeding every s2 quantum cell. The correlation
  weakens to −0.41 ± 0.27 (4/5 negative). Increasing `lr_qfm` to `1e-2` removes
  the preconditioner's accuracy cost.
- Across the full grid, coupling appears only for floor-free arms on clustered
  inputs; the floored `XY_AllPairs` control has `r(dP,dL)` = 0.04 ± 0.06. On
  clustered inputs, the preconditioner adds 0.010–0.013 accuracy and
  0.011–0.016 Perfect-LCAG, recovering about one-third of the encoding gap. The
  highest cell is floored `XY_AllPairs` on clustered inputs (0.563), although
  capacity remains a confound.

## Reproduce

```sh
python dev/s3-readout-channel/run.py
python dev/s3-readout-channel/run.py --channels 4
python dev/s3-readout-channel/run.py --opt
python dev/s3-readout-channel/run.py --full
python dev/s3-readout-channel/run.py --report
python dev/s3-readout-channel/figures.py
```

`results/` and `figures/` are gitignored.
