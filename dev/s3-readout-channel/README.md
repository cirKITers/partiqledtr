# s3: the readout channel

Does the in-algebra readout (D107) open the channel from the encoded input
distribution to the trained loss? ROADMAP phase 4c item 1's success criterion:
the g-purity trajectory and the loss become correlated *within* a run — the
thing phase 4 could not show and phase 4b showed the absence of, under a
readout whose observable purity was zero on every floor-free arm
(`FINDINGS.md` §2).

## Design

Smoke block, 10 runs: `XY_Ring` × {`none`, `mlp` preconditioner} on the
clustered `legacy` encoding, 5 seeds, 40 epochs. The clustered arm is where the
prediction is falsifiable (purity starts far below `mu_n`, so a learned
preconditioner has headroom); the `none` cells are the frozen-distribution
control. Same dataset splits as s2 (`dev/s2-expressivity/data`, generate run
`1787760161002-8bde9189`), so the readout is the only change between studies.

Headline number: Spearman correlation of the **first-differenced** g-purity and
validation-loss series per run — raw correlations of two co-trending series are
trivial. Success looks like consistently negative values (purity up, loss down)
on the mlp cells, absent on the control.

Expansion, only if the smoke block shows the correlation: `XY_Brickwork` (the
manuscript's own family), `pair_polar` inputs, and the floored `XY_AllPairs`
specificity control where the correlation must be absent.

## Run

```sh
python dev/s3-readout-channel/run.py            # 10 runs, resumable
python dev/s3-readout-channel/run.py --report   # table + correlations
python dev/s3-readout-channel/figures.py        # trajectories per cell
```

`results/` and `figures/` are gitignored; findings go to `RESEARCH.md`.

## Findings (smoke block, 2026-08-29)

The channel opens: `r(dP, dL)` Pearson is **−0.67 ± 0.18 with 5/5 seeds
negative** (circular-shift null `p <= 0.026` per seed) on the mlp cells, absent
by construction on the frozen control. The coupling lives in the first ~10
epochs; end purities converge on the `mu_n` band from both sides while
`mean_sin2` moves toward uniform. Task accuracy is unchanged (0.425 vs 0.428
known) — the purity dynamics now reach the loss, and the loss still does not
reach the task. Full reading and caveats: `RESEARCH.md` §12.

## Findings (widening smoke, `--channels 4`, 2026-08-29)

Capacity was a real bottleneck: K=4 lifts known accuracy 0.428 → **0.498 ±
0.004** (`none`; every widened seed above every narrow seed, spread collapses
10x), beating every phase-4b quantum cell. The purity-loss channel survives,
attenuated (r(dP,dL) −0.41 ± 0.27, 4/5 negative). The preconditioner now
*costs* accuracy at K=4 (0.475) while leading Perfect-LCAG (0.104) — the
remaining confound is the shared learning rate (ROADMAP 4c item 4). Decision:
the full experiment runs widened. Details: `RESEARCH.md` §13.

## Findings (optimizer smoke `--opt` + full grid `--full`, 2026-08-29/30)

The preconditioner's accuracy cost was the *circuit's* rate: `lr_qfm = 1e-2`
recovers it exactly (RESEARCH §14). The full grid at that pick (§15): the
purity–loss coupling appears on exactly the floor-free × clustered cells and
nowhere else (floored `XY_AllPairs` control: r(dP,dL) = 0.04 ± 0.06); the
preconditioner pays where the theory says it should (+0.010–0.013 acc,
+0.011–0.016 perfect) and is neutral on favourable inputs; the rescue is
partial (~1/3 of the encoding gap); and the floored arm on clustered inputs is
the best cell overall (0.563 — capacity-confounded, follow-up cells named in
§15).
