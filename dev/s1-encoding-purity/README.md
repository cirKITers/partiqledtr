# s1 — Encoding purity before training

## Question

Does the input distribution fall in the barren regime, and can the encoding
spectrum change that?

## Method

Measure g-purity from the data and encoding before fitting a model. The
`generate` flow produces `encoding_report`; `characterize` produces
`encoding_cells` and `arm_report` for the encoding-weight comparison.

## Findings

- **Encoding determines the regime.** The `pair_polar` map reaches 1.61 `mu_n`;
  partiqlegan's `p*E*pi` product reaches 0.28, with 85% of edges below the
  threshold. Only the legacy encoding reaches the clustered regime, making it a
  control arm.
- **Preconditioning contracts purity toward `mu_n`.** It moves the clustered arm
  from 0.28 to 0.55 and `pair_polar` from 1.61 to 1.0. Richer spectra help only
  when the input distribution is worse than uniform.
- `ternary_pair-cyclic` is the only weight cell that is both dissociated and
  invariant under endpoint swaps, as verified exhaustively.

## Reproduce

```sh
fluksio run generate --sync partiqledtr --seed 0 --wait
python dev/s1-encoding-purity/encoding_table.py <generate-run-id>
fluksio run characterize --sync partiqledtr --wait
```

The renderer and gitignored `results/` exports are in this study directory.
