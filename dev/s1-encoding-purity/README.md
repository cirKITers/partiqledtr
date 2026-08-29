# s1 — Encoding purity, before anything trains

**Question.** Does this task's input distribution land in the barren regime at all,
and can a richer encoding spectrum move it there or out of it?

Spectral preconditioning is a property of *data plus encoding*, not of a trained
model, so the whole question is answerable without fitting anything. That is what
makes this study worth keeping separate: a training run can then confirm the answer
rather than be the only evidence for it (`docs/DECISIONS.md` D97).

## What it found

- **The encoding decides the regime.** The project's own `pair_polar` map sits at
  1.61 `mu_n`; partiqlegan's `p*E*pi` product sits at 0.28 with 85% of edges below
  threshold. Only the legacy encoding reaches the clustered regime, which is why it
  is now a deliberate control arm rather than a baseline to beat.
- **Preconditioning is a contraction toward `mu_n`, not a rescue.** It lifts the
  clustered arm 0.28 → 0.55 and drags `pair_polar` 1.61 → 1.0. Richer spectra help
  only where the input distribution is worse than uniform.
- `ternary_pair-cyclic` is the only weight cell that is both dissociated **and**
  invariant under the endpoint swap — measured by exhaustion, not asserted (D100).

`docs/RESEARCH.md` §1 and §10 carry the argument and the tables.

## How to re-run it

The measurement itself is a flow node, so it is versioned and cached rather than
living here: `encoding_report` runs inside `generate`, and `encoding_cells` and
`arm_report` inside `characterize`. This folder holds the renderer, and its
gitignored `results/` the exported reports of the runs the tables above quote.

```sh
fluksio run generate --sync partiqledtr --seed 0 --wait    # or reuse a finished run
python dev/s1-encoding-purity/encoding_table.py <generate-run-id>

fluksio run characterize --sync partiqledtr --wait         # the prediction + arm table
```
