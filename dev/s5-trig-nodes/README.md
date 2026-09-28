# s5 — Trigonometric node updates

## Question

Does a sine node update improve the quantum model beyond the effect of adding
classical capacity, and does the gain depend on the graph algebra?

## Method

The baseline uses linear classical components between quantum blocks; the
second block consumes the node update directly as RY angles. A boundary probe
measures the second block's encoded distribution, and linear probes compare
pair-invariant features with trigonometric features of the six input angles.

Training compares linear, wider linear, SIREN, and parameter-matched ELU node
updates on `XY_Cycle` with the boost chart, four channels, and no
preconditioner. Five additional cells complete the SIREN grid across cycle,
ladder, and odd-chord with `none` and `mlp`. Each training cell uses two seeds;
the linear s4 baselines use three.

## Findings

- The second block does not start in a clustered regime: its encoded angles
  have standard deviation 0.89 rad, mean sin² 0.43, and purity 0.92 `mu_n`,
  while the first block's chart is at 1.89 `mu_n`. Increasing `node_omega` to
  16 moves initial purity only to 1.01 `mu_n`. The distribution after training
  remains an open measurement.
- A cubic in the pair invariant `m_ij` reaches 0.495 held-out edge accuracy,
  above the best integer-frequency trigonometric probe on the six angles
  (0.471; majority baseline 0.381). Angle information saturates at total
  degree 2, and frequencies beyond the L=2 QFM box add nothing. Encoding
  `m_ij` in the algebra remains open; this probe covers the edge function only.
- On cycle, SIREN reaches 0.587 ± 0.007 accuracy, ahead of matched ELU
  (0.565 ± 0.002) and linear (0.555 ± 0.011). Increasing `node_omega` to 8
  alone lowers accuracy to 0.530. The SIREN cycle nearly matches s4's best
  cell (0.591 ± 0.012) with a 17-fold smaller DLA.
- Without a preconditioner, SIREN adds 0.032 accuracy on cycle and 0.044 on
  ladder, versus 0.006 on the floored odd-chord control. Ladder reaches
  0.602 ± 0.001; odd-chord × mlp × SIREN sets the best observed result at
  0.607 ± 0.002. Under SIREN, mlp is neutral on floor-free arms and adds
  0.029 on the floored arm, where the gain cannot be unflattening rescue.

## Reproduce

Requires the s2 dataset export. From the repository root:

```sh
python dev/s5-trig-nodes/spectrum.py
python dev/s5-trig-nodes/run.py --fluksio
python dev/s5-trig-nodes/run.py --expand --fluksio
python dev/s5-trig-nodes/run.py --report
```

Results are stored in the gitignored `results/` directory.
