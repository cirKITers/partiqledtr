# Findings

ROADMAP phase 4b, complete. 280 training runs: every cell at **10 seeds**, 40
epochs, on the 30k-event dataset (8000 train / 6000 val / 16000 test, 30 topologies,
`max_depth=4`). Read on the **known-topology subset** -- the test split is 94%
unseen topologies by construction, so an overall number is dominated by a subset
nothing solves.

`RESEARCH.md` has the argument and the measurement history, `DECISIONS.md` why each
choice was made, `NOTEPAD.md` what the tooling cost. This file is the claims.
Figures and the data behind them are in `figures/`.

## 1. The task is learnable; the quantum arm does not learn it

| arm | params | train loss | acc known | Perfect-LCAG | valid tree |
| --- | --- | --- | --- | --- | --- |
| classical GNN, `dim=64` | 129 028 | 0.080 | **0.949** | **0.731** | 0.781 |
| best quantum cell (`XY_AllPairs`) | 106 | 0.772 | 0.453 | 0.087 | 0.487 |
| MLP-only control | 36 | 0.822 | 0.402 | 0.000 | 0.155 |

None of phase 4b's three axes closes that gap. On *unseen* topologies the GNN scores
0.401 against a majority-class rate of about 0.40: it memorises topologies and does
not generalise at all. That gap is the more interesting open problem, and it is not
a quantum-versus-classical question.

## 2. Two mechanisms pin the encoded distribution -- and what they pin is the *spread*

*(figure 2, `figures/fig2_frontend_channel.csv`)*

A trained front end is the only thing in this architecture that can move the encoded
angle distribution. What varies across arms is not how far it moves on average but
**how reproducibly** -- mean standard deviation of the end purity over 10 seeds, in
units of the arm's own uniform-prior mean:

| what limits it | mean sd | cells |
| --- | --- | --- |
| nothing -- floor-free, Hamming weights | **0.529** | `ham-dia`, `Ring`, `ham-cyc` |
| the encoding -- floor-free, dissociated weights | 0.088 | `ter-dia`, `bin-cyc`, `ter-cyc`, `pair-cyc` |
| the algebra -- floored (`d_Z > 0`) | 0.040 | `AllPairs`, `C19` |

The phase-4 baseline `ham-dia` starts at 1.61 and its ten seeds end anywhere in
**[0.01, 1.91]** -- four land at or below 0.14, an annihilated state, while two end
*above* where they started. `Circuit_19` returns exactly 1.000 on all ten, because
`P_g = 2^n - 1` for every pure state and no front end can change it.

So the front end on a floor-free Hamming arm is not a preconditioner but an
**unconstrained perturbation**: it can destroy the encoded state or amplify it, and
which one happens is the seed. A dissociated encoding suppresses that by 6x and a
DLA floor by 13x. Both suppressions are predicted; the manuscript names the second
and our own arm-B analysis names the first.

**Why the movement does not reach the loss** is a property of our readout, not of the
task. The variance law is `Var = P_g(rho) P_g(O) / dim g`, which needs the observable
inside the algebra; ours is per-qubit Pauli-Z, and single-qubit Z is **not** in the DLA
of any XY arm (0/4 on `XY_Brickwork`, `XY_Ring` and `XY_AllPairs`; only `Circuit_19`,
saturating `su(2^n)`, contains it). So `P_g(O) = 0` exactly on the floor-free arms, and
the front end's effect on the encoded state falls in the sector the theory says "only
shifts the mean". The manuscript's own isolated experiment reads `XX + YY` on a bond for
the off-diagonal arm and Pauli-Z only for the matchgate arm, where `d_Z = n > 0` puts Z
in the algebra — both of its arms satisfy the premise, ours satisfies it on `Circuit_19`
alone. Fixing that is ROADMAP phase 4c item 1, and until it is fixed §2 should be read as
a statement about the encoded state, not about trainability.

This supersedes the mean-shift reading in an earlier draft. At 3 seeds the same
cell looked like a mild, coherent decrease (1.61 -> 1.47); the mean was hiding a
190-fold range. It is also the sharpened form of `RESEARCH.md` §8's seed-dependent
total-variation direction, now quantified on a second observable.

## 3. Depth is not the bottleneck

| `n_layers` | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- |
| train loss | 0.795 | 0.777 | 0.772 | **0.770** |
| acc known | 0.396 | 0.433 | 0.441 | **0.459** |
| Perfect-LCAG | 0.060 | **0.084** | 0.081 | 0.082 |

Accuracy rises monotonically with depth and Perfect-LCAG saturates after `L = 4`.
Eight times the frequency support and 3.4x the parameters buy +0.063 accuracy
against a gap of 0.49 to the GNN. The encoded purity is identical across all four
(1.61 `mu_n`), so this isolates expressivity from trainability -- and expressivity
is not what is missing.

*(A 3-seed draft reported a peak at `L = 8` reversing at 16. That reversal was
noise; the monotone rise is what 10 seeds show.)*

Practical note: the *step* cost is flat in depth (2.5 ms at `L=2`, 2.2 ms at 16) but
the *compile* cost is not -- 200 s per run at `L=2` against 6800 s at 16, because
the circuit is unrolled. Depth is cheap per gradient and expensive per experiment.

## 4. Richer spectra do not help the task, and preconditioning is a contraction

Encoding weights move the encoded g-purity over a 5.7x range and move the task score
*down*. Every dissociated cell scores at or below the Hamming baseline:

| cell | `P/mu_n` | acc known | Perfect-LCAG |
| --- | --- | --- | --- |
| `hamming-diagonal` (phase-4 baseline) | 1.61 | 0.396 | 0.060 |
| `ternary_pair-cyclic` | 0.99 | **0.408** | 0.056 |
| `hamming-cyclic` | 1.04 | 0.386 | 0.071 |
| `binary-cyclic` | 0.95 | 0.379 | 0.055 |
| `ternary-cyclic` | 1.02 | 0.332 | 0.038 |
| `ternary-diagonal` | 0.84 | 0.318 | 0.024 |

The mechanism is confirmed and its sign is not what the ROADMAP predicted.
Spectral preconditioning drives `E[P_g]` **to** `mu_n`, from whichever side the raw
data sits:

- the clustered `legacy` encoding sits at 0.28 `mu_n` and ternary weights lift it to
  0.55 -- the manuscript's rescue, reproduced on real kinematics;
- the project's own `pair_polar` encoding sits at **1.61 `mu_n`**, and every richer
  encoding drags it *down* toward 1.0.

Preconditioning is therefore a rescue only when the input distribution is worse than
uniform. `pair_polar` is better than uniform, because `E >= |p_z|` confines the alpha
angles near `pi/2` -- the favourable RY point -- so the theory's headline
intervention has nothing to offer the encoding this project uses.

And it costs. On the clustered arm, doubling the purity takes Perfect-LCAG from
0.081 to **0.010** and the valid-tree rate from 0.704 to 0.212. The purity rescue is
real, reproducible, and negative for the task.

## 5. A partition-respecting ansatz exists that is still input-distribution sensitive

The ROADMAP expected two criteria to conflict -- respecting the two-particle
partition, and keeping an algebra on which the input distribution still decides. At
`n = 4` they do not.

| ansatz | bonds | `dim_g` | `d_Z` | swap-inv. | params | acc known | Perfect-LCAG | valid tree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `XY_Brickwork` | 01, 12, 23 | 12 | 0 | **no** | 70 | 0.396 | 0.060 | 0.373 |
| **`XY_Ring`** | 01, 23, 02, 13 | 24 | 0 | yes | 70 | 0.386 | **0.076** | **0.507** |
| `XY_AllPairs` | all six | 60 | 6 | yes | 106 | **0.453** | 0.087 | 0.487 |
| `Circuit_19` | CRX ring | 255 | 15 | yes | 106 | 0.417 | 0.086 | 0.417 |

`XY_Ring` is `Topology.bricks(offset=0)` together with `Topology.stairs(span=2)`:
the 4-cycle `0-1-3-2-0`. An even cycle is bipartite, which is the manuscript's own
criterion for `d_Z = 0`, at twice the DLA dimension of the arm it replaces. Tying
the intra-particle bonds (`Block(shared=True)`) makes it *exactly* equivariant under
the endpoint swap, so the output averaging of D35 becomes a no-op rather than a
patch. `XY_AllPairs` adds the triangles and the floor returns at `d_Z = 6`, which is
what let `Matchgate` be retired.

**At equal parameter count `XY_Ring` gives +27% Perfect-LCAG and +36% valid trees
over the arm it replaces, with per-element accuracy slightly lower.** That split is
the signature a symmetry gain should have: it buys *structural* correctness, not
more correct cells. It also beats the universal `Circuit_19` on valid trees with
two-thirds the parameters.

## 6. The unflattening classification is a safety criterion, not a selection criterion

Confirmed with a parameter-free hit: `Circuit_19` returns `P_g = 15.0000` exactly and
`Var = P_g / dim_g = 15/255 = 0.0588 = 1/(2^n + 1)` to four digits. The `d_Z`
certificate separates the arms cleanly -- floor-free at 1.60-1.61 `mu_n`, floored at
1.07 and 1.00.

But it does not rank them for the task, **and at `n = 4` it cannot**: the predicted
gradient variance spans only 1.85x across all four arms (0.109 -> 0.059) and bottoms
out at a perfectly trainable 1/17. Nothing here is untrainable, so a certificate
predicting trainability predicts nothing about performance -- and accuracy is in fact
highest on the two lowest-variance arms.

It is also blind to the difference that mattered: `XY_Brickwork` and `XY_Ring` agree
on every unflattening descriptor and differ only in bond geometry relative to the
physical partition. Conversely, at matched capacity a 4x change in `dim_g`
(`XY_AllPairs` against `Circuit_19`) moves accuracy by 3%.

**The tension worth writing up:** the arms that perform best are exactly the ones on
which the phase-4 mechanism is unobservable by construction -- `XY_AllPairs` and
`Circuit_19` lead, and both are input-distribution indifferent, provably so for
`Circuit_19`. `XY_Ring` is the only arm that is both floor-free and competitive,
which is an argument for it the theory cannot supply for itself.

## Standing caveats

- **Capacity is confounded with algebra** in §5: `XY_AllPairs` and `Circuit_19` carry
  106 parameters against 70. Within matched pairs the DLA differences are large and
  the score differences small, which is the point of §6 -- but the cross-pair
  comparison is not clean.
- **`n = 4` is a design choice** (D24), not a constraint, and it is where the
  theory's discriminating variable has no range. Scaling the constellation is the
  clean test of §6 and the obvious next experiment.
- **The dataset is one seed.** All 280 runs share `generate --seed 0`; seed variation
  is over model initialisation and batching, not over the data. A second dataset
  would test whether §5's structural gain survives resampling the topologies.
- **The generalisation probe has a ceiling of 85 distinct tree shapes** at
  `max_depth = 4`, so the known/unknown split cannot be widened at this depth
  whatever the event count.
- **The readout is outside the algebra on the floor-free arms**, so the variance law
  does not cover our loss (see §2). This does not affect §4, which is about the encoded
  state alone, nor §5's task numbers, but it does mean no result here tests the
  *trainability* half of the theory.
- **The purity observable was measured on a biased subset until 2026-08-27**: splits
  are ordered by topology, so the first 64 validation events are one topology at one
  multiplicity, which erased a 1.8x encoding effect. Fixed (D105); every number here
  post-dates it. Figures in `RESEARCH.md` §7-8 do not.
