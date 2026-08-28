# Research findings

What the phase 1-4 implementation measured, and what it means for the study.
Implementation reasoning lives in `DECISIONS.md`; this file is about the physics.

Status: §1 and §2 are measured on the 3200/2400/6400-event dataset (24 topologies,
8 per known/unknown group, 500 events each) and hold. The phase-4 trajectories of
§3 are still provisional -- one seed per cell -- and are the subject of the
learning-rate sweep in §7.

Numbers here postdate the phase-1-4 review; §6 records what it changed and which
earlier measurements it invalidated.

---

## 1. The task is not in the barren regime -- the old encoding was

Mean off-diagonal g-purity over sampled real edges of a generated dataset (384
training events, 4 topologies per group, `max_depth=4`), against the iid-uniform
mean `mu_4 = 0.8125` and the whitening acceptance threshold `mu_4 / 2 = 0.406`.
All three arms are scored on the *same* sampled edges, so they differ by their
encoding and nothing else:

| encoding | mean `P_g` | edges below threshold |
| --- | --- | --- |
| pair-polar, the arm in use | 1.242 | 15.4 % |
| direct `(theta, phi)` | 1.048 | 22.5 % |
| partiqlegan's `p_a * E * pi` | **0.164** | **85.4 %** |

The ROADMAP's motivating premise is that kinematic features cluster encoding
angles naturally, putting this task in exactly the regime where the unflattening
theory makes falsifiable predictions. **That holds for the old encoding and not
for either replacement.** `p * E * pi` sits far below the acceptance threshold
with the large majority of its edges collapsed; both new encodings sit above
`mu_n`.

The mechanism is a kinematic bound, not softness. Under the pair-polar map the
`(pz, E)` angle is `atan2(E, pz)`, and `E >= |p| >= |pz|` pins it inside roughly
`[pi/4, 3pi/4]` for *every* particle -- measured soft-quartile mean 1.545 against
hard-quartile 1.516, i.e. the same place. `pi/2` is the *favourable* RY-encoding
point, the one of maximal superposition. The old encoding instead multiplied two
quantities that both live in the unit interval after its max-based normalisation,
so their product concentrated near zero, which is the collapsed point.

> Provenance: these numbers replace an earlier table that was measured before the
> observable was made consistent (see §6) and whose two comparison encodings
> existed nowhere in the repository. They now come from `analysis.encoding_purity`,
> which runs as a node in the `generate` flow, so the table is reproducible from
> committed code.

### Implications

- **The encoding switch is itself the cleanest result this project has produced.**
  It quantifies, in the unflattening theory's own currency, exactly why the prior
  work's encoding was hard to train. That is a direct, falsifiable link between
  the two papers on a real HEP task -- arguably a stronger contribution than the
  rescue experiment it was meant to enable.
- **The fixed-whitening arm has little to rescue.** On real kinematics the raw arm
  already scores above `mu_n`, a Haar rotation is accepted on the first draw, and
  whitening moves the mean *down* toward `mu_n` rather than up. What remains is
  the 15 % of edges still below threshold: a sub-population effect, a weaker and
  more delicate claim than the ROADMAP anticipated. Worth deciding whether to
  pursue it, report it as a bounded negative, or re-frame the arm as a control.
- **The clustered arm is now built, so the prediction is testable again.** The
  `"legacy"` encoding plus `angle_map="legacy"` runs the prior work's own product
  encoding as a fourth input arm, deep in the collapsed regime by measurement.
  That is where the rescue experiment has data to work with, and it uses the
  reference implementation's encoding rather than a synthetic construction.

## 2. Ansatz certificates, recorded before any training

Dynamical Lie algebra of each arm at the constellation's size (`n_qubits = 4`),
computed from each ansatz's own circuit structure. `n_diag_words` counts Z-only
closure words — these have expectation 1 in the clustered limit, so their count
*is* the deterministic g-purity floor.

| arm | DLA | `dim_g` | `dim_su` | ratio | `n_diag_words` |
| --- | --- | --- | --- | --- | --- |
| `XY_Brickwork` | `so(n) (+) so(n)` | 12 | 255 | 0.047 | 0 |
| `Matchgate` | `so(2n)` | 28 | 255 | 0.110 | 4 |
| `Circuit_19` | `su(2^n)` | 255 | 255 | 1.000 | 15 |

Cross-checked: the Matchgate closure equals `n(2n-1)` for every `n` tested, and
the generators derived from the circuit reproduce upstream's `matchgate_generators`
exactly.

### Implications

- **`Circuit_19` cannot show front-end effects at all.** Its DLA saturates
  `su(2^n)`, where the g-purity is `2^n - 1` for *every* pure state. Its loss
  variance is the textbook `1/(2^n + 1)` regardless of what the input
  distribution does. Measured: exactly 15.000 on clustered inputs and 15.000 after
  whitening. This must be stated as a structural fact, not reported as an
  experimental null result — the arm is a legacy bridge to ACAT'22/CHEP'23, and
  that is all it can be.
- **"Indifference" is the wrong word for the Matchgate arm.** Clustering does not
  leave its purity unchanged; it *maximises* it. Measured 3.997 on clustered
  inputs against a uniform mean of `n - 1 + 2^-n = 3.06`, with whitening pulling
  it *down* to 3.095. The confirmed prediction is that it never collapses, so
  trainability is protected either way and a learned front end has nothing to
  rescue. Writing "indifferent" would misstate the result.
- **Only `XY_Brickwork` is a live experimental arm** for the input-distribution
  question. The other two are controls of different kinds: one floored, one
  structurally immune.

---

## 3. The phase-4 observable works; the trajectory it reports is not the predicted one

On synthetic clustered inputs (encoding angles driven near `{0, pi}`), measured
two ways -- the closed form the theory is written in, and the exact g-purity of
the statevector the circuit actually prepares:

| arm | closed form | exact | reading |
| --- | --- | --- | --- |
| `XY_Brickwork` | 0.000 | 0.000 | collapse, no floor |
| `Matchgate` | 3.997 | 3.997 | floored at `n`, never collapses |
| `Circuit_19` | 15.000 | 15.000 | exactly `2^n - 1`, input-independent |

**The two agree here, and only here.** In the clustered limit every encoding
rotation tends to the identity and `P_g` is invariant under the algebra's own
group, so the product-state closed form describes the real state. At generic
angles they diverge -- qml-essentials orders each layer ansatz-first, so the
prepared state is not a product state at all. Both are now measured and reported
(`g_purity` per epoch, `g_purity_exact` once at the end), which is what lets a
claim say which object it is about.

### And during training the trajectory does *not* go the way the earlier draft said

Re-measured under the corrected convention (§6), 15 epochs, 384 training events,
one seed, `mu_4 = 0.8125`. The epoch-0 anchor is the raw arm's level by
construction, since the front end starts as the identity:

| arm | epoch 0 | trajectory | end |
| --- | --- | --- | --- |
| `XY_Brickwork` + learned | 1.247 | dips to 0.178 by epoch 6, then recovers | 0.436 |
| legacy + learned | 0.148 | 0.23, then collapses | 0.020 |

**Neither is the predicted rescue.** On the XY arm the learned front end drives
purity *down* by a factor of seven before partially recovering, ending well below
both its start and `mu_4`. On the clustered legacy arm -- the one that is actually
in the regime the theory is about -- purity falls by an order of magnitude and
never recovers, while validation accuracy *degrades* from 0.423 to 0.230.

An earlier version of this file reported the XY arm rising from 0.714 to 0.930
across `mu_4`. That measurement was taken under the `n_layers * u` convention and
on a different dataset, and does not survive the fix; it should not be cited.

**Superseded by §7.** The sweep shows the direction is set by the learning rate,
and that the guess in the previous sentence -- that `lr=5e-3` was too *high* -- had
the sign backwards: at `lr = 1e-2` the same arms rescue. Read §7, not this table,
for what the trajectory does.

### Implications

- **The observable works and is cheap enough to stream every epoch.** What it
  reports, at this scale, is not yet the predicted rescue.
- **The fixed arms are flat by construction, not by result.** Raw and whitened
  encodings do not change during training, so their purity cannot move. The
  comparison across arms is therefore between *levels* for the fixed arms and a
  *trajectory* for the learned one -- a point to make explicitly, because a flat
  line in a figure otherwise reads as a measured null.
- **A closed-form purity is a statement about data plus encoding, not about a
  trained circuit.** Saying "measured through the actual model" of the closed-form
  series would overstate it; that phrase belongs to `g_purity_exact` alone.
- The whitening acceptance test behaves as the theory says (acceptance well above
  the Markov bound of ~1/5; immediate on real data), so the machinery is not the
  limiting factor. The limiting factor is §1: the input distribution.

## 4. Cost, and what it permits

One QFM evaluation over 3584 folded edges (batch 64, 8 particles) takes 10 ms
forward and 39 ms with gradient on CPU. Two blocks put a training step near 80 ms,
so roughly 12 s per epoch at 10k events — a 100-epoch run is about 20 minutes and
the full nine-cell arm matrix is a few CPU-hours.

**Implication:** the simulation cost that killed the original partiqlegan approach
is no longer the binding constraint. Analytic expectation values, small
shared-weight circuits and native batching bring the whole ablation matrix inside
a working day. Scaling in event size is the open question, not scaling in runs.

---

## 5. Corrections to the reference implementations

Found while re-deriving the algorithms. Two of these were themselves corrected on
review, which is why each now carries its evidence:

- **Withdrawn: partiqlegan's accuracy and Perfect-LCAG metrics are correct.** An
  earlier version of this file claimed they scored agreement between two unrelated
  booleans. They do not. Three lines above the comparison,
  `prediction = t.where(label == ignore_index, label, prediction)` forces agreement
  on every ignored cell, so `(a == b).sum() / b.sum()` reduces exactly to the
  intended masked accuracy -- verified by hand on a worked example. The code is
  obfuscated and fragile (a sibling method applies `two_child_fix` *after* that
  overwrite and does break the identity), but the published numbers need no
  re-derivation on this account. **Any comparison text drafted from the earlier
  claim must be corrected.**
- **partiqlegan's topology sampler mis-accounts the mass budget** -- confirmed. It
  subtracts the running child-mass *total* on every iteration of its child loop, so
  an extra `m0` comes off the budget each time. The direction is conservative, so
  it never produces an unphysical decay, but the effect is large: simulated over
  20 000 trees at the shipped configuration, 3-child nodes drop from ~50 % to
  22-36 % depending on depth, i.e. 14-27 points of intended structure silently
  truncated to binary.
- **Qualified: the same sampler's early exit is a latent hazard, not an observed
  defect.** The claim that it leaves nodes with fewer than two children, producing
  a final-state particle carrying an intermediate-state mass, does not occur at the
  shipped mass pools: 120 000 simulated trees over six configurations produced zero
  such nodes. The guard on the root mass makes the first `break` unreachable and the
  second provably dead given the over-subtraction. Our reserve-based construction
  still rules it out by design.
- **The valid-tree rate is an optimistic metric, in two ways rather than one.**
  Reconstruction is greedy, so a matrix consistent with no single tree can still
  reduce to one (~70 % of accepted random symmetric matrices do not reproduce their
  own input LCAG). And because leaves the prediction calls disconnected are dropped
  first, a prediction that keeps a *single pair* scores a perfect 1.0 -- measured --
  with nothing in the loss to discourage it, since class 0 is never a target and
  carries weight 0. A strict variant now closes both holes and is reported
  alongside; the lenient definition is kept for comparability with the prior papers
  and must not be quoted without the caveat.
- Our LCAG construction was cross-checked against baumbauen's own implementation
  on 9198 sampled topologies spanning `max_depth` 2-5 and `n_fsps` 2-8: matrices
  and leaf order matched exactly in every case.

## 6. What the review changed, and what it invalidates

The phase-1-4 verification found three things that had already been written up as
results. Recording them here because the correction is part of the finding:

- **The g-purity observable had three inconsistent definitions** (`DECISIONS.md`
  D78): the tracked series used `n_layers * u`, the whitening acceptance test used
  `u`, and §1's table used `u`. So the test that gated the rotation and the series
  that reported the outcome were a factor of `n_layers` apart. Resolved in favour
  of `u`, on the unflattening manuscript's own terms -- under re-uploading the
  closed forms describe the state entering the first trainable block. Every purity
  number in this file has been re-measured under the single convention.
- **The known/unknown probe was 26 % contaminated** (D82). Topologies were deduped
  on their mass-labelled form, but the LCAG label sees only the unlabelled shape
  and masses are not model inputs, so a quarter of group C's "unseen" topologies
  carried a label matrix from training. Now 0 %. The fix exposed the constraint the
  old key was hiding: distinct shapes are genuinely scarce at small leaf counts (2
  at three leaves, 4 at four, 8 at five, 15 at six, for `max_depth=4`), which
  **bounds how many truly-unseen topologies any dataset at this depth can hold**.
  That belongs in the paper's description of the probe.
- **Whitened arms were being evaluated on unrotated angles** (D81). The rotation
  is not a trainable parameter and was never stored in the checkpoint, so
  `evaluate` rebuilt three of the nine phase-4 cells as the raw arm. No published
  number depends on it -- the cells had not been run at scale -- but any
  whitened-arm test metric produced before this fix should be discarded.

Also fixed, without invalidating a result: the front end no longer re-seeds the
whole model (D84), so the raw-versus-learned comparison differs by the front end
alone; and the "parameter-matched" classical baseline was off by a factor of
sixteen (D86) and is now computed rather than asserted.

## 7. The learning-rate sweep: the rescue is real, reproducible, and does not pay

The sweep §3 called for, on the 3200-event dataset, 40 epochs, through the `train`
flow. Two arms crossed five learning rates, then an extension pushed past the grid
edge and added the controls.

### The direction of the effect is set by the learning rate

Change in closed-form g-purity from epoch 0 to epoch 40:

| lr | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy + learned | -0.106 | -0.099 | -0.030 | +0.022 | **+0.635** | +0.091 | +0.451 |
| XY + learned | -0.898 | -0.966 | -1.008 | -0.977 | **+0.205** | -- | -- |

Monotone in `lr` through the sign change, in both arms. **§3's "the front end drives
purity down" was an artefact of too small a step**, and the earlier guess that it
was an instability at too *high* a rate had the sign backwards. At `lr = 1e-2` the
legacy arm rises from 0.397 to 1.032, crossing `mu_4 = 0.8125`; the XY arm rises
from 1.045 to 1.251.

This is what the unflattening mechanism predicts. The encoder channel's gradient
scales as `sigma^2` against the circuit channel's `sigma^4`, so what survives at
clustered inputs is small but non-zero, and Adam's per-parameter normalisation
turns it into an escape -- *if* the step is large enough to take it.

Three checks that it is not an artefact:

- **The exact statevector purity moves the same way** (legacy 0.258 -> 0.590, XY
  0.119 -> 0.641 at `lr = 1e-2`), so this is a property of the state the circuit
  prepares, not only of the closed form.
- **Three seeds all rise** on the headline cell: 0.397 -> 1.032, 0.941, 0.626.
- **The fixed arms are flat to the digit** -- raw 0.397 -> 0.397, whitened 0.297 ->
  0.297 -- confirming they are flat by construction and that the moving arm is
  moving because it is learning.

### But it buys nothing on the task

At `lr = 1e-2`, legacy encoding, everything else equal:

| arm | g-purity | val acc | test acc | test known / unknown |
| --- | --- | --- | --- | --- |
| raw (no front end) | 0.397 flat | **0.460** | **0.439** | 0.474 / 0.437 |
| fixed whitening | 0.297 flat | 0.406 | 0.426 | 0.441 / 0.425 |
| learned front end | 0.397 -> **1.032** | 0.453 | 0.394 | 0.488 / 0.388 |

**The arm whose purity is rescued does not beat the arm whose purity never moves.**
Raw matches it on validation and beats it on test. Within-run correlation between
the per-epoch purity and the per-epoch accuracy is inconsistent in sign across
cells (-0.81 to +0.59), so there is no coupling to point at either.

That is the answer to the open question §3 raised, and it is a negative one: on
this task, at this scale, **g-purity moves without paying**. The trainability
obstruction the unflattening theory describes is real and the front end does lift
it -- but lifting it is not what decides how well the model reconstructs a decay
tree.

### The floored-ansatz prediction holds

`Matchgate`, the floored control, at the same learning rate:

| arm | g-purity | val acc | test acc |
| --- | --- | --- | --- |
| raw | 3.239 flat | 0.430 | 0.443 |
| learned front end | 3.239 -> 3.099 | 0.391 | 0.410 |

The purity barely moves -- there is nothing to rescue, since the floor already
protects it -- and the learned front end is *harmful*, costing 0.039 validation
and 0.033 test accuracy. This is `DECISIONS.md` D29's falsifiable prediction,
confirmed: on a floored ansatz a learned front end is useless-or-harmful, and
preconditioning there should be fixed rather than learned.

### What this means for the study

- The **encoding comparison (§1) is the robust result** and it survives the
  scale-up. The rescue is a genuine second result, but a mechanistic one: it
  demonstrates the theory's dynamics on real kinematic data without demonstrating
  a benefit.
- The honest headline is therefore *"the theory's mechanism is observable on a real
  HEP task, and its observable does not predict task performance"*. That is
  publishable as an application study and is a stronger contribution than an
  overstated benefit claim would be -- but it is not the "rescue improves
  reconstruction" story the ROADMAP anticipated.
- **The learning rate is a confound that has to be reported.** One optimiser and
  one rate drive both the front end and the circuit here. The prior work (CHEP'23)
  used separate learning rates for the quantum and classical parts for exactly this
  reason. Whether the front end simply needs a different rate from the circuit --
  rather than "a large one" -- is untested and is the obvious next experiment.

### Caveats

40 epochs, one dataset, `n=3` seeds only on the headline cell. Test accuracies sit
in a narrow 0.29-0.45 band across every arm, so the task differences quoted above
are small; they are reported because the *ordering* is consistent, not because any
single gap is established. The whitening arm's numbers predate `DECISIONS.md` D91
and should be re-run.

## 8. What the angle distribution says: not pinning, and not flattening either

§7 left the mechanism of the rescue undetermined: `P_offdiag` is built from
`sin^2(theta)` factors, so it climbs both when the angles *spread* toward uniform
and when they *pin* near `pi/2`, which is the maximum `n - 1 = 3` and the
configuration the unflattening manuscript notes destroys the input information.
The second would have explained §7's flat accuracy neatly. `analysis.angle_stats`
(D92) measures the distribution per qubit; the headline cells re-run with it, at
`lr = 1e-2`, 40 epochs, on the legacy (clustered) encoding:

| arm | seed | P end | sin2 phi | sin2 alpha | dTV phi | dTV alpha |
| --- | --- | --- | --- | --- | --- | --- |
| raw (control) | 0 | 0.397 | 0.341 | 0.293 | +0.000 | +0.000 |
| learned | 0 | 1.032 | 0.595 | 0.504 | -0.064 | -0.173 |
| learned | 1 | 0.941 | 0.581 | 0.512 | +0.203 | -0.171 |
| learned | 2 | 0.626 | 0.394 | 0.472 | -0.143 | +0.118 |
| learned, Matchgate | 0 | 3.099 | 0.579 | 0.316 | +0.004 | +0.015 |

All arms start from the same place (`sin2 = 0.341 / 0.293`, `TV = 0.406 / 0.453`),
because the front end is the identity at epoch 0. A uniform law would read
`sin2 = 0.5`; the raw control is flat to three decimals, confirming that anything
that moves, moves because the front end moved it.

**The pinning hypothesis is refuted.** `mean_sin2` rises to 0.39-0.60, i.e. toward
the uniform value 0.5, and nowhere near the 1.0 that pinning at `pi/2` would give.
The rescue is not buying purity by discarding the input, so §7's flat accuracy
needs a different explanation than that one.

**But it is not flattening either.** The total variation to uniform moves in
*different directions per seed and per site*: seed 0 becomes more uniform at both
site types, seed 1 becomes markedly *less* uniform at the `phi` sites while
becoming more uniform at `alpha`, and seed 2 does the opposite. Averaged over
sites the change per seed is `-0.119, +0.016, -0.013` -- no consistent direction.
What *is* consistent is `sin2` tracking the purity almost exactly across seeds
(correlation 0.98 over the three end states, which the closed form implies since
it is built from those factors).

So the front end moves the one moment the purity depends on toward its uniform
value, without making the distribution uniform. **That is a replication, on real
kinematic data, of the unflattening manuscript's own latent-drift finding** -- that
training does not converge to the uniform law, and that a pooled histogram which
appears to say otherwise is an artefact. It also settles the ROADMAP's wording:
*"the input distribution flattening can be observed"* is **not** what happens, and
the hypothesis should be restated in terms of purity recovery.

### The Matchgate row is the sharpest result here

On the floored ansatz the front end moved the `phi` sites just as far as on the
live arm -- `sin2` from 0.341 to 0.579, a larger change than seed 2 managed -- and
the purity did not follow, drifting from 3.239 to 3.099. The distribution changed;
the observable the theory cares about did not, because the diagonal words floor it.
And the task got *worse* (val 0.391 against 0.430 raw).

That is a much stronger form of D29's prediction than §7 could state. It is not
merely that a learned front end fails to help when the ansatz is floored: **the
front end does the same amount of work and the floor absorbs all of it**, so the
capacity spent on preconditioning is capacity taken from the task. Preconditioning
on a floored ansatz should be fixed, not learned -- and this is the measurement
that shows why.

### Reading caution carried from D92

`tv_uniform` has a floor set by kinematics rather than by training: the `alpha`
sites cannot leave `(0, pi)` and sit inside roughly `[pi/4, 3pi/4]` (D79), so they
can never reach a uniform law. Every TV number above is quoted as a *change from
the raw arm at the same site* for that reason, and the absolute values should not
be compared against zero. The statistics are per qubit throughout; pooling them
would reproduce exactly the flat-looking artefact this section is about.

## 9. The ceiling check: the QFM constellation does not learn the task

Before comparing architectures it was worth asking whether *any* architecture beats
the trivial baseline here, since a comparison between two models that both sit at
the majority-class rate measures nothing. Four arms, same dataset, 40 epochs:

| arm | params | train loss | acc known | acc unknown | Perfect-LCAG known |
| --- | --- | --- | --- | --- | --- |
| GNN `dim=64`, lr 1e-3 | 128 964 | 0.825 -> **0.392** | **0.732** | 0.400 | **0.330** |
| GNN `dim=64`, lr 1e-2 | 128 964 | 1.044 -> 0.825 | 0.380 | 0.392 | 0.000 |
| MLP-only control | 28 | 1.021 -> 0.805 | 0.363 | 0.289 | 0.125 |
| QFM XY raw | 70 | 0.975 -> 0.820 | 0.415 | 0.364 | **0.000** |

Loss trajectories, every fifth epoch, tell the story more plainly than the table:

    classical GNN   0.825  0.624  0.516  0.463  0.438  0.423  0.414  0.382   (still falling)
    QFM             0.975  0.823  0.822  0.821  0.820  0.820  0.820  0.820   (flat from epoch 5)

**The task is learnable and the classical GNN learns it.** On topologies it has
seen, it reaches 0.732 accuracy and reconstructs a third of the LCAG matrices
exactly. **The QFM constellation does not learn it at all**: its loss is flat from
epoch 5 onward and it reconstructs zero matrices, on the same events, with the same
loss and the same schedule.

Three things this exposes, none of which the earlier sections could see:

- **Overall accuracy was hiding it.** The test split is 94 % unknown-topology by
  construction, so an *overall* number is dominated by a subset every arm fails.
  The classical GNN's overall 0.419 looks like the QFM's 0.367; its known-subset
  0.732 against 0.415 does not. Every arm comparison from here has to be read on
  the known subset, or it is measuring the generalisation gap instead of the model.
- **`lr = 1e-2` is a rate at which nothing learns.** It costs the classical GNN
  everything -- loss stalls at 0.825, Perfect-LCAG drops to zero, exactly the QFM's
  failure mode. And `lr = 1e-2` is precisely where 7's rescue appears. So the
  purity rescue and task learning currently live at **incompatible learning
  rates**, which is a far more specific reading of "it moves without paying" than
  7 could give: with one shared optimiser we cannot have both at once.
- **The classical GNN does not generalise either** -- 0.732 known against 0.400
  unknown, essentially the majority-class rate on unseen topologies. That gap is
  now honest to quote, because the known/unknown split is genuinely disjoint in
  LCAG labels (D82); it was 26 % contaminated before. Whether *any* model here
  generalises across topologies is a separate open question from whether the
  quantum arm fits.

### What this says about the architecture

The QFM arm has 70 trainable parameters against the classical arm's 128 964. The
parameter-matched comparison (D86) is fair in the sense it was meant to be and
useless in the sense that matters: at 70 parameters neither model has the capacity
to fit this task, and the classical arm only succeeds when allowed 1800x more.
Underfitting this complete -- flat loss after five epochs -- is a statement about
capacity and optimisation, not about quantum versus classical.

A concrete structural flaw found while reviewing the design, independent of
capacity: **the ansatz bond topology does not respect the particle partition.**
`XY_Brickwork` at `n = 4` has bonds (0,1), (2,3), (1,2). Qubits 0,1 carry particle
A's two angles and 2,3 carry particle B's, so two of the three bonds are
*intra*-particle and only (1,2) crosses between them -- joining `alpha_A` to
`phi_B`, an arbitrary asymmetric pairing. The edge function is therefore not
symmetric under swapping its two endpoints, and the symmetry is recovered by
averaging the output afterwards (D35). One third of the circuit's entangling
structure carries the cross-particle information the edge function exists to
compute, and the symmetry it fails to learn is imposed by hand.

## 10. Phase 4b, before any training: what the three arms are

Three measurements that need no fit at all, and that fix what the arms mean before
their numbers exist. All at the constellation size `n_qubits = 4`, and all from the
`characterize` flow (run `1787767689395-9acacc69`) rather than from a scratch
script -- a table that decides how the arms are read should carry a run id and a
commit stamp like everything else (`DECISIONS.md` D102).

### The ansatz certificates, and the criterion that turned out not to conflict

The edge QFM spends two qubits per particle, so the endpoint swap acts on the
wires as `pi = (0 2)(1 3)`. §9 found that `XY_Brickwork` does not respect that
partition: two of its three bonds are *intra*-particle and the third joins
`alpha_A` to `phi_B`. The ROADMAP expected the fix to cost the floor -- that a
partition-respecting ansatz would either be trivial or saturate `su(2^n)`, and so
stop carrying the preconditioning question. Measured, it does not:

| ansatz | bonds | `dim_g` / 255 | `d_Z` | `pi`-invariant | cross bonds |
| --- | --- | --- | --- | --- | --- |
| `XY_Brickwork` | 01, 12, 23 | 12 | 0 | **no** | 1/3 |
| `XY_Cross` | 02, 13 | 4 | 0 | yes | 2/2 |
| **`XY_Ring`** | 01, 23, 02, 13 | **24** | **0** | **yes** | 2/4 |
| `XY_AllPairs` | all six | 60 | **6** | yes | 4/6 |
| `Circuit_19` | CRX ring | 255 | 15 | yes | -- |

`XY_Ring` is the 4-cycle `0-1-3-2-0`, assembled from the qml-essentials topologies
`bricks(offset=0)` (the intra-particle bonds) and `stairs(span=2)` (the
cross-particle ones). An even cycle is bipartite, and bipartite XY couplings keep
`d_Z = 0` -- the manuscript's own graph criterion. So the arm respects the particle
partition **and** keeps the input distribution decisive, at twice the DLA dimension
of the arm it replaces. `XY_AllPairs` adds the triangles, and the odd cycles
rebuild the diagonal sector exactly as the manuscript predicts: `d_Z` goes 0 -> 6.
That is a floored control which *also* respects the partition, which is what
allowed `Matchgate` to be retired.

Bond invariance is not parameter equivariance. `pi` maps the bond `(0,1)` to
`(2,3)`, so those two gates must carry the same angle; `XY_Ring` ties them
(`Block(shared=True)`) and is therefore exactly equivariant under the endpoint
swap, which makes the output averaging of D35 a no-op instead of a patch. The
span-2 gates are `pi`-fixed and need no tying. `XY_AllPairs` ties nothing and is
equivariant only in its bonds.

The certificates are read off each arm's own `structure()` and `Topology` helpers,
so they describe the circuit that runs. A useful side effect: the whole per-ansatz
purity dispatch could go. On an RY product state `<X> = sin`, `<Y> = 0`, `<Z> =
cos`, so `P_g` is a sum over the Y-free DLA basis words, computable for *any* arm
in `O(|basis| * n)` and agreeing with all three hand-derived closed forms to
float32 (D94). Without that, none of the new arms could be measured at all.

### The encoding cells, and why "ternary" alone is not an arm

Taken literally -- `Encoding("ternary")` with a fully widened re-upload mask --
arm B destroys the input: every qubit `q` would encode `RY(3^q * sum_f u_f)`, so
the model sees one number instead of four features. The widening has to be
partial, and the cell that works is `cyclic`: qubit `q` sees features `q` and
`q + 1`. Measured per-feature reachable spectrum (`Encoding.get_spectrum`, two
layers, minimum over the four features), and dissociation checked by exhaustion
over all 80 nonzero `eps` in `{-1,0,1}^4`:

| cell | weight on qubit `q` | freq/feature | dissociated | swap-equivariant |
| --- | --- | --- | --- | --- |
| `hamming-diagonal` | 1 | 5 | separable | yes |
| `hamming-cyclic` | 1 | 9 | **no** | yes |
| `binary-diagonal` | `2^q` | 5 | separable | no |
| `binary-cyclic` | `2^q` | 13 | yes | no |
| `ternary-diagonal` | `3^q` | 5 | separable | no |
| `ternary-cyclic` | `3^q` | 17 | yes | **no** |
| `ternary_pair-diagonal` | `3^(q mod 2)` | 5 | separable | yes |
| **`ternary_pair-cyclic`** | `3^(q mod 2)` | **17** | **yes** | **yes** |

Two things move between the phase-4 baseline and the arm -- spectrum size and
feature mixing -- and the two off-diagonal rows separate them. `hamming-cyclic`
mixes without enriching and is the only cell that fails dissociation;
`ternary-diagonal` enriches the *angle* without enriching the per-feature comb.
Without both, "ternary helped" cannot be told apart from "you multiplied by 27" or
"you mixed the features" (D95). The `diagonal` cells are dissociated only in the
vacuous sense that a diagonal `W` has no cross terms to kill; there is no
cross-feature preconditioning in them at all.

### The last column is a result, not bookkeeping

Exponential weights are dissociated *because* they distinguish the qubits -- and
the endpoint swap `pi = (0 2)(1 3)` exchanges the two particles' qubits, so
`3^q` is not invariant under it. **Arm B as first written would have undone the
equivariance arm C restores**, which is the kind of interaction independent arms
are supposed to be free of.

It is avoidable, and the reason is worth stating: the manuscript's dissociation
condition is on a weight *vector* `w` acting on a scalar input,
`sum_k eps_k w_k != 0`. Here it is a condition on the weight *matrix*,
`W^T eps != 0`, which is strictly weaker because the widened mask gives each
feature its own column. Repeating the exponent per particle, `w = (1,3,1,3)`,
satisfies the matrix condition and is `pi`-invariant, whereas no `pi`-invariant
vector satisfies the vector condition -- `(a,b,a,b)` always admits
`eps = (1,-1,1,-1)` when `a = b`, and the matrix form is what rescues it
otherwise. Checked by exhaustion over all 80 nonzero `eps`.

`ternary_pair-cyclic` reaches 17 frequencies on *every* feature, against
`ternary-cyclic`'s `[25, 17, 17, 17]`, so the symmetry costs no spectrum (D100).
Among the cells that enrich the spectrum at all it is the only one that is both
dissociated and swap-invariant, and therefore the only one that composes with
`XY_Ring` rather than undoing it. The `diagonal` cells satisfy both conditions
vacuously -- a diagonal `W` has no cross terms to kill and no per-feature comb to
widen -- which is why the comparison is drawn over the enriching cells only.

The compatibility claim the ROADMAP made holds and is now the code's contract:
with weights, one encoding layer rotates qubit `q` by `theta_q = sum_f W[q,f] u_f`,
the state stays an RY product state, and every purity form applies with `theta` in
place of `u`. The zero-parameter law generalises from `cos(L u)` to `cos(L W u)`
and is checked at every cell and depth (D96).

### Arm B's prediction, registered on synthetic angles before the data arrives

The manuscript's identity `E_x[P_g(rho(wx))] = E_phi[P_g]` needs a *uniform scalar*
input; ours is neither. What it also states, and what is testable here, is the
weaker claim: under clustered inputs exponential weights *partially recover*
through jitter amplification while Hamming weights stay collapsed. Run on
synthetic angle laws (2e5 draws, `XY_Ring`, `mu_4 = 1.25`), so the number on real
data is a confirmation rather than a discovery:

| cell | uniform | clustered `s=0.30` | clustered `s=0.03` |
| --- | --- | --- | --- |
| `hamming-diagonal` | 1.250 (26%) | 0.050 (100%) | 0.0000 (100%) |
| `hamming-cyclic` | 1.283 (25%) | 0.175 (94%) | 0.0000 (100%) |
| `ternary-diagonal` | 1.251 (26%) | 0.774 (50%) | 0.055 (100%) |
| `binary-cyclic` | 1.249 (26%) | 0.834 (46%) | 0.010 (100%) |
| `ternary-cyclic` | 1.249 (26%) | **0.895 (42%)** | 0.141 (95%) |
| `ternary_pair-cyclic` | 1.251 (26%) | 0.638 (59%) | 0.001 (100%) |

(mean g-purity, and the fraction of samples below the `mu_4 / 2` acceptance
threshold in brackets.)

Three things this fixes in advance:

- **The uniform column does not discriminate, and that is the identity holding.**
  Every cell sits on `mu_4` to three digits, Hamming included -- with four
  *independent* uniform features the Hamming encoding already reproduces the iid
  prior. So the arm has to be read on the clustered column; a uniform-input
  comparison would show nothing and mean nothing.
- **The ordering is the manuscript's, and the controls do their job.** At
  `s = 0.30` the recovery over the phase-4 baseline is 18x for `ternary-cyclic`.
  Mixing alone (`hamming-cyclic`) buys 3.5x and scaling alone
  (`ternary-diagonal`) buys 15x, so the effect is mostly *weight magnitude*, not
  feature mixing -- which is exactly what the two controls exist to separate, and
  it would have been invisible with a single ternary switch. `binary-cyclic` falls
  between, giving the dose-response in spectrum size.
- **No cell removes the input-distribution dependence.** At `s = 0.03` every cell
  is still below the threshold for 95-100% of samples. "Spectral preconditioner"
  therefore has to mean *partial recovery*, and the ROADMAP's stated prediction --
  "better training performance and **no** input-distribution dependence" -- is
  already too strong in its second half, before any training.

**The symmetric cell costs preconditioning strength.** `ternary_pair-cyclic`
recovers 0.638 against `ternary-cyclic`'s 0.895, and the reason is structural: the
paired weighting has only two distinct values, so the weight-1 qubits stay
clustered whatever the base is. Scanning `w = (1, b, 1, b)` -- all dissociated,
all swap-equivariant -- confirms it saturates:

| `b` | 2 | 3 | 9 | 27 | 81 |
| --- | --- | --- | --- | --- | --- |
| freq/feature | 13 | 17 | 25 | 25 | 25 |
| purity at `s=0.30` | 0.471 | 0.638 | 0.675 | 0.676 | 0.674 |
| purity at `s=0.03` | 0.000 | 0.001 | 0.033 | 0.433 | 0.502 |

So endpoint symmetry and preconditioning strength genuinely trade off at
`s = 0.30`, and the ceiling is about 0.68 against the asymmetric arm's 0.895.
`ternary_pair` is kept at the principled definition -- the exponent runs over one
particle's qubits and repeats across particles, `3^(q mod 2)` -- rather than tuned
to this table, since the real pair-polar angles are not clustered at all
(§1: 1.235). A larger base is available if the clustered `legacy` arm needs one.

### Arm B, measured on real kinematics: preconditioning is a contraction, not a rescue

The prediction above, run through `encoding_report` on the 30k dataset. Mean
g-purity against the `XY_Ring` basis (`mu_4 = 1.25`), with the fraction of edges
below the `mu_4 / 2` acceptance threshold in brackets:

| cell | `pair_polar` | `direct` | `legacy` (clustered) |
| --- | --- | --- | --- |
| `hamming-diagonal` (phase-4 baseline) | **1.958 (0%)** | 1.668 (12%) | **0.271 (85%)** |
| `hamming-cyclic` (mixing only) | 1.267 (32%) | 1.283 (29%) | 0.444 (73%) |
| `binary-diagonal` | 1.272 (25%) | 1.424 (17%) | 0.570 (65%) |
| `binary-cyclic` | 1.242 (26%) | 1.261 (26%) | 0.747 (54%) |
| `ternary-diagonal` (scaling only) | 1.115 (31%) | 1.431 (18%) | 0.705 (56%) |
| **`ternary-cyclic`** | 1.224 (28%) | 1.233 (27%) | **0.831 (48%)** |
| `ternary_pair-diagonal` | 1.037 (36%) | 1.656 (11%) | 0.406 (76%) |
| `ternary_pair-cyclic` | 1.248 (26%) | 1.254 (26%) | 0.623 (61%) |

**On the clustered arm the prediction holds, and the decomposition survives
contact with real data.** `legacy` goes 0.271 -> 0.831, a 3.1x recovery, with the
fraction of collapsed edges falling from 85% to 48%. The ordering is monotone in
spectrum size -- 0.271 < 0.444 < 0.570 < 0.705 < 0.747 < 0.831 -- and the two
controls split the effect the same way the synthetic table did: mixing alone buys
1.6x, scaling alone 2.6x, so it is mostly weight magnitude. `ternary_pair-cyclic`
lands at 0.623, below the asymmetric cell, reproducing the symmetry/strength
trade-off exactly as the synthetic scan said it would.

**On the encoding the quantum arm actually uses, it goes the other way.**
`pair_polar` at the phase-4 baseline sits at **1.958, well above `mu_4 = 1.25`**,
with essentially no edge below threshold -- and every richer encoding drags it
*down*, to 1.04-1.27. Every dissociated cell lands within a few percent of
`mu_4`, from either side.

That is the identity doing exactly what it says, and it forces a correction to
how the whole arm was framed. `E_x[P_g(rho(wx))] = E_phi[P_g]` sets the
preconditioned average **to** the uniform-prior mean; it does not raise it.
Preconditioning is therefore a *contraction toward `mu_n`*, and whether that is a
rescue or a loss depends on which side the raw input sits:

- `legacy` sits far below (0.271), so contraction is the rescue the manuscript
  describes, and the manuscript only ever has cause to consider this side;
- `pair_polar` sits ~1.6x **above** (1.958), because `alpha` is kinematically
  confined near `pi/2` -- the *favourable* RY point (§1, D79) -- so contraction
  destroys an advantage the raw encoding already had.

So the ROADMAP's arm-B prediction needs replacing rather than qualifying. It
predicted "better training performance and no input-distribution dependence". What
is true is: **richer spectra act as a spectral preconditioner, and preconditioning
only helps when the input distribution is worse than uniform.** On this task's own
encoding it is worth about a factor 1.6 in the wrong direction.

This is consistent with §1 rather than new, and it sharpens it into a mechanism.
§1 measured pair-polar at 1.235 against `XY_Brickwork`'s `mu_4 = 0.8125`, i.e. 1.52x
above the prior mean; here it is 1.958 against 1.25, i.e. 1.57x. The same fact in
two algebras. What §1 could not say is *why* that matters: the encoding this
project chose is already better than the preconditioned limit, so the theory's
headline intervention has nothing to offer it.

The open question the training runs now have to answer is no longer a formality:
on `pair_polar`, arm B trades **spectrum** (5 -> 17 frequencies per feature) against
**purity** (1.96 -> 1.22). Whether the expressivity gain pays for the trainability
loss is a question about the task, and only a fit can answer it.

### The generalisation probe's ceiling, measured

§6 flagged that distinct tree shapes are scarce at shallow depth and that the
known/unknown probe therefore has a ceiling, without saying where it is. Asking
the sampler for `n` distinct shapes per group and letting it fail says exactly
where:

| `max_depth` | 10/group | 20/group | 34/group |
| --- | --- | --- | --- |
| 4 | 30 shapes | 60 shapes | **fails at 85** |
| 5 | 30 shapes | 60 shapes | 102 shapes |

**At `max_depth = 4` the whole shape space is 85 trees** for leaf counts in
`[3, 8]`, so a dataset cannot hold more than that many genuinely distinct
topologies however many events it generates. The 30k sweep dataset uses 30 of
them, which is comfortable; the scale-up needs 102 and is therefore *only*
reachable at `max_depth = 5`. Raising the depth for the larger dataset is a
requirement, not a preference, and the ceiling is now a number rather than a
caveat.

### The compute the arms actually need, and where the GPU goes

Measured on this machine (Tesla P100, jax 0.11.1), because the plan assumed the
GPU would carry the generation:

| workload | CPU | GPU |
| --- | --- | --- |
| phase-space generation, 20k events | **6074 ev/s** | 1206 ev/s |
| QFM train step, `n_layers=2` | **2.5 ms** | 2.6 ms |
| QFM train step, `n_layers=16` | **2.2 ms** | 19.3 ms |
| GNN train step, `dim=64` | **13.7 ms** | 14.1 ms |

**The GPU loses on both.** The arrays are small -- 16 amplitudes per edge QFM, a
few thousand edges per batch -- so kernel launch dominates, and phasespace-jax
needs float64 throughout. It is also an active hazard here: the engine starts one
worker process per node, each JAX process preallocates 75% of the device by
default, and the first generation run deadlocked in an allocator retry loop for 21
minutes at 0.4% CPU and 0% GPU utilisation without surfacing an error. CUDA is now
an opt-in extra and everything runs with `JAX_PLATFORMS=cpu` (D99).

The useful consequence for the plan looked like "arm A is nearly free", since the
QFM *step* cost is flat in depth on the CPU up to `n_layers = 16` (2.5 ms at depth
2, 2.2 ms at depth 16). **That was the wrong reading, and the sweep corrected it.**
Measured end-to-end wall-clock per run, four concurrent on the 8000-event training
split: `n_layers = 2` takes about 200 s, `n_layers = 8` between 900 and 1850 s.

The step cost is flat; the *compile* cost is not. qml-essentials unrolls the
circuit, so the jitted graph grows with `n_layers + 1` ansatz layers and XLA pays
for it once per run -- which a steady-state microbenchmark measures away by design,
because it compiles before it times. Depth is cheap per gradient step and expensive
per experiment, and only the second of those decides how long a sweep takes.

## 11. What the unflattening classification does and does not decide

Phase 4b's four ansatz arms are a direct test of the manuscript's own ansatz
taxonomy on a real task, because they were chosen to span it. Measured at
`n_qubits = 4`, `frontend=none`, three seeds, purity on the repaired subset
(D105):

| ansatz | `dim_g` | `d_Z` | `P_g` | `P/mu_4` | `Var = P_g/dim_g` | params | acc known | perfect known |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `XY_Brickwork` | 12 | 0 | 1.306 | **1.61** | 0.1088 | 70 | 0.379 | 0.068 |
| `XY_Ring` | 24 | 0 | 2.000 | **1.60** | 0.0833 | 70 | 0.371 | 0.099 |
| `XY_AllPairs` | 60 | 6 | 3.618 | 1.07 | 0.0603 | 106 | 0.452 | 0.078 |
| `Circuit_19` | 255 | 15 | 15.000 | **1.00** | 0.0588 | 106 | 0.436 | 0.089 |

### Confirmed, and sharply

**The `d_Z` certificate predicts input-distribution sensitivity exactly.** Both
floor-free arms sit 60% above their own prior mean, so the encoded distribution is
decisive for them; the floored arms sit at 1.07 and 1.00, pinned. `Circuit_19`
returns `P_g = 15.0000` -- exactly `2^n - 1`, as it must for *every* pure state --
and its variance `15/255 = 0.0588` reproduces the textbook `1/(2^n + 1) = 1/17` to
four digits. This is the cleanest confirmation of the theory anywhere in the
project: a number predicted with no free parameter, met exactly, on real kinematics.

**The graph criterion transfers.** The 4-cycle is even, hence bipartite, hence
`d_Z = 0` at `dim_g = 24`; adding the triangles gives `d_Z = 6` at 60. The
manuscript's bipartite-versus-odd-cycle rule holds on the edge QFM's geometry.

### Not confirmed: the classification does not rank ansaetze for the task

- **The predicted gradient variance is monotone decreasing in `dim_g`
  (0.109, 0.083, 0.060, 0.059) while per-element accuracy is *highest* for the two
  lowest-variance arms.** Anti-correlated, though confounded: those two also carry
  106 parameters against 70.
- **The spread is 1.85x, and the floor of it is `1/17`.** A "barren plateau" of
  0.059 is perfectly trainable. The theory is asymptotic in `n`, and at the
  constellation size its discriminating variable has no dynamic range -- so a
  certificate that predicts trainability predicts nothing about performance here,
  because nothing here is untrainable. That is not a failure of the theory; it is a
  statement about where `n = 4` sits relative to it.
- **It is blind to the difference that mattered.** `XY_Brickwork` and `XY_Ring`
  agree in every unflattening descriptor -- `d_Z = 0`, `P/mu` 1.61 against 1.60,
  70 parameters each -- and differ consistently on the structural metrics
  (Perfect-LCAG 0.068 -> 0.099, strict valid-tree 0.297 -> 0.514, the latter in
  3 of 3 seeds). At `n = 3` seeds that is 1.4-1.6 pooled standard deviations:
  a direction, not yet a result. The distinguishing property is bond geometry
  relative to the two-particle partition, which the taxonomy has no vocabulary for.
  Conversely, at matched capacity a **4x difference in `dim_g`** (`XY_AllPairs`
  against `Circuit_19`) moves accuracy by 3%.

This answers open question 4 negatively: the DLA ratios do **not** rank final task
performance. Ratio 0.047 / 0.094 / 0.235 / 1.000 against Perfect-LCAG
0.068 / 0.099 / 0.078 / 0.089 is not monotone in either direction.

### The tension it leaves

**The arms that perform best are exactly the ones on which the phase-4 mechanism
cannot be observed.** `XY_AllPairs` and `Circuit_19` lead on accuracy and are, by
certificate, input-distribution indifferent -- the front-end rescue is unobservable
on them by construction, and on `Circuit_19` provably so. So the scientifically
live arm and the best-performing arm are different arms. `XY_Ring` is the only one
that is both floor-free and competitive on the structural metrics, which is an
argument for it as the live arm that the theory itself cannot supply.

The honest framing for the write-up: the unflattening classification is a *safety*
criterion -- can the input distribution decide trainability here -- and not a
*selection* criterion. At `n = 4` the two questions decouple completely. The
classification would acquire teeth by scaling the constellation, since at `n = 8` a
universal arm sits near `1/257` while a polynomial-DLA arm stays `Theta(1/n)`. That
is a concrete experiment rather than a concession: `n = 4` is a design choice (D24),
not a constraint.

## Open questions this raises

1. Does the §1 result survive at full dataset scale? It is the load-bearing
   measurement and still rests on a few hundred events.
2. Given §1, what is the study's headline: the encoding comparison, the
   sub-population rescue, or the clustered legacy arm now that it exists?
3. *(Answered in §7.)* The direction is set by the learning rate: too small a
   step and purity falls, at `lr = 1e-2` it rises past `mu_4` and reproduces across
   seeds. The rescue does not buy task performance, which is the finding that
   matters.
4. Does the front end need a *different* learning rate from the circuit, rather
   than merely a large one? One optimiser drives both here, which confounds §7's
   axis; CHEP'23 split them for this reason. This is the next experiment.
5. Does anything the task cares about track g-purity at all? §7 says no at this
   scale. If that holds up, the honest framing of the whole project is mechanistic
   rather than performance-driven, and the paper should say so up front.
6. *(Answered in §8.)* The rescue is not `pi/2` pinning and not flattening: the
   front end moves `sin^2` toward its uniform value without making the law uniform,
   replicating the manuscript's latent-drift result on real data.
7. Why is the TV direction seed-dependent (§8)? Three seeds is too few to say
   whether the front end has several equally good solutions or whether one site
   type is simply easier to move. Worth `n >= 5` before it is written up.
4. Do the DLA ratios of §2 rank final task performance, and does the FCC
   (phase 5) rank it the same way?
5. How should the generalisation probe be read given that distinct shapes are
   scarce at shallow depth (§6)? Either `max_depth` rises for the full runs, or
   the probe's ceiling gets stated explicitly.
6. Some sampled topologies are effectively ungeneratable -- daughters nearly
   saturating the parent mass drive the unweighting acceptance rate below `1e-6`,
   and generation raises rather than finishing. Harmless at the scales run so far;
   it needs a policy (resample the topology, or cap the mass ratio) before a large
   offline generation runs unattended.

## What to run next

The fixes make three cells worth running before phase 5, in this order:

Done: the encoding comparison at scale (§1), the learning-rate sweep (§7), the
`Matchgate` prediction cell (§7) and the legacy clustered arm (§7). What is left,
in order:

1. **Split the optimiser.** One learning rate drives the front end and the circuit,
   which confounds §7's only axis. Separate rates -- CHEP'23's approach, for this
   exact reason -- would say whether the front end needs a *different* step or
   merely a large one.
2. **Re-run the whitening arm** under D91, since the rotation it used was fitted on
   a different encoding than it was applied to.
3. **Seeds across the board.** Only the headline cell has `n=3`; the task-accuracy
   differences in §7 are small enough that everything quoted needs the same
   treatment before it goes in a paper.
4. **Then phase 5.** The spectrum/FCC instrumentation is the remaining ROADMAP
   phase, and §7 gives it a sharper question to answer than it had: if g-purity
   does not predict task performance, does the FCC? The §9 caveat applies equally:
   a descriptor cannot be shown to rank task performance while no quantum arm
   achieves any.

Read every future arm comparison on the **known-topology subset** (§9): the test
split is 94 % unknown, so an overall number is dominated by a subset nothing solves.
