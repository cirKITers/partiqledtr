# PartiqleDTR Roadmap

Framing: inductive-bias / application study. No quantum-advantage claim (classical surrogates
via RFF exist for the relevant spectra).

`RESEARCH.md` holds the measurements and what they mean; `DECISIONS.md` holds why each
implementation choice was made. This file is the plan and the state of it.

## Status

Phases 1-4 and 4b are done and measured, the last at 10 seeds over 280 runs
(`FINDINGS.md`). Four results carry forward:

- **The encoding decides whether this task is in the barren regime at all** — pair-polar
  sits at 1.61 `mu_n`, the legacy `p*E*pi` product at 0.28 with 85% of edges below
  threshold (`RESEARCH.md` §1, §10). The clustered regime is reached only under the prior
  work's encoding, which is now a deliberate control arm rather than a baseline to beat.
- **Preconditioning is a contraction toward `mu_n`, not a rescue.** It lifts the clustered
  arm 0.28 -> 0.55 and drags pair-polar 1.61 -> 1.0, so richer spectra help only where the
  input distribution is worse than uniform. On this task's own encoding they cost
  accuracy, and on the clustered arm the purity gain takes Perfect-LCAG from 0.081 to
  0.010 (§10).
- **Neither depth nor spectrum is the bottleneck** (§10, `FINDINGS.md` §3-4). `n_layers`
  2 -> 16 buys +0.063 accuracy; every dissociated encoding scores at or below the Hamming
  baseline. The quantum arm reaches 0.45 accuracy and 0.09 Perfect-LCAG against the
  classical GNN's 0.949 and 0.731.
- **A partition-respecting, floor-free ansatz exists and helps structurally.** `XY_Ring`
  gives +27% Perfect-LCAG and +36% valid trees over `XY_Brickwork` at equal parameters,
  with per-element accuracy slightly lower — the split a symmetry gain should show.

**And one finding that reframes what is left.** The unflattening variance law
`Var = P_g(rho) P_g(O) / dim g` needs the readout inside the algebra, `O in i g`. Our
readout is per-qubit Pauli-Z, and single-qubit Z is **not** in the DLA of any XY arm
(0/4 on `XY_Brickwork`, `XY_Ring`, `XY_AllPairs`; it is only in `Circuit_19`, which
saturates `su(2^n)`). So `P_g(O) = 0` exactly on the floor-free arms: the encoded-state
purity has no channel to our loss, and the preconditioner's effect on it is the "only shifts
the mean" sector of the theory. That is why the preconditioner moves the distribution over a
190-fold range and the task score barely responds — which is what phase 4c exists to fix,
and it comes before phase 5.

## Fixed decisions

- Simulation: analytic expectation values only (no shots), JAX autodiff. No parameter-shift,
  no Qiskit machinery from partiqlegan. **On the CPU**: the GPU was installed and measured,
  and it loses on both generation (1206 vs 6074 ev/s) and training (19.3 vs 2.2 ms/step at
  `n_layers=16`), because the arrays are small enough that kernel launch dominates (D99).
- Architecture: constellation of small shared-weight QFMs per edge inside message passing —
  NOT the old one-qubit-per-FSP monolith. Shared weights give permutation equivariance;
  small circuits keep vmap cheap and per-QFM spectra tractable.
- **The preconditioner never touches the frequencies, deliberately.** It is an elementwise
  residual MLP (per-feature 1->16->1, zero-init output, `phi = x` at epoch 0): it reshapes
  per-feature marginals and cannot mix features. That is what isolates *distribution
  shaping* as the thing being measured, and it is what leaves the QFM solely responsible
  for the frequency content. A linear preconditioner would silently be a trainable-frequency
  model; a nonlinear one is a warp, which is neither a frequency change nor a way to reduce
  spectral redundancy (see phase 4b arm B).
- **Trainable frequencies (`trainable_frequencies=True`) stay out for now**, in every arm.
  Fixed-weight expressivity has to be settled first: §9 shows the optimisation is already
  fragile, and learned encoding weights add the Fourier-locking failure mode (encoding
  weights and entangling layers couple non-linearly into spurious local minima). A separate
  axis later, on whichever arm wins.
- Ansatz arms so far: `XY_Brickwork` (floor-free poly DLA, `dim_g` 12/255, `d_A` 0 — the
  live arm), `Matchgate` (floored, 28/255, `d_A` 4 — the control), `Circuit_19` (saturates
  `su(2^n)`, 255/255 — a legacy bridge that is input-distribution independent by
  construction). Every arm's DLA and floor count are recorded before training, by the flow's
  shape rather than by convention.
- We are **not bound by partiqlegan's design**. New architectures, encodings and metrics are
  in scope. The legacy encoding survives only as the clustered *physics regime* phase 4
  needs, and prior published numbers are context, not a target.
- Dataset: known/unknown-topology split, deduplicated on unlabelled tree *shape* so the
  probe is real. Read every arm comparison on the **known subset** — the test split is 94%
  unknown by construction, so an overall number is dominated by a subset nothing solves.
- Stack: qml-essentials (`Model` API), JAX, Flax NNX, Optax, Fluksio (nodes in plain
  modules, flows declared in `partiqledtr/pipeline.py`). Fluksio or qml-essentials
  limitations go to `NOTEPAD.md` rather than being worked around.

## Phases

### 1. Data — done
Phase-space generation with unweighting, shape-deduplicated topology sampling with a
viability probe, LCAG conversion, leaf shuffling, `-1` padding. Three encodings stored per
split (`angles`, `cartesian`, `legacy`); `encoding_report` prices each in g-purity.

### 2. Baselines — done
Classical message-passing GNN and a linear MLP-only control, parameter-matched via
`models.matched_dim` rather than by hand. Metrics: masked accuracy, Perfect-LCAG, and the
**strict** valid-tree rate as primary (the lenient one is kept only for comparability).

### 3. QFM constellation — done
Per-edge QFM via `Model.apply` (analytic expval, folded batch x edges), shared parameters
across edges, per-qubit Pauli-Z readout plus a shared linear head. DLA certificate recorded
upstream of every fit.

### 4. Preconditioner / input-distribution study — done
Arms: raw | fixed whitening | learned elementwise MLP, crossed with pair-polar or the
clustered legacy encoding. Observables: closed-form g-purity of the encoded angle
distribution (argument `u`, never `n_layers * u`), the exact statevector purity, and the
per-qubit angle statistics that separate spreading from `pi/2` pinning. Results in
`RESEARCH.md` §7-8. Left over: split the optimiser, re-run the whitening arm under D91, and
seeds beyond the headline cell.

### 4b. Expressivity: make the quantum arm fit the task — done

Opened by §9. Three arms, independent, cheapest first. **Trainable frequencies stay off in
all of them** so the axes remain separable. All three are axes of the existing `train`
flow; `dev/s2-expressivity/run.py` submits them. Decisions in `DECISIONS.md` D94-D99.

**Arm A -- depth scan, current Hamming product encoding.**
The per-feature spectrum is `2L + 1` and grows only linearly in depth: measured (3,3,3,3)
at `n_layers=1`, (5,5,5,5) at 2 (our default), (9,9,9,9) at 4. So each angle currently
enters as a *degree-2 trigonometric polynomial*, which is a strong candidate for §9's flat
loss. `n_layers` is already a flow port, training is fast (the step cost is flat in depth
up to 16 on CPU), so scan it wide (2, 4, 8, 16) and read the known-subset fit. This is the
cheapest possible test of "is expressivity the bottleneck", and it changes nothing else.

**Arm B -- encoding weights, as a 2x3 grid rather than one switch (D95).**
The ROADMAP's original wording -- "ternary weights with the re-upload mask widened" --
does not survive contact with `Encoding("ternary")`: with a *full* mask every qubit `q`
encodes `RY(3^q * sum_f u_f)`, so the model sees only the sum of the four features. The
widening has to be partial, and once it is, spectrum size and feature mixing move
together and need separating:

| weights \ mask | `diagonal` | `cyclic` (qubit q sees features q, q+1) |
| --- | --- | --- |
| `hamming` | phase-3/4 baseline, 5 freq/feature | mixing only, 9, **not dissociated** |
| `binary` | scaling only, 5 | dissociated, 9-13 |
| `ternary` | scaling only, 5 | **dissociated, 17-25** -- the arm |

crossed with `preconditioner` in {none, mlp}, plus the clustered `legacy` encoding at
`hamming` and `ternary`.

Prediction, stated in advance so it cannot be fitted after the fact: **better training
performance and no input-distribution dependence.** The second half is the interesting one
and it is the project's own result -- for dissociated weights `E_x[P_g(rho(wx))] =
E_phi[P_g]` exactly, so an exponential encoding is a *built-in preconditioner*, while
Hamming weights leave the input distribution fully decisive. If that lands, it explains why
phase 4's effect exists at all: it is a property of the flat-weight encoding, not of the
task. Confirming a hypothesis is the point; the arm is worth running either way.

The manuscript's identity holds for a *uniform scalar* input; ours are neither, so the
falsifiable form here is the weaker one it also states -- under clustered inputs
exponential weights *partially recover* through jitter amplification while Hamming stays
collapsed. `encoding_purity` answers that **without training anything** (D97), which is
what keeps the training runs from being the only evidence.

Compatibility, checked and now the code's contract: with weights, qubit `q` encodes
`RY(sum_f W[q,f] u_f)`, so the state is *still* an RY product state and every purity form
applies unchanged with `theta = W u` (D96). The D55 zero-parameter test generalised from
`cos(L u)` to `cos(L W u)` and is checked at every cell.

**Arm C -- ansatz and architecture.**
Two motivations that happen to point the same way: §9's structural finding, and the
unflattening manuscript's own open question of *which* ansaetze are sensitive to
preconditioning. Stays in the **Hamming** regime, which is what makes the published
fourier-fingerprints results transferable.

Selection criteria: (1) the bond set is invariant under the endpoint swap
`pi = (0 2)(1 3)`, so the edge function is symmetric by construction; (2) the set spans the
DLA trichotomy; (3) FCC computable at n=4. The ROADMAP expected 1 and 2 to pull apart.
**At n = 4 they do not** -- measured (D98):

| ansatz | bonds | `dim_g` / 255 | `d_A` | `pi`-invariant | reading |
| --- | --- | --- | --- | --- | --- |
| `XY_Brickwork` | 01, 12, 23 | 12 | 0 | **no** | phase-3/4 continuity control |
| `XY_Ring` | 01, 23, 02, 13 | 24 | 0 | yes | **the arm**: floor-free *and* partition-respecting |
| `XY_AllPairs` | all six | 60 | 6 | yes | floored control, also partition-respecting |
| `Circuit_19` | CRX ring | 255 | 15 | yes | universal leg, input-independent by construction |

`XY_Ring` is `Topology.bricks(offset=0)` plus `Topology.stairs(span=2)`: the 4-cycle
`0-1-3-2-0`. An even cycle is bipartite, which is the manuscript's own criterion for
`d_Z = 0`. `XY_AllPairs` adds the triangles and the floor comes back, which is why
`Matchgate` could be **retired** -- its floored role is taken over by an arm that also
respects the partition. Bond invariance is not parameter equivariance: `pi` swaps the two
`bricks` bonds, so that block ties its angle (`shared=True`) and `XY_Ring` is exactly
equivariant, making D35's output averaging a no-op rather than a patch.

Still open inside arm C, from §9: **capacity** (70 parameters against the classical arm's
128 964 -- the parameter-matched arm stays as a control but cannot be the only quantum arm)
and **splitting the optimiser** (`lr = 1e-2` is where the rescue appears *and* where the
classical GNN stops learning entirely, so one shared rate cannot serve both). Neither is
built yet.

### 4c. Architecture: give the model a channel the mechanism can act through

Opened by the readout finding above and by `FINDINGS.md` §1. Phase 4b ruled out depth
and spectrum as the bottleneck; what is left is the architecture itself. Ordered by
expected value, not by cost.

**1. Put the readout in the algebra.** Replace the per-qubit Pauli-Z observables with
the arm's own generators — `XX + YY` on its bonds — so `P_g(O) = 2^n` and
`eq:variance` applies to the loss we actually train. This is what the manuscript's own
`exp_latent_drift` does: `obs = [X_i X_{i+1}, Y_i Y_{i+1}]` for the off-diagonal arm,
and `PauliZ` only for the matchgate arm, where `d_Z = n > 0` puts Z inside the algebra.
Both of its arms satisfy the premise; ours satisfies it on `Circuit_19` alone.

`observables` is a `Model` argument, so the change is small — but it resets every
quantum number in `RESEARCH.md`, so it is a new phase rather than a patch. Success
criterion: the g-purity trajectory and the loss become correlated within a run, which
is the thing phase 4 could not show and phase 4b showed the absence of.

**2. Widen the node state.** `w_node` maps to `N_ANGLES = 2`, so a particle's hidden
representation between message-passing blocks is **two numbers** — it has to be
re-encodable as two angles. The classical GNN carries 64. This is the most likely
reason the architecture caps out, and it is independent of anything quantum. Three
ways out, in increasing cost to the design:

- more qubits per edge, which widens the state and gives the theory's variance its
  range back (see phase 6);
- several QFMs per edge with independent parameters, read jointly into the head;
- letting the classical channel carry width alongside the quantum one, which weakens
  D27's "cross-particle structure only from the QFM" and has to be argued for rather
  than slipped in.

**3. Capacity.** 70-238 parameters against the classical arm's 128 964. Listed under
arm C and never tested. Arm A's accuracy still rising at `n_layers = 16`, where the
count reaches 238, reads as a capacity signal more than a depth one.

**4. Split the optimiser.** Flagged as "the next experiment" since §7 and still undone.
`lr = 1e-2` is where the rescue appears *and* where the classical GNN stops learning
entirely, so one shared rate cannot serve both. CHEP'23 split them for this reason.

**5. Train longer.** The loss is still descending at epoch 40, by 0.0003-0.0007 per
epoch on every arm. The cheapest lever and the weakest: extrapolation gives about 0.60
after another 400 epochs, against the GNN's 0.080. Worth doing as a control so that
"underfitting" stops being a live explanation, not as a fix.

Items 1 and 2 rank far above the rest: without 1 the project cannot observe the effect
it exists to study, and 2 is the most likely reason the architecture cannot reach the
task at all.

### 5. Spectrum analysis (fourier-fingerprints connection)
- Offline: fingerprint + FCC per ansatz arm at the constellation's size. Phases 4 and 4b
  sharpened the question: g-purity does *not* predict task performance (`FINDINGS.md` §6),
  so does the FCC? Same caveat as everywhere — a descriptor cannot be shown to rank task
  performance while every quantum arm sits near 0.45 accuracy, so this waits on **4c**
  rather than 4b. 4b answered its own question and moved the blocker rather than clearing
  it.
- Online: spectrum tracking during training, in latent `phi` (post-preconditioner). This is the
  fingerprints paper's own open question about nonlinear classical preprocessing, and our
  preconditioner is exactly that case.

### 6. Ablation matrix & scaling
- {MLP-only, QFM-only, MLP+QFM} x {raw, fixed whitening, learned preconditioner} x ansatz arm.
- Scaling in event size and dataset size, against the classical baselines, citing PASCL and
  Kahn et al. as external reference points.
- Read the known/unknown probe against its ceiling: distinct tree shapes are scarce at
  shallow depth (2 at three leaves, 4 at four, 8 at five for `max_depth=4`), which bounds
  how many genuinely unseen topologies a dataset can hold.

### 7. Writing
Target narrative: an application leg connecting unflattening (trainability / input
distribution) and fourier-fingerprints (spectrum / inductive bias) on a real HEP task. On
current evidence the honest headline is mechanistic — the theory's mechanism is observable
and its observable does not predict task performance — and arm B may add a second one: the
encoding change that makes the model expressive enough to learn is the same one that removes
the input-distribution dependence.

## Open questions

- Does *anything* here generalise across topologies? The classical GNN scores 0.732 known
  against 0.400 unknown, i.e. the majority-class rate on unseen shapes. Honest to quote now
  the split is uncontaminated, and separate from the quantum arm's underfitting.
- Why is the total-variation direction seed-dependent (§8)? Three seeds cannot say whether
  the preconditioner has several equally good solutions or whether one site type is simply easier
  to move. Wants `n >= 5`.
- Formal link between spectral redundancy (`Omega-hat` degeneracies) and DLA structure? Both
  toolchains sit in qml-essentials. Speculative — do not promise it in the paper.
- Spectrum definition under a trainable preconditioner: report against latent `phi` (clean) or
  1D/2D slices in `x` (end-to-end)? Decide before phase 5 writes anything down.
