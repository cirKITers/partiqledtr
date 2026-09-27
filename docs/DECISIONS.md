# Decision log

Every non-obvious choice made while implementing ROADMAP phases 1-4, with its reason.
Chronological within sections. `[user]` marks decisions the user confirmed explicitly.

## Stack and tooling

- **D1 `[user]` qml-essentials pinned to git rev, not PyPI.**
  `[tool.uv.sources] qml-essentials = { git = ..., rev = "59ed6ad" }`. The pin has
  moved several times during implementation (`e72ac03` -> `3cc19f1` -> `07d325f` ->
  `59ed6ad`): each bump picked up the user's fixes for issues this project had
  recorded in `NOTEPAD.md`.
  PyPI 0.2.2 has neither `qml_essentials.algebra` (`lie_closure_paulis`,
  `g_purity_from_basis`, `matchgate_generators`, `dim_so2n`) nor the
  `XY_Brickwork` / `Matchgate` ansaetze that ROADMAP phases 3-4 require. Verified
  after each `uv sync`: the three ansatz classes and the five algebra symbols are
  importable, and the full test suite passes against the pinned revision.
- **D2 `matplotlib` declared as a direct dependency,** for the phase-1 statistics
  figures. It was originally also required because qml-essentials imported it
  unconditionally and so could not be imported without it; the user has since made
  those imports lazy, leaving our own use as the only reason.
- **D3 `[user]` phase-space generation uses `phasespace-jax`, and the project has
  no TensorFlow dependency at all.** Pinned to
  `git+https://github.com/stroblme/phasespace-jax@4f06c1b`. The distribution is
  named `phasespace-jax` but imports as `phasespace`, so it is a drop-in at the
  import line. Removing it dropped tensorflow, tensorflow-probability and five
  more packages from the lock, and cut the generation import from seconds to
  0.34 s. The API differs in one place: randomness is an explicit JAX key
  (`key=`) rather than a `tf.random.Generator` (`seed=`). Verified after the port
  that the layout, weight normalisation, conservation laws and reproducibility
  all behave exactly as measured for the TensorFlow version (D13, D13a, D14,
  D14a still hold verbatim).
- **D4 `requires-python = ">=3.12,<3.13"`.** qml-essentials declares
  `>=3.11,<3.13`; without the upper bound uv resolves a lock that the pinned
  dependency cannot satisfy.
- **D5 ty configured via `[tool.ty.src]` / `[tool.ty.environment]` in
  `pyproject.toml`.** Verified against ty 0.0.74: a standalone `ty.toml` takes
  top-level `src`/`environment`/`rules`/`terminal`/`analysis`/`overrides`
  sections, and the same sections live under `[tool.ty.*]` in `pyproject.toml`.
  `include = ["partiqledtr", "tests"]` keeps `reference/` (PyTorch code we do not
  own) out of the check.
- **D6 ruff rule set `E,F,W,I,UP,B,NPY,RUF,D` at line length 100,** google
  docstring convention, `D104/D105/D107` off (no nagging for package `__init__`
  and dunders), all `D` off under `tests/`. Research code: docstrings on public
  functions are the interface documentation, but test names document themselves.

## Layout

- **D7 `partiqledtr/data/` and `partiqledtr/models/` are packages, not flat
  modules.** The original reason was isolating the seconds-long TensorFlow import
  to `data/generation.py`. That cost is gone (D3: phasespace-jax adds 1.6 ms on
  top of jax), and so is the x64 side effect that briefly replaced it (D7a), so
  the layout now stands purely on separation of concerns and test granularity --
  which is reason enough, and cheaper than churning it back. `models/` gets the
  same treatment so the phase-3 QFM model lands beside the baselines without
  touching them.
- **D7a `[user]` phasespace scopes x64 to its own calls, so the containment for
  it is gone -- but the dtype pinning it exposed stays.** For one revision
  (`983a6b2`) `import phasespace` called
  `jax.config.update("jax_enable_x64", True)` at import time, flipping the default
  dtype of every array in the process and making results depend on import order.
  The user fixed it in `4f06c1b`: a `with jax.enable_x64()` scoped per call, in a
  dedicated `precision` module. Verified here -- the global flag reads `False`
  before the import, after it, and after a `generate` call, while conservation
  residuals are ~1e-13 (full float64). What that episode changed permanently:
  1. **Reverted**: `assemble_dataset` imports `generation` at module scope again,
     `data/__init__.py` says nothing about x64, and `tests/conftest.py` is gone.
  2. **Kept**: `ElementwiseResidualMLP` pins `param_dtype=jnp.float32`. This was a
     latent bug **of ours**, not of phasespace: its parameters followed the global
     flag while every `nnx.Linear` pinned float32, so with x64 on a float64 front
     end silently promoted the whole forward pass. It could not manifest while
     nothing enabled x64, but any dependency or user may flip that flag.
     `tests/test_models.py::test_parameter_dtype_does_not_follow_the_global_x64_flag`
     asserts the invariant under both settings.
  3. **Kept**: the numpy boundary in `generate_events`. phasespace still returns
     **float64** arrays (that is the point of the scoped mode), and combining one
     directly with a float32 JAX array warns and silently truncates -- measured:
     `f64 + jnp.zeros(4, jnp.float32)` gives a `UserWarning` and a float32 result,
     as does a `.at[].set()` of a float64 scalar. Converting through numpy warns
     on neither and keeps the precision, so `generate_events` returns numpy
     float64 and downstream code never meets a stray float64 JAX array.
     `tests/test_generation.py::test_output_crosses_the_jax_boundary_as_numpy`
     pins that, because a regression there would only warn, not fail. The whole
     suite is additionally run with `-W error` on `UserWarning`, `FutureWarning`
     and `DeprecationWarning` to confirm no implicit cast survives anywhere.
- **D8 Neither package `__init__.py` re-exports its modules.** Callers import the
  specific module (`from partiqledtr.data.lcag import topology_to_lcag`), so each
  module names exactly what it depends on. The original reason -- keeping a heavy
  or side-effecting import out of every data import -- no longer applies (D3, D7a);
  the convention stays because it is clearer, not because it is load-bearing.
- **D9 No `configs/` directory and no YAML layer.** Fluksio already records node
  settings (keyword defaults) and flow inputs (`Port`) per run; a parallel config
  system would duplicate that and split the source of truth.
- **D10 No dataset or config directory is *tracked*.** Fluksio stores artifacts
  content-addressed and records their digests per run, so the repo carries no
  copy. Superseded in its detail by D106: the engine's store is `./.fluksio`
  rather than `~/.fluksio`, and the exported splits, results and figures sit at
  the repo root in `data/`, `results/` and `figures/` -- all gitignored. There is
  no `reports/`.
- **D11 Dataset statistics are a flow node, not a notebook.** The ROADMAP said
  "notebook/flow". A node is seeded, re-runnable and its figures are attached to
  the run as artifacts; a notebook is none of these.

## Pipeline (Fluksio)

- **D11a Fluksio semantics verified by running them, not by reading docs.** A
  scratch two-node flow confirmed: keyword-only node parameters work (the
  generated shim calls `fn(port=port, **settings)`); a node may be a generator,
  `yield`ing streamed values and `return`ing final ones; `Flow(outputs=...)` takes
  plain message-name **strings**, not `Port` objects; decorated nodes stay
  directly callable in tests without an engine.
- **D11b Every value a node yields is declared as a `provides` port with
  `stream=True`.** This began as a defensive convention -- 0.1.0 accepted an
  undeclared yield key silently, so a typo lost data with no diagnostic. Fluksio
  0.1.2 now refuses one, both statically (an AST check over the node's literal
  yields at decoration time) and at the first yield, so the convention is enforced
  rather than merely advisable.

## Physics and data

- **D12 `[user]` phasespace weights are handled by unweighting** (accept-reject
  against the maximum weight, generate with a margin, trim to `n_events`), not
  ignored as in baumbauen/partiqlegan. The study is *about* input angle
  distributions; using weighted events as if unweighted biases exactly the
  marginals phase 1 must characterise. Cost is a one-time factor on offline
  generation.
- **D13 phasespace 4-vector layout verified as `[px, py, pz, E]`.** Checked
  empirically rather than trusted from the reference code: for
  `root(100) -> a(5) + b(3)`, components 0..2 sum to zero and component 3 sums to
  100 across events, and `sqrt(E^2 - |p|^2)` returns the declared child masses.
  Two-body decays return weights identically 1, as expected.
- **D13a phasespace momenta are events-major, `(n_events, 4)`.** The TensorFlow
  version's docstring wrongly claimed `(4, n_events)`; phasespace-jax documents
  the layout correctly, and both were measured to be events-major.
- **D14 Per-topology seeding uses an explicit JAX key.** `generate(key=...)`
  accepts an int seed or a PRNG key. Verified: two calls
  with the same seed give bit-identical events, different seeds differ.
- **D14a Unweighting needs no maximum-weight estimate.** With the default
  `normalize_weights=True`, phasespace already divides each weight by the maximum
  attainable weight (constant across events), so weights lie in `[0, 1]` and
  accepting event `i` with probability `w_i` is exact. `generate_events` draws in
  chunks sized from the running acceptance rate rather than a fixed 2x margin,
  because the rate varies by an order of magnitude between topologies (measured
  around 0.30 for a symmetric 3-body decay, but as low as 0.001 for one whose
  daughters nearly saturate the parent mass).
- **D14b The accept-reject loop splits its key per round and sizes chunks from a
  pilot draw.** Two traps, both hit during the port and both now covered by
  tests. Reusing one JAX key would redraw the *identical* chunk forever, since
  the same key reproduces a draw exactly. And `generate` is jitted with
  `n_events` static, so the old adaptive chunk size would recompile every round --
  but a *fixed* size cannot reach a topology whose acceptance rate is 0.001. The
  loop therefore draws one pilot chunk to measure the rate, derives a single
  chunk size from it, and holds that: two compiled sizes per topology.
- **D14c The unweighting test is calibrated to its own standard error.** It
  previously asserted a fixed 0.05 tolerance on a 40k sample whose standard error
  is 0.04 -- about 1.2 sigma, so it was always ~25% flaky and had passed by luck.
  A 20-seed check against a 1.6M-event reference put the bias at -0.023 +/- 0.010
  (t = -2.3, consistent with zero), confirming the estimator is unbiased and the
  threshold was the problem. It now uses 100k events and asserts agreement within
  4 sigma and separation from the raw sample beyond 5 sigma.
- **D15 Topology sampling gets an exact FSP count by rejection.** partiqlegan let
  the leaf count emerge from the mass-budget walk and then *scanned seeds* to hit
  a target count. The ROADMAP wants a controllable FSP count, so `sample_topology`
  retries until the leaf count matches and raises after `max_tries`.
- **D16 Known/unknown topology split restored** (baumbauen scheme: three
  topology groups, A -> train/val/test, B -> val/test, C -> test only).
  partiqlegan dropped it; ROADMAP phase 1 asks for it back as the generalisation
  probe.
- **D17 Fixed-size padding applied once, at generation time** (features padded
  with `0.0`, labels with `-1`, LCAG diagonal also set to `-1`). JAX wants static
  shapes, so one artifact shape means one jitted step and trivial batching. The
  two independent `-1` mechanisms of the reference code (diagonal fill at dataset
  read time, label padding in the collate function) are merged into this single
  place so the convention has one owner.
- **D18 Baseline features are `(theta, phi, E)`, angles kept raw.** Normalisation
  is scale-only and separate for momentum and energy, never shifting (energy must
  stay non-negative); angles are already `O(1)` and periodic, and scaling them
  would break the periodicity the QFM encoding relies on. The Cartesian
  `(px, py, pz, E)` variant is stored in the same artifact for the ablation.
  This replaces partiqlegan's `p_[xyz] * E * pi` product encoding, whose factors
  both live in `[-1, 1]` so their product concentrates near zero -- the clustered
  regime the unflattening theory predicts is untrainable.
- **D19 Both feature encodings live in one artifact per split.** Ablation arms
  then never re-run phasespace.

## LCAG, topology sampling and features (phase 1 internals)

- **D39 The LCAG entry is the *height* of the lowest common ancestor.**
  baumbauen's pull-down is exactly `level(v) = depth - height(v)` with
  `height(leaf) = 0`, so `level(leaf) - level(lca)` collapses to `height(lca)`.
  `topology_to_lcag` computes heights bottom-up in one pass instead of the
  reference's repeated queue sweeps. Cross-checked against baumbauen's own
  `decay2lca.py`, loaded directly, on **9198 sampled topologies** spanning
  `max_depth` 2-5 and `n_fsps` 2-8: matrices and leaf-name order matched exactly
  in every case.
- **D40 `lcag_to_adjacency` emits leaves first, in input order, then internal
  nodes breadth-first** (so the last row is the root). baumbauen renumbers every
  node breadth-first and loses the correspondence between adjacency rows and LCAG
  rows, which makes a round-trip test impossible.
- **D41 Inconsistent LCAGs are detected explicitly, not via an `IndexError`.**
  Two leaves whose entry claims a level but that already share a lower ancestor
  raise immediately; baumbauen builds a node with a duplicated child and only
  fails later when breadth-first indexing overruns the matrix.
- **D42 `is_valid_lcag` reproduces baumbauen's valid-*tree* criterion, which is
  weaker than "is a genuine LCAG" -- and that matters when reading the metric.**
  The reconstruction is greedy, so a matrix inconsistent with any single tree can
  still reduce to one: measured on random symmetric matrices, about 70% of
  accepted ones do not reproduce their own input LCAG. Kept deliberately, because
  the phase-2 valid-tree rate has to mean what it meant in the prior papers to be
  comparable. Consequence: **the valid-tree rate is an optimistic metric and must
  be reported as such.** The strict version (reconstruct, then re-derive the LCAG
  and require equality) is a small addition if a tighter number is wanted.
- **D43 Treeness is checked as symmetric + connected + `|E| = |V| - 1`,** which is
  equivalent to baumbauen's recursive DFS cycle search, iterative and O(V+E).
- **D44 `lcag_to_adjacency` ignores the diagonal,** because padded labels and model
  predictions carry `-1` there (D17) and validity must not depend on it.
- **D45 partiqlegan's mass-budget accounting is a bug and is corrected here.** The
  reference subtracts the running child-mass *total* on every iteration of the
  child loop, over-counting the budget. We subtract the mass of the child just
  placed.
- **D46 Every internal node gets at least two children by construction.** A
  reserve of `min(MASSES_FSP)` on the first child guarantees a second fits under
  the parent mass. The reference's early `break` *can* leave a node with one or
  zero children, producing a leaf that carries an *ISP* mass and breaking the
  "identity is the mass" invariant `canonical_form` depends on. **Qualification
  (review):** that path is unreachable at partiqlegan's shipped mass pools -- 120k
  simulated trees over six configurations produced no such node -- so it is a
  latent hazard in the reference, not a defect of its published data. D45's
  arithmetic bug is the one with measurable consequences.
- **D47 Children per internal node are fixed at 2..3** (partiqlegan's configured
  values) rather than exposed as arguments. `max_depth` counts levels with the
  root as level 1, so `max_depth=4` allows at most 3 edges root-to-leaf.
- **D48 Zero-momentum rows map to `theta = phi = 0`,** computed with a guarded
  divisor so padded rows produce neither NaN nor a numpy warning, and
  `to_cartesian` maps them back to the zero four-vector.
- **D49 Class 0 never occurs as a label.** Any two real leaves share at least the
  root, so every scored cell is at least 1, and all padding is `-1` (D17). Class 0
  stays in the model's output space because predicting it means "unrelated", which
  the valid-tree metric interprets -- but it is never a target, so `class_weights`
  gives it weight 0 and the `_primary` metric variants (D21) coincide with the
  plain ones on this dataset. `n_classes` is measured from the data rather than
  assumed from `max_depth`, since pull-down can leave a tree shallower than the
  sampling bound.
- **D50 Dataset artifacts are written with `allow_pickle=False`.** Every stored
  array is numeric; an object array would be a mistake worth raising on rather
  than silently pickling into a research artifact.

## Metrics and training

- **D20 Metrics follow baumbauen, not partiqlegan.** The mask is built from ground
  truth only, as in baumbauen. **Correction (review):** an earlier version of this
  entry called partiqlegan's `edge_accuracy` / `perfect_lcag` wrong, on the grounds
  that `(a == b)` compares "prediction correct" against "not ignored". That is
  incorrect and has been withdrawn. The line three above them,
  `prediction = t.where(label == ignore_index, label, prediction)`, forces
  agreement on every ignored cell, so `(a == b).sum() / b.sum()` reduces exactly to
  the intended masked accuracy. The reference code is obfuscated and fragile -- a
  sibling method applies `two_child_fix` *after* that overwrite and does break the
  identity -- but the two metrics named here are numerically right, and published
  numbers from that line need no re-derivation on this account.
- **D21 `_primary` metric variants** (ignoring both `-1` and class `0`) are
  reported alongside the plain ones: they cost nothing and exclude the trivially
  correct zero entries from the score.
- **D22 Loss is class-weighted cross-entropy with `-1` ignored.** Focal loss and
  the ordinal/EMD losses (LCAG classes are generation counts, hence ordinal) are
  noted as options but not built: they are small additions once there is a
  baseline number to compare against.
- **D23 Checkpoints are `np.savez` state dictionaries stored as one artifact,**
  not orbax. One artifact, no extra dependency, and the flat `nnx.state` mapping
  round-trips directly.

## Classical baselines (phase 2)

- **D31 The model interface is a call convention plus a registry, not a base
  class.** `__call__(features (B,L,F), mask (B,L) bool) -> logits (B,L,L,C)`,
  symmetric in the two `L` axes; a preconditioner is `(...,F) -> (...,F)`.
  `MODELS` / `PRECONDITIONERS` in `partiqledtr/models/__init__.py` are the
  string-selectable extension points (phase 3 adds `"qfm"`, phase 4 `"whiten"`).
  An ABC would add a file and enforce nothing that the first call does not.
- **D32 `models/__init__.py` *does* re-export, unlike `data/__init__.py` (D8).**
  The registry cannot exist without importing the classes. D7/D8 was about
  TensorFlow isolation; jax/flax are already loaded by anything touching a model.
  Watch: adding `"qfm"` makes any `partiqledtr.models` import pull qml-essentials.
- **D33 Edge tensors keep a `(B, L, L, .)` layout; no `(L*L, L)` incidence
  matmuls.** baumbauen builds one-hot `rel_rec`/`rel_send` matrices and flattens
  edges to `L*L`. Plain broadcasting is the same operation, cheaper, and keeps the
  LCAG's two-index structure visible all the way to the head.
- **D34 `edge2node` divides by the true degree, not by `L`.** The mean runs over
  `mask_i & mask_j & (i != j)` with the denominator clamped to `>= 1`. Dividing by
  a constant would make a padded event a *different* function of the same physics.
  The padding-invariance test pins this down exactly (bit-identical logits), not
  approximately.
- **D35 Output symmetrisation is architectural, in every model.**
  `(out + out^T)/2` removes half the hypothesis space for free rather than asking
  the loss to learn a property the LCAG has by construction. It is bit-exact, so
  the test asserts equality rather than closeness.
- **D36 Parameter-matched and unconstrained GNN arms are two `dim` values of one
  class.** Count is `dim^2 (7 + 8 n_blocks) + dim (F + C + 6 + 6 n_blocks) + C`;
  `n_params()` is the matching tool. At `F=3, C=4`: 128 964 params at the default
  `dim=64, n_blocks=3`; 4 148 at `dim=16, n_blocks=2`.
- **D37 No dropout and no batchnorm in the baselines.** Both need train/eval state
  (dropout also an rng stream) threaded through every caller, and `nnx.Dropout`
  is non-deterministic *by default*, so a trainer that forgets `model.eval()`
  silently corrupts validation metrics. Ceiling: the unconstrained arm is
  unregularised. Marked `# ponytail:` in `MLPBlock`.
- **D38 `MLPBaseline` has a strictly linear head, per the ROADMAP's wording
  ("preconditioner + linear head, no QFM").** It is the phase-3 quantum model with
  everything quantum deleted, which is what makes it the right control: 28
  parameters at `F=3, C=4`, deliberately weak, there to establish the floor. An
  earlier draft gave it the GNN's nonlinear pair MLP, which isolates *message
  passing* instead -- a defensible but different experiment, and one that would
  no longer answer "does the elementwise MLP alone do everything?". Re-adding it
  as a third arm is a one-line change if the comparison is wanted.

## Theory instrumentation (phases 3-4)

- **D51 The three ansatz arms map to three different g-purity closed forms, and
  that mapping *is* the phase-4 experiment.** Measured at the constellation size
  `n_qubits = 4` by `analysis.dla_check`, before any training:

  | arm | DLA | `dim_g` | `dim_su` | ratio | `n_diag_words` | purity form |
  |---|---|---|---|---|---|---|
  | `XY_Brickwork` | `so(n) (+) so(n)` | 12 | 255 | 0.047 | 0 | `g_purity_offdiag` |
  | `Matchgate` | `so(2n)` | 28 | 255 | 0.110 | 4 | `g_purity_full` |
  | `Circuit_19` | `su(2**n)` | 255 | 255 | 1.000 | 15 | `g_purity_su` |

  `XY_Brickwork` is floor-free, so clustered angles kill the purity as
  `O(theta^4)` -- the rescue regime. `Matchgate` has `n` diagonal `Z_k` words
  giving a floor of exactly `n`; clustered angles *maximise* it, above the uniform
  mean `n - 1 + 2^-n` (measured 3.99 clustered vs 3.06 uniform at n=4), which is
  the predicted indifference. `Circuit_19` saturates `su(2^n)`, where the g-purity
  is `2^n - 1` for *every* pure state: **the arm is input-distribution independent
  by construction**, with the textbook `1/(2^n + 1)` variance. It therefore cannot
  show preconditioner effects at all -- consistent with the ROADMAP calling it a legacy
  bridge with "no theoretical protection", and worth stating explicitly in the
  paper rather than presenting it as a null result.
- **D52 `n_diag_words`, the count of Z-only closure words, is the
  floored/floor-free certificate.** Diagonal words have expectation 1 in the
  clustered limit, so their count is the deterministic purity floor. Measured 0
  for XY_Brickwork, `n` for Matchgate and `2^n - 1` for Circuit_19 at every
  `n` from 2 to 6.
- **D53 `ansatz_generators` walks each ansatz's own `structure()` and `Topology`
  helpers instead of hardcoding bond sets.** The XY_Brickwork bonds at n=4 are
  (0,1), (2,3), (1,2) -- an open chain with no wrap -- which is not obvious from
  the class and would have been guessed wrong. `CRX(c,t)` has generator
  `(I - Z_c) X_t / 2`, contributing `X_t` and `Z_c X_t`; Circuit_19's `RX` block
  already produces the bare `X_t`, so deduplication makes the split exact. A test
  cross-checks that the walk reproduces upstream `matchgate_generators(n)` exactly
  for n=2..5, which validates upstream against the circuit rather than assuming
  the two agree.
- **D54 `dla_check` caps the Lie closure at `max_dim=2000` rather than hanging,
  using upstream's own cap.** Circuit_19 saturates `su(2^n)`, so an uncapped
  closure enumerates `4^n - 1` words -- 72 s at n=6. We briefly carried a local
  copy of the algorithm purely to add an early exit; `lie_closure_paulis` now takes
  `max_dim` with the same semantics (growth stops at the cap, a result of exactly
  `max_dim` words means `dim_g >= max_dim` and the basis is partial), so the copy
  is deleted. Verified after the swap: the certificates are unchanged and n=6 caps
  in 0.08 s. The returned record carries `capped`, and `dim_g` is then a lower
  bound. Note the return type differs -- upstream yields `PauliWord` objects, so
  the diagonal-word count converts with `to_pauli_string()` first.

## Model architecture (phases 3-4)

- **D24 `[user]` quantum arms encode pair-polar angles at 4 qubits per edge QFM.**
  Two angles per particle, obtained from the Cartesian 4-vector by taking polar
  angles of coordinate pairs (`(px,py) -> phi`, `(pz,E) -> alpha`); an edge QFM
  encodes both endpoints, so 4 features on 4 qubits -- inside the ROADMAP's 2-4
  qubit range. This makes the whitening arm literally the unflattening paper's
  construction (`polar_angles(x @ Q.T)` with a shared Haar `Q`) and lets the
  closed-form g-purities apply verbatim. Accepted cost: the pair radii (transverse
  momentum and the `(p_z, E)` magnitude) never enter the quantum path, so the
  parameter-matched classical baseline is additionally run on the same two-angle
  features to keep the comparison clean.
- **D25 Encoding is one RY-encoded feature per qubit via a diagonal
  `data_reupload` mask.** `Model._iec` otherwise applies every feature to every
  qubit; the mask `[layer, qubit, feature] = (qubit == feature)` reproduces
  exactly the `prod_q RY(u_q)|0>` product state the unflattening closed forms are
  derived for.
- **D26 Readout is per-qubit Pauli-Z expectation values plus a shared linear
  head** (resolves the ROADMAP's open question on edge-logit aggregation). Linear
  in qubit count, so no exponential readout; the qubit order is fixed and shared
  across edges so leaf-permutation equivariance survives; and unflattening's
  Theorem 1 is stated for single-qubit `Z_i` observables, so the trained
  observable family is the instrumented one. A scalar readout (`force_mean` or a
  parity observable) would make the logits rank-1 over `C > 2` classes and is
  kept only as an ordinal-regression ablation.
- **D27 QFMs are the edge function of the message passing; every classical part
  is particle-local.** Preconditioner is elementwise per feature, aggregation is a
  parameter-free masked mean, the node update is a per-node linear map. This is
  the strict reading of "cross-particle structure must come from the quantum
  part".
- **D28 Two message-passing blocks with separate QFM parameters per block.**
  Weight sharing *across edges* is what gives permutation equivariance; sharing
  across blocks is not required for it and would constrain two functions that do
  different jobs. Shared-across-blocks stays a one-line ablation.
- **D29 The Matchgate x learned-MLP cell is deliberately runnable.** The theory
  predicts a learned preconditioner is useless-or-harmful on a floored ansatz; that is
  a falsifiable prediction, so the cell must be executable. Default configurations
  pair Matchgate with fixed whitening.
- **D55 The encoding contract is verified by measurement, not assumed.** With
  `encoding=["RY"] * 4` and the diagonal `data_reupload` mask, the zero-parameter
  readout is exactly `cos(n_layers * u_q)` per qubit -- checked for all three arms
  at `n_layers` 1, 2 and 3. That single assertion pins the whole wiring: feature
  `f` reaches qubit `f` alone, observables come back in qubit order, and the
  encoded state is the RY product state the closed forms assume. **Reuploading
  multiplies the encoding angle**, which is easy to miss: the first draft of the
  test expected `cos(u)` and failed until the factor was understood. Measured
  trainable parameters per QFM at `n_qubits=4, n_layers=2` (3 implemented ansatz
  layers): XY_Brickwork 18, Matchgate 21, Circuit_19 36.
- **D56 The QFM arm requires the `"cartesian"` encoding and rejects `"angles"`
  with an explanatory error.** `pair_polar` needs four-vectors, and the angle
  encoding `(theta, phi, E)` drops `|p|`, which is *not* recoverable for our
  final-state particles because they are massive (`|p| != E`). Reconstructing them
  under a massless assumption would quietly corrupt the quantum arm's input, so
  the constructor refuses instead.
- **D57 `[user]` The forward pass goes through the functional `Model.apply`.**
  Mid-implementation the user fixed the upstream issues this project had recorded
  and pushed rev `07d325f`, which `pyproject.toml` now pins. `apply` writes no
  model state, so it is safe under an outer `jax.jit`, and it keeps the full
  `(B_I, B_P, B_R, O)` output rank instead of squeezing -- which removes the
  shape hazard too. The constellation asserts that exact shape. Verified here:
  jitted and un-jitted gradients agree across repeated calls and the circuit
  instance's parameters stay concrete. A test asserts the repeated- and
  jitted-gradient properties, so a regression back to `__call__` would be caught.
- **D58 Measured cost, which is what makes the study affordable.** One QFM call
  over 3584 folded edges (B=64, L=8) takes 10 ms forward and 39 ms with gradient
  on CPU. Two blocks per step puts a training step around 80 ms, so roughly 12 s
  per epoch at 10k events -- a 100-epoch run is about 20 minutes and the nine-cell
  arm matrix is a few CPU-hours. No outer-jit fix is needed to make phase 3 run.
- **D59 The whitening acceptance test reproduces the paper's mechanism, verified
  on synthetic clustered data.** Four-vectors whose pair second components are
  negligible encode angles near `{0, pi}`: the measured off-diagonal g-purity is
  `0.0` against a threshold of `mu_4 / 2 = 0.406`. A Haar `SO(4)` rotation lifts
  it to `0.986`, accepted after 3 draws with an empirical acceptance rate of 0.80,
  comfortably above Markov's `~1/5` guarantee. Rotations are drawn by QR of a
  Gaussian with the sign and determinant fixed. Fitted on the training split only
  -- the whitening is part of the model, and fitting it on evaluation data would
  leak.
- **D60 The phase-4 predictions are reproduced end-to-end through the actual
  model, before any training.** `QFMConstellation.g_purity` measured on clustered
  synthetic four-vectors (pair second components negligible, so encoding angles
  sit near `{0, pi}`), raw versus a fixed accepted Haar rotation, at `mu_4 =
  0.8125`:

  | arm | clustered | whitened | reading |
  |---|---|---|---|
  | `XY_Brickwork` | 0.00000 | 0.858 | collapse, then rescue above `mu_4` |
  | `Matchgate` | 3.997 | 3.095 | floored at `n = 4`, never collapses |
  | `Circuit_19` | 15.000 | 15.000 | exactly `2^n - 1`, input-independent |

  Note what "indifference" means for the floored arm: the purity is not *unchanged*
  by whitening -- clustering actually *maximises* it (4.00 against the uniform mean
  `n - 1 + 2^-n = 3.06`) and whitening pulls it *down* toward that mean. The
  prediction being confirmed is that it never collapses, so trainability is
  protected either way and a preconditioner has nothing to rescue. Reporting this as
  "unchanged" would misstate the result.
- **D61 MEASURED, AND IT REVISES A ROADMAP PREMISE: the new encodings do not put
  this task in the barren regime -- the old one did.** Mean off-diagonal g-purity
  over all real edges of a generated dataset (1280 training events, 4 topologies
  per group, `max_depth=4`), against `mu_4 = 0.8125` and the acceptance threshold
  `mu_4 / 2 = 0.4062`:

  | encoding | mean `P_g` | fraction of edges below threshold |
  |---|---|---|
  | pair-polar, the arm in use (D24) | 1.265 | 0.165 |
  | direct `(theta, phi)` | 1.077 | 0.232 |
  | partiqlegan's `p_a * E * pi` | **0.395** | **0.639** |

  The ROADMAP argues that "kinematic features cluster encoding angles naturally
  (soft particles yield near-zero angles), so this task lands in exactly the
  regime where the theory makes falsifiable predictions". That holds for the
  **old** encoding -- `p * E * pi` sits below the acceptance threshold, with
  nearly two thirds of its edges in the collapsed regime -- and *not* for either
  replacement. The mechanism is visible in the marginals: the pair-polar
  `(pz, E)` angle concentrates at `pi/2` (mean 1.573, std 0.486), because a soft
  particle has small momentum but `E >= m > 0`, and `pi/2` is the *favourable*
  RY-encoding point, not a barren one.

  Two consequences for the study, neither fatal but both needing to be stated
  rather than discovered late:
  1. **The fixed-whitening arm has little to rescue at the dataset level.** On
     real data the raw arm already scores above `mu_n`, acceptance is immediate,
     and a whitening rotation moves the mean *down* toward `mu_n`. The remaining
     16% of edges below threshold is a sub-population effect, which is a weaker
     and more delicate claim than the ROADMAP anticipated.
  2. **The encoding switch is itself a publishable, falsifiable result.** The
     table above quantifies exactly why the prior work's encoding was hard to
     train, in the unflattening theory's own currency, and it is the cleanest
     empirical link between the two papers this project has produced so far.

  Worth confirming on the full-scale dataset before it goes in a paper; these are
  small-sample numbers, though `0.395` versus `1.265` is not a marginal call.
- **D30 `[user]` the upstream qml-essentials fixes are the user's own work, not
  ours.** The plan had us implement a pure, jit-safe `Model.apply` in
  `~/Documents/CodeWorkspace/qml-essentials`; mid-implementation the user said they
  are already fixing the issues recorded in `NOTEPAD.md` (they have since branched
  to `compatibility`). We therefore do not touch that checkout at all. Downstream
  consequence: the QFM code is written so it works either way -- it uses
  qml-essentials' *native* batching (fold `batch x edges` into the model's input
  batch axis), which is correct with or without an outer `jax.jit`, and gains a
  jitted training step for free once the upstream fix lands. `pyproject.toml` must
  be re-pinned to the new rev once the user pushes.

## Training loop and flows (phase 2 wiring)

- **D68 The checkpoint artifact is self-describing.** `state_to_npz` stores the
  `build_model` keyword arguments plus the encoding as JSON under the reserved
  npz key `"#config"`, beside the parameter arrays. Parameter keys are `'/'`-joins
  of Python identifiers and `nnx.List` indices, so `'#'` cannot collide. That is
  what lets `evaluate` take only `(checkpoint, dataset_test, dataset_meta)`, and
  it means a checkpoint moved between runs still knows what it is.
- **D69 The training loop is a plain generator, `train_model`, and the `fit` node
  is a thin wrapper.** Same split as `assemble_dataset` / `build_dataset` (D11a),
  for the same reason: `save_artifact` raises outside a running node. The overfit
  smoke test therefore exercises the real loop -- optimiser, shuffling, per-epoch
  evaluation -- instead of a re-implementation of it.
- **D70 `build_model` forwards optional keywords only to classes whose `__init__`
  accepts them, read via `inspect.signature(cls.__init__)`.** `signature(cls)` on
  an NNX module resolves to the metaclass' `(*args, **kwargs)` and silently
  reports no parameters, so the GNN kept its default `n_blocks=3` while the caller
  believed it had set 2. Anything introspecting an `nnx.Module` class must read
  `__init__` directly. This is also how `ansatz`, `n_layers` and `whitening` reach
  the quantum arm without the registry needing per-model special cases.
- **D71 A model may declare `preconditioner_features` when its preconditioner does not see
  the raw features.** The quantum arm's preconditioner acts on the two pair-polar
  angles, not the four input components, so building it at `F = 4` would raise a
  shape error at the first call. The attribute keeps that knowledge in the model
  that owns it rather than in the trainer.
- **D72 The particle mask comes from `n_fsps`, never from the features.**
  `dataset_statistics` uses `E > 0` as a real-particle heuristic, fine for a
  histogram but wrong as a contract -- a genuine zero-energy particle would be
  masked out. Padding puts real particles in the first `n_fsps` rows (D17), so
  `arange(L) < n_fsps[:, None]` is exact.
- **D73 Training drops the partial tail batch; evaluation keeps it.** Static
  shapes are what let the jitted train step compile once, and a dropped partial
  batch is resampled next epoch. Scoring must cover every event, so
  `evaluate_split` accepts one extra compilation for the remainder.
- **D74 `npz_to_state` validates the key set and per-parameter shapes before
  writing.** `nnx.update` accepts a mismatched pytree and the failure surfaces
  much later inside a jitted step as an opaque shape error with no mention of the
  checkpoint. Loading a `dim=8` checkpoint into a `dim=16` model now names the
  offending parameter.
- **D75 An undefined metric travels as `None` in a record, and is omitted from a
  stream.** A subset with no events -- the `unknown` split at `n_groups=1` -- has
  no accuracy to report. Dropping the key from a record would make its shape depend
  on the dataset and break cross-run comparison, so `evaluate` and `fit` publish
  `None` there instead, via `train.jsonable`. A *stream* port cannot do that:
  Fluksio's ports reject any non-finite float (rightly -- JSON cannot spell one)
  and a `float` port rejects `None` too, so the `g_purity` stream is simply not
  emitted for a model that encodes no quantum state. Both halves are pinned by
  `tests/test_train.py::test_node_payloads_are_port_legal`, which checks our
  payloads against the real port specs rather than a copy of the rule.
- **D76 The DLA certificate is a required input of `fit`, not a convention.** The
  `dla_report` node runs upstream in the `train` flow and its record is passed
  into `fit`, which stores it in `final_metrics`. The ROADMAP asks for the DLA to
  be "recorded before any training"; making it an edge in the flow graph is what
  enforces that, rather than trusting anyone to run it first.
- **D77 The g-purity is measured on a fixed validation subset, not a fresh sample
  each epoch.** Otherwise the series would track sampling noise as well as the
  model, and the phase-4 claim is specifically about what *training* does to the
  encoded distribution. It costs `O(n)` per edge, so it is streamed every epoch.
  Models that encode no quantum state report NaN on that port.

## Review fixes (post phase-1-4 verification)

Decisions taken while acting on the phase-1-4 review. Each names the thing that was
wrong and what replaced it, because several of these correct a claim that had already
been written down as a result.

- **D78 `[user]` the g-purity observable is the *encoded angle distribution*, and
  its argument is `u`, never `n_layers * u`.** Three definitions were in play:
  `qfm.g_purity` evaluated the closed form at `n_layers * angles`, the whitening
  acceptance test at the raw angles, and the RESEARCH table at the raw angles. So
  the tracked series and the test that gated the rotation were a factor of
  `n_layers` apart, and the headline table did not describe the arm in use.
  Resolved in favour of the raw angle, on the unflattening manuscript's own
  grounds: under re-uploading the closed forms describe *the state entering the
  first trainable block*, because later encoding layers act on parameter-dependent
  entangled states and are not product states at all. A closed-form purity is
  therefore a property of data plus encoding, not of a trained circuit.
  Consequence: every purity number in RESEARCH.md predates the fix and has been
  re-measured -- and the re-measurement changed a reported result, not just its
  scale. Under the corrected convention the learned preconditioner drives the XY arm's
  purity *down* (1.247 -> 0.178 -> 0.436 over 15 epochs) rather than up across
  `mu_4` as previously written, and on the clustered legacy arm it collapses by an
  order of magnitude while accuracy degrades. Provisional at this scale, but it is
  now an open question rather than a confirmation (RESEARCH.md 3).
- **D79 the two pair-polar angles are not equivalent, and the docstring said they
  were.** `phi = atan2(py, px)` covers `[0, 2pi)`, but `alpha = atan2(E, pz)`
  cannot leave `(0, pi)` (energy is positive) and `E >= |p| >= |pz|` confines it to
  about `[pi/4, 3pi/4]`, so the `mod 2pi` is a no-op on half the qubits. Measured
  soft-versus-hard quartile means 1.545 and 1.516: the concentration at `pi/2` is
  the kinematic bound, **not** a softness effect as first written.
- **D80 `[user]` the legacy `p * E * pi` encoding is added as a fourth, deliberately
  clustered input arm.** The ROADMAP's premise was that kinematic features cluster
  the encoding angles naturally, which is where the unflattening prediction is
  falsifiable. Measurement said otherwise for both replacement encodings (D61), so
  without a clustered arm the phase-4 rescue experiment has no data in the regime
  it is about. `legacy_angles` keeps two of partiqlegan's three product angles
  (`px E pi`, `pz E pi`) -- the edge QFM spends two qubits per particle, not three
  -- and the `"legacy"` encoding normalises by the **maximum** rather than the mean
  so both factors really do lie in the unit interval, which is the whole mechanism.
  Measured through the full path: mean `P_g` 0.16 with 85% of edges below
  threshold, against 1.24 and 15% for pair-polar.
- **D81 the whitening rotation travels inside the checkpoint config.** It is not an
  `nnx.Param`, so `state_to_npz` never stored it, and `evaluate` rebuilt every
  whitened arm as the raw one and scored it on unrotated angles -- silently, for
  three of the nine phase-4 cells. Sixteen floats as json keeps the checkpoint
  self-describing (D68) and needs no second artifact; `QFMConstellation` now
  coerces before validating so a nested list is accepted.
- **D84 the preconditioner gets its own rng stream.** Built from the model's, its three
  draws shifted every later draw, so `preconditioner="mlp"` did not merely attach a front
  end -- it re-initialised the whole model. The raw-versus-learned comparison is
  the phase-4 experiment, so an initialisation difference sitting inside it is a
  confound, not a detail. Offset `1 << 20`, and a test pins the shared parameters
  equal across the two arms.
- **D85 `valid_tree_rate` gains a strict variant, and it is the primary number.**
  The lenient definition is permissive twice over: greedy reconstruction accepts
  matrices consistent with no single tree (D42, already known), *and* dropping the
  leaves a prediction calls disconnected lets a prediction that keeps one pair
  score 1.0 -- with nothing in the loss to discourage it, since class 0 is never a
  target and carries weight 0 (D49). Measured: an all-zero-but-one-pair prediction
  on five leaves scores 1.0 lenient, 0.0 strict. `strict=True` requires every
  scored leaf to survive and the reconstructed tree to re-derive its own LCAG
  (`lcag_roundtrip`). Both are reported; the lenient one stays for comparability.
- **D86 parameter matching is computed, not asserted.** The docs called
  `dim=8, n_blocks=1` the parameter-matched arm; it has 1115 parameters against the
  quantum arm's 70, a factor of sixteen. `matched_dim` bisects for the width whose
  count is closest -- `dim=1, n_blocks=3` at 67 -- and a test asserts the match is
  within 10%. Worth stating in the paper rather than hiding: at 70 parameters a
  classical GNN is necessarily width-1, which is itself part of the comparison.
- **D87 `numpy>=2.3`.** `np.savez(..., allow_pickle=False)` relies on
  `allow_pickle` being a real keyword, which arrived in 2.3. On 2.0-2.2 it is
  swallowed into `**kwds` and written as a stray array, which `npz_to_state`'s key
  check would then reject -- a failure at load time, on a machine that resolved an
  older numpy.
- **D88 the encoding comparison is a flow node, not a one-off measurement.** The
  RESEARCH table compared three encodings, but two of them existed nowhere in the
  repo or in its history, so the project's cleanest empirical claim was not
  reproducible from committed code. `analysis.encoding_purity` scores all three
  arms on one shared edge sample (so they differ by encoding and nothing else) and
  `encoding_report` runs it in the `generate` flow.

## Running the experiments

- **D89 the long-running nodes declare a silence `timeout`.** The engine kills a
  node that has been quiet for 30s, and `build_dataset` is quiet by nature: it
  produces one artifact at the end and nothing in between, for as long as
  phase-space generation takes. Three earlier `generate` runs died that way. The
  `@node` docstring says the inherited default is "no limit", so the 30s is either
  an engine setting or a doc mismatch -- either way the fix is to say what quiet
  means per node, which is what `timeout` is for ("set one where silence means
  stuck rather than working"). Set on `build_dataset` (24h), `evaluate` (12h),
  `fit` (2h, covering the first epoch's jit compilation rather than a steady one),
  `whitening_rotation` (1h), `dataset_stats`, `dla_report` and `encoding_report`
  (30m). Nothing here is a workaround for a Fluksio limitation; the alternative
  would be to stream progress out of generation, which is a larger change to a
  function the tests call directly.

## Generation robustness

- **D90 an ungeneratable topology is detected by probing, not by a mass rule, and
  the sampler resamples it.** Generating the larger dataset died part-way through
  with "unweighting did not reach 500 events in 100 rounds ... measured acceptance
  rate 6e-7". Three changes, in the order the reasoning went:

  1. **The obvious guard is wrong.** The intuition was that a decay whose daughters
     nearly saturate the parent mass has no phase space, so a minimum released-energy
     ratio would filter them. Measured over 60 sampled topologies, `Q/M` barely
     predicts the acceptance rate at all -- at `Q/M >= 0.20` the worst rate is still
     zero in 4096 draws -- and the actual worst case runs the other way: a *wide*
     mass spread (`100 -> 2 + 25`, `Q/M = 0.73`). With `normalize_weights=True`
     phasespace divides by the maximum attainable weight, so what suppresses
     acceptance is a peaked weight distribution, which a mass ratio does not see.
     No mass rule was added.
  2. **`generate_events` spends a draw budget, not a round count** (`max_draws`,
     default 40M, replacing `max_rounds=100`) and raises a named
     `UngeneratableTopologyError`. "Ungeneratable" is thereby defined as the thing
     the caller actually cares about -- more draws than we are willing to spend --
     rather than as an arbitrary number of chunks whose size varies by orders of
     magnitude anyway.
  3. **`sample_topologies` takes an `is_viable` predicate**, and `assemble_dataset`
     supplies one that probes each candidate with a short generation
     (`probe_events=32`, `probe_draws=2M`). The rejection has to live in the sampler
     because that is where shape uniqueness is tracked (D82); the predicate is
     *injected* so `topology.py` stays free of the generator and of phasespace.

  **Caveat that belongs in the paper**: this biases the topology distribution toward
  decays that are cheap to sample. The bias is in the sampling of *kinematics*, not
  of tree shapes, so the LCAG label distribution is untouched -- but the statement
  "topologies are drawn uniformly from the mass pools" is no longer exactly true.

- **D91 the whitening rotation is fitted on the encoding it will be applied to.**
  `whitening_rotation` read `features_cartesian` unconditionally while `fit` applied
  the result to whatever `encoding` the run selected. Adding the `legacy` encoding
  (D80) made that mismatch reachable and measurable: the whitened legacy arm scored
  a *lower* g-purity than the raw one (0.297 against 0.397), because a rotation
  accepted on one encoding's four-vectors says nothing about another's. The node
  now takes an `encoding` port, fits on that split's features, records it in the
  report, and rejects `"angles"` outright (D56 -- it has no four-vectors to rotate).
  Every whitened-arm number measured before this is void.

- **D92 the angle distribution is measured beside the purity, per qubit.** The
  g-purity answers "is this encoded state trainable" and the ROADMAP's own
  hypothesis -- "the input distribution flattening can be observed" -- asks
  something else that the purity cannot answer. Worse, the two readings of a
  rising purity are opposite: `P_offdiag` is built from `sin^2(theta)` factors, so
  it climbs when the angles *spread* toward uniform and also when they *pin* near
  `pi/2`, which is the true maximum `n - 1` and the configuration the unflattening
  manuscript notes destroys the input information. `analysis.angle_stats` reports
  `tv_uniform`, `mean_sin2` and `circular_variance`; `mean_sin2` is the
  discriminator, measuring 0.5 / 1.0 / 0.0 for uniform / pinned / clustered laws
  where the total variation cannot tell the last two apart (both ~0.94).

  Three properties of the record, each load-bearing:
  1. **Per qubit, never pooled.** Sites peaking at different angles average into
     something that looks flat, which is the artefact the unflattening latent-drift
     memo warns about; a test pins that the pooled reading is misleadingly low.
  2. **`tv_uniform` has a floor set by kinematics, not by training.** The
     `(p_z, E)` sites cannot leave `(0, pi)` and sit inside about `[pi/4, 3pi/4]`
     (D79), so those qubits can never be uniform however the preconditioner moves them.
     Read a run against the *raw* arm at the same site, not against zero.
  3. **`TV_BINS` is fixed at 36 and reported.** A total variation over a histogram
     is meaningless without its resolution, and two runs only compare at the same one.

  Not factored with `dataset_statistics`' circular variance despite the shared
  one-line formula: that one describes raw dataset features, this one the latent
  angles a specific model encodes, and coupling `data/` to `analysis/` to share a
  single expression would cost more than it saves.

- **D93 `fit` opts out of node caching.** Its fingerprint covers its own source,
  not `train_model` where the loop lives, so the D69 split that makes the loop
  testable also makes the cache blind to every change in it. It bit immediately:
  the first re-run with the D92 observable came back from cache with no
  `angle_stats` and empty streams, and would have been read as "the measurement
  did not fire" rather than "this is a pre-change result". `cache=False` on `fit`
  only -- generation, whitening and the DLA check stay cached, because they are the
  expensive ones and their helpers change rarely. Re-running a fit is a few
  minutes; silently reporting a stale one is a wrong result in a paper.

- **D94 the whitening node falls back rather than failing a run it cannot serve.**
  D91's fix put the encoding check in the wrong place. `whitening_rotation` runs
  for *every* run so its acceptance report is always recorded, so rejecting an
  encoding with no four-vectors did not guard the whitening arm -- it broke every
  classical run on the `angles` encoding, which never wanted a rotation in the
  first place. Caught immediately: three of four cells in the ceiling check failed
  with it. The node now fits on the training encoding when that encoding has
  four-vectors and falls back to `cartesian` otherwise, recording `fitted_on` and
  `applicable` in the report. D91's substance survives untouched, because the only
  consumer of the rotation is the QFM and the QFM accepts four-vectors alone (D56):
  wherever a rotation is *applied*, it was fitted on exactly the features the
  circuit encodes.

  General lesson worth keeping: **validate at the point of use, not at the point of
  production**, when a node produces something optional for the rest of the flow.

- **D95 `[user]` the preconditioner stays frequency-neutral, and trainable frequencies stay
  out of every arm for now.** The question came up whether the MLP already gives
  trainable frequencies or a richer spectrum "for free". It does not, and the
  distinction matters enough to record:

  * A trainable frequency is a *linear* rescale `RY(w x)`. The preconditioner is a
    nonlinear elementwise residual warp, so the composite is not a trigonometric
    polynomial in `x` at all -- it has no discrete spectrum to speak of. In `phi`
    the circuit's frequencies are untouched.
  * It cannot reduce spectral redundancy either. The degeneracy `Omega-hat` is fixed
    by the encoding generators and the ansatz; the preconditioner changes which input
    maps to which `phi`, not how many degenerate terms exist. A nonlinear preconditioner
    actually *breaks* the framework the FCC is defined in, which is why the
    fingerprints manuscript lists it as an open question rather than a tool.
  * It is elementwise per feature and applied before `node2edge`, so it could not
    give different weights to one feature across several qubits even if it were
    linear -- which is exactly what a ternary comb is.

  Kept deliberately, because frequency-neutrality is what isolates *distribution
  shaping* as the measured effect and leaves the QFM solely responsible for the
  frequency content. Richer spectra go through `enc_params` / `Encoding` in a
  separate arm (ROADMAP 4b arm B), with fixed weights; `trainable_frequencies`
  stays off everywhere until fixed-weight expressivity is settled, since D9's
  optimisation is already fragile and learned encoding weights add Fourier locking.

- **D96 the measured spectrum of the current encoding, and why the ansatz set cannot
  be chosen on symmetry alone.** Recorded because both facts are cheap to measure
  and easy to assume wrongly:

  * Per-feature spectrum is `2L + 1`, growing *linearly* in depth -- measured
    (3,3,3,3) at `n_layers=1`, (5,5,5,5) at 2, (9,9,9,9) at 4. At our default each
    angle enters as a degree-2 trigonometric polynomial.
  * `Permutation_Equivariant` -- the obvious answer to the edge function's broken
    endpoint symmetry -- saturates `su(2^n)` (255/255, `d_A` 15), so it is
    input-distribution independent by construction and cannot carry the
    preconditioning question. **Symmetry does not imply a benign algebra**, and the
    ansatz arm has to be selected against the DLA criterion as well as the physical
    one.

## Split hygiene

- **D82 topologies are deduplicated on their *unlabelled shape*, not their
  mass-labelled canonical form.** The LCAG label depends only on the shape, and
  masses are not model inputs, so mass-labelled dedup let group C repeat labels
  group A had trained on: measured 26% of "unseen" topologies over 20 seeds. Now 0%.
  The fix exposed the constraint the old key was hiding -- distinct shapes are
  genuinely scarce at small leaf counts (at `max_depth=4`: 2 shapes at 3 leaves, 4
  at 4, 8 at 5, 15 at 6) -- so a slot whose leaf count is exhausted falls through
  to another count rather than failing. **This bounds how many truly-unseen
  topologies a dataset can hold, and it has to be stated when the probe is read.**
- **D83 groups are dealt round-robin from the draw sorted by leaf count.** Slicing
  the draw order interacted with D82's fallthrough to hand group A the scarce small
  trees and group C only large ones, so the known/unknown probe would have compared
  multiplicities rather than familiarity. Dealing gives the groups near-identical
  count profiles. The `per_group >= span` guard stays: below it a group cannot
  cover the range at all.

## End-to-end verification

Run on a generated dataset (720 train / 540 val / 1440 test events, 3 topologies
per group, `max_fsps=6`, `n_classes=4`), 8 epochs each -- enough to show every arm
trains and every observable streams, far too few to mean anything scientifically:

| arm | params | train loss | test acc | valid-tree | g-purity |
|---|---|---|---|---|---|
| GNN `dim=64` unconstrained | 128 964 | 1.519 -> 0.508 | 0.347 | 0.685 | n/a |
| GNN `dim=8` parameter-matched | 1 116 | 0.882 -> 0.580 | 0.400 | 0.626 | n/a |
| MLP-only control | 28 | 1.352 -> 0.821 | 0.352 | 0.385 | n/a |
| QFM XY raw | 70 | 1.102 -> 0.840 | 0.417 | 0.794 | 0.883 (flat) |
| QFM XY whitened | 70 | 1.095 -> 0.833 | 0.390 | 0.434 | 0.828 (flat) |
| QFM XY + learned preconditioner | 166 | 0.969 -> 0.828 | 0.376 | 0.642 | **0.714 -> 0.930** |
| QFM Matchgate whitened | 76 | 1.048 -> 0.831 | 0.389 | 0.737 | 3.055 (flat) |
| QFM Circuit_19 raw | 106 | 0.997 -> 0.823 | 0.357 | 0.326 | 15.000 (flat) |

The one row that already says something: with a **learned** preconditioner the XY arm's
g-purity *rises* during training, from below `mu_4 = 0.8125` to above it, while
every fixed-encoding arm is flat by construction. That is the rescue dynamics the
unflattening work predicts, observable in this pipeline. Whether it survives on
the full dataset, and whether it buys accuracy, is what the real runs decide.

Also verified: the complete test suite (184 tests, `ruff format`, `ruff check` and
`ty` all clean), both Fluksio flows sync (`fluksio sync --dry-run partiqledtr`
renders shims for all six nodes), and an overfit smoke test drives the GNN to
1.000 per-element accuracy and 1.000 Perfect-LCAG on 8 events.

## Phase 4b: expressivity (arms A, B, C)

- **D94 One general product-state purity replaces the three hand-derived closed
  forms.** On an `RY` product state `<X> = sin(theta)`, `<Y> = 0`, `<Z> =
  cos(theta)`, so only the Y-free DLA basis words survive and
  `P_g = sum_B prod_{q in X(B)} sin^2 theta_q prod_{q in Z(B)} cos^2 theta_q`.
  `analysis.product_state_purity` evaluates that against the arm's own DLA basis:
  `O(|basis| * n)`, jittable, and it reproduces `g_purity_offdiag`,
  `g_purity_full` and `g_purity_su` to float32 (max |diff| 3e-6, tested at n = 2,
  3, 4).

  The reason it had to exist: **arm C's ansaetze have no published closed form.**
  A per-arm dispatch table cannot be extended to a new bond topology without
  deriving a new series by hand, which is exactly the step this removes.
  `G_PURITY_BY_ANSATZ` is gone; `g_purity_offdiag` and `offdiag_uniform_mean`
  stay in `analysis.py` because `encoding_purity` and the whitening acceptance
  test use them, and `g_purity_full` / `g_purity_su` moved into
  `tests/test_analysis.py` as independent oracles -- production carries only what
  it runs, and the manuscript's derivations still pin the general routine.
  `uniform_prior_mean(ansatz, n)` generalises `offdiag_uniform_mean` the same way:
  a Y-free word on `m` qubits contributes `2^-m`, and the two agree exactly for
  `XY_Brickwork` (0.8125 at n = 4).

- **D95 Arm B is a 2x3 factorisation (`enc_reupload` x `enc_weights`), not a
  single ternary switch.** The ROADMAP asked for "ternary encoding with the
  re-upload mask widened". Taken literally with `Encoding("ternary")` and a *full*
  mask, every qubit `q` would encode `RY(3^q * sum_f u_f)` -- the model would see
  only the sum of the four features, which destroys the input. The widening has to
  be partial, and once it is, two things change at once (spectrum size and feature
  mixing) and have to be separated:

  | weights \ mask | `diagonal` | `cyclic` (qubit q sees features q, q+1) |
  |---|---|---|
  | `hamming` | phase-3/4 baseline, 5 freq/feature | mixing only, 9, **not dissociated** |
  | `binary` | scaling only, 5 | dissociated, 9-13 |
  | `ternary` | scaling only, 5 | **dissociated, 17-25** -- arm B |

  Measured with `Encoding.get_spectrum`. The two off-diagonal cells are the
  controls that stop "ternary helped" from meaning "you multiplied by 27" or "you
  mixed the features". Dissociation is checked by exhaustion over all 80 nonzero
  `eps in {-1,0,1}^4` rather than cited: `hamming-cyclic` fails it, every other
  cell passes.

- **D96 The purity argument is `theta = W u`, where `W` is the effective encoding
  weight matrix.** Re-uploaded `RY` gates on one wire add, so one encoding layer
  rotates qubit `q` by `theta_q = sum_f W[q,f] u_f` with `W = mask * base^q`
  (`models.qfm.encoding_matrix`). The state stays an `RY` product state, so every
  purity form applies unchanged with `theta` in place of `u` -- D78's "never
  `L * u`" convention is untouched, only the argument is now weighted.
  `QFMConstellation.encoded_angles` is that step, and `g_purity` and `angle_stats`
  both read it, so the angle statistics describe what the qubits actually rotate
  by. The zero-parameter contract generalises from `cos(L u)` to `cos(L W u)` and
  is tested at every `(weights, mask, depth)` cell.

- **D97 The encoding-weight comparison runs before any model exists.** Spectral
  preconditioning is a property of data plus encoding, so `encoding_purity` now
  reports `report[feature_arm][weight_cell]` -- three feature encodings crossed
  with the six weight cells, 18 numbers from one pass over sampled edges, no
  training. If an exponential spectrum lifts the clustered `legacy` encoding off
  the floor, it shows there, and a training run cannot then be the *only* evidence
  for it. The report is keyed by the ansatz whose DLA basis it sums over
  (`purity_ansatz`, default `XY_Ring`), because the threshold `mu_n / 2` is
  arm-specific once arms are no longer all `so(n) (+) so(n)`.

- **D98 Arm C's arms come from `Topology`, and "retired" means out of the reported
  set, not out of the codebase.** The edge QFM spends two qubits per particle, so
  the endpoint swap acts on the wires as `pi = (0 2)(1 3)`; an ansatz respects the
  two-particle partition when its bond set is `pi`-invariant. Measured at n = 4:

  | ansatz | bonds | `dim_g`/255 | `d_Z` | `pi`-invariant | cross bonds |
  |---|---|---|---|---|---|
  | `XY_Brickwork` | 01, 12, 23 | 12 | 0 | **no** | 1/3 |
  | `XY_Ring` | 01, 23, 02, 13 | 24 | 0 | yes | 2/4 |
  | `XY_AllPairs` | all six | 60 | 6 | yes | 4/6 |
  | `Circuit_19` | CRX ring | 255 | 15 | yes | -- |

  `XY_Ring` is `Topology.bricks(offset=0)` (the intra-particle bonds) plus
  `Topology.stairs(span=2)` (the cross-particle ones), i.e. the 4-cycle
  `0-1-3-2-0`. Even cycle -> bipartite -> `d_Z = 0`, which is the manuscript's own
  graph criterion, so it is partition-respecting **and** still input-distribution
  sensitive -- the ROADMAP expected criteria 1 and 2 to pull apart and at n = 4
  they do not. `XY_AllPairs` adds the triangles and the floor returns at
  `d_Z = 6`, giving a floored control that also respects the partition, which is
  what let `Matchgate` be retired. Bond invariance is not parameter equivariance:
  `pi` swaps the two `bricks` bonds, so that block carries `shared=True` and
  `XY_Ring` is exactly equivariant under the endpoint swap, making D35's output
  averaging a no-op rather than a patch. `XY_AllPairs` ties nothing and is
  equivariant only in its bonds.

  Retired arms stay resolvable: `ansaetze.circuit` falls back to any qml-essentials
  ansatz, so `Matchgate` remains runnable and certifiable and `RESEARCH.md` §7 stays
  reproducible. New arms are project-local `DeclarativeCircuit` subclasses --
  `Model(circuit_type=...)` accepts a class, so no fork of qml-essentials is needed.

- **D99 Everything runs on the CPU; CUDA is an opt-in extra.** The GPU was
  installed and measured rather than assumed, and it loses on both workloads on
  this box (Tesla P100, jax 0.11.1):

  | workload | CPU | GPU |
  |---|---|---|
  | phase-space generation, 20k events | 6074 ev/s | 1206 ev/s |
  | QFM train step, `n_layers=2` | 2.5 ms | 2.6 ms |
  | QFM train step, `n_layers=16` | 2.2 ms | 19.3 ms |
  | GNN train step, `dim=64` | 13.7 ms | 14.1 ms |

  The arrays are small -- 16 amplitudes per edge QFM, a few thousand edges per
  batch -- so kernel launch dominates, and phasespace-jax needs float64. Worse, it
  is an active hazard: the engine starts one worker process per node and each JAX
  process preallocates 75% of the device, so three concurrent nodes deadlocked the
  first generation run in an allocator retry loop (0.4% CPU, 0% GPU utilisation,
  21 minutes, no progress). `jax[cuda12]` therefore sits in an optional `gpu`
  extra, and every engine and script runs with `JAX_PLATFORMS=cpu`.

- **D100 The encoding weights live in `enc_params`, and `ternary_pair` is the cell
  that composes with a partition-respecting ansatz.** Two changes, one cause.

  Exponential weights are dissociated *because* they distinguish the qubits, and
  the endpoint swap `pi = (0 2)(1 3)` exchanges the two particles' qubits -- so
  `3^q` is not `pi`-invariant and arm B as first written would have undone the
  equivariance arm C restores. Independent arms are not supposed to interact like
  that. The escape is that the manuscript's dissociation condition is on a weight
  *vector* acting on a scalar input (`sum_k eps_k w_k != 0`), while ours is on the
  weight *matrix* (`W^T eps != 0`), which the widened mask makes strictly weaker.
  `w = (1,3,1,3)` satisfies the matrix condition **and** is `pi`-invariant, at the
  same 17 frequencies per feature as full ternary. Verified by exhaustion over all
  80 nonzero `eps` in `{-1,0,1}^4`.

  qml-essentials has no such strategy, and it should not: it is specific to this
  model's two-qubits-per-particle layout. So the weights moved out of the
  `Encoding` strategy and into `model.enc_params`, which `_iec` multiplies the
  input by directly. One matrix -- `encoding_matrix` -- is now the whole of what
  arm B varies, any weighting is expressible, and `Encoding` stays at `hamming`
  (the identity wrapper). The cost is that `Model.get_spectrum` no longer knows
  the weights, so `encoding_spectrum` reads the comb off `W` instead; it is the
  same Minkowski sum, and it agrees with `Encoding.get_spectrum` on the cells that
  qml-essentials can express.

- **D101 The viability probe has to demand the acceptance rate the real generation
  needs.** D90's probe used a fixed budget -- 32 events in 2e6 draws, so it passed
  anything above 1.6e-5 -- while the real generation asks for 1000 events in 4e7
  draws and needs 2.5e-5. A topology in between passes the probe and then kills the
  run, which is exactly what happened: `UngeneratableTopologyError: unweighting
  reached 719/1000 events in 40961024 draws ... acceptance rate 1.76e-05`, ten
  minutes in, with nothing kept.

  The probe budget is now proportional to the real one,
  `max_draws * probe_events / (probe_margin * n_events_per_topology)`, capped by
  `probe_draws`. At the defaults that is 640 000 draws, so the probe demands
  5e-5 against the run's 2.5e-5. The `probe_margin` exists because the rate
  estimate from `probe_events = 32` events carries about 18% sampling noise; a
  factor of two puts the boundary roughly four sigma clear. `max_draws` is now an
  argument of `assemble_dataset` and passed to both calls, so the two budgets
  cannot drift apart again. This is the policy `RESEARCH.md` open question 6 asked
  for before a large offline generation runs unattended.

- **D102 Everything phase 4b records *about* its arms is a flow, not a number in a
  document.** The arms themselves always ran through Fluksio -- every cell is a
  `train` run with a commit stamp, a DLA certificate recorded upstream of the fit,
  and streamed per-epoch metrics. Three tables in `RESEARCH.md` §10 did not: the
  ansatz certificates, the encoding-cell characterisation and the sampler's
  distinct-shape ceiling were computed in scratch scripts and typed in.

  They are now the `characterize` flow -- `arm_report`, `encoding_cells`,
  `shape_ceiling` -- which consumes no dataset, is deterministic given its seed,
  and takes seconds. `arm_report` records each arm's `dim_g`, `d_Z`, uniform-prior
  mean, endpoint-swap invariance and bond set; `encoding_cells` records each cell's
  weight matrix, per-feature spectrum, dissociation (checked by exhaustion over all
  80 nonzero signs, not cited) and its purity under a uniform and two clustered
  angle laws; `shape_ceiling` records where the topology sampler runs out. First
  run: `1787767689395-9acacc69`.

  `bonds` and `swap_invariant` moved into `partiqledtr.ansaetze` so the node and
  the test read the same definition rather than each reimplementing the swap.

- **D103 The sweep driver survives a busy engine; the engine survives a busy
  sweep.** Both halves were learned by losing a sweep. Two independent faults:

  1. **Oversubscription starved the engine.** Fluksio starts one worker process per
     node, and each JAX process sizes its Eigen pool to all 20 cores. Five
     concurrent workers plus the generation node left the engine's own event loop
     unscheduled; the API stopped answering within 10 s, and every driver died on
     `httpx.ReadTimeout` with six of eighty-four runs finished. The engine now runs
     with `OMP_NUM_THREADS=2` and
     `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2`.
     Capping is not a compromise here, it is **faster**: 1.82 ms per train step
     against 3.42 ms uncapped, because a 16-amplitude circuit over a few thousand
     edges is far too small to amortise the thread synchronisation. The same
     reasoning as D99 -- these arrays are small, and parallelism costs more than it
     returns.
  2. **The driver had no tolerance and no memory.** `Client()` defaults to a 30 s
     read timeout, one slow response killed the run, and because `fit` opts out of
     caching (D93) a restart retrained everything from zero. It now uses a 300 s
     timeout, retries engine calls with backoff, and looks up a finished `train`
     run with identical parameters before submitting -- so a driver that dies costs
     the runs in flight, not the arm.

  Consequence for scheduling: the 204k generation is **not** run concurrently with
  a sweep. It is one node that holds a slot for hours and competes for the same
  cores; sequential is both faster in total and the reason the engine stayed up.

- **D104 The arms run in process; the flows stay as the record of what they are.**
  Fluksio gave the study real things -- a run id and certificate per cell, streamed
  per-epoch metrics, cached generation -- but running 84 fits through it cost two
  sweeps to engine contention and client timeouts (D103, `NOTEPAD.md`), and the
  research is the point. `dev/s2-expressivity/run.py` now calls `train_model` and
  `evaluate_split` directly over a `ProcessPoolExecutor`, on splits exported to
  `/mnt/cache/partiqledtr/data`.

  Nothing was forked to do it. The nodes were always plain functions -- that split
  exists so they are testable without an engine -- so the driver calls the same
  code the `fit` node calls, and `partiqledtr/pipeline.py` still declares the same
  flows. Re-running the study through the engine needs no code change.

  Three properties of the flow were worth keeping by hand: the DLA certificate is
  computed *before* the fit (the flow enforced that by wiring `dla_report`
  upstream); results are written after every cell rather than at the end; and a
  restart skips cells already recorded, since `fit` opts out of caching. One
  process per cell, because JAX caches compiled executables per process and arm A's
  deep circuits would otherwise hold the whole scan's compilation cache.

  What is genuinely lost: the commit stamp per run, and the streamed metric series.
  The stamp was already not identifying the code (`NOTEPAD.md` item 3), and the
  per-epoch loss trace is kept in each record's `trace` field instead.

- **D106 The study lives in the repo, and the research record is tracked.** Run
  data sat on `/mnt/cache/partiqledtr` on the assumption it was large. Measured, the
  whole study -- engine store, dataset splits, every arm's results, figures and logs
  -- is **39 MB** against 22 GB free on the working disk, so the mount bought
  nothing and cost the study its self-containedness: absolute paths in three
  drivers, and a layout no one could reproduce without knowing about the mount.

  It now sits at the repo root: `.fluksio/` (the engine's own default data dir, so
  `serve` needs no `--data-dir` and there is no settings file to keep in step),
  plus `data/`, `results/`, `figures/`, `logs/`. The drivers resolve those from
  their own location rather than the working directory, so they behave the same
  started from anywhere. All five are gitignored: they are reproducible from
  `generate` plus `dev/`, and too churny to track.

  The same edit fixed something that had been wrong since the start. `.gitignore`
  carried `*.md` with only `!README.md`, so `RESEARCH.md`, `DECISIONS.md`,
  `ROADMAP.md` and `NOTEPAD.md` -- the entire research record, and the reason this
  repo exists -- were untracked, with no history and no restore point. They are
  tracked now; only `reference/` (vendored papers) and the run directories are not.

- **D107 The readout is the arm's own bond generators, `<XX_b> + <YY_b>` summed
  per coupling bond.** Supersedes the readout half of D26. The unflattening
  variance law `Var = P_g(rho) P_g(O) / dim g` needs the observable inside the
  arm's algebra, and single-qubit Z is in no XY arm's DLA (asserted per arm in
  `tests/test_qfm.py`, with the Z_q negative control) -- so the per-qubit Pauli-Z
  readout had `P_g(O) = 0` on every floor-free arm and the preconditioner's
  effect on the encoded state had no channel to the loss (`FINDINGS.md` §2,
  ROADMAP 4c item 1). D26's citation of Theorem 1 was wrong for the arms in use:
  the manuscript states it for `Z_i` only on the matchgate family, where
  `d_Z = n > 0` puts Z inside the algebra, and reads `X_i X_{i+1} + Y_i Y_{i+1}`
  on the off-diagonal family for exactly this reason (`exp_latent_drift`).

  The bond set is the arm's own coupling graph, read off the circuit structure
  (`ansaetze.bonds`) rather than assumed, and sorted: 3/4/6/4 bonds for
  `XY_Brickwork`/`XY_Ring`/`XY_AllPairs`/`Circuit_19`. The `Model` carries the
  `2 * n_bonds` interleaved strings `[XX_b, YY_b, ...]` and `_edges` sums each
  pair, because the summed operator is itself the in-algebra generator the law
  prices and because `<YY_b> = 0` identically on the RY product state at zero
  parameters -- unsummed features would start half-dead. D26's surviving
  constraints hold: the readout stays a vector (`n_bonds >= 3`, so no rank-1
  logits over `C > 2` classes) in a fixed order shared across edges.

  `Circuit_19` gets the same uniform bond rule rather than keeping per-qubit Z
  (which *is* in its `su(2^n)`): the cross-arm comparison should vary the
  algebra, not the algebra and the readout family at once, and continuity with
  phase 4b is already gone -- this change resets every quantum number, which is
  why 4c is a phase and not a patch. Consequence: `w_node` and `head` widths now
  follow `n_bonds`, so phase-4b QFM checkpoints no longer load (the shape check
  rejects them, correctly).

- **D108 Node-state widening is `n_channels` parallel QFMs per block, off by
  default.** ROADMAP 4c item 2's middle option. The inter-block node state was
  two numbers -- `w_node` maps to `N_ANGLES = 2` and a particle's whole hidden
  representation must be re-encodable as two angles -- against the classical
  GNN's 64. `n_channels = K` stacks K independently initialised parameter sets
  per block on a leading axis, widens `w_node` to `Linear(2 + K*n_bonds, 2K)` so
  each channel of block 2 encodes its own angle pair, and reads all `K*n_bonds`
  bond features into the head. Cross-particle structure still comes only from
  the QFMs (D27), edge sharing and equivariance are untouched, and `K = 1` is
  bit-identical to the previous architecture (channel c of block b seeds
  `seed + 2c + b`, so the K=1 draws are the old `(seed, seed+1)` pair and
  channel 0 of a wide model starts at the narrow model's parameters).

  Rejected alternatives: more qubits per edge (that is phase 6 -- it changes the
  algebra, the readout and the purity scale all at once), and a classical width
  channel beside the quantum one (weakens D27 and would have to be argued for).
  Channels are unrolled in the forward pass rather than vmapped: K stays small,
  and the unrolled form keeps the per-channel `Model.apply` calls identical to
  the K=1 path. `g_purity` is unchanged (the encoded state is channel-independent);
  `g_purity_exact` averages over the block's channels. The smoke dose is K=4
  (`dev/s3-readout-channel/`, `--channels 4`).

- **D109 `[user]` Per-group learning rates for the hybrid, as `lr_preconditioner`
  and `lr_qfm` overrides on one shared Adam.** The 4c item 4 confound, made
  testable. Adam equalises per-*parameter* step sizes (the update is
  `lr * m/sqrt(v)`, ~`lr` in steady state), so it erases the natural
  gradient-magnitude asymmetry between the encoder channel (`sigma^2`
  sensitivity) and the circuit channel (`sigma^4`) and forces both groups to
  march at the shared base rate whatever their curvature -- a shared rate is a
  shared trust region, and the two groups live on different geometries
  (unbounded MLP weights vs a 2pi-periodic bounded-curvature trig landscape).
  Implemented as `optax.multi_transform` over path-derived labels
  {preconditioner, qfm, classical}; all-`None` overrides reproduce the single
  Adam exactly, and a rate of zero provably freezes its group (test).

  **Rotosolve was considered for the circuit group and dropped** (user, after
  review): its closed-form jump assumes the loss is a single sinusoid per
  parameter, which fails here twice -- the XY gate generators have eigenvalue
  gaps {1, 2}, so even each expectation value carries two harmonics, and the
  softmax cross-entropy of the head output is not a trigonometric polynomial at
  all. Only an approximate 3-point variant would apply, at ~3x the evaluation
  cost per update for 144 shared parameters.

- **D110 The register is a parameter (`n_qubits`), and the phase-6 graph arms are
  a separate registry.** ROADMAP phase 6: `n_qubits` threads from
  `train_model`/`build_model` into the constellation, the masks, the readout and
  the checkpoint config (old checkpoints rebuild at the default 4). The angle
  maps declare their width (`ANGLE_WIDTHS`) and the constellation enforces
  `n_qubits == 2 * width` at construction, so a map/register mismatch fails
  before a jitted reshape can. `preconditioner_features` became a static method
  of the register for the same reason. The three `n = 6` arms (`XY_Cycle`,
  `XY_Ladder`, `XY_OddChord`, built on qml-essentials' new `Topology.graph`)
  live in `ANSAETZE_N6` rather than `ANSAETZE`: the phase-4b surfaces that
  iterate `ANSAETZE` at `n = 4` (`arm_report`, the parametrised tests) keep
  their meaning, and the arms themselves are explicit edge lists pinned to six
  qubits -- any other register raises, because the same list at `n > 6` would
  silently be a different graph and the certificates would lie. Certificates,
  measured: cycle 60/0, ladder 510/0, odd-chord 1020/30 (`dim_g`/`d_Z`), all
  invariant under the endpoint swap `pi = (0 3)(1 4)(2 5)`.

- **D111 `[user]` The s4 chart is `pair_polar_boost`, gated by the encoding
  report; the clustered `legacy` axis is dropped from s4.** The third
  per-particle angle was picked by the phase-4b discipline, not by tuning: the
  three candidates priced in g-purity on real kinematics before any training
  (`dev/s4-scaling/results/encoding.json`). `atan2(|p|, m)` wins at 1.87/1.43
  `mu_n` on the floor-free arms with 0% of edges below threshold;
  `atan2(p_T, p_z)` follows at 1.56/1.30; even `atan2(m, E)` sits at ~1.0 --
  the FSPs are not relativistic enough to cluster any chart, so this dataset
  offers no clustered three-angle arm. On normalised features the mass proxy is
  a deformed invariant (momenta and energy carry different scales, the same
  caveat that already widens `alpha`); the report prices the result, which is
  the decision criterion. Dropping `legacy` (user, 2026-08-31) defers the
  phase-6 annihilation/rescue prediction; the smoke grid tests the certificate
  -range, hardness and specificity predictions at 3 arms x {none, mlp} x 3
  seeds, smoke-first.

- **D112 `[user]` The node update is an axis (`node_update`), the re-encoding
  boundary has a scale (`node_omega`), and block 2's encoded distribution is
  instrumented.** The trig-interface program (user, 2026-09-02): the quantum
  arm's classical parts were purely linear, so the composite was
  trig-polynomial -> linear -> trig-polynomial -> linear with no classical
  nonlinear capacity, and the block-2 re-encoding boundary -- a sine of a linear
  map -- had no scale discipline (SIREN's own lever, arXiv:2006.09661; the
  QFM-SIREN bridge is QIREN, `LITERATURE.md`). `node_update` in
  {linear, siren, elu}: `linear` reproduces the old construction and rng draw
  order bit-for-bit; `siren` is a two-layer sine MLP with the SIREN first-layer
  init and `omega_0 = 30`; `elu` is the matched-parameter control that separates
  "a nonlinearity pays" from "the trigonometric one pays". Both keep the output
  layer linear, because the consumer re-encodes it as angles. `node_omega`
  scales the node state before block 2 encodes it (default 1.0, bit-identical);
  it applies inside `_node_state`, so the diagnostic and the circuit see the
  same thing. `block2_encoded_angles`/`block2_g_purity`/`block2_angle_stats`
  mirror the block-1 instrumentation one block deeper and land in
  `final_metrics` only -- the flow's declared stream ports are untouched. The
  init-time diagnostic then *falsified* the collapse hypothesis that motivated
  `node_omega` (block 2 starts at 0.92 mu_n, the uniform level), so the scale is
  a probe axis rather than a fix; `RESEARCH.md` §17 has both gate measurements.

- **D113 `[user]` Studies run versioned through the Fluksio engine; in-process
  drivers are the sandbox path.** (User, 2026-09-03.) The s3-s5 drivers ran
  `train_model` in process, so their cells carried no run id, commit stamp,
  params digest or streamed metrics -- the phase-4c/6 results are reproducible
  from the JSON records but not *versioned*. From the next experiment on, every
  study cell is one run of the `train` flow: the flow now carries every study
  axis (`n_channels` D108, `lr_preconditioner`/`lr_qfm` D109, `n_qubits` D110,
  `node_update`/`node_hidden`/`node_omega` D112), `dla_report` takes the
  register, and the s2 `run_arm_fluksio` pattern is the driver template --
  extended in s5 with `_trace`, which reassembles the per-epoch series from the
  run's streamed metrics so the correlation analysis needs no second code path.
  Verified end to end on two versioned smoke runs (gnn, and qfm at n=6 with
  siren + `lr_qfm`), certificates and configs intact. One fluksio limitation
  surfaced and is flagged rather than worked around (NOTEPAD.md 2026-09-03):
  nullable flow inputs are not expressible, so the two lr overrides ride as
  float ports with the flow-level contract "non-positive means share `lr`" --
  to revert once fluksio can register a null-initial input. The engine runs
  detached (`dev/serve.sh`, parented to init) with the store at `./.fluksio`.
