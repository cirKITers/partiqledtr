# PartiqleDTR

This repo facilitates to revive the former partiqlegan project.
The goal is to reconstruct intermediate decay products based on simulated decay events as reference data.

Tech stack:
- qml-essentials : for quantum Fourier models, simulated by its JAX backend jaqsi
- phasespace-jax : JAX port of phasespace, for decay event generation
- JAX : array computation and autodiff
- Flax : neural network modules
- Optax : optimizer and training loop
- Fluksio : data science pipeline and experiment tracking

References (`./reference/`, gitignored symlinks):
- baumbauen: classical gnn based approach with message passing
- partiqlegan: hybrid quantum-classical approach
- reconstructing-paper & improving-paper: paper corresponding to the hybrid approach
- fourier-fingerprints: correlations between frequency components (FCC) of QFMs as an inductive-bias descriptor
- unflattening: latest research concerning input distribution dependence for quantum circuits

Idea:
Create a new approach for the decay tree reconstruction problem by using quantum Fourier models (through `Model` in qml-essentials) in combination with a classical MLP.
This should serve as an application scenario for the `unflattening` paper, where we showed that the input distribution for a data re-uploading model get's whitened for some ansaetze when being combined with an MLP (and whitening is actually preferrable for these ansaetze).
Problem back then was that the simulation of a quantum model took an awful long time and therefore a purely quantum based gnn wasn't feasible.
However we showed that the qnn seems to be beneficial for the training.
While there is a low chance that we can acutally show an advantage compared to the purely classical case, it would already be interesting to research if the trigonometric properties of a QFM fit in the context of this given problem.
This formulates the main hypothesis:
> A constellation of small, shared-weight QFMs embedded in a message-passing architecture, with a constrained elementwise MLP preconditioner, can predict the LCAG of particle decay events.
> On floor-free polynomial-DLA ansaetze the encoder-channel rescue and g-purity dynamics predicted by the unflattening work are observable on naturally clustered kinematic inputs, while floored ansaetze show the predicted indifference.
> The ansatz FCC (fourier-fingerprints) acts as an inductive-bias descriptor for the task.

This is an inductive-bias study, not an advantage claim -- the relevant spectra admit
classical surrogates, and the honest question is whether the trigonometric structure
helps. `docs/` carries what has been measured against the hypothesis so far; several
of its clauses have since been amended.

Note:
Fluksio is a relatively new framework (developed by myself).
Documentation is available here: https://docs.fluksio.com/getting-started/data-science/
If we hit any limitations or encounter problems, we should stop and flag them in `docs/NOTEPAD.md` instead of trying workarounds.
Then Fluksio will be fixed and we can continue.
The same holds true for any limitations/issues with qml-essentials.

## Architecture

![the quantum model, end to end](docs/architecture.svg)

The classical `gnn` and `mlp` arms take the same features straight into their own
preconditioner and head; only the quantum arm is drawn.

The model is a constellation of 4-qubit QFMs used as the *edge function* of a
message-passing network, sharing one parameter set across every edge (which is
what makes it permutation-equivariant). Every classical part is particle-local --
elementwise preconditioner, parameter-free masked mean, per-node linear map -- so
cross-particle structure can only come from the quantum part. Readout is per-qubit
Pauli-Z plus a shared linear head, so nothing scales exponentially.

Three flows: `generate` runs the phase-space simulation once, `train` consumes its
artifacts and is the part a sweep repeats, and `characterize` records what the arms
*are* -- DLA certificates, encoding cells, the sampler's shape ceiling -- without
touching a dataset. Fluksio caches node results on their inputs, so re-running an
unchanged stage is nearly free. `fit` opts out (`cache=False`, `docs/DECISIONS.md`
D93).

## Layout

```
partiqledtr/    the model and its data generation -- nothing study-specific
├── data/       topology sampler · decay -> LCAG · features · phasespace generation
│               · dataset assembly + stats · whitening (phase 4)
├── models/     elementwise residual preconditioner · NRI message-passing GNN
│               · linear control · QFM constellation (phase 3)
├── ansaetze.py the ansatz arms and their bond structure (phase 4b arm C)
├── metrics.py  per-element / Perfect-LCAG / valid-tree rate, class weights
├── analysis.py g-purity (product-state + exact) · DLA pre-check · encoding comparison
├── train.py    loss, training loop, checkpoints, fit/evaluate nodes
└── pipeline.py the three Fluksio flows
dev/            the research: one folder per study, plus the engine script
├── serve.sh            the engine, on ./.fluksio with enough runs in flight
├── s1-encoding-purity/ encoding x weight g-purity, before anything trains
└── s2-expressivity/    phase 4b -- the three arms, the sweep, the figures
                        each study keeps its own data/ results/ figures/ logs/
docs/           the research record, and the diagram above
├── architecture.d2 / .svg
├── ROADMAP.md   the plan and the state of it
├── RESEARCH.md  the measurement history
├── FINDINGS.md  the claims
├── LITERATURE.md candidate references for the scaling study, pending review
├── ALGEBRA.md   open theory question: in-algebra encoding of pair invariants
├── DECISIONS.md why each implementation choice was made
└── NOTEPAD.md   what the tooling cost
tests/          run with `uv run pytest`
```

Everything a run reads or writes is gitignored: `.fluksio/` (the engine's store, at
the repo root, its own default location), the root `logs/` (the engine's own output),
and inside each study folder its `data/` (dataset splits exported from a `generate`
run), `results/` (one json per arm or report), `figures/` and `logs/`. All of it is
reproducible from `generate` plus `dev/`, so none of it is tracked -- but the research
record in `docs/` now is.

**Read `docs/FINDINGS.md` before designing runs** -- it is the current set of claims,
`docs/RESEARCH.md` the measurement history behind them, `docs/ROADMAP.md` the plan,
and `docs/DECISIONS.md` why each implementation choice was made.

## Getting started

Prerequisites: `uv sync`, and a Fluksio engine (`dev/serve.sh`, or add `--local`
to any command to boot one in-process). **Run the engine and every script with
`JAX_PLATFORMS=cpu`**: the arrays here are small enough that the GPU loses on both
generation and training, and with a CUDA jaxlib installed the engine's per-node
worker processes fight over the device (`docs/DECISIONS.md` D99). Pass
`--sync partiqledtr` to every `fluksio run`: the default sync root is the whole
directory, which walks the vendored `reference/` checkouts and aborts there
(`docs/NOTEPAD.md`). Node results are cached on their inputs, so re-running
something unchanged is nearly free; `--no-cache` forces re-execution.

**1. Generate a dataset.** Defaults are 10 topologies *per group* (30 total, in
three known/unknown groups) at 1000 events each:

```sh
fluksio run generate --sync partiqledtr --seed 0 --wait
```

Check the `stats` output before going further -- `split_integrity_ok` must be true.

**2. Train.** A dataset artifact is named on the command line by its digest, or by
the run that made it (`@run:<id>.dataset_train`); `dataset_meta` is a `json` input,
so it still has to be passed inline. The Python API hands both over directly:

```python
from partiqledtr.pipeline import generate, train

data = generate.submit(seed=0).wait().result
data_refs = {k: data[k] for k in ("dataset_train", "dataset_val", "dataset_test", "dataset_meta")}

run = train.submit(
    **data_refs, model="qfm", encoding="cartesian", ansatz="XY_Brickwork", epochs=100
).wait()

print(run.result["dla_report"])     # recorded before training
print(run.result["test_metrics"])   # overall / known / unknown
print(run.metrics("train.g_purity"))
```

The arms, all selected through flow inputs (`partiqledtr/pipeline.py` has every
input and its default):

| axis | values |
| --- | --- |
| `model` | `gnn`, `mlp`, `qfm` -- `qfm` requires `encoding="cartesian"` |
| `ansatz` | `XY_Brickwork`, `XY_Ring`, `XY_AllPairs`, `Circuit_19`; quantum arm only |
| `preconditioner` | `none`, `mlp` -- `mlp` is the learned elementwise preconditioner |
| `whiten` | `false`, `true` -- fixed isotropic preconditioning |
| `encoding` | `angles`, `cartesian`, `legacy` -- `cartesian` keeps `\|p\|`, needed by `qfm` |
| `angle_map` | `pair_polar`, `legacy` -- pair with `encoding=legacy` for the clustered arm |
| `n_layers` | int -- data-reuploading depth |
| `enc_weights` | `hamming`, `binary`, `ternary`, `ternary_pair` -- per-qubit encoding weight |
| `enc_reupload` | `diagonal`, `cyclic` -- which features reach which qubit |
| `dim`, `n_blocks` | ints -- GNN width/depth; use for parameter matching |

**3. Sweep.** An ablation cell is one run of the same flow, and since D113 that is
literal: studies submit their cells through the engine (each driver's `--fluksio`
flag), so every cell carries a run id, commit stamp, params digest and streamed
metrics; running a driver without the flag executes the same cells in process,
which is the sandbox path, not the record. The phase-4b arms are driven by
`dev/s2-expressivity/run.py`, which submits every cell at several seeds, keeps a
bounded number in flight and writes one JSON per arm:

```sh
python dev/s2-expressivity/run.py --arm a --fluksio <generate-run-id>
python dev/s2-expressivity/run.py --report
```

Dropping `--fluksio` runs the same cells in process instead, which is the sandbox
path rather than a fork of the flow. `dev/s2-expressivity/README.md` has the whole
study; `dev/s1-encoding-purity/README.md` the one that needs no training at all.

`fluksio sweep train --param ansatz=A,B --param preconditioner=none,mlp` is the CLI
equivalent for any other grid and takes it directly; the dataset inputs come along
as digests (`--dataset_train sha256:...`) with `dataset_meta` inline.

**Without an engine.** The nodes are plain functions, so `assemble_dataset(...)`
and `train_model(...)` can be called directly, which is what the test suite does.
Run the tests with `uv run pytest` (`-m "not gen"` skips the ones that run
phase-space generation).
