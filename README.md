# PartiqleDTR

This repo facilitates to revive the former partiqlegan project.
The goal is to reconstruct intermediate decay products based on simulated decay events as reference data.

Tech stack:
- qml-essentials : for quantum Fourier models and JAX based simulation
- JAX : array computation and autodiff
- Flax : neural network modules
- Optax : optimizer and training loop
- Fluksio : data science pipeline and experiment tracking

References (`./references/`):
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
> A constellation of small, shared-weight QFMs embedded in a message-passing architecture, with a constrained elementwise MLP front end, can predict the LCAG of particle decay events.
> On floor-free polynomial-DLA ansaetze the encoder-channel rescue and g-purity dynamics predicted by the unflattening work are observable on naturally clustered kinematic inputs, while floored ansaetze show the predicted indifference.
> The ansatz FCC (fourier-fingerprints) acts as an inductive-bias descriptor for the task.

Note:
Fluksio is a relatively new framework (developed by myself).
Documentation is available here: https://docs.fluksio.com/getting-started/data-science/
If we hit any limitations or encounter problems, we should stop and flag them in `NOTEPAD.md` instead of trying workarounds.
Then Fluksio will be fixed and we can continue.
The same holds true for any limitations/issues with qml-essentials.

## Architecture

```
partiqledtr/
├── data/       topology sampler · decay -> LCAG · features · phasespace generation
│               · dataset assembly + stats · whitening (phase 4)
├── models/     elementwise residual front end · NRI message-passing GNN
│               · linear control · QFM constellation (phase 3)
├── metrics.py  per-element / Perfect-LCAG / valid-tree rate, class weights
├── analysis.py g-purity closed forms · DLA pre-check
├── train.py    loss, training loop, checkpoints, fit/evaluate nodes
└── pipeline.py the two Fluksio flows
```

The model is a constellation of 4-qubit QFMs used as the *edge function* of a
message-passing network, sharing one parameter set across every edge (which is
what makes it permutation-equivariant). Every classical part is particle-local --
elementwise front end, parameter-free masked mean, per-node linear map -- so
cross-particle structure can only come from the quantum part. Readout is per-qubit
Pauli-Z plus a shared linear head, so nothing scales exponentially.

Two flows: `generate` runs the phase-space simulation once, `train` consumes its
artifacts and is the part a sweep repeats. `data/generation.py` is the only module
that imports TensorFlow, so nothing else pays that import.

**Read `RESEARCH.md` before designing runs** -- the measurements there revise one of
the premises below (see *Theoretical Motivation*), and `DECISIONS.md` records why
each implementation choice was made.

## Running the experiments

Prerequisites: `uv sync`, and a Fluksio engine (`fluksio serve`, or add `--local`
to any command to boot one in-process). `fluksio run` syncs the flows first, so
there is no separate upload step. Node results are cached on their inputs, so
re-running something unchanged is nearly free; `--no-cache` forces re-execution.

**1. Generate a dataset.** Defaults are 10 topologies *per group* (30 total, in
three known/unknown groups) at 1000 events each:

```sh
fluksio run generate --seed 0 --wait
```

Check the `stats` output before going further: `split_integrity_ok` must be true,
and the angle-marginal figures are the phase-1 evidence.

**2. Train.** Artifact references are JSON on the command line, so the Python API
is the practical way to hand a dataset to a training run:

```python
from partiqledtr.pipeline import generate, train

data = generate.submit(seed=0).wait().result
data_refs = {k: data[k] for k in ("dataset_train", "dataset_val", "dataset_test", "dataset_meta")}

run = train.submit(
    **data_refs, model="qfm", encoding="cartesian", ansatz="XY_Brickwork", epochs=100
).wait()

print(run.result["dla_report"])  # recorded before training
print(run.result["test_metrics"])  # overall / known / unknown
print(run.metrics("train.g_purity"))  # the phase-4 observable
```

The arms, all selected through flow inputs:

| axis | values | notes |
| --- | --- | --- |
| `model` | `gnn`, `mlp`, `qfm` | `qfm` requires `encoding="cartesian"` |
| `ansatz` | `XY_Brickwork`, `Matchgate`, `Circuit_19` | quantum arm only |
| `frontend` | `none`, `mlp` | `mlp` is the learned elementwise front end |
| `whiten` | `false`, `true` | fixed isotropic preconditioning |
| `encoding` | `angles`, `cartesian` | `cartesian` keeps `\|p\|`, needed by `qfm` |
| `dim`, `n_blocks` | ints | GNN width/depth; use for parameter matching |

The ROADMAP's three phase-4 input arms are `frontend=none, whiten=false` (raw),
`frontend=none, whiten=true` (fixed whitening) and `frontend=mlp, whiten=false`
(learned). The whitening rotation is fitted on the training split for every run
regardless, so its acceptance report is always recorded.

**3. Sweep the ablation matrix.** The nine-cell phase-4 study is nine runs of the
same flow. Because the dataset arrives as artifact references, drive it from
Python:

```python
from itertools import product

arms = [("none", False), ("none", True), ("mlp", False)]  # raw | whitened | learned
runs = [
    train.submit(
        **data_refs,
        model="qfm",
        encoding="cartesian",
        ansatz=ansatz,
        frontend=fe,
        whiten=wh,
        epochs=100,
    )
    for ansatz, (fe, wh) in product(["XY_Brickwork", "Matchgate", "Circuit_19"], arms)
]
for r in runs:
    r.wait()
```

`fluksio sweep train --param ansatz=A,B --param frontend=none,mlp` is the CLI
equivalent and takes a grid directly, but every required input still has to be
given, and an artifact one has to be spelled as JSON (`NOTEPAD.md`).

**4. Baselines.** Same `train.submit`, with `model="gnn", dim=64` for the
unconstrained GNN, `model="gnn", dim=8, n_blocks=1` for one parameter-matched to
the quantum arm (`n_params` is recorded in `final_metrics`), and `model="mlp"`
for the "MLP does everything" control. For a clean cross-arm comparison, also run
the matched GNN on `encoding="cartesian"`, the features the quantum arm sees.

Everything is also usable without an engine: the nodes are plain functions, so
`assemble_dataset(...)` and `train_model(...)` can be called directly, which is
what the test suite does. Run the tests with `uv run pytest` (`-m "not gen"`
skips the ones needing TensorFlow).

## Theoretical Motivation

A QFM realizes a truncated Fourier series in its encoded features, so its hypothesis
class is a trigonometric polynomial with a spectrum fixed by the encoding. Decay
kinematics fit this class naturally: momentum directions are periodic quantities, and
decay angular distributions are low-order expansions in spherical harmonics. Encoding
direction angles directly aligns the model's basis with the structure of the data — the
inductive-bias question this project probes.

Two prior results shape the design. First, the unflattening work shows that for
floor-free, polynomial-DLA ansaetze the input angle distribution decides trainability:
clustered angles annihilate the loss signal, while a classical front end rescues it
through the encoder channel. The expectation was that kinematic features cluster encoding
angles naturally (soft particles yield near-zero angles), putting this task in exactly the
regime where the theory makes falsifiable predictions — observable as g-purity dynamics
during training, with a floored (Matchgate) arm as control. **Measured, this holds for the
old `p·E·π` encoding and not for the replacements** (`RESEARCH.md` §1), which changes what
the input-distribution arm can claim. Second, the fourier-fingerprints work
provides the FCC as a cheap, hardware-compatible descriptor of an ansatz's coefficient
correlations; here it serves as an ansatz-selection metric, and tracking the spectrum
under a trainable front end addresses that paper's open question about nonlinear
classical preprocessing.

The architecture consequence: many small shared-weight QFMs inside message passing
(permutation-equivariant by construction, tractable spectra, cheap analytic simulation)
instead of the former one-qubit-per-particle monolith; an elementwise residual MLP as
front end so cross-particle structure must come from the quantum part. This is an
inductive-bias study, not an advantage claim — the relevant spectra admit classical
surrogates, and the honest question is whether the trigonometric structure helps.

See `ROADMAP.md` for the experimental plan.