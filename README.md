# PartiqleDTR

This is a revived version of the initial attempt to tackle the particle **d**ecay **t**ree **r**econstruction problem using a hybrid (quantum-classical) graph neural network architecture.
See the [corresponding section](#architecture) below for details concerning the current approach.
The project builds upon the same foundation as the [BaumBauen](https://github.com/helmholtz-ai-energy/baumbauen) project.

Technology:
- [qml-essentials](https://github.com/cirKITers/qml-essentials): quantum Fourier models, simulated with its JAX backend, jaqsi
- [phasespace-jax](https://github.com/cirKITers/phasespace-jax): JAX port of phasespace for decay event generation
- JAX: array computation and automatic differentiation
- Flax: neural network modules
- Optax: optimization and training
- Fluksio: data pipeline and experiment tracking

## Layout

```
partiqledtr/    model and data generation shared by all studies
├── data/       topology sampler · decay -> LCAG · features · phasespace generation
│               · dataset assembly + stats · whitening
├── models/     elementwise residual preconditioner · NRI message-passing GNN
│               · linear control · QFM constellation
├── ansaetze.py the ansatz arms and their bond structure
├── metrics.py  per-element / Perfect-LCAG / valid-tree rate, class weights
├── analysis.py g-purity (product-state + exact) · DLA pre-check · encoding comparison
├── train.py    loss, training loop, checkpoints, fit/evaluate nodes
└── pipeline.py the three Fluksio flows
dev/            one folder per study, plus the engine script
├── serve.sh            the engine, using ./.fluksio
├── s1-encoding-purity/ encoding and weight g-purity before training
├── s2-expressivity/    depth, encoding weights and ansatz at n = 4
├── s3-readout-channel/ in-algebra readout, widening, split learning rates
├── s4-scaling/         the graph trichotomy at n = 6
└── s5-trig-nodes/      the node update between message-passing blocks
                        each study has its own data/ results/ figures/ logs/
docs/           the diagram above (architecture.d2 / .svg)
tests/          run with `uv run pytest`
```

## Getting started

Run `uv sync`, then start a Fluksio engine with `dev/serve.sh` or add `--local` to commands to start one in process.
Set `JAX_PLATFORMS=cpu` for the engine and all scripts: these small arrays run faster on CPU, and engine workers contend for a CUDA device.
Pass `--sync partiqledtr` to every `fluksio run` so syncing skips the vendored `reference/` checkouts. Use `--no-cache` to force a stage to rerun.

**1. Generate a dataset.** By default, this generates 1,000 events for each of
10 topologies in each of three topology groups (30 topologies total):

```sh
fluksio run generate --sync partiqledtr --seed 0 --wait
```

Check that `split_integrity_ok` is true in the `stats` output.

**2. Train.** On the command line, identify a dataset artifact by its digest or
source run (`@run:<id>.dataset_train`). Pass the JSON `dataset_meta` input inline.
The Python API passes both directly:

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

Select arms through flow inputs; `partiqledtr/pipeline.py` defines their defaults:

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

**3. Sweep.** Each ablation cell runs the `train` flow. With `--fluksio`, study drivers submit cells through the engine, recording a run ID, commit stamp, parameter digest, and streamed metrics. 
The s2 driver runs each cell at several seeds, limits concurrent runs, and writes one JSON file per arm:

```sh
python dev/s2-expressivity/run.py --arm a --fluksio <generate-run-id>
python dev/s2-expressivity/run.py --report
```

Without `--fluksio`, the driver runs the same cells in process for exploration.
See `dev/s2-expressivity/README.md` for the full study and
`dev/s1-encoding-purity/README.md` for a study that needs no training.

For other grids, `fluksio sweep train` accepts parameters such as
`--param ansatz=A,B --param preconditioner=none,mlp`. Pass dataset inputs by
digest (for example, `--dataset_train sha256:...`) and `dataset_meta` inline.

## Architecture

![the quantum model, end to end](docs/architecture.svg)

The diagram shows only the quantum arm. The classical `gnn` and `mlp` arms use the same features with their own preconditioner and head.

The model uses 4-qubit QFMs as the *edge function* of a message-passing network.
They share parameters across edges, making the model permutation-equivariant.
The classical components are particle-local: an elementwise preconditioner, a parameter-free masked mean, and a per-node linear map. Cross-particle structure therefore comes from the quantum component.
Readout uses per-qubit Pauli-Z measurements and a shared linear head, avoiding exponential readout size.

The project is organized in three flows: `generate` simulates events, `train` consumes those artifacts for each sweep cell, and `characterize` records DLA certificates, encoding cells, and the sampler's shape ceiling without a dataset.
Fluksio caches node results by input, making unchanged stages cheap to rerun. `fit` uses `cache=False` because its fingerprint covers only its own source, not the training loop it calls.