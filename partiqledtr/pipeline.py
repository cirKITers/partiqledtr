"""Fluksio flow declarations.

Three flows rather than one. Fluksio caches node results on their inputs, so a merged
flow would skip regeneration anyway, but the split still earns its place: generation
and training have different parameter axes, generation is expensive and offline, and
one dataset feeds many training runs.

``characterize`` is the third, and it consumes no data at all. Everything phase 4b
records *about* its arms rather than *from* them -- each ansatz's DLA certificate,
floor count, prior scale and endpoint symmetry; each encoding cell's weight matrix,
spectrum, dissociation and synthetic purity; the sampler's distinct-shape ceiling --
belongs to a run with a commit stamp, not to a number pasted into a document
(``DECISIONS.md`` D102). It is cheap and deterministic, so re-running it after any
change to an arm is close to free.

    fluksio serve                                     # the engine, once
    fluksio sync partiqledtr                          # upload the flows
    fluksio run generate --seed 0 --wait
    fluksio run train --dataset_train <ref> --dataset_val <ref> \
        --dataset_test <ref> --dataset_meta <ref> --model gnn --epochs 100

``generate`` also runs ``encoding_report``, which prices each candidate encoding in
g-purity (D88), crossed with every encoding-weight arm of phase 4b (D97): whether a
decay-tree model's inputs land in the barren regime at all is decided there, before
any model exists, and the table is a result in its own right.

The ``train`` flow also carries the phase-3/4 instrumentation. ``dla_report`` runs
upstream of ``fit`` and its certificate is a *required* input there, so the arm's
dynamical Lie algebra and floor count are recorded before any training happens --
enforced by the flow's shape rather than by anyone remembering. ``whitening_rotation``
fits the fixed isotropic preconditioning on the training split and records its
acceptance report for *every* run; the model applies it only when ``whiten`` is set,
independently of ``preconditioner`` -- ``preconditioner=mlp`` with ``whiten=true`` is a reachable
(and deliberately runnable) cell.

An ablation cell is one run of this one flow:

    for arm in XY_Brickwork XY_Ring XY_AllPairs Circuit_19; do
      for fe in raw whiten mlp; do
        fluksio run train --model qfm --encoding cartesian --ansatz $arm ...
      done
    done

The clustered control arm is the same flow with ``--encoding legacy --angle_map
legacy``: partiqlegan's ``p * E * pi`` product, whose encoding angles collapse toward
zero, which is where the unflattening rescue prediction is falsifiable (D80).

Phase 4b adds three axes to the same flow and nothing else: ``n_layers`` (arm A),
``enc_weights`` x ``enc_reupload`` (arm B) and the new ``ansatz`` arms (arm C).

Declarations only: the nodes live in :mod:`partiqledtr.data.dataset`,
:mod:`partiqledtr.data.whitening`, :mod:`partiqledtr.analysis` and
:mod:`partiqledtr.train`, and are wired by matching provides/requires names.
"""

from fluksio import Flow, Port

from partiqledtr.analysis import arm_report, dla_report, encoding_cells, encoding_report
from partiqledtr.data.dataset import build_dataset, dataset_stats, shape_ceiling
from partiqledtr.data.whitening import whitening_rotation
from partiqledtr.train import evaluate, fit

generate = Flow(
    "generate",
    title="Generate the decay dataset",
    nodes=[build_dataset, dataset_stats, encoding_report],
    inputs=[
        Port("seed", "int", initial=0),
        # Per group, so the default is 30 topologies over the three known/unknown
        # groups -- the ROADMAP's ">=10 topologies, ~1000 events/topology".
        Port("n_topologies", "int", initial=10),
        Port("n_events_per_topology", "int", initial=1000),
        Port("min_fsps", "int", initial=3),
        Port("max_fsps", "int", initial=8),
        Port("max_depth", "int", initial=4),
    ],
    outputs=[
        "dataset_train",
        "dataset_val",
        "dataset_test",
        "dataset_meta",
        "stats",
        "encoding_report",
    ],
)

characterize = Flow(
    "characterize",
    title="Characterise the phase-4b arms, before any data",
    nodes=[arm_report, encoding_cells, shape_ceiling],
    inputs=[
        Port("n_qubits", "int", initial=4),
        Port("purity_ansatz", "str", initial="XY_Ring"),
        Port("min_fsps", "int", initial=3),
        Port("max_fsps", "int", initial=8),
    ],
    outputs=["arm_report", "encoding_cells", "shape_ceiling"],
)

train = Flow(
    "train",
    title="Train and evaluate an LCAG model",
    nodes=[dla_report, whitening_rotation, fit, evaluate],
    inputs=[
        Port("dataset_train", "artifact"),
        Port("dataset_val", "artifact"),
        Port("dataset_test", "artifact"),
        Port("dataset_meta", "json"),
        Port("seed", "int", initial=0),
        Port("model", "str", initial="gnn"),
        Port("preconditioner", "str", initial="none"),
        Port("encoding", "str", initial="angles"),
        Port("dim", "int", initial=64),
        Port("n_blocks", "int", initial=3),
        Port("epochs", "int", initial=100),
        Port("batch_size", "int", initial=64),
        Port("lr", "float", initial=1e-3),
        Port("ansatz", "str", initial="XY_Brickwork"),
        Port("n_layers", "int", initial=2),
        Port("angle_map", "str", initial="pair_polar"),
        Port("enc_weights", "str", initial="hamming"),
        Port("enc_reupload", "str", initial="diagonal"),
        Port("whiten", "bool", initial=False),
        Port("whitening_seed", "int", initial=0),
    ],
    outputs=[
        "checkpoint",
        "final_metrics",
        "test_metrics",
        "dla_report",
        "whitening_report",
    ],
)
