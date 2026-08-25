"""Fluksio flow declarations.

Two flows rather than one, because Fluksio has no stage caching (``NOTEPAD.md``): a
run always re-executes every node it contains. Generation runs phase-space simulation
through TensorFlow and is expensive; training is the part an ablation sweep repeats.
Splitting them means ``generate`` runs once and every ``train`` run consumes the
artifact references it produced.

    fluksio serve                                     # the engine, once
    fluksio sync partiqledtr                          # upload the flows
    fluksio run generate --seed 0 --wait
    fluksio run train --dataset_train <ref> --dataset_val <ref> \
        --dataset_test <ref> --dataset_meta <ref> --model gnn --epochs 100

The ``train`` flow also carries the phase-3/4 instrumentation. ``dla_report`` runs
upstream of ``fit`` and its certificate is a *required* input there, so the arm's
dynamical Lie algebra and floor count are recorded before any training happens --
enforced by the flow's shape rather than by anyone remembering. ``whitening_rotation``
fits the fixed isotropic preconditioning on the training split; the quantum arm uses
it when ``frontend`` is left at ``"none"`` and it is wired in, and ignores it
otherwise.

The nine-cell phase-4 study is nine runs of this one flow:

    for arm in XY_Brickwork Matchgate Circuit_19; do
      for fe in raw whiten mlp; do
        fluksio run train --model qfm --encoding cartesian --ansatz $arm ...
      done
    done

Declarations only: the nodes live in :mod:`partiqledtr.data.dataset`,
:mod:`partiqledtr.data.whitening`, :mod:`partiqledtr.analysis` and
:mod:`partiqledtr.train`, and are wired by matching provides/requires names.
"""

from fluksio import Flow, Port

from partiqledtr.analysis import dla_report
from partiqledtr.data.dataset import build_dataset, dataset_stats
from partiqledtr.data.whitening import whitening_rotation
from partiqledtr.train import evaluate, fit

generate = Flow(
    "generate",
    title="Generate the decay dataset",
    nodes=[build_dataset, dataset_stats],
    inputs=[
        Port("seed", "int", initial=0),
        # Per group, so the default is 30 topologies over the three known/unknown
        # groups -- the ROADMAP's ">=10 topologies, ~1000 events/topology".
        Port("n_topologies", "int", initial=10),
        Port("n_events_per_topology", "int", initial=1000),
        Port("min_fsps", "int", initial=3),
        Port("max_fsps", "int", initial=8),
    ],
    outputs=["dataset_train", "dataset_val", "dataset_test", "dataset_meta", "stats"],
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
        Port("frontend", "str", initial="none"),
        Port("encoding", "str", initial="angles"),
        Port("dim", "int", initial=64),
        Port("n_blocks", "int", initial=3),
        Port("epochs", "int", initial=100),
        Port("batch_size", "int", initial=64),
        Port("lr", "float", initial=1e-3),
        Port("ansatz", "str", initial="XY_Brickwork"),
        Port("whiten", "bool", initial=False),
    ],
    outputs=[
        "checkpoint",
        "final_metrics",
        "test_metrics",
        "dla_report",
        "whitening_report",
    ],
)
