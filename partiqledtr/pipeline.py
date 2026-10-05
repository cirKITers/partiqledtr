"""Declare the ``generate``, ``characterize``, and ``train`` Fluksio flows.

Generation produces datasets and encoding reports. Characterization records
ansatz and encoding certificates without data. Training records the DLA
certificate, fits whitening on the training split, then fits and evaluates a
model. Nodes are connected by matching provides/requires names.
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
        # groups, at 1000 events each.
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
    title="Characterise the study arms, before any data",
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
        # Per-group overrides. Fluksio cannot express a nullable flow input
        # (Port(initial=None) means "no initial"), so these are float ports where
        # non-positive means "share lr". The 0.0-freeze diagnostic stays on the
        # in-process path.
        Port("lr_preconditioner", "float", initial=0.0),
        Port("lr_qfm", "float", initial=0.0),
        Port("ansatz", "str", initial="XY_Brickwork"),
        Port("n_layers", "int", initial=2),
        Port("n_channels", "int", initial=1),
        Port("n_qubits", "int", initial=4),
        Port("angle_map", "str", initial="pair_polar"),
        Port("enc_weights", "str", initial="hamming"),
        Port("enc_reupload", "str", initial="diagonal"),
        Port("node_update", "str", initial="linear"),
        Port("node_hidden", "int", initial=32),
        Port("node_omega", "float", initial=1.0),
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
