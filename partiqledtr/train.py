"""Training loop, checkpointing and the training flow's nodes (ROADMAP phase 2).

The loop itself lives in :func:`train_model`, a plain generator over numpy arrays:
Fluksio's ``save_artifact`` raises outside a running node, so keeping the pure part
separate is what makes the whole thing testable without an engine (the same split
:mod:`partiqledtr.data.dataset` uses for generation).

Loss is class-weighted cross-entropy with the ``-1`` padding class ignored
(``DECISIONS.md`` D22); checkpoints are ``np.savez`` state dictionaries rather than
orbax (D23), carrying the settings needed to rebuild the model beside its parameters
so :func:`evaluate` needs nothing but the artifact.

Everything here is classical, so the train and eval steps are wrapped in
:func:`flax.nnx.jit`. Phase 3 replaces the model, not this file.
"""

from __future__ import annotations

import inspect
import io
import json
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import fluksio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from fluksio import Port, node

from partiqledtr.data.dataset import ENCODINGS, load_split
from partiqledtr.metrics import (
    IGNORE_PRIMARY,
    class_weights,
    masked_accuracy,
    perfect_lcag_rate,
    valid_tree_rate,
)
from partiqledtr.models import FRONTENDS, MODELS, n_params

__all__ = [
    "CONFIG_KEY",
    "batches",
    "build_model",
    "evaluate",
    "evaluate_split",
    "fit",
    "lcag_loss",
    "npz_config",
    "npz_to_state",
    "state_to_npz",
    "train_model",
]

#: Reserved npz entry holding the model configuration. Parameter paths are ``'/'``
#: joins of Python identifiers and list indices, so ``'#'`` cannot collide with one.
CONFIG_KEY = "#config"


def lcag_loss(logits: jax.Array, labels: jax.Array, weights: jax.Array) -> jax.Array:
    """Class-weighted softmax cross-entropy over the scored LCAG cells.

    A cell is scored iff its label is not ``-1``; masked cells contribute exactly
    zero, both to the value and to the gradient. The labels are clamped to a valid
    class *before* they reach optax, because ``-1`` is out of range there and would
    silently index the last class (or produce a NaN under ``jax_debug_nans``).

    Args:
        logits: Unnormalised log-probabilities of shape ``(..., C)``.
        labels: Integer classes of shape ``(...)``; ``-1`` marks an ignored cell.
        weights: Per-class weights of shape ``(C,)``; the weight of a cell is
            ``weights[label]``.

    Returns:
        Scalar mean over the scored cells, or ``0.0`` if the batch has none.
    """
    scored = labels != -1
    safe = jnp.where(scored, labels, 0)
    entropy = optax.softmax_cross_entropy_with_integer_labels(logits, safe)
    per_cell = jnp.where(scored, entropy * weights[safe], 0.0)
    return jnp.sum(per_cell) / jnp.maximum(jnp.sum(scored), 1)


def batches(rng: np.random.Generator, n_events: int, batch_size: int) -> Iterator[np.ndarray]:
    """Yield shuffled index batches covering one epoch.

    The last partial batch is dropped so every step sees the same shapes and the
    jitted train step compiles once.

    Args:
        rng: Generator supplying the shuffle; seed it to make an epoch reproducible.
        n_events: Number of events to draw indices from.
        batch_size: Events per batch.

    Yields:
        Integer index arrays of shape ``(batch_size,)``, disjoint within an epoch.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    order = rng.permutation(n_events)
    for start in range(0, n_events - batch_size + 1, batch_size):
        yield order[start : start + batch_size]


def build_model(
    *,
    model: str,
    frontend: str,
    n_features: int,
    n_classes: int,
    dim: int = 64,
    n_blocks: int = 3,
    seed: int = 0,
    ansatz: str = "XY_Brickwork",
    n_layers: int = 2,
    whitening: Any = None,
) -> nnx.Module:
    """Construct a model and its optional front end from the registry strings.

    Optional keywords are forwarded only to classes whose ``__init__`` accepts
    them, so the registry stays the single extension point: ``n_blocks`` reaches
    the GNN, and ``ansatz``/``n_layers``/``whitening`` reach the quantum arm.

    Args:
        model: Key into :data:`partiqledtr.models.MODELS`.
        frontend: Key into :data:`partiqledtr.models.FRONTENDS`; ``"none"`` feeds the
            raw features.
        n_features: Number of per-particle input features ``F``.
        n_classes: Number of LCAG classes ``C``.
        dim: Width of the model's internal representations.
        n_blocks: Number of message-passing blocks.
        seed: Seed of the parameter-initialisation rng.
        ansatz: Ansatz arm of the quantum model.
        n_layers: Data-reuploading depth of the quantum model.
        whitening: Optional fixed rotation for the quantum model's whitening arm.

    Returns:
        The constructed model, with the front end already attached.

    Raises:
        ValueError: If ``model`` or ``frontend`` names no registry entry.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; valid models are {sorted(MODELS)}")
    if frontend not in FRONTENDS:
        raise ValueError(f"unknown frontend {frontend!r}; valid frontends are {sorted(FRONTENDS)}")

    rngs = nnx.Rngs(seed)
    frontend_cls = FRONTENDS[frontend]
    cls = MODELS[model]
    # ``signature(cls)`` would resolve to the NNX metaclass' ``(*args, **kwargs)``,
    # so the check has to read ``__init__`` directly (DECISIONS.md D62).
    accepted = inspect.signature(cls.__init__).parameters
    optional = {"n_blocks": n_blocks, "ansatz": ansatz, "n_layers": n_layers, "seed": seed}
    if whitening is not None:
        optional["whitening"] = whitening
    extra = {name: value for name, value in optional.items() if name in accepted}

    # The quantum arm's front end sees its pair-polar angles, not the raw features.
    frontend_features = getattr(cls, "frontend_features", n_features)
    return cls(
        n_features,
        n_classes,
        dim=dim,
        frontend=None if frontend_cls is None else frontend_cls(frontend_features, rngs=rngs),
        rngs=rngs,
        **extra,
    )


def _flat_keys(state: Any) -> list[str]:
    """Flat ``'/'``-joined parameter paths, in the order of ``jax.tree`` leaves."""
    return ["/".join(str(part) for part in path) for path, _ in nnx.to_flat_state(state)]


def state_to_npz(module: nnx.Module, config: dict[str, Any] | None = None) -> bytes:
    """Serialise a module's parameters, and optionally how to rebuild it, to npz bytes.

    Args:
        module: Any NNX module; only :class:`flax.nnx.Param` leaves are stored.
        config: Keyword arguments for :func:`build_model` plus ``encoding``, stored as
            JSON under :data:`CONFIG_KEY` so a checkpoint is self-describing.

    Returns:
        The bytes of an uncompressed npz archive.
    """
    state = nnx.state(module, nnx.Param)
    arrays: dict[str, np.ndarray] = {
        key: np.asarray(leaf)
        for key, leaf in zip(_flat_keys(state), jax.tree.leaves(state), strict=True)
    }
    if config is not None:
        arrays[CONFIG_KEY] = np.asarray(json.dumps(config))
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    return buffer.getvalue()


def npz_config(payload: bytes) -> dict[str, Any]:
    """Read the model configuration back out of a checkpoint.

    Args:
        payload: Bytes written by :func:`state_to_npz` with a ``config``.

    Returns:
        The configuration record: :func:`build_model`'s keyword arguments plus
        ``encoding``.

    Raises:
        ValueError: If the checkpoint carries no configuration.
    """
    with np.load(io.BytesIO(payload)) as data:
        if CONFIG_KEY not in data.files:
            raise ValueError("checkpoint carries no model configuration")
        config: dict[str, Any] = json.loads(str(data[CONFIG_KEY].item()))
    return config


def npz_to_state(module: nnx.Module, payload: bytes) -> None:
    """Load parameters from a checkpoint into a module, in place.

    Args:
        module: A module built the same way as the one that was saved.
        payload: Bytes written by :func:`state_to_npz`.

    Raises:
        ValueError: If the stored parameters do not match the module's, by name or by
            shape. Loading a mismatched checkpoint would otherwise fail much later,
            inside a jitted step.
    """
    state = nnx.state(module, nnx.Param)
    leaves, treedef = jax.tree.flatten(state)
    keys = _flat_keys(state)
    with np.load(io.BytesIO(payload)) as data:
        stored = {name: data[name] for name in data.files if name != CONFIG_KEY}

    if set(stored) != set(keys):
        missing = sorted(set(keys) - set(stored)) or ["none"]
        extra = sorted(set(stored) - set(keys)) or ["none"]
        raise ValueError(
            f"checkpoint does not match the module: missing {missing}, unexpected {extra}"
        )
    for key, leaf in zip(keys, leaves, strict=True):
        if stored[key].shape != leaf.shape:
            raise ValueError(
                f"checkpoint parameter '{key}' has shape {stored[key].shape}, "
                f"but the module's is {leaf.shape}"
            )
    nnx.update(module, jax.tree.unflatten(treedef, [jnp.asarray(stored[key]) for key in keys]))


def _split_arrays(
    split: dict[str, np.ndarray], encoding: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack one split into the model's call convention.

    Args:
        split: Arrays from :func:`partiqledtr.data.dataset.load_split`.
        encoding: Feature encoding to train on, one of
            :data:`partiqledtr.data.dataset.ENCODINGS`.

    Returns:
        ``(features (N, L, F) float32, mask (N, L) bool, labels (N, L, L) int32)``. The
        mask comes from ``n_fsps``, not from a feature heuristic: padding puts the real
        particles in the first ``n_fsps`` rows (``DECISIONS.md`` D17).

    Raises:
        ValueError: If the encoding is unknown or the split lacks an expected array.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; valid encodings are {list(ENCODINGS)}")
    key = f"features_{encoding}"
    for name in (key, "lcag", "n_fsps"):
        if name not in split:
            raise ValueError(f"split has no '{name}' array; it holds {sorted(split)}")

    features = np.asarray(split[key], dtype=np.float32)
    labels = np.asarray(split["lcag"], dtype=np.int32)
    mask = np.arange(features.shape[1]) < np.asarray(split["n_fsps"])[:, None]
    return features, mask, labels


@nnx.jit
def _logits(module: nnx.Module, x: jax.Array, mask: jax.Array) -> jax.Array:
    """One jitted forward pass."""
    return module(x, mask)


@nnx.jit
def _train_step(
    module: nnx.Module,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    mask: jax.Array,
    labels: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """One jitted optimisation step; mutates ``module`` and ``optimizer`` in place."""
    loss, grads = nnx.value_and_grad(lambda m: lcag_loss(m(x, mask), labels, weights))(module)
    optimizer.update(module, grads)
    return loss


def evaluate_split(
    module: nnx.Module,
    split: dict[str, np.ndarray],
    *,
    encoding: str,
    batch_size: int = 256,
    weights: np.ndarray | jax.Array | None = None,
    valid_trees: bool = False,
) -> dict[str, float]:
    """Score a module on one dataset split.

    Args:
        module: The model to evaluate.
        split: Arrays from :func:`partiqledtr.data.dataset.load_split`.
        encoding: Feature encoding to read.
        batch_size: Events per forward pass. Unlike training, the partial tail batch is
            kept, so no event is dropped from the score.
        weights: Class weights; when given, the split's loss is reported too.
        valid_trees: Also compute the valid-tree rate. Off by default because it
            reconstructs a tree per event in Python -- run it once at the end, not per
            epoch.

    Returns:
        ``accuracy``, ``accuracy_primary``, ``perfect`` and ``perfect_primary``, plus
        ``loss`` if ``weights`` was given and ``valid_tree`` if ``valid_trees``. Every
        entry is NaN for an empty split.
    """
    features, mask, labels = _split_arrays(split, encoding)
    class_weight = None if weights is None else jnp.asarray(weights, dtype=jnp.float32)

    chunks: list[np.ndarray] = []
    total, n_scored = 0.0, 0
    for start in range(0, len(labels), batch_size):
        stop = start + batch_size
        # ponytail: the split is re-uploaded batch by batch every epoch. Ceiling is a
        # host-to-device copy per step, invisible next to the forward pass at this size;
        # upgrade is jax.device_put of the whole split once, when it still fits.
        logits = _logits(module, jnp.asarray(features[start:stop]), jnp.asarray(mask[start:stop]))
        chunks.append(np.asarray(jnp.argmax(logits, axis=-1)))
        if class_weight is not None:
            scored = int(np.count_nonzero(labels[start:stop] != -1))
            total += (
                float(lcag_loss(logits, jnp.asarray(labels[start:stop]), class_weight)) * scored
            )
            n_scored += scored

    predictions = np.concatenate(chunks) if chunks else np.zeros_like(labels)
    metrics = {
        "accuracy": masked_accuracy(predictions, labels),
        "accuracy_primary": masked_accuracy(predictions, labels, IGNORE_PRIMARY),
        "perfect": perfect_lcag_rate(predictions, labels),
        "perfect_primary": perfect_lcag_rate(predictions, labels, IGNORE_PRIMARY),
    }
    if class_weight is not None:
        metrics["loss"] = total / n_scored if n_scored else float("nan")
    if valid_trees:
        metrics["valid_tree"] = valid_tree_rate(predictions, labels)
    return metrics


def train_model(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    meta: dict[str, Any],
    *,
    seed: int,
    model: str = "gnn",
    frontend: str = "none",
    encoding: str = "angles",
    dim: int = 64,
    n_blocks: int = 3,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    ansatz: str = "XY_Brickwork",
    n_layers: int = 2,
    whitening: Any = None,
    n_purity_events: int = 64,
) -> Generator[dict[str, float], None, tuple[nnx.Module, dict[str, Any]]]:
    """Fit a model on the training split, reporting validation metrics per epoch.

    Class weights are fitted on the training labels (``DECISIONS.md`` D22); class 0 is
    never a target on this dataset and so gets weight 0 (D49).

    When the model exposes a ``g_purity`` method -- the quantum arm does -- the mean
    g-purity of its encoded states is measured on a fixed subset of validation
    events every epoch and streamed alongside the losses. That is the phase-4
    observable: read it against
    :func:`partiqledtr.analysis.offdiag_uniform_mean` to see the rescue dynamics
    the unflattening theory predicts (``DECISIONS.md`` D51, D60).

    Args:
        train: Training split arrays.
        val: Validation split arrays.
        meta: Dataset metadata; only ``n_classes`` is read.
        seed: Seeds parameter initialisation and the epoch shuffles.
        model: Key into :data:`partiqledtr.models.MODELS`.
        frontend: Key into :data:`partiqledtr.models.FRONTENDS`.
        encoding: Feature encoding to train on.
        dim: Width of the model's internal representations.
        n_blocks: Number of message-passing blocks.
        epochs: Number of passes over the training split.
        batch_size: Events per optimisation step.
        lr: Adam learning rate.
        ansatz: Ansatz arm of the quantum model; ignored by the classical ones.
        n_layers: Data-reuploading depth of the quantum model.
        whitening: Optional fixed ``(4, 4)`` rotation for the whitening arm.
        n_purity_events: Validation events the g-purity is measured on each epoch.

    Yields:
        One dict per epoch with ``epoch``, ``train_loss``, ``val_loss``,
        ``val_accuracy``, ``val_perfect`` and ``g_purity`` (NaN for models that
        encode no quantum state).

    Returns:
        A ``(module, final_metrics)`` pair. ``final_metrics["config"]`` is what
        :func:`build_model` needs to rebuild the module, plus the encoding.

    Raises:
        ValueError: If ``epochs`` is not positive or the training split holds fewer
            events than one batch, which would silently train on nothing.
    """
    features, mask, labels = _split_arrays(train, encoding)
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if len(labels) < batch_size:
        raise ValueError(
            f"train split has {len(labels)} events, fewer than batch_size={batch_size}"
        )

    n_classes = int(meta["n_classes"])
    config: dict[str, Any] = {
        "model": model,
        "frontend": frontend,
        "n_features": int(features.shape[2]),
        "n_classes": n_classes,
        "dim": int(dim),
        "n_blocks": int(n_blocks),
        "ansatz": ansatz,
        "n_layers": int(n_layers),
    }
    module = build_model(**config, seed=seed, whitening=whitening)
    optimizer = nnx.Optimizer(module, optax.adam(lr), wrt=nnx.Param)
    weights = jnp.asarray(class_weights(labels, n_classes), dtype=jnp.float32)

    # A fixed validation subset, so the purity series tracks the model rather than
    # the sample. Only the arms that encode quantum states expose g_purity.
    measure_purity = getattr(module, "g_purity", None)
    purity_batch = None
    if measure_purity is not None:
        val_features, val_mask, _ = _split_arrays(val, encoding)
        take = slice(0, min(n_purity_events, len(val_mask)))
        purity_batch = (jnp.asarray(val_features[take]), jnp.asarray(val_mask[take]))

    rng = np.random.default_rng(seed)
    train_loss = float("nan")
    val_metrics: dict[str, float] = {}
    for epoch in range(epochs):
        losses = [
            float(
                _train_step(
                    module,
                    optimizer,
                    jnp.asarray(features[index]),
                    jnp.asarray(mask[index]),
                    jnp.asarray(labels[index]),
                    weights,
                )
            )
            for index in batches(rng, len(labels), batch_size)
        ]
        train_loss = float(np.mean(losses))
        val_metrics = evaluate_split(
            module, val, encoding=encoding, batch_size=batch_size, weights=weights
        )
        if measure_purity is not None and purity_batch is not None:
            val_metrics["g_purity"] = float(measure_purity(*purity_batch))
        yield {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_perfect": val_metrics["perfect"],
            "g_purity": val_metrics.get("g_purity", float("nan")),
        }

    final = {
        "config": {"encoding": encoding, **config},
        "n_params": n_params(module),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "train_loss": train_loss,
        **{f"val_{name}": value for name, value in val_metrics.items()},
    }
    return module, final


@node(
    requires=[
        Port("dataset_train", "artifact"),
        Port("dataset_val", "artifact"),
        Port("dataset_meta", "json"),
        Port("seed", "int"),
        Port("model", "str"),
        Port("frontend", "str"),
        Port("encoding", "str"),
        Port("dim", "int"),
        Port("n_blocks", "int"),
        Port("epochs", "int"),
        Port("batch_size", "int"),
        Port("lr", "float"),
        Port("ansatz", "str"),
        Port("whitening", "artifact"),
        Port("whiten", "bool"),
        Port("dla_report", "json"),
    ],
    provides=[
        Port("epoch", "int", stream=True),
        Port("train_loss", "float", stream=True),
        Port("val_loss", "float", stream=True),
        Port("val_accuracy", "float", stream=True),
        Port("val_perfect", "float", stream=True),
        Port("g_purity", "float", stream=True),
        Port("checkpoint", "artifact"),
        Port("final_metrics", "json"),
    ],
)
def fit(
    *,
    dataset_train: dict[str, Any],
    dataset_val: dict[str, Any],
    dataset_meta: dict[str, Any],
    seed: int,
    model: str = "gnn",
    frontend: str = "none",
    encoding: str = "angles",
    dim: int = 64,
    n_blocks: int = 3,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    ansatz: str = "XY_Brickwork",
    whitening: dict[str, Any] | None = None,
    whiten: bool = False,
    dla_report: dict[str, Any] | None = None,
    n_layers: int = 2,
) -> Generator[dict[str, float], None, dict[str, Any]]:
    """Train a model and store its checkpoint as a run artifact.

    Args:
        dataset_train: Training split artifact reference.
        dataset_val: Validation split artifact reference.
        dataset_meta: Metadata record from
            :func:`partiqledtr.data.dataset.build_dataset`.
        seed: Seeds parameter initialisation and the epoch shuffles.
        model: Key into :data:`partiqledtr.models.MODELS`.
        frontend: Key into :data:`partiqledtr.models.FRONTENDS`.
        encoding: Feature encoding to train on.
        dim: Width of the model's internal representations.
        n_blocks: Number of message-passing blocks.
        epochs: Number of passes over the training split.
        batch_size: Events per optimisation step.
        lr: Adam learning rate.
        ansatz: Ansatz arm of the quantum model.
        whitening: Artifact reference to the fixed whitening rotation fitted by
            :func:`partiqledtr.data.whitening.whitening_rotation`. Always wired in
            the flow; applied only when ``whiten`` is set.
        whiten: Select the phase-4 fixed-preconditioning arm. The rotation is
            fitted regardless, so its acceptance report is recorded for every run,
            but with ``whiten=False`` the model encodes the raw angles -- that is
            the ROADMAP's "raw" arm.
        dla_report: Optional DLA certificate from
            :func:`partiqledtr.analysis.dla_report`. Not used by the fit itself --
            requiring it here is what makes the flow record the arm's algebra
            before any training happens, as the ROADMAP asks.
        n_layers: Data-reuploading depth of the quantum model.

    Yields:
        The per-epoch metrics of :func:`train_model`, one message per declared stream.

    Returns:
        The checkpoint artifact reference and the final metrics record.
    """
    rotation = None
    if whiten and whitening is not None:
        rotation = jnp.asarray(np.load(fluksio.load_artifact(whitening))["rotation"])

    module, final = yield from train_model(
        load_split(dataset_train),
        load_split(dataset_val),
        dataset_meta,
        seed=seed,
        model=model,
        frontend=frontend,
        encoding=encoding,
        dim=dim,
        n_blocks=n_blocks,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        ansatz=ansatz,
        n_layers=n_layers,
        whitening=rotation,
    )
    final["whitened"] = rotation is not None
    final["dla_report"] = dla_report
    payload = state_to_npz(module, final["config"])
    return {
        "checkpoint": fluksio.save_artifact(payload, "checkpoint.npz"),
        "final_metrics": final,
    }


@node(
    requires=[
        Port("checkpoint", "artifact"),
        Port("dataset_test", "artifact"),
        Port("dataset_meta", "json"),
    ],
    provides=[Port("test_metrics", "json")],
)
def evaluate(
    *,
    checkpoint: dict[str, Any],
    dataset_test: dict[str, Any],
    dataset_meta: dict[str, Any],
) -> dict[str, Any]:
    """Score a checkpoint on the test split, overall and by topology familiarity.

    The test split mixes topologies the model trained on (group 0) with topologies it
    has never seen (groups 1 and 2), so the ``known``/``unknown`` pair is the
    generalisation probe of ``DECISIONS.md`` D16. Reporting only the overall number
    would average the two together and hide the answer.

    Args:
        checkpoint: Checkpoint artifact reference from :func:`fit`.
        dataset_test: Test split artifact reference.
        dataset_meta: Metadata record; ``topology_group`` maps a topology id to its
            group.

    Returns:
        ``{"test_metrics": {"overall": {...}, "known": {...}, "unknown": {...}}}``, each
        entry the metrics of :func:`evaluate_split` plus ``n_events``. A subset with no
        events reports NaN metrics rather than being omitted, so the record's shape does
        not depend on the dataset.
    """
    payload = Path(fluksio.load_artifact(checkpoint)).read_bytes()
    config = npz_config(payload)
    encoding = config.pop("encoding")
    module = build_model(**config)
    npz_to_state(module, payload)

    split = load_split(dataset_test)
    group = np.asarray(dataset_meta["topology_group"])
    known = group[split["topology_id"]] == 0
    subsets = {"overall": np.ones_like(known), "known": known, "unknown": ~known}
    return {
        "test_metrics": {
            name: {
                "n_events": int(selection.sum()),
                **evaluate_split(
                    module,
                    {key: array[selection] for key, array in split.items()},
                    encoding=encoding,
                    valid_trees=True,
                ),
            }
            for name, selection in subsets.items()
        }
    }
