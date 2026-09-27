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
import math
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
from partiqledtr.models import MODELS, PRECONDITIONERS, n_params

__all__ = [
    "CONFIG_KEY",
    "batches",
    "build_model",
    "evaluate",
    "evaluate_split",
    "fit",
    "jsonable",
    "lcag_loss",
    "npz_config",
    "npz_to_state",
    "state_to_npz",
    "train_model",
]

#: Reserved npz entry holding the model configuration. Parameter paths are ``'/'``
#: joins of Python identifiers and list indices, so ``'#'`` cannot collide with one.
CONFIG_KEY = "#config"

#: Offset separating the preconditioner's rng stream from the model's, so attaching one
#: does not re-initialise the other (``DECISIONS.md`` D84).
_PRECONDITIONER_SEED_OFFSET = 1 << 20

#: Seed of the g-purity measurement subset. Deliberately *not* the run's seed: every
#: arm and every seed has to measure the observable on the same events, or a purity
#: difference between two cells could be a difference between two samples (D105).
_PURITY_SEED = 987654321


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
    preconditioner: str,
    n_features: int,
    n_classes: int,
    dim: int = 64,
    n_blocks: int = 3,
    seed: int = 0,
    ansatz: str = "XY_Brickwork",
    n_layers: int = 2,
    n_channels: int = 1,
    n_qubits: int = 4,
    angle_map: str = "pair_polar",
    enc_weights: str = "hamming",
    enc_reupload: str = "diagonal",
    node_update: str = "linear",
    node_hidden: int = 32,
    node_omega: float = 1.0,
    whitening: Any = None,
) -> nnx.Module:
    """Construct a model and its optional preconditioner from the registry strings.

    Optional keywords are forwarded only to classes whose ``__init__`` accepts
    them, so the registry stays the single extension point: ``n_blocks`` reaches
    the GNN, and ``ansatz``/``n_layers``/``whitening`` reach the quantum arm.

    Args:
        model: Key into :data:`partiqledtr.models.MODELS`.
        preconditioner: Key into :data:`partiqledtr.models.PRECONDITIONERS`; ``"none"`` feeds the
            raw features.
        n_features: Number of per-particle input features ``F``.
        n_classes: Number of LCAG classes ``C``.
        dim: Width of the model's internal representations.
        n_blocks: Number of message-passing blocks.
        seed: Seed of the parameter-initialisation rng.
        ansatz: Ansatz arm of the quantum model.
        n_layers: Data-reuploading depth of the quantum model.
        n_channels: Independent QFMs per block of the quantum model (D108).
        n_qubits: Qubits per edge QFM (ROADMAP phase 6); must match the angle
            map's width, which the constellation checks at construction.
        angle_map: Four-vector-to-angle map of the quantum model, a key of
            :data:`partiqledtr.models.qfm.ANGLE_MAPS`.
        enc_weights: Encoding weight strategy of the quantum model, a key of
            :data:`partiqledtr.models.qfm.ENC_WEIGHTS`.
        enc_reupload: Re-upload mask of the quantum model, a key of
            :data:`partiqledtr.models.qfm.ENC_REUPLOAD`.
        node_update: Node-update variant of the quantum model, one of
            :data:`partiqledtr.models.qfm.NODE_UPDATES` (the trig-interface arm).
        node_hidden: Hidden width of the non-linear node updates.
        node_omega: Scale at the block-2 re-encoding boundary of the quantum model.
        whitening: Optional fixed rotation for the quantum model's whitening arm.
            A nested list is accepted, which is how a checkpoint carries it (D81).

    Returns:
        The constructed model, with the preconditioner already attached.

    Raises:
        ValueError: If ``model`` or ``preconditioner`` names no registry entry.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; valid models are {sorted(MODELS)}")
    if preconditioner not in PRECONDITIONERS:
        raise ValueError(
            f"unknown preconditioner {preconditioner!r}; "
            f"valid preconditioners are {sorted(PRECONDITIONERS)}"
        )

    # Two independent streams. Built from one, the preconditioner's own draws would shift
    # every later draw, so a model *with* a preconditioner would not merely gain the front
    # end -- its whole parameter set would be re-initialised, and the phase-4
    # raw-versus-learned comparison would differ by an initialisation too (D84).
    rngs = nnx.Rngs(seed)
    preconditioner_rngs = nnx.Rngs(seed + _PRECONDITIONER_SEED_OFFSET)
    preconditioner_cls = PRECONDITIONERS[preconditioner]
    cls = MODELS[model]
    # ``signature(cls)`` would resolve to the NNX metaclass' ``(*args, **kwargs)``,
    # so the check has to read ``__init__`` directly (DECISIONS.md D62).
    accepted = inspect.signature(cls.__init__).parameters
    optional = {
        "n_blocks": n_blocks,
        "ansatz": ansatz,
        "n_layers": n_layers,
        "n_channels": n_channels,
        "n_qubits": n_qubits,
        "angle_map": angle_map,
        "enc_weights": enc_weights,
        "enc_reupload": enc_reupload,
        "node_update": node_update,
        "node_hidden": node_hidden,
        "node_omega": node_omega,
        "seed": seed,
    }
    if whitening is not None:
        optional["whitening"] = whitening
    extra = {name: value for name, value in optional.items() if name in accepted}

    # The quantum arm's preconditioner sees its angle-map output, not the raw
    # features, so its width follows the register rather than F.
    width = getattr(cls, "preconditioner_features", None)
    preconditioner_features = width(n_qubits) if callable(width) else n_features
    return cls(
        n_features,
        n_classes,
        dim=dim,
        preconditioner=(
            None
            if preconditioner_cls is None
            else preconditioner_cls(preconditioner_features, rngs=preconditioner_rngs)
        ),
        rngs=rngs,
        **extra,
    )


def jsonable(value: Any) -> Any:
    """Replace every non-finite float with ``None``, however deeply it sits.

    JSON cannot spell NaN or infinity, and Fluksio's ports reject both rather than
    letting one leave the engine as a response nobody can parse. A metric over an
    empty subset is legitimately undefined, though -- an ``unknown`` split with no
    events, say -- so it travels as ``None``, which says the same thing and does
    survive the wire (``DECISIONS.md`` D75).

    Args:
        value: Any json-shaped structure.

    Returns:
        The same structure with non-finite floats replaced by ``None``.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


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


def _group_optimizer(
    module: nnx.Module, lr: float, lr_preconditioner: float | None, lr_qfm: float | None
) -> nnx.Optimizer:
    """One Adam per parameter group, when any group rate differs (D109).

    Adam equalises per-parameter step sizes, so a shared base rate forces the
    preconditioner and the circuit to move at the same speed whatever their
    curvature; a per-group base rate is the only way to decouple them. ``None``
    means "the shared rate", and all-``None`` is exactly the old single Adam.
    """
    if lr_preconditioner is None and lr_qfm is None:
        return nnx.Optimizer(module, optax.adam(lr), wrt=nnx.Param)

    def label(path: tuple, _leaf: Any) -> str:
        keys = [str(getattr(entry, "key", entry)) for entry in path]
        if "preconditioner" in keys:
            return "preconditioner"
        if any(key.startswith("qfm") for key in keys):
            return "qfm"
        return "classical"

    # One label per variable, with the Param wrapper as the leaf: the gradient
    # tree optax sees carries plain arrays at the wrapper's position.
    labels = jax.tree_util.tree_map_with_path(
        label,
        nnx.state(module, nnx.Param),
        is_leaf=lambda node: isinstance(node, nnx.Variable),
    )
    transform = optax.multi_transform(
        {
            "classical": optax.adam(lr),
            "preconditioner": optax.adam(lr if lr_preconditioner is None else lr_preconditioner),
            "qfm": optax.adam(lr if lr_qfm is None else lr_qfm),
        },
        labels,
    )
    return nnx.Optimizer(module, transform, wrt=nnx.Param)


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
        ``loss`` if ``weights`` was given and ``valid_tree``/``valid_tree_strict`` if
        ``valid_trees``. Every entry is NaN for an empty split.
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
        # The strict variant is the primary number; the lenient one is kept for
        # comparability with the prior papers (``DECISIONS.md`` D85).
        metrics["valid_tree_strict"] = valid_tree_rate(predictions, labels, strict=True)
    return metrics


def train_model(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    meta: dict[str, Any],
    *,
    seed: int,
    model: str = "gnn",
    preconditioner: str = "none",
    encoding: str = "angles",
    dim: int = 64,
    n_blocks: int = 3,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    lr_preconditioner: float | None = None,
    lr_qfm: float | None = None,
    ansatz: str = "XY_Brickwork",
    n_layers: int = 2,
    n_channels: int = 1,
    n_qubits: int = 4,
    angle_map: str = "pair_polar",
    enc_weights: str = "hamming",
    enc_reupload: str = "diagonal",
    node_update: str = "linear",
    node_hidden: int = 32,
    node_omega: float = 1.0,
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
        preconditioner: Key into :data:`partiqledtr.models.PRECONDITIONERS`.
        encoding: Feature encoding to train on.
        dim: Width of the model's internal representations.
        n_blocks: Number of message-passing blocks.
        epochs: Number of passes over the training split.
        batch_size: Events per optimisation step.
        lr: Adam learning rate.
        lr_preconditioner: Optional separate Adam rate for the preconditioner's
            parameters (D109); ``None`` shares ``lr``.
        lr_qfm: Optional separate Adam rate for the circuit parameters (D109);
            ``None`` shares ``lr``.
        ansatz: Ansatz arm of the quantum model; ignored by the classical ones.
        n_layers: Data-reuploading depth of the quantum model.
        n_channels: Independent QFMs per block of the quantum model (D108).
        n_qubits: Qubits per edge QFM of the quantum model (ROADMAP phase 6).
        angle_map: Four-vector-to-angle map of the quantum model. Pair it with the
            matching ``encoding``: ``"legacy"`` with ``"legacy"``, otherwise
            ``"cartesian"``.
        enc_weights: Encoding weight strategy of the quantum model (arm B).
        enc_reupload: Re-upload mask of the quantum model (arm B).
        node_update: Node-update variant of the quantum model (trig-interface arm).
        node_hidden: Hidden width of the non-linear node updates.
        node_omega: Scale at the quantum model's block-2 re-encoding boundary.
        whitening: Optional fixed ``(4, 4)`` rotation for the whitening arm.
        n_purity_events: Validation events the g-purity is measured on each epoch.

    Yields:
        One dict per epoch with ``epoch``, ``train_loss``, ``val_loss``,
        ``val_accuracy``, ``val_perfect``, and -- for the arms that encode a
        quantum state -- ``g_purity`` plus the site means ``tv_uniform`` and
        ``mean_sin2`` of :func:`partiqledtr.analysis.angle_stats`.

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
        "preconditioner": preconditioner,
        "n_features": int(features.shape[2]),
        "n_classes": n_classes,
        "dim": int(dim),
        "n_blocks": int(n_blocks),
        "ansatz": ansatz,
        "n_layers": int(n_layers),
        "n_channels": int(n_channels),
        "n_qubits": int(n_qubits),
        "angle_map": angle_map,
        "enc_weights": enc_weights,
        "enc_reupload": enc_reupload,
        "node_update": node_update,
        "node_hidden": int(node_hidden),
        "node_omega": float(node_omega),
        # In the config, not beside it: the rotation is part of what the model *is*,
        # and it is not an nnx.Param, so a checkpoint that did not carry it would
        # rebuild the whitened arm as the raw one and score it on the wrong angles
        # (D81). 16 floats travel fine as json.
        "whitening": None if whitening is None else np.asarray(whitening).tolist(),
    }
    module = build_model(**config, seed=seed)
    optimizer = _group_optimizer(module, lr, lr_preconditioner, lr_qfm)
    weights = jnp.asarray(class_weights(labels, n_classes), dtype=jnp.float32)

    # A fixed validation subset, so the purity series tracks the model rather than
    # the sample -- fixed *and* drawn across the split, which is not the same thing
    # (D105). Only the arms that encode quantum states expose g_purity.
    measure_purity = getattr(module, "g_purity", None)
    measure_angles = getattr(module, "angle_stats", None)
    measure_block2 = getattr(module, "block2_g_purity", None)
    purity_batch = None
    initial_purity = float("nan")
    initial_block2 = float("nan")
    initial_angles: dict[str, list[float]] | None = None
    angles: dict[str, list[float]] | None = None
    if measure_purity is not None:
        val_features, val_mask, _ = _split_arrays(val, encoding)
        # Drawn at random, not sliced off the front: the split is ordered by
        # topology, so `val[:64]` is one topology at one multiplicity, and the
        # observable then describes that corner rather than the data (D105).
        take = np.random.default_rng(_PURITY_SEED).choice(
            len(val_mask), size=min(n_purity_events, len(val_mask)), replace=False
        )
        purity_batch = (jnp.asarray(val_features[take]), jnp.asarray(val_mask[take]))
        # Epoch 0, before any step. The preconditioner starts as the identity, so this is
        # also the raw arm's level -- without it a learned-preconditioner trajectory has
        # no anchor and its first plotted point is already one epoch of training old.
        initial_purity = float(measure_purity(*purity_batch))
        initial_angles = measure_angles(*purity_batch) if measure_angles else None
        if measure_block2 is not None:
            initial_block2 = float(measure_block2(*purity_batch))

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
        if measure_angles is not None and purity_batch is not None:
            # Streamed as the site mean, because a stream port carries one float;
            # the per-site vectors the reading actually needs ride in final_metrics.
            angles = measure_angles(*purity_batch)
            val_metrics["tv_uniform"] = float(np.mean(angles["tv_uniform"]))
            val_metrics["mean_sin2"] = float(np.mean(angles["mean_sin2"]))
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_perfect": val_metrics["perfect"],
        }
        # Omitted rather than NaN when the model encodes no quantum state: a float
        # port accepts neither a non-finite value nor None (``DECISIONS.md`` D75).
        if "g_purity" in val_metrics:
            record["g_purity"] = val_metrics["g_purity"]
        for name in ("tv_uniform", "mean_sin2"):
            if name in val_metrics:
                record[name] = val_metrics[name]
        yield record

    final = {
        "config": {"encoding": encoding, **config},
        "n_params": n_params(module),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lr_preconditioner": lr_preconditioner,
        "lr_qfm": lr_qfm,
        "seed": seed,
        "train_loss": train_loss,
        **{f"val_{name}": value for name, value in val_metrics.items()},
    }
    if purity_batch is not None:
        final["g_purity_initial"] = initial_purity
        # The closed form is the theory's object; this is the state the circuit
        # actually prepares. Reporting both is what lets a claim name which one it
        # is about (D78). Measured once, at the end: it is not jittable.
        exact = getattr(module, "g_purity_exact", None)
        if exact is not None:
            final["g_purity_exact"] = exact(*purity_batch)
        # Per-site, start and end. Pooling hides the thing worth seeing: sites
        # peaking at different angles average into something that looks flat (D92).
        if initial_angles is not None:
            final["angle_stats_initial"] = initial_angles
        if angles is not None:
            final["angle_stats_final"] = angles
        # The re-encoding boundary: what block 2 encodes, start and end. Kept in
        # final_metrics rather than streamed, so the flow's declared ports are
        # untouched.
        if measure_block2 is not None:
            final["block2_g_purity_initial"] = initial_block2
            final["block2_g_purity"] = float(measure_block2(*purity_batch))
            block2_stats = getattr(module, "block2_angle_stats", None)
            if block2_stats is not None:
                final["block2_angle_stats_final"] = block2_stats(*purity_batch)
    return module, final


@node(
    requires=[
        Port("dataset_train", "artifact"),
        Port("dataset_val", "artifact"),
        Port("dataset_meta", "json"),
        Port("seed", "int"),
        Port("model", "str"),
        Port("preconditioner", "str"),
        Port("encoding", "str"),
        Port("dim", "int"),
        Port("n_blocks", "int"),
        Port("epochs", "int"),
        Port("batch_size", "int"),
        Port("lr", "float"),
        # Per-group overrides (D109). Nullable flow inputs are not expressible
        # (NOTEPAD.md 2026-09-03), so the flow contract is: non-positive means
        # "share lr" -- the body maps 0.0 to None before train_model.
        Port("lr_preconditioner", "float"),
        Port("lr_qfm", "float"),
        Port("ansatz", "str"),
        Port("n_layers", "int"),
        Port("n_channels", "int"),
        Port("n_qubits", "int"),
        Port("angle_map", "str"),
        Port("enc_weights", "str"),
        Port("enc_reupload", "str"),
        Port("node_update", "str"),
        Port("node_hidden", "int"),
        Port("node_omega", "float"),
        Port("whitening", "artifact"),
        Port("whiten", "bool"),
        Port("dla_report", "json"),
    ],
    # Yields per epoch, which resets the watchdog -- but the *first* epoch also pays
    # for jit compilation, so the limit has to cover that rather than a steady one
    # (``DECISIONS.md`` D89).
    timeout=2 * 60 * 60,
    # The fingerprint covers this function's source, not `train_model` where the
    # loop actually lives, so editing the helper silently replays a pre-change run
    # -- which for the node that produces the study's numbers is a correctness
    # hazard, not an inconvenience (``DECISIONS.md`` D93).
    cache=False,
    provides=[
        Port("epoch", "int", stream=True),
        Port("train_loss", "float", stream=True),
        Port("val_loss", "float", stream=True),
        Port("val_accuracy", "float", stream=True),
        Port("val_perfect", "float", stream=True),
        Port("g_purity", "float", stream=True),
        Port("tv_uniform", "float", stream=True),
        Port("mean_sin2", "float", stream=True),
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
    preconditioner: str = "none",
    encoding: str = "angles",
    dim: int = 64,
    n_blocks: int = 3,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    lr_preconditioner: float = 0.0,
    lr_qfm: float = 0.0,
    ansatz: str = "XY_Brickwork",
    n_layers: int = 2,
    n_channels: int = 1,
    n_qubits: int = 4,
    angle_map: str = "pair_polar",
    enc_weights: str = "hamming",
    enc_reupload: str = "diagonal",
    node_update: str = "linear",
    node_hidden: int = 32,
    node_omega: float = 1.0,
    whitening: dict[str, Any] | None = None,
    whiten: bool = False,
    dla_report: dict[str, Any] | None = None,
    n_purity_events: int = 64,
) -> Generator[dict[str, float], None, dict[str, Any]]:
    """Train a model and store its checkpoint as a run artifact.

    Args:
        dataset_train: Training split artifact reference.
        dataset_val: Validation split artifact reference.
        dataset_meta: Metadata record from
            :func:`partiqledtr.data.dataset.build_dataset`.
        seed: Seeds parameter initialisation and the epoch shuffles.
        model: Key into :data:`partiqledtr.models.MODELS`.
        preconditioner: Key into :data:`partiqledtr.models.PRECONDITIONERS`.
        encoding: Feature encoding to train on.
        dim: Width of the model's internal representations.
        n_blocks: Number of message-passing blocks.
        epochs: Number of passes over the training split.
        batch_size: Events per optimisation step.
        lr: Adam learning rate.
        lr_preconditioner: Separate rate for the preconditioner (D109);
            non-positive shares ``lr`` (the flow contract, NOTEPAD.md 2026-09-03).
        lr_qfm: Separate rate for the circuit parameters (D109); non-positive
            shares ``lr``.
        ansatz: Ansatz arm of the quantum model.
        n_layers: Data-reuploading depth of the quantum model.
        n_channels: Independent QFMs per block of the quantum model (D108).
        n_qubits: Qubits per edge QFM of the quantum model (D110).
        angle_map: Four-vector-to-angle map of the quantum model; pair ``"legacy"``
            with the ``"legacy"`` encoding.
        enc_weights: Encoding weight strategy of the quantum model (arm B).
        enc_reupload: Re-upload mask of the quantum model (arm B).
        node_update: Node-update variant of the quantum model (D112).
        node_hidden: Hidden width of the non-linear node updates (D112).
        node_omega: Scale at the quantum model's re-encoding boundary (D112).
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
        n_purity_events: Validation events the g-purity is measured on each epoch.

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
        preconditioner=preconditioner,
        encoding=encoding,
        dim=dim,
        n_blocks=n_blocks,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lr_preconditioner=lr_preconditioner if lr_preconditioner > 0 else None,
        lr_qfm=lr_qfm if lr_qfm > 0 else None,
        ansatz=ansatz,
        n_layers=n_layers,
        n_channels=n_channels,
        n_qubits=n_qubits,
        angle_map=angle_map,
        enc_weights=enc_weights,
        enc_reupload=enc_reupload,
        node_update=node_update,
        node_hidden=node_hidden,
        node_omega=node_omega,
        whitening=rotation,
        n_purity_events=n_purity_events,
    )
    final["whitened"] = rotation is not None
    final["dla_report"] = dla_report
    payload = state_to_npz(module, final["config"])
    return {
        "checkpoint": fluksio.save_artifact(payload, "checkpoint.npz"),
        "final_metrics": jsonable(final),
    }


@node(
    # Reconstructs a tree per test event in Python, silent throughout (D89).
    timeout=43200,
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
    # A subset with no events scores NaN, which no port accepts; `jsonable` turns
    # those into None so "not measured" still travels (DECISIONS.md D75).
    return jsonable(
        {
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
    )
