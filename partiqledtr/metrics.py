"""LCAG evaluation metrics (ROADMAP phase 2).

All metrics take *predicted class indices*, not logits, so the argmax convention
lives at the call site.  Masks are built from the ground truth only: a cell is
scored iff its label is not an ignored class.  This follows baumbauen's
definitions rather than partiqlegan's, whose masking compares "prediction
correct" against "cell not ignored" and so scores agreement between two unrelated
booleans (``DECISIONS.md`` D20).

Everything here is numpy and runs at evaluation time only -- the valid-tree rate
has to reconstruct a tree per sample, which is inherently sequential Python.

Two ignore sets are used throughout:

* ``IGNORE`` -- the padding/diagonal sentinel alone.
* ``IGNORE_PRIMARY`` -- also drops class 0, giving a score over structural edges
  only, excluding the trivially correct zero entries (``DECISIONS.md`` D21).
"""

from __future__ import annotations

import numpy as np

from partiqledtr.data.lcag import InvalidLCAGError, lcag_roundtrip, lcag_to_adjacency

__all__ = [
    "IGNORE",
    "IGNORE_PRIMARY",
    "class_weights",
    "masked_accuracy",
    "perfect_lcag_rate",
    "valid_tree_rate",
]

IGNORE: tuple[int, ...] = (-1,)
IGNORE_PRIMARY: tuple[int, ...] = (-1, 0)


def _scored(labels: np.ndarray, ignore: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of the cells a metric scores, derived from the labels alone."""
    mask = np.ones(labels.shape, dtype=bool)
    for value in ignore:
        mask &= labels != value
    return mask


def masked_accuracy(
    predictions: np.ndarray, labels: np.ndarray, ignore: tuple[int, ...] = IGNORE
) -> float:
    """Fraction of scored cells predicted correctly.

    Args:
        predictions: Integer class indices, shape ``(..., L, L)``.
        labels: Ground-truth class indices of the same shape.
        ignore: Label values that are not scored.

    Returns:
        Accuracy over all scored cells, or NaN if no cell is scored.

    Raises:
        ValueError: If the shapes differ.
    """
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if predictions.shape != labels.shape:
        raise ValueError(f"shape mismatch: predictions {predictions.shape}, labels {labels.shape}")

    mask = _scored(labels, ignore)
    if not mask.any():
        return float("nan")
    return float((predictions[mask] == labels[mask]).mean())


def perfect_lcag_rate(
    predictions: np.ndarray, labels: np.ndarray, ignore: tuple[int, ...] = IGNORE
) -> float:
    """Fraction of events whose every scored cell is predicted correctly.

    The all-or-nothing metric of the reconstruction papers: an event counts only
    if the complete LCAG matrix is right.

    Args:
        predictions: Integer class indices, shape ``(B, L, L)``.
        labels: Ground-truth class indices of the same shape.
        ignore: Label values that are not scored.

    Returns:
        Fraction of events reconstructed exactly, over the events that have at
        least one scored cell; NaN if no event has one.

    Raises:
        ValueError: If the shapes differ or are not three-dimensional.
    """
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if predictions.shape != labels.shape:
        raise ValueError(f"shape mismatch: predictions {predictions.shape}, labels {labels.shape}")
    if labels.ndim != 3:
        raise ValueError(f"expected (B, L, L) labels, got shape {labels.shape}")

    mask = _scored(labels, ignore)
    correct = (predictions == labels) | ~mask
    scored_events = mask.any(axis=(1, 2))
    if not scored_events.any():
        return float("nan")
    # An event with no scored cell carries no information, so it is left out of
    # the denominator rather than counted as a vacuous success.
    return float(correct.all(axis=(1, 2))[scored_events].mean())


def _reconstructs(
    prediction: np.ndarray, labels: np.ndarray, ignore_disconnected: bool, strict: bool
) -> bool:
    """Whether one predicted LCAG describes a realisable decay tree."""
    scored = _scored(labels, IGNORE)
    if not scored.any() or not prediction[scored].any():
        # A prediction that is entirely class 0 on the scored cells describes no
        # tree at all; baumbauen rejects it before attempting reconstruction.
        return False

    keep = scored.copy()
    if ignore_disconnected:
        # Applied to the prediction, not the truth: we are only ignoring what the
        # model itself claims is disconnected.
        keep &= prediction != 0
    np.fill_diagonal(keep, False)

    rows = keep.any(axis=0)
    if strict:
        # Every scored leaf has to survive. Otherwise a prediction that calls all
        # but two leaves disconnected reduces to a trivially valid two-leaf tree,
        # and nothing in the loss discourages that: class 0 is never a target, so
        # it carries weight 0 (D49, D85).
        if not np.array_equal(rows, scored.any(axis=0)):
            return False
    elif rows.sum() < 2:
        return False

    candidate = prediction[np.ix_(rows, rows)].copy()
    np.fill_diagonal(candidate, 0)
    try:
        if strict:
            # Off-diagonal only: the diagonal carries the ignore label, not a level.
            implied = np.triu(lcag_roundtrip(candidate), 1)
            return bool(np.array_equal(implied, np.triu(candidate, 1)))
        lcag_to_adjacency(candidate)
    except InvalidLCAGError:
        return False
    return True


def valid_tree_rate(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    ignore_disconnected: bool = True,
    strict: bool = False,
) -> float:
    """Fraction of predicted LCAGs that reconstruct into a valid tree.

    The prediction need not be *correct* -- only realisable. Reconstruction is the
    definition of validity, so this reuses
    :func:`partiqledtr.data.lcag.lcag_to_adjacency`.

    The default is an **optimistic** measure and has to be reported as one. It is
    permissive in two separate ways (``DECISIONS.md`` D42, D85): reconstruction is
    greedy, so a matrix consistent with no single tree can still reduce to one --
    on random symmetric matrices roughly 70% of the accepted ones do not reproduce
    their own input LCAG -- and dropping the leaves a prediction calls
    disconnected means a prediction that keeps only a single pair scores a valid
    tree, which nothing in the loss discourages because class 0 carries weight 0.
    The lenient definition is kept as the default because it is the one the
    reconstruction papers used and the numbers have to stay comparable.

    ``strict=True`` closes both holes: every scored leaf must survive, and the
    reconstructed tree must re-derive the very matrix it came from. Report it as
    the primary number and the lenient one for comparability.

    Args:
        predictions: Integer class indices, shape ``(B, L, L)``.
        labels: Ground-truth class indices of the same shape, used only for its
            padding mask.
        ignore_disconnected: Drop leaves the prediction marks as disconnected
            (class 0) before reconstructing, as baumbauen does.
        strict: Require that no scored leaf is dropped and that the reconstructed
            tree reproduces the predicted LCAG.

    Returns:
        Fraction of events that reconstruct, or NaN for an empty batch.

    Raises:
        ValueError: If the shapes differ or are not three-dimensional.
        Exception: Any error from reconstruction other than
            :class:`~partiqledtr.data.lcag.InvalidLCAGError` propagates -- a
            silent ``False`` would hide real bugs.
    """
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if predictions.shape != labels.shape:
        raise ValueError(f"shape mismatch: predictions {predictions.shape}, labels {labels.shape}")
    if labels.ndim != 3:
        raise ValueError(f"expected (B, L, L) labels, got shape {labels.shape}")
    if labels.shape[0] == 0:
        return float("nan")

    valid = [
        _reconstructs(p, y, ignore_disconnected, strict)
        for p, y in zip(predictions, labels, strict=True)
    ]
    return float(np.mean(valid))


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced class weights over the scored cells.

    ``weight[c] = n_scored / (n_classes * count[c])``, the usual balanced
    weighting. Classes absent from ``labels`` get weight 0: they cannot be a
    target, and any other value would be an invented penalty.

    Args:
        labels: Ground-truth class indices; ignored cells are excluded.
        n_classes: Size of the class set.

    Returns:
        Float64 array of shape ``(n_classes,)``.

    Raises:
        ValueError: If ``n_classes`` is below 2 or a label exceeds it.
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}")

    labels = np.asarray(labels)
    scored = labels[_scored(labels, IGNORE)]
    if scored.size and scored.max() >= n_classes:
        raise ValueError(f"label {scored.max()} exceeds n_classes={n_classes}")

    counts = np.bincount(scored, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = scored.size / (n_classes * counts[present])
    return weights
