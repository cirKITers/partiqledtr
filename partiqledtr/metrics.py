"""Evaluate LCAG class predictions on ground-truth valid cells.

Inputs are predicted class indices. ``IGNORE`` excludes padding and diagonal
cells; ``IGNORE_PRIMARY`` also excludes class 0. Valid-tree metrics reconstruct
one tree per event.
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
    """Return the fraction of events with every scored LCAG cell correct.

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
        # it carries weight 0.
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
    """Return the fraction of predictions reconstructing to a tree.

    The default greedy criterion is permissive: it can drop disconnected leaves
    and accept a matrix that the reconstructed tree cannot reproduce.
    ``strict=True`` requires every scored leaf and an exact LCAG round trip.

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
