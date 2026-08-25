"""LCAG metrics: hand-computed values, mask semantics and degenerate inputs."""

import numpy as np
import pytest

from partiqledtr.metrics import (
    IGNORE_PRIMARY,
    class_weights,
    masked_accuracy,
    perfect_lcag_rate,
    valid_tree_rate,
)

# The reference tree a -> {b -> {d, e, f}, c -> {g, h}}: siblings share a parent
# one generation up, the two sibling groups meet two generations up.
GOLDEN_LCAG = np.array(
    [
        [0, 1, 1, 2, 2],
        [1, 0, 1, 2, 2],
        [1, 1, 0, 2, 2],
        [2, 2, 2, 0, 1],
        [2, 2, 2, 1, 0],
    ]
)

# Leaf 1 would need two different parents (it pairs with leaf 0 and with leaf 2
# one generation up), while leaves 0 and 2 meet only two generations up.
IMPOSSIBLE_LCAG = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])


def _labelled(lcag: np.ndarray) -> np.ndarray:
    """Apply the dataset convention: -1 on the diagonal."""
    out = lcag.copy()
    np.fill_diagonal(out, -1)
    return out


def test_masked_accuracy_scores_only_unignored_cells():
    labels = _labelled(GOLDEN_LCAG)
    predictions = GOLDEN_LCAG.copy()
    # The diagonal is ignored, so garbage there must not matter.
    np.fill_diagonal(predictions, 7)
    assert masked_accuracy(predictions, labels) == 1.0

    predictions[0, 3] = 1  # one of the 20 scored off-diagonal cells
    assert masked_accuracy(predictions, labels) == pytest.approx(19 / 20)


def test_masked_accuracy_primary_drops_class_zero():
    labels = np.array([[-1, 0, 1], [0, -1, 2], [1, 2, -1]])
    predictions = np.array([[0, 5, 1], [5, 0, 2], [1, 2, 0]])

    # Plain: 4 of 6 scored cells correct (the two zeros are predicted as 5).
    assert masked_accuracy(predictions, labels) == pytest.approx(4 / 6)
    # Primary: the zeros are not scored at all, so the rest is perfect.
    assert masked_accuracy(predictions, labels, IGNORE_PRIMARY) == 1.0


def test_masked_accuracy_is_nan_when_nothing_is_scored():
    labels = np.full((3, 3), -1)
    assert np.isnan(masked_accuracy(labels.copy(), labels))


def test_perfect_lcag_rate_is_all_or_nothing_per_event():
    labels = np.stack([_labelled(GOLDEN_LCAG)] * 2)
    predictions = np.stack([GOLDEN_LCAG, GOLDEN_LCAG])
    assert perfect_lcag_rate(predictions, labels) == 1.0

    predictions[1, 0, 3] = predictions[1, 3, 0] = 1
    assert perfect_lcag_rate(predictions, labels) == 0.5


def test_perfect_lcag_rate_excludes_events_with_nothing_to_score():
    """A fully padded event is not a free success -- it leaves the denominator."""
    labels = np.stack([_labelled(GOLDEN_LCAG), np.full((5, 5), -1)])
    predictions = np.stack([GOLDEN_LCAG, np.full((5, 5), 3)])
    assert perfect_lcag_rate(predictions, labels) == 1.0

    assert np.isnan(perfect_lcag_rate(np.full((1, 5, 5), 3), np.full((1, 5, 5), -1)))


def test_valid_tree_rate_counts_reconstructable_predictions():
    labels = np.stack([_labelled(GOLDEN_LCAG)] * 2)
    predictions = np.stack([GOLDEN_LCAG, GOLDEN_LCAG])
    assert valid_tree_rate(predictions, labels) == 1.0

    padded_impossible = np.zeros((5, 5), dtype=int)
    padded_impossible[:3, :3] = IMPOSSIBLE_LCAG
    labels = np.stack([_labelled(GOLDEN_LCAG), _labelled(padded_impossible)])
    predictions = np.stack([GOLDEN_LCAG, padded_impossible])
    assert valid_tree_rate(predictions, labels) == 0.5


def test_valid_tree_rate_rejects_an_all_zero_prediction():
    labels = np.stack([_labelled(GOLDEN_LCAG)])
    assert valid_tree_rate(np.zeros((1, 5, 5), dtype=int), labels) == 0.0


def test_class_weights_ignore_padding_and_balance_counts():
    # Scored cells (diagonal ignored): 6 ordered pairs within {d, e, f} plus 2 within
    # {g, h} give eight 1s; the 3 x 2 cross pairs, both orders, give twelve 2s.
    labels = _labelled(GOLDEN_LCAG)[None]
    weights = class_weights(labels, n_classes=3)

    assert weights.shape == (3,)
    assert weights[0] == 0.0  # class 0 never occurs
    counts = np.array([0, 8, 12])
    expected = np.where(counts > 0, 20 / (3 * np.maximum(counts, 1)), 0.0)
    assert weights == pytest.approx(expected)
    # The rarer class is weighted up.
    assert weights[1] > weights[2]


def test_metrics_validate_their_inputs():
    labels = _labelled(GOLDEN_LCAG)[None]
    with pytest.raises(ValueError, match="shape mismatch"):
        masked_accuracy(np.zeros((1, 4, 4), dtype=int), labels)
    with pytest.raises(ValueError, match=r"\(B, L, L\)"):
        perfect_lcag_rate(labels[0], labels[0])
    with pytest.raises(ValueError, match="n_classes"):
        class_weights(labels, n_classes=1)
    with pytest.raises(ValueError, match="exceeds n_classes"):
        class_weights(labels, n_classes=2)


def test_strict_valid_tree_rejects_a_prediction_that_keeps_only_one_pair():
    """The lenient metric's second hole, and why strict is the primary number (D85).

    Dropping the leaves a prediction calls disconnected means a model can score a
    valid tree by predicting class 0 everywhere but one pair -- and nothing in the
    loss discourages that, since class 0 is never a target and carries weight 0.
    """
    labels = np.ones((1, 5, 5), dtype=int)
    np.fill_diagonal(labels[0], -1)
    degenerate = np.zeros((1, 5, 5), dtype=int)
    degenerate[0, 0, 1] = degenerate[0, 1, 0] = 1

    assert valid_tree_rate(degenerate, labels) == 1.0
    assert valid_tree_rate(degenerate, labels, strict=True) == 0.0


def test_strict_valid_tree_accepts_a_genuine_lcag():
    """A real LCAG has to survive the strict test, or it would only measure rigour."""
    from partiqledtr.data.lcag import topology_to_lcag
    from partiqledtr.data.topology import sample_topology

    lcag = topology_to_lcag(sample_topology(np.random.default_rng(3), n_fsps=5))[0].astype(int)
    labels = lcag.copy()[None]
    np.fill_diagonal(labels[0], -1)

    assert valid_tree_rate(lcag[None], labels, strict=True) == 1.0
