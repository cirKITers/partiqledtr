"""Kinematic features, scale-only normalisation and padding."""

import numpy as np
import pytest

from partiqledtr.data.features import (
    apply_normalization,
    featurize,
    normalization_scales,
    pad_events,
    to_cartesian,
)
from partiqledtr.models.qfm import pair_polar


def test_featurize_angles_by_hand():
    # [px, py, pz, E]; massless, so |p| = E.
    p4 = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],  # along +z:  theta = 0
            [0.0, 0.0, -1.0, 1.0],  # along -z: theta = pi
            [1.0, 0.0, 0.0, 1.0],  # along +x: theta = pi/2, phi = 0
            [0.0, 1.0, 0.0, 1.0],  # along +y: theta = pi/2, phi = pi/2
            [-1.0, 0.0, 0.0, 1.0],  # along -x: theta = pi/2, phi = pi (arctan2 upper bound)
        ]
    )
    expected = [
        [0.0, 0.0, 1.0],
        [np.pi, 0.0, 1.0],
        [np.pi / 2, 0.0, 1.0],
        [np.pi / 2, np.pi / 2, 1.0],
        [np.pi / 2, np.pi, 1.0],
    ]
    np.testing.assert_allclose(featurize(p4), expected, atol=1e-12)


def test_featurize_angle_ranges():
    rng = np.random.default_rng(3)
    features = featurize(rng.normal(size=(200, 7, 4)))

    assert features.shape == (200, 7, 3)
    assert np.all((features[..., 0] >= 0.0) & (features[..., 0] <= np.pi))
    assert np.all((features[..., 1] > -np.pi) & (features[..., 1] <= np.pi))


def test_featurize_zero_momentum_is_finite_and_quiet():
    p4 = np.zeros((3, 4))
    p4[1, 3] = 5.0  # zero momentum, non-zero energy
    with np.errstate(all="raise"):
        features = featurize(p4)
    np.testing.assert_array_equal(features, [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]])


def test_featurize_cartesian_passes_through_without_aliasing():
    p4 = np.arange(8, dtype=float).reshape(2, 4)
    features = featurize(p4, encoding="cartesian")

    np.testing.assert_array_equal(features, p4)
    features[0, 0] = -1.0
    assert p4[0, 0] == 0.0


def test_featurize_rejects_bad_input():
    with pytest.raises(ValueError, match="4 components"):
        featurize(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="encoding"):
        featurize(np.zeros((2, 4)), encoding="spherical")


def test_to_cartesian_round_trips_massless_four_vectors():
    rng = np.random.default_rng(5)
    momentum = rng.normal(size=(100, 3))
    p4 = np.concatenate([momentum, np.linalg.norm(momentum, axis=-1, keepdims=True)], axis=-1)

    np.testing.assert_allclose(to_cartesian(featurize(p4)), p4, atol=1e-12)


def test_to_cartesian_rejects_bad_input():
    with pytest.raises(ValueError, match="3 components"):
        to_cartesian(np.zeros((2, 4)))


def test_pair_polar_matches_the_unflattening_convention():
    # arctan2(x[1::2], x[0::2]) mapped into [0, 2pi): the second pair is the one the
    # mod matters for (arctan2 returns -pi/4 there).
    x = np.array([[1.0, 0.0, 1.0, -1.0]])
    np.testing.assert_allclose(pair_polar(x), [[0.0, 7 * np.pi / 4]], atol=1e-6)


def test_pair_polar_range_and_shape():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4, 6, 4))
    angles = np.asarray(pair_polar(x))

    assert angles.shape == (4, 6, 2)
    assert np.all((angles >= 0.0) & (angles < 2 * np.pi))
    np.testing.assert_allclose(
        angles, np.mod(np.arctan2(x[..., 1::2], x[..., 0::2]), 2 * np.pi), atol=1e-6
    )


def test_pair_polar_rejects_non_four_vectors():
    with pytest.raises(ValueError, match="four-vectors"):
        pair_polar(np.zeros((2, 5)))


def test_normalization_is_scale_only_and_ignores_padding():
    features = np.array([[[0.3, 1.2, 4.0], [1.0, -2.0, 6.0], [0.0, 0.0, 0.0]]])
    scales = normalization_scales(features, "angles")

    assert scales == {"energy": 5.0}  # mean over the two unpadded rows, not the padded one
    normalized = apply_normalization(features, scales, "angles")
    np.testing.assert_allclose(normalized[..., :2], features[..., :2])  # angles untouched
    np.testing.assert_allclose(normalized[..., 2], [[0.8, 1.2, 0.0]])
    assert np.all(normalized[..., 2] >= 0.0)


def test_cartesian_normalization_preserves_direction():
    features = np.array([[3.0, 0.0, 4.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
    scales = normalization_scales(features, "cartesian")

    assert scales == {"energy": 10.0, "momentum": 5.0}
    normalized = apply_normalization(features, scales, "cartesian")
    np.testing.assert_allclose(normalized[0], [0.6, 0.0, 0.8, 1.0])
    assert isinstance(scales["momentum"], float)  # json-serialisable


def test_normalization_rejects_bad_input():
    with pytest.raises(ValueError, match="unpadded"):
        normalization_scales(np.zeros((4, 3)), "angles")
    with pytest.raises(ValueError, match="angles features"):
        normalization_scales(np.ones((4, 4)), "angles")
    with pytest.raises(ValueError, match="encoding"):
        normalization_scales(np.ones((4, 3)), "polar")
    with pytest.raises(ValueError, match="positive"):
        apply_normalization(np.ones((4, 3)), {"energy": 0.0}, "angles")
    with pytest.raises(KeyError):
        apply_normalization(np.ones((4, 4)), {"energy": 1.0}, "cartesian")


def test_pad_events_writes_the_ignore_convention():
    features = np.ones((2, 3, 3))
    lcag = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]], dtype=np.int8)

    padded_features, padded_lcag = pad_events(features, lcag, max_fsps=5)

    assert padded_features.shape == (2, 5, 3)
    np.testing.assert_array_equal(padded_features[:, :3], 1.0)
    np.testing.assert_array_equal(padded_features[:, 3:], 0.0)

    assert padded_lcag.shape == (2, 5, 5)
    assert padded_lcag.dtype == np.int8
    expected = np.array(
        [
            [-1, 1, 2, -1, -1],
            [1, -1, 2, -1, -1],
            [2, 2, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(padded_lcag[0], expected)
    np.testing.assert_array_equal(padded_lcag[1], expected)


def test_pad_events_accepts_per_event_labels():
    features = np.zeros((2, 2, 4))
    lcag = np.array([[[0, 1], [1, 0]], [[0, 2], [2, 0]]])

    _, padded_lcag = pad_events(features, lcag, max_fsps=3)

    np.testing.assert_array_equal(padded_lcag[..., 0, 1], [1, 2])
    np.testing.assert_array_equal(padded_lcag[:, 2, :], -1)


def test_pad_events_rejects_oversized_events():
    with pytest.raises(ValueError, match="max_fsps=3"):
        pad_events(np.zeros((1, 4, 3)), np.zeros((4, 4)), max_fsps=3)
    with pytest.raises(ValueError, match="lcag must be"):
        pad_events(np.zeros((1, 4, 3)), np.zeros((3, 3)), max_fsps=5)
    with pytest.raises(ValueError, match=r"\(N, L, F\)"):
        pad_events(np.zeros((4, 3)), np.zeros((4, 4)), max_fsps=5)
