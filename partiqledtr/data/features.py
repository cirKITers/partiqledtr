"""Encode, normalise, and pad particle four-vectors.

Input order is ``[px, py, pz, E]``. The ``angles``, ``cartesian``, and
``legacy`` encodings are stored per split. Scale-only normalisation excludes
padded rows and preserves non-negative energy; padding uses the ``-1`` label
sentinel.
"""

import numpy as np

_FEATURE_DIM = {"angles": 3, "cartesian": 4, "legacy": 4}

#: Encodings whose four-vectors are scaled by the maximum rather than the mean, so
#: that momenta land in ``[-1, 1]`` and energies in ``[0, 1]`` exactly as partiqlegan
#: normalised them.
LEGACY_ENCODING = "legacy"


def featurize(p4: np.ndarray, *, encoding: str = "angles") -> np.ndarray:
    """Turn four-vectors into model features.

    Args:
        p4: ``(..., 4)`` four-vectors laid out ``[px, py, pz, E]``.
        encoding: ``"angles"`` for ``(theta, phi, E)`` with ``theta`` in ``[0, pi]`` and
            ``phi`` in ``(-pi, pi]``, or ``"cartesian"``/``"legacy"`` for the
            four-vectors unchanged. The two four-vector encodings differ only in the
            normalisation applied later, which is what makes ``"legacy"`` clustered.

    Returns:
        ``(..., 3)`` for ``"angles"``, ``(..., 4)`` otherwise.

    Raises:
        ValueError: If the last axis is not 4, or the encoding is unknown.
    """
    p4 = np.asarray(p4, dtype=float)
    if p4.shape[-1] != 4:
        raise ValueError(f"p4 must have 4 components on its last axis, got {p4.shape[-1]}")
    if encoding not in _FEATURE_DIM:
        raise ValueError(f"encoding must be one of {sorted(_FEATURE_DIM)}, got {encoding!r}")
    if encoding in ("cartesian", LEGACY_ENCODING):
        return p4.copy()

    px, py, pz, energy = p4[..., 0], p4[..., 1], p4[..., 2], p4[..., 3]
    momentum = np.sqrt(px**2 + py**2 + pz**2)
    # Divide by 1 where the momentum vanishes, then discard that branch: no NaN, no warning.
    safe = np.where(momentum > 0.0, momentum, 1.0)
    theta = np.where(momentum > 0.0, np.arccos(np.clip(pz / safe, -1.0, 1.0)), 0.0)
    return np.stack((theta, np.arctan2(py, px), energy), axis=-1)


def to_cartesian(features: np.ndarray) -> np.ndarray:
    """Invert :func:`featurize` for the ``"angles"`` encoding.

    Assumes massless final-state particles, ``|p| = E``. Exact for the massless case and
    an approximation otherwise: the reconstructed momentum is too large by ``E - |p|``.

    Args:
        features: ``(..., 3)`` features ``(theta, phi, E)``.

    Returns:
        ``(..., 4)`` four-vectors ``[E sin(theta) cos(phi), E sin(theta) sin(phi),
        E cos(theta), E]``.

    Raises:
        ValueError: If the last axis is not 3.
    """
    features = np.asarray(features, dtype=float)
    if features.shape[-1] != 3:
        raise ValueError(f"features must have 3 components on its last axis, got {features.shape}")
    theta, phi, energy = features[..., 0], features[..., 1], features[..., 2]
    transverse = energy * np.sin(theta)
    return np.stack(
        (transverse * np.cos(phi), transverse * np.sin(phi), energy * np.cos(theta), energy),
        axis=-1,
    )


def normalization_scales(train_features: np.ndarray, encoding: str) -> dict[str, float]:
    """Fit feature scales on unpadded training rows.

    ``angles`` remain unscaled. Other encodings use mean scales, except
    ``legacy``, which uses maxima to keep momenta and energy within unit bounds.

    Args:
        train_features: ``(..., F)`` features of the *training* split only.
        encoding: One of ``"angles"``, ``"cartesian"``, ``"legacy"``.

    Returns:
        ``{"energy": ...}`` for ``"angles"``, plus a single shared ``"momentum"`` scale for
        the four-vector encodings (one scale for all three components, so directions are
        preserved). Plain floats, ready for a json port.

    Raises:
        ValueError: If the encoding is unknown, the feature width does not match it, no
            unpadded row is left, or a scale comes out non-positive.
    """
    features = _as_rows(train_features, encoding)
    unpadded = features[:, -1] > 0.0
    if not unpadded.any():
        raise ValueError("no unpadded rows (E > 0) to compute a normalisation scale from")

    reduce = np.max if encoding == LEGACY_ENCODING else np.mean
    scales = {"energy": float(reduce(features[unpadded, -1]))}
    if encoding in ("cartesian", LEGACY_ENCODING):
        # The legacy arm bounds each *component*, since it is components that get
        # multiplied by the energy; the cartesian arm bounds the vector norm so the
        # direction is untouched.
        magnitude = (
            np.abs(features[unpadded, :3])
            if encoding == LEGACY_ENCODING
            else np.linalg.norm(features[unpadded, :3], axis=-1)
        )
        scales["momentum"] = float(reduce(magnitude))
    if any(scale <= 0.0 for scale in scales.values()):
        raise ValueError(f"normalisation scales must be positive, got {scales}")
    return scales


def apply_normalization(
    features: np.ndarray, scales: dict[str, float], encoding: str
) -> np.ndarray:
    """Divide features by the scales of :func:`normalization_scales`.

    Args:
        features: ``(..., F)`` features.
        scales: Scales from :func:`normalization_scales` for the same encoding.
        encoding: One of ``"angles"``, ``"cartesian"``, ``"legacy"``.

    Returns:
        The rescaled features, same shape. Energies stay non-negative and angles are
        untouched.

    Raises:
        ValueError: If the encoding is unknown, the feature width does not match it, or a
            scale is non-positive.
        KeyError: If a scale required by the encoding is missing.
    """
    features = np.asarray(features, dtype=float)
    if encoding not in _FEATURE_DIM:
        raise ValueError(f"encoding must be one of {sorted(_FEATURE_DIM)}, got {encoding!r}")
    if features.shape[-1] != _FEATURE_DIM[encoding]:
        raise ValueError(
            f"{encoding} features have {_FEATURE_DIM[encoding]} columns, got {features.shape[-1]}"
        )
    if encoding == "angles":
        divisor = np.array([1.0, 1.0, scales["energy"]])
    else:
        momentum = scales["momentum"]
        divisor = np.array([momentum, momentum, momentum, scales["energy"]])
    if np.any(divisor <= 0.0):
        raise ValueError(f"normalisation scales must be positive, got {scales}")
    return features / divisor


def pad_events(
    features: np.ndarray, lcag: np.ndarray, max_fsps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pad features with zero and ignored labels with ``-1``.

    The label diagonal also uses ``-1``.

    Args:
        features: ``(N, L, F)`` features.
        lcag: ``(N, L, L)`` labels, or one ``(L, L)`` matrix shared by all ``N`` events.
        max_fsps: Padded leaf count.

    Returns:
        ``(N, max_fsps, F)`` float features and ``(N, max_fsps, max_fsps)`` int8 labels.

    Raises:
        ValueError: If the shapes are inconsistent or ``L`` exceeds ``max_fsps``.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 3:
        raise ValueError(f"features must be (N, L, F), got shape {features.shape}")
    n_events, n_leaves, n_columns = features.shape

    lcag = np.asarray(lcag)
    if lcag.ndim == 2:
        lcag = np.broadcast_to(lcag, (n_events, *lcag.shape))
    if lcag.shape != (n_events, n_leaves, n_leaves):
        raise ValueError(
            f"lcag must be {(n_events, n_leaves, n_leaves)} or {(n_leaves, n_leaves)}, "
            f"got shape {lcag.shape}"
        )
    if n_leaves > max_fsps:
        raise ValueError(f"event has {n_leaves} leaves, more than max_fsps={max_fsps}")

    padded_features = np.zeros((n_events, max_fsps, n_columns), dtype=float)
    padded_features[:, :n_leaves] = features
    padded_lcag = np.full((n_events, max_fsps, max_fsps), -1, dtype=np.int8)
    padded_lcag[:, :n_leaves, :n_leaves] = lcag
    diagonal = np.arange(max_fsps)
    padded_lcag[:, diagonal, diagonal] = -1
    return padded_features, padded_lcag


def _as_rows(features: np.ndarray, encoding: str) -> np.ndarray:
    """Flatten features to ``(rows, F)`` and check the width against the encoding."""
    if encoding not in _FEATURE_DIM:
        raise ValueError(f"encoding must be one of {sorted(_FEATURE_DIM)}, got {encoding!r}")
    features = np.asarray(features, dtype=float)
    if features.ndim < 2 or features.shape[-1] != _FEATURE_DIM[encoding]:
        raise ValueError(
            f"{encoding} features must be (..., {_FEATURE_DIM[encoding]}), got {features.shape}"
        )
    return features.reshape(-1, features.shape[-1])
