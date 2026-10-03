"""Measure encoded-state purity, circuit-state purity, and dynamical Lie algebras.

Product-state purity uses angles from one encoding layer before the first
trainable block. Exact purity uses the circuit's prepared state. The DLA
certificate records the basis dimension and diagonal-word floor of each arm.
"""

import functools
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from fluksio import Port, node
from jaqsi import PauliWord
from qml_essentials.algebra import g_purity_from_basis, lie_closure_paulis

from partiqledtr.ansaetze import circuit

_SINGLE_QUBIT_GENERATOR = {"RX": "X", "RY": "Y", "RZ": "Z"}
_TWO_QUBIT_GENERATOR = {"RXX": "XX", "RYY": "YY", "RZZ": "ZZ"}


# --- g-purity closed forms --------------------------------------------------


def g_purity_offdiag(theta: jax.Array) -> jax.Array:
    """Return the ``XY_Brickwork`` g-purity of an RY product state.

    Only odd-separation X..X basis words contribute. An O(n) two-parity
    recurrence evaluates their sum; clustered angles drive it to zero. Ported
    from ``reference/unflattening/unflattening/utils/purity.py``.

    Args:
        theta: Encoding angles of shape ``(..., n)``.

    Returns:
        The g-purity of shape ``(...)``.
    """
    c, s = jnp.cos(theta) ** 2, jnp.sin(theta) ** 2
    purity = jnp.zeros(theta.shape[:-1])
    # odd/even = sum over j < k of s_j prod_{j<l<k} c_l, restricted to k - j of
    # that parity; both flip parity when k advances.
    odd = jnp.zeros(theta.shape[:-1])
    even = jnp.zeros(theta.shape[:-1])
    for k in range(theta.shape[-1]):
        purity = purity + s[..., k] * odd
        odd, even = c[..., k] * even + s[..., k], c[..., k] * odd
    return purity


def offdiag_uniform_mean(n: int) -> float:
    r"""Return :math:`\mathbb E_\Theta[P_{\mathfrak g}]` of :func:`g_purity_offdiag`.

    Under the iid uniform prior :math:`\theta_k \sim U[0, 2\pi)` every
    :math:`\sin^2` and :math:`\cos^2` factor averages to :math:`1/2`, so
    :math:`\mu_n = \sum_{d \ \mathrm{odd}} (n - d) 2^{-(d+1)} \to n/3 - 5/9`.
    This is the reference scale of the whitening acceptance test of the
    unflattening manuscript, which accepts a rotation when the dataset-averaged
    purity reaches :math:`\mu_n / 2`.  Port of ``offdiag_uniform_mean`` in
    ``reference/unflattening/unflattening/utils/purity.py``.

    Args:
        n: Number of qubits.

    Returns:
        The prior mean :math:`\mu_n`.
    """
    return sum((n - d) * 2.0 ** -(d + 1) for d in range(1, n, 2))


@functools.cache
def _yfree_masks(ansatz: str, n_qubits: int) -> tuple[jax.Array, jax.Array]:
    """Return the ``X`` and ``Z`` placement masks of the Y-free DLA basis words."""
    words = [word.to_pauli_string() for word in dla_basis(ansatz, n_qubits)]
    keep = [word for word in words if "Y" not in word]
    # Shape (0, n) rather than a list comprehension over nothing: an algebra whose
    # every word carries a Y has purity 0 on an RY product state, not 1.
    placements = np.zeros((len(keep), n_qubits, 2), dtype=bool)
    for row, word in enumerate(keep):
        for column, letter in enumerate(word):
            placements[row, column] = (letter == "X", letter == "Z")
    return jnp.asarray(placements[..., 0]), jnp.asarray(placements[..., 1])


def product_state_purity(theta: jax.Array, ansatz: str) -> jax.Array:
    """Return the g-purity of an RY product state for any ansatz arm.

    Only Y-free DLA words contribute. Each word contributes the product of
    squared sine factors at X sites and squared cosine factors at Z sites.

    Args:
        theta: Encoded angles of shape ``(..., n_qubits)`` of a single encoding
            layer, never scaled by the depth (see the module docstring).
        ansatz: One of :data:`partiqledtr.ansaetze.ANSAETZE`.

    Returns:
        The g-purity of shape ``(...)``.
    """
    xs, zs = _yfree_masks(ansatz, theta.shape[-1])
    sin2, cos2 = jnp.sin(theta) ** 2, jnp.cos(theta) ** 2
    term = jnp.where(xs, sin2[..., None, :], 1.0) * jnp.where(zs, cos2[..., None, :], 1.0)
    return jnp.prod(term, axis=-1).sum(-1)


def uniform_prior_mean(ansatz: str, n_qubits: int) -> float:
    r"""Return :math:`\mathbb E_\Theta[P_{\mathfrak g}]` under the iid uniform prior.

    Every :math:`\sin^2` and :math:`\cos^2` factor averages to :math:`1/2`, so a
    Y-free basis word acting non-trivially on :math:`m` qubits contributes
    :math:`2^{-m}`. The generalisation of :func:`offdiag_uniform_mean` to an
    arbitrary arm, and the reference scale every purity is read against: the
    whitening acceptance test of the unflattening manuscript accepts a rotation
    at half this value.

    Args:
        ansatz: One of :data:`partiqledtr.ansaetze.ANSAETZE`.
        n_qubits: Number of qubits.

    Returns:
        The prior mean.
    """
    xs, zs = _yfree_masks(ansatz, n_qubits)
    return float(jnp.sum(0.5 ** jnp.sum(xs | zs, axis=-1)))


# --- exact g-purity ---------------------------------------------------------


@functools.cache
def dla_basis(ansatz: str, n_qubits: int) -> tuple[PauliWord, ...]:
    """Return the DLA basis of an ansatz arm, cached across calls.

    Args:
        ansatz: One of :data:`partiqledtr.ansaetze.ANSAETZE`.
        n_qubits: Number of qubits, at least 2.

    Returns:
        The Lie-closure basis words of :func:`ansatz_generators`.
    """
    return tuple(lie_closure_paulis(ansatz_generators(ansatz, n_qubits)))


def g_purity_exact(states: np.ndarray, basis: Sequence[PauliWord]) -> np.ndarray:
    """Sum squared DLA-basis expectations on the supplied statevectors.

    Unlike product-state purity, this measures the prepared circuit state.

    Args:
        states: ``(..., 2 ** n)`` statevectors.
        basis: DLA basis words, e.g. from :func:`dla_basis`.

    Returns:
        ``(...)`` g-purities.

    Raises:
        ValueError: If ``states`` is empty along its last axis.
    """
    states = np.asarray(states)
    if states.shape[-1] < 2:
        raise ValueError(f"expected statevectors on the last axis, got {states.shape}")
    flat = states.reshape(-1, states.shape[-1])
    values = np.array([g_purity_from_basis(row, basis) for row in flat], dtype=float)
    return values.reshape(states.shape[:-1])


# --- angle distribution -----------------------------------------------------

#: Bins used by :func:`angle_stats` for the total-variation distance to uniform.
#: TV over a histogram depends on the binning, so it is fixed here and reported
#: rather than passed in: two runs are only comparable at the same resolution.
TV_BINS = 36


def angle_stats(angles: np.ndarray) -> dict[str, list[float]]:
    """Describe encoded angle distributions separately for each qubit.

    ``mean_sin2`` distinguishes angles spread toward uniform from angles pinned
    near pi/2, which can have similar g-purity. ``tv_uniform`` depends on the
    sample size and :data:`TV_BINS`; kinematics bounds some sites away from
    uniformity, so compare those sites against the raw encoding.

    Args:
        angles: ``(n_samples, n_qubits)`` encoded angles, any real values; they are
            wrapped into ``[0, 2 pi)`` first.

    Returns:
        ``tv_uniform``, ``mean_sin2`` and ``circular_variance``, each a list with
        one entry per qubit, plus ``n_samples`` and ``n_bins`` as one-element lists
        so the record stays json-shaped.

    Raises:
        ValueError: If ``angles`` is not two-dimensional or has no samples.
    """
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 2:
        raise ValueError(f"expected (n_samples, n_qubits) angles, got shape {angles.shape}")
    if angles.shape[0] == 0:
        raise ValueError("no angles to describe")

    wrapped = np.mod(angles, 2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, TV_BINS + 1)
    uniform = 1.0 / TV_BINS

    tv, mean_sin2, circular = [], [], []
    for site in range(wrapped.shape[1]):
        column = wrapped[:, site]
        density = np.histogram(column, bins=edges)[0] / len(column)
        tv.append(float(0.5 * np.abs(density - uniform).sum()))
        mean_sin2.append(float(np.mean(np.sin(column) ** 2)))
        circular.append(float(1.0 - abs(np.exp(1j * column).mean())))
    return {
        "tv_uniform": tv,
        "mean_sin2": mean_sin2,
        "circular_variance": circular,
        "n_samples": [float(wrapped.shape[0])],
        "n_bins": [float(TV_BINS)],
    }


# --- encoding comparison ----------------------------------------------------

#: Feature encoding -> (dataset array, angle columns) for :func:`encoding_purity`.
#: ``"angles"`` is the direct ``(theta, phi)`` arm; the other two go through an angle
#: map in :mod:`partiqledtr.models.qfm`.
_PURITY_ARMS = {
    "pair_polar": ("features_cartesian", "pair_polar"),
    "direct": ("features_angles", None),
    "legacy": ("features_legacy", "legacy"),
}


def encoding_purity(
    split: dict[str, np.ndarray],
    *,
    ansatz: str = "XY_Ring",
    n_pairs: int = 4096,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, float]]]:
    """Measure g-purity for each feature map and encoding-weight cell.

    The feature maps are ``pair_polar``, direct direction angles, and the
    clustered ``legacy`` product map. All cells use the same sampled edges.

    Args:
        split: A loaded dataset split; needs the feature array of every arm plus
            ``n_fsps``.
        ansatz: Ansatz arm whose DLA basis the purity is summed over. Defaults to
            the live floor-free arm, on which the input distribution decides.
        n_pairs: Edges sampled per arm.
        seed: Seed of the edge sampling; the same edges are used for every cell, so
            the cells differ only by their encoding.

    Returns:
        ``report[feature_arm][weight_cell]`` with ``mean_purity``,
        ``below_threshold`` (fraction of edges under ``mu_n / 2``), ``threshold``
        and ``uniform_mean``, plus a top-level ``"ansatz"`` and ``"n_pairs"``.

    Raises:
        ValueError: If ``split`` is missing an arm's feature array.
    """
    import itertools

    from partiqledtr.data.whitening import _sample_pairs
    from partiqledtr.models.qfm import (
        ANGLE_MAPS,
        ENC_REUPLOAD,
        ENC_WEIGHTS,
        N_QUBITS,
        encoding_matrix,
    )

    n_fsps = np.asarray(split["n_fsps"])
    # One edge sample shared by every cell: the cells must differ by their encoding
    # and by nothing else.
    events, pairs = _sample_pairs(np.random.default_rng(seed), n_fsps, n_pairs)
    uniform_mean = uniform_prior_mean(ansatz, N_QUBITS)
    threshold = uniform_mean / 2.0

    report: dict[str, Any] = {"ansatz": ansatz, "n_pairs": n_pairs}
    for arm, (key, angle_map) in _PURITY_ARMS.items():
        if key not in split:
            raise ValueError(
                f"split has no '{key}' array for arm {arm!r}; it holds {sorted(split)}"
            )
        features = jnp.asarray(np.asarray(split[key])[events])
        angles = features[..., :2] if angle_map is None else ANGLE_MAPS[angle_map](features)
        taken = jnp.take_along_axis(angles, jnp.asarray(pairs)[:, :, None], axis=1)
        edge = taken.reshape(n_pairs, N_QUBITS)
        cells: dict[str, dict[str, float]] = {}
        for weights, reupload in itertools.product(ENC_WEIGHTS, ENC_REUPLOAD):
            matrix = jnp.asarray(encoding_matrix(weights, reupload, N_QUBITS))
            purity = np.asarray(product_state_purity(edge @ matrix.T, ansatz))
            cells[f"{weights}-{reupload}"] = {
                "mean_purity": float(purity.mean()),
                "below_threshold": float((purity < threshold).mean()),
                "threshold": threshold,
                "uniform_mean": uniform_mean,
            }
        report[arm] = cells
    return report


@node(
    # One purity pass per encoding arm, silent throughout.
    timeout=1800,
    requires=[Port("dataset_train", "artifact")],
    provides=[Port("encoding_report", "json")],
)
def encoding_report(
    *,
    dataset_train: dict[str, Any],
    purity_ansatz: str = "XY_Ring",
    n_pairs: int = 4096,
    encoding_seed: int = 0,
) -> dict[str, Any]:
    """Record the encoding comparison of :func:`encoding_purity` for a dataset.

    Args:
        dataset_train: Training split artifact reference.
        purity_ansatz: Ansatz arm whose DLA basis the purity is summed over.
        n_pairs: Edges sampled per arm.
        encoding_seed: Seed of the edge sampling.

    Returns:
        The per-arm report under the ``encoding_report`` port.
    """
    from partiqledtr.data.dataset import load_split

    return {
        "encoding_report": encoding_purity(
            load_split(dataset_train),
            ansatz=purity_ansatz,
            n_pairs=n_pairs,
            seed=encoding_seed,
        )
    }


# --- DLA pre-check ----------------------------------------------------------


def _pauli_word(n_qubits: int, *placements: tuple[int, str]) -> str:
    """Return the Pauli string with the given ``(qubit, pauli)`` placements.

    Args:
        n_qubits: Length of the string; all other positions are ``I``.
        *placements: Pairs of qubit index (0 leftmost) and Pauli label.

    Returns:
        The Pauli string over ``{'I', 'X', 'Y', 'Z'}``.
    """
    word = ["I"] * n_qubits
    for qubit, pauli in placements:
        word[qubit] = pauli
    return "".join(word)


def ansatz_generators(ansatz: str, n_qubits: int) -> list[str]:
    """Return deduplicated Pauli generators from an ansatz's gate structure.

    Single- and two-qubit rotations contribute their Pauli words. ``CRX(c, t)``
    contributes ``X_t`` and ``Z_c X_t`` from its controlled generator.

    Args:
        ansatz: One of :data:`partiqledtr.ansaetze.ANSAETZE`.
        n_qubits: Number of qubits, at least 2.

    Returns:
        The deduplicated generator Pauli strings (qubit 0 leftmost).

    Raises:
        ValueError: If ``ansatz`` names no arm and no qml-essentials ansatz, or
            ``n_qubits < 2``.
        NotImplementedError: If the ansatz contains a rotation gate whose
            generator is not covered here.
    """
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be at least 2, got {n_qubits}")

    words: list[str] = []
    for block in circuit(ansatz).structure():
        gate = block.gate.__name__
        if gate in _SINGLE_QUBIT_GENERATOR:
            pauli = _SINGLE_QUBIT_GENERATOR[gate]
            wires = block.wires if block.wires is not None else range(n_qubits)
            words += [_pauli_word(n_qubits, (q, pauli)) for q in wires]
        elif gate in _TWO_QUBIT_GENERATOR:
            pj, pk = _TWO_QUBIT_GENERATOR[gate]
            for j, k in block.topology(n_qubits=n_qubits, **block.kwargs):
                words.append(_pauli_word(n_qubits, (j, pj), (k, pk)))
        elif gate == "CRX":
            for control, target in block.topology(n_qubits=n_qubits, **block.kwargs):
                words.append(_pauli_word(n_qubits, (target, "X")))
                words.append(_pauli_word(n_qubits, (control, "Z"), (target, "X")))
        else:
            raise NotImplementedError(f"no generator rule for gate {gate!r} in {ansatz!r}")
    return list(dict.fromkeys(words))


def dla_check(
    ansatz: str = "XY_Brickwork", n_qubits: int = 4, max_dim: int = 2000
) -> dict[str, str | int | float | bool]:
    """Measure an ansatz's DLA dimension and diagonal-word purity floor.

    The Lie closure of :func:`ansatz_generators` supplies the basis. Its Z-only
    word count is the clustered-angle g-purity floor.

    Args:
        ansatz: One of :data:`partiqledtr.ansaetze.ANSAETZE`.
        n_qubits: Number of qubits, at least 2.
        max_dim: Cap on the number of basis words, so a runaway closure stops
            instead of hanging.

    Returns:
        A json-serialisable dict with ``ansatz``, ``n_qubits``, ``dim_g``,
        ``dim_su`` (:math:`= 4^n - 1`), ``ratio`` (``dim_g / dim_su``),
        ``n_diag_words`` and ``capped``.  When ``capped`` is true ``dim_g`` is
        only a lower bound.

    Raises:
        ValueError: If ``ansatz`` is unknown, ``n_qubits < 2``, or ``max_dim``
            is smaller than the number of generators.
    """
    generators = ansatz_generators(ansatz, n_qubits)
    if max_dim < len(generators):
        raise ValueError(
            f"max_dim={max_dim} is below the {len(generators)} generators of {ansatz!r}"
        )
    # A result of exactly `max_dim` words means growth was stopped there, so the
    # basis is partial and `dim_g` is a lower bound.
    words = [word.to_pauli_string() for word in lie_closure_paulis(generators, max_dim=max_dim)]
    capped = len(words) >= max_dim
    dim_su = 4**n_qubits - 1
    return {
        "ansatz": ansatz,
        "n_qubits": n_qubits,
        "dim_g": len(words),
        "dim_su": dim_su,
        "ratio": len(words) / dim_su,
        "n_diag_words": sum(1 for w in words if set(w) <= {"I", "Z"}),
        "capped": capped,
    }


@node(
    # Circuit_19's Lie closure is slow and silent.
    timeout=1800,
    requires=[Port("ansatz", "str"), Port("n_qubits", "int")],
    provides=[Port("dla_report", "json")],
)
def dla_report(*, ansatz: str = "XY_Brickwork", n_qubits: int = 4, max_dim: int = 4200) -> dict:
    """Record an ansatz arm's dynamical Lie algebra before any training.

    The DLA and floor count of each arm are recorded *before* training. Wiring
    this node upstream of the fit makes that a property of the flow rather than
    of anyone's discipline.

    Args:
        ansatz: Ansatz arm, one of :data:`partiqledtr.ansaetze.ANSAETZE`.
        n_qubits: Qubits per edge QFM.
        max_dim: Cap on the Lie closure; see :func:`dla_check`.

    Returns:
        The certificate of :func:`dla_check`, under the ``dla_report`` port.
    """
    return {"dla_report": dla_check(ansatz=ansatz, n_qubits=n_qubits, max_dim=max_dim)}


# --- arm characterisation ---------------------------------------------------


@node(
    # Circuit_19's Lie closure dominates and is silent throughout.
    timeout=1800,
    requires=[Port("n_qubits", "int")],
    provides=[Port("arm_report", "json")],
)
def arm_report(*, n_qubits: int = 4, max_dim: int = 2000) -> dict[str, Any]:
    """Record each arm's DLA certificate, uniform-prior scale, and swap symmetry.

    Args:
        n_qubits: Qubits per edge QFM.
        max_dim: Cap on each Lie closure; see :func:`dla_check`.

    Returns:
        One record per arm under the ``arm_report`` port, plus its bonds.
    """
    from partiqledtr.ansaetze import ANSAETZE, bonds, swap_invariant

    return {
        "arm_report": {
            arm: {
                **dla_check(arm, n_qubits=n_qubits, max_dim=max_dim),
                "uniform_mean": uniform_prior_mean(arm, n_qubits),
                "swap_invariant": swap_invariant(arm, n_qubits),
                "bonds": sorted(sorted(bond) for bond in bonds(arm, n_qubits)),
            }
            for arm in ANSAETZE
        }
    }


@node(
    timeout=1800,
    requires=[Port("n_qubits", "int"), Port("purity_ansatz", "str")],
    provides=[Port("encoding_cells", "json")],
)
def encoding_cells(
    *,
    n_qubits: int = 4,
    purity_ansatz: str = "XY_Ring",
    n_layers: int = 2,
    n_samples: int = 200_000,
    clustered_sigmas: tuple[float, ...] = (0.30, 0.03),
    cells_seed: int = 0,
) -> dict[str, Any]:
    """Characterise encoding weights and price synthetic angle laws.

    For each cell, record its weight matrix, spectrum, dissociation, endpoint
    symmetry, and purity on uniform and clustered angles.

    Args:
        n_qubits: Qubits per edge QFM.
        purity_ansatz: Arm whose DLA basis the purities are summed over.
        n_layers: Data-reuploading depth the spectrum is computed at.
        n_samples: Angle draws per law.
        clustered_sigmas: Standard deviations of the clustered laws, in radians.
        cells_seed: Seed of the angle draws.

    Returns:
        Per cell: ``weights`` (the matrix), ``n_freqs``, ``dissociated``,
        ``swap_invariant`` and a ``purity`` map over the angle laws.
    """
    import itertools

    from partiqledtr.models.qfm import (
        ENC_REUPLOAD,
        ENC_WEIGHTS,
        encoding_matrix,
        encoding_spectrum,
    )

    rng = np.random.default_rng(cells_seed)
    laws = {"uniform": rng.uniform(0.0, 2.0 * np.pi, (n_samples, n_qubits))}
    laws |= {f"clustered_{s}": rng.normal(0.0, s, (n_samples, n_qubits)) for s in clustered_sigmas}
    # Every eps in {-1,0,1}^n except zero: dissociation is checked by exhaustion
    # rather than cited, which is cheap at this size and is the whole premise.
    signs = np.array(list(itertools.product((-1, 0, 1), repeat=n_qubits)))
    signs = signs[np.any(signs != 0, axis=1)]
    half = n_qubits // 2
    order = [(q + half) % n_qubits for q in range(n_qubits)]
    threshold = uniform_prior_mean(purity_ansatz, n_qubits) / 2.0

    report: dict[str, Any] = {
        "ansatz": purity_ansatz,
        "n_layers": n_layers,
        "n_samples": n_samples,
        "threshold": threshold,
    }
    for weights, reupload in itertools.product(ENC_WEIGHTS, ENC_REUPLOAD):
        matrix = encoding_matrix(weights, reupload, n_qubits)
        purity = {}
        for name, angles in laws.items():
            values = np.asarray(
                product_state_purity(jnp.asarray(angles) @ jnp.asarray(matrix).T, purity_ansatz)
            )
            purity[name] = {
                "mean_purity": float(values.mean()),
                "below_threshold": float((values < threshold).mean()),
            }
        report[f"{weights}-{reupload}"] = {
            "weights": matrix.astype(int).tolist(),
            "n_freqs": [int(encoding_spectrum(matrix, f, n_layers).size) for f in range(n_qubits)],
            "dissociated": bool(np.all(np.any(signs @ matrix != 0, axis=1))),
            "swap_invariant": bool(np.array_equal(matrix[order][:, order], matrix)),
            "purity": purity,
        }
    return {"encoding_cells": report}
