r"""Theory instrumentation for the unflattening connection.

Three things live here, all recorded or tracked around training rather than trained:

1. **Product-state g-purity** -- the trainability observable.  For a
   floor-free, polynomial-DLA ansatz the unflattening result (Theorem 1) states
   :math:`\mathrm{Var}_W[\langle Z_i \rangle] = P_{\mathfrak g}(\rho) / \dim
   \mathfrak g`, so the g-purity of the *encoded input state* alone decides
   trainability.  The encoded state is the RY product state
   :math:`\bigotimes_q R_y(\theta_q)|0\rangle`, on which
   :math:`\langle X \rangle = \sin\theta`, :math:`\langle Y \rangle = 0` and
   :math:`\langle Z \rangle = \cos\theta`.  :func:`product_state_purity` sums the
   squared expectations over the arm's own DLA basis, which reproduces the
   manuscript's hand-derived closed forms to float32 *and* extends to arms that
   have none -- the project's own ansatz arms.
   :func:`g_purity_offdiag` is kept as the ported closed form of
   ``reference/unflattening/unflattening/utils/purity.py``: it prices encodings
   in :func:`encoding_purity` and pins the general form in the tests.

   **Which angles go in.**  The closed forms describe the
   state *entering the first trainable block*, which is the scope the
   unflattening manuscript claims for them under re-uploading: later encoding
   layers act on parameter-dependent entangled states and are not product
   states at all.  So the argument is the encoded angle
   :math:`\theta_q = \sum_f w_{qf} u_f` of a *single* layer, never
   :math:`L \theta`.  A purity is therefore a property of *data plus encoding*
   -- weights and re-upload mask included -- not of the trained circuit.

2. **Exact g-purity** -- :func:`g_purity_exact` sums
   :math:`\langle\psi|B|\psi\rangle^2` over the DLA basis of the *actual*
   statevector the circuit prepares, parameters and all.  At ``n_qubits = 4``
   the basis has at most 255 words and the state 16 amplitudes, so it is cheap.
   It is the honest counterpart of the product-state form: the two agree exactly
   in the clustered limit (where every encoding rotation tends to the identity
   and :math:`P_{\mathfrak g}` is Ad-invariant under :math:`e^{\mathfrak g}`) and
   diverge at generic angles, because qml-essentials orders each layer
   *ansatz first, then encoding*.  Reporting both is what lets a claim say
   which object it is about.

3. **DLA pre-check** -- :func:`dla_check` records, per ansatz arm and before any
   training, the DLA dimension and the number of Z-only (diagonal) basis words.
   The latter is the floored/floor-free certificate: diagonal words have
   expectation 1 on the clustered-angle limit :math:`\theta \to 0`, so their
   count is the deterministic g-purity floor that makes an arm indifferent to
   the input distribution.  The arms and their bond structure are in
   :mod:`partiqledtr.ansaetze`; what the certificate says about each is the
   experiment, and it is measured rather than asserted.

Spectrum/FCC instrumentation (fourier-fingerprints) is not here yet.
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
    r"""Return the off-diagonal (``XY_Brickwork``) g-purity of an RY product state.

    The DLA of ``XY_Brickwork`` is spanned by
    :math:`\{X_j Z \cdots Z X_k,\, Y_j Z \cdots Z Y_k\}` at odd separation and
    :math:`\{X_j Z \cdots Z Y_k,\, Y_j Z \cdots Z X_k\}` at even separation.  On
    :math:`\bigotimes_k R_y(\theta_k)|0\rangle` only the ``X..X`` words survive
    (:math:`\langle Y_k \rangle = 0`), leaving

    .. math::
        P_{\mathfrak g} = \sum_{j < k,\ k - j \ \mathrm{odd}}
            \sin^2\theta_j \, \sin^2\theta_k \prod_{j < l < k} \cos^2\theta_l .

    There is no diagonal word, hence no floor: clustered angles
    (:math:`\theta \to 0`) drive this to zero as :math:`O(\theta^4)`.
    Evaluated by an O(n) two-parity recurrence.  Port of ``offdiag_closed_form``
    in ``reference/unflattening/unflattening/utils/purity.py``.

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
    r"""Return the g-purity of an RY product state, for any ansatz arm.

    On :math:`\bigotimes_q R_y(\theta_q)|0\rangle` a Pauli word has
    :math:`\langle X_q \rangle = \sin\theta_q`, :math:`\langle Y_q \rangle = 0`
    and :math:`\langle Z_q \rangle = \cos\theta_q`, so only the Y-free basis
    words survive and

    .. math::
        P_{\mathfrak g} = \sum_{B \ \mathrm{Y-free}}
            \prod_{q \in X(B)} \sin^2\theta_q \prod_{q \in Z(B)} \cos^2\theta_q .

    This is the same object the manuscript's closed forms describe -- it agrees
    with :func:`g_purity_offdiag` to float32 -- but it is read off the arm's own
    DLA basis instead of a hand-derived series, which is what lets a *new*
    ansatz be measured at all. Cost is ``O(|basis| * n)`` and it is jittable.

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
    r"""Return the exact g-purity of statevectors against a DLA basis.

    :math:`P_{\mathfrak g} = \sum_B \langle\psi|B|\psi\rangle^2`, evaluated on
    the state the circuit *actually* prepares rather than on the product state
    the closed forms describe.  Use it to check a closed-form series rather than
    to replace it: the closed form is the theory's object (the encoded angle
    distribution), this is the model's.

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
    r"""Describe the *shape* of an encoded angle distribution, per qubit.

    The g-purity says how trainable an encoded state is; it does not say what the
    distribution looks like, and two very different laws reach the same purity.
    ``P_{\mathfrak g}`` for the off-diagonal algebra is built from
    :math:`\sin^2\theta` factors, so it climbs both when the angles *spread*
    toward uniform -- the flattening the study looks for -- and when they *pin*
    near :math:`\pi/2`, which is the true maximum :math:`n - 1` and the
    configuration the unflattening manuscript notes destroys the input
    information. Telling those apart needs the distribution, not the purity.

    Reported **per qubit**, never pooled: sites peaking at different angles average
    into something that looks flat, which is the artefact the unflattening latent
    -drift memo warns about.

    Two cautions on reading ``tv_uniform``:

    * It has a **nonzero floor set by kinematics, not by training**. The
      ``(p_z, E)`` sites cannot leave ``(0, pi)`` and in practice sit inside about
      ``[pi/4, 3pi/4]`` (as ``E >= |p_z|``), so those qubits can never be uniform
      however the preconditioner moves them. Compare a run against the *raw* arm's value at the same
      site, not against zero.
    * It depends on :data:`TV_BINS` and on how many angles went in.

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
    """Price every feature encoding, crossed with every encoding-weight arm.

    This is the measurement behind the project's clearest empirical claim, and the
    reason it lives here rather than in a notebook: the encoding a decay-tree model
    picks decides whether its inputs land in the barren regime at all, and the
    unflattening theory prices that decision in a currency both papers share.
    Crossing it with the encoding-weight arms answers their central question
    *without training anything* -- spectral preconditioning is a property of data
    plus encoding, so if an exponential spectrum lifts a collapsed encoding off the
    floor, it shows up here.

    Three feature arms, all read off the same events:

    * ``pair_polar`` -- the quantum arm's map, polar angles of ``(px, py)`` and
      ``(pz, E)``;
    * ``direct`` -- the ``(theta, phi)`` direction angles of the ``"angles"``
      encoding, used as-is;
    * ``legacy`` -- partiqlegan's ``p * E * pi`` product, the clustered arm,
      and the one where the manuscript's jitter amplification has room to act.

    crossed with the six ``weights-reupload`` cells of
    :func:`partiqledtr.models.qfm.encoding_matrix`.

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
    r"""Return the Pauli-word generators of an ansatz layer's rotation gates.

    Walks the ansatz's own ``structure()`` and its ``Topology`` helpers, so the
    wire sets are the ones the circuit actually applies rather than an assumed
    pattern.  Single-qubit ``R\sigma`` contributes :math:`\sigma_q`, two-qubit
    ``R\sigma\sigma`` contributes :math:`\sigma_j \sigma_k` on each topology
    bond, and ``CRX(c, t)`` -- whose generator is
    :math:`|1\rangle\langle 1|_c \otimes X_t = (I - Z_c) X_t / 2` -- contributes
    the two words :math:`X_t` and :math:`Z_c X_t`.  Duplicates are dropped, which
    makes the ``CRX`` split exact for ``Circuit_19``: its bare :math:`X_t` are
    already generated by the ``RX`` block.

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
    r"""Return the DLA certificate of an ansatz arm, recorded before any training.

    Computes the Lie closure of :func:`ansatz_generators` and reports its
    dimension together with ``n_diag_words``, the number of Z-only basis words.
    Diagonal words have expectation 1 on :math:`|0\rangle^{\otimes n}`, so their
    count is the deterministic g-purity floor of the clustered-angle limit: 0
    certifies a floor-free arm on which the unflattening rescue is live, while
    :math:`n` (``Matchgate``) certifies the floored control arm.

    Measured at the constellation size ``n_qubits=4``::

        ansatz          dim_g  dim_su   ratio  n_diag_words  runtime
        XY_Brickwork       12     255  0.0471             0   0.6 ms
        Matchgate          28     255  0.1098             4   1.9 ms
        Circuit_19        255     255  1.0000            15   464 ms

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
    """Record every ansatz arm's certificate, prior scale and partition symmetry.

    The ansatz-arm table, as a run rather than as a number someone typed into a
    document: the DLA dimension and floor count that decide whether an arm is
    input-distribution sensitive, the uniform-prior mean its purities have to be
    read against, and whether its bond set survives the endpoint swap.

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
    """Characterise every encoding-weight cell, and price it on synthetic angle laws.

    Two things per encoding-weight cell, neither of which needs a dataset: what
    the cell *is* -- its weight matrix, per-feature spectrum, dissociation and
    endpoint symmetry -- and what it *does* to the g-purity of a uniform and of a
    clustered angle law.

    The synthetic prices are the arm's prediction, recorded before the real
    encodings are read so the measured table confirms rather than discovers.
    ``encoding_purity`` is the same question on real kinematics.

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
