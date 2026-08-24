r"""Theory instrumentation for the unflattening connection (ROADMAP phases 3-4).

Two things live here, both recorded/tracked around training rather than trained:

1. **g-purity closed forms** -- the observable of ROADMAP phase 4.  For a
   floor-free, polynomial-DLA ansatz the unflattening result (Theorem 1) states
   :math:`\mathrm{Var}_W[\langle Z_i \rangle] = P_{\mathfrak g}(\rho) / \dim
   \mathfrak g`, so the g-purity of the *encoded input state* alone decides
   trainability.  Our edge QFMs encode one feature per qubit through a single
   ``RY`` (DECISIONS.md D25), i.e. the state is the product state
   :math:`\bigotimes_k R_y(\theta_k)|0\rangle` for which the g-purity has an
   O(n) closed form per DLA.  Ported from
   ``reference/unflattening/unflattening/utils/purity.py``.

2. **DLA pre-check** -- :func:`dla_check` records, per ansatz arm and before any
   training, the DLA dimension and the number of Z-only (diagonal) basis words.
   The latter is the floored/floor-free certificate: diagonal words have
   expectation 1 on the clustered-angle limit :math:`\theta \to 0`, so their
   count is the deterministic g-purity floor that makes an arm indifferent to
   the input distribution.

Ansatz arm -> purity function (see :data:`G_PURITY_BY_ANSATZ`; this mapping *is*
the experiment):

- ``XY_Brickwork`` -> :func:`g_purity_offdiag`.  DLA
  :math:`\mathfrak{so}(n) \oplus \mathfrak{so}(n)`, no diagonal word, hence no
  floor.  The live arm: its purity collapses as :math:`O(\theta^4)` when the
  encoding angles cluster at zero -- the regime kinematic features land in --
  and a front end that spreads the angles rescues it.
- ``Matchgate`` -> :func:`g_purity_full`.  DLA :math:`\mathfrak{so}(2n)`, whose
  :math:`n` single-qubit :math:`Z_k` give a floor of :math:`n`, so clustered
  angles *maximise* the purity: the predicted indifference, and the control arm.
- ``Circuit_19`` -> :func:`g_purity_su`.  DLA is all of
  :math:`\mathfrak{su}(2^n)`, whose g-purity is :math:`2^n - 1` for *every* pure
  state -- input-independent, and with :math:`\dim \mathfrak g = 4^n - 1` the
  variance is the textbook :math:`1/(2^n + 1)` barren plateau, so no front end
  can help.

The phase-5 spectrum/FCC instrumentation (fourier-fingerprints) is not here yet.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from fluksio import Port, node
from qml_essentials.ansaetze import Ansaetze
from qml_essentials.operations import PauliWord

#: Ansatz arms of ROADMAP phase 3, in the order they are reported.
ANSAETZE = ("XY_Brickwork", "Matchgate", "Circuit_19")

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


def g_purity_full(theta: jax.Array) -> jax.Array:
    r"""Return the matchgate (:math:`\mathfrak{so}(2n)`) g-purity of an RY product state.

    The matchgate DLA basis is :math:`\{Z_k\}` plus
    :math:`\sigma_j (\prod_{j<l<k} Z_l) \sigma'_k` with
    :math:`\sigma, \sigma' \in \{X, Y\}` (``qml_essentials.algebra.matchgate_basis``).
    On :math:`\bigotimes_k R_y(\theta_k)|0\rangle` this gives

    .. math::
        P_{\mathfrak g} = \sum_k \cos^2\theta_k
            + \sum_{j<k} \sin^2\theta_j \, \sin^2\theta_k
              \prod_{j<l<k} \cos^2\theta_l ,

    evaluated by the O(n) recurrence of ``g_purity_closed_form`` in
    ``reference/unflattening/unflattening/utils/purity.py``.  The first sum is
    the deterministic floor contributed by the :math:`n` diagonal words: it
    equals :math:`n` exactly in the clustered limit :math:`\theta \to 0`, above
    the uniform-prior mean :math:`n - 1 + 2^{-n}`.  This is why the ``Matchgate``
    arm is predicted indifferent to the input distribution.

    Args:
        theta: Encoding angles of shape ``(..., n)``.

    Returns:
        The g-purity of shape ``(...)``.
    """
    c, s = jnp.cos(theta) ** 2, jnp.sin(theta) ** 2
    cross = jnp.zeros(theta.shape[:-1])
    w = jnp.zeros(theta.shape[:-1])  # sum over j < k of s_j prod_{j<l<k} c_l
    for k in range(theta.shape[-1]):
        cross = cross + s[..., k] * w
        w = c[..., k] * w + s[..., k]
    return c.sum(axis=-1) + cross


def g_purity_su(theta: jax.Array) -> jax.Array:
    r"""Return the :math:`\mathfrak{su}(2^n)` (``Circuit_19``) g-purity, :math:`2^n - 1`.

    Summing :math:`\langle P \rangle^2` over all :math:`4^n - 1` non-identity
    Pauli words gives :math:`2^n \mathrm{tr}(\rho^2) - 1`, which is
    :math:`2^n - 1` for any pure state -- the encoding cannot change it.  Held
    only where :math:`\dim \mathfrak g = 4^n - 1`, which :func:`dla_check`
    confirms for ``Circuit_19`` at :math:`n = 2, 3, 4`.

    Args:
        theta: Encoding angles of shape ``(..., n)``; only the shape is used.

    Returns:
        The constant g-purity, broadcast to shape ``(...)``.
    """
    return jnp.full(theta.shape[:-1], 2.0 ** theta.shape[-1] - 1.0)


#: Purity observable of ROADMAP phase 4, per ansatz arm (see the module docstring).
G_PURITY_BY_ANSATZ: dict[str, Callable[[jax.Array], jax.Array]] = {
    "XY_Brickwork": g_purity_offdiag,
    "Matchgate": g_purity_full,
    "Circuit_19": g_purity_su,
}


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
        ansatz: One of :data:`ANSAETZE`.
        n_qubits: Number of qubits, at least 2.

    Returns:
        The deduplicated generator Pauli strings (qubit 0 leftmost).

    Raises:
        ValueError: If ``ansatz`` is unknown or ``n_qubits < 2``.
        NotImplementedError: If the ansatz contains a rotation gate whose
            generator is not covered here.
    """
    if ansatz not in ANSAETZE:
        raise ValueError(f"unknown ansatz {ansatz!r}, expected one of {ANSAETZE}")
    if n_qubits < 2:
        raise ValueError(f"n_qubits must be at least 2, got {n_qubits}")

    words: list[str] = []
    for block in getattr(Ansaetze, ansatz).structure():
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


def _lie_closure_capped(generators: list[str], max_dim: int) -> tuple[list[str], bool]:
    """Return the Pauli-word Lie closure, stopped at ``max_dim`` words.

    Args:
        generators: Hermitian generator Pauli strings.
        max_dim: Maximum number of basis words to collect.

    Returns:
        The basis Pauli strings and whether the cap stopped the search (in which
        case the basis is a subset of the true closure).
    """
    # ponytail: duplicates qml_essentials.algebra.lie_closure_paulis purely to add
    # the cap -- upstream grows the closure unconditionally, which is O(4**n) words
    # and hangs well before n = 8. Upgrade path: a max_dim kwarg upstream, then
    # delete this and call lie_closure_paulis.
    n = len(generators[0])
    basis = [PauliWord.from_pauli_string(s, list(range(n)), n) for s in generators]
    seen = {word.to_pauli_string() for word in basis}
    frontier = list(basis)
    while frontier:
        new: list[PauliWord] = []
        pool = list(basis)
        for a in frontier:
            for b in pool:
                if a.commutes_with(b):
                    continue
                product = a.compose(b)
                string = product.to_pauli_string()
                if string in seen:
                    continue
                seen.add(string)
                basis.append(product)
                new.append(product)
                if len(basis) >= max_dim:
                    return [w.to_pauli_string() for w in basis], True
        frontier = new
    return [w.to_pauli_string() for w in basis], False


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

    Measured at the constellation size ``n_qubits=4`` (DECISIONS.md D24)::

        ansatz          dim_g  dim_su   ratio  n_diag_words  runtime
        XY_Brickwork       12     255  0.0471             0   0.6 ms
        Matchgate          28     255  0.1098             4   1.9 ms
        Circuit_19        255     255  1.0000            15   464 ms

    Args:
        ansatz: One of :data:`ANSAETZE`.
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
    words, capped = _lie_closure_capped(generators, max_dim)
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
    requires=[Port("ansatz", "str")],
    provides=[Port("dla_report", "json")],
)
def dla_report(*, ansatz: str = "XY_Brickwork", n_qubits: int = 4, max_dim: int = 2000) -> dict:
    """Record an ansatz arm's dynamical Lie algebra before any training.

    The ROADMAP asks for the DLA and floor count of each arm to be recorded
    *before* training. Wiring this node upstream of the fit makes that a property
    of the flow rather than of anyone's discipline.

    Args:
        ansatz: Ansatz arm, one of :data:`ANSAETZE`.
        n_qubits: Qubits per edge QFM.
        max_dim: Cap on the Lie closure (``DECISIONS.md`` D54).

    Returns:
        The certificate of :func:`dla_check`, under the ``dla_report`` port.
    """
    return {"dla_report": dla_check(ansatz=ansatz, n_qubits=n_qubits, max_dim=max_dim)}
