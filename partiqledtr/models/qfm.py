"""QFM constellation: shared-weight quantum Fourier models as the edge function.

ROADMAP phase 3. Many small quantum Fourier models (4 qubits each) sit inside a
message-passing network, one evaluation per directed edge, sharing one parameter
set across every edge. Sharing is what makes the model permutation-equivariant;
staying small is what keeps the analytic simulation cheap and the per-QFM spectrum
tractable.

Every classical part is deliberately particle-local -- an elementwise preconditioner, a
parameter-free masked mean, a per-node linear map -- so cross-particle structure
can only come from the quantum edge function (``DECISIONS.md`` D27).

Encoding contract, verified by measurement rather than assumed (``DECISIONS.md``
D25, D55, D96): re-uploaded ``RY`` gates on one wire add, so a layer rotates qubit
``q`` by ``theta_q = sum_f W[q, f] u_f`` for the effective weight matrix ``W`` of
:func:`encoding_matrix`, and with zero ansatz parameters the bond readout is
exactly ``sin(n_layers * theta_j) sin(n_layers * theta_k)`` (``<Y_q> = 0`` on the
RY product state, so the ``YY`` half vanishes there). The phase-3/4 arm has ``W = I`` -- feature
``f`` on qubit ``f`` alone, weight one. ROADMAP phase 4b arm B varies ``W`` and
nothing else: exponential weights and a widened (``cyclic``) mask, which is what
the unflattening manuscript's spectral preconditioning needs and what enlarges the
per-feature spectrum from 5 to 17 frequencies.

The state the *product-state* g-purity describes is ``prod_q RY(theta_q)|0>``
entering the first trainable block -- the encoded angle distribution, not the
trained circuit, and not ``L theta`` (``DECISIONS.md`` D78).
:meth:`QFMConstellation.g_purity` reports that; :meth:`QFMConstellation.g_purity_exact`
reports the same quantity for the state the circuit actually prepares. Both are
streamed, because they answer different questions and only agree in the
clustered limit.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jaqsi
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import ArrayLike
from qml_essentials.ansaetze import Encoding
from qml_essentials.model import Model

from partiqledtr.analysis import angle_stats, dla_basis, g_purity_exact, product_state_purity
from partiqledtr.ansaetze import ANSAETZE, bonds, circuit
from partiqledtr.models.gnn import _edge_mask, edge2node, node2edge

__all__ = [
    "ANGLE_MAPS",
    "ANGLE_WIDTHS",
    "ANSAETZE",
    "ENC_REUPLOAD",
    "ENC_WEIGHTS",
    "NODE_UPDATES",
    "N_QUBITS",
    "NodeMLP",
    "QFMConstellation",
    "encoding_matrix",
    "encoding_spectrum",
    "legacy_angles",
    "make_qfm",
    "pair_polar",
    "pair_polar_boost",
    "pair_polar_mass",
    "pair_polar_theta",
    "readout_bonds",
    "readout_observables",
    "reupload_mask",
]

#: Qubits per edge QFM: two pair-polar angles from each of the two endpoints (D24).
N_QUBITS = 4
N_ANGLES = N_QUBITS // 2

#: Re-upload masks: which features reach which qubit (D95).
ENC_REUPLOAD = ("diagonal", "cyclic")


def _strategy_weights(strategy: str, n_qubits: int) -> np.ndarray:
    """Per-qubit weights of a qml-essentials encoding strategy."""
    return np.asarray(Encoding(strategy, ["RY"] * n_qubits).get_weights(n_qubits), dtype=float)


#: Per-qubit encoding weights of ROADMAP phase 4b arm B, as functions of the qubit
#: count.  The first three are qml-essentials' own strategies (``base ** q``);
#: ``ternary_pair`` repeats the exponent per *particle* (``n // 2`` qubits each),
#: which is the only family that is dissociated **and** invariant under the
#: endpoint swap (D100).
ENC_WEIGHTS: dict[str, Callable[[int], np.ndarray]] = {
    name: functools.partial(_strategy_weights, name) for name in ("hamming", "binary", "ternary")
} | {"ternary_pair": lambda n: 3.0 ** (np.arange(n) % (n // 2))}


def reupload_mask(reupload: str, n_layers: int, n_qubits: int = N_QUBITS) -> np.ndarray:
    """Return the ``(n_layers, n_qubits, n_features)`` re-upload mask of an arm.

    ``"diagonal"`` sends feature ``f`` to qubit ``f`` alone, which is the phase-3/4
    encoding.  ``"cyclic"`` also sends it to qubit ``f - 1``, so every feature
    reaches two qubits -- the widening ROADMAP phase 4b arm B asks for, and what
    turns per-qubit weights into a per-*feature* weight vector.

    Args:
        reupload: One of :data:`ENC_REUPLOAD`.
        n_layers: Data-reuploading depth; every layer carries the same mask.
        n_qubits: Number of qubits, which here also fixes the feature count.

    Returns:
        The boolean mask.

    Raises:
        ValueError: If ``reupload`` is unknown.
    """
    if reupload not in ENC_REUPLOAD:
        raise ValueError(f"reupload must be one of {list(ENC_REUPLOAD)}, got {reupload!r}")
    mask = np.zeros((n_layers, n_qubits, n_qubits), dtype=bool)
    for qubit in range(n_qubits):
        mask[:, qubit, qubit] = True
        if reupload == "cyclic":
            mask[:, qubit, (qubit + 1) % n_qubits] = True
    return mask


def encoding_matrix(
    weights: str = "hamming", reupload: str = "diagonal", n_qubits: int = N_QUBITS
) -> np.ndarray:
    r"""Return the effective encoding weight matrix ``W`` of an arm.

    One layer of the encoding rotates qubit ``q`` by
    :math:`\theta_q = \sum_f W_{qf} u_f`, since re-uploaded ``RY`` gates on the
    same wire add.  ``W`` is the re-upload mask scaled row-wise by the strategy's
    per-qubit weight, and it is the whole of what arm B varies: the state stays an
    ``RY`` product state, so every purity closed form applies unchanged with
    :math:`\theta` in place of :math:`u` (``DECISIONS.md`` D96).

    At ``n_qubits = 4``, dissociation -- no
    :math:`\epsilon \in \{-1,0,1\}^4 \setminus 0` with :math:`W^\top \epsilon = 0`
    -- needs both a widened mask and unequal weights, so ``hamming-cyclic`` mixes
    the features without preconditioning them and is the control that says so.
    Among the dissociated cells only ``ternary_pair`` is also invariant under the
    endpoint swap, which is what lets it compose with a partition-respecting
    ansatz (``DECISIONS.md`` D100).

    Args:
        weights: One of :data:`ENC_WEIGHTS`.
        reupload: One of :data:`ENC_REUPLOAD`.
        n_qubits: Number of qubits, which here also fixes the feature count.

    Returns:
        ``(n_qubits, n_features)`` weights.

    Raises:
        ValueError: If ``weights`` or ``reupload`` is unknown.
    """
    if weights not in ENC_WEIGHTS:
        raise ValueError(f"weights must be one of {list(ENC_WEIGHTS)}, got {weights!r}")
    return reupload_mask(reupload, 1, n_qubits)[0] * ENC_WEIGHTS[weights](n_qubits)[:, None]


def encoding_spectrum(matrix: np.ndarray, feature: int, n_layers: int) -> np.ndarray:
    r"""Return the frequencies one feature reaches through an encoding matrix.

    Feature ``f`` enters qubit ``q`` with weight ``W[q, f]``, once per layer, so
    its reachable comb is the Minkowski sum over qubits of
    ``{k W[q,f] : |k| <= n_layers}``.  ``Encoding.get_spectrum`` computes the same
    thing from a strategy plus a mask; this reads it off the weight matrix, which
    is what the circuit actually applies once the weights live in ``enc_params``
    (``DECISIONS.md`` D100).

    Args:
        matrix: ``(n_qubits, n_features)`` weights from :func:`encoding_matrix`.
        feature: Which feature's spectrum to return.
        n_layers: Data-reuploading depth.

    Returns:
        The sorted reachable frequencies.
    """
    reach = {0.0}
    for weight in matrix[:, feature]:
        if weight:
            reach = {a + k * weight for a in reach for k in range(-n_layers, n_layers + 1)}
    return np.array(sorted(reach))


def readout_bonds(ansatz: str, n_qubits: int = N_QUBITS) -> tuple[tuple[int, int], ...]:
    """The bonds an arm's readout measures, in a frozen order.

    The arm's own coupling graph, read off its circuit structure via
    :func:`partiqledtr.ansaetze.bonds` and sorted lexicographically -- the same
    convention :func:`partiqledtr.analysis.arm_report` records.

    Args:
        ansatz: One of :data:`ANSAETZE`.
        n_qubits: Number of qubits.

    Returns:
        Sorted ``(j, k)`` bonds with ``j < k``.
    """
    return tuple(sorted((min(bond), max(bond)) for bond in bonds(ansatz, n_qubits)))


def readout_observables(ansatz: str, n_qubits: int = N_QUBITS) -> list[jaqsi.Operation]:
    """The in-algebra readout: ``X_j X_k`` and ``Y_j Y_k`` per coupling bond.

    The unflattening variance law ``Var = P_g(rho) P_g(O) / dim g`` needs the
    observable inside the arm's dynamical Lie algebra; single-qubit ``Z`` is in
    no XY arm's algebra, so the previous per-qubit readout had ``P_g(O) = 0``
    and no channel from the encoded state to the loss (``DECISIONS.md`` D107).
    The bond words are the arm's own generators, so they are in-algebra by
    construction -- asserted per arm in the tests. The two strings of a bond are
    interleaved ``[XX_b, YY_b, ...]`` and summed to ``<XX_b> + <YY_b>`` by the
    consumer, the manuscript's own readout convention (``exp_latent_drift``).

    Args:
        ansatz: One of :data:`ANSAETZE`.
        n_qubits: Number of qubits.

    Returns:
        ``2 * n_bonds`` Pauli-string observables.
    """
    observables = []
    for j, k in readout_bonds(ansatz, n_qubits):
        observables.append(jaqsi.PauliX(wires=j) @ jaqsi.PauliX(wires=k))
        observables.append(jaqsi.PauliY(wires=j) @ jaqsi.PauliY(wires=k))
    return observables


def make_qfm(
    ansatz: str,
    *,
    n_layers: int = 2,
    seed: int = 0,
    enc_weights: str = "hamming",
    enc_reupload: str = "diagonal",
    n_qubits: int = N_QUBITS,
) -> Model:
    """Build one edge QFM.

    Args:
        ansatz: One of :data:`ANSAETZE`.
        n_layers: Data-reuploading depth. The re-uploaded rotations add, so this
            also multiplies the model's frequency support.
        seed: Seed for the model's own parameter initialisation. The constellation
            copies those values into its NNX state and never uses the model's copy
            again.
        enc_weights: Encoding weight strategy, one of :data:`ENC_WEIGHTS`.
        enc_reupload: Re-upload mask, one of :data:`ENC_REUPLOAD`.
        n_qubits: Qubits per edge QFM, which here also fixes the feature count.

    Returns:
        A qml-essentials :class:`~qml_essentials.model.Model` with analytic
        expectation values and the arm's in-algebra bond observables of
        :func:`readout_observables`.

    Raises:
        ValueError: If the ansatz, weights or mask is unknown, or ``n_layers`` is
            not positive.
    """
    pqc = circuit(ansatz)
    if n_layers < 1:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    if enc_weights not in ENC_WEIGHTS:
        raise ValueError(f"enc_weights must be one of {list(ENC_WEIGHTS)}, got {enc_weights!r}")

    model = Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type=pqc,
        encoding=["RY"] * n_qubits,
        data_reupload=reupload_mask(enc_reupload, n_layers, n_qubits),
        observables=readout_observables(ansatz, n_qubits),
        shots=None,
        random_seed=seed,
    )
    # The weights ride in `enc_params` rather than in an `Encoding` strategy, so
    # one matrix is the whole of what arm B varies and a weighting qml-essentials
    # does not ship -- `ternary_pair` -- needs no new strategy (D100). They stay
    # frozen: trainable frequencies are a separate ROADMAP axis.
    matrix = encoding_matrix(enc_weights, enc_reupload, n_qubits)
    model.enc_params = jnp.broadcast_to(jnp.asarray(matrix), (n_layers, n_qubits, n_qubits))
    return model


def pair_polar(p4: ArrayLike) -> jax.Array:
    """Map four-vectors to the two pair-polar angles the QFMs encode.

    Coordinate pairs become polar angles, ``(px, py) -> phi`` and ``(pz, E) ->
    alpha``. This is the unflattening paper's ``polar_angles`` map, which is what
    lets the phase-4 whitening arm be that paper's construction verbatim
    (``DECISIONS.md`` D24). The pair radii are dropped, so momentum and energy
    magnitudes do not enter the quantum path.

    The two angles do **not** cover the circle equally, and the asymmetry is
    physical rather than incidental (``DECISIONS.md`` D79):

    * ``phi`` is a genuine azimuth and covers ``[0, 2 pi)``;
    * ``alpha`` cannot leave ``(0, pi)`` at all, because ``E > 0`` puts the pair
      in the upper half-plane, and ``E >= |p| >= |pz|`` confines it further to
      about ``[pi/4, 3 pi/4]`` -- a quarter of the circle, widened only slightly
      by momentum and energy carrying different normalisation scales. The
      ``jnp.mod`` is therefore a no-op on the ``alpha`` components.

    That concentration around ``pi/2`` is a kinematic bound, not a softness
    effect: soft and hard particles sit at the same place. It matters for the
    phase-4 reading, because ``pi/2`` is the *favourable* RY point, so this
    encoding starts well away from the clustered regime -- see
    :func:`legacy_angles` for the arm that does cluster.

    Args:
        p4: ``(..., 4)`` four-vectors laid out ``[px, py, pz, E]``; anything
            :func:`jax.numpy.asarray` accepts, since callers hand over numpy too.

    Returns:
        ``(..., 2)`` angles ``(phi, alpha)`` in ``[0, 2 pi)``.

    Raises:
        ValueError: If the last axis is not 4.
    """
    p4 = jnp.asarray(p4)
    if p4.shape[-1] != 4:
        raise ValueError(f"expected four-vectors on the last axis, got {p4.shape[-1]}")
    return jnp.mod(jnp.arctan2(p4[..., 1::2], p4[..., 0::2]), 2 * jnp.pi)


def legacy_angles(p4: ArrayLike) -> jax.Array:
    """Map four-vectors to partiqlegan's product encoding angles.

    The prior work encoded each particle on one qubit as ``RX(px E pi)``,
    ``RY(py E pi)``, ``RZ(pz E pi)`` with momenta scaled into ``[-1, 1]`` and
    energy into ``[0, 1]``. Our edge QFM spends two qubits per particle, so this
    keeps two of those three angles -- the ``RX`` and ``RZ`` ones, ``(px E pi,
    pz E pi)`` -- and drops the ``py`` one.

    It exists as a deliberately *clustered* input arm (``DECISIONS.md`` D80).
    Both factors live in the unit interval, so their product concentrates near
    zero, which is the collapsed point of the RY encoding and the regime where
    the unflattening rescue prediction is falsifiable. Feed it the ``"legacy"``
    encoding, whose normalisation is max-based precisely so the ``[-1, 1]``
    premise of that argument holds.

    Args:
        p4: ``(..., 4)`` four-vectors laid out ``[px, py, pz, E]``, normalised by
            :data:`partiqledtr.data.features.LEGACY_ENCODING`.

    Returns:
        ``(..., 2)`` angles ``(px E pi, pz E pi)``.

    Raises:
        ValueError: If the last axis is not 4.
    """
    p4 = jnp.asarray(p4)
    if p4.shape[-1] != 4:
        raise ValueError(f"expected four-vectors on the last axis, got {p4.shape[-1]}")
    energy = p4[..., 3]
    return jnp.stack([p4[..., 0] * energy, p4[..., 2] * energy], axis=-1) * jnp.pi


def _third_angle(p4: ArrayLike, third: Callable[[jax.Array], jax.Array]) -> jax.Array:
    """The pair-polar angles plus one derived third angle per particle."""
    p4 = jnp.asarray(p4)
    return jnp.concatenate([pair_polar(p4), third(p4)[..., None]], axis=-1)


def _mass(p4: jax.Array) -> jax.Array:
    """A mass proxy from a (normalised) four-vector.

    ``sqrt(max(E^2 - |p|^2, 0))``. The features carry separate momentum and
    energy scales (``data/features.py``), so on normalised inputs this is a
    *deformed* invariant -- a fixed quadratic form of the features, not the rest
    mass in physical units. The same caveat already widens ``alpha``; what
    decides whether a chart is usable is its induced angle law, priced by the
    phase-6 encoding report before any training (ROADMAP phase 6).
    """
    return jnp.sqrt(jnp.clip(p4[..., 3] ** 2 - jnp.sum(p4[..., :3] ** 2, axis=-1), min=0.0))


def pair_polar_boost(p4: ArrayLike) -> jax.Array:
    """Pair-polar plus the boost chart ``atan2(|p|, m)`` -- three angles per particle.

    The third angle is a velocity measure (``gamma beta`` against 1): a
    relativistic particle sits near ``pi/2``, the favourable RY point, so the
    chart is expected to land in the favourable band the way ``alpha`` does
    (``RESEARCH.md`` §1). One of the three ROADMAP phase-6 candidate charts.
    """
    p4 = jnp.asarray(p4)
    return _third_angle(p4, lambda v: jnp.arctan2(jnp.linalg.norm(v[..., :3], axis=-1), _mass(v)))


def pair_polar_mass(p4: ArrayLike) -> jax.Array:
    """Pair-polar plus the inverse-boost chart ``atan2(m, E)`` -- three angles.

    The third angle is ``1/gamma``-like: a relativistic particle sits near zero,
    the collapsed RY point, so this candidate is expected to cluster. Kept as a
    candidate precisely so the encoding report decides rather than intuition.
    """
    return _third_angle(jnp.asarray(p4), lambda v: jnp.arctan2(_mass(v), v[..., 3]))


def pair_polar_theta(p4: ArrayLike) -> jax.Array:
    """Pair-polar plus the polar angle ``atan2(p_T, p_z)`` -- three angles.

    The third angle is the momentum direction's polar angle in ``(0, pi)``,
    frame-dependent but broadly distributed. One of the three phase-6 candidates.
    """
    return _third_angle(
        jnp.asarray(p4),
        lambda v: jnp.arctan2(jnp.linalg.norm(v[..., :2], axis=-1), v[..., 2]),
    )


#: Four-vectors to the angles each particle contributes to an edge QFM.
#: ``"legacy"`` is the clustered control arm and needs the ``"legacy"`` encoding.
#: The ``pair_polar_*`` charts are the phase-6 three-angle candidates for the
#: ``n = 6`` constellation, priced against each other by the s4 encoding report.
ANGLE_MAPS = {
    "pair_polar": pair_polar,
    "legacy": legacy_angles,
    "pair_polar_boost": pair_polar_boost,
    "pair_polar_mass": pair_polar_mass,
    "pair_polar_theta": pair_polar_theta,
}

#: Angles each map yields per particle; the constellation checks
#: ``n_qubits == 2 * ANGLE_WIDTHS[angle_map]`` so a map/register mismatch fails
#: at construction rather than as a reshape error inside a jitted step.
ANGLE_WIDTHS = {
    "pair_polar": 2,
    "legacy": 2,
    "pair_polar_boost": 3,
    "pair_polar_mass": 3,
    "pair_polar_theta": 3,
}


#: Node-update variants of the trig-interface arm: the default linear map, the
#: sine-activated SIREN block, and its matched-parameter ELU control.
NODE_UPDATES = ("linear", "siren", "elu")


class NodeMLP(nnx.Module):
    """Two-layer node update with a *linear* output, sine- or ELU-activated.

    The trig-interface arm: the constellation's node update is otherwise a bare
    linear map, so the whole quantum arm is trig-polynomial -> linear ->
    trig-polynomial -> linear, with no classical nonlinear capacity at all.
    ``"siren"`` applies ``sin(omega_0 (W x + b))`` with the SIREN first-layer
    initialisation ``W ~ U(-1/n_in, 1/n_in)`` and ``omega_0 = 30`` (Sitzmann et
    al., arXiv:2006.09661) -- the scale discipline that keeps post-sine
    activations distributed instead of collapsing ``sin`` to its linear regime.
    ``"elu"`` is the matched-parameter control that separates "a nonlinearity
    pays" from "the trigonometric one pays". The output layer stays linear in
    both, because the consumer re-encodes it as angles.

    Args:
        n_in: Size of the trailing input axis.
        n_hidden: Width of the hidden layer.
        n_out: Size of the trailing output axis.
        activation: ``"siren"`` or ``"elu"``.
        rngs: Rng container used for parameter initialisation.

    Raises:
        ValueError: If ``activation`` is neither variant.
    """

    #: SIREN's first-layer frequency scale, from the paper's recipe.
    OMEGA_0 = 30.0

    def __init__(
        self, n_in: int, n_hidden: int, n_out: int, *, activation: str, rngs: nnx.Rngs
    ) -> None:
        if activation not in ("siren", "elu"):
            raise ValueError(f"activation must be 'siren' or 'elu', got {activation!r}")
        self.activation = activation
        kwargs: dict[str, Any] = {}
        if activation == "siren":
            bound = 1.0 / n_in
            kwargs["kernel_init"] = lambda key, shape, dtype: jax.random.uniform(
                key, shape, dtype, -bound, bound
            )
        self.fc1 = nnx.Linear(n_in, n_hidden, rngs=rngs, **kwargs)
        self.fc2 = nnx.Linear(n_hidden, n_out, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the block to the trailing axis."""
        h = self.fc1(x)
        h = jnp.sin(self.OMEGA_0 * h) if self.activation == "siren" else nnx.elu(h)
        return self.fc2(h)


class QFMConstellation(nnx.Module):
    """Two message-passing blocks whose edge function is a shared-weight QFM.

    Shapes, with ``B`` events, ``L`` padded particles, ``C`` classes, ``nb`` the
    arm's bond count (:func:`readout_bonds`), ``A = n_qubits // 2`` the angles
    per particle (2 at the phase-4b size, 3 for the phase-6 arms) and ``K``
    the channel count (``n_channels``, default 1 -- D108's node-state widening:
    ``K`` independently initialised QFMs per block, so the inter-block node state
    is ``A * K`` numbers rather than ``A``)::

        p4    (B, L, 4)        four-vectors
        ang   (B, L, A)        angle-map output, optionally whitened
        a     (B, L, A)        preconditioner (identity or elementwise residual MLP)
        u1    (B, L, L, 2A)    concat(a_i, a_j) -> folded to (B*L*L, 2A) per channel
        e1    (B, L, L, K*nb)  <XX_b> + <YY_b> per coupling bond and channel (D107)
        m     (B, L, K*nb)     masked mean over real neighbours
        h     (B, L, A*K)      [a ; m] @ w_node, channel c's angles at h[..., A*c:A*(c+1)]
        e2    (B, L, L, K*nb)  second QFM block, its own parameters per channel
        out   (B, L, L, C)     symmetrised linear readout

    The QFM parameters live here as :class:`flax.nnx.Param` leaves and reach the
    circuit through :meth:`~qml_essentials.model.Model.apply`, the functional call
    path that writes no model state -- so the whole forward pass is safe under an
    outer ``jax.jit``.

    Only the ansatz parameters train. The encoding weights are fixed by the arm
    (``enc_weights``, ``enc_reupload``), i.e. this is a fixed-frequency model whose
    spectrum is set by the encoding alone; making them trainable is a separate
    ROADMAP axis and a separate hazard (Fourier locking), so it is deliberately not
    folded in here.

    Args:
        n_classes: Number of LCAG classes ``C``.
        ansatz: Ansatz arm, one of :data:`ANSAETZE` or :data:`ANSAETZE_N6`.
        n_layers: Data-reuploading depth of each QFM.
        n_channels: Independently initialised QFMs per block (D108). Widens the
            inter-block node state to ``n_angles * n_channels`` numbers; the
            default 1 is the original architecture.
        n_qubits: Qubits per edge QFM; must equal twice the angle map's width.
            The default 4 is the phase-4b constellation (D24), 6 the phase-6 one.
        angle_map: Key into :data:`ANGLE_MAPS`: ``"pair_polar"`` for the polar map
            of ``DECISIONS.md`` D24, ``"legacy"`` for the clustered control arm,
            which expects the ``"legacy"`` encoding, or a three-angle
            ``pair_polar_*`` chart for the phase-6 register.
        enc_weights: Encoding weight strategy, one of :data:`ENC_WEIGHTS`.
        enc_reupload: Re-upload mask, one of :data:`ENC_REUPLOAD`. Crossed with
            ``enc_weights`` these are ROADMAP phase 4b arm B.
        node_update: Node-update variant, one of :data:`NODE_UPDATES`. The
            default ``"linear"`` is the original architecture; ``"siren"`` and
            ``"elu"`` are the trig-interface arm and its control
            (:class:`NodeMLP`).
        node_hidden: Hidden width of the non-linear node updates; ignored by
            ``"linear"``.
        node_omega: Scale applied to the node update's output before block 2
            re-encodes it as angles. The re-encoding boundary is a sine of a
            linear map, and nothing else sets its frequency scale: at the
            default init the hidden values start small, so block 2's encoding
            starts clustered near zero -- the collapsed regime on a floor-free
            arm. ``1.0`` is the original behaviour; SIREN's ``omega_0`` is the
            same lever one layer earlier.
        preconditioner: Optional elementwise preconditioner ``(..., 2) -> (..., 2)`` applied to
            the angles; ``None`` feeds them raw.
        whitening: Optional ``(4, 4)`` rotation applied to the four-vectors before
            the polar map -- the phase-4 fixed-whitening arm. Accepts anything
            :func:`jax.numpy.asarray` takes, so a checkpoint can carry it as a
            nested list (``DECISIONS.md`` D81).
        seed: Seed for the quantum parameter initialisation.
        rngs: Rng container for the classical parameters.

    Raises:
        ValueError: If ``n_classes < 2``, the ansatz or angle map is unknown, or
            ``whitening`` is not ``(4, 4)``.
    """

    @staticmethod
    def preconditioner_features(n_qubits: int = N_QUBITS) -> int:
        """Feature width of an attached preconditioner: the per-particle angles.

        A preconditioner here sees the angle-map output, not the raw four-vectors,
        so it is built for ``n_qubits // 2`` features rather than ``F``
        (:func:`partiqledtr.train.build_model` calls this before construction).
        """
        return n_qubits // 2

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        ansatz: str = "XY_Brickwork",
        n_layers: int = 2,
        n_channels: int = 1,
        n_qubits: int = N_QUBITS,
        angle_map: str = "pair_polar",
        enc_weights: str = "hamming",
        enc_reupload: str = "diagonal",
        node_update: str = "linear",
        node_hidden: int = 32,
        node_omega: float = 1.0,
        preconditioner: nnx.Module | None = None,
        whitening: Any = None,
        seed: int = 0,
        rngs: nnx.Rngs,
        dim: int = 0,
    ) -> None:
        del dim  # the quantum path's width is n_qubits, set by the arm, not by `dim`
        if n_features != 4:
            raise ValueError(
                f"the QFM constellation consumes four-vectors, so it needs the "
                f'"cartesian" encoding (4 features), got {n_features}. The "angles" '
                f"encoding drops |p|, which is not recoverable for massive final-state "
                f"particles."
            )
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if n_channels < 1:
            raise ValueError(f"n_channels must be positive, got {n_channels}")
        if angle_map not in ANGLE_MAPS:
            raise ValueError(f"angle_map must be one of {sorted(ANGLE_MAPS)}, got {angle_map!r}")
        if node_update not in NODE_UPDATES:
            raise ValueError(f"node_update must be one of {NODE_UPDATES}, got {node_update!r}")
        if node_omega <= 0.0:
            raise ValueError(f"node_omega must be positive, got {node_omega}")
        if n_qubits != 2 * ANGLE_WIDTHS[angle_map]:
            raise ValueError(
                f"angle_map {angle_map!r} yields {ANGLE_WIDTHS[angle_map]} angles per "
                f"particle, so it needs n_qubits={2 * ANGLE_WIDTHS[angle_map]}, "
                f"got {n_qubits}"
            )
        circuit(ansatz)  # resolve early, so an unknown arm fails here and not mid-build
        # Converted before the shape check so a checkpoint may carry the rotation
        # as a nested list rather than an array (D81).
        rotation = None if whitening is None else jnp.asarray(whitening, dtype=jnp.float32)
        if rotation is not None and rotation.shape != (4, 4):
            raise ValueError(f"whitening must be (4, 4), got {rotation.shape}")

        self.ansatz = ansatz
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.n_angles = n_qubits // 2
        self.angle_map = angle_map
        self.enc_weights = enc_weights
        self.enc_reupload = enc_reupload
        self.preconditioner = preconditioner
        self.whitening = rotation
        # Not an nnx.Param: the encoding weights are fixed by the arm, and making
        # them trainable is a separate ROADMAP axis with its own failure mode.
        self.enc_matrix = jnp.asarray(encoding_matrix(enc_weights, enc_reupload, n_qubits))

        # Separate parameters per block: the two blocks do different jobs, and
        # equivariance only needs sharing across *edges* (D28). Each block holds
        # `n_channels` independent draws stacked on a leading axis; channel c of
        # block b seeds `seed + 2c + b`, so `n_channels = 1` is exactly the old
        # `(seed, seed + 1)` pair (D108).
        blocks = [
            [
                make_qfm(
                    ansatz,
                    seed=seed + 2 * c + b,
                    n_layers=n_layers,
                    enc_weights=enc_weights,
                    enc_reupload=enc_reupload,
                    n_qubits=n_qubits,
                )
                for c in range(n_channels)
            ]
            for b in (0, 1)
        ]
        stack = [jnp.stack([jnp.asarray(np.asarray(m.params)) for m in block]) for block in blocks]
        self.qfm1_params = nnx.Param(stack[0])
        self.qfm2_params = nnx.Param(stack[1])
        self._qfm1, self._qfm2 = blocks[0][0], blocks[1][0]

        self.n_channels = n_channels
        self.n_bonds = len(readout_bonds(ansatz, n_qubits))
        self.node_update = node_update
        self.node_omega = float(node_omega)
        node_in = self.n_angles + n_channels * self.n_bonds
        node_out = n_channels * self.n_angles
        # The default keeps the exact construction (and rng draw order) of the
        # original architecture, so `node_update="linear"` stays bit-identical.
        self.w_node = (
            nnx.Linear(node_in, node_out, rngs=rngs)
            if node_update == "linear"
            else NodeMLP(node_in, node_hidden, node_out, activation=node_update, rngs=rngs)
        )
        self.head = nnx.Linear(n_channels * self.n_bonds, n_classes, rngs=rngs)

    def _edges(self, qfm: Model, params: jax.Array, angles: jax.Array) -> jax.Array:
        """Evaluate one QFM on every directed edge of every event.

        Uses the functional :meth:`~qml_essentials.model.Model.apply`, which writes
        no model state and keeps every batch axis unsqueezed, so the call is safe
        inside an outer ``jax.jit`` and its shape does not depend on the batch
        sizes. The guard pins that rank: a silent change to it would otherwise
        reshape into the wrong edge grid rather than fail.
        """
        batch, n_leaves = angles.shape[0], angles.shape[1]
        flat = node2edge(angles).reshape(-1, self.n_qubits)
        out = qfm.apply(params=params, inputs=flat)
        expected = (flat.shape[0], 1, 1, 1, 2 * self.n_bonds)
        if out.shape != expected:
            raise ValueError(f"QFM returned {out.shape}, expected {expected}")
        # <XX_b> + <YY_b> per bond: the summed pair is the arm's own generator,
        # the in-algebra observable the variance law prices (D107).
        return out.reshape(batch, n_leaves, n_leaves, self.n_bonds, 2).sum(-1)

    def encode(self, x: jax.Array) -> jax.Array:
        """Turn four-vectors into the angles the first QFM block encodes.

        Args:
            x: ``(B, L, 4)`` four-vectors.

        Returns:
            ``(B, L, 2)`` angles after optional whitening and the preconditioner.
        """
        if self.whitening is not None:
            x = x @ self.whitening.T
        angles = ANGLE_MAPS[self.angle_map](x)
        return angles if self.preconditioner is None else self.preconditioner(angles)

    def edge_angles(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """The angle vectors the first QFM block encodes, over real edges only.

        Args:
            x: ``(B, L, 4)`` four-vectors.
            mask: Boolean ``(B, L)``, True on real particles.

        Returns:
            ``(n_edges, n_qubits)`` angles, one row per directed edge between two
            distinct real particles.
        """
        pairs = node2edge(self.encode(x)).reshape(-1, self.n_qubits)
        return pairs[_edge_mask(mask).reshape(-1)]

    def encoded_angles(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """The angles the qubits actually rotate by, over real edges only.

        One encoding layer turns the edge's feature vector ``u`` into
        ``theta_q = sum_f W[q, f] u_f``, since re-uploaded ``RY`` gates on one wire
        add. With the phase-3/4 arm (``hamming``, ``diagonal``) ``W`` is the
        identity and this is :meth:`edge_angles` itself; arm B is exactly the
        change of ``W`` (``DECISIONS.md`` D96).

        Args:
            x: ``(B, L, 4)`` four-vectors.
            mask: Boolean ``(B, L)``, True on real particles.

        Returns:
            ``(n_edges, 4)`` per-qubit encoding angles.
        """
        return self.edge_angles(x, mask) @ self.enc_matrix.T

    def g_purity(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """Mean closed-form g-purity of the angle distribution this arm encodes.

        The phase-4 observable. The argument is the encoded angle ``theta = W u``
        of a single layer, which makes this the g-purity of the product state
        ``prod_q RY(theta_q)|0>`` entering the first trainable block -- the scope the
        unflattening closed forms claim under re-uploading, and the same
        convention the whitening acceptance test uses (``DECISIONS.md`` D78). It
        is a property of *data plus encoding*, not of the trained circuit; for
        that, see :meth:`g_purity_exact`. Costs ``O(n)`` per edge, so it is cheap
        enough to track every epoch.

        Read it against :func:`partiqledtr.analysis.uniform_prior_mean` for the same
        arm: on a floor-free arm (``XY_Brickwork``, ``XY_Ring``) the theory predicts
        a collapse on clustered inputs and a rise over training when a preconditioner
        rescues it, on a floored one (``XY_AllPairs``) indifference, and on
        ``Circuit_19`` a constant (``DECISIONS.md`` D51).

        Args:
            x: ``(B, L, 4)`` four-vectors.
            mask: Boolean ``(B, L)``, True on real particles.

        Returns:
            Scalar mean g-purity over the real edges.
        """
        return jnp.mean(product_state_purity(self.encoded_angles(x, mask), self.ansatz))

    def angle_stats(self, x: jax.Array, mask: jax.Array) -> dict[str, list[float]]:
        """Shape of the angle distribution this arm's first block encodes, per qubit.

        The companion to :meth:`g_purity`, and the reason both are needed: a purity
        can rise because the angles spread toward uniform or because they pin near
        ``pi/2``, and those are opposite in what they do to the input information
        (``DECISIONS.md`` D92). ``mean_sin2`` separates them -- it tends to 0.5 for
        a uniform law, to 1 when pinned at ``pi/2`` and to 0 when clustered at zero.

        Args:
            x: ``(B, L, 4)`` four-vectors.
            mask: Boolean ``(B, L)``, True on real particles.

        Returns:
            The per-qubit record of :func:`partiqledtr.analysis.angle_stats`.
        """
        return angle_stats(np.asarray(self.encoded_angles(x, mask)))

    def g_purity_exact(self, x: jax.Array, mask: jax.Array) -> float:
        """Mean g-purity of the state the first QFM block actually prepares.

        Runs the circuit to its statevector, parameters and all, and sums
        ``<psi|B|psi>**2`` over the arm's DLA basis. This is the model-side
        counterpart of :meth:`g_purity`: the two agree in the clustered limit,
        where every encoding rotation tends to the identity, and diverge at
        generic angles because qml-essentials orders each layer ansatz-first
        (``DECISIONS.md`` D78).

        Not jittable and ``O(4**n)`` in the basis, so it is measured once at the
        end of training rather than per step.

        Args:
            x: ``(B, L, 4)`` four-vectors.
            mask: Boolean ``(B, L)``, True on real particles.

        Returns:
            Mean exact g-purity over the real edges, or NaN if there are none.
        """
        angles = self.edge_angles(x, mask)
        if angles.shape[0] == 0:
            return float("nan")
        basis = dla_basis(self.ansatz, self.n_qubits)
        params = self.qfm1_params[...]
        # Mean over channels: every channel encodes the same angles, so this is
        # the average over the block's independent circuits (D108).
        values = [
            g_purity_exact(
                np.asarray(
                    self._qfm1.apply(params=params[c], inputs=angles, execution_type="state")
                ).reshape(-1, 2**self.n_qubits),
                basis,
            )
            for c in range(self.n_channels)
        ]
        return float(np.mean(values))

    def _node_state(self, x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Block 1 plus the node update: what block 2 encodes, and the edge mask.

        Channels are unrolled: n_channels stays small (D108), and each call is
        the same jitted edge evaluation with a different parameter slice.

        Returns:
            ``((B, L, K, A) angles, (B, L, L) edge mask)``, with ``node_omega``
            already applied -- this is exactly the state block 2 consumes.
        """
        edge_mask = _edge_mask(mask)
        angles = self.encode(x)
        params1 = self.qfm1_params[...]
        messages = jnp.concatenate(
            [
                edge2node(self._edges(self._qfm1, params1[c], angles), edge_mask)
                for c in range(self.n_channels)
            ],
            axis=-1,
        )
        hidden = self.node_omega * self.w_node(jnp.concatenate([angles, messages], axis=-1))
        return hidden.reshape(*hidden.shape[:-1], self.n_channels, self.n_angles), edge_mask

    def block2_encoded_angles(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """The angles the *second* QFM block actually rotates by, over real edges.

        The diagnostic of the re-encoding boundary: block 2 consumes the node
        update's output directly as RY angles, and nothing gates that
        distribution the way D111 gates the block-1 charts -- at the default
        init it starts near zero, the collapsed point of a floor-free arm.
        Channels are stacked as extra rows (same sites, independent circuits).

        Args:
            x: ``(B, L, 4)`` four-vectors.
            mask: Boolean ``(B, L)``, True on real particles.

        Returns:
            ``(K * n_edges, n_qubits)`` encoded angles ``theta = W u``.
        """
        pairs, edge_mask = self._node_state(x, mask)
        keep = edge_mask.reshape(-1)
        per_channel = [
            node2edge(pairs[..., c, :]).reshape(-1, self.n_qubits)[keep]
            for c in range(self.n_channels)
        ]
        return jnp.concatenate(per_channel, axis=0) @ self.enc_matrix.T

    def block2_g_purity(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """Mean closed-form g-purity of the distribution block 2 encodes.

        The companion of :meth:`g_purity` one block deeper. Under re-uploading
        the closed form's premise (a product state entering the block) holds
        only approximately here -- block 1's circuit acts in between -- so read
        it as the same kind of diagnostic the D78 convention gives block 1, not
        as an exact variance-law input.
        """
        return jnp.mean(product_state_purity(self.block2_encoded_angles(x, mask), self.ansatz))

    def block2_angle_stats(self, x: jax.Array, mask: jax.Array) -> dict[str, list[float]]:
        """Shape of the distribution block 2 encodes, per qubit (cf. :meth:`angle_stats`)."""
        return angle_stats(np.asarray(self.block2_encoded_angles(x, mask)))

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """Predict LCAG class logits.

        Args:
            x: ``(B, L, 4)`` four-vectors of the final-state particles.
            mask: Boolean ``(B, L)``, True on real particles.

        Returns:
            Logits of shape ``(B, L, L, C)``, symmetric in the two ``L`` axes.
        """
        pairs, _ = self._node_state(x, mask)
        params2 = self.qfm2_params[...]
        edges = jnp.concatenate(
            [self._edges(self._qfm2, params2[c], pairs[..., c, :]) for c in range(self.n_channels)],
            axis=-1,
        )
        logits = self.head(edges)
        return (logits + jnp.swapaxes(logits, 1, 2)) / 2
