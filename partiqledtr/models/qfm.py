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
:func:`encoding_matrix`, and with zero ansatz parameters the per-qubit readout is
exactly ``cos(n_layers * theta_q)``. The phase-3/4 arm has ``W = I`` -- feature
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

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import ArrayLike
from qml_essentials.ansaetze import Encoding
from qml_essentials.model import Model

from partiqledtr.analysis import angle_stats, dla_basis, g_purity_exact, product_state_purity
from partiqledtr.ansaetze import ANSAETZE, circuit
from partiqledtr.models.gnn import _edge_mask, edge2node, node2edge

__all__ = [
    "ANGLE_MAPS",
    "ANSAETZE",
    "ENC_REUPLOAD",
    "ENC_WEIGHTS",
    "N_QUBITS",
    "QFMConstellation",
    "encoding_matrix",
    "encoding_spectrum",
    "legacy_angles",
    "make_qfm",
    "pair_polar",
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
#: ``ternary_pair`` repeats the exponent per *particle*, which is the only family
#: that is dissociated **and** invariant under the endpoint swap (D100).
ENC_WEIGHTS: dict[str, Callable[[int], np.ndarray]] = {
    name: functools.partial(_strategy_weights, name) for name in ("hamming", "binary", "ternary")
} | {"ternary_pair": lambda n: 3.0 ** (np.arange(n) % N_ANGLES)}


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


def make_qfm(
    ansatz: str,
    *,
    n_layers: int = 2,
    seed: int = 0,
    enc_weights: str = "hamming",
    enc_reupload: str = "diagonal",
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

    Returns:
        A qml-essentials :class:`~qml_essentials.model.Model` with analytic
        expectation values and one Pauli-Z observable per qubit.

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
        n_qubits=N_QUBITS,
        n_layers=n_layers,
        circuit_type=pqc,
        encoding=["RY"] * N_QUBITS,
        data_reupload=reupload_mask(enc_reupload, n_layers),
        observables=list(range(N_QUBITS)),
        shots=None,
        random_seed=seed,
    )
    # The weights ride in `enc_params` rather than in an `Encoding` strategy, so
    # one matrix is the whole of what arm B varies and a weighting qml-essentials
    # does not ship -- `ternary_pair` -- needs no new strategy (D100). They stay
    # frozen: trainable frequencies are a separate ROADMAP axis.
    matrix = encoding_matrix(enc_weights, enc_reupload, N_QUBITS)
    model.enc_params = jnp.broadcast_to(jnp.asarray(matrix), (n_layers, N_QUBITS, N_QUBITS))
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


#: Four-vectors to the two angles each particle contributes to an edge QFM.
#: ``"legacy"`` is the clustered control arm and needs the ``"legacy"`` encoding.
ANGLE_MAPS = {"pair_polar": pair_polar, "legacy": legacy_angles}


class QFMConstellation(nnx.Module):
    """Two message-passing blocks whose edge function is a shared-weight QFM.

    Shapes, with ``B`` events, ``L`` padded particles and ``C`` classes::

        p4    (B, L, 4)      four-vectors
        ang   (B, L, 2)      pair-polar angles, optionally whitened
        a     (B, L, 2)      preconditioner (identity or elementwise residual MLP)
        u1    (B, L, L, 4)   concat(a_i, a_j) -> folded to (B*L*L, 4) for the QFM
        e1    (B, L, L, 4)   per-qubit Pauli-Z expectation values
        m     (B, L, 4)      masked mean over real neighbours
        h     (B, L, 2)      [a ; m] @ w_node
        e2    (B, L, L, 4)   second QFM block, its own parameters
        out   (B, L, L, C)   symmetrised linear readout

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
        ansatz: Ansatz arm, one of :data:`ANSAETZE`.
        n_layers: Data-reuploading depth of each QFM.
        angle_map: Key into :data:`ANGLE_MAPS`: ``"pair_polar"`` for the polar map
            of ``DECISIONS.md`` D24, ``"legacy"`` for the clustered control arm,
            which expects the ``"legacy"`` encoding.
        enc_weights: Encoding weight strategy, one of :data:`ENC_WEIGHTS`.
        enc_reupload: Re-upload mask, one of :data:`ENC_REUPLOAD`. Crossed with
            ``enc_weights`` these are ROADMAP phase 4b arm B.
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

    #: A preconditioner attached here sees the pair-polar angles, not the raw
    #: four-vectors, so it is built for this many features rather than ``F``.
    preconditioner_features = N_ANGLES

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        ansatz: str = "XY_Brickwork",
        n_layers: int = 2,
        angle_map: str = "pair_polar",
        enc_weights: str = "hamming",
        enc_reupload: str = "diagonal",
        preconditioner: nnx.Module | None = None,
        whitening: Any = None,
        seed: int = 0,
        rngs: nnx.Rngs,
        dim: int = 0,
    ) -> None:
        del dim  # the quantum path is fixed at N_QUBITS (D24), so there is no width
        if n_features != 4:
            raise ValueError(
                f"the QFM constellation consumes four-vectors, so it needs the "
                f'"cartesian" encoding (4 features), got {n_features}. The "angles" '
                f"encoding drops |p|, which is not recoverable for massive final-state "
                f"particles."
            )
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if angle_map not in ANGLE_MAPS:
            raise ValueError(f"angle_map must be one of {sorted(ANGLE_MAPS)}, got {angle_map!r}")
        circuit(ansatz)  # resolve early, so an unknown arm fails here and not mid-build
        # Converted before the shape check so a checkpoint may carry the rotation
        # as a nested list rather than an array (D81).
        rotation = None if whitening is None else jnp.asarray(whitening, dtype=jnp.float32)
        if rotation is not None and rotation.shape != (4, 4):
            raise ValueError(f"whitening must be (4, 4), got {rotation.shape}")

        self.ansatz = ansatz
        self.n_layers = n_layers
        self.angle_map = angle_map
        self.enc_weights = enc_weights
        self.enc_reupload = enc_reupload
        self.preconditioner = preconditioner
        self.whitening = rotation
        # Not an nnx.Param: the encoding weights are fixed by the arm, and making
        # them trainable is a separate ROADMAP axis with its own failure mode.
        self.enc_matrix = jnp.asarray(encoding_matrix(enc_weights, enc_reupload))

        arm = {"n_layers": n_layers, "enc_weights": enc_weights, "enc_reupload": enc_reupload}
        first, second = make_qfm(ansatz, seed=seed, **arm), make_qfm(ansatz, seed=seed + 1, **arm)
        # Separate parameters per block: the two blocks do different jobs, and
        # equivariance only needs sharing across *edges* (D28).
        self.qfm1_params = nnx.Param(jnp.asarray(np.asarray(first.params)))
        self.qfm2_params = nnx.Param(jnp.asarray(np.asarray(second.params)))
        self._qfm1, self._qfm2 = first, second

        self.w_node = nnx.Linear(N_ANGLES + N_QUBITS, N_ANGLES, rngs=rngs)
        self.head = nnx.Linear(N_QUBITS, n_classes, rngs=rngs)

    def _edges(self, qfm: Model, params: jax.Array, angles: jax.Array) -> jax.Array:
        """Evaluate one QFM on every directed edge of every event.

        Uses the functional :meth:`~qml_essentials.model.Model.apply`, which writes
        no model state and keeps every batch axis unsqueezed, so the call is safe
        inside an outer ``jax.jit`` and its shape does not depend on the batch
        sizes. The guard pins that rank: a silent change to it would otherwise
        reshape into the wrong edge grid rather than fail.
        """
        batch, n_leaves = angles.shape[0], angles.shape[1]
        flat = node2edge(angles).reshape(-1, N_QUBITS)
        out = qfm.apply(params=params, inputs=flat)
        expected = (flat.shape[0], 1, 1, 1, N_QUBITS)
        if out.shape != expected:
            raise ValueError(f"QFM returned {out.shape}, expected {expected}")
        return out.reshape(batch, n_leaves, n_leaves, N_QUBITS)

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
            ``(n_edges, 4)`` angles, one row per directed edge between two distinct
            real particles.
        """
        pairs = node2edge(self.encode(x)).reshape(-1, N_QUBITS)
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
        states = self._qfm1.apply(
            params=self.qfm1_params[...], inputs=angles, execution_type="state"
        )
        basis = dla_basis(self.ansatz, N_QUBITS)
        return float(np.mean(g_purity_exact(np.asarray(states).reshape(-1, 2**N_QUBITS), basis)))

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """Predict LCAG class logits.

        Args:
            x: ``(B, L, 4)`` four-vectors of the final-state particles.
            mask: Boolean ``(B, L)``, True on real particles.

        Returns:
            Logits of shape ``(B, L, L, C)``, symmetric in the two ``L`` axes.
        """
        edge_mask = _edge_mask(mask)
        angles = self.encode(x)

        messages = edge2node(self._edges(self._qfm1, self.qfm1_params[...], angles), edge_mask)
        hidden = self.w_node(jnp.concatenate([angles, messages], axis=-1))

        edges = self._edges(self._qfm2, self.qfm2_params[...], hidden)
        logits = self.head(edges)
        return (logits + jnp.swapaxes(logits, 1, 2)) / 2
