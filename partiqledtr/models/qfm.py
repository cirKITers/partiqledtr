"""QFM constellation: shared-weight quantum Fourier models as the edge function.

ROADMAP phase 3. Many small quantum Fourier models (4 qubits each) sit inside a
message-passing network, one evaluation per directed edge, sharing one parameter
set across every edge. Sharing is what makes the model permutation-equivariant;
staying small is what keeps the analytic simulation cheap and the per-QFM spectrum
tractable.

Every classical part is deliberately particle-local -- an elementwise front end, a
parameter-free masked mean, a per-node linear map -- so cross-particle structure
can only come from the quantum edge function (``DECISIONS.md`` D27).

Encoding contract, verified by measurement rather than assumed (``DECISIONS.md``
D25, D55): with ``encoding=["RY"] * n_qubits`` and a diagonal ``data_reupload``
mask, feature ``f`` is encoded on qubit ``f`` alone, once per layer. The encoded
state is then exactly the product state ``prod_q RY(n_layers * u_q)|0>`` the
unflattening closed forms are written for, and with zero ansatz parameters the
per-qubit readout is exactly ``cos(n_layers * u_q)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from qml_essentials.model import Model

from partiqledtr.models.gnn import _edge_mask, edge2node, node2edge

__all__ = ["ANSAETZE", "N_QUBITS", "QFMConstellation", "make_qfm"]

ANSAETZE = ("XY_Brickwork", "Matchgate", "Circuit_19")

#: Qubits per edge QFM: two pair-polar angles from each of the two endpoints (D24).
N_QUBITS = 4
N_ANGLES = N_QUBITS // 2


def make_qfm(ansatz: str, *, n_layers: int = 2, seed: int = 0) -> Model:
    """Build one edge QFM.

    Args:
        ansatz: One of :data:`ANSAETZE`.
        n_layers: Data-reuploading depth. The effective encoding angle is
            ``n_layers * u``, so this also multiplies the model's frequency support.
        seed: Seed for the model's own parameter initialisation. The constellation
            copies those values into its NNX state and never uses the model's copy
            again.

    Returns:
        A qml-essentials :class:`~qml_essentials.model.Model` with analytic
        expectation values and one Pauli-Z observable per qubit.

    Raises:
        ValueError: If the ansatz is unknown or ``n_layers`` is not positive.
    """
    if ansatz not in ANSAETZE:
        raise ValueError(f"ansatz must be one of {list(ANSAETZE)}, got {ansatz!r}")
    if n_layers < 1:
        raise ValueError(f"n_layers must be positive, got {n_layers}")

    # Diagonal mask: feature f reaches qubit f only, in every layer. Passed as a
    # nested list rather than a NumPy array because that is the form
    # qml-essentials annotates; the setter converts either to the same array, and
    # both were checked to give identical outputs.
    reupload = [
        [[qubit == feature for feature in range(N_QUBITS)] for qubit in range(N_QUBITS)]
        for _ in range(n_layers)
    ]

    return Model(
        n_qubits=N_QUBITS,
        n_layers=n_layers,
        circuit_type=ansatz,
        encoding=["RY"] * N_QUBITS,
        data_reupload=reupload,
        observables=list(range(N_QUBITS)),
        shots=None,
        random_seed=seed,
    )


def pair_polar(p4: jax.Array) -> jax.Array:
    """Map four-vectors to the two pair-polar angles the QFMs encode.

    Coordinate pairs become polar angles, ``(px, py) -> phi`` and ``(pz, E) ->
    alpha``, both in ``[0, 2 pi)``. This is the unflattening paper's ``polar_angles``
    map, which is what lets the phase-4 whitening arm be that paper's construction
    verbatim (``DECISIONS.md`` D24). The pair radii are dropped, so momentum and
    energy magnitudes do not enter the quantum path.

    Note that ``alpha`` depends on the momentum and energy normalisation scales,
    which differ slightly, so the ``(pz, E)`` plane is mildly anisotropic. That is
    a fixed reparametrisation of the input, and the whitening arm rotates the
    four-vector anyway.

    Args:
        p4: ``(..., 4)`` four-vectors laid out ``[px, py, pz, E]``.

    Returns:
        ``(..., 2)`` angles in ``[0, 2 pi)``.

    Raises:
        ValueError: If the last axis is not 4.
    """
    if p4.shape[-1] != 4:
        raise ValueError(f"expected four-vectors on the last axis, got {p4.shape[-1]}")
    return jnp.mod(jnp.arctan2(p4[..., 1::2], p4[..., 0::2]), 2 * jnp.pi)


class QFMConstellation(nnx.Module):
    """Two message-passing blocks whose edge function is a shared-weight QFM.

    Shapes, with ``B`` events, ``L`` padded particles and ``C`` classes::

        p4    (B, L, 4)      four-vectors
        ang   (B, L, 2)      pair-polar angles, optionally whitened
        a     (B, L, 2)      front end (identity or elementwise residual MLP)
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

    Args:
        n_classes: Number of LCAG classes ``C``.
        ansatz: Ansatz arm, one of :data:`ANSAETZE`.
        n_layers: Data-reuploading depth of each QFM.
        frontend: Optional elementwise front end ``(..., 2) -> (..., 2)`` applied to
            the angles; ``None`` feeds them raw.
        whitening: Optional ``(4, 4)`` rotation applied to the four-vectors before
            the polar map -- the phase-4 fixed-whitening arm.
        seed: Seed for the quantum parameter initialisation.
        rngs: Rng container for the classical parameters.

    Raises:
        ValueError: If ``n_classes < 2``, the ansatz is unknown, or ``whitening``
            is not ``(4, 4)``.
    """

    #: A front end attached here sees the pair-polar angles, not the raw
    #: four-vectors, so it is built for this many features rather than ``F``.
    frontend_features = N_ANGLES

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        ansatz: str = "XY_Brickwork",
        n_layers: int = 2,
        frontend: nnx.Module | None = None,
        whitening: jax.Array | None = None,
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
        if whitening is not None and tuple(whitening.shape) != (4, 4):
            raise ValueError(f"whitening must be (4, 4), got {tuple(whitening.shape)}")

        self.ansatz = ansatz
        self.n_layers = n_layers
        self.frontend = frontend
        self.whitening = None if whitening is None else jnp.asarray(whitening)

        first, second = (
            make_qfm(ansatz, n_layers=n_layers, seed=seed),
            make_qfm(ansatz, n_layers=n_layers, seed=seed + 1),
        )
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
        no model state and keeps the full ``(B_I, B_P, B_R, O)`` output rank, so the
        call is safe inside an outer ``jax.jit`` and its shape does not depend on
        the batch sizes.
        """
        batch, n_leaves = angles.shape[0], angles.shape[1]
        flat = node2edge(angles).reshape(-1, N_QUBITS)
        out = qfm.apply(params=params, inputs=flat)
        expected = (flat.shape[0], 1, 1, N_QUBITS)
        if out.shape != expected:
            raise ValueError(f"QFM returned {out.shape}, expected {expected}")
        return out.reshape(batch, n_leaves, n_leaves, N_QUBITS)

    def encode(self, x: jax.Array) -> jax.Array:
        """Turn four-vectors into the angles the first QFM block encodes.

        Args:
            x: ``(B, L, 4)`` four-vectors.

        Returns:
            ``(B, L, 2)`` angles after optional whitening and the front end.
        """
        if self.whitening is not None:
            x = x @ self.whitening.T
        angles = pair_polar(x)
        return angles if self.frontend is None else self.frontend(angles)

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

    def g_purity(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """Mean g-purity of the states this arm's first block encodes.

        The phase-4 observable. The encoded state is the RY product state
        ``prod_q RY(n_layers * u_q)|0>``, so the closed forms of
        :mod:`partiqledtr.analysis` apply directly and the measurement costs
        ``O(n)`` per edge -- cheap enough to track every epoch.

        Read it against :func:`partiqledtr.analysis.offdiag_uniform_mean`: on
        ``XY_Brickwork`` the theory predicts a collapse on clustered inputs and a
        rise over training when a front end rescues it, on ``Matchgate``
        indifference, and on ``Circuit_19`` a constant (``DECISIONS.md`` D51).

        Args:
            x: ``(B, L, 4)`` four-vectors.
            mask: Boolean ``(B, L)``, True on real particles.

        Returns:
            Scalar mean g-purity over the real edges.
        """
        from partiqledtr.analysis import G_PURITY_BY_ANSATZ

        angles = self.edge_angles(x, mask)
        return jnp.mean(G_PURITY_BY_ANSATZ[self.ansatz](self.n_layers * angles))

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
