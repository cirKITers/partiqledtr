"""NRI-style message-passing baseline for LCAG prediction (ROADMAP phase 2)."""

import jax
import jax.numpy as jnp
from flax import nnx


def node2edge(h: jax.Array) -> jax.Array:
    """Build every ordered pair of node representations.

    Args:
        h: Node representations of shape ``(B, L, d)``.

    Returns:
        Edge representations of shape ``(B, L, L, 2 * d)`` with
        ``out[b, i, j] = concat(h[b, i], h[b, j])``.
    """
    b, n, d = h.shape
    sends = jnp.broadcast_to(h[:, :, None, :], (b, n, n, d))
    recvs = jnp.broadcast_to(h[:, None, :, :], (b, n, n, d))
    return jnp.concatenate([sends, recvs], axis=-1)


def edge2node(edges: jax.Array, edge_mask: jax.Array) -> jax.Array:
    """Mean-aggregate the edges incident to each node.

    The mean runs over real neighbours only and divides by the true degree, not by
    ``L``: padded columns contribute nothing and padded rows aggregate to zero. A node
    with no real neighbour (a padded row, or a one-particle event) divides by one
    instead of by zero.

    Args:
        edges: Edge representations of shape ``(B, L, L, d)``.
        edge_mask: Boolean ``(B, L, L)``, True on edges between two distinct real
            particles (see :func:`_edge_mask`).

    Returns:
        Node representations of shape ``(B, L, d)``.
    """
    weights = edge_mask[..., None].astype(edges.dtype)
    degree = jnp.sum(weights, axis=2)
    return jnp.sum(edges * weights, axis=2) / jnp.maximum(degree, 1.0)


def _edge_mask(mask: jax.Array) -> jax.Array:
    """Turn a particle mask into an edge mask that excludes self-edges.

    Args:
        mask: Boolean ``(B, L)``, True on real particles.

    Returns:
        Boolean ``(B, L, L)``, True where both endpoints are real and ``i != j``.
    """
    pair = mask[:, :, None] & mask[:, None, :]
    return pair & ~jnp.eye(mask.shape[1], dtype=bool)


class MLPBlock(nnx.Module):
    """Two-layer ELU perceptron applied to the trailing axis.

    The only nonlinear unit of both phase-2 baselines:
    :class:`partiqledtr.models.mlp.MLPBaseline` reuses it so the two differ *only* in
    message passing.

    Args:
        n_in: Size of the trailing input axis.
        n_hidden: Width of the hidden layer.
        n_out: Size of the trailing output axis.
        rngs: Rng container used for parameter initialisation.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, *, rngs: nnx.Rngs) -> None:
        self.fc1 = nnx.Linear(n_in, n_hidden, rngs=rngs)
        self.fc2 = nnx.Linear(n_hidden, n_out, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the block.

        Args:
            x: Array of shape ``(..., n_in)``.

        Returns:
            Array of shape ``(..., n_out)``.
        """
        # ponytail: baumbauen's unit is ELU + dropout + optional batchnorm. Both
        # regularisers are dropped -- each forces train/eval state (and dropout an rng
        # stream) through every caller for no demonstrated gain at this scale. Ceiling:
        # the unconstrained (large `dim`) arm has no regularisation and may overfit;
        # upgrade is nnx.Dropout here plus model.train()/model.eval() in the trainer.
        return nnx.elu(self.fc2(nnx.elu(self.fc1(x))))


class _Block(nnx.Module):
    """One NRI block: edge MLP, edge2node, node MLP, node2edge, skip, reduce.

    Args:
        dim: Width of the node and edge representations.
        rngs: Rng container used for parameter initialisation.
    """

    def __init__(self, dim: int, *, rngs: nnx.Rngs) -> None:
        self.edge_mlp = MLPBlock(dim, dim, dim, rngs=rngs)
        self.node_mlp = MLPBlock(dim, dim, dim, rngs=rngs)
        self.reduce = MLPBlock(3 * dim, dim, dim, rngs=rngs)

    def __call__(self, edges: jax.Array, edge_mask: jax.Array) -> jax.Array:
        """Run one node <-> edge alternation.

        Args:
            edges: Edge representations of shape ``(B, L, L, dim)``.
            edge_mask: Boolean ``(B, L, L)`` from :func:`_edge_mask`.

        Returns:
            Edge representations of shape ``(B, L, L, dim)``.
        """
        skip = edges
        h = self.node_mlp(edge2node(self.edge_mlp(edges), edge_mask))
        return self.reduce(jnp.concatenate([node2edge(h), skip], axis=-1))


class LCAGGNN(nnx.Module):
    """Message-passing LCAG predictor with alternating node and edge updates.

    Node representations become edges by pairwise concatenation and edges become nodes
    by a masked mean over real neighbours, so the model is permutation-equivariant in
    the particle axis and invariant to whatever sits in the padded rows. The output is
    symmetrised architecturally because the LCAG is symmetric by construction.

    The parameter-matched and unconstrained variants of ROADMAP phase 2 are two values
    of ``dim``, not two classes.

    Args:
        n_features: Number of per-particle input features ``F``.
        n_classes: Number of LCAG classes ``C``.
        dim: Width of the node and edge representations.
        n_blocks: Number of node <-> edge blocks.
        preconditioner: Optional elementwise preconditioner ``(..., F) -> (..., F)`` applied to the
            features first; ``None`` feeds the raw features.
        rngs: Rng container used for parameter initialisation.

    Raises:
        ValueError: If ``dim < 1``, ``n_blocks < 1`` or ``n_classes < 2``.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        dim: int = 64,
        n_blocks: int = 3,
        preconditioner: nnx.Module | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.preconditioner = preconditioner
        self.initial_mlp = MLPBlock(n_features, dim, dim, rngs=rngs)
        self.pre_blocks_mlp = MLPBlock(2 * dim, dim, dim, rngs=rngs)
        self.blocks = nnx.List([_Block(dim, rngs=rngs) for _ in range(n_blocks)])
        self.final_mlp = MLPBlock(2 * dim, dim, dim, rngs=rngs)
        self.head = nnx.Linear(dim, n_classes, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """Predict LCAG class logits.

        Args:
            x: Features of shape ``(B, L, F)``.
            mask: Boolean ``(B, L)``, True on real particles.

        Returns:
            Logits of shape ``(B, L, L, C)``, symmetric in the two ``L`` axes.
        """
        if self.preconditioner is not None:
            x = self.preconditioner(x)
        edge_mask = _edge_mask(mask)
        edges = self.pre_blocks_mlp(node2edge(self.initial_mlp(x)))
        global_skip = edges
        for block in self.blocks:
            edges = block(edges, edge_mask)
        edges = self.final_mlp(jnp.concatenate([edges, global_skip], axis=-1))
        logits = self.head(edges)
        return (logits + jnp.swapaxes(logits, 1, 2)) / 2
