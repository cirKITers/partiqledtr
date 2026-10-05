"""Message-passing-free control baseline."""

import jax
import jax.numpy as jnp
from flax import nnx

from partiqledtr.models.gnn import node2edge


class MLPBaseline(nnx.Module):
    """Predict symmetric pair logits with a linear head on preconditioned features.

    Each pair uses only its two particles, without neighbour aggregation.

    Args:
        n_features: Number of per-particle input features ``F``.
        n_classes: Number of LCAG classes ``C``.
        dim: Unused. Accepted so every entry of :data:`~partiqledtr.models.MODELS` is
            constructed the same way; a linear control has no width to set.
        preconditioner: Optional elementwise preconditioner ``(..., F) -> (..., F)`` applied to the
            features first; ``None`` feeds the raw features.
        rngs: Rng container used for parameter initialisation.

    Raises:
        ValueError: If ``n_classes < 2``.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        dim: int = 64,
        preconditioner: nnx.Module | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        del dim
        self.preconditioner = preconditioner
        self.head = nnx.Linear(2 * n_features, n_classes, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """Predict LCAG class logits.

        Args:
            x: Features of shape ``(B, L, F)``.
            mask: Boolean ``(B, L)``, True on real particles. Unused: without
                aggregation, ``logits[i, j]`` cannot depend on a padded row, so padding
                invariance is structural here. Accepted to satisfy the call convention.

        Returns:
            Logits of shape ``(B, L, L, C)``, symmetric in the two ``L`` axes.
        """
        if self.preconditioner is not None:
            x = self.preconditioner(x)
        logits = self.head(node2edge(x))
        return (logits + jnp.swapaxes(logits, 1, 2)) / 2
