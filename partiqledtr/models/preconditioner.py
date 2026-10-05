"""Elementwise residual preconditioner."""

import jax
import jax.numpy as jnp
from flax import nnx


class ElementwiseResidualMLP(nnx.Module):
    """Apply an independent residual MLP to each feature.

    Feature axes never mix. A zero-initialised output layer makes the module an
    exact identity at initialisation.

    Args:
        n_features: Size of the trailing feature axis ``F``.
        hidden: Width of the per-feature hidden layer.
        rngs: Rng container used for parameter initialisation.
        param_dtype: Dtype of the parameters, matching the ``nnx.Linear`` default so
            the whole model stays one precision whatever the global x64 flag says.

    Raises:
        ValueError: If ``n_features < 1`` or ``hidden < 1``.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 16,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        shape = (n_features, hidden)
        # Pinned for the same reason `nnx.Linear` pins `param_dtype`: without an
        # explicit dtype these follow the global `jax_enable_x64` flag, so with x64
        # on they would come out float64 while every Linear in the model stayed
        # float32, and the preconditioner would silently promote the whole forward pass.
        # Random slopes and biases spread the tanh kinks over the input range. Zeroing
        # w2 also zeroes the gradient w.r.t. w1 and b1 at step 0, so w2 moves first.
        self.w1 = nnx.Param(jax.random.normal(rngs.params(), shape, dtype=param_dtype))
        self.b1 = nnx.Param(jax.random.normal(rngs.params(), shape, dtype=param_dtype))
        self.w2 = nnx.Param(jnp.zeros(shape, dtype=param_dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the preconditioner elementwise.

        Args:
            x: Array of shape ``(..., F)``.

        Returns:
            Array of the same shape, output feature ``f`` depending on ``x[..., f]`` only.
        """
        h = jnp.tanh(jnp.einsum("...f,fh->...fh", x, self.w1[...]) + self.b1[...])
        return x + jnp.einsum("...fh,fh->...f", h, self.w2[...])
