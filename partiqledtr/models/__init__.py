"""LCAG models and the registries the training node selects them by.

Call convention, satisfied by every entry of :data:`MODELS` including the quantum
one:

    A model is an ``nnx.Module`` with
    ``__call__(features: (B, L, F), mask: (B, L) bool) -> logits: (B, L, L, C)``,
    symmetric in the two ``L`` axes.

``B`` is the batch size, ``L`` the padded number of final-state particles, ``F`` the
number of per-particle features and ``C`` the number of LCAG classes. ``mask`` is True
on real particles; whatever sits in the padded rows must not reach the logits of the
valid block. A preconditioner is an ``nnx.Module`` mapping ``(..., F) -> (..., F)``, passed
to a model as its ``preconditioner`` argument.

There is no abstract base class: the convention plus the two registries is the whole
interface.

:func:`n_params` counts a model and :func:`matched_dim` inverts that count, which is
how the parameter-matched classical arm of a study is chosen.
"""

import jax
from flax import nnx

from partiqledtr.models.gnn import LCAGGNN
from partiqledtr.models.mlp import MLPBaseline
from partiqledtr.models.preconditioner import ElementwiseResidualMLP
from partiqledtr.models.qfm import QFMConstellation

#: ``"qfm"`` consumes four-vectors, so it requires the ``"cartesian"`` encoding;
#: the classical baselines accept either.
MODELS: dict[str, type] = {"gnn": LCAGGNN, "mlp": MLPBaseline, "qfm": QFMConstellation}
PRECONDITIONERS: dict[str, type | None] = {"none": None, "mlp": ElementwiseResidualMLP}


def n_params(module: nnx.Module) -> int:
    """Count the trainable scalars of a module.

    Args:
        module: Any NNX module.

    Returns:
        Total number of elements over all :class:`flax.nnx.Param` leaves. Used to
        parameter-match the quantum model against a classical baseline.
    """
    return sum(int(leaf.size) for leaf in jax.tree.leaves(nnx.state(module, nnx.Param)))


def matched_dim(target: int, build, *, max_dim: int = 256, **kwargs) -> int:
    """Return the ``dim`` whose parameter count sits closest to ``target``.

    A hand-picked "matched" width is easily off by an order of magnitude. This makes
    the match computable, so the matched arm of a study is derived rather than asserted.

    Args:
        target: Parameter count to match, e.g. that of the quantum arm.
        build: A callable taking ``dim=`` plus ``kwargs`` and returning a module,
            normally :func:`partiqledtr.train.build_model`. Its parameter count has
            to grow with ``dim``, which every model here satisfies.
        max_dim: Largest width to consider.
        **kwargs: Forwarded to ``build`` unchanged.

    Returns:
        The best ``dim`` in ``[1, max_dim]``.

    Raises:
        ValueError: If ``target`` is not positive or ``max_dim`` is below 1.
    """
    if target < 1 or max_dim < 1:
        raise ValueError(f"target and max_dim must be positive, got {target}, {max_dim}")

    # Parameter count grows with width, so bisect rather than build every candidate:
    # a linear scan to 256 costs hundreds of model constructions for the same answer.
    low, high = 1, max_dim
    while low < high:
        mid = (low + high) // 2
        if n_params(build(dim=mid, **kwargs)) < target:
            low = mid + 1
        else:
            high = mid
    below = max(low - 1, 1)
    return min((below, low), key=lambda dim: abs(n_params(build(dim=dim, **kwargs)) - target))
