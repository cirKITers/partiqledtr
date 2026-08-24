"""LCAG models and the registries the training node selects them by (ROADMAP phases 2-4).

Call convention, satisfied by every entry of :data:`MODELS` including the phase-3
quantum one:

    A model is an ``nnx.Module`` with
    ``__call__(features: (B, L, F), mask: (B, L) bool) -> logits: (B, L, L, C)``,
    symmetric in the two ``L`` axes.

``B`` is the batch size, ``L`` the padded number of final-state particles, ``F`` the
number of per-particle features and ``C`` the number of LCAG classes. ``mask`` is True
on real particles; whatever sits in the padded rows must not reach the logits of the
valid block. A front end is an ``nnx.Module`` mapping ``(..., F) -> (..., F)``, passed
to a model as its ``frontend`` argument.

There is no abstract base class: the convention plus the two registries is the whole
interface. Phase 3 adds a ``"qfm"`` model, phase 4 a ``"whiten"`` front end.
"""

import jax
from flax import nnx

from partiqledtr.models.frontend import ElementwiseResidualMLP
from partiqledtr.models.gnn import LCAGGNN
from partiqledtr.models.mlp import MLPBaseline
from partiqledtr.models.qfm import QFMConstellation

#: ``"qfm"`` consumes four-vectors, so it requires the ``"cartesian"`` encoding;
#: the classical baselines accept either.
MODELS: dict[str, type] = {"gnn": LCAGGNN, "mlp": MLPBaseline, "qfm": QFMConstellation}
FRONTENDS: dict[str, type | None] = {"none": None, "mlp": ElementwiseResidualMLP}


def n_params(module: nnx.Module) -> int:
    """Count the trainable scalars of a module.

    Args:
        module: Any NNX module.

    Returns:
        Total number of elements over all :class:`flax.nnx.Param` leaves. Used to
        parameter-match the phase-3 quantum model against a classical baseline.
    """
    return sum(int(leaf.size) for leaf in jax.tree.leaves(nnx.state(module, nnx.Param)))
