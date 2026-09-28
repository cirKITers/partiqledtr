import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from partiqledtr.models import MODELS, PRECONDITIONERS, n_params
from partiqledtr.models.gnn import LCAGGNN
from partiqledtr.models.mlp import MLPBaseline
from partiqledtr.models.preconditioner import ElementwiseResidualMLP

B, L, F, C, DIM, SEED = 2, 5, 3, 4, 8, 0
N_VALID = (5, 3)  # batch row 0 is unpadded, row 1 has two padded particles
TOL = 1e-5  # float32, after ~10 matmuls


def _batch(seed: int = 1, n_valid: tuple[int, ...] = N_VALID):
    """Data contract: float (B, L, F) features zeroed on padded rows, bool (B, L) mask."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((B, L), dtype=bool)
    for b, n in enumerate(n_valid):
        mask[b, :n] = True
    x = rng.normal(size=(B, L, F)).astype(np.float32) * mask[..., None]
    return jnp.asarray(x), jnp.asarray(mask)


def _build(cls, *, dim: int = DIM, preconditioner=None, seed: int = SEED):
    return cls(F, C, dim=dim, preconditioner=preconditioner, rngs=nnx.Rngs(seed))


@pytest.mark.parametrize("cls", [LCAGGNN, MLPBaseline])
def test_output_shape_and_symmetry(cls):
    x, mask = _batch()
    out = _build(cls)(x, mask)
    assert out.shape == (B, L, L, C)
    # Symmetrisation is architectural, so equality is exact, not approximate.
    assert jnp.array_equal(out, jnp.swapaxes(out, 1, 2))


@pytest.mark.parametrize("cls", [LCAGGNN, MLPBaseline])
def test_padding_invariance(cls):
    """Garbage in the masked-out rows must not move the logits of the valid block."""
    x, mask = _batch()
    garbage = jnp.asarray(np.random.default_rng(2).normal(size=x.shape).astype(np.float32) * 10.0)
    x_garbage = jnp.where(mask[..., None], x, garbage)
    model = _build(cls)
    out, out_garbage = model(x, mask), model(x_garbage, mask)
    for b, n in enumerate(N_VALID):
        assert jnp.allclose(out[b, :n, :n], out_garbage[b, :n, :n], atol=TOL)
    # Not vacuous: the garbage does reach the padded block, it just stays there.
    assert not jnp.allclose(out, out_garbage, atol=TOL)


def test_gnn_permutation_equivariance():
    """logits(x[:, p])[i, j] == logits(x)[p[i], p[j]] on an unpadded batch."""
    x, _ = _batch(n_valid=(L, L))
    mask = jnp.ones((B, L), dtype=bool)
    perm = np.asarray([2, 0, 4, 1, 3])
    model = _build(LCAGGNN)
    out, out_perm = model(x, mask), model(x[:, perm], mask)
    assert jnp.allclose(out_perm, out[:, perm][:, :, perm], atol=TOL)
    assert not jnp.allclose(out_perm, out, atol=TOL)  # not vacuous: the permutation bites


def test_gnn_degree_normalisation_uses_true_degree():
    """A padded event must give the same logits as the same event without padding."""
    n = N_VALID[1]
    x, mask = _batch()
    padded = _build(LCAGGNN)(x, mask)[1, :n, :n]
    unpadded = _build(LCAGGNN)(x[1:, :n], mask[1:, :n])[0]
    assert jnp.allclose(padded, unpadded, atol=TOL)


def test_preconditioner_is_identity_at_init():
    x, _ = _batch()
    preconditioner = ElementwiseResidualMLP(F, 4, rngs=nnx.Rngs(SEED))
    assert jnp.array_equal(preconditioner(x), x)  # zero-initialised w2, so exactly identity


def test_preconditioner_cannot_mix_features():
    preconditioner = ElementwiseResidualMLP(F, 4, rngs=nnx.Rngs(SEED))
    dtype = preconditioner.w2[...].dtype
    shape = preconditioner.w2.shape
    preconditioner.w2[...] = jax.random.normal(jax.random.key(3), shape, dtype=dtype)
    x = jax.random.normal(jax.random.key(4), (F,), dtype=dtype)
    jac = jax.jacobian(preconditioner)(x)
    assert jac.shape == (F, F)
    # Exactly zero off-diagonal: the einsums never contract over the feature axis.
    assert jnp.count_nonzero(jac - jnp.diag(jnp.diagonal(jac))) == 0
    # ... and the check is not vacuous: the perturbed preconditioner is no longer identity.
    assert not jnp.allclose(jnp.diagonal(jac), 1.0)


@pytest.mark.parametrize("cls", [LCAGGNN, MLPBaseline])
def test_identity_preconditioner_leaves_model_output_unchanged(cls):
    x, mask = _batch()
    preconditioner = ElementwiseResidualMLP(F, 4, rngs=nnx.Rngs(9))
    plain, wrapped = _build(cls), _build(cls, preconditioner=preconditioner)
    assert jnp.array_equal(plain(x, mask), wrapped(x, mask))
    assert n_params(wrapped) == n_params(plain) + 3 * F * 4


def test_parameter_dtype_does_not_follow_the_global_x64_flag():
    """Every parameter is float32, whether or not something enabled x64.

    Parameters created without an explicit dtype follow the global
    `jax_enable_x64` flag, while every `nnx.Linear` pins `param_dtype=float32`.
    Mixing the two silently promotes the forward pass to float64, so a model built
    under one setting would not match one built under the other. Pinning the dtype
    everywhere is what keeps the models independent of a global flag that any
    dependency, or the user, may flip.
    """
    was_enabled = jax.config.jax_enable_x64
    try:
        for enabled in (False, True):
            jax.config.update("jax_enable_x64", enabled)
            for module in (_build(LCAGGNN), ElementwiseResidualMLP(F, 4, rngs=nnx.Rngs(SEED))):
                dtypes = {leaf.dtype for leaf in jax.tree.leaves(nnx.state(module, nnx.Param))}
                assert dtypes == {np.dtype("float32")}, f"x64={enabled} gave {dtypes}"
    finally:
        jax.config.update("jax_enable_x64", was_enabled)


@pytest.mark.parametrize("cls", [LCAGGNN, MLPBaseline, ElementwiseResidualMLP])
def test_same_seed_gives_identical_parameters(cls):
    def build():
        if cls is ElementwiseResidualMLP:
            return cls(F, 4, rngs=nnx.Rngs(SEED))
        return _build(cls)

    left, right = nnx.state(build(), nnx.Param), nnx.state(build(), nnx.Param)
    pairs = jax.tree.leaves(jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), left, right))
    assert pairs and all(pairs)


@pytest.mark.parametrize("dim", [8, 16, 32, 64])
def test_parameter_counts(dim, capsys):
    # The quantum arm has no `dim` and takes four-vectors; it is counted in
    # tests/test_qfm.py instead.
    classical = {name: cls for name, cls in MODELS.items() if name != "qfm"}
    counts = {name: n_params(_build(cls, dim=dim)) for name, cls in classical.items()}
    with capsys.disabled():
        print(f"\nn_params(dim={dim:3d}): " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    assert counts["mlp"] < counts["gnn"]  # the control is the cheaper model
    assert all(v > 0 for v in counts.values())


_BASE_ARGS = {
    LCAGGNN: {"n_features": F, "n_classes": C},
    MLPBaseline: {"n_features": F, "n_classes": C},
    ElementwiseResidualMLP: {"n_features": F},
}


@pytest.mark.parametrize(
    ("cls", "kwargs", "match"),
    [
        (LCAGGNN, {"dim": 0}, "dim"),
        (LCAGGNN, {"n_blocks": 0}, "n_blocks"),
        (LCAGGNN, {"n_classes": 1}, "n_classes"),
        # MLPBaseline takes no `dim`: a linear control has no width to validate.
        (MLPBaseline, {"n_classes": 1}, "n_classes"),
        (ElementwiseResidualMLP, {"n_features": 0}, "n_features"),
        (ElementwiseResidualMLP, {"hidden": 0}, "hidden"),
    ],
)
def test_rejects_invalid_arguments(cls, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cls(**(_BASE_ARGS[cls] | kwargs), rngs=nnx.Rngs(SEED))


def test_registries_expose_the_selectable_components():
    assert MODELS["gnn"] is LCAGGNN
    assert MODELS["mlp"] is MLPBaseline
    assert set(MODELS) == {"gnn", "mlp", "qfm"}  # "qfm" is the quantum arm
    assert PRECONDITIONERS == {"none": None, "mlp": ElementwiseResidualMLP}
