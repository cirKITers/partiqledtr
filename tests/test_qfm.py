"""QFM constellation: encoding wiring, equivariance, gradients and the whitening arm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from partiqledtr.analysis import G_PURITY_BY_ANSATZ, offdiag_uniform_mean
from partiqledtr.data.features import apply_normalization, normalization_scales
from partiqledtr.data.whitening import fit_whitening, purity_of_rotation, sample_rotation
from partiqledtr.models import MODELS, n_params
from partiqledtr.models.frontend import ElementwiseResidualMLP
from partiqledtr.models.qfm import (
    ANSAETZE,
    N_QUBITS,
    QFMConstellation,
    legacy_angles,
    make_qfm,
    pair_polar,
)

SEED = 7
C = 3

# Trainable parameters per QFM at n_qubits=4, n_layers=2 (so 3 implemented ansatz
# layers, Schuld's L+1 for a data-reuploading model). Recorded from the ansatz
# definitions and frozen: a change here means the arm's circuit changed.
QFM_PARAMS = {"XY_Brickwork": 18, "Matchgate": 21, "Circuit_19": 36}


def _batch(rng, batch=3, n_leaves=4):
    x = jnp.asarray(rng.normal(size=(batch, n_leaves, 4)))
    return x, jnp.ones((batch, n_leaves), dtype=bool)


def _model(**kwargs):
    return QFMConstellation(4, C, rngs=nnx.Rngs(SEED), seed=SEED, **kwargs)


@pytest.mark.parametrize("ansatz", ANSAETZE)
@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_encoding_is_a_ry_product_state(ansatz, n_layers):
    """With zero ansatz parameters the readout must be exactly cos(n_layers * u).

    This pins the whole encoding contract at once: that the diagonal
    ``data_reupload`` mask sends feature f to qubit f alone, that the observables
    come back in qubit order, that reuploading multiplies the angle, and that the
    encoded state is the RY product state the g-purity closed forms assume (D25).
    """
    qfm = make_qfm(ansatz, n_layers=n_layers, seed=SEED)
    u = jnp.asarray(np.random.default_rng(SEED).uniform(0, np.pi, size=(5, N_QUBITS)))

    out = qfm(params=jnp.zeros_like(jnp.asarray(np.asarray(qfm.params))), inputs=u)

    assert out.shape == (5, N_QUBITS)
    assert np.allclose(np.asarray(out), np.cos(n_layers * np.asarray(u)), atol=1e-6)


@pytest.mark.parametrize("ansatz", ANSAETZE)
def test_parameter_counts_per_arm(ansatz):
    qfm = make_qfm(ansatz, n_layers=2, seed=SEED)
    assert qfm.params.size == QFM_PARAMS[ansatz]


def test_make_qfm_validates_its_arguments():
    with pytest.raises(ValueError, match="ansatz"):
        make_qfm("Circuit_42")
    with pytest.raises(ValueError, match="n_layers"):
        make_qfm("Matchgate", n_layers=0)


def test_pair_polar_maps_coordinate_pairs_into_the_full_circle():
    # The pairs are (px, py) -> arctan2(py, px) and (pz, E) -> arctan2(E, pz).
    p4 = jnp.asarray([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, -1.0, 0.0]])
    angles = pair_polar(p4)

    assert angles.shape == (2, 2)
    assert np.allclose(np.asarray(angles[0]), [0.0, 3 * np.pi / 2])
    assert np.allclose(np.asarray(angles[1]), [np.pi / 2, np.pi])
    assert np.all(np.asarray(angles) >= 0.0) and np.all(np.asarray(angles) < 2 * np.pi)

    with pytest.raises(ValueError, match="four-vectors"):
        pair_polar(jnp.zeros((2, 3)))


@pytest.mark.parametrize("ansatz", ANSAETZE)
def test_logits_are_shaped_and_symmetric(ansatz):
    x, mask = _batch(np.random.default_rng(SEED))
    logits = _model(ansatz=ansatz)(x, mask)

    assert logits.shape == (*x.shape[:2], x.shape[1], C)
    assert jnp.array_equal(logits, jnp.swapaxes(logits, 1, 2))


def test_permutation_equivariance():
    """Shared edge weights make the model equivariant: permuting leaves conjugates
    the logits. This is the property the whole shared-weight design exists for."""
    rng = np.random.default_rng(SEED)
    x, mask = _batch(rng)
    model = _model()
    permutation = rng.permutation(x.shape[1])

    permuted = model(x[:, permutation], mask[:, permutation])
    expected = model(x, mask)[:, permutation][:, :, permutation]
    assert np.allclose(np.asarray(permuted), np.asarray(expected), atol=1e-5)


def test_padded_particles_cannot_reach_the_valid_logits():
    rng = np.random.default_rng(SEED)
    x, _ = _batch(rng, n_leaves=4)
    mask = jnp.asarray(np.array([[True, True, True, False]] * x.shape[0]))
    model = _model()

    polluted = np.array(x)
    polluted[:, 3] = 1e3
    clean = model(x, mask)[:, :3, :3]
    dirty = model(jnp.asarray(polluted), mask)[:, :3, :3]
    assert np.allclose(np.asarray(clean), np.asarray(dirty), atol=1e-5)


def test_gradients_flow_through_the_quantum_edge_function_repeatedly():
    """Repeated and jitted gradients must agree.

    The forward pass goes through the functional ``Model.apply``, which writes no
    state onto the circuit object, so an outer transform is safe. Calling
    ``Model.__call__`` instead would stash the traced parameters on the instance
    and leak them into a later call; this test is what catches a regression into
    that pattern.
    """
    x, mask = _batch(np.random.default_rng(SEED))
    graphdef, state = nnx.split(_model())

    def loss(params):
        return jnp.sum(nnx.merge(graphdef, params)(x, mask) ** 2)

    first, second = jax.grad(loss)(state), jax.grad(loss)(state)
    jitted = jax.jit(jax.grad(loss))(state)
    leaves = jax.tree.leaves(first)

    assert leaves and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(bool(jnp.any(leaf != 0.0)) for leaf in leaves)
    for other in (second, jitted):
        for a, b in zip(leaves, jax.tree.leaves(other), strict=True):
            assert jnp.allclose(a, b, atol=1e-6)


def test_a_front_end_starts_as_the_identity():
    x, mask = _batch(np.random.default_rng(SEED))
    frontend = ElementwiseResidualMLP(2, rngs=nnx.Rngs(SEED))

    plain = _model()(x, mask)
    fronted = _model(frontend=frontend)(x, mask)
    assert jnp.allclose(plain, fronted)


def test_whitening_rotates_the_input_and_changes_the_prediction():
    rng = np.random.default_rng(SEED)
    x, mask = _batch(rng)
    rotation = jnp.asarray(sample_rotation(rng))

    plain = _model()
    whitened = _model(whitening=rotation)
    assert not np.allclose(np.asarray(plain(x, mask)), np.asarray(whitened(x, mask)))
    assert np.allclose(
        np.asarray(whitened.encode(x)), np.asarray(pair_polar(x @ rotation.T)), atol=1e-6
    )


def test_constellation_validates_its_arguments():
    with pytest.raises(ValueError, match="cartesian"):
        QFMConstellation(3, C, rngs=nnx.Rngs(SEED))
    with pytest.raises(ValueError, match="n_classes"):
        QFMConstellation(4, 1, rngs=nnx.Rngs(SEED))
    with pytest.raises(ValueError, match="whitening"):
        QFMConstellation(4, C, whitening=jnp.eye(3), rngs=nnx.Rngs(SEED))


def test_registered_as_a_model_arm():
    assert MODELS["qfm"] is QFMConstellation
    # 2 x 18 quantum + Linear(6, 2) + Linear(4, 3) for the XY arm at C = 3.
    assert n_params(_model()) == 2 * QFM_PARAMS["XY_Brickwork"] + 14 + 15


def test_sample_rotation_is_a_rotation():
    q = sample_rotation(np.random.default_rng(SEED))
    assert np.allclose(q @ q.T, np.eye(4), atol=1e-12)
    assert np.linalg.det(q) == pytest.approx(1.0)


def test_whitening_rescues_clustered_angles():
    """The phase-4 mechanism, end to end.

    Four-vectors whose pair second components are negligible encode angles near
    {0, pi}, where the floor-free g-purity collapses. A Haar rotation spreads them
    and the acceptance test passes.
    """
    rng = np.random.default_rng(SEED)
    n_events, n_leaves = 300, 4
    p4 = rng.normal(size=(n_events, n_leaves, 4))
    p4[..., 1] *= 1e-3
    p4[..., 3] = np.abs(p4[..., 3]) * 1e-3
    n_fsps = np.full(n_events, n_leaves)

    threshold = offdiag_uniform_mean(N_QUBITS) / 2.0
    raw = purity_of_rotation(p4, n_fsps, np.eye(4), np.random.default_rng(SEED))
    assert raw < threshold

    rotation, report = fit_whitening(p4, n_fsps, seed=3)
    assert rotation.shape == (4, 4)
    assert report["accepted_purity"] >= report["threshold"] == threshold
    # The report's own raw estimate uses its own pair sample, so compare the
    # verdict rather than the value: both must sit far below the threshold.
    assert report["raw_purity"] < threshold / 100
    # Markov guarantees acceptance with probability at least about 1/5.
    assert report["acceptance_rate"] >= 0.2


def test_fit_whitening_validates_its_arguments():
    p4 = np.zeros((4, 3, 4))
    with pytest.raises(ValueError, match=r"\(N, L, 4\)"):
        fit_whitening(np.zeros((4, 3)), np.full(4, 3))
    with pytest.raises(ValueError, match="n_fsps"):
        fit_whitening(p4, np.full(5, 3))
    with pytest.raises(ValueError, match="two real particles"):
        fit_whitening(p4, np.full(4, 1))


def _clustered(rng, batch=6, n_leaves=4):
    """Four-vectors whose pair second components vanish: encoding angles near {0, pi}."""
    x = np.zeros((batch, n_leaves, 4))
    x[..., 0] = rng.uniform(1.0, 2.0, (batch, n_leaves))
    x[..., 2] = rng.uniform(1.0, 2.0, (batch, n_leaves))
    x[..., 1] = x[..., 3] = 1e-6
    return jnp.asarray(x), jnp.ones((batch, n_leaves), dtype=bool)


@pytest.mark.parametrize("ansatz", ANSAETZE)
def test_closed_form_and_exact_purity_agree_in_the_clustered_limit(ansatz):
    """The one regime where the two purity observables must coincide (D78).

    Every encoding rotation tends to the identity as the angles cluster, and the
    g-purity is Ad-invariant under exp(g), so the product-state closed form and the
    real statevector agree there -- and only there. This is what licenses reading
    the clustered-limit predictions of D51 off the closed form.
    """
    model = _model(ansatz=ansatz)
    x, mask = _clustered(np.random.default_rng(SEED))

    closed = float(model.g_purity(x, mask))
    exact = model.g_purity_exact(x, mask)
    floor = {"XY_Brickwork": 0.0, "Matchgate": float(N_QUBITS), "Circuit_19": 2.0**N_QUBITS - 1}

    assert closed == pytest.approx(exact, abs=1e-4)
    assert closed == pytest.approx(floor[ansatz], abs=1e-4)


def test_g_purity_uses_the_encoded_angle_not_the_reuploaded_one():
    """The closed form describes the state entering the first trainable block (D78).

    Re-uploading multiplies the *effective* angle, but the product state the
    unflattening forms are derived for is the one at the first encoding, so the
    depth must not enter the observable. Without this the whitening acceptance test
    and the tracked series would disagree by a factor of n_layers.
    """
    x, mask = _batch(np.random.default_rng(SEED))
    purities = {n: float(_model(n_layers=n).g_purity(x, mask)) for n in (1, 2, 3)}
    assert purities[1] == pytest.approx(purities[2]) == pytest.approx(purities[3])

    expected = float(jnp.mean(G_PURITY_BY_ANSATZ["XY_Brickwork"](_model().edge_angles(x, mask))))
    assert purities[2] == pytest.approx(expected)


def test_legacy_angle_map_clusters_where_pair_polar_does_not():
    """The clustered control arm (D80), through the whole normalise-then-encode path.

    partiqlegan multiplied two unit-interval quantities. Real kinematics are mostly
    soft, so after max-normalisation both factors are small and their product
    collapses onto zero -- the barren point of the RY encoding. The pair-polar map
    instead concentrates near pi/2, which is the favourable one. Measured as
    g-purity against the whitening threshold, the two arms land on opposite sides
    of it, which is what makes the legacy arm the place the rescue prediction is
    falsifiable.
    """
    rng = np.random.default_rng(SEED)
    # Mostly-soft momenta with a hard tail, which is what max-normalisation sees.
    momenta = rng.normal(0.0, 1.0, (512, 2, 3)) * rng.gamma(2.0, 0.5, (512, 2, 1))
    energy = np.sqrt((momenta**2).sum(-1, keepdims=True) + 1.0)
    p4 = np.concatenate([momenta, energy], axis=-1)

    purity = G_PURITY_BY_ANSATZ["XY_Brickwork"]
    mu_4 = offdiag_uniform_mean(N_QUBITS)

    def edge_purity(encoding, angle_map):
        scales = normalization_scales(p4.reshape(-1, 4), encoding)
        angles = angle_map(jnp.asarray(apply_normalization(p4, scales, encoding)))
        return float(jnp.mean(purity(jnp.asarray(np.asarray(angles).reshape(-1, N_QUBITS)))))

    assert edge_purity("legacy", legacy_angles) < mu_4 / 2
    assert edge_purity("cartesian", pair_polar) > mu_4


def test_whitening_accepts_a_nested_list_so_a_checkpoint_can_carry_it():
    """The rotation is not an nnx.Param, so json is how it survives a round trip (D81)."""
    rotation = sample_rotation(np.random.default_rng(SEED))
    x, mask = _batch(np.random.default_rng(SEED))

    from_array = _model(whitening=jnp.asarray(rotation))(x, mask)
    from_list = _model(whitening=rotation.tolist())(x, mask)
    np.testing.assert_allclose(from_array, from_list, atol=1e-6)

    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        _model(whitening=np.eye(3).tolist())
