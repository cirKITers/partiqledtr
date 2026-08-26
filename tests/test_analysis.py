import importlib.util
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest
from qml_essentials.algebra import (
    dim_so2n,
    g_purity_from_basis,
    lie_closure_paulis,
    matchgate_basis,
    matchgate_generators,
)

from partiqledtr.analysis import (
    ANSAETZE,
    G_PURITY_BY_ANSATZ,
    angle_stats,
    ansatz_generators,
    dla_basis,
    dla_check,
    g_purity_exact,
    g_purity_full,
    g_purity_offdiag,
    g_purity_su,
    offdiag_uniform_mean,
)

SEED = 20260824
# jax runs in float32 by default; the references are float64.
RTOL, ATOL = 1e-5, 1e-6

_REF_PURITY = (
    pathlib.Path(__file__).resolve().parents[1]
    / "reference/unflattening/unflattening/utils/purity.py"
)

needs_reference = pytest.mark.skipif(
    not _REF_PURITY.exists(), reason="reference/unflattening symlink not present"
)


def _load_reference():
    """The unflattening purity module, loaded from the `reference/` symlink.

    Loaded by path rather than imported: `unflattening` is a sibling research
    repo, not a declared dependency, so it is not on the path.
    """
    spec = importlib.util.spec_from_file_location("ref_purity", _REF_PURITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _closure(generators: list[str]) -> list[str]:
    """Bare Pauli strings of the Lie closure of `generators`."""
    return [word.to_pauli_string() for word in lie_closure_paulis(generators)]


def _angles(shape, rng):
    return jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, size=shape))


def _product_state(theta):
    """prod_k RY(theta_k)|0> as an explicit statevector, qubit 0 leftmost."""
    psi = np.array([1.0], dtype=complex)
    for t in np.asarray(theta):
        psi = np.kron(psi, np.array([np.cos(t / 2), np.sin(t / 2)], dtype=complex))
    return psi


# --- DLA pre-check ----------------------------------------------------------


def test_ansatz_generators_matchgate_matches_upstream():
    # Walking Matchgate.structure() must reproduce algebra.matchgate_generators
    # exactly: RZ on every qubit plus RXX on the even then odd open-chain bonds.
    for n in (2, 3, 4, 5):
        assert set(ansatz_generators("Matchgate", n)) == set(matchgate_generators(n))


@pytest.mark.parametrize(("n", "dim_g"), [(3, 15), (4, 28)])
def test_matchgate_closure_is_so2n(n, dim_g):
    # GOLDEN, derived: the matchgate closure is so(2n), dim = n(2n-1).
    assert dim_g == n * (2 * n - 1) == dim_so2n(n)
    assert dla_check("Matchgate", n_qubits=n)["dim_g"] == dim_g


def test_xy_brickwork_closure_dimension():
    # GOLDEN, derived: XY_Brickwork applies RXX then RYY on the open-chain
    # nearest-neighbour bonds (Topology.bricks, offsets 0 and 1, no wrap), i.e.
    # {X_k X_{k+1}} u {Y_k Y_{k+1}}; at n = 4 the bonds are (0,1), (2,3), (1,2).
    # Their closure is the off-diagonal algebra so(n) (+) so(n) of dimension
    # 2 * n(n-1)/2 = n(n-1) = 12, spanned by X_j Z..Z X_k and Y_j Z..Z Y_k at odd
    # separation plus X_j Z..Z Y_k and Y_j Z..Z X_k at even separation.
    assert dla_check("XY_Brickwork", n_qubits=4)["dim_g"] == 12 == 4 * 3


@pytest.mark.parametrize(("n", "dim_g"), [(2, 15), (3, 63)])
def test_circuit_19_closure_dimension(n, dim_g):
    # GOLDEN, recorded (not derived): what the installed Circuit_19 produces.
    # Sanity check: it saturates su(2**n), the "no theoretical protection" arm.
    assert dim_g == 4**n - 1
    assert dla_check("Circuit_19", n_qubits=n)["dim_g"] == dim_g


@pytest.mark.parametrize("n", [2, 3, 4])
def test_diagonal_word_counts_certify_the_floor(n):
    # The floored/floor-free certificate: Matchgate keeps the n single-qubit Z_k,
    # XY_Brickwork has no Z-only word at all.
    assert dla_check("Matchgate", n_qubits=n)["n_diag_words"] == n
    assert dla_check("XY_Brickwork", n_qubits=n)["n_diag_words"] == 0


@pytest.mark.parametrize("ansatz", ANSAETZE)
@pytest.mark.parametrize("n", [2, 3, 4])
def test_uncapped_closure_agrees_with_upstream(ansatz, n):
    """`dla_check` must report the closure of the generators it derived.

    Not a tautology: it checks that `ansatz_generators` feeds through to the
    reported dimension, and that a cap large enough to be inert stays inert.
    """
    report = dla_check(ansatz, n_qubits=n)
    assert report["dim_g"] == len(_closure(ansatz_generators(ansatz, n)))
    assert report["capped"] is False


def test_dla_check_result_shape():
    result = dla_check()
    assert result["ansatz"] == "XY_Brickwork" and result["n_qubits"] == 4
    assert set(result) == {
        "ansatz",
        "n_qubits",
        "dim_g",
        "dim_su",
        "ratio",
        "n_diag_words",
        "capped",
    }
    assert isinstance(result["dim_g"], int)
    assert isinstance(result["dim_su"], int)
    assert isinstance(result["ratio"], float)
    assert isinstance(result["n_diag_words"], int)
    assert isinstance(result["capped"], bool)
    assert result["capped"] is False
    assert result["dim_su"] == 4**4 - 1
    assert result["ratio"] == pytest.approx(result["dim_g"] / result["dim_su"])


def test_dla_check_respects_max_dim():
    capped = dla_check("Circuit_19", n_qubits=3, max_dim=20)
    dim_g = capped["dim_g"]
    assert capped["capped"] is True
    assert isinstance(dim_g, int) and 20 <= dim_g < 63
    assert dla_check("Circuit_19", n_qubits=3, max_dim=200)["capped"] is False


def test_dla_check_validates_arguments():
    with pytest.raises(ValueError, match="unknown ansatz"):
        dla_check("Circuit_42")
    with pytest.raises(ValueError, match="n_qubits"):
        dla_check("Matchgate", n_qubits=1)
    with pytest.raises(ValueError, match="max_dim"):
        dla_check("Matchgate", n_qubits=4, max_dim=3)


# --- g-purity closed forms --------------------------------------------------


@needs_reference
@pytest.mark.parametrize("n", [2, 3, 4, 7])
def test_purity_matches_reference_implementation(n):
    rng = np.random.default_rng(SEED)
    theta = _angles((11, 5, n), rng)
    ref = _load_reference()
    np.testing.assert_allclose(
        np.asarray(g_purity_full(theta)), ref.g_purity_closed_form(theta), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        np.asarray(g_purity_offdiag(theta)), ref.offdiag_closed_form(theta), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("n", [2, 3, 4, 7])
def test_purity_matches_naive_double_sum(n):
    # Independent transcription of the two closed forms as naive O(n**2) double
    # sums, so the O(n) recurrences are checked even without the reference repo.
    rng = np.random.default_rng(SEED + 1)
    theta = np.asarray(_angles((6, n), rng))
    s, c = np.sin(theta) ** 2, np.cos(theta) ** 2
    cross = np.zeros(theta.shape[:-1])
    odd = np.zeros(theta.shape[:-1])
    for j in range(n):
        for k in range(j + 1, n):
            term = s[..., j] * s[..., k] * np.prod(c[..., j + 1 : k], axis=-1)
            cross = cross + term
            if (k - j) % 2 == 1:
                odd = odd + term
    np.testing.assert_allclose(
        np.asarray(g_purity_full(jnp.asarray(theta))), c.sum(-1) + cross, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        np.asarray(g_purity_offdiag(jnp.asarray(theta))), odd, rtol=RTOL, atol=ATOL
    )


def test_purity_matches_g_purity_from_basis_on_statevector():
    # The closed forms against sum_B <B>**2 on the explicit statevector, for the
    # DLA basis each one belongs to.
    n = 4
    rng = np.random.default_rng(SEED + 2)
    xy_basis = _closure(ansatz_generators("XY_Brickwork", n))
    mg_basis = matchgate_basis(n)
    assert len(xy_basis) == 12 and len(mg_basis) == dim_so2n(n)
    for _ in range(6):
        theta = _angles((n,), rng)
        psi = _product_state(theta)
        assert float(g_purity_offdiag(theta)) == pytest.approx(
            g_purity_from_basis(psi, xy_basis), rel=RTOL, abs=ATOL
        )
        assert float(g_purity_full(theta)) == pytest.approx(
            g_purity_from_basis(psi, mg_basis), rel=RTOL, abs=ATOL
        )


def test_su_purity_is_input_independent():
    # Circuit_19 saturates su(2**n), whose g-purity is 2**n - 1 for any pure state.
    n = 3
    rng = np.random.default_rng(SEED + 3)
    basis = _closure(ansatz_generators("Circuit_19", n))
    assert len(basis) == 4**n - 1
    theta = _angles((5, n), rng)
    np.testing.assert_allclose(np.asarray(g_purity_su(theta)), np.full(5, 2**n - 1.0))
    for row in theta:
        assert g_purity_from_basis(_product_state(row), basis) == pytest.approx(2**n - 1.0)


def test_purity_batches_over_leading_axes():
    rng = np.random.default_rng(SEED + 4)
    theta = _angles((3, 5, 4), rng)
    for purity in (g_purity_offdiag, g_purity_full, g_purity_su):
        assert purity(theta).shape == (3, 5)
        assert float(purity(theta)[2, 1]) == pytest.approx(
            float(purity(theta[2, 1])), rel=RTOL, abs=ATOL
        )


def test_g_purity_by_ansatz_covers_every_arm():
    assert set(G_PURITY_BY_ANSATZ) == set(ANSAETZE)


@needs_reference
@pytest.mark.parametrize("n", [2, 4, 6, 10])
def test_offdiag_uniform_mean_matches_reference(n):
    assert offdiag_uniform_mean(n) == pytest.approx(_load_reference().offdiag_uniform_mean(n))


def test_offdiag_uniform_mean_approaches_asymptote():
    assert offdiag_uniform_mean(4) == pytest.approx(0.8125)
    for n in (30, 100):
        assert offdiag_uniform_mean(n) == pytest.approx(n / 3 - 5 / 9, abs=1e-6)


# --- the physics the whole module exists for --------------------------------


def test_clustered_angles_collapse_the_floor_free_purity():
    # The unflattening prediction, and the assertion that catches a wrong formula:
    # clustered angles annihilate the XY_Brickwork purity (well below the mu_n / 2
    # acceptance threshold) while uniform angles sit at mu_n; the floored Matchgate
    # arm instead sits at its floor n, above its own uniform mean n - 1 + 2**-n.
    n = 4
    rng = np.random.default_rng(SEED + 5)
    mu = offdiag_uniform_mean(n)

    uniform = _angles((20000, n), rng)
    assert float(g_purity_offdiag(uniform).mean()) == pytest.approx(mu, rel=0.05)
    assert float(g_purity_full(uniform).mean()) == pytest.approx(n - 1 + 2.0**-n, rel=0.05)

    clustered = jnp.asarray(rng.normal(0.0, 0.05, size=(2000, n)))
    assert float(g_purity_offdiag(clustered).max()) < mu / 2
    assert float(g_purity_offdiag(clustered).mean()) < mu / 100
    assert float(g_purity_full(clustered).min()) > 0.9 * n


@pytest.mark.parametrize("ansatz", ANSAETZE)
def test_cap_truncates_and_says_so(ansatz):
    """A cap stops the closure early and marks `dim_g` as a lower bound.

    This is the whole reason `max_dim` exists: Circuit_19 saturates `su(2**n)`, so
    an uncapped closure enumerates `4**n - 1` words and takes over a minute at
    n=6. A capped result must be exactly `max_dim` words with `capped` set, so a
    reader knows not to treat `dim_g` as the true dimension.
    """
    dim_g = int(dla_check(ansatz, n_qubits=4)["dim_g"])
    cap = len(ansatz_generators(ansatz, 4)) + 1
    assert cap < dim_g, "the cap has to bite for this to test anything"

    capped = dla_check(ansatz, n_qubits=4, max_dim=cap)
    assert capped["capped"] is True
    assert capped["dim_g"] == cap  # a lower bound, not the dimension

    with pytest.raises(ValueError, match="max_dim"):
        dla_check(ansatz, n_qubits=4, max_dim=1)


@pytest.mark.parametrize(
    ("ansatz", "expected"),
    [("XY_Brickwork", 0), ("Matchgate", 4), ("Circuit_19", 15)],
)
def test_diagonal_word_count_at_the_constellation_size(ansatz, expected):
    """The floored/floor-free certificate at the size actually used, n_qubits=4 (D52).

    Circuit_19's 2**n - 1 was the one documented value with no test behind it, and
    it is the arm whose whole reading -- input-distribution independent, no front
    end can help -- rests on saturating su(2**n).
    """
    record = dla_check(ansatz, n_qubits=4)
    assert record["n_diag_words"] == expected
    assert not record["capped"]


def test_exact_purity_matches_the_closed_form_on_a_product_state():
    """The two purity routes have to agree where they describe the same state (D78).

    With no ansatz in the way, the encoded state *is* the RY product state, so the
    closed form and the sum over the DLA basis must coincide -- which is what makes
    the exact route a check on the closed one rather than a second guess.
    """
    rng = np.random.default_rng(5)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(6, 4))
    # prod_q RY(theta_q)|0>, built directly: cos(theta/2)|0> + sin(theta/2)|1>.
    states = np.ones((len(angles), 1), dtype=complex)
    for qubit in range(4):
        half = angles[:, qubit, None] / 2.0
        states = np.concatenate([states * np.cos(half), states * np.sin(half)], axis=1)

    for ansatz, closed in G_PURITY_BY_ANSATZ.items():
        exact = g_purity_exact(states, dla_basis(ansatz, 4))
        np.testing.assert_allclose(exact, np.asarray(closed(jnp.asarray(angles))), atol=1e-6)


def test_angle_stats_separates_uniform_pinned_and_clustered():
    """The discriminator the g-purity cannot provide (D92).

    A purity rises both when angles spread toward uniform and when they pin near
    pi/2 -- the true maximum, and the configuration that destroys the input
    information. Both look identical in total variation (~0.94 here), so
    ``mean_sin2`` is what tells them apart: 0.5 uniform, 1 pinned, 0 clustered.
    """
    rng = np.random.default_rng(0)
    uniform = rng.uniform(0.0, 2.0 * np.pi, (4000, 4))
    pinned = np.pi / 2 + rng.normal(0.0, 0.05, (4000, 4))
    clustered = rng.normal(0.0, 0.05, (4000, 4))

    stats = {
        name: angle_stats(a)
        for name, a in (("uniform", uniform), ("pinned", pinned), ("clustered", clustered))
    }

    assert stats["uniform"]["tv_uniform"][0] < 0.1
    assert stats["pinned"]["tv_uniform"][0] > 0.8
    assert stats["clustered"]["tv_uniform"][0] > 0.8
    # ... and the two high-TV laws are opposite in what they do to the input.
    assert stats["uniform"]["mean_sin2"][0] == pytest.approx(0.5, abs=0.05)
    assert stats["pinned"]["mean_sin2"][0] > 0.95
    assert stats["clustered"]["mean_sin2"][0] < 0.05


def test_angle_stats_is_per_site_not_pooled():
    """Sites peaking at different angles must not average into a flat-looking law.

    This is the artefact the unflattening latent-drift memo warns about, and the
    reason the record is a list per qubit rather than one number.
    """
    rng = np.random.default_rng(1)
    # Two sites, each sharply peaked, but at different places.
    angles = np.stack([rng.normal(0.5, 0.05, 4000), rng.normal(4.0, 0.05, 4000)], axis=-1)
    stats = angle_stats(angles)

    assert len(stats["tv_uniform"]) == 2
    assert all(tv > 0.8 for tv in stats["tv_uniform"])
    # Pooled, the same samples would look far closer to uniform than either site is.
    pooled = angle_stats(angles.reshape(-1, 1))
    assert pooled["tv_uniform"][0] < min(stats["tv_uniform"])


def test_angle_stats_rejects_malformed_input():
    with pytest.raises(ValueError, match="n_qubits"):
        angle_stats(np.zeros(10))
    with pytest.raises(ValueError, match="no angles"):
        angle_stats(np.zeros((0, 4)))
