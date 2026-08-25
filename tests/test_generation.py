"""Event generation: conservation laws, determinism and correct unweighting."""

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.gen

# Nested decay: r -> (i0 -> a b) c d.  Three generations, four final-state particles.
TOPOLOGY: dict[str, Any] = {
    "name": "r",
    "mass": 100.0,
    "children": [
        {
            "name": "i0",
            "mass": 50.0,
            "children": [
                {"name": "a", "mass": 5.0, "children": []},
                {"name": "b", "mass": 3.0, "children": []},
            ],
        },
        {"name": "c", "mass": 12.0, "children": []},
        {"name": "d", "mass": 2.0, "children": []},
    ],
}
MASSES = {"a": 5.0, "b": 3.0, "c": 12.0, "d": 2.0}


def test_leaf_names_are_depth_first():
    from partiqledtr.data.generation import leaf_names

    assert leaf_names(TOPOLOGY) == ["a", "b", "c", "d"]


def test_events_conserve_energy_momentum_and_mass():
    from partiqledtr.data.generation import generate_events

    events = generate_events(TOPOLOGY, 400, seed=11)

    assert sorted(events) == ["a", "b", "c", "d"]
    for array in events.values():
        assert array.shape == (400, 4)

    total = np.sum(list(events.values()), axis=0)
    # Generated in the root rest frame: energies sum to the root mass, momenta cancel.
    assert np.allclose(total[:, 3], TOPOLOGY["mass"])
    assert np.allclose(total[:, :3], 0.0, atol=1e-6)

    for name, mass in MASSES.items():
        p = events[name]
        invariant = np.sqrt(np.abs(p[:, 3] ** 2 - (p[:, :3] ** 2).sum(axis=1)))
        assert np.allclose(invariant, mass, atol=1e-5)


def test_generation_is_deterministic_and_seed_sensitive():
    from partiqledtr.data.generation import generate_events

    a = generate_events(TOPOLOGY, 200, seed=11)
    b = generate_events(TOPOLOGY, 200, seed=11)
    c = generate_events(TOPOLOGY, 200, seed=12)

    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["a"], c["a"])


def test_unweighting_reproduces_the_weighted_distribution():
    """Accepted events must follow the weighted sample, not the raw one.

    A multi-body decay has non-uniform phase-space weights, so the mean energy of
    a daughter differs between the raw sample and the weight-corrected one. The
    unweighted sample must agree with the weighted mean and be clearly separated
    from the raw one -- which is the whole reason for unweighting (D12).

    The tolerance is the sample's own standard error rather than a fixed number.
    Measured at 100k events: the unweighted mean lands 0.01-0.05 from the weighted
    one against a 4-sigma bound of 0.097, while the raw sample sits 0.202 away --
    comfortably outside the 5-sigma bound of 0.121. An earlier fixed tolerance of
    0.05 was about 1.2 sigma and passed only by luck.
    """
    from partiqledtr.data.generation import _build_particle, generate_events

    root = _build_particle(TOPOLOGY)
    drawn = root.generate(200_000, key=5)
    weights = np.asarray(drawn[0])
    energy = np.asarray(drawn[-1]["a"])[:, 3]
    weighted_mean = float(np.average(energy, weights=weights))

    sample = generate_events(TOPOLOGY, 100_000, seed=5)["a"][:, 3]
    standard_error = sample.std() / np.sqrt(sample.size)

    assert abs(sample.mean() - weighted_mean) < 4 * standard_error
    assert abs(energy.mean() - weighted_mean) > 5 * standard_error


def test_output_crosses_the_jax_boundary_as_numpy():
    """`generate_events` must hand back numpy, not phasespace's float64 JAX arrays.

    phasespace runs its kinematics under a scoped `jax.enable_x64()` and returns
    float64 arrays even though the calling program is float32. Combining those
    directly with a float32 JAX array warns and silently truncates::

        f64 + jnp.zeros(4, jnp.float32)   -> UserWarning, result float32

    Converting through numpy is the clean boundary and is what this function does,
    so downstream code never meets a stray float64 JAX array. A future edit that
    returned the arrays as-is would only warn, not fail, hence this guard.
    """
    import warnings

    import jax.numpy as jnp

    from partiqledtr.data.generation import generate_events

    events = generate_events(TOPOLOGY, 32, seed=0)

    for array in events.values():
        assert isinstance(array, np.ndarray)
        assert not isinstance(array, jnp.ndarray)
        assert array.dtype == np.float64  # full precision survives the crossing

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stacked = jnp.asarray(np.stack(list(events.values()), axis=1))
    assert stacked.dtype == jnp.float32  # the training path's dtype, no warning


def test_rejects_invalid_arguments():
    from partiqledtr.data.generation import generate_events

    with pytest.raises(ValueError, match="n_events"):
        generate_events(TOPOLOGY, 0, seed=0)
    with pytest.raises(ValueError, match="no children"):
        generate_events({"name": "r", "mass": 1.0, "children": []}, 10, seed=0)
