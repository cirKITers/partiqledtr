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
    """Check accepted events match the weighted energy distribution.

    The raw sample differs because multi-body phase-space weights vary. Compare
    means against the sample's standard error.
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
    """Check generation returns NumPy arrays across the float64 JAX boundary.

    This prevents scoped ``jax.enable_x64()`` arrays from entering a float32 JAX
    computation and being silently truncated.
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


def test_generation_gives_up_on_a_budget_rather_than_grinding():
    """Check generation raises after its bounded draw budget is exhausted.

    Near-threshold decays can reject almost every draw; a caller must be able to
    sample another topology.
    """
    from partiqledtr.data.generation import UngeneratableTopologyError, generate_events

    with pytest.raises(UngeneratableTopologyError, match="acceptance rate"):
        generate_events(TOPOLOGY, 5000, seed=0, max_draws=2000)

    # ... and the same request finishes when the budget allows it.
    events = generate_events(TOPOLOGY, 50, seed=0, max_draws=2_000_000)
    assert len(next(iter(events.values()))) == 50


def test_viability_probe_keeps_generation_finishing():
    """assemble_dataset must not hand the generator a topology it cannot sample.

    Before the probe this raised part-way through a dataset, after minutes of work.
    """
    from partiqledtr.data.dataset import assemble_dataset

    splits, meta = assemble_dataset(
        seed=0, n_topologies=4, n_events_per_topology=40, min_fsps=3, max_fsps=6
    )
    assert all(len(split["lcag"]) > 0 for split in splits.values())
    assert meta["counts"]["train"] > 0
