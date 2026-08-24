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
    """Accepted events must match the weighted sample, not the raw one.

    A 3-body decay has non-uniform phase-space weights, so the mean energy of a
    daughter differs between the raw sample and the weight-corrected one.  The
    unweighted sample must agree with the weighted mean and, at this statistics,
    be distinguishable from the raw mean -- which is the whole reason for
    unweighting (DECISIONS.md D12).
    """
    import tensorflow as tf

    from partiqledtr.data.generation import _build_particle, generate_events

    root = _build_particle(TOPOLOGY)
    weights, raw = root.generate(200_000, seed=tf.random.Generator.from_seed(5))
    weights = np.asarray(weights)
    energy = np.asarray(raw["a"])[:, 3]

    raw_mean = energy.mean()
    weighted_mean = float(np.average(energy, weights=weights))
    unweighted_mean = generate_events(TOPOLOGY, 40_000, seed=5)["a"][:, 3].mean()

    assert abs(unweighted_mean - weighted_mean) < 0.05
    assert abs(raw_mean - weighted_mean) > 0.1


def test_rejects_invalid_arguments():
    from partiqledtr.data.generation import generate_events

    with pytest.raises(ValueError, match="n_events"):
        generate_events(TOPOLOGY, 0, seed=0)
    with pytest.raises(ValueError, match="no children"):
        generate_events({"name": "r", "mass": 1.0, "children": []}, 10, seed=0)
