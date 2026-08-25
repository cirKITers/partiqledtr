"""Topology sampling: reproducibility, the requested FSP count and the mass budget."""

import numpy as np
import pytest

from partiqledtr.data.topology import (
    MASSES_FSP,
    MASSES_ISP,
    MAX_CHILDREN,
    MIN_CHILDREN,
    canonical_form,
    count_fsps,
    sample_topologies,
    sample_topology,
    shape_form,
    tree_depth,
)


def walk(topology):
    """Yield every node of a topology."""
    yield topology
    for child in topology["children"]:
        yield from walk(child)


def test_mass_pools_are_disjoint():
    # Particles are identified by mass alone, so a shared mass would make an ISP and an
    # FSP indistinguishable -- and canonical_form an incomplete isomorphism test.
    assert set(MASSES_ISP).isdisjoint(MASSES_FSP)


def test_sampling_is_deterministic_at_a_fixed_seed():
    def draw(seed):
        rng = np.random.default_rng(seed)
        return [canonical_form(sample_topology(rng, n_fsps=5, max_depth=4)) for _ in range(5)]

    assert draw(42) == draw(42)
    assert draw(42) != draw(43)


@pytest.mark.parametrize("n_fsps", [2, 3, 4, 5, 6, 7, 8])
def test_exact_fsp_count_is_honoured(n_fsps):
    rng = np.random.default_rng(n_fsps)
    for _ in range(20):
        assert count_fsps(sample_topology(rng, n_fsps=n_fsps, max_depth=4)) == n_fsps


@pytest.mark.parametrize("max_depth", [2, 3, 4, 5])
def test_structure_invariants(max_depth):
    rng = np.random.default_rng(11)
    # A tree of max_depth levels holds at most MAX_CHILDREN ** (max_depth - 1) leaves.
    reachable = min(8, MAX_CHILDREN ** (max_depth - 1))
    for _ in range(50):
        n_fsps = int(rng.integers(2, reachable + 1))
        topology = sample_topology(rng, n_fsps=n_fsps, max_depth=max_depth)
        assert tree_depth(topology) <= max_depth
        for node in walk(topology):
            if not node["children"]:
                assert node["mass"] in MASSES_FSP
                continue
            # phasespace requires the children to be strictly lighter than the parent.
            assert node["mass"] in MASSES_ISP
            assert len(node["children"]) >= MIN_CHILDREN
            assert sum(child["mass"] for child in node["children"]) < node["mass"]


def test_rejected_arguments():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_fsps"):
        sample_topology(rng, n_fsps=1)
    with pytest.raises(ValueError, match="max_depth"):
        sample_topology(rng, n_fsps=3, max_depth=1)
    with pytest.raises(ValueError, match="isp_weight"):
        sample_topology(rng, n_fsps=3, isp_weight=-1.0)


def test_exhausted_tries_names_the_budget():
    rng = np.random.default_rng(0)
    # A two-level tree has at most MAX_CHILDREN leaves, so 40 is unreachable.
    with pytest.raises(ValueError, match=r"n_fsps=40.*max_depth=2.*5 draws"):
        sample_topology(rng, n_fsps=40, max_depth=2, max_tries=5)


def test_canonical_form_ignores_names_and_child_order():
    left = {
        "name": "isp0",
        "mass": 100.0,
        "children": [
            {"name": "fsp0", "mass": 1.0, "children": []},
            {
                "name": "isp1",
                "mass": 10.0,
                "children": [
                    {"name": "fsp1", "mass": 2.0, "children": []},
                    {"name": "fsp2", "mass": 3.0, "children": []},
                ],
            },
        ],
    }
    right = {
        "name": "root",
        "mass": 100.0,
        "children": [
            {
                "name": "x",
                "mass": 10.0,
                "children": [
                    {"name": "z", "mass": 3.0, "children": []},
                    {"name": "y", "mass": 2.0, "children": []},
                ],
            },
            {"name": "w", "mass": 1.0, "children": []},
        ],
    }
    heavier = {**right, "mass": 90.0}

    assert canonical_form(left) == canonical_form(right) == "(100(1)(10(2)(3)))"
    assert canonical_form(left) != canonical_form(heavier)


def test_sample_topologies_groups_are_pairwise_non_isomorphic():
    groups = sample_topologies(
        np.random.default_rng(2026), n_groups=3, per_group=10, min_fsps=3, max_fsps=8
    )

    assert [len(group) for group in groups] == [10, 10, 10]
    flat = [topology for group in groups for topology in group]
    assert len({canonical_form(topology) for topology in flat}) == len(flat)
    assert {count_fsps(topology) for topology in flat} == {3, 4, 5, 6, 7, 8}
    # Shapes are distinct as *unlabelled trees*, which is what the LCAG label sees
    # (D82) -- a stronger property than the mass-labelled form asserted above.
    assert len({shape_form(topology) for topology in flat}) == len(flat)
    for group in groups:
        # Every group has to span most of the range, not just the collection as a
        # whole, so the known/unknown probe is not secretly a multiplicity probe.
        # Not *all* of it: at three leaves and max_depth=4 there are only a couple
        # of distinct shapes in existence, so a slot whose count is exhausted falls
        # through to another one (D82).
        assert len({count_fsps(topology) for topology in group}) >= 5


def test_sample_topologies_rejects_an_empty_range():
    with pytest.raises(ValueError, match="min_fsps"):
        sample_topologies(np.random.default_rng(0), min_fsps=6, max_fsps=5)
