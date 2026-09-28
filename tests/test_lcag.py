"""LCAG construction and reconstruction.

Golden matrices are transcribed from baumbauen's own fixtures
(``reference/baumbauen/tests/test_decay2lca.py`` and ``test_lca2adjacency.py``), rebuilt on
our dict topologies. Where the reference gives no fixture the expectation is derived from
the algorithm in a comment above it.
"""

import numpy as np
import pytest

from partiqledtr.data.lcag import (
    InvalidLCAGError,
    is_valid_lcag,
    lcag_roundtrip,
    lcag_to_adjacency,
    shuffle_leaves,
    topology_to_lcag,
)
from partiqledtr.data.topology import sample_topology


def node(name, *children, mass=1.0):
    return {"name": name, "mass": mass, "children": list(children)}


# baumbauen test_decay2lca.test_valid_bfs_example_1: a -> {b -> {d, e, f}, c -> {g, h}}.
EXAMPLE_1 = node(
    "a",
    node("b", node("d"), node("e"), node("f")),
    node("c", node("g"), node("h")),
)
EXAMPLE_1_LCAG = [
    # d  e  f  g  h
    [0, 1, 1, 2, 2],  # d
    [1, 0, 1, 2, 2],  # e
    [1, 1, 0, 2, 2],  # f
    [2, 2, 2, 0, 1],  # g
    [2, 2, 2, 1, 0],  # h
]

# Skipped generation: h hangs directly off the root while d and e sit two levels deeper,
# so the raw generations (h: 1, f/g: 2, d/e: 3) are not all equal on the leaf plane.
# Pull-down puts every leaf on the deepest leaf level (3) and every interior node one
# level above its highest child, which is the same as reading off the height of the
# lowest common ancestor:
#   height(c) = 1        -> (d, e)                     = 1
#   height(b) = 2        -> (d, f), (d, g), (f, g)     = 2
#   height(a) = 3        -> every pair involving h     = 3
# baumbauen's test_decay2lca.test_valid_dfs_example_4 lists the same matrix.
EXAMPLE_4 = node(
    "a",
    node("b", node("c", node("d"), node("e")), node("f"), node("g")),
    node("h"),
)
EXAMPLE_4_LCAG = [
    # d  e  f  g  h
    [0, 1, 2, 2, 3],  # d
    [1, 0, 2, 2, 3],  # e
    [2, 2, 0, 2, 3],  # f
    [2, 2, 2, 0, 3],  # g
    [3, 3, 3, 3, 0],  # h
]

# Two skipped generations, one per branch: g and m are leaves one level above their
# siblings' subtrees. Heights: e = f = 1, c = d = 2, b = 3, a = 4, hence
#   (h, i) = height(e) = 1, (k, l) = height(f) = 1,
#   (g, h) = height(c) = 2, (j, k) = height(d) = 2,
#   (j, m) = height(b) = 3, anything across the two branches = height(a) = 4.
# baumbauen's test_decay2lca.test_valid_example_6 lists the same matrix.
EXAMPLE_6 = node(
    "a",
    node("c", node("g"), node("e", node("h"), node("i"))),
    node("b", node("d", node("j"), node("f", node("k"), node("l"))), node("m")),
)
EXAMPLE_6_LCAG = [
    # g  h  i  j  k  l  m
    [0, 2, 2, 4, 4, 4, 4],  # g
    [2, 0, 1, 4, 4, 4, 4],  # h
    [2, 1, 0, 4, 4, 4, 4],  # i
    [4, 4, 4, 0, 2, 2, 3],  # j
    [4, 4, 4, 2, 0, 1, 3],  # k
    [4, 4, 4, 2, 1, 0, 3],  # l
    [4, 4, 4, 3, 3, 3, 0],  # m
]


@pytest.mark.parametrize(
    ("topology", "expected", "names"),
    [
        (EXAMPLE_1, EXAMPLE_1_LCAG, ["d", "e", "f", "g", "h"]),
        (EXAMPLE_4, EXAMPLE_4_LCAG, ["d", "e", "f", "g", "h"]),
        (EXAMPLE_6, EXAMPLE_6_LCAG, ["g", "h", "i", "j", "k", "l", "m"]),
    ],
)
def test_topology_to_lcag_golden(topology, expected, names):
    lcag, leaf_names = topology_to_lcag(topology)
    assert leaf_names == names
    assert lcag.dtype == np.int8
    np.testing.assert_array_equal(lcag, np.array(expected, dtype=np.int8))


def test_topology_to_lcag_flat_decay():
    # baumbauen test_decay2lca.test_valid_bfs_example_3: every leaf is a direct child.
    lcag, names = topology_to_lcag(node("a", node("b"), node("c"), node("d")))
    assert names == ["b", "c", "d"]
    np.testing.assert_array_equal(lcag, [[0, 1, 1], [1, 0, 1], [1, 1, 0]])


def test_topology_to_lcag_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        topology_to_lcag(node("a", node("b"), node("b")))


def test_lcag_to_adjacency_golden():
    # Same tree as EXAMPLE_1. Rows 0..4 are the leaves d, e, f, g, h in input order,
    # then the internal nodes breadth-first from the root: 5 = a, 6 = b, 7 = c.
    expected = [
        # d  e  f  g  h  a  b  c
        [0, 0, 0, 0, 0, 0, 1, 0],  # d
        [0, 0, 0, 0, 0, 0, 1, 0],  # e
        [0, 0, 0, 0, 0, 0, 1, 0],  # f
        [0, 0, 0, 0, 0, 0, 0, 1],  # g
        [0, 0, 0, 0, 0, 0, 0, 1],  # h
        [0, 0, 0, 0, 0, 0, 1, 1],  # a
        [1, 1, 1, 0, 0, 1, 0, 0],  # b
        [0, 0, 0, 1, 1, 1, 0, 0],  # c
    ]
    np.testing.assert_array_equal(lcag_to_adjacency(np.array(EXAMPLE_1_LCAG)), expected)


def test_lcag_to_adjacency_skips_missing_levels():
    # baumbauen test_lca2adjacency.test_valid_shuffled_bfs_example_2: levels 2 and 5 encode
    # the same tree as levels 1 and 2, because only the ordering of the values matters.
    golden = np.array(EXAMPLE_1_LCAG)
    disjoint = golden * 3 + 2 * (golden > 0)  # off-diagonal values 5 and 8, no 1 or 2
    np.testing.assert_array_equal(
        lcag_to_adjacency(disjoint), lcag_to_adjacency(np.array(EXAMPLE_1_LCAG))
    )


@pytest.mark.parametrize(
    ("name", "matrix"),
    [
        ("asymmetric", [[0, 1, 2], [1, 0, 2], [1, 2, 0]]),
        ("negative off-diagonal", [[0, -1, 2], [-1, 0, 2], [2, 2, 0]]),
        ("zero off-diagonal", [[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        # Contradictory levels: a and c meet at level 1 and b and c meet at level 1, so a
        # and b cannot first meet at level 2 (baumbauen test_illegal_connections_1).
        ("contradictory levels", [[0, 2, 1], [2, 0, 1], [1, 1, 0]]),
        ("not square", [[0, 1, 1], [1, 0, 1]]),
        ("not 2-d", [[[0, 1], [1, 0]]]),
    ],
)
def test_lcag_to_adjacency_rejects(name, matrix):
    with pytest.raises(InvalidLCAGError):
        lcag_to_adjacency(np.array(matrix))
    assert not is_valid_lcag(np.array(matrix))


def test_is_valid_lcag_accepts_golden():
    assert is_valid_lcag(np.array(EXAMPLE_1_LCAG))
    assert is_valid_lcag(np.array(EXAMPLE_6_LCAG))


def test_lcag_to_adjacency_ignores_the_diagonal():
    # pad_events writes -1 on the diagonal and models predict the ignore label
    # there, so the diagonal must not decide validity.
    ignored = np.array(EXAMPLE_1_LCAG)
    np.fill_diagonal(ignored, -1)
    np.testing.assert_array_equal(
        lcag_to_adjacency(ignored), lcag_to_adjacency(np.array(EXAMPLE_1_LCAG))
    )


def test_is_valid_lcag_does_not_swallow_other_errors():
    # A non-numeric matrix is a caller bug, not an invalid LCAG: the TypeError numpy
    # raises has to surface instead of being reported as a merely invalid matrix.
    with pytest.raises(TypeError):
        is_valid_lcag(np.array([["0", "1"], ["1", "0"]]))


def test_shuffle_leaves_conjugates_the_lcag():
    rng = np.random.default_rng(20260824)
    lcag = np.array(EXAMPLE_6_LCAG, dtype=np.int8)
    # Row index as the feature, so the drawn permutation can be read back off the result.
    features = np.arange(lcag.shape[0], dtype=float)[:, None]

    shuffled_features, shuffled_lcag = shuffle_leaves(rng, features, lcag)

    perm = shuffled_features[:, 0].astype(int)
    assert sorted(perm.tolist()) == list(range(lcag.shape[0]))
    np.testing.assert_array_equal(shuffled_lcag, lcag[perm][:, perm])
    np.testing.assert_array_equal(shuffled_lcag, shuffled_lcag.T)
    np.testing.assert_array_equal(np.diag(shuffled_lcag), 0)


def test_shuffle_leaves_checks_shapes():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="rows"):
        shuffle_leaves(rng, np.zeros((3, 2)), np.zeros((5, 5)))
    with pytest.raises(ValueError, match="square"):
        shuffle_leaves(rng, np.zeros((3, 2)), np.zeros((3, 4)))


def test_reordering_children_conjugates_the_lcag():
    """Relabelling leaves must permute the LCAG, not change it."""
    lcag, names = topology_to_lcag(EXAMPLE_6)
    mirrored = _mirror(EXAMPLE_6)
    mirrored_lcag, mirrored_names = topology_to_lcag(mirrored)

    perm = np.array([names.index(name) for name in mirrored_names])
    np.testing.assert_array_equal(mirrored_lcag, lcag[perm][:, perm])


def test_round_trip_through_adjacency():
    rng = np.random.default_rng(7)
    for _ in range(50):
        topology = sample_topology(rng, n_fsps=int(rng.integers(3, 9)), max_depth=4)
        lcag, names = topology_to_lcag(topology)
        n_leaves = len(names)
        # Leaves reach the model shuffled, so reconstruct from a shuffled matrix too.
        _, lcag = shuffle_leaves(rng, np.zeros((n_leaves, 1)), lcag)
        adjacency = lcag_to_adjacency(lcag)

        assert adjacency.shape[0] > n_leaves
        # Leaves keep their input rows and have exactly one neighbour.
        np.testing.assert_array_equal(adjacency[:n_leaves].sum(axis=1), 1)
        np.testing.assert_array_equal(_lcag_from_adjacency(adjacency, n_leaves), lcag)


def test_round_trip_baumbauen_shuffled_fixture():
    # baumbauen test_lca2adjacency.test_valid_shuffled_bfs_example_1: EXAMPLE_1 with the
    # leaves ordered d, g, f, h, e.
    shuffled = np.array(
        [
            # d  g  f  h  e
            [0, 2, 1, 2, 1],  # d
            [2, 0, 2, 1, 2],  # g
            [1, 2, 0, 2, 1],  # f
            [2, 1, 2, 0, 2],  # h
            [1, 2, 1, 2, 0],  # e
        ]
    )
    adjacency = lcag_to_adjacency(shuffled)

    assert adjacency.shape == (8, 8)
    np.testing.assert_array_equal(_lcag_from_adjacency(adjacency, 5), shuffled)


def _mirror(topology):
    """Return the topology with every child list reversed."""
    return {
        **topology,
        "children": [_mirror(child) for child in reversed(topology["children"])],
    }


def _lcag_from_adjacency(adjacency, n_leaves):
    """Rebuild the LCAG from an adjacency matrix, independently of lcag.py.

    Uses the documented node order: rows ``0..n_leaves-1`` are the leaves, row
    ``n_leaves`` is the root.
    """
    n_nodes = adjacency.shape[0]
    parent = [-1] * n_nodes
    children = [[] for _ in range(n_nodes)]
    order = [n_leaves]
    head = 0
    while head < len(order):
        current = order[head]
        head += 1
        for neighbour in np.flatnonzero(adjacency[current]).tolist():
            if neighbour != parent[current]:
                parent[neighbour] = current
                children[current].append(neighbour)
                order.append(neighbour)
    assert len(order) == n_nodes

    heights = [0] * n_nodes
    for current in reversed(order):
        if children[current]:
            heights[current] = 1 + max(heights[child] for child in children[current])

    chains = []
    for leaf in range(n_leaves):
        chain = [leaf]
        while parent[chain[-1]] >= 0:
            chain.append(parent[chain[-1]])
        chains.append(chain[::-1])

    lcag = np.zeros((n_leaves, n_leaves), dtype=np.int8)
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            shared = 0
            for a, b in zip(chains[i], chains[j], strict=False):
                if a != b:
                    break
                shared = a
            lcag[i, j] = lcag[j, i] = heights[shared]
    return lcag


def test_lcag_roundtrip_reproduces_a_real_lcag_and_exposes_a_greedy_acceptance():
    """The strict test the greedy reconstruction is not.

    A genuine LCAG re-derives itself exactly. A matrix the greedy reconstruction
    accepts need not: it reduces to *a* tree, but not to one that would produce it
    back, which is the loophole the lenient valid-tree rate leaves open.
    """
    lcag = topology_to_lcag(sample_topology(np.random.default_rng(11), n_fsps=6))[0].astype(int)
    np.testing.assert_array_equal(lcag_roundtrip(lcag), lcag)

    # Three leaves pairwise claiming different ancestors: greedy accepts, strict does not.
    inconsistent = np.array([[0, 1, 2, 2], [1, 0, 2, 2], [2, 2, 0, 1], [2, 2, 1, 0]])
    inconsistent[0, 2] = inconsistent[2, 0] = 1
    if is_valid_lcag(inconsistent):
        assert not np.array_equal(lcag_roundtrip(inconsistent), inconsistent)


def test_lcag_to_adjacency_rejects_a_single_leaf():
    """One leaf is no decay: there is no pair and no ancestor to reconstruct."""
    with pytest.raises(InvalidLCAGError, match="at least two leaves"):
        lcag_to_adjacency(np.array([[0]]))
