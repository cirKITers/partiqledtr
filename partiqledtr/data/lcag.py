"""Build LCAG matrices and reconstruct decay-tree adjacency.

An LCAG entry is the height of the lowest common ancestor of two leaves;
the diagonal is zero. Reconstruction keeps input leaves in their original
order, places the root at row ``L``, and rejects inconsistent ancestry. Matrix
construction reproduces baumbauen's ``decay2lca`` using node heights.
"""

import numpy as np


class InvalidLCAGError(ValueError):
    """Raised for a malformed LCAG matrix or one that does not encode a tree."""


def topology_to_lcag(topology: dict) -> tuple[np.ndarray, list[str]]:
    """Build the LCAG matrix of a topology.

    Args:
        topology: Nested ``{"name", "mass", "children"}`` dict, see
            :mod:`partiqledtr.data.topology`.

    Returns:
        The ``(L, L)`` int8 LCAG matrix and the ``L`` leaf names, both in the tree's own
        depth-first order.

    Raises:
        ValueError: If two nodes share a name.
    """
    names, parents, children = _walk(topology)
    if len(set(names)) != len(names):
        raise ValueError("node names have to be unique within a topology")

    heights = [0] * len(names)
    for idx in range(len(names) - 1, -1, -1):  # depth-first order: children come after
        if children[idx]:
            heights[idx] = 1 + max(heights[c] for c in children[idx])

    leaves = [idx for idx, kids in enumerate(children) if not kids]
    chains = [_ancestry(idx, parents) for idx in leaves]

    lcag = np.zeros((len(leaves), len(leaves)), dtype=np.int8)
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            lcag[i, j] = lcag[j, i] = heights[_lowest_common(chains[i], chains[j])]
    return lcag, [names[idx] for idx in leaves]


def _walk(topology: dict) -> tuple[list[str], list[int], list[list[int]]]:
    """Index the nodes depth-first; return their names, parents and children."""
    names: list[str] = []
    parents: list[int] = []
    children: list[list[int]] = []
    stack = [(topology, -1)]
    while stack:
        node, parent = stack.pop()
        idx = len(names)
        names.append(node["name"])
        parents.append(parent)
        children.append([])
        if parent >= 0:
            children[parent].append(idx)
        stack.extend((child, idx) for child in reversed(node["children"]))
    return names, parents, children


def _ancestry(idx: int, parents: list[int]) -> list[int]:
    """Return the node indices from the root down to ``idx``."""
    chain = [idx]
    while parents[chain[-1]] >= 0:
        chain.append(parents[chain[-1]])
    chain.reverse()
    return chain


def _lowest_common(chain_a: list[int], chain_b: list[int]) -> int:
    """Return the last node shared by two root-down ancestries."""
    common = 0
    for node_a, node_b in zip(chain_a, chain_b, strict=False):
        if node_a != node_b:
            break
        common = node_a
    return common


def shuffle_leaves(
    rng: np.random.Generator, features: np.ndarray, lcag: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one random leaf permutation to a feature matrix and its LCAG.

    The LCAG is conjugated, ``P L P^T``, so that its rows keep tracking the feature rows.

    Args:
        rng: Random generator; the only source of randomness.
        features: ``(L, F)`` features of the leaves.
        lcag: ``(L, L)`` LCAG matrix.

    Returns:
        The permuted features and the conjugated LCAG.

    Raises:
        ValueError: If ``lcag`` is not square or its size differs from the leaf count.
    """
    if lcag.ndim != 2 or lcag.shape[0] != lcag.shape[1]:
        raise ValueError(f"lcag must be square, got shape {lcag.shape}")
    if features.shape[0] != lcag.shape[0]:
        raise ValueError(f"features has {features.shape[0]} rows but lcag has {lcag.shape[0]}")
    perm = rng.permutation(lcag.shape[0])
    return features[perm], lcag[perm][:, perm]


def lcag_to_adjacency(lcag: np.ndarray) -> np.ndarray:
    """Reconstruct a decay tree from an LCAG matrix.

    Rank distinct off-diagonal values as ancestor levels; ignore the diagonal.

    Args:
        lcag: ``(L, L)`` LCAG matrix.

    Returns:
        The ``(N, N)`` int8 adjacency matrix over all ``N`` nodes: rows ``0..L-1`` are the
        leaves in input order, rows ``L..N-1`` the reconstructed internal nodes
        breadth-first from the root, so row ``L`` is the root.

    Raises:
        InvalidLCAGError: If the matrix is not a square 2-d matrix, is asymmetric, has an
            off-diagonal entry below 1, is internally inconsistent, or if the
            reconstruction is not a tree.
    """
    lcag = np.asarray(lcag)
    if lcag.ndim != 2 or lcag.shape[0] != lcag.shape[1]:
        raise InvalidLCAGError(f"lcag must be a square 2-d matrix, got shape {lcag.shape}")
    n_leaves = lcag.shape[0]
    if n_leaves < 2:
        # One leaf is no decay: there is no pair, hence no ancestor to reconstruct,
        # and the promised "row L is the root" would not hold.
        raise InvalidLCAGError(
            f"lcag needs at least two leaves to describe a decay, got {n_leaves}"
        )
    if not np.array_equal(lcag, lcag.T):
        raise InvalidLCAGError("lcag must be symmetric")
    off_diagonal = ~np.eye(n_leaves, dtype=bool)
    if np.any(lcag[off_diagonal] < 1):
        raise InvalidLCAGError(
            "off-diagonal entries must be at least 1: every leaf pair needs a common ancestor"
        )

    level, parent, children = _reconstruct(lcag)
    root = _ancestor(0, parent)
    order = _bfs(root, children)
    if len(order) != len(level):
        raise InvalidLCAGError("reconstruction is disconnected: not every node reaches the root")

    # Leaves keep their input row, internal nodes follow breadth-first from the root.
    index = {leaf: leaf for leaf in range(n_leaves)}
    index.update(
        (node, n_leaves + rank)
        for rank, node in enumerate(node for node in order if node >= n_leaves)
    )
    adjacency = np.zeros((len(order), len(order)), dtype=np.int8)
    for node, kids in enumerate(children):
        for kid in kids:
            adjacency[index[node], index[kid]] = 1
            adjacency[index[kid], index[node]] = 1
    if not _is_tree(adjacency):
        raise InvalidLCAGError("reconstruction is not a tree")
    return adjacency


def lcag_roundtrip(lcag: np.ndarray) -> np.ndarray:
    """Reconstruct a tree and return the LCAG it implies.

    This detects matrices accepted by greedy reconstruction that do not describe
    the resulting tree. Preserve the input's level values, including gaps.

    Args:
        lcag: ``(L, L)`` LCAG matrix; the diagonal is ignored.

    Returns:
        The ``(L, L)`` LCAG of the reconstructed tree, equal to ``lcag`` off the
        diagonal exactly when the matrix is self-consistent.

    Raises:
        InvalidLCAGError: If ``lcag`` does not reconstruct into a tree at all.
    """
    lcag = np.asarray(lcag)
    lcag_to_adjacency(lcag)  # validates: symmetric, consistent, connected, a tree
    level, parent, _ = _reconstruct(lcag)

    n_leaves = lcag.shape[0]
    values = np.unique(lcag[~np.eye(n_leaves, dtype=bool)]).tolist()
    chains = [_ancestry(leaf, parent) for leaf in range(n_leaves)]
    implied = np.zeros_like(lcag)
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            rank = level[_lowest_common(chains[i], chains[j])]
            if not 2 <= rank <= len(values) + 1:
                raise InvalidLCAGError(f"reconstructed level {rank} has no matching entry")
            implied[i, j] = implied[j, i] = values[rank - 2]
    return implied


def _reconstruct(lcag: np.ndarray) -> tuple[list[int], list[int], list[list[int]]]:
    """Grow the tree bottom-up from the LCAG; return node levels, parents and children."""
    n_leaves = lcag.shape[0]
    level = [1] * n_leaves
    parent = [-1] * n_leaves
    children: list[list[int]] = [[] for _ in range(n_leaves)]
    off_diagonal = ~np.eye(n_leaves, dtype=bool)
    values = np.unique(lcag[off_diagonal]).tolist()

    # Rank the distinct entries: consecutive levels above the leaves, skipped generations
    # in the input collapse away.
    for rank, value in enumerate(values, start=2):
        for col in range(n_leaves):
            for row in range(col + 1, n_leaves):
                if lcag[row, col] != value:
                    continue
                node_a = _ancestor(col, parent)
                node_b = _ancestor(row, parent)
                level_a, level_b = level[node_a], level[node_b]
                if level_a > rank or level_b > rank:
                    raise InvalidLCAGError(
                        f"leaves {col} and {row} already have an ancestor above level {rank}"
                    )
                if level_a == rank and level_b == rank:
                    if node_a != node_b:
                        raise InvalidLCAGError(
                            f"leaves {col} and {row} sit under two different ancestors "
                            f"at level {rank}"
                        )
                elif level_a < rank and level_b < rank:
                    if node_a == node_b:
                        raise InvalidLCAGError(
                            f"leaves {col} and {row} already share an ancestor below level {rank}"
                        )
                    parent[node_a] = parent[node_b] = len(level)
                    children.append([node_a, node_b])
                    level.append(rank)
                    parent.append(-1)
                elif level_b < rank:
                    parent[node_b] = node_a
                    children[node_a].append(node_b)
                else:
                    parent[node_a] = node_b
                    children[node_b].append(node_a)
    return level, parent, children


def _ancestor(node: int, parent: list[int]) -> int:
    """Return the topmost ancestor of a node, itself if it has no parent."""
    while parent[node] >= 0:
        node = parent[node]
    return node


def _bfs(root: int, children: list[list[int]]) -> list[int]:
    """Return the nodes reachable from ``root``, breadth-first."""
    order = [root]
    head = 0
    while head < len(order):
        order.extend(children[order[head]])
        head += 1
    return order


def _is_tree(adjacency: np.ndarray) -> bool:
    """Report whether an adjacency matrix encodes an undirected, connected, acyclic graph.

    Uses the ``|E| = |V| - 1`` characterisation of a tree instead of the recursive cycle
    search of baumbauen's ``tree_utils.is_valid_tree``; connected plus ``|V| - 1`` edges is
    equivalent to connected plus acyclic.
    """
    n_nodes = adjacency.shape[0]
    if adjacency.shape[0] != adjacency.shape[1] or not np.array_equal(adjacency, adjacency.T):
        return False
    if np.any(np.diag(adjacency) != 0) or int(adjacency.sum()) != 2 * (n_nodes - 1):
        return False
    reached = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbour in np.flatnonzero(adjacency[node]).tolist():
            if neighbour not in reached:
                reached.add(neighbour)
                stack.append(neighbour)
    return len(reached) == n_nodes


def is_valid_lcag(lcag: np.ndarray) -> bool:
    """Test whether an LCAG matrix reconstructs to a tree.

    Greedy reconstruction can accept an inconsistent matrix; use
    :func:`lcag_roundtrip` for the stricter check. Only :class:`InvalidLCAGError`
    is treated as an invalid matrix.

    Args:
        lcag: ``(L, L)`` LCAG matrix.

    Returns:
        ``True`` if :func:`lcag_to_adjacency` succeeds.
    """
    try:
        lcag_to_adjacency(lcag)
    except InvalidLCAGError:
        return False
    return True
