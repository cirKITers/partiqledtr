"""Decay topology sampling (ROADMAP phase 1).

A topology is a plain nested dict ``{"name": str, "mass": float, "children": list}``;
a final-state particle (FSP) is a node with an empty child list. Particles are identified
by their mass alone, hence the intermediate (ISP) and final-state mass pools must be
disjoint -- the invariant that makes :func:`canonical_form` a complete isomorphism test.
Masses are the ones partiqlegan ships in ``conf/base/parameters/data_generation.yml``.

The mass-budget walk follows partiqlegan's ``gen_structure_from_parameters``
(``reference/partiqlegan/src/partiqleDTR/pipelines/data_generation/nodes.py``) with three
deviations:

* the available mass is reduced by the mass of the child just placed; the reference
  subtracts the *running total* each iteration and therefore over-counts,
* a reserve of one minimal FSP mass on the first child guarantees every internal node
  gets at least two children, instead of the reference's early ``break`` that can leave a
  node with one or zero children,
* the requested FSP count is reached by rejecting whole draws instead of the reference's
  offline seed scan (DECISIONS.md D15).
"""

import numpy as np

# Intermediate- and final-state particle mass pools, descending. Physics identity is the
# mass, so the two pools must stay disjoint.
MASSES_ISP: tuple[float, ...] = (100.0, 90.0, 80.0, 70.0, 50.0, 25.0, 20.0, 10.0)
MASSES_FSP: tuple[float, ...] = (12.0, 5.0, 3.0, 2.0, 1.0)

# ponytail: children per internal node are fixed at partiqlegan's configured 2..3 rather
# than exposed as arguments; promote them to keyword arguments if an experiment needs
# wider decays.
MIN_CHILDREN = 2
MAX_CHILDREN = 3

# Isomorphism retries per slot in sample_topologies before giving up.
_ISO_TRIES = 100

_MIN_FSP = min(MASSES_FSP)
_MIN_ISP = min(MASSES_ISP)
_ROOT_MASS = max(MASSES_ISP)

if not set(MASSES_ISP).isdisjoint(MASSES_FSP):
    raise ValueError("MASSES_ISP and MASSES_FSP must be disjoint: particles are identified by mass")
if _MIN_ISP <= 2 * _MIN_FSP:
    # Every ISP is expanded into at least two children, each at least _MIN_FSP heavy, and
    # the sum must stay strictly below the parent mass.
    raise ValueError("every ISP mass must exceed twice the smallest FSP mass")


def sample_topology(
    rng: np.random.Generator,
    *,
    n_fsps: int,
    max_depth: int = 4,
    isp_weight: float = 1.0,
    max_tries: int = 10_000,
) -> dict:
    """Draw one topology with exactly ``n_fsps`` final-state particles.

    A breadth-first mass-budget walk expands the root: every node's children are drawn
    from the ISP or FSP pool such that their masses sum to strictly less than the parent
    mass (phasespace rejects anything else), children at level ``max_depth`` are forced to
    be final-state, and each internal node receives at least two children. Draws whose
    leaf count differs from ``n_fsps`` are rejected and repeated.

    Args:
        rng: Random generator; the only source of randomness.
        n_fsps: Required number of final-state particles, at least 2.
        max_depth: Number of levels counting the root as level 1, at least 2.
        isp_weight: Relative weight of the intermediate-state pool; larger values make
            deeper trees more likely. Must be non-negative.
        max_tries: Number of rejected draws before giving up.

    Returns:
        The sampled topology.

    Raises:
        ValueError: If an argument is out of range, or if no draw matched ``n_fsps``
            within ``max_tries`` attempts.
    """
    if n_fsps < 2:
        raise ValueError(f"n_fsps must be at least 2, got {n_fsps}")
    if max_depth < 2:
        raise ValueError(f"max_depth must be at least 2, got {max_depth}")
    if isp_weight < 0:
        raise ValueError(f"isp_weight must be non-negative, got {isp_weight}")

    for _ in range(max_tries):
        topology = _draw(rng, max_depth, isp_weight)
        if count_fsps(topology) == n_fsps:
            return topology
    raise ValueError(
        f"no topology with n_fsps={n_fsps} found within max_depth={max_depth} after "
        f"{max_tries} draws; raise max_depth, lower n_fsps, or raise isp_weight"
    )


def _draw(rng: np.random.Generator, max_depth: int, isp_weight: float) -> dict:
    """Run one unconditioned mass-budget walk and return its topology."""
    p_fsp = len(MASSES_FSP) / (len(MASSES_FSP) + isp_weight * len(MASSES_ISP))
    root: dict = {"name": "isp0", "mass": _ROOT_MASS, "children": []}
    n_isp, n_fsp = 1, 0
    queue = [(root, 1)]
    while queue:
        node, level = queue.pop(0)
        child_level = level + 1
        budget = node["mass"]
        for k in range(int(rng.integers(MIN_CHILDREN, MAX_CHILDREN + 1))):
            # Reserve room for a second child so the node never ends up with only one.
            reserve = _MIN_FSP if k == 0 else 0.0
            use_fsp = (
                child_level >= max_depth or budget - reserve <= _MIN_ISP or rng.random() < p_fsp
            )
            pool = MASSES_FSP if use_fsp else MASSES_ISP
            candidates = [m for m in pool if m < budget - reserve]
            if not candidates:
                break
            mass = candidates[int(rng.integers(len(candidates)))]
            budget -= mass
            if use_fsp:
                child = {"name": f"fsp{n_fsp}", "mass": mass, "children": []}
                n_fsp += 1
            else:
                child = {"name": f"isp{n_isp}", "mass": mass, "children": []}
                n_isp += 1
                queue.append((child, child_level))
            node["children"].append(child)
    return root


def canonical_form(topology: dict) -> str:
    """Return the parenthetical mass tuple identifying the topology up to isomorphism.

    Children are sorted, so two topologies are isomorphic (equal up to child order and
    node names) exactly when their canonical forms are equal. After A. Aho, J. Hopcroft
    and J. Ullman, *The Design and Analysis of Computer Algorithms*, 1974, pp. 84-85, as
    used by partiqlegan's ``assign_parenthetical_weight_tuples``.

    Args:
        topology: The topology to encode.

    Returns:
        A string such as ``"(100(10(1)(2))(5))"``; masses are formatted with ``%g``.
    """
    if not topology["children"]:
        return f"({topology['mass']:g})"
    children = "".join(sorted(canonical_form(c) for c in topology["children"]))
    return f"({topology['mass']:g}{children})"


def sample_topologies(
    rng: np.random.Generator,
    *,
    n_groups: int = 3,
    per_group: int = 10,
    min_fsps: int = 3,
    max_fsps: int = 8,
    max_depth: int = 4,
    isp_weight: float = 1.0,
) -> list[list[dict]]:
    """Draw pairwise non-isomorphic topologies, grouped for the known/unknown split.

    FSP counts cycle through ``[min_fsps, max_fsps]`` so every group covers the range.
    The groups feed the generalisation probe of DECISIONS.md D16 (group A to
    train/val/test, B to val/test, C to test only).

    Args:
        rng: Random generator; the only source of randomness.
        n_groups: Number of topology groups.
        per_group: Topologies per group.
        min_fsps: Smallest final-state particle count.
        max_fsps: Largest final-state particle count.
        max_depth: Number of levels counting the root as level 1.
        isp_weight: Relative weight of the intermediate-state pool.

    Returns:
        ``n_groups`` lists of ``per_group`` topologies, all pairwise non-isomorphic
        across the whole result.

    Raises:
        ValueError: If an argument is out of range, or if no unseen topology was found
            for one of the slots.
    """
    if n_groups < 1 or per_group < 1:
        raise ValueError(f"n_groups and per_group must be positive, got {n_groups}, {per_group}")
    if max_fsps < min_fsps:
        raise ValueError(f"max_fsps={max_fsps} is below min_fsps={min_fsps}")

    span = max_fsps - min_fsps + 1
    seen: set[str] = set()
    flat: list[dict] = []
    for i in range(n_groups * per_group):
        n_fsps = min_fsps + i % span
        for _ in range(_ISO_TRIES):
            topology = sample_topology(
                rng, n_fsps=n_fsps, max_depth=max_depth, isp_weight=isp_weight
            )
            key = canonical_form(topology)
            if key not in seen:
                seen.add(key)
                flat.append(topology)
                break
        else:
            raise ValueError(
                f"no unseen topology with n_fsps={n_fsps} after {_ISO_TRIES} draws; "
                f"ask for fewer topologies or widen the [min_fsps, max_fsps] range"
            )
    return [flat[g * per_group : (g + 1) * per_group] for g in range(n_groups)]


def count_fsps(topology: dict) -> int:
    """Count the final-state particles (leaves) of a topology.

    Args:
        topology: The topology to measure.

    Returns:
        The number of leaves.
    """
    if not topology["children"]:
        return 1
    return sum(count_fsps(c) for c in topology["children"])


def tree_depth(topology: dict) -> int:
    """Return the number of levels of a topology, counting the root as level 1.

    Args:
        topology: The topology to measure.

    Returns:
        The depth, comparable against the ``max_depth`` of :func:`sample_topology`.
    """
    if not topology["children"]:
        return 1
    return 1 + max(tree_depth(c) for c in topology["children"])
