"""Define XY ansatz arms and their particle-swap bond symmetries.

At four qubits, particles occupy wires (0, 1) and (2, 3); at six qubits they
occupy (0, 1, 2) and (3, 4, 5). A swap-symmetric bond set needs shared
parameters on bonds exchanged by that swap. Even-cycle and ladder arms have
no diagonal-word purity floor; odd-chord arms provide floored controls.
"""

from qml_essentials.ansaetze import Ansaetze, Block, DeclarativeCircuit, Gates
from qml_essentials.topologies import Topology

__all__ = [
    "ANSAETZE",
    "ANSAETZE_N6",
    "XY_AllPairs",
    "XY_Cycle",
    "XY_Ladder",
    "XY_OddChord",
    "XY_Ring",
    "bonds",
    "circuit",
    "swap_invariant",
]


def _xy(topology, *, shared: bool = False, **kwargs) -> tuple[Block, ...]:
    """An ``RXX`` then ``RYY`` pair on one topology: the XY coupling of a bond set."""
    return tuple(
        Block(gate=gate, topology=topology, shared=shared, **kwargs)
        for gate in (Gates.RXX, Gates.RYY)
    )


class XY_Ring(DeclarativeCircuit):
    r"""XY coupling on the cycle ``0-1-3-2-0``: partition-respecting and floor-free.

    Intra-particle bonds ``(0,1), (2,3)`` from ``Topology.bricks`` share one angle
    each, since :math:`\pi` exchanges them; the cross-particle bonds
    ``(0,2), (1,3)`` from ``Topology.stairs(span=2)`` are :math:`\pi`-fixed and
    keep their own.  The layer is therefore exactly equivariant under the endpoint
    swap.
    """

    @classmethod
    def structure(cls) -> tuple[Block, ...]:
        """Return the intra-particle then cross-particle XY blocks."""
        return (
            *_xy(Topology.bricks, shared=True, offset=0, reverse=False, mirror=False),
            *_xy(Topology.stairs, span=2, modulo=False, reverse=False, mirror=False),
        )


class XY_AllPairs(DeclarativeCircuit):
    r"""XY coupling on every bond: partition-respecting, and floored by its triangles.

    The bond set is :math:`\pi`-invariant, but ``all_pairs`` is one block, so the
    orbits ``{(0,1),(2,3)}`` and ``{(0,3),(1,2)}`` are not tied and the layer is
    equivariant only in its bonds.  Its role is the floored control -- the floor
    is a property of the algebra, not of the parameterisation -- and symmetrising
    the output supplies the rest of the symmetry.
    """

    @classmethod
    def structure(cls) -> tuple[Block, ...]:
        """Return the XY blocks on every bond."""
        return _xy(Topology.all_pairs)


def _graph6(n_qubits: int, *, edges: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    """``Topology.graph`` pinned to six qubits.

    The ``n = 6`` arms are written as explicit edge lists at the study size, so at
    any other register the same list would silently be a different graph (a
    subgraph at ``n > 6``); failing loudly is what keeps the certificates honest.
    """
    if n_qubits != 6:
        raise ValueError(f"this arm is defined at 6 qubits, got n_qubits={n_qubits}")
    return Topology.graph(n_qubits, edges=edges)


#: The intra-particle chains every ``n = 6`` graph arm shares, one ``_xy`` block
#: pair per pi-orbit: pi maps (0,1) to (3,4) and (1,2) to (4,5), so each orbit
#: ties its angle (``shared=True``), exactly the ``XY_Ring`` mechanism.
_CHAIN_ORBITS = (((0, 1), (3, 4)), ((1, 2), (4, 5)))


def _chains() -> tuple[Block, ...]:
    return tuple(
        block for orbit in _CHAIN_ORBITS for block in _xy(_graph6, edges=orbit, shared=True)
    )


class XY_Cycle(DeclarativeCircuit):
    r"""XY coupling on the even 6-cycle ``0-1-2-5-4-3-0``: poly floor-free at ``n = 6``.

    The intra-particle chains plus the end rungs ``(0,3), (2,5)``.  An even cycle
    is bipartite, so ``d_Z = 0`` (measured: ``dim_g = 60``), and the bond set is
    :math:`\pi`-invariant with the rungs :math:`\pi`-fixed.  The ``n = 6`` analogue
    of ``XY_Ring``, which is this construction at ``n = 4``.
    """

    @classmethod
    def structure(cls) -> tuple[Block, ...]:
        """Return the chain orbits then the end rungs."""
        return (*_chains(), *_xy(_graph6, edges=((0, 3), (2, 5))))


class XY_Ladder(DeclarativeCircuit):
    r"""XY coupling on the 3-rung ladder: the hard floor-free arm at ``n = 6``.

    The cycle plus the middle rung ``(1,4)``, giving the bipartite graph two
    degree-3 vertices -- the unflattening manuscript's encoded-universality
    criterion -- while keeping ``d_Z = 0`` (measured: ``dim_g = 510``).  All
    rungs are :math:`\pi`-fixed.
    """

    @classmethod
    def structure(cls) -> tuple[Block, ...]:
        """Return the chain orbits then all three rungs."""
        return (*_chains(), *_xy(_graph6, edges=((0, 3), (1, 4), (2, 5))))


class XY_OddChord(DeclarativeCircuit):
    r"""XY coupling on the cycle plus intra-particle chords: the floored hard control.

    The chords ``(0,2), (3,5)`` close one odd triangle inside each particle, and
    the floor returns exactly as the manuscript's odd-cycle rule predicts
    (measured: ``dim_g = 1020``, ``d_Z = 30``).  :math:`\pi` exchanges the two
    chords, so their orbit ties its angle like the chain orbits.
    """

    @classmethod
    def structure(cls) -> tuple[Block, ...]:
        """Return the chain orbits, the end rungs, then the tied chord orbit."""
        return (
            *_chains(),
            *_xy(_graph6, edges=((0, 3), (2, 5))),
            *_xy(_graph6, edges=((0, 2), (3, 5)), shared=True),
        )


#: Ansatz arms, name -> circuit class.  ``Model`` takes either a name it knows or
#: a class, so a project arm needs no fork of qml-essentials.  ``XY_Brickwork`` is
#: kept as the original arm for continuity and ``Circuit_19`` as the universal
#: leg of the DLA trichotomy; ``Matchgate`` is retired, its floored role taken
#: over by ``XY_AllPairs``, which also respects the partition.
ANSAETZE: dict[str, type[DeclarativeCircuit]] = {
    "XY_Brickwork": Ansaetze.XY_Brickwork,
    "XY_Ring": XY_Ring,
    "XY_AllPairs": XY_AllPairs,
    "Circuit_19": Ansaetze.Circuit_19,
}

#: The graph arms, defined at ``n = 6`` only.  A separate registry so the
#: surfaces that iterate :data:`ANSAETZE` at ``n = 4`` (``arm_report``, the
#: parametrised tests) keep their meaning; :func:`circuit` resolves both.
ANSAETZE_N6: dict[str, type[DeclarativeCircuit]] = {
    "XY_Cycle": XY_Cycle,
    "XY_Ladder": XY_Ladder,
    "XY_OddChord": XY_OddChord,
}


def circuit(name: str) -> type[DeclarativeCircuit]:
    """Resolve an ansatz name to its circuit class.

    An ``n = 4`` arm from :data:`ANSAETZE` first, then an ``n = 6`` arm from
    :data:`ANSAETZE_N6`, then any ansatz qml-essentials ships.  The split is what
    "retired" means here: ``Matchgate`` is no longer a reported arm but stays
    certifiable and runnable, so earlier runs on it remain reproducible.

    Args:
        name: Ansatz name.

    Returns:
        The circuit class.

    Raises:
        ValueError: If no arm and no qml-essentials ansatz has that name.
    """
    if name in ANSAETZE:
        return ANSAETZE[name]
    if name in ANSAETZE_N6:
        return ANSAETZE_N6[name]
    found = getattr(Ansaetze, name, None)
    if found is None or not isinstance(found, type):
        raise ValueError(
            f"unknown ansatz {name!r}; the arms are {list(ANSAETZE) + list(ANSAETZE_N6)}"
        )
    return found


def bonds(ansatz: str, n_qubits: int) -> set[frozenset[int]]:
    """Return the unordered two-qubit bonds an ansatz's entangling blocks touch.

    Read off each block's own ``Topology`` call, so it describes the circuit that
    runs rather than an assumed pattern -- the same reason
    :func:`partiqledtr.analysis.ansatz_generators` walks ``structure()``.

    Args:
        ansatz: Ansatz name, resolved by :func:`circuit`.
        n_qubits: Number of qubits.

    Returns:
        One frozenset per bond.
    """
    return {
        frozenset(bond)
        for block in circuit(ansatz).structure()
        if block.topology is not None
        for bond in block.topology(n_qubits=n_qubits, **block.kwargs)
    }


def swap_invariant(ansatz: str, n_qubits: int) -> bool:
    """Test whether an ansatz's bonds survive swapping the two particles.

    The swap shifts wires by half the register.

    Args:
        ansatz: Ansatz name.
        n_qubits: Number of qubits; must be even.

    Returns:
        True if the bond set maps to itself.

    Raises:
        ValueError: If ``n_qubits`` is odd, so there is no such swap.
    """
    if n_qubits % 2:
        raise ValueError(f"the endpoint swap needs an even register, got {n_qubits}")
    half = n_qubits // 2
    have = bonds(ansatz, n_qubits)
    return {frozenset((q + half) % n_qubits for q in bond) for bond in have} == have
