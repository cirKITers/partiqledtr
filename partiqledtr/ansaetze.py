r"""Ansatz arms of the edge QFM, and the bond structure that motivates them.

ROADMAP phase 4b arm C.  The edge QFM spends two qubits per particle, so at
``n = 4`` qubits 0, 1 carry particle A's angles and 2, 3 carry particle B's, and
the swap of the two endpoints acts on the wires as the permutation
:math:`\pi = (0\,2)(1\,3)`.  An ansatz is *partition-respecting* when its bond
set is invariant under :math:`\pi`; the edge function is then symmetric in its
two endpoints by construction rather than by the output averaging of
``DECISIONS.md`` D35.

``XY_Brickwork``, the phase-3/4 arm, is not: its bonds ``(0,1), (1,2), (2,3)``
put two of three inside a particle and join ``alpha_A`` to ``phi_B`` with the
third (``RESEARCH.md`` §9).  Replacing the odd brick layer with the span-2
bonds fixes exactly that, and the qml-essentials topologies spell both out --
``Topology.bricks(offset=0)`` is ``(0,1), (2,3)`` and
``Topology.stairs(span=2)`` is ``(0,2), (1,3)``.

Measured at ``n = 4`` (:func:`partiqledtr.analysis.dla_check`):

    ansatz          bonds                    dim_g/255  d_Z  pi-invariant
    XY_Brickwork    01, 12, 23                      12    0  no
    XY_Ring         01, 23, 02, 13                  24    0  yes
    XY_AllPairs     all six                         60    6  yes
    Circuit_19      CRX ring                       255   15  --

``XY_Ring`` is the 4-cycle ``0-1-3-2-0``.  Even cycles are bipartite, which is
the unflattening manuscript's graph criterion for :math:`d_Z = 0`, so it is
partition-respecting *and* still input-distribution sensitive -- the two
criteria the ROADMAP expected to pull apart.  ``XY_AllPairs`` adds the
triangles: an odd cycle rebuilds the diagonal sector, giving a floored control
that respects the partition as well.

Bonds alone do not give equivariance, though: :math:`\pi` maps ``(0,1)`` to
``(2,3)``, so the two gates of that orbit must carry the *same* angle, which is
what ``Block(shared=True)`` ties.  The span-2 gates map to themselves and need
no tying.
"""

from qml_essentials.ansaetze import Ansaetze, Block, DeclarativeCircuit, Gates
from qml_essentials.topologies import Topology

__all__ = ["ANSAETZE", "XY_AllPairs", "XY_Ring", "bonds", "circuit", "swap_invariant"]


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
    is a property of the algebra, not of the parameterisation -- and the output
    averaging of D35 supplies the rest of the symmetry.
    """

    @classmethod
    def structure(cls) -> tuple[Block, ...]:
        """Return the XY blocks on every bond."""
        return _xy(Topology.all_pairs)


#: Ansatz arms, name -> circuit class.  ``Model`` takes either a name it knows or
#: a class, so a project arm needs no fork of qml-essentials.  ``XY_Brickwork`` is
#: kept as the phase-3/4 continuity control and ``Circuit_19`` as the universal
#: leg of the DLA trichotomy; ``Matchgate`` was retired in phase 4b, its floored
#: role taken over by ``XY_AllPairs``, which also respects the partition.
ANSAETZE: dict[str, type[DeclarativeCircuit]] = {
    "XY_Brickwork": Ansaetze.XY_Brickwork,
    "XY_Ring": XY_Ring,
    "XY_AllPairs": XY_AllPairs,
    "Circuit_19": Ansaetze.Circuit_19,
}


def circuit(name: str) -> type[DeclarativeCircuit]:
    """Resolve an ansatz name to its circuit class.

    A phase-4b arm from :data:`ANSAETZE` first, then any ansatz qml-essentials
    ships.  The split is what "retired" means here: ``Matchgate`` is no longer a
    reported arm but stays certifiable and runnable, so the phase-4 cells of
    ``RESEARCH.md`` §7 remain reproducible.

    Args:
        name: Ansatz name.

    Returns:
        The circuit class.

    Raises:
        ValueError: If no arm and no qml-essentials ansatz has that name.
    """
    if name in ANSAETZE:
        return ANSAETZE[name]
    found = getattr(Ansaetze, name, None)
    if found is None or not isinstance(found, type):
        raise ValueError(f"unknown ansatz {name!r}; the arms are {list(ANSAETZE)}")
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
    r"""Whether an ansatz's bond set survives the endpoint swap.

    With two qubits per particle the swap of an edge's two endpoints acts on the
    wires as :math:`\pi = (0\,2)(1\,3)` at ``n = 4``, and generally as the shift by
    half the register.  A ``True`` here is what makes the edge function symmetric
    by construction rather than by the output averaging of ``DECISIONS.md`` D35.

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
