# Notepad

Small intermediate bugs/features land here (see README note).

## qml-essentials

**Open (0.2.3): `Model.apply`'s docstring names one axis fewer than it returns.**
Dropping the squeeze was deliberate; the axis list beside it was not updated. The
docstring promises `(B_I, B_P, B_R, O)` -- four axes -- and the method returns five.

Measured on `make_qfm(...)`: inputs `(7, 4)` with params `(1, 3, 6)` give
`(7, 1, 1, 1, 4)`, params `(2, 3, 6)` give `(7, 2, 1, 1, 4)`. So axis 0 is `B_I`,
axis 1 is `B_P`, axis -1 is `O`, and two singleton axes sit where the docstring
names one `B_R`. Naming both would fix it. The rest of that paragraph still holds:
`__call__` does squeeze, and returns `(7, 4)` for the same call.

Callers that reshape or squeeze are unaffected -- `g_purity_model`'s
`execution_type="state"` path here reshapes to `(-1, 2**n)` and did not notice.
Ones that assert the documented shape are: `models/qfm.py` `_edges` did, and now
pins the five-axis rank instead.

**Resolved upstream (2026-08-31): `Topology.graph` shipped.** The landed version
matches the sketch below except that it deduplicates on the *ordered* pair (both
orientations of one qubit pair are allowed, for directed gates) and preserves
orientation. The phase-6 arms build on it (`partiqledtr/ansaetze.py`, D110). Original
flag, kept for the record -- the phase-6 graph arms (even cycle, 3-rung ladder,
odd-chord control at `n = 6`, ROADMAP phase 6) are not expressible through
`bricks`/`stairs`/`all_pairs`, and the per-orbit parameter tying needs one `Block` per
edge orbit. Sketch of the missing helper -- validation only (indices in range, no
self-loops, no unordered duplicates), order preserved so the circuit is deterministic;
gate choice and parameter sharing stay with `Block`, exactly as for the built-in
topologies:

    class Topology:
        @staticmethod
        def graph(n_qubits: int, *, edges: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
            """Explicit edge list as a topology."""
            seen = set()
            for q, r in edges:
                if not (0 <= q < n_qubits and 0 <= r < n_qubits) or q == r:
                    raise ValueError(f"edge ({q}, {r}) invalid on {n_qubits} qubits")
                if frozenset((q, r)) in seen:
                    raise ValueError(f"duplicate edge ({q}, {r})")
                seen.add(frozenset((q, r)))
            return [tuple(edge) for edge in edges]

Usage, the ladder arm at `n = 6` -- one block pair per `pi`-orbit, tied where the
orbit has two edges, which is exactly the `XY_Ring` mechanism:

    def _xy_graph(edges, *, shared=False):
        return tuple(
            Block(gate=gate, topology=Topology.graph, edges=edges, shared=shared)
            for gate in (Gates.RXX, Gates.RYY)
        )

    class XY_Ladder(DeclarativeCircuit):
        @classmethod
        def structure(cls):
            return (
                *_xy_graph(((0, 1), (3, 4)), shared=True),  # chain link 1, pi-orbit
                *_xy_graph(((1, 2), (4, 5)), shared=True),  # chain link 2, pi-orbit
                *_xy_graph(((0, 3),)),                      # rungs: pi-fixed,
                *_xy_graph(((1, 4),)),                      # each its own angle
                *_xy_graph(((2, 5),)),
            )

`ansaetze.bonds()` already calls `block.topology(n_qubits=..., **block.kwargs)`, so it
reads the edge lists with no change on this side. To be resolved upstream before the
s4 study starts.

**Open (2026-09-03): no way to declare a nullable flow input.** A flow input
declared as `Port("lr_qfm", "json", initial=None)` is silently not registered:
a submitted value for it is dropped from the run's params, and the run then
fails with `TypeError: process() missing ... positional argument`, because the
worker wrapper makes every port in a node's `requires` a required argument
regardless of the Python signature's default. Two fixes would help, either one
sufficient: register inputs whose `initial` is `None` (delivering the null),
or let a node's Python default stand in for an absent optional port. Until
then the `train` flow carries `lr_preconditioner`/`lr_qfm` as float ports with
`initial=0.0` and the *flow-level* contract "non-positive means share `lr`"
(the 0.0-freezes-its-group diagnostic of D109 stays reachable through the
in-process path). Revert to nullable ports when this lands.

Also observed, minor: when `fit` fails, `evaluate` is still invoked and fails
with `missing ... 'checkpoint'` rather than being skipped as downstream of a
failed node — cosmetic (the run errors either way), but it doubles the error
count in `status_reason`.

## phasespace-jax


## Fluksio

