"""Generate unweighted decay events with phasespace-jax.

Four-vectors use ``[px, py, pz, E]`` order. Normalised phase-space weights
support accept-reject sampling; each draw uses a fresh JAX key. Generated
arrays cross the backend boundary as NumPy float64 arrays.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import phasespace

__all__ = ["UngeneratableTopologyError", "generate_events", "leaf_names"]


class UngeneratableTopologyError(RuntimeError):
    """A topology whose phase-space acceptance rate is too low to sample.

    Not a bug and not a bad topology in itself -- a decay whose daughter masses
    nearly saturate the parent mass genuinely has almost no phase space, so
    unweighting rejects nearly every draw. The caller's remedy is to sample a
    different topology.
    """


def leaf_names(topology: dict[str, Any]) -> list[str]:
    """Collect the names of a topology's final-state particles, in tree order.

    Args:
        topology: Nested ``{"name", "mass", "children"}`` dict.

    Returns:
        Names of the childless nodes, in depth-first order.
    """
    if not topology["children"]:
        return [topology["name"]]
    return [name for child in topology["children"] for name in leaf_names(child)]


def _build_particle(topology: dict[str, Any]) -> phasespace.GenParticle:
    """Translate a topology dict into a phasespace decay chain."""
    particle = phasespace.GenParticle(topology["name"], topology["mass"])
    if topology["children"]:
        particle.set_children(*[_build_particle(c) for c in topology["children"]])
    return particle


def generate_events(
    topology: dict[str, Any],
    n_events: int,
    seed: int,
    *,
    chunk_factor: float = 4.0,
    max_draws: int = 40_000_000,
) -> dict[str, np.ndarray]:
    """Generate unweighted phase-space events for one topology.

    Accept each draw according to its normalised phase-space weight.

    Args:
        topology: Nested ``{"name", "mass", "children"}`` dict describing the decay.
        n_events: Number of unweighted events to return.
        seed: Seed for the JAX generation key and the acceptance draws.
        chunk_factor: Headroom on the chunk size derived from the measured
            acceptance rate, so a round normally overshoots rather than needing
            another one.
        max_draws: Total weighted draws to spend before giving up. This is what
            "ungeneratable" means operationally -- a decay whose acceptance rate is
            too low to reach ``n_events`` within the budget -- so it is stated as a
            budget rather than as a round count.

    Returns:
        Mapping from final-state particle name to an ``(n_events, 4)`` float array
        of ``[px, py, pz, E]`` four-momenta.

    Raises:
        ValueError: If ``n_events`` is not positive or the topology has no decay.
        UngeneratableTopologyError: If ``max_draws`` was spent without reaching
            ``n_events``. The caller's remedy is a different topology, so the
            message carries the measured acceptance rate.
    """
    if n_events <= 0:
        raise ValueError(f"n_events must be positive, got {n_events}")
    if not topology["children"]:
        raise ValueError("topology must describe a decay, but its root has no children")

    names = leaf_names(topology)
    root = _build_particle(topology)
    key = jax.random.key(seed)
    accept_rng = np.random.default_rng(seed)

    accepted: list[np.ndarray] = []
    n_accepted = 0

    def draw(size: int) -> int:
        """Generate `size` events, keep the accepted ones, return how many."""
        nonlocal key, n_accepted
        key, subkey = jax.random.split(key)
        # `generate` returns (weights, momenta) normalised and (weights, max,
        # momenta) otherwise, so index the ends rather than unpacking a union.
        drawn = root.generate(size, key=subkey)
        weights = np.asarray(drawn[0])
        keep = accept_rng.random(weights.shape) < weights
        # (n_kept, n_leaves, 4): stack leaves in the order `names` fixes.
        accepted.append(np.stack([np.asarray(drawn[-1][name])[keep] for name in names], axis=1))
        n_accepted += int(keep.sum())
        return int(keep.sum())

    # One pilot chunk to measure the acceptance rate, which ranges over orders of
    # magnitude across topologies, then one fixed size for every round after it.
    pilot = max(1024, n_events)
    n_kept = draw(pilot)
    rate = max(n_kept / pilot, 1.0 / pilot)
    chunk = int(np.ceil(chunk_factor * max(n_events - n_accepted, 1) / rate))

    # Spend the budget rather than a round count: what decides whether a topology is
    # worth keeping is how many draws it costs, and the chunk size already varies by
    # orders of magnitude with the measured rate.
    drawn_total = pilot
    while n_accepted < n_events and drawn_total < max_draws:
        draw(chunk)
        drawn_total += chunk
    if n_accepted < n_events:
        raise UngeneratableTopologyError(
            f"unweighting reached {n_accepted}/{n_events} events in {drawn_total} draws "
            f"(budget {max_draws}); measured acceptance rate is {n_accepted / drawn_total:.3g}, "
            f"so this decay leaves too little phase space to sample"
        )

    sample = np.concatenate(accepted)[:n_events]
    return {name: sample[:, i] for i, name in enumerate(names)}
