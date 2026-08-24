"""Phase-space event generation for a sampled decay topology (ROADMAP phase 1).

This is the only module that imports phasespace and, through it, TensorFlow.
Nothing else in the project should import it, so that the seconds-long TensorFlow
import is paid once, in the generation flow only.

Facts about the phasespace API established by measurement rather than from its
documentation (see ``DECISIONS.md`` D13, D14):

* ``GenParticle.generate`` returns momenta as ``(n_events, 4)`` arrays laid out
  ``[px, py, pz, E]``.  Its docstring claims ``(4, n_events)``; the implementation
  is events-major.
* With ``normalize_weights=True`` (the default) the returned weights are already
  divided by the maximum attainable weight, which is constant across events, so
  they lie in ``[0, 1]`` and accept-reject against a uniform draw is exact
  unweighting -- no maximum has to be estimated.
* The global TensorFlow seed does not control the generator; a
  ``tf.random.Generator`` must be passed per call.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import phasespace
import tensorflow as tf

__all__ = ["generate_events", "leaf_names"]


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
    max_rounds: int = 100,
) -> dict[str, np.ndarray]:
    """Generate unweighted decay events for one topology.

    Events are drawn in chunks and accepted with probability equal to their
    normalised phase-space weight, so the returned sample follows the phase-space
    density.  Prior work (baumbauen, partiqlegan) discarded the weights and used
    the raw sample, which biases exactly the angular marginals this project
    studies (``DECISIONS.md`` D12).

    Args:
        topology: Nested ``{"name", "mass", "children"}`` dict describing the decay.
        n_events: Number of unweighted events to return.
        seed: Seed for both the TensorFlow generator and the acceptance draws.
        chunk_factor: Multiplier on the running acceptance-rate estimate used to
            size each generation chunk.  Larger values trade memory for fewer
            rounds.
        max_rounds: Safety bound on the accept-reject loop.

    Returns:
        Mapping from final-state particle name to an ``(n_events, 4)`` float64
        array of ``[px, py, pz, E]`` four-momenta.

    Raises:
        ValueError: If ``n_events`` is not positive or the topology has no decay.
        RuntimeError: If ``max_rounds`` chunks did not yield ``n_events`` accepted
            events, which indicates a pathologically low acceptance rate.
    """
    if n_events <= 0:
        raise ValueError(f"n_events must be positive, got {n_events}")
    if not topology["children"]:
        raise ValueError("topology must describe a decay, but its root has no children")

    names = leaf_names(topology)
    root = _build_particle(topology)
    tf_rng = tf.random.Generator.from_seed(seed)
    accept_rng = np.random.default_rng(seed)

    accepted: list[np.ndarray] = []
    n_accepted = 0
    n_drawn = 0
    for _ in range(max_rounds):
        # Size the next chunk from the acceptance rate seen so far, starting
        # optimistic and correcting downwards as evidence arrives.
        rate = max(n_accepted / n_drawn, 1e-3) if n_drawn else 1.0
        chunk = int(np.ceil(chunk_factor * (n_events - n_accepted) / rate))
        weights, events = root.generate(chunk, seed=tf_rng)

        weights = np.asarray(weights)
        keep = accept_rng.random(weights.shape) < weights
        # (n_kept, n_leaves, 4): stack leaves in the order `names` fixes.
        accepted.append(np.stack([np.asarray(events[name])[keep] for name in names], axis=1))
        n_accepted += int(keep.sum())
        n_drawn += chunk
        if n_accepted >= n_events:
            break
    else:
        raise RuntimeError(
            f"unweighting did not reach {n_events} events in {max_rounds} rounds "
            f"({n_accepted} accepted from {n_drawn} drawn); acceptance rate is "
            f"{n_accepted / max(n_drawn, 1):.3g}"
        )

    sample = np.concatenate(accepted)[:n_events]
    return {name: sample[:, i] for i, name in enumerate(names)}
