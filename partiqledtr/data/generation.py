"""Phase-space event generation for a sampled decay topology (ROADMAP phase 1).

Uses `phasespace-jax <https://github.com/stroblme/phasespace-jax>`_, a JAX port of
phasespace, so the project has no TensorFlow dependency and generation runs on the
same array backend as everything else.

Facts about the phasespace API established by measurement rather than from its
documentation (see ``DECISIONS.md`` D13, D14):

* ``GenParticle.generate`` returns momenta as ``(n_events, 4)`` arrays laid out
  ``[px, py, pz, E]``.
* With ``normalize_weights=True`` (the default) the returned weights are already
  divided by the maximum attainable weight, which is constant across events, so
  they lie in ``[0, 1]`` and accept-reject against a uniform draw is exact
  unweighting -- no maximum has to be estimated.
* Randomness is an explicit JAX key. The same key reproduces a draw exactly, so
  the accept-reject loop must *split* its key per round; reusing one key would
  redraw the identical chunk forever.
* The kinematics run under a scoped ``jax.enable_x64()`` and therefore return
  **float64** arrays, whatever the calling program's default is. Combining those
  directly with a float32 JAX array warns and silently truncates, so this module
  converts to numpy at the boundary and hands back numpy float64. Downstream code
  never meets a stray float64 JAX array.
* ``generate`` is jitted with ``n_events`` as a static argument, so every distinct
  chunk size costs a compilation. Acceptance rates vary by orders of magnitude
  between topologies -- a decay whose daughters nearly saturate the parent mass
  has very little phase space -- so the loop draws one pilot chunk to measure the
  rate, then holds a single derived chunk size for the rest. That is two compiled
  sizes per topology instead of one per round.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import phasespace

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
    density. Prior work (baumbauen, partiqlegan) discarded the weights and used
    the raw sample, which biases exactly the angular marginals this project
    studies (``DECISIONS.md`` D12).

    Args:
        topology: Nested ``{"name", "mass", "children"}`` dict describing the decay.
        n_events: Number of unweighted events to return.
        seed: Seed for the JAX generation key and the acceptance draws.
        chunk_factor: Headroom on the chunk size derived from the measured
            acceptance rate, so a round normally overshoots rather than needing
            another one.
        max_rounds: Safety bound on the accept-reject loop.

    Returns:
        Mapping from final-state particle name to an ``(n_events, 4)`` float array
        of ``[px, py, pz, E]`` four-momenta.

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

    for _ in range(max_rounds):
        if n_accepted >= n_events:
            break
        draw(chunk)
    else:
        raise RuntimeError(
            f"unweighting did not reach {n_events} events in {max_rounds} rounds of "
            f"{chunk} ({n_accepted} accepted); measured acceptance rate is {rate:.3g}"
        )

    sample = np.concatenate(accepted)[:n_events]
    return {name: sample[:, i] for i, name in enumerate(names)}
