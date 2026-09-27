"""The task-spectrum probe: how much of the edge decision is low-degree trig?

ROADMAP phase 5's alignment question, pulled forward as a measurement (no
training). Per sampled real edge, the LCA class is regressed on integer-frequency
trigonometric features ``cos/sin(w . theta)`` of the six encoded angles
(`pair_polar_boost`, the gated s4 chart), by cumulative total degree ``|w|_1``.
The ``L = 2`` QFM reaches per-feature frequencies ``|w_f| <= 2``, so comparing
``k_max = 2`` against ``k_max = 4`` at the same degree says whether the model's
frequency box is spectrally sufficient for the *single-edge* part of the task --
message passing sees more than one edge, so these numbers bound the edge function
alone, not the model.

A second, one-line probe fits the same ridge on powers of the pair invariant
``m_ij`` (the physically decisive quantity for the LCA level): if that alone
matches the trig probe, the information is in the invariant, not in richer
spectra -- the `ALGEBRA.md` question in empirical form.

    python dev/s5-trig-nodes/spectrum.py            # prints the table, writes results/spectrum.json
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
DATA = STUDY.parent / "s2-expressivity" / "data"

N_EDGES = 16384
SEED = 987654321
RIDGE = 1e-3


def sample_edges(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample real labelled edges: six encoded angles, the pair mass, the LCA class."""
    import jax.numpy as jnp

    from partiqledtr.models.qfm import pair_polar_boost

    with np.load(DATA / "train.npz") as data:
        features, lcag, n_fsps = data["features_cartesian"], data["lcag"], data["n_fsps"]

    events = rng.integers(0, len(n_fsps), size=2 * N_EDGES)
    i = rng.integers(0, n_fsps[events])
    j = rng.integers(0, n_fsps[events] - 1)
    j = np.where(j >= i, j + 1, j)  # j != i, uniform over the others
    labels = lcag[events, i, j]
    keep = np.flatnonzero(labels != -1)[:N_EDGES]
    events, i, j, labels = events[keep], i[keep], j[keep], labels[keep]

    angles = np.asarray(pair_polar_boost(jnp.asarray(features)))
    theta = np.concatenate([angles[events, i], angles[events, j]], axis=-1)

    pair = features[events, i] + features[events, j]
    m2 = pair[:, 3] ** 2 - np.sum(pair[:, :3] ** 2, axis=-1)
    return theta, np.sqrt(np.clip(m2, 0.0, None)), labels.astype(int)


def frequencies(k_max: int, d_max: int) -> list[tuple[int, ...]]:
    """Canonical integer frequency vectors with ``|w_f| <= k_max``, ``1 <= |w|_1 <= d_max``.

    Canonical means the first nonzero entry is positive: ``w`` and ``-w`` span the
    same cos/sin pair, so only one of each is kept.
    """
    kept = []
    for w in itertools.product(range(-k_max, k_max + 1), repeat=6):
        degree = sum(abs(k) for k in w)
        if not 1 <= degree <= d_max:
            continue
        first = next(k for k in w if k)
        if first > 0:
            kept.append(w)
    return kept


def trig_features(theta: np.ndarray, freqs: list[tuple[int, ...]]) -> np.ndarray:
    """``[1, cos(w . theta), sin(w . theta), ...]`` for every frequency vector."""
    phase = theta @ np.asarray(freqs, dtype=float).T
    return np.concatenate([np.ones((len(theta), 1)), np.cos(phase), np.sin(phase)], axis=1).astype(
        np.float32
    )


def probe(train: tuple[np.ndarray, np.ndarray], test: tuple[np.ndarray, np.ndarray]) -> float:
    """Held-out accuracy of a ridge fit to one-hot classes, argmax-decoded."""
    features, labels = train
    onehot = np.eye(labels.max() + 1)[labels]
    gram = features.T @ features + RIDGE * len(features) * np.eye(features.shape[1])
    weights = np.linalg.solve(gram, features.T @ onehot)
    held_features, held_labels = test
    return float((np.argmax(held_features @ weights, axis=1) == held_labels).mean())


def main() -> None:
    """Run the probes and print the degree table."""
    rng = np.random.default_rng(SEED)
    theta, mass, labels = sample_edges(rng)
    half = len(labels) // 2
    theta_tr, theta_te = theta[:half], theta[half:]
    y_tr, y_te = labels[:half], labels[half:]

    majority = float((y_te == np.bincount(y_tr).argmax()).mean())
    report = {"n_edges": len(labels), "majority": majority, "cells": {}}
    print(f"edges: {len(labels)}, classes: {labels.max() + 1}, majority: {majority:.3f}")
    print("| probe | features | held-out acc |")
    print("| --- | --- | --- |")

    powers = mass[:, None] ** np.arange(4)[None, :]
    acc = probe((powers[:half], y_tr), (powers[half:], y_te))
    report["cells"]["m_ij poly3"] = acc
    print(f"| m_ij powers 0-3 | 4 | {acc:.3f} |")

    for k_max, d_max in ((2, 1), (2, 2), (2, 3), (4, 2), (4, 3)):
        freqs = frequencies(k_max, d_max)
        train_f = trig_features(theta_tr, freqs)
        test_f = trig_features(theta_te, freqs)
        acc = probe((train_f, y_tr), (test_f, y_te))
        report["cells"][f"trig k<={k_max} d<={d_max}"] = acc
        print(f"| trig k<={k_max} d<={d_max} | {train_f.shape[1]} | {acc:.3f} |")

    out = STUDY / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "spectrum.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
