"""Fixed isotropic whitening of the encoded input distribution (ROADMAP phase 4).

The unflattening construction: draw one rotation ``Q ~ Haar(SO(4))``, hold it fixed
across the whole dataset, and read the pair-polar angles of the rotated
four-vectors. Rotational invariance in each coordinate plane makes the resulting
angles close to uniform, which is what lifts the g-purity of the encoded product
state off the floor-free ansatz's clustered regime.

A drawn ``Q`` is *accepted* only if the dataset-mean off-diagonal g-purity it
produces reaches half the iid-uniform mean,

    mean_i P_g(rho(phi(Q x_i)))  >=  mu_n / 2,

which Markov's inequality guarantees with probability at least about 1/5, so
rejection sampling terminates in a handful of draws. The test costs one rotation
of the sample plus an O(n) purity evaluation per row.

Fitted on the **training split only**: the whitening is part of the model, and
fitting it on data the model is later evaluated on would leak.
"""

from __future__ import annotations

import io
from typing import Any

import fluksio
import jax.numpy as jnp
import numpy as np
from fluksio import Port, node

from partiqledtr.analysis import g_purity_offdiag, offdiag_uniform_mean
from partiqledtr.models.qfm import N_QUBITS, pair_polar

__all__ = ["fit_whitening", "purity_of_rotation", "sample_rotation"]

#: Encodings that carry four-vectors, hence the ones a rotation can be fitted on.
_ROTATABLE = ("cartesian", "legacy")


def sample_rotation(rng: np.random.Generator, dim: int = 4) -> np.ndarray:
    """Draw one Haar-distributed rotation from ``SO(dim)``.

    QR of a Gaussian matrix gives a Haar-distributed orthogonal matrix once the
    sign ambiguity of ``R``'s diagonal is removed; flipping one column when the
    determinant is negative restricts it to the rotation subgroup.

    Args:
        rng: Random generator; the only source of randomness.
        dim: Dimension of the rotation.

    Returns:
        A ``(dim, dim)`` orthogonal matrix with determinant ``+1``.
    """
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _sample_pairs(
    rng: np.random.Generator, n_fsps: np.ndarray, n_pairs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw random within-event ordered leaf pairs, avoiding the padded rows.

    Returns the event index and the two leaf indices of each sampled edge. The
    second leaf is drawn as an offset in ``1 .. count - 1`` so it never equals the
    first, and both stay below that event's real particle count.
    """
    events = rng.integers(len(n_fsps), size=n_pairs)
    counts = n_fsps[events]
    first = (rng.random(n_pairs) * counts).astype(np.int64)
    offset = 1 + (rng.random(n_pairs) * (counts - 1)).astype(np.int64)
    return events, np.stack([first, (first + offset) % counts], axis=1)


def purity_of_rotation(
    p4: np.ndarray,
    n_fsps: np.ndarray,
    rotation: np.ndarray,
    rng: np.random.Generator,
    n_pairs: int = 4096,
) -> float:
    """Mean off-diagonal g-purity of the states one rotation encodes.

    Args:
        p4: ``(N, L, 4)`` padded four-vectors.
        n_fsps: ``(N,)`` real particle count per event.
        rotation: ``(4, 4)`` candidate rotation.
        rng: Random generator used to sample the leaf pairs.
        n_pairs: Number of edge samples to average over.

    Returns:
        The mean of :func:`partiqledtr.analysis.g_purity_offdiag` over sampled edges.
    """
    events, pairs = _sample_pairs(rng, n_fsps, n_pairs)
    angles = np.asarray(pair_polar(np.asarray(p4[events] @ rotation.T)))
    taken = np.take_along_axis(angles, pairs[:, :, None], axis=1)
    purity = g_purity_offdiag(jnp.asarray(taken.reshape(n_pairs, N_QUBITS)))
    return float(np.mean(np.asarray(purity)))


def fit_whitening(
    p4: np.ndarray,
    n_fsps: np.ndarray,
    *,
    seed: int = 0,
    max_draws: int = 200,
    n_pairs: int = 4096,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rejection-sample a whitening rotation that passes the purity acceptance test.

    Args:
        p4: ``(N, L, 4)`` padded four-vectors of the **training** split.
        n_fsps: ``(N,)`` real particle count per event.
        seed: Seed for the rotation draws and the pair sampling.
        max_draws: Maximum candidate rotations before giving up.
        n_pairs: Edge samples used to estimate each candidate's mean purity.

    Returns:
        The accepted rotation and a report recording the threshold, the accepted
        purity, the raw (unrotated) purity it had to beat, the number of draws and
        the empirical acceptance rate.

    Raises:
        ValueError: If the inputs are malformed.
        RuntimeError: If no candidate passed within ``max_draws``. That is a
            result about the data, not a bug: log it rather than lowering the bar.
    """
    p4 = np.asarray(p4, dtype=float)
    if p4.ndim != 3 or p4.shape[-1] != 4:
        raise ValueError(f"p4 must be (N, L, 4), got shape {p4.shape}")
    n_fsps = np.asarray(n_fsps)
    if n_fsps.shape != (p4.shape[0],):
        raise ValueError(f"n_fsps must be ({p4.shape[0]},), got {n_fsps.shape}")
    if np.any(n_fsps < 2):
        raise ValueError("every event needs at least two real particles to form an edge")
    if max_draws < 1 or n_pairs < 1:
        raise ValueError(f"max_draws and n_pairs must be positive, got {max_draws}, {n_pairs}")

    threshold = offdiag_uniform_mean(N_QUBITS) / 2.0
    rng = np.random.default_rng(seed)
    raw = purity_of_rotation(p4, n_fsps, np.eye(4), np.random.default_rng(seed), n_pairs)

    accepted: np.ndarray | None = None
    purity = float("nan")
    first_draw = 0
    n_accepted = 0
    # Keep drawing past the first acceptance only long enough to estimate the
    # acceptance rate, which is the quantity the Markov bound is about.
    min_draws = min(max_draws, 20)
    for draw in range(1, max_draws + 1):
        candidate = sample_rotation(rng)
        value = purity_of_rotation(p4, n_fsps, candidate, rng, n_pairs)
        if value >= threshold:
            n_accepted += 1
            if accepted is None:
                accepted, purity, first_draw = candidate, value, draw
        if accepted is not None and draw >= min_draws:
            break

    if accepted is None:
        raise RuntimeError(
            f"no rotation reached the purity threshold {threshold:.4g} in {max_draws} draws "
            f"(raw angles score {raw:.4g}); this is a statement about the data, not a bug"
        )

    report = {
        "threshold": threshold,
        "uniform_mean": offdiag_uniform_mean(N_QUBITS),
        "accepted_purity": purity,
        "raw_purity": raw,
        "draws_to_first_acceptance": first_draw,
        "acceptance_rate": n_accepted / draw,
        "n_pairs": n_pairs,
        "seed": seed,
    }
    return accepted, report


@node(
    # Rejection sampling over the training split, silent throughout (D89).
    timeout=3600,
    requires=[
        Port("dataset_train", "artifact"),
        Port("whitening_seed", "int"),
        Port("encoding", "str"),
    ],
    provides=[Port("whitening", "artifact"), Port("whitening_report", "json")],
)
def whitening_rotation(
    *,
    dataset_train: dict[str, Any],
    encoding: str = "cartesian",
    whitening_seed: int = 0,
    max_draws: int = 200,
    n_pairs: int = 4096,
) -> dict[str, Any]:
    """Fit the phase-4 whitening rotation on the training split.

    Args:
        dataset_train: Training split artifact reference.
        encoding: The feature encoding the rotation will be *applied* to. It has to
            be the one the model trains on: a rotation accepted on one encoding's
            four-vectors says nothing about another's, and fitting on ``cartesian``
            while training on ``legacy`` measurably lowered the purity it was meant
            to raise (``DECISIONS.md`` D91). An encoding with no four-vectors falls
            back to ``cartesian`` and is marked inapplicable rather than failing the
            run, since nothing will apply it (D94).
        whitening_seed: Seed for the rotation draws and the pair sampling. Its own
            port rather than the run seed, so the acceptance rate can be swept
            without also re-seeding the model.
        max_draws: Maximum candidate rotations before giving up.
        n_pairs: Edge samples used to estimate each candidate's mean purity.

    Returns:
        The rotation as an artifact and the acceptance report.
    """
    from partiqledtr.data.dataset import ENCODINGS, load_split

    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; valid encodings are {list(ENCODINGS)}")
    # This node runs for every run so its acceptance report is always recorded, and a
    # classical arm ignores the rotation entirely -- so an encoding with no
    # four-vectors to rotate must not fail the run, it just has nothing to fit on.
    # Falling back keeps D91's substance: wherever the rotation is *applied*, it was
    # fitted on the very features the circuit encodes, because only the QFM applies
    # it and the QFM accepts four-vectors alone (D56, D94).
    fitted_on = encoding if encoding in _ROTATABLE else "cartesian"
    split = load_split(dataset_train)
    rotation, report = fit_whitening(
        split[f"features_{fitted_on}"],
        split["n_fsps"],
        seed=whitening_seed,
        max_draws=max_draws,
        n_pairs=n_pairs,
    )
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, rotation=rotation)
    return {
        "whitening": fluksio.save_artifact(buffer.getvalue(), "whitening.npz"),
        "whitening_report": {
            **report,
            "encoding": encoding,
            "fitted_on": fitted_on,
            # False means the rotation is a formality for this run: nothing will
            # apply it, because the encoding carries no four-vectors.
            "applicable": fitted_on == encoding,
        },
    }
