"""Dataset assembly and the generation flow's nodes (ROADMAP phase 1).

The heavy lifting lives in :func:`assemble_dataset` and :func:`dataset_statistics`,
which are plain functions returning arrays: Fluksio's ``save_artifact`` raises
outside a running node, so keeping the pure part separate is what makes the
pipeline testable without an engine.

Splits follow the known/unknown topology scheme (``DECISIONS.md`` D16): with three
topology groups, group A's events are spread over train, validation and test,
group B's over validation and test, and group C's go to test alone. Evaluating on
the test split therefore measures generalisation to unseen topologies as well as
to unseen events, and ``topology_id`` together with ``meta["topology_group"]``
tells the two apart.
"""

from __future__ import annotations

import io
from typing import Any

import fluksio
import numpy as np
from fluksio import Port, node

from partiqledtr.data.features import (
    apply_normalization,
    featurize,
    normalization_scales,
    pad_events,
)
from partiqledtr.data.lcag import shuffle_leaves, topology_to_lcag
from partiqledtr.data.topology import canonical_form, count_fsps, sample_topologies

__all__ = ["assemble_dataset", "build_dataset", "dataset_statistics", "dataset_stats", "load_split"]

ENCODINGS = ("angles", "cartesian")
SPLITS = ("train", "val", "test")

# Which splits each topology group may contribute events to (D16).
_GROUP_SPLITS: tuple[tuple[str, ...], ...] = (("train", "val", "test"), ("val", "test"), ("test",))


def _split_sizes(group: int, n_events: int, val_frac: float, test_frac: float) -> dict[str, int]:
    """Divide one topology's events among the splits its group may feed."""
    if group == 0:
        n_val = round(val_frac * n_events)
        n_test = round(test_frac * n_events)
        return {"train": n_events - n_val - n_test, "val": n_val, "test": n_test}
    if group == 1:
        n_val = round(n_events * val_frac / (val_frac + test_frac))
        return {"train": 0, "val": n_val, "test": n_events - n_val}
    return {"train": 0, "val": 0, "test": n_events}


def assemble_dataset(
    *,
    seed: int,
    n_topologies: int = 10,
    n_events_per_topology: int = 1000,
    min_fsps: int = 3,
    max_fsps: int = 8,
    max_depth: int = 4,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    isp_weight: float = 1.0,
    n_groups: int = 3,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Sample topologies, generate events and assemble the three dataset splits.

    Both feature encodings are stored side by side so the ablation arms never have
    to re-run phase-space generation (``DECISIONS.md`` D19). Normalisation scales
    are fitted on the training split alone and applied to all three.

    Args:
        seed: Master seed; every random stream is derived from it.
        n_topologies: Topologies **per group**, so the total is
            ``n_groups * n_topologies`` (``DECISIONS.md`` D3 note).
        n_events_per_topology: Unweighted events generated per topology.
        min_fsps: Smallest final-state particle count to sample.
        max_fsps: Largest final-state particle count; also the padded width ``L``.
        max_depth: Maximum topology depth, counting the root as level 1.
        val_frac: Fraction of a group-A topology's events used for validation.
        test_frac: Fraction of a group-A topology's events used for testing.
        isp_weight: Relative weight of the intermediate-state mass pool.
        n_groups: Number of topology groups; 3 gives the known/unknown scheme.

    Returns:
        A ``(splits, meta)`` pair. ``splits`` maps each split name to a dict of
        arrays: ``features_angles`` ``(N, L, 3)``, ``features_cartesian``
        ``(N, L, 4)``, ``lcag`` ``(N, L, L)`` int8, ``n_fsps`` ``(N,)`` and
        ``topology_id`` ``(N,)``. ``meta`` is json-serialisable.

    Raises:
        ValueError: If an argument is out of range.
    """
    if n_topologies < 1:
        raise ValueError(f"n_topologies must be positive, got {n_topologies}")
    if n_events_per_topology < 1:
        raise ValueError(f"n_events_per_topology must be positive, got {n_events_per_topology}")
    if min_fsps < 2 or max_fsps < min_fsps:
        raise ValueError(f"need 2 <= min_fsps <= max_fsps, got {min_fsps}, {max_fsps}")
    if not 0.0 < val_frac < 1.0 or not 0.0 < test_frac < 1.0 or val_frac + test_frac >= 1.0:
        raise ValueError(
            f"need val_frac + test_frac < 1 with both positive, got {val_frac}, {test_frac}"
        )
    if not 1 <= n_groups <= len(_GROUP_SPLITS):
        raise ValueError(f"n_groups must be between 1 and {len(_GROUP_SPLITS)}, got {n_groups}")

    # Imported here so that importing this module does not pull in TensorFlow (D7).
    from partiqledtr.data.generation import generate_events

    rng = np.random.default_rng(seed)
    groups = sample_topologies(
        rng,
        n_groups=n_groups,
        per_group=n_topologies,
        min_fsps=min_fsps,
        max_fsps=max_fsps,
        max_depth=max_depth,
        isp_weight=isp_weight,
    )

    parts: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in SPLITS}
    topology_group: list[int] = []
    topology_form: list[str] = []
    topology_id = 0
    for group, topologies in enumerate(groups):
        for topology in topologies:
            lcag, names = topology_to_lcag(topology)
            events = generate_events(topology, n_events_per_topology, int(rng.integers(2**31)))
            p4 = np.stack([events[name] for name in names], axis=1)

            # One permutation per event, applied to the rows of both encodings and
            # conjugated onto the LCAG, so leaf order carries no information.
            per_event = [shuffle_leaves(rng, p4[i], lcag) for i in range(len(p4))]
            p4 = np.stack([f for f, _ in per_event])
            labels = np.stack([m for _, m in per_event])

            order = rng.permutation(len(p4))
            start = 0
            for name, size in _split_sizes(group, len(p4), val_frac, test_frac).items():
                if size == 0:
                    continue
                take = order[start : start + size]
                start += size
                padded: dict[str, np.ndarray] = {}
                for enc in ENCODINGS:
                    features, padded["lcag"] = pad_events(
                        featurize(p4[take], encoding=enc), labels[take], max_fsps
                    )
                    padded[f"features_{enc}"] = features
                padded["n_fsps"] = np.full(size, count_fsps(topology), dtype=np.int16)
                padded["topology_id"] = np.full(size, topology_id, dtype=np.int32)
                parts[name].append(padded)

            topology_group.append(group)
            topology_form.append(canonical_form(topology))
            topology_id += 1

    splits = {
        name: {key: np.concatenate([p[key] for p in chunks]) for key in chunks[0]}
        for name, chunks in parts.items()
        if chunks
    }
    missing = set(SPLITS) - set(splits)
    if missing:
        raise ValueError(f"splits {sorted(missing)} came out empty; raise n_events_per_topology")

    scales = {
        enc: normalization_scales(splits["train"][f"features_{enc}"], enc) for enc in ENCODINGS
    }
    for split in splits.values():
        for enc in ENCODINGS:
            key = f"features_{enc}"
            split[key] = apply_normalization(split[key], scales[enc], enc)

    # Class 0 is never a target: any two real leaves share at least the root, so
    # every scored cell is at least 1. It stays in the output space because a
    # prediction of 0 means "unrelated", which the valid-tree metric interprets.
    n_classes = int(max(split["lcag"].max() for split in splits.values())) + 1
    meta = {
        "seed": seed,
        "max_fsps": max_fsps,
        "max_depth": max_depth,
        "n_classes": n_classes,
        "n_groups": n_groups,
        "scales": scales,
        "topology_group": topology_group,
        "topology_form": topology_form,
        "group_splits": [list(s) for s in _GROUP_SPLITS[:n_groups]],
        "counts": {name: len(split["lcag"]) for name, split in splits.items()},
    }
    return splits, meta


def _to_npz(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialise one split to compressed npz bytes.

    ``allow_pickle=False`` keeps object arrays out of the dataset artifacts: every
    stored array is numeric, and anything else is a mistake worth raising on.
    """
    buffer = io.BytesIO()
    np.savez_compressed(buffer, allow_pickle=False, **arrays)
    return buffer.getvalue()


def load_split(ref: dict[str, Any]) -> dict[str, np.ndarray]:
    """Read a split artifact back into arrays.

    Args:
        ref: Artifact reference produced by :func:`build_dataset`.

    Returns:
        The arrays stored for that split.
    """
    with np.load(fluksio.load_artifact(ref)) as data:
        return {key: data[key] for key in data.files}


@node(
    requires=[
        Port("seed", "int"),
        Port("n_topologies", "int"),
        Port("n_events_per_topology", "int"),
        Port("min_fsps", "int"),
        Port("max_fsps", "int"),
    ],
    provides=[
        Port("dataset_train", "artifact"),
        Port("dataset_val", "artifact"),
        Port("dataset_test", "artifact"),
        Port("dataset_meta", "json"),
    ],
)
def build_dataset(
    *,
    seed: int,
    n_topologies: int = 10,
    n_events_per_topology: int = 1000,
    min_fsps: int = 3,
    max_fsps: int = 8,
    max_depth: int = 4,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    isp_weight: float = 1.0,
) -> dict[str, Any]:
    """Generate the dataset and store one artifact per split.

    Args:
        seed: Master seed; every random stream is derived from it.
        n_topologies: Topologies per group.
        n_events_per_topology: Unweighted events generated per topology.
        min_fsps: Smallest final-state particle count.
        max_fsps: Largest final-state particle count and the padded width.
        max_depth: Maximum topology depth.
        val_frac: Validation fraction of a group-A topology's events.
        test_frac: Test fraction of a group-A topology's events.
        isp_weight: Relative weight of the intermediate-state mass pool.

    Returns:
        Artifact references for the three splits plus the metadata record.
    """
    splits, meta = assemble_dataset(
        seed=seed,
        n_topologies=n_topologies,
        n_events_per_topology=n_events_per_topology,
        min_fsps=min_fsps,
        max_fsps=max_fsps,
        max_depth=max_depth,
        val_frac=val_frac,
        test_frac=test_frac,
        isp_weight=isp_weight,
    )
    refs = {
        f"dataset_{name}": fluksio.save_artifact(_to_npz(arrays), f"{name}.npz")
        for name, arrays in splits.items()
    }
    return {**refs, "dataset_meta": meta}


def dataset_statistics(
    splits: dict[str, dict[str, np.ndarray]], meta: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Summarise the dataset and render the phase-1 verification figures.

    The angular marginals are the point of the exercise: the ROADMAP predicts
    kinematic features cluster the encoding angles, which is the regime where the
    unflattening theory makes falsifiable predictions. Circular variance near 0
    means clustered, near 1 means spread.

    Args:
        splits: Split arrays from :func:`assemble_dataset`.
        meta: The accompanying metadata record.

    Returns:
        A ``(stats, figures)`` pair; ``stats`` is json-serialisable and ``figures``
        maps a file name to PNG bytes.

    Raises:
        ValueError: If a split is missing from ``splits``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if set(SPLITS) - set(splits):
        raise ValueError(f"expected splits {SPLITS}, got {sorted(splits)}")

    stats: dict[str, Any] = {"counts": {n: len(s["lcag"]) for n, s in splits.items()}}

    train = splits["train"]["features_angles"]
    real = splits["train"]["features_angles"][..., 2] > 0.0
    theta, phi = train[..., 0][real], train[..., 1][real]
    stats["angles"] = {
        # 1 - |mean of unit vectors|: 0 is perfectly clustered, 1 is uniform.
        "theta_circular_variance": float(1.0 - abs(np.exp(1j * theta).mean())),
        "phi_circular_variance": float(1.0 - abs(np.exp(1j * phi).mean())),
        "theta_mean": float(theta.mean()),
        "energy_mean": float(train[..., 2][real].mean()),
    }

    labels = splits["train"]["lcag"]
    scored = labels[labels != -1]
    counts = np.bincount(scored, minlength=meta["n_classes"])
    stats["class_counts"] = counts.tolist()
    stats["class_balance"] = (counts / max(counts.sum(), 1)).round(6).tolist()

    group = np.asarray(meta["topology_group"])
    stats["split_integrity"] = {
        name: sorted({int(g) for g in group[np.unique(split["topology_id"])]})
        for name, split in splits.items()
    }
    stats["split_integrity_ok"] = all(
        all(name in _GROUP_SPLITS[g] for g in stats["split_integrity"][name]) for name in SPLITS
    )

    figures: dict[str, bytes] = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
    for ax, values, title in (
        (axes[0], theta, r"$\theta$ (polar angle)"),
        (axes[1], phi, r"$\varphi$ (azimuth)"),
        (axes[2], train[..., 2][real], "scaled energy"),
    ):
        ax.hist(values, bins=80)
        ax.set_title(title)
        ax.set_ylabel("events")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    figures["angle_marginals.png"] = buffer.getvalue()

    fig, ax = plt.subplots(figsize=(4, 3.2), constrained_layout=True)
    ax.bar(np.arange(len(counts)), counts)
    ax.set_xlabel("LCAG class (generations to the common ancestor)")
    ax.set_ylabel("cells")
    ax.set_yscale("log")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    figures["class_balance.png"] = buffer.getvalue()
    return stats, figures


@node(
    requires=[
        Port("dataset_train", "artifact"),
        Port("dataset_val", "artifact"),
        Port("dataset_test", "artifact"),
        Port("dataset_meta", "json"),
    ],
    provides=[Port("stats", "json"), Port("figures", "list")],
)
def dataset_stats(
    *,
    dataset_train: dict[str, Any],
    dataset_val: dict[str, Any],
    dataset_test: dict[str, Any],
    dataset_meta: dict[str, Any],
) -> dict[str, Any]:
    """Compute dataset statistics and attach the verification figures to the run.

    Args:
        dataset_train: Training split artifact reference.
        dataset_val: Validation split artifact reference.
        dataset_test: Test split artifact reference.
        dataset_meta: Metadata record from :func:`build_dataset`.

    Returns:
        The statistics record and the list of figure artifact references.
    """
    splits = {
        "train": load_split(dataset_train),
        "val": load_split(dataset_val),
        "test": load_split(dataset_test),
    }
    stats, figures = dataset_statistics(splits, dataset_meta)
    refs = [
        fluksio.save_artifact(payload, name, media_type="image/png")
        for name, payload in figures.items()
    ]
    return {"stats": stats, "figures": refs}
