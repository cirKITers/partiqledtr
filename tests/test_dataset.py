"""Dataset assembly: shapes, reproducibility, split integrity and the -1 convention."""

import numpy as np
import pytest

from partiqledtr.data.dataset import ENCODINGS, SPLITS, assemble_dataset, dataset_statistics

pytestmark = pytest.mark.gen

SMALL = {
    "n_topologies": 2,
    "n_events_per_topology": 40,
    "min_fsps": 3,
    "max_fsps": 5,
    "max_depth": 3,
}


@pytest.fixture(scope="module")
def dataset():
    return assemble_dataset(seed=0, **SMALL)


def test_splits_have_the_documented_arrays_and_shapes(dataset):
    splits, meta = dataset
    assert set(splits) == set(SPLITS)

    for split in splits.values():
        n = len(split["lcag"])
        assert split["features_angles"].shape == (n, meta["max_fsps"], 3)
        assert split["features_cartesian"].shape == (n, meta["max_fsps"], 4)
        assert split["lcag"].shape == (n, meta["max_fsps"], meta["max_fsps"])
        assert split["lcag"].dtype == np.int8
        assert split["n_fsps"].shape == (n,)
        assert split["topology_id"].shape == (n,)


def test_padding_and_ignore_conventions(dataset):
    """Padded feature rows are zero, padded and diagonal labels are -1."""
    splits, meta = dataset
    for split in splits.values():
        n_fsps, lcag = split["n_fsps"], split["lcag"]
        for encoding in ENCODINGS:
            features = split[f"features_{encoding}"]
            for i in range(min(len(features), 20)):
                assert np.all(features[i, n_fsps[i] :] == 0.0)

        diagonal = np.arange(meta["max_fsps"])
        assert np.all(lcag[:, diagonal, diagonal] == -1)
        for i in range(min(len(lcag), 20)):
            real = n_fsps[i]
            assert np.all(lcag[i, real:] == -1)
            assert np.all(lcag[i, :, real:] == -1)
            # Every off-diagonal pair of real leaves shares at least the root.
            block = lcag[i, :real, :real]
            assert np.all(block[~np.eye(real, dtype=bool)] >= 1)


def test_labels_are_symmetric(dataset):
    splits, _ = dataset
    lcag = splits["train"]["lcag"]
    assert np.array_equal(lcag, np.transpose(lcag, (0, 2, 1)))


def test_assembly_is_reproducible_and_seed_sensitive():
    same = assemble_dataset(seed=0, **SMALL)[0]
    other = assemble_dataset(seed=1, **SMALL)[0]
    reference = assemble_dataset(seed=0, **SMALL)[0]

    for name in SPLITS:
        for key in reference[name]:
            assert np.array_equal(same[name][key], reference[name][key])
    assert not np.array_equal(other["train"]["lcag"], reference["train"]["lcag"])


def test_known_unknown_split_integrity(dataset):
    """Group A feeds every split, B only val and test, C only test (D16)."""
    splits, meta = dataset
    group = np.asarray(meta["topology_group"])
    groups_in = {
        name: {int(g) for g in group[np.unique(split["topology_id"])]}
        for name, split in splits.items()
    }

    assert groups_in["train"] == {0}
    assert groups_in["val"] == {0, 1}
    assert groups_in["test"] == {0, 1, 2}


def test_topologies_are_pairwise_non_isomorphic(dataset):
    _, meta = dataset
    forms = meta["topology_form"]
    assert len(set(forms)) == len(forms)


def test_normalisation_is_fitted_on_train_and_leaves_energy_non_negative(dataset):
    splits, meta = dataset
    train = splits["train"]["features_angles"]
    real = train[..., 2] > 0.0

    # The energy scale is the training mean, so the normalised training mean is 1.
    assert train[..., 2][real].mean() == pytest.approx(1.0)
    for split in splits.values():
        for encoding in ENCODINGS:
            assert np.all(split[f"features_{encoding}"][..., -1] >= 0.0)
    assert set(meta["scales"]) == set(ENCODINGS)


def test_statistics_report_marginals_balance_and_integrity(dataset):
    splits, meta = dataset
    stats, figures = dataset_statistics(splits, meta)

    assert stats["split_integrity_ok"] is True
    assert set(figures) == {"angle_marginals.png", "class_balance.png"}
    assert all(payload.startswith(b"\x89PNG") for payload in figures.values())

    assert len(stats["class_counts"]) == meta["n_classes"]
    assert stats["class_counts"][0] == 0  # class 0 is never a label (D49)
    assert sum(stats["class_counts"]) > 0
    for key in ("theta_circular_variance", "phi_circular_variance"):
        assert 0.0 <= stats["angles"][key] <= 1.0


def test_assembly_validates_its_arguments():
    with pytest.raises(ValueError, match="n_topologies"):
        assemble_dataset(seed=0, n_topologies=0)
    with pytest.raises(ValueError, match="min_fsps"):
        assemble_dataset(seed=0, min_fsps=1)
    with pytest.raises(ValueError, match="val_frac"):
        assemble_dataset(seed=0, val_frac=0.6, test_frac=0.6)
