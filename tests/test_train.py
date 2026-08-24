import io
from itertools import pairwise
from typing import Any

import fluksio
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from partiqledtr.train import (
    batches,
    build_model,
    evaluate,
    evaluate_split,
    fit,
    lcag_loss,
    npz_config,
    npz_to_state,
    state_to_npz,
    train_model,
)

N, L, F, C, SEED = 8, 4, 3, 3, 0


def _split(seed: int = 0, n: int = N, n_fsps: int = L):
    """A synthetic dataset split: the arrays load_split returns, built directly.

    Labels are symmetric with a -1 diagonal and -1 on padded rows/columns, and never
    use class 0 -- the generated dataset's convention (DECISIONS.md D17, D49). Half the
    events carry topology 1, so a known/unknown split has both sides populated.
    """
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.integers(1, C, size=(n, L, L)), 1)
    lcag = (upper + np.swapaxes(upper, 1, 2)).astype(np.int8)
    lcag[:, n_fsps:, :] = -1
    lcag[:, :, n_fsps:] = -1
    np.einsum("nii->ni", lcag)[...] = -1
    features = rng.normal(size=(n, L, F))
    features[:, n_fsps:] = 0.0
    return {
        "features_angles": features,
        "features_cartesian": rng.normal(size=(n, L, 4)),
        "lcag": lcag,
        "n_fsps": np.full(n, n_fsps, dtype=np.int16),
        "topology_id": (np.arange(n) >= n // 2).astype(np.int32),
    }


def _logit_batch(seed: int = 1, n_fsps: int = L - 1):
    """Random logits with a matching label matrix that has real padding in it."""
    split = _split(seed, n_fsps=n_fsps)
    rng = np.random.default_rng(seed + 100)
    logits = jnp.asarray(rng.normal(size=(N, L, L, C)), dtype=jnp.float32)
    return logits, jnp.asarray(split["lcag"], dtype=jnp.int32), jnp.ones(C, dtype=jnp.float32)


def _model(seed: int = SEED, **kwargs):
    settings: dict[str, Any] = {
        "model": "gnn",
        "frontend": "none",
        "n_features": F,
        "n_classes": C,
        "dim": 8,
        "n_blocks": 2,
    }
    return build_model(**(settings | kwargs), seed=seed)


def _call(module, split, encoding: str = "angles"):
    x = jnp.asarray(split[f"features_{encoding}"], dtype=jnp.float32)
    mask = jnp.asarray(np.arange(L) < split["n_fsps"][:, None])
    return module(x, mask)


def test_masked_cells_contribute_exactly_nothing():
    logits, labels, weights = _logit_batch()
    ignored = (labels == -1)[..., None]
    assert bool(ignored.any())  # not vacuous: the batch really is padded

    noise = jnp.asarray(np.random.default_rng(3).normal(size=logits.shape) * 10.0, jnp.float32)
    perturbed = jnp.where(ignored, noise, logits)
    assert not jnp.array_equal(perturbed, logits)
    # Bit-identical, not close: a masked cell is multiplied by zero, never averaged in.
    assert lcag_loss(perturbed, labels, weights) == lcag_loss(logits, labels, weights)

    gradient = jax.grad(lcag_loss)(logits, labels, weights)
    assert jnp.count_nonzero(jnp.where(ignored, gradient, 0.0)) == 0
    assert jnp.count_nonzero(gradient) > 0


def test_fully_masked_batch_is_finite():
    """The -1 clamp has to hold even when every label is the sentinel."""
    logits, _, weights = _logit_batch()
    labels = jnp.full(logits.shape[:-1], -1, dtype=jnp.int32)
    loss = lcag_loss(logits, labels, weights)
    assert float(loss) == 0.0
    assert jnp.all(jnp.isfinite(jax.grad(lcag_loss)(logits, labels, weights)))


def test_class_weighting_matches_hand_computation():
    logits = jnp.asarray([[0.0, 1.0, 2.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    labels = jnp.asarray([1, 2, -1])
    weights = jnp.asarray([0.0, 2.0, 5.0])

    values = np.asarray(logits)[:2]
    entropy = np.log(np.exp(values).sum(axis=1)) - values[[0, 1], [1, 2]]
    # Weighted by the label's class, averaged over the two scored cells only -- the
    # third row is ignored and does not enter the denominator.
    expected = (2.0 * entropy[0] + 5.0 * entropy[1]) / 2.0
    assert float(lcag_loss(logits, labels, weights)) == pytest.approx(expected, rel=1e-6)


def test_overfits_a_tiny_batch(capsys):
    """The gradient path end to end: a model with capacity to spare must memorise 8 events."""
    split = _split()
    records = []
    loop = train_model(split, split, {"n_classes": C}, seed=SEED, dim=16, n_blocks=2, epochs=120,
                       batch_size=N, lr=5e-3)  # fmt: skip
    while True:
        try:
            records.append(next(loop))
        except StopIteration as stop:
            _, final = stop.value
            break

    losses = [record["train_loss"] for record in records]
    with capsys.disabled():
        print(f"\noverfit: loss {losses[0]:.4f} -> {losses[-1]:.5f}, {len(losses)} epochs")
    assert all(b < a for a, b in pairwise(losses[:10]))  # smooth early descent
    assert losses[-1] < 0.05 * losses[0]
    assert records[-1]["val_accuracy"] == 1.0
    assert records[-1]["val_perfect"] == 1.0
    # Everything build_model needs to rebuild the module, including the quantum
    # knobs, which the classical arms record but ignore.
    assert final["config"] == {
        "encoding": "angles",
        "model": "gnn",
        "frontend": "none",
        "n_features": F,
        "n_classes": C,
        "dim": 16,
        "n_blocks": 2,
        "ansatz": "XY_Brickwork",
        "n_layers": 2,
    }
    assert final["n_params"] > 0
    assert set(records[0]) == {
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_perfect",
        "g_purity",
    }
    # A classical model encodes no quantum state, so it reports no purity.
    assert np.isnan(records[0]["g_purity"])


def test_train_model_rejects_a_split_smaller_than_a_batch():
    split = _split()
    with pytest.raises(ValueError, match="fewer than batch_size"):
        next(train_model(split, split, {"n_classes": C}, seed=SEED, batch_size=N + 1))
    with pytest.raises(ValueError, match="epochs"):
        next(train_model(split, split, {"n_classes": C}, seed=SEED, epochs=0, batch_size=N))


def test_checkpoint_round_trip():
    split = _split()
    saved, loaded = _model(frontend="mlp"), _model(frontend="mlp", seed=SEED + 7)
    before = _call(saved, split)
    assert not jnp.array_equal(_call(loaded, split), before)

    npz_to_state(loaded, state_to_npz(saved))
    assert jnp.array_equal(_call(loaded, split), before)
    equal = jax.tree.leaves(
        jax.tree.map(
            lambda a, b: bool(jnp.array_equal(a, b)),
            nnx.state(saved, nnx.Param),
            nnx.state(loaded, nnx.Param),
        )
    )
    assert equal and all(equal)


def test_checkpoint_carries_the_model_configuration():
    config = {"encoding": "angles", "model": "gnn", "n_features": F}
    assert npz_config(state_to_npz(_model(), config)) == config
    with pytest.raises(ValueError, match="no model configuration"):
        npz_config(state_to_npz(_model()))


@pytest.mark.parametrize(
    ("other", "match"),
    [({"dim": 16}, "shape"), ({"model": "mlp"}, "does not match")],
)
def test_npz_to_state_rejects_a_checkpoint_from_a_different_model(other, match):
    with pytest.raises(ValueError, match=match):
        npz_to_state(_model(**other), state_to_npz(_model()))


def test_batches_are_deterministic_disjoint_and_drop_the_tail():
    first = list(batches(np.random.default_rng(SEED), 10, 4))
    again = list(batches(np.random.default_rng(SEED), 10, 4))
    assert len(first) == 2  # the two leftover indices are dropped, so shapes stay static
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))

    drawn = np.concatenate(first)
    assert sorted(drawn.tolist()) == sorted(set(drawn.tolist()))  # each index at most once
    assert set(drawn.tolist()) <= set(range(10))
    assert not np.array_equal(np.concatenate(list(batches(np.random.default_rng(1), 10, 4))), drawn)
    assert list(batches(np.random.default_rng(SEED), 3, 4)) == []
    with pytest.raises(ValueError, match="batch_size"):
        list(batches(np.random.default_rng(SEED), 10, 0))


def test_build_model_rejects_unknown_registry_keys():
    with pytest.raises(ValueError, match="unknown model 'nri'; valid models are"):
        _model(model="nri")
    with pytest.raises(ValueError, match="unknown frontend 'linear'; valid frontends are"):
        _model(frontend="linear")
    # The message names the keys that would have worked.
    with pytest.raises(ValueError, match=r"gnn.*mlp"):
        _model(model="nri")
    with pytest.raises(ValueError, match=r"none"):
        _model(frontend="linear")


def test_build_model_attaches_the_frontend_and_honours_n_blocks():
    assert _model(frontend="none").frontend is None
    assert _model(frontend="mlp").frontend is not None
    assert len(_model(n_blocks=4).blocks) == 4
    # MLPBaseline takes no n_blocks; passing one must not raise.
    assert _model(model="mlp", n_blocks=4) is not None


def test_evaluate_split_returns_the_documented_keys():
    split, module = _split(), _model()
    plain = evaluate_split(module, split, encoding="angles", batch_size=4)
    assert set(plain) == {"accuracy", "accuracy_primary", "perfect", "perfect_primary"}

    full = evaluate_split(
        module, split, encoding="angles", batch_size=4, weights=np.ones(C), valid_trees=True
    )
    assert set(full) == set(plain) | {"loss", "valid_tree"}
    assert all(0.0 <= full[key] <= 1.0 for key in plain)
    assert full["loss"] > 0.0
    # Class 0 is never a label here, so the _primary variants coincide (D49).
    assert full["accuracy"] == full["accuracy_primary"]


def test_evaluate_split_reports_nan_for_an_empty_split():
    split = {key: array[:0] for key, array in _split().items()}
    metrics = evaluate_split(split=split, module=_model(), encoding="angles", weights=np.ones(C))
    assert all(np.isnan(value) for value in metrics.values())


def test_evaluate_split_rejects_an_unknown_encoding():
    with pytest.raises(ValueError, match="unknown encoding 'polar'"):
        evaluate_split(_model(), _split(), encoding="polar")


@pytest.fixture
def artifacts(monkeypatch, tmp_path):
    """Stand in for the Fluksio artifact store, which only exists inside a running node."""

    def save(source, name="", media_type="application/octet-stream"):
        path = tmp_path / (name or "artifact")
        path.write_bytes(source)
        return {"path": str(path)}

    monkeypatch.setattr(fluksio, "save_artifact", save)
    monkeypatch.setattr(fluksio, "load_artifact", lambda ref: ref["path"])
    return save


def test_fit_and_evaluate_run_end_to_end(artifacts):
    """The node bodies, including the known/unknown topology split of D16."""
    buffer = io.BytesIO()
    np.savez(buffer, **_split())
    dataset = artifacts(buffer.getvalue(), "split.npz")
    # Topology 0 was trained on, topology 1 is in group 2 and was not.
    meta = {"n_classes": C, "topology_group": [0, 2]}

    loop = fit(dataset_train=dataset, dataset_val=dataset, dataset_meta=meta, seed=SEED,
               dim=8, n_blocks=2, epochs=3, batch_size=N, lr=5e-3)  # fmt: skip
    streamed = []
    while True:
        try:
            streamed.append(next(loop))
        except StopIteration as stop:
            outputs = stop.value
            break

    assert [record["epoch"] for record in streamed] == [0, 1, 2]
    assert set(outputs) == {"checkpoint", "final_metrics"}

    scores = evaluate(checkpoint=outputs["checkpoint"], dataset_test=dataset, dataset_meta=meta)[
        "test_metrics"
    ]
    assert set(scores) == {"overall", "known", "unknown"}
    assert scores["overall"]["n_events"] == N
    assert scores["known"]["n_events"] == N // 2
    assert scores["unknown"]["n_events"] == N - N // 2
    assert "valid_tree" in scores["overall"]
    # The checkpoint alone says how to rebuild the model, so evaluate reproduces fit's
    # final validation accuracy on the same events.
    assert scores["overall"]["accuracy"] == pytest.approx(outputs["final_metrics"]["val_accuracy"])
