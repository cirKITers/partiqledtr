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
        "angle_map": "pair_polar",
        # Carried in the config, not beside it, so `evaluate` rebuilds the arm it
        # scored rather than silently dropping the rotation (D81).
        "whitening": None,
    }
    assert final["n_params"] > 0
    # A classical model encodes no quantum state, so it omits the purity key
    # entirely: a float port accepts neither NaN nor None (D75).
    assert set(records[0]) == {"epoch", "train_loss", "val_loss", "val_accuracy", "val_perfect"}


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
    assert set(full) == set(plain) | {"loss", "valid_tree", "valid_tree_strict"}
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


def test_the_whitening_arm_is_opt_in():
    """`whiten` gates the fixed-preconditioning arm, and defaults to off.

    The rotation is always wired into `fit` by the flow, so without this switch
    every run would silently be whitened and the ROADMAP's "raw" arm would be
    unreachable -- which would invalidate the phase-4 comparison rather than
    merely break it. The default has to stay off.
    """
    import inspect

    from partiqledtr.pipeline import train as train_flow

    assert inspect.signature(fit).parameters["whiten"].default is False

    ports = {port.name: port for port in train_flow.inputs}
    assert "whiten" in ports, "the flow must expose `whiten` so the arm is sweepable"
    assert ports["whiten"].initial is False


def test_node_payloads_are_port_legal():
    """Whatever the nodes publish must satisfy the declared ports.

    Fluksio rejects a non-finite float on any port rather than letting one leave
    the engine as unparseable JSON, and it does so however deeply the value sits.
    Metrics over an empty subset are legitimately undefined, so they have to
    travel as None (json) or be omitted (float) -- checked here against the real
    port specs, because the failure would otherwise only appear mid-run.
    """
    from fluksio.flow.messages import DType, MessageSpec

    from partiqledtr.train import jsonable

    undefined = {
        "overall": {"n_events": 4, "accuracy": 0.5, "valid_tree": float("nan")},
        "unknown": {"n_events": 0, "accuracy": float("nan"), "perfect": float("nan")},
    }
    record = MessageSpec(name="test_metrics", port="test_metrics", dtype=DType.JSON)
    with pytest.raises(TypeError, match="cannot travel as JSON"):
        record.check(undefined)
    record.check(jsonable(undefined))  # must not raise
    assert jsonable(undefined)["unknown"]["accuracy"] is None
    assert jsonable(undefined)["overall"]["accuracy"] == 0.5

    stream = MessageSpec(name="g_purity", port="g_purity", dtype=DType.FLOAT)
    stream.check(0.8)
    for rejected in (float("nan"), None):
        with pytest.raises(TypeError):
            stream.check(rejected)


def test_frontend_does_not_reseed_the_model():
    """Attaching a front end must add one, not re-initialise everything (D84).

    Built from a single rng stream, the front end's own draws shift every later
    draw, so the raw and learned arms of the phase-4 study would have differed by a
    full re-initialisation as well as by the front end -- a confound in exactly the
    comparison the study is about.
    """
    common: dict[str, Any] = {"model": "gnn", "n_features": F, "n_classes": C, "dim": 8, "seed": 3}
    plain = build_model(frontend="none", **common)
    with_frontend = build_model(frontend="mlp", **common)

    attached = dict(nnx.to_flat_state(nnx.state(with_frontend, nnx.Param)))
    shared = [
        (path, leaf)
        for path, leaf in nnx.to_flat_state(nnx.state(plain, nnx.Param))
        if path[0] != "frontend"
    ]
    assert shared, "the baseline must have parameters outside the front end"
    for path, leaf in shared:
        np.testing.assert_array_equal(np.asarray(leaf[...]), np.asarray(attached[path][...]))


def test_parameter_matched_arm_is_derived_not_asserted():
    """The documented classical baseline has to actually match the quantum one (D86).

    The write-up claimed dim=8 was parameter-matched; it is sixteen times larger.
    matched_dim computes the width instead, so the claim is checkable.
    """
    from functools import partial

    from partiqledtr.models import matched_dim, n_params

    quantum = build_model(model="qfm", frontend="none", n_features=4, n_classes=C)
    target = n_params(quantum)
    build = partial(build_model, model="gnn", frontend="none", n_features=4, n_classes=C)

    dim = matched_dim(target, build, n_blocks=3)
    matched = n_params(build(dim=dim, n_blocks=3))
    assert abs(matched - target) / target < 0.1
    # ... and the number the docs used to carry is nowhere near.
    assert n_params(build(dim=8, n_blocks=3)) > 10 * target


def test_checkpoint_round_trip_preserves_the_whitening_arm():
    """A whitened checkpoint must not be rebuilt as the raw arm (D81).

    The rotation is not an nnx.Param, so before it moved into the config `evaluate`
    silently scored the whitened arms of the phase-4 matrix on unrotated angles.
    """
    from partiqledtr.data.whitening import sample_rotation
    from partiqledtr.models.qfm import QFMConstellation

    rotation = sample_rotation(np.random.default_rng(SEED))
    config: dict[str, Any] = {
        "model": "qfm",
        "frontend": "none",
        "n_features": 4,
        "n_classes": C,
        "whitening": rotation.tolist(),
    }
    module = build_model(**config)
    payload = state_to_npz(module, config)

    rebuilt = build_model(**npz_config(payload))
    npz_to_state(rebuilt, payload)
    assert isinstance(rebuilt, QFMConstellation)
    assert rebuilt.whitening is not None
    np.testing.assert_allclose(np.asarray(rebuilt.whitening), rotation, atol=1e-6)

    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, L, 4)))
    mask = jnp.ones((2, L), dtype=bool)
    np.testing.assert_allclose(module(x, mask), rebuilt(x, mask), atol=1e-6)


def test_quantum_arm_streams_the_angle_distribution_beside_the_purity():
    """Both observables have to travel, because either alone is ambiguous (D92).

    The purity says how trainable the encoded state is; the angle statistics say
    what its distribution looks like, which is what separates a rescue that spreads
    the angles from one that pins them at pi/2.
    """
    split = _split()
    cartesian = dict(split)
    cartesian["features_cartesian"] = np.concatenate(
        [split["features_angles"], np.ones((len(split["lcag"]), L, 1), np.float32)], axis=-1
    )
    loop = train_model(cartesian, cartesian, {"n_classes": C}, seed=SEED, model="qfm",
                       encoding="cartesian", epochs=2, batch_size=N, lr=1e-3)  # fmt: skip
    records = []
    while True:
        try:
            records.append(next(loop))
        except StopIteration as stop:
            _, final = stop.value
            break

    assert {"g_purity", "tv_uniform", "mean_sin2"} <= set(records[0])
    # Per-site vectors ride in final_metrics; the stream carries only the site mean.
    for key in ("angle_stats_initial", "angle_stats_final"):
        assert len(final[key]["tv_uniform"]) == 4
        assert len(final[key]["mean_sin2"]) == 4
    assert final["angle_stats_final"]["n_bins"] == [36.0]


def test_classical_arm_omits_the_angle_ports():
    """A model that encodes no quantum state has no angle distribution to report."""
    split = _split()
    loop = train_model(split, split, {"n_classes": C}, seed=SEED, dim=8, n_blocks=1,
                       epochs=1, batch_size=N)  # fmt: skip
    records = []
    while True:
        try:
            records.append(next(loop))
        except StopIteration as stop:
            _, final = stop.value
            break
    assert not {"g_purity", "tv_uniform", "mean_sin2"} & set(records[0])
    assert "angle_stats_final" not in final
