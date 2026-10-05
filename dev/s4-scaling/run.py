"""Run the six-qubit cycle, ladder, and odd-chord scaling grid.

``--encodings`` prices three-angle charts before training. ``--fluksio`` runs
the grid on registered datasets; ``--report`` summarises results.

Usage: ``python dev/s4-scaling/run.py [--encodings | --fluksio | --report]``.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

STUDY = Path(__file__).resolve().parent
#: The s2 export of the pinned generate run, reused unchanged (module docstring).
DATA = STUDY.parent / "s2-expressivity" / "data"
OUT = STUDY / "results"

#: The pinned generate run `DATA` was exported from; `--fluksio` resolves the
#: dataset artifacts from it.
GENERATE_RUN = "1787760161002-8bde9189"
#: Dataset seed -> generate run id, shared with s3.
DATASETS = STUDY.parent / "datasets.json"

#: The day the in-process smoke seeds landed. Their records carry durations but no
#: timestamps, so the import is stamped with the day.
INPROCESS_DATE = "2026-09-02T00:00:00+00:00"

#: The chart the grid encodes, set by the ``--encodings`` gate (results/encoding.json).
CHART = "pair_polar_boost"

ARMS = ("XY_Cycle", "XY_Ladder", "XY_OddChord")

BASE: dict[str, Any] = {
    "model": "qfm",
    "n_qubits": 6,
    "n_channels": 4,
    "lr_qfm": 1e-2,
    "encoding": "cartesian",
    "angle_map": CHART,
}

#: Trace keys kept per epoch, matching s3 so the correlation analysis carries over.
TRACE_KEYS = ("epoch", "train_loss", "val_loss", "g_purity", "tv_uniform", "mean_sin2")


def cells() -> list[dict[str, Any]]:
    """The smoke grid: three graph arms x {none, mlp} on the gated chart."""
    return [{**BASE, "ansatz": arm, "preconditioner": f} for arm in ARMS for f in ("none", "mlp")]


def gate_cell() -> list[dict[str, Any]]:
    """The cost-gate cell: the largest closure (1020 words) with the preconditioner."""
    return [{**BASE, "ansatz": "XY_OddChord", "preconditioner": "mlp"}]


def price_encodings(n_pairs: int = 4096, seed: int = 0) -> dict[str, Any]:
    """Price the three-angle charts in g-purity on real kinematics, per arm.

    The chart gate: mean purity of the induced angle law over sampled real
    edges, against each floor-free arm's own basis and its uniform-prior mean.
    ``W = I`` (hamming-diagonal), the study's encoding-weight regime.
    """
    import jax.numpy as jnp
    import numpy as np

    from partiqledtr.analysis import product_state_purity, uniform_prior_mean
    from partiqledtr.data.whitening import _sample_pairs
    from partiqledtr.models.qfm import ANGLE_MAPS

    with np.load(DATA / "train.npz") as data:
        features, n_fsps = data["features_cartesian"], data["n_fsps"]
    events, pairs = _sample_pairs(np.random.default_rng(seed), n_fsps, n_pairs)

    report: dict[str, Any] = {"n_pairs": n_pairs, "charts": {}}
    for chart in ("pair_polar_boost", "pair_polar_mass", "pair_polar_theta"):
        angles = ANGLE_MAPS[chart](jnp.asarray(features[events]))
        taken = jnp.take_along_axis(angles, jnp.asarray(pairs)[:, :, None], axis=1)
        edge = taken.reshape(n_pairs, 6)
        arms = {}
        for arm in ARMS:
            mu = uniform_prior_mean(arm, 6)
            purity = np.asarray(product_state_purity(edge, arm))
            arms[arm] = {
                "mean_purity": float(purity.mean()),
                "ratio": float(purity.mean() / mu),
                "below_threshold": float((purity < mu / 2).mean()),
                "uniform_mean": mu,
            }
        report["charts"][chart] = arms
    return report


def load(name: str) -> dict[str, Any]:
    """Read one exported dataset split."""
    import numpy as np

    with np.load(DATA / f"{name}.npz") as data:
        return {key: data[key] for key in data.files}


def one_cell(settings: dict[str, Any], epochs: int) -> dict[str, Any]:
    """Fit one cell exactly as the ``train`` flow's nodes would."""
    import numpy as np

    from partiqledtr.analysis import dla_check
    from partiqledtr.train import evaluate_split, train_model

    started = time.monotonic()
    seed = settings.pop("seed")
    encoding = settings.get("encoding", "cartesian")
    train, val, test = load("train"), load("val"), load("test")
    meta = json.loads((DATA / "meta.json").read_text())

    # The arm's algebra, recorded before the fit -- the property the flow enforces
    # by wiring `dla_report` upstream of `fit`, kept here by doing it first. The
    # exponential closures need the cap above dim su(2^6) = 4095.
    certificate = dla_check(settings.get("ansatz"), n_qubits=settings["n_qubits"], max_dim=4200)

    loop = train_model(train, val, meta, seed=seed, epochs=epochs, **settings)
    trace: list[dict[str, float]] = []
    while True:
        try:
            trace.append(next(loop))
        except StopIteration as stop:
            module, final = stop.value
            break

    group = np.asarray(meta["topology_group"])
    known = group[test["topology_id"]] == 0
    scores = {
        name: evaluate_split(
            module,
            {key: array[selection] for key, array in test.items()},
            encoding=encoding,
            valid_trees=True,
        )
        for name, selection in (("known", known), ("unknown", ~known))
    }
    return {
        "cell": {**settings, "seed": seed},
        "final_metrics": final,
        "test_metrics": scores,
        "dla_report": certificate,
        "trace": [{k: r[k] for k in TRACE_KEYS if k in r} for r in trace],
        "seconds": round(time.monotonic() - started, 1),
    }


def _safe(args: tuple[dict[str, Any], int]) -> dict[str, Any]:
    """Run a cell, returning the failure rather than killing the pool."""
    settings, epochs = args
    try:
        return one_cell(dict(settings), epochs)
    except Exception:  # a single bad cell must not cost the study
        return {"cell": settings, "error": traceback.format_exc(limit=4)}


def run(
    block: list[dict[str, Any]], filename: str, *, seeds: int, jobs: int, epochs: int, out: Path
) -> list[dict[str, Any]]:
    """Run every cell x seed of one block, ``jobs`` at a time, saving as it goes."""
    path = out / filename
    done: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
    seen = {json.dumps(r["cell"], sort_keys=True) for r in done if "error" not in r}
    work = [
        ({**cell, "seed": seed}, epochs)
        for cell in block
        for seed in range(seeds)
        if json.dumps({**cell, "seed": seed}, sort_keys=True) not in seen
    ]
    print(f"s4: {len(work)} to run ({len(seen)} already done), {jobs} at a time", flush=True)

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for record in pool.map(_safe, work):
            done.append(record)
            # Written after every cell: a driver that dies costs one run, not the study.
            path.write_text(json.dumps(done, indent=1))
            mark = "FAILED" if "error" in record else f"{record['seconds']:.0f}s"
            print(f"  [{len(done)}] {mark} {label(record['cell'])}", flush=True)
    return done


def _trace(client: Any, run_id: str) -> list[dict[str, float]]:
    """Reassemble the per-epoch trace from a run's streamed metrics (s5's helper).

    Zips the streamed ports into the same per-epoch rows the in-process path
    writes, so ``report`` and ``correlations`` need no second code path.
    """
    series: dict[str, list[float]] = {}
    for key in TRACE_KEYS:
        # Streamed metrics are named `<flow>.<port>`, one {step, value} row each.
        rows = client.metrics(run_id, f"train.{key}")
        if rows:
            series[key] = [row["value"] for row in sorted(rows, key=lambda row: row["step"])]
    length = min((len(v) for v in series.values()), default=0)
    return [{key: series[key][i] for key in series} for i in range(length)]


def datasets() -> dict[int, str]:
    """The registry: dataset seed -> generate run id, the pinned run alone if absent."""
    if not DATASETS.exists():
        return {0: GENERATE_RUN}
    return {int(seed): run for seed, run in json.loads(DATASETS.read_text()).items()}


def _key(record: dict[str, Any]) -> str:
    return json.dumps({**record["cell"], "dataset": record.get("dataset", 0)}, sort_keys=True)


def _dataset(client: Any, dataset: str) -> dict[str, Any]:
    """The dataset inputs of the ``train`` flow, as the generate run recorded them."""
    result = client.run(dataset).get("result") or {}
    names = ("dataset_train", "dataset_val", "dataset_test", "dataset_meta")
    missing = [name for name in names if name not in result]
    if missing:
        raise SystemExit(f"run {dataset} has no {missing}; is it a finished generate run?")
    return {name: result[name] for name in names}


def run_fluksio(
    block: list[dict[str, Any]],
    filename: str,
    *,
    seeds: int,
    jobs: int,
    epochs: int,
    out: Path,
) -> list[dict[str, Any]]:
    """Run a block through the engine: every cell one versioned run.

    The s5 pattern: artifacts resolved from each registry dataset's generate
    run, ``jobs`` submissions in flight, records appended to the same JSON shape
    the in-process path writes plus their dataset, with the trace rebuilt from
    the streamed metrics. Seed-major, so a stopped grid stays balanced.
    """
    import time as clock

    from fluksio.sdk.client import Client

    client = Client()
    data = {seed: _dataset(client, run) for seed, run in datasets().items()}

    path = out / filename
    done: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
    seen = {_key(r) for r in done if "error" not in r}
    queue = [
        (dataset, {**cell, "seed": seed})
        for seed in range(seeds)
        for dataset in sorted(data)
        for cell in block
        if _key({"dataset": dataset, "cell": {**cell, "seed": seed}}) not in seen
    ]
    total = len(seen) + len(queue)
    print(f"s4: {len(queue)} to submit ({len(seen)} already done), {jobs} in flight", flush=True)

    inflight: list[tuple[int, dict[str, Any], Any]] = []
    while queue or inflight:
        while queue and len(inflight) < jobs:
            dataset, settings = queue.pop(0)
            params = {**data[dataset], **settings, "epochs": epochs}
            handle = client.submit("train", params, seed=settings["seed"])
            inflight.append((dataset, settings, handle))
        clock.sleep(10.0)
        for dataset, settings, handle in list(inflight):
            if not handle.refresh().done:
                continue
            inflight.remove((dataset, settings, handle))
            payload = handle.result or {}
            done.append(
                {
                    "dataset": dataset,
                    "cell": settings,
                    "run": handle.id,
                    "final_metrics": payload.get("final_metrics"),
                    "test_metrics": {
                        name: (payload.get("test_metrics") or {}).get(name)
                        for name in ("known", "unknown")
                    },
                    "dla_report": payload.get("dla_report"),
                    "trace": _trace(client, handle.id),
                }
                if handle.status == "ok"
                else {
                    "dataset": dataset,
                    "cell": settings,
                    "run": handle.id,
                    "error": handle.status,
                }
            )
            path.write_text(json.dumps(done, indent=1))
            ok = sum("error" not in r for r in done)
            print(f"  [{ok}/{total}] {handle.status} d{dataset} {label(settings)}", flush=True)
    return done


def import_inprocess(filename: str, dataset: str, *, out: Path) -> None:
    """Import unversioned in-process results as finished engine runs.

    Imported records retain their source identity and parameter digest. Repeated
    imports use the same external ID.
    """
    from datetime import datetime

    from fluksio.sdk.client import Client

    client = Client()
    data = _dataset(client, dataset)
    path = out / filename
    done: list[dict[str, Any]] = json.loads(path.read_text())
    todo = [r for r in done if "error" not in r and "run" not in r]
    ts = datetime.fromisoformat(INPROCESS_DATE).timestamp()
    entries = [
        {
            "flow": "train",
            "external_id": f"s4/{label(r['cell'])}/seed={r['cell']['seed']}",
            "params": {
                **data,
                **{k: v for k, v in r["cell"].items() if k != "seed"},
                "epochs": r["final_metrics"]["epochs"],
            },
            "seed": r["cell"]["seed"],
            "created_at": INPROCESS_DATE,
            "result": {k: r[k] for k in ("final_metrics", "test_metrics", "dla_report", "seconds")},
            # The in-process trace kept TRACE_KEYS only, streamed as `fit` would.
            "metrics": [
                {"name": f"train.{key}", "step": row["epoch"], "ts": ts, "value": value}
                for row in r["trace"]
                for key, value in row.items()
            ],
            "labels": ["s4", "inprocess"],
        }
        for r in todo
    ]
    for record, answer in zip(todo, client.import_runs(entries), strict=True):
        record["run"], record["imported"] = answer["id"], True
    path.write_text(json.dumps(done, indent=1))
    print(f"s4: imported {len(todo)} in-process records from {filename}", flush=True)


def label(cell: dict[str, Any]) -> str:
    """A short, stable name for a cell: only what it varies from :data:`BASE`."""
    parts = [f"{k}={v}" for k, v in cell.items() if k != "seed" and BASE.get(k) != v]
    return " ".join(parts) or "base"


def correlations(trace: list[dict[str, float]]) -> dict[str, float | None]:
    """Compute raw and first-differenced purity-loss correlations per run.

    Return ``None`` for constant series.
    """
    import numpy as np
    from scipy import stats

    purity = np.asarray([r["g_purity"] for r in trace])
    loss = np.asarray([r["val_loss"] for r in trace])

    def corr(kind: Any, a: np.ndarray, b: np.ndarray) -> float | None:
        if a.std() == 0.0 or b.std() == 0.0:
            return None
        return float(kind(a, b).statistic)

    return {
        "pearson": corr(stats.pearsonr, purity, loss),
        "spearman": corr(stats.spearmanr, purity, loss),
        "pearson_diff": corr(stats.pearsonr, np.diff(purity), np.diff(loss)),
        "spearman_diff": corr(stats.spearmanr, np.diff(purity), np.diff(loss)),
    }


def _stat(values: list[float | None], places: int = 2) -> str:
    """``mean +- sd (k/n neg)`` over the runs that have a value, or ``--``."""
    import numpy as np

    clean = [v for v in values if v is not None]
    if not clean:
        return "--"
    neg = sum(v < 0 for v in clean)
    return f"{np.mean(clean):.{places}f} +- {np.std(clean):.{places}f} ({neg}/{len(clean)} neg)"


def report(out: Path) -> None:
    """One row per cell: task numbers, the purity trajectory, and the correlations."""
    import numpy as np

    from partiqledtr.analysis import uniform_prior_mean

    rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out.glob("*.json")):
        if path.name == "encoding.json":
            continue
        for record in json.loads(path.read_text()):
            if "error" not in record:
                rows.setdefault(label(record["cell"]), []).append(record)

    print(
        "\n| cell | n | train loss | acc known | sd over datasets | perfect | purity/mu_n "
        "| var pred | r(dP,dL) pearson |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, group in rows.items():
        final = [r["final_metrics"] for r in group]
        known = [r["test_metrics"]["known"] for r in group]
        series = [correlations(r["trace"]) for r in group]
        config = final[0]["config"]
        mu = uniform_prior_mean(config["ansatz"], config["n_qubits"])
        start = sum(f["g_purity_initial"] for f in final) / len(final) / mu
        end = sum(f["val_g_purity"] for f in final) / len(final) / mu
        # The variance-law scale of prediction 1: 2 P_g / dim g at the mean end purity.
        dim_g = group[0]["dla_report"]["dim_g"]
        var = 2.0 * end * mu / dim_g
        by_data: dict[int, list[float]] = {}
        for r, t in zip(group, known, strict=True):
            by_data.setdefault(r.get("dataset", 0), []).append(t["accuracy"])
        means = [sum(v) / len(v) for v in by_data.values()]
        spread = f"{np.std(means):.3f}" if len(means) > 1 else "--"
        print(
            f"| {name} | {len(group)} "
            f"| {sum(f['train_loss'] for f in final) / len(final):.4f} "
            f"| {sum(t['accuracy'] for t in known) / len(known):.3f} "
            f"| {spread} "
            f"| {sum(t['perfect'] for t in known) / len(known):.3f} "
            f"| {start:.2f} -> {end:.2f} "
            f"| {var:.1e} "
            f"| {_stat([s['pearson_diff'] for s in series])} |"
        )


def main() -> None:
    """Parse arguments and run the requested block, the chart gate, or the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5, help="the smoke's 3, topped up (user)")
    # Two cells at a time: what this VM's 15 GB holds with the engine's warm workers.
    parser.add_argument("--jobs", type=int, default=2)
    # 40, matching s2/s3 so the numbers are comparable run for run.
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--encodings", action="store_true", help="price the charts, no training")
    parser.add_argument("--gate", action="store_true", help="one worst-case cell, cost projection")
    parser.add_argument(
        "--fluksio",
        action="store_true",
        help="submit through the engine as versioned runs, on every registry dataset",
    )
    parser.add_argument(
        "--import-inprocess",
        action="store_true",
        help="record the in-process seeds in the engine, against the pinned run",
    )
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if not DATA.exists():
        raise SystemExit(f"{DATA} missing; export the pinned generate run first (s2 README)")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(args.out)
        return
    if args.import_inprocess:
        import_inprocess("smoke.json", GENERATE_RUN, out=args.out)
        return
    if args.encodings:
        table = price_encodings()
        (args.out / "encoding.json").write_text(json.dumps(table, indent=1))
        print("| chart | arm | purity | /mu_n | below mu_n/2 |")
        print("| --- | --- | --- | --- | --- |")
        for chart, arms in table["charts"].items():
            for arm, row in arms.items():
                print(
                    f"| {chart} | {arm} | {row['mean_purity']:.3f} "
                    f"| {row['ratio']:.2f} | {row['below_threshold']:.0%} |"
                )
        return
    if args.gate:
        done = run(gate_cell(), "gate.json", seeds=1, jobs=1, epochs=args.epochs, out=args.out)
        good = [r["seconds"] for r in done if "seconds" in r]
        if good:
            total = good[-1] * len(cells()) * args.seeds / args.jobs / 3600
            print(f"gate: {good[-1]:.0f}s/run -> ~{total:.1f}h for the smoke grid")
        return
    common = {"seeds": args.seeds, "jobs": args.jobs, "epochs": args.epochs, "out": args.out}
    if args.fluksio:
        run_fluksio(cells(), "smoke.json", **common)
    else:
        run(cells(), "smoke.json", **common)
    report(args.out)


if __name__ == "__main__":
    main()
