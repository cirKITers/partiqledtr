"""Does the in-algebra readout open the purity-loss channel?

The s2 expressivity study measured a preconditioner that moves the encoded
distribution over a 190-fold purity range while the task loss barely responds --
under a per-qubit Z readout whose observable purity is zero on every floor-free
arm. The readout is now in the algebra (``<XX_b> + <YY_b>`` per coupling bond);
this study measures the thing that fix exists to enable: whether the g-purity
trajectory and the loss become correlated *within* a run.

Smoke block first, 10 runs: ``XY_Ring`` x {none, mlp} on the clustered ``legacy``
encoding at 5 seeds. The clustered arm is where the prediction is falsifiable --
g-purity starts far below the uniform-prior mean, so a learned preconditioner has
headroom to move it, and the new readout gives that movement a channel to the
loss. The ``none`` cells are the within-study control: their encoded distribution
is frozen, so their purity series is constant and carries no correlation to
explain away. Expansion (``XY_Brickwork``, ``pair_polar``, the floored
``XY_AllPairs`` specificity control) waits on this block showing the correlation
at all.

Runs in process against the splits ``dev/s2-expressivity/data`` exported from
generate run ``1787760161002-8bde9189`` -- the same data, deliberately, so the
readout is the only thing that moved between the studies.

``--fluksio`` repeats the full grid, plus the classical GNN baseline, on every
dataset of the shared registry ``dev/datasets.json`` as versioned runs: dataset 0
is the pinned run above, ``--generate`` adds generate seeds 1-4 (README, dataset
repeat). Its records land in ``results/datasets/``, apart from the in-process
history.

    python dev/s3-readout-channel/run.py            # run the smoke block
    python dev/s3-readout-channel/run.py --report   # tables + correlations
    python dev/s3-readout-channel/run.py --generate           # datasets 1-4 into the registry
    python dev/s3-readout-channel/run.py --fluksio            # full grid x datasets, versioned
    python dev/s3-readout-channel/run.py --fluksio --report   # the same tables, per dataset
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

#: The pinned generate run ``DATA`` was exported from: dataset 0 of the registry.
GENERATE_RUN = "1787760161002-8bde9189"
#: Dataset seed -> generate run id, shared with s4.
DATASETS = STUDY.parent / "datasets.json"
DATASET_SEEDS = (0, 1, 2, 3, 4)

BASE: dict[str, Any] = {
    "model": "qfm",
    "encoding": "legacy",
    "angle_map": "legacy",
    "ansatz": "XY_Ring",
}

#: Trace keys kept per epoch. The correlation analysis needs the purity and the
#: angle-shape series beside the losses, which is what s2's three-key trace lacks.
TRACE_KEYS = ("epoch", "train_loss", "val_loss", "g_purity", "tv_uniform", "mean_sin2")


def cells(n_channels: int = 1) -> list[dict[str, Any]]:
    """The smoke block: the learned preconditioner against its frozen control.

    ``n_channels > 1`` is the widening dose (that many parallel QFMs per block);
    the key is added only then, so the widened cells dedup against their own
    records and not the narrow ones.
    """
    wide = {"n_channels": n_channels} if n_channels > 1 else {}
    return [{**BASE, **wide, "preconditioner": f} for f in ("none", "mlp")]


def cells_opt() -> list[dict[str, Any]]:
    """The optimizer smoke: per-group learning rates at the widened dose.

    Against the K=4 shared-rate baselines already in ``smoke.json``. The MLP
    rate moves down and up around the shared 1e-3 (is the accuracy cost of the
    preconditioner an optimisation artifact?); the circuit rate moves up with
    and without the preconditioner (an earlier learning-rate sweep saw the rescue
    at 1e-2, and the ``none`` cell says whether a faster circuit helps regardless).
    """
    wide = {**BASE, "n_channels": 4}
    return [
        {**wide, "preconditioner": "mlp", "lr_preconditioner": 1e-4},
        {**wide, "preconditioner": "mlp", "lr_preconditioner": 1e-2},
        {**wide, "preconditioner": "mlp", "lr_qfm": 1e-2},
        {**wide, "preconditioner": "none", "lr_qfm": 1e-2},
    ]


def cells_full() -> list[dict[str, Any]]:
    """The full grid, at the widened dose and the optimizer smoke's pick.

    ``lr_qfm = 1e-2`` throughout (the circuit must track the latent distribution
    the preconditioner moves), everything else at the shared 1e-3.
    Arms: both floor-free ansaetze x {none, mlp} x {clustered legacy, pair_polar},
    plus the floored ``XY_AllPairs`` specificity control, where the purity-loss
    correlation must be absent.
    """
    pick = {"model": "qfm", "n_channels": 4, "lr_qfm": 1e-2}
    grid = [
        {
            **pick,
            "ansatz": ansatz,
            "preconditioner": pre,
            "encoding": enc,
            "angle_map": "legacy" if enc == "legacy" else "pair_polar",
        }
        for ansatz in ("XY_Ring", "XY_Brickwork")
        for pre in ("none", "mlp")
        for enc in ("legacy", "cartesian")
    ]
    control = {
        **pick,
        "ansatz": "XY_AllPairs",
        "preconditioner": "mlp",
        "encoding": "legacy",
        "angle_map": "legacy",
    }
    return [*grid, control]


def cells_datasets() -> list[dict[str, Any]]:
    """The full grid plus the phase-4b classical baseline, for the dataset repeat."""
    return [*cells_full(), {"model": "gnn", "encoding": "cartesian", "dim": 64}]


def datasets() -> dict[int, str]:
    """The registry: dataset seed -> generate run id, the pinned run alone if absent."""
    if not DATASETS.exists():
        return {0: GENERATE_RUN}
    return {int(seed): run for seed, run in json.loads(DATASETS.read_text()).items()}


def generate(client: Any) -> None:
    """Generate the missing registry datasets at the flow defaults (the pinned design).

    The pinned run stands in for seed 0 rather than a fresh generate, which need
    not reproduce it with today's code (README, dataset repeat). All pending
    seeds are submitted at once; a run enters the registry only once its split
    integrity check passes.
    """
    runs = datasets()
    pending = {
        seed: client.submit("generate", {"seed": seed}, seed=seed)
        for seed in DATASET_SEEDS
        if seed not in runs
    }
    for seed, handle in pending.items():
        handle.wait(poll=30.0)
        ok = (handle.result.get("stats") or {}).get("split_integrity_ok")
        print(f"  dataset {seed}: {handle.id} {handle.status}, split_integrity_ok={ok}", flush=True)
        if handle.status == "ok" and ok:
            runs[seed] = handle.id
            DATASETS.write_text(json.dumps({str(k): v for k, v in sorted(runs.items())}, indent=1))


def _dataset(client: Any, run_id: str) -> dict[str, Any]:
    """The dataset inputs of the ``train`` flow, as a generate run recorded them."""
    result = client.run(run_id).get("result") or {}
    names = ("dataset_train", "dataset_val", "dataset_test", "dataset_meta")
    missing = [name for name in names if name not in result]
    if missing:
        raise SystemExit(f"run {run_id} has no {missing}; is it a finished generate run?")
    return {name: result[name] for name in names}


def _trace(client: Any, run_id: str) -> list[dict[str, float]]:
    """Reassemble the per-epoch trace from a run's streamed metrics (s4's helper)."""
    series: dict[str, list[float]] = {}
    for key in TRACE_KEYS:
        rows = client.metrics(run_id, f"train.{key}")
        if rows:
            series[key] = [row["value"] for row in sorted(rows, key=lambda row: row["step"])]
    length = min((len(v) for v in series.values()), default=0)
    return [{key: series[key][i] for key in series} for i in range(length)]


def _key(record: dict[str, Any]) -> str:
    return json.dumps({**record["cell"], "dataset": record.get("dataset", 0)}, sort_keys=True)


def run_fluksio(
    block: list[dict[str, Any]], filename: str, *, seeds: int, jobs: int, epochs: int, out: Path
) -> list[dict[str, Any]]:
    """Every cell x seed on every registry dataset, as versioned ``train`` runs.

    Seed-major order, so a grid stopped part-way leaves every dataset at the same
    number of seeds. Failed runs are not counted as done and are resubmitted.
    """
    import time as clock

    from fluksio.sdk.client import Client

    client = Client()
    data = {seed: _dataset(client, run) for seed, run in datasets().items()}
    path = out / filename
    done: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
    seen = {_key(r) for r in done if "error" not in r}
    queue = [
        {"dataset": dataset, "cell": {**cell, "seed": seed}}
        for seed in range(seeds)
        for dataset in sorted(data)
        for cell in block
    ]
    queue = [entry for entry in queue if _key(entry) not in seen]
    total = len(seen) + len(queue)
    print(f"datasets {sorted(data)}: {len(queue)} to submit ({len(seen)} done), {jobs} in flight")

    inflight: list[tuple[dict[str, Any], Any]] = []
    while queue or inflight:
        while queue and len(inflight) < jobs:
            entry = queue.pop(0)
            params = {**data[entry["dataset"]], **entry["cell"], "epochs": epochs}
            inflight.append((entry, client.submit("train", params, seed=entry["cell"]["seed"])))
        clock.sleep(10.0)
        for entry, handle in list(inflight):
            if not handle.refresh().done:
                continue
            inflight.remove((entry, handle))
            record = {**entry, "run": handle.id}
            if handle.status == "ok":
                payload = handle.result
                record |= {
                    "final_metrics": payload.get("final_metrics"),
                    "test_metrics": {
                        name: (payload.get("test_metrics") or {}).get(name)
                        for name in ("known", "unknown")
                    },
                    "trace": _trace(client, handle.id),
                }
            else:
                record["error"] = handle.status
            done.append(record)
            path.write_text(json.dumps(done, indent=1))
            ok = sum("error" not in r for r in done)
            name = f"d{entry['dataset']} seed={entry['cell']['seed']} {label(entry['cell'])}"
            print(f"  [{ok}/{total}] {handle.status} {name}", flush=True)
    return done


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
    # by wiring `dla_report` upstream of `fit`, kept here by doing it first.
    certificate = dla_check(settings.get("ansatz", "XY_Brickwork"), n_qubits=4)

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
    print(f"smoke: {len(work)} to run ({len(seen)} already done), {jobs} at a time", flush=True)

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for record in pool.map(_safe, work):
            done.append(record)
            # Written after every cell: a driver that dies costs one run, not the study.
            path.write_text(json.dumps(done, indent=1))
            mark = "FAILED" if "error" in record else f"{record['seconds']:.0f}s"
            print(f"  [{len(done)}] {mark} {label(record['cell'])}", flush=True)
    return done


def label(cell: dict[str, Any]) -> str:
    """A short, stable name for a cell: only what it varies from :data:`BASE`."""
    parts = [f"{k}={v}" for k, v in cell.items() if k != "seed" and BASE.get(k) != v]
    return " ".join(parts) or "base"


def correlations(trace: list[dict[str, float]]) -> dict[str, float | None]:
    """Within-run association of the g-purity and validation-loss series.

    Both raw and first-differenced: two monotone series correlate trivially, so
    the de-trended number is the honest one and the headline. ``None`` where a
    series is constant (the ``none`` control) -- there is nothing to correlate.
    The success criterion expects *negative* values on the mlp cells: purity up,
    loss down.
    """
    import numpy as np
    from scipy import stats

    if not trace or "g_purity" not in trace[0]:  # classical baseline: no encoded state
        return dict.fromkeys(("pearson", "spearman", "pearson_diff", "spearman_diff"))
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
    """One row per cell: task numbers, the purity trajectory, and the correlations.

    Records from ``--fluksio`` carry their dataset: the accuracy then also gets its
    spread over the per-dataset means, and a second table pairs every mlp run with
    its none twin (same arm, input law, dataset and seed).
    """
    import numpy as np

    from partiqledtr.analysis import uniform_prior_mean

    rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out.glob("*.json")):
        for record in json.loads(path.read_text()):
            if "error" not in record:
                rows.setdefault(label(record["cell"]), []).append(record)

    print(
        "\n| cell | n | train loss | acc known | sd over datasets | purity/mu_n "
        "| r(P,L) spearman | r(dP,dL) pearson | r(dP,dL) spearman |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    acc: dict[tuple, float] = {}
    for name, group in rows.items():
        final = [r["final_metrics"] for r in group]
        known = [r["test_metrics"]["known"] for r in group]
        series = [correlations(r["trace"]) for r in group]
        by_data: dict[int, list[float]] = {}
        for r, t in zip(group, known, strict=True):
            by_data.setdefault(r.get("dataset", 0), []).append(t["accuracy"])
            acc[(name, r.get("dataset", 0), r["cell"]["seed"])] = t["accuracy"]
        spread = np.std([np.mean(v) for v in by_data.values()]) if len(by_data) > 1 else None
        purity = "--"
        if "g_purity_initial" in final[0]:
            mu = uniform_prior_mean(final[0]["config"].get("ansatz", "XY_Brickwork"), 4)
            start = sum(f["g_purity_initial"] for f in final) / len(final) / mu
            end = sum(f["val_g_purity"] for f in final) / len(final) / mu
            purity = f"{start:.2f} -> {end:.2f}"
        print(
            f"| {name} | {len(group)} "
            f"| {sum(f['train_loss'] for f in final) / len(final):.4f} "
            f"| {sum(t['accuracy'] for t in known) / len(known):.3f} "
            f"| {'--' if spread is None else f'{spread:.3f}'} "
            f"| {purity} "
            f"| {_stat([s['spearman'] for s in series])} "
            f"| {_stat([s['pearson_diff'] for s in series])} "
            f"| {_stat([s['spearman_diff'] for s in series])} |"
        )

    deltas: dict[str, dict[int, list[float]]] = {}
    for (name, dataset, seed), value in acc.items():
        twin = (name.replace("preconditioner=mlp", "preconditioner=none"), dataset, seed)
        if twin[0] != name and twin in acc:
            deltas.setdefault(name, {}).setdefault(dataset, []).append(value - acc[twin])
    if deltas:
        print("\n| mlp cell | mlp - none, paired | per dataset | positive |")
        print("| --- | --- | --- | --- |")
    for name, by_data in deltas.items():
        values = [v for vs in by_data.values() for v in vs]
        per = " ".join(f"{np.mean(by_data[d]):+.3f}" for d in sorted(by_data))
        print(
            f"| {name} | {np.mean(values):+.3f} | {per} "
            f"| {sum(v > 0 for v in values)}/{len(values)} |"
        )


def main() -> None:
    """Parse arguments and either run the smoke block or report what has been run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=5)
    # 40, matching s2 so the numbers are comparable run for run.
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--channels", type=int, default=1, help="widening dose: QFMs per block")
    parser.add_argument("--opt", action="store_true", help="run the per-group learning-rate block")
    parser.add_argument("--full", action="store_true", help="run the full grid (lr_qfm=1e-2)")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--generate", action="store_true", help="generate registry datasets 1-4")
    parser.add_argument(
        "--fluksio",
        action="store_true",
        help="full grid + GNN on every registry dataset, versioned, into results/datasets/",
    )
    args = parser.parse_args()

    if args.generate:
        from fluksio.sdk.client import Client

        generate(Client())
        return
    if args.fluksio:
        args.out = args.out / "datasets"
    elif not DATA.exists():
        raise SystemExit(f"{DATA} missing; export the pinned generate run first (s2 README)")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(args.out)
        return
    common = {"seeds": args.seeds, "jobs": args.jobs, "epochs": args.epochs, "out": args.out}
    if args.fluksio:
        run_fluksio(cells_datasets(), "full.json", **common)
        report(args.out)
        return
    if args.full:
        block, filename = cells_full(), "full.json"
    elif args.opt:
        block, filename = cells_opt(), "opt.json"
    else:
        block, filename = cells(args.channels), "smoke.json"
    run(block, filename, **common)
    report(args.out)


if __name__ == "__main__":
    main()
