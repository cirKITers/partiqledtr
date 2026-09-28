"""Trig-interface probe: does classical trigonometric capacity fit the QFM arm?

The quantum arm's classical parts are purely linear, so the whole model is
trig-polynomial -> linear -> trig-polynomial -> linear -- and block 2 consumes
the node update's output directly as RY angles. The init-time diagnostic
(2026-09-02) showed that boundary is *not* collapsed at start (std 0.89 rad,
0.92 mu_n, against the collapse hypothesis), sitting at the uniform level while
the gated block-1 chart sits at 1.89 mu_n. What is open is what training does to
it, and whether classical nonlinear capacity in the node update pays -- and if
so, whether the *trigonometric* kind (SIREN, cf. QIREN) pays
beyond a matched-parameter ELU control.

Four cells on the s4 baseline configuration (`XY_Cycle`, floor-free, cheapest;
`pair_polar_boost`; K=4; `lr_qfm` 1e-2; no preconditioner, so the node axis is
the only thing varying):

    linear-w1   the s4 architecture, now with block-2 instrumentation (control)
    linear-w8   the boundary-scale axis alone (`node_omega = 8`)
    siren       sine node MLP, SIREN init (the trig-interface arm)
    elu         the same MLP with ELU -- matched parameters, different activation

Reads: task scores against the s4 smoke baseline (0.555 +- 0.009), and the
trained block-2 purity endpoints the instrumentation now records.

    python dev/s5-trig-nodes/run.py             # the probe (4 cells x 2 seeds)
    python dev/s5-trig-nodes/run.py --report    # tables + correlations
    python dev/s5-trig-nodes/spectrum.py        # the task-spectrum probe (no training)
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
#: The s2 export of the pinned generate run, reused unchanged (as s3/s4).
DATA = STUDY.parent / "s2-expressivity" / "data"
OUT = STUDY / "results"

#: The pinned generate run `DATA` was exported from. `--fluksio` submits cells
#: as versioned runs of the `train` flow against its artifacts; the
#: in-process path below stays the sandbox.
GENERATE_RUN = "1787760161002-8bde9189"

BASE: dict[str, Any] = {
    "model": "qfm",
    "n_qubits": 6,
    "n_channels": 4,
    "lr_qfm": 1e-2,
    "encoding": "cartesian",
    "angle_map": "pair_polar_boost",
    "ansatz": "XY_Cycle",
    "preconditioner": "none",
}

TRACE_KEYS = ("epoch", "train_loss", "val_loss", "g_purity", "tv_uniform", "mean_sin2")


def cells() -> list[dict[str, Any]]:
    """The four node-update cells of the probe."""
    return [
        {**BASE},
        {**BASE, "node_omega": 8.0},
        {**BASE, "node_update": "siren"},
        {**BASE, "node_update": "elu"},
    ]


def cells_expand() -> list[dict[str, Any]]:
    """The siren expansion: arm generality and the preconditioner interaction.

    siren across the graph trichotomy x {none, mlp}; the cycle x none cell
    already exists in the probe, so only its mlp completion runs here. The
    linear baselines are s4's smoke grid, and the capacity control (elu) was
    priced on the cycle in the probe. Reading rule in the README, registered
    before the numbers.
    """
    siren = {**BASE, "node_update": "siren"}
    return [
        {**siren, "ansatz": arm, "preconditioner": pre}
        for arm in ("XY_Ladder", "XY_OddChord")
        for pre in ("none", "mlp")
    ] + [{**siren, "preconditioner": "mlp"}]


def cells_confirm() -> list[dict[str, Any]]:
    """Confirmation of the probe and expansion headline cells, plus the missing elu controls.

    Submit with ``--seeds 5``: dedup is study-wide, so the probe/expand seeds
    are not re-run -- the siren headline cells gain three seeds each and the
    elu control reaches five seeds on every arm, which is what closes the
    capacity caveat behind the floor-free-specific reading.
    The secondary mlp-interaction cells stay at their two seeds.
    """
    siren = {**BASE, "node_update": "siren"}
    elu = {**BASE, "node_update": "elu"}
    return [
        {**siren},
        {**siren, "ansatz": "XY_Ladder"},
        {**siren, "ansatz": "XY_OddChord"},
        {**siren, "ansatz": "XY_OddChord", "preconditioner": "mlp"},
        {**elu},
        {**elu, "ansatz": "XY_Ladder"},
        {**elu, "ansatz": "XY_OddChord"},
    ]


def _seen(out: Path) -> set[str]:
    """Every finished cell across the study's result files.

    Study-wide rather than per-file (the s2 lesson): the confirmation block
    shares cells with the probe and the expansion, and a per-file check would
    re-run their seeds for identical numbers.
    """
    seen: set[str] = set()
    for path in out.glob("*.json"):
        if path.name == "spectrum.json":
            continue
        for record in json.loads(path.read_text()):
            if "error" not in record:
                seen.add(json.dumps(record["cell"], sort_keys=True))
    return seen


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
    seen = _seen(out)
    work = [
        ({**cell, "seed": seed}, epochs)
        for cell in block
        for seed in range(seeds)
        if json.dumps({**cell, "seed": seed}, sort_keys=True) not in seen
    ]
    print(f"s5: {len(work)} to run ({len(seen)} already done), {jobs} at a time", flush=True)

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for record in pool.map(_safe, work):
            done.append(record)
            path.write_text(json.dumps(done, indent=1))
            mark = "FAILED" if "error" in record else f"{record['seconds']:.0f}s"
            print(f"  [{len(done)}] {mark} {label(record['cell'])}", flush=True)
    return done


def _trace(client: Any, run_id: str) -> list[dict[str, float]]:
    """Reassemble the per-epoch trace from a run's streamed metrics.

    The engine records every stream port of ``fit``; this fetches the ones the
    correlation analysis reads and zips them into the same per-epoch rows the
    in-process path writes, so ``report`` and ``correlations`` need no second
    code path.
    """
    series: dict[str, list[float]] = {}
    for key in TRACE_KEYS:
        # Streamed metrics are named `<flow>.<port>`, one {step, value} row each.
        rows = client.metrics(run_id, f"train.{key}")
        if rows:
            series[key] = [row["value"] for row in sorted(rows, key=lambda row: row["step"])]
    length = min((len(v) for v in series.values()), default=0)
    return [{key: series[key][i] for key in series} for i in range(length)]


def run_fluksio(
    block: list[dict[str, Any]],
    filename: str,
    dataset: str,
    *,
    seeds: int,
    jobs: int,
    epochs: int,
    out: Path,
) -> list[dict[str, Any]]:
    """Run a block through the engine: every cell one versioned run.

    The s2 ``run_arm_fluksio`` pattern: artifacts resolved from the pinned
    generate run, a bounded number of submissions in flight, results appended to
    the same JSON the in-process path writes -- plus the streamed trace, so the
    correlation analysis carries over.
    """
    import time as clock

    from fluksio.sdk.client import Client

    client = Client()
    result = client.run(dataset).get("result") or {}
    names = ("dataset_train", "dataset_val", "dataset_test", "dataset_meta")
    missing = [name for name in names if name not in result]
    if missing:
        raise SystemExit(f"run {dataset} has no {missing}; is it a finished generate run?")
    data = {name: result[name] for name in names}

    path = out / filename
    done: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
    seen = _seen(out)
    queue = [
        {**cell, "seed": seed}
        for cell in block
        for seed in range(seeds)
        if json.dumps({**cell, "seed": seed}, sort_keys=True) not in seen
    ]
    total = len(done) + len(queue)
    print(f"s5: {len(queue)} to submit ({len(seen)} already done), {jobs} in flight", flush=True)

    inflight: list[tuple[dict[str, Any], Any]] = []
    while queue or inflight:
        while queue and len(inflight) < jobs:
            settings = queue.pop(0)
            params = {**data, **settings, "epochs": epochs}
            handle = client.submit("train", params, seed=settings["seed"])
            inflight.append((settings, handle))
        clock.sleep(10.0)
        for settings, handle in list(inflight):
            if not handle.refresh().done:
                continue
            inflight.remove((settings, handle))
            payload = handle.result or {}
            done.append(
                {
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
                else {"cell": settings, "error": handle.status}
            )
            path.write_text(json.dumps(done, indent=1))
            print(f"  [{len(done)}/{total}] {handle.status} {label(settings)}", flush=True)
    return done


def label(cell: dict[str, Any]) -> str:
    """A short, stable name for a cell: only what it varies from :data:`BASE`."""
    parts = [f"{k}={v}" for k, v in cell.items() if k != "seed" and BASE.get(k) != v]
    return " ".join(parts) or "linear-w1"


def report(out: Path) -> None:
    """Task numbers beside the block-2 boundary trajectory, one row per cell."""
    import numpy as np

    from partiqledtr.analysis import uniform_prior_mean

    rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out.glob("*.json")):
        if path.name == "spectrum.json":
            continue
        for record in json.loads(path.read_text()):
            if "error" not in record:
                rows.setdefault(label(record["cell"]), []).append(record)

    print("\n| cell | n | params | acc known | perfect | P1/mu | P2/mu start -> end |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, group in rows.items():
        final = [r["final_metrics"] for r in group]
        known = [r["test_metrics"]["known"] for r in group]
        config = final[0]["config"]
        mu = uniform_prior_mean(config["ansatz"], config["n_qubits"])
        acc = [t["accuracy"] for t in known]
        b2 = [
            (f.get("block2_g_purity_initial"), f.get("block2_g_purity"))
            for f in final
            if f.get("block2_g_purity") is not None
        ]
        boundary = (
            f"{np.mean([a for a, _ in b2]) / mu:.2f} -> {np.mean([b for _, b in b2]) / mu:.2f}"
            if b2
            else "--"
        )
        print(
            f"| {name} | {len(group)} | {final[0]['n_params']} "
            f"| {np.mean(acc):.3f} +- {np.std(acc):.3f} "
            f"| {np.mean([t['perfect'] for t in known]):.3f} "
            f"| {np.mean([f['val_g_purity'] for f in final]) / mu:.2f} "
            f"| {boundary} |"
        )


def main() -> None:
    """Run the probe block or report what has been run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--expand", action="store_true", help="the siren arm/preconditioner grid")
    parser.add_argument("--confirm", action="store_true", help="headline seeds + elu controls")
    parser.add_argument(
        "--fluksio",
        nargs="?",
        const=GENERATE_RUN,
        metavar="GENERATE_RUN",
        help="submit through the engine as versioned runs (default: the pinned run)",
    )
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if not DATA.exists():
        raise SystemExit(f"{DATA} missing; export the pinned generate run first (s2 README)")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(args.out)
        return
    if args.confirm:
        block, filename = cells_confirm(), "confirm.json"
    elif args.expand:
        block, filename = cells_expand(), "expand.json"
    else:
        block, filename = cells(), "probe.json"
    common = {"seeds": args.seeds, "jobs": args.jobs, "epochs": args.epochs, "out": args.out}
    if args.fluksio:
        run_fluksio(block, filename, args.fluksio, **common)
    else:
        run(block, filename, **common)
    report(args.out)


if __name__ == "__main__":
    main()
