"""ROADMAP phase 4c item 1: does the in-algebra readout open the purity-loss channel?

Phase 4b measured a preconditioner that moves the encoded distribution over a
190-fold purity range while the task loss barely responds -- under a readout whose
observable purity is zero on every floor-free arm (``FINDINGS.md`` §2). D107 put
the readout in the algebra; this study measures the thing that fix exists to
enable: whether the g-purity trajectory and the loss become correlated *within* a
run, ROADMAP 4c's success criterion.

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
readout is the only thing that moved between the studies (D104, D107).

    python dev/s3-readout-channel/run.py            # run the smoke block
    python dev/s3-readout-channel/run.py --report   # tables + correlations
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

BASE: dict[str, Any] = {
    "model": "qfm",
    "encoding": "legacy",
    "angle_map": "legacy",
    "ansatz": "XY_Ring",
}

#: Trace keys kept per epoch. The correlation analysis needs the purity and the
#: angle-shape series beside the losses, which is what s2's three-key trace lacks.
TRACE_KEYS = ("epoch", "train_loss", "val_loss", "g_purity", "tv_uniform", "mean_sin2")


def cells() -> list[dict[str, Any]]:
    """The smoke block: the learned preconditioner against its frozen control."""
    return [{**BASE, "preconditioner": f} for f in ("none", "mlp")]


def load(name: str) -> dict[str, Any]:
    """Read one exported dataset split."""
    import numpy as np

    with np.load(DATA / f"{name}.npz") as data:
        return {key: data[key] for key in data.files}


def one_cell(settings: dict[str, Any], epochs: int) -> dict[str, Any]:
    """Fit one cell exactly as the ``train`` flow's nodes would (D104)."""
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


def run(*, seeds: int, jobs: int, epochs: int, out: Path) -> list[dict[str, Any]]:
    """Run every cell x seed, ``jobs`` at a time, saving as it goes."""
    path = out / "smoke.json"
    done: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
    seen = {json.dumps(r["cell"], sort_keys=True) for r in done if "error" not in r}
    work = [
        ({**cell, "seed": seed}, epochs)
        for cell in cells()
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
    from partiqledtr.analysis import uniform_prior_mean

    path = out / "smoke.json"
    rows: dict[str, list[dict[str, Any]]] = {}
    for record in json.loads(path.read_text()):
        if "error" not in record:
            rows.setdefault(label(record["cell"]), []).append(record)

    print(
        "\n| cell | n | train loss | acc known | purity/mu_n | r(P,L) spearman "
        "| r(dP,dL) pearson | r(dP,dL) spearman |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, group in rows.items():
        final = [r["final_metrics"] for r in group]
        known = [r["test_metrics"]["known"] for r in group]
        series = [correlations(r["trace"]) for r in group]
        mu = uniform_prior_mean(final[0]["config"].get("ansatz", "XY_Brickwork"), 4)
        start = sum(f["g_purity_initial"] for f in final) / len(final) / mu
        end = sum(f["val_g_purity"] for f in final) / len(final) / mu
        print(
            f"| {name} | {len(group)} "
            f"| {sum(f['train_loss'] for f in final) / len(final):.4f} "
            f"| {sum(t['accuracy'] for t in known) / len(known):.3f} "
            f"| {start:.2f} -> {end:.2f} "
            f"| {_stat([s['spearman'] for s in series])} "
            f"| {_stat([s['pearson_diff'] for s in series])} "
            f"| {_stat([s['spearman_diff'] for s in series])} |"
        )


def main() -> None:
    """Parse arguments and either run the smoke block or report what has been run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=5)
    # 40, matching phase 4b so the numbers are comparable run for run.
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if not DATA.exists():
        raise SystemExit(f"{DATA} missing; export the pinned generate run first (s2 README)")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(args.out)
        return
    run(seeds=args.seeds, jobs=args.jobs, epochs=args.epochs, out=args.out)
    report(args.out)


if __name__ == "__main__":
    main()
