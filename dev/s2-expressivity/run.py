"""The three expressivity arms, run in process.

The nodes are plain functions, so an arm needs no engine: this calls
:func:`partiqledtr.train.train_model` and :func:`partiqledtr.train.evaluate_split`
directly on dataset splits read from disk. The Fluksio flows in
:mod:`partiqledtr.pipeline` are unchanged and still describe the same work, so the
study is re-runnable through the engine whenever that is wanted -- this driver is
the sandbox path, not a fork of it.

Arms, deliberately independent -- each holds everything else at the baseline
configuration (:data:`BASE`) so its axis is the only thing that moves:

* **A, depth.** ``n_layers`` 2, 4, 8, 16 on the baseline. The per-feature
  spectrum is ``2L + 1``, so this is the cheapest test of "is expressivity the
  bottleneck" and it changes nothing else. Run last: deep circuits are unrolled,
  so their *compile* cost dominates even though the step cost is flat in depth.
* **B, encoding weights.** ``enc_weights`` x ``enc_reupload``, crossed with the raw
  and learned preconditioners, plus the clustered ``legacy`` encoding where the
  manuscript's jitter amplification has room to act.
* **C, ansatz.** The partition-respecting arms against the baseline's
  ``XY_Brickwork``, crossed with the preconditioner.

Every cell runs at ``--seeds`` seeds, because the differences between cells
are small enough that a single seed says nothing.

    python dev/s2-expressivity/run.py --arm c                 # in process
    python dev/s2-expressivity/run.py --arm c --fluksio <run>  # through the engine
    python dev/s2-expressivity/run.py --report
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

#: This study's folder, resolved from this module rather than the working directory,
#: so a driver behaves the same wherever it is started from.
STUDY = Path(__file__).resolve().parent
DATA = STUDY / "data"
OUT = STUDY / "results"

BASE: dict[str, Any] = {"model": "qfm", "encoding": "cartesian", "ansatz": "XY_Brickwork"}
ANSATZ_ARMS = ("XY_Brickwork", "XY_Ring", "XY_AllPairs", "Circuit_19")

#: Arm B's cells, chosen so each isolates a different thing rather than filling the
#: grid: the baseline, mixing without enrichment, enrichment of the angle
#: without enrichment of the comb, the dose-response in spectrum size, and the cell
#: that is dissociated *and* endpoint-equivariant.
WEIGHT_CELLS = (
    ("hamming", "diagonal"),
    ("hamming", "cyclic"),
    ("ternary", "diagonal"),
    ("binary", "cyclic"),
    ("ternary", "cyclic"),
    ("ternary_pair", "cyclic"),
)


def cells(arm: str) -> list[dict[str, Any]]:
    """Return the settings of every cell of one arm."""
    if arm == "a":
        return [{**BASE, "n_layers": n} for n in (2, 4, 8, 16)]
    if arm == "b":
        grid = [
            {**BASE, "enc_weights": w, "enc_reupload": r, "preconditioner": f}
            for (w, r), f in itertools.product(WEIGHT_CELLS, ("none", "mlp"))
        ]
        # The clustered arm: p*E*pi angles collapse toward zero, which is where an
        # exponential spectrum can amplify the jitter and a Hamming one cannot.
        return grid + [
            {
                **BASE,
                "encoding": "legacy",
                "angle_map": "legacy",
                "enc_weights": w,
                "enc_reupload": "cyclic",
            }
            for w in ("hamming", "ternary")
        ]
    if arm == "c":
        return [
            {**BASE, "ansatz": a, "preconditioner": f}
            for a, f in itertools.product(ANSATZ_ARMS, ("none", "mlp"))
        ]
    if arm == "baseline":
        return [
            {"model": "gnn", "encoding": "cartesian", "dim": 64},
            {"model": "mlp", "encoding": "cartesian"},
        ]
    raise ValueError(f"unknown arm {arm!r}; pick a, b, c or baseline")


def _already_run(out: Path) -> set[str]:
    """Every cell that has finished, in any arm's file.

    Arms share cells -- the baseline belongs to all three -- so a per-arm
    check submits the same configuration once per arm and pays for it twice, which
    the engine export made visible as fourteen runs where seven would do.
    """
    seen: set[str] = set()
    for path in sorted(out.glob("arm_*.json")):
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
    """Fit one cell and score it, exactly as the ``train`` flow's nodes would.

    Runs in its own process: JAX keeps compiled executables per process, and the
    deep cells of arm A would otherwise retain the whole scan's compilation cache.
    """
    import numpy as np

    from partiqledtr.analysis import dla_check
    from partiqledtr.train import evaluate_split, train_model

    started = time.monotonic()
    seed = settings.pop("seed")
    encoding = settings.get("encoding", "cartesian")
    train, val, test = load("train"), load("val"), load("test")
    meta = json.loads((DATA / "meta.json").read_text())

    # The arm's algebra, recorded before the fit -- the property the flow enforced
    # by wiring `dla_report` upstream of `fit`, kept here by doing it first.
    certificate = (
        dla_check(settings.get("ansatz", "XY_Brickwork"), n_qubits=4)
        if settings.get("model") == "qfm"
        else None
    )

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
        "trace": [{k: r[k] for k in ("epoch", "train_loss", "val_loss")} for r in trace],
        "seconds": round(time.monotonic() - started, 1),
    }


def _safe(args: tuple[dict[str, Any], int]) -> dict[str, Any]:
    """Run a cell, returning the failure rather than killing the pool."""
    settings, epochs = args
    try:
        return one_cell(dict(settings), epochs)
    except Exception:  # a single bad cell must not cost the arm
        return {"cell": settings, "error": traceback.format_exc(limit=4)}


def run_arm(arm: str, *, seeds: int, jobs: int, epochs: int, out: Path) -> list[dict[str, Any]]:
    """Run every cell x seed of one arm, ``jobs`` at a time, saving as it goes."""
    work = [({**cell, "seed": seed}, epochs) for cell in cells(arm) for seed in range(seeds)]
    path = out / f"arm_{arm}.json"
    done: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
    seen = _already_run(out)
    work = [w for w in work if json.dumps(w[0], sort_keys=True) not in seen]
    print(f"arm {arm}: {len(work)} to run ({len(seen)} already done), {jobs} at a time", flush=True)

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for record in pool.map(_safe, work):
            done.append(record)
            # Written after every cell: a driver that dies costs one run, not an arm.
            path.write_text(json.dumps(done, indent=1))
            mark = "FAILED" if "error" in record else f"{record['seconds']:.0f}s"
            print(
                f"  [{len(done)}/{len(done) + len(work)}] {mark} {label(record['cell'])}",
                flush=True,
            )
    return done


#: The `generate` run whose splits `DATA` was exported from. Passing it to
#: `--fluksio` runs the same cells through the engine instead of in process, which
#: is where a study belongs when the engine is healthy: each cell then carries a run
#: id, a certificate recorded upstream of its fit, and its streamed metrics.
def run_arm_fluksio(
    arm: str, dataset: str, *, seeds: int, jobs: int, epochs: int, out: Path
) -> list[dict[str, Any]]:
    """Run one arm through a Fluksio engine, resuming whatever already finished."""
    import time as clock

    from fluksio.sdk.client import Client

    client = Client()
    result = client.run(dataset).get("result") or {}
    names = ("dataset_train", "dataset_val", "dataset_test", "dataset_meta")
    missing = [name for name in names if name not in result]
    if missing:
        raise SystemExit(f"run {dataset} has no {missing}; is it a finished generate run?")
    data = {name: result[name] for name in names}

    path = out / f"arm_{arm}.json"
    done: list[dict[str, Any]] = json.loads(path.read_text()) if path.exists() else []
    # Across every arm, not just this one: the baseline is a cell of all
    # three, and submitting it per arm ran it twice for identical numbers.
    seen = _already_run(out)
    queue = [
        {**cell, "seed": seed}
        for cell in cells(arm)
        for seed in range(seeds)
        if json.dumps({**cell, "seed": seed}, sort_keys=True) not in seen
    ]
    total = len(done) + len(queue)
    print(f"arm {arm}: {len(queue)} to submit ({len(seen)} already done), {jobs} in flight")

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
            payload = handle.result
            done.append(
                {
                    "cell": settings,
                    "run": handle.id,
                    "final_metrics": payload.get("final_metrics"),
                    "test_metrics": {
                        name: (payload.get("test_metrics") or {}).get(name)
                        for name in ("known", "unknown")
                    },
                    # The flow certifies an ansatz for every run; a classical arm
                    # has none, and printing one in its row would invent a circuit.
                    "dla_report": (
                        payload.get("dla_report") if settings.get("model") == "qfm" else None
                    ),
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
    return " ".join(parts) or "base"


def _mean(values: list[Any], places: int = 3) -> str:
    """Mean of the finite numbers in ``values``, or ``--`` if there are none."""
    clean = [v for v in values if isinstance(v, (int, float))]
    return f"{sum(clean) / len(clean):.{places}f}" if clean else "--"


def report(paths: list[Path]) -> None:
    """Print one markdown table per arm, averaged over seeds.

    Read the **known** subset: the test split is 94% unknown topologies by
    construction, so an overall number is dominated by a subset nothing solves.
    Purity is start -> end, because the purity observable is a trajectory rather
    than a level, and it is reported **in units of that arm's own uniform-prior
    mean** -- `mu_n` is 0.8125 for `XY_Brickwork`, 1.25 for `XY_Ring`
    and 15 for `Circuit_19`, so raw purities are not comparable across arm C.
    """
    from partiqledtr.analysis import uniform_prior_mean

    for path in sorted(paths):
        rows: dict[str, list[dict[str, Any]]] = {}
        for record in json.loads(path.read_text()):
            if "error" not in record:
                rows.setdefault(label(record["cell"]), []).append(record)
        if not rows:
            continue
        print(f"\n### {path.stem}\n")
        print(
            "| cell | n | params | dim_g | d_Z | train loss | purity/mu_n | acc known "
            "| perfect known | valid known | acc unknown |"
        )
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, group in rows.items():
            final = [r["final_metrics"] or {} for r in group]
            known = [(r["test_metrics"] or {}).get("known") or {} for r in group]
            unknown = [(r["test_metrics"] or {}).get("unknown") or {} for r in group]
            dla = group[0].get("dla_report") or {}
            mu = uniform_prior_mean((final[0].get("config") or {}).get("ansatz", "XY_Brickwork"), 4)

            # `*_repaired` where a now-retired repair script recomputed the
            # observable on a representative subset; the raw keys are what the run
            # recorded on the biased one, and are not comparable across cells.
            def _p(record: dict[str, Any], name: str) -> float | None:
                return record.get(f"{name}_repaired", record.get(name))

            # `_p` prefers a repaired value where one exists -- the preconditioner-free
            # cells, whose observable was recomputed exactly offline -- and otherwise
            # takes the recorded one, which every remaining record produced after the
            # purity-subset fix, so there is no longer an untrustworthy end value to flag.
            start = _mean([v / mu for f in final if (v := _p(f, "g_purity_initial"))], 2)
            end = _mean([v / mu for f in final if (v := _p(f, "val_g_purity"))], 2)
            print(
                f"| {name} | {len(group)} | {final[0].get('n_params', '--')} "
                f"| {dla.get('dim_g', '--')} | {dla.get('n_diag_words', '--')} "
                f"| {_mean([f.get('train_loss') for f in final], 4)} "
                f"| {start} -> {end} "
                f"| {_mean([t.get('accuracy') for t in known])} "
                f"| {_mean([t.get('perfect') for t in known])} "
                f"| {_mean([t.get('valid_tree_strict') for t in known])} "
                f"| {_mean([t.get('accuracy') for t in unknown])} |"
            )


def main() -> None:
    """Parse arguments and either run an arm or report what has been run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["a", "b", "c", "baseline"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=6)
    # 40, to match the earlier ceiling check of the QFM against the classical GNN --
    # the table this study exists to beat. The quantum arm's loss was flat there
    # from epoch 5, so 40 separates "learns" from "does not" at 60% less cost than
    # the flow default.
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--fluksio", metavar="GENERATE_RUN", help="submit through the engine")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(list(args.out.glob("arm_*.json")))
        return
    if not args.arm:
        parser.error("--arm is required unless --report")

    common = {"seeds": args.seeds, "jobs": args.jobs, "epochs": args.epochs, "out": args.out}
    if args.fluksio:
        run_arm_fluksio(args.arm, args.fluksio, **common)
    else:
        run_arm(args.arm, **common)
    report([args.out / f"arm_{args.arm}.json"])


if __name__ == "__main__":
    main()
