"""The one-figure summary of the phase-4c study (RESEARCH §15).

Four panels, one claim each: (a) the preconditioner rescues the clustered
distribution to the uniform-prior band; (b) the loss descends alongside; (c) the
per-epoch purity-loss coupling exists exactly on the floor-free x clustered
cells and not on the floored control; (d) the task effect follows the same
pattern. Floor-free arms (`XY_Ring`, `XY_Brickwork`) are pooled -- they behave
identically here and the per-arm split stays in the CSV.

The figure is drawn from ``results/summary.csv`` alone, so the plot can be
restyled or rebuilt without touching the run records:

    python dev/s3-readout-channel/summary.py          # rebuild csv, then figure
    python dev/s3-readout-channel/summary.py --plot   # figure from existing csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parent
CSV = STUDY / "results" / "summary.csv"
FLOORED = "XY_AllPairs"

#: Condition -> colour. Clustered (legacy p*E*pi) is the barren input law, the
#: pair-polar law starts favourable, the floored arm is the control.
COLOR = {"clustered": "#cc4c02", "pair_polar": "#2a78d6", "floored": "#8d8b86"}


def export() -> None:
    """Flatten ``full.json`` into one tidy CSV: one row per run per epoch.

    Per-run constants (accuracy, first-difference correlation) repeat on every
    row of their run, so the file is self-contained for any regrouping.
    """
    from scipy import stats

    records = [
        r for r in json.loads((STUDY / "results" / "full.json").read_text()) if "error" not in r
    ]
    from partiqledtr.analysis import uniform_prior_mean

    rows = []
    for record in records:
        cell, trace = record["cell"], record["trace"]
        mu = uniform_prior_mean(cell.get("ansatz", "XY_Ring"), 4)
        purity = np.array([t["g_purity"] for t in trace])
        loss = np.array([t["val_loss"] for t in trace])
        r_diff = (
            f"{stats.pearsonr(np.diff(purity), np.diff(loss)).statistic:.4f}"
            if purity.std() > 0
            else ""
        )
        base = {
            "ansatz": cell.get("ansatz", "XY_Ring"),
            "algebra": "floored" if cell.get("ansatz") == FLOORED else "floor-free",
            "input_law": (
                "clustered" if cell.get("encoding", "legacy") == "legacy" else "pair_polar"
            ),
            "preconditioner": cell["preconditioner"],
            "seed": cell["seed"],
            "acc_known": f"{record['test_metrics']['known']['accuracy']:.4f}",
            "r_diff_pearson": r_diff,
        }
        # Epoch 0 is the pre-training anchor: the purity is measured before the
        # first step (`g_purity_initial`), the loss is not -- without this row
        # the rescue's first jump is invisible, because the streamed trace
        # starts after one epoch of training.
        rows.append(
            {
                **base,
                "epoch": 0,
                "g_purity_over_mu": f"{record['final_metrics']['g_purity_initial'] / mu:.5f}",
                "val_loss": "",
            }
        )
        for t in trace:
            rows.append(
                {
                    **base,
                    "epoch": t["epoch"] + 1,
                    "g_purity_over_mu": f"{t['g_purity'] / mu:.5f}",
                    "val_loss": f"{t['val_loss']:.5f}",
                }
            )
    with CSV.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {CSV} ({len(rows)} rows)")


def _load() -> list[dict[str, str]]:
    with CSV.open() as source:
        return list(csv.DictReader(source))


def _runs(rows: list[dict[str, str]]) -> dict[tuple, list[dict[str, str]]]:
    """Group rows by run (ansatz, input law, preconditioner, seed)."""
    runs: dict[tuple, list[dict[str, str]]] = {}
    for row in rows:
        key = (row["ansatz"], row["input_law"], row["preconditioner"], row["seed"])
        runs.setdefault(key, []).append(row)
    return runs


def _band(ax, rows: list[dict[str, str]], column: str, color: str, style: str, label: str) -> None:
    """Mean +- sd band over the runs in ``rows``, per epoch.

    Blank cells (the epoch-0 loss anchor) are dropped before grouping.
    """
    kept = [r for r in rows if r[column]]
    per_run = _runs(kept)
    series = np.array(
        [[float(r[column]) for r in sorted(run, key=lambda r: int(r["epoch"]))]
         for run in per_run.values()]
    )  # fmt: skip
    start = min(int(r["epoch"]) for r in kept)
    epochs = np.arange(start, start + series.shape[1])
    mean, sd = series.mean(axis=0), series.std(axis=0)
    ax.plot(epochs, mean, style, color=color, lw=1.6, label=label)
    ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)


def _dots(ax, x: float, values: list[float], color: str, filled: bool) -> None:
    """One seed per dot, jittered, with a heavy mean tick."""
    jitter = (np.random.default_rng(0).random(len(values)) - 0.5) * 0.18
    ax.scatter(
        x + jitter,
        values,
        s=18,
        facecolors=color if filled else "none",
        edgecolors=color,
        lw=1.2,
        zorder=3,
    )
    ax.hlines(float(np.mean(values)), x - 0.18, x + 0.18, color=color, lw=2.5, zorder=4)


def plot() -> None:
    """Draw the 2x2 summary figure from the CSV alone."""
    rows = _load()
    free = [r for r in rows if r["algebra"] == "floor-free"]
    floored = [r for r in rows if r["algebra"] == "floored"]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    (pur, loss), (corr, acc) = axes

    # (a) + (b): trajectories, floor-free arms pooled.
    for law, pre, style, label in (
        ("clustered", "mlp", "-", "clustered, MLP"),
        ("clustered", "none", "--", "clustered, raw"),
        ("pair_polar", "mlp", "-", "pair-polar, MLP"),
        ("pair_polar", "none", "--", "pair-polar, raw"),
    ):
        sel = [r for r in free if r["input_law"] == law and r["preconditioner"] == pre]
        _band(pur, sel, "g_purity_over_mu", COLOR[law], style, label)
        _band(loss, sel, "val_loss", COLOR[law], style, label)
    pur.axhline(1.0, color="black", ls=":", lw=0.8)
    pur.text(4.5, 1.04, r"uniform-prior mean $\mu_n$", ha="left", va="bottom", fontsize=8)
    pur.set_ylabel(r"g-purity / $\mu_n$")
    pur.set_title("(a) the preconditioner rescues the clustered law", fontsize=10, loc="left")
    pur.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    loss.set_ylabel("validation loss")
    loss.set_title("(b) the loss descends alongside", fontsize=10, loc="left")
    for ax in (pur, loss):
        ax.set_xlabel("epoch")

    # (c): first-difference coupling, one dot per run that has one.
    polar = [r for r in free if r["input_law"] == "pair_polar"]
    groups = [
        ("floor-free\nclustered", [r for r in free if r["input_law"] == "clustered"], "clustered"),
        ("floor-free\npair-polar", polar, "pair_polar"),
        ("floored\ncontrol", floored, "floored"),
    ]
    for x, (_name, sel, color) in enumerate(groups):
        values = [
            float(run[0]["r_diff_pearson"])
            for run in _runs(sel).values()
            if run[0]["r_diff_pearson"]
        ]
        _dots(corr, x, values, COLOR[color], filled=True)
    corr.axhline(0.0, color="black", ls=":", lw=0.8)
    corr.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
    corr.set_ylabel(r"Pearson $r(\Delta P_{\mathfrak{g}}, \Delta L)$")
    corr.set_title("(c) the coupling is specific (MLP runs)", fontsize=10, loc="left")
    corr.set_xlim(-0.5, len(groups) - 0.5)

    # (d): accuracy, none -> mlp per condition, floored control beside.
    pairs = [("clustered", 0.0), ("pair_polar", 1.6)]
    for law, x in pairs:
        for pre, offset, filled in (("none", 0.0, False), ("mlp", 0.6, True)):
            sel = _runs([r for r in free if r["input_law"] == law and r["preconditioner"] == pre])
            values = [float(run[0]["acc_known"]) for run in sel.values()]
            _dots(acc, x + offset, values, COLOR[law], filled)
    control = [float(run[0]["acc_known"]) for run in _runs(floored).values()]
    _dots(acc, 3.2, control, COLOR["floored"], filled=True)
    acc.set_xticks(
        [0.0, 0.6, 1.6, 2.2, 3.2],
        ["raw", "MLP", "raw", "MLP", "MLP"],
        fontsize=8,
    )
    # Condition labels as a second row below the raw/MLP ticks.
    conditions = [(0.3, "clustered"), (1.9, "pair_polar"), (3.2, "floored")]
    for x, law in conditions:
        acc.text(
            x,
            -0.13,
            law.replace("_", "-"),
            transform=acc.get_xaxis_transform(),
            ha="center",
            fontsize=8,
            color=COLOR[law],
            clip_on=False,
        )
    acc.set_ylabel("test accuracy (known topologies)")
    acc.set_title("(d) the task effect follows the mechanism", fontsize=10, loc="left")
    acc.set_xlim(-0.5, 3.7)

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (STUDY / "figures").mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(STUDY / "figures" / f"summary.{suffix}", dpi=200)
    print(f"wrote {STUDY / 'figures' / 'summary.png'} (and .pdf)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="skip the csv rebuild")
    if not parser.parse_args().plot:
        export()
    plot()
