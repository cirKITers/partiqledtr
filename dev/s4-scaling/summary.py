"""Build the s4 summary CSV and four-panel figure.

Panels compare g-purity, validation loss, their association, and known-topology
accuracy across the graph arms. ``--plot`` redraws from the existing CSV.

Usage: ``python dev/s4-scaling/summary.py [--plot]``.
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
ARMS = ("XY_Cycle", "XY_Ladder", "XY_OddChord")

#: Arm -> colour. The floored control keeps s3's grey; the two floor-free arms
#: take s3's two input-law colours.
COLOR = {"XY_Cycle": "#2a78d6", "XY_Ladder": "#cc4c02", "XY_OddChord": "#8d8b86"}
NAME = {"XY_Cycle": "cycle", "XY_Ladder": "ladder", "XY_OddChord": "odd-chord"}


def export() -> None:
    """Flatten ``smoke.json`` into one tidy CSV: one row per run per epoch.

    Per-run constants (accuracy, first-difference correlation, where the run
    came from) repeat on every row of their run, so the file is self-contained
    for any regrouping.
    """
    from scipy import stats

    from partiqledtr.analysis import uniform_prior_mean

    records = [
        r for r in json.loads((STUDY / "results" / "smoke.json").read_text()) if "error" not in r
    ]
    rows = []
    for record in records:
        cell, trace = record["cell"], record["trace"]
        mu = uniform_prior_mean(cell["ansatz"], cell["n_qubits"])
        purity = np.array([t["g_purity"] for t in trace])
        loss = np.array([t["val_loss"] for t in trace])
        r_diff = (
            f"{stats.pearsonr(np.diff(purity), np.diff(loss)).statistic:.4f}"
            if purity.std() > 0
            else ""
        )
        known = record["test_metrics"]["known"]
        base = {
            "dataset": record.get("dataset", 0),
            "ansatz": cell["ansatz"],
            "dim_g": record["dla_report"]["dim_g"],
            "algebra": "floored" if record["dla_report"]["n_diag_words"] else "floor-free",
            "preconditioner": cell["preconditioner"],
            "seed": cell["seed"],
            # Seeds 0-2 ran in process on the pre-jaqsi stack and were imported.
            "source": "import" if record.get("imported") else "engine",
            "run": record.get("run", ""),
            "acc_known": f"{known['accuracy']:.4f}",
            "perfect_known": f"{known['perfect']:.4f}",
            "r_diff_pearson": r_diff,
        }
        # Epoch 0 is the pre-training anchor, as in s3: the purity is measured
        # before the first step, the loss is not.
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
                    # int: the engine streams every port as a float.
                    "epoch": int(t["epoch"]) + 1,
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
    """Group rows by run (dataset, ansatz, preconditioner, seed)."""
    runs: dict[tuple, list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("dataset", "0"), row["ansatz"], row["preconditioner"], row["seed"])
        runs.setdefault(key, []).append(row)
    return runs


def _band(ax, rows: list[dict[str, str]], column: str, color: str, style: str, label: str) -> None:
    """Mean +- sd band over the runs in ``rows``, per epoch (blank cells dropped)."""
    kept = [r for r in rows if r[column]]
    series = np.array(
        [[float(r[column]) for r in sorted(run, key=lambda r: int(r["epoch"]))]
         for run in _runs(kept).values()]
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
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    (pur, loss), (corr, acc) = axes

    # (a) + (b): trajectories, one colour per arm, dashed for the raw control.
    for arm in ARMS:
        for pre, style, suffix in (("mlp", "-", "MLP"), ("none", "--", "raw")):
            sel = [r for r in rows if r["ansatz"] == arm and r["preconditioner"] == pre]
            _band(pur, sel, "g_purity_over_mu", COLOR[arm], style, f"{NAME[arm]}, {suffix}")
            _band(loss, sel, "val_loss", COLOR[arm], style, f"{NAME[arm]}, {suffix}")
    # Below the line: every arm starts at or above it on this chart.
    pur.axhline(1.0, color="black", ls=":", lw=0.8)
    pur.text(0.5, 0.98, r"uniform-prior mean $\mu_n$", ha="left", va="top", fontsize=8)
    pur.set_ylim(0.9, 2.45)  # headroom for the legend
    pur.set_ylabel(r"g-purity / $\mu_n$")
    pur.set_title(r"(a) the preconditioner contracts toward $\mu_n$", fontsize=10, loc="left")
    pur.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    loss.set_ylabel("validation loss")
    loss.set_title("(b) validation loss bottoms out near epoch 10", fontsize=10, loc="left")
    for ax in (pur, loss):
        ax.set_xlabel("epoch")

    # (c): first-difference coupling, one dot per mlp run.
    for x, arm in enumerate(ARMS):
        values = [
            float(run[0]["r_diff_pearson"])
            for run in _runs([r for r in rows if r["ansatz"] == arm]).values()
            if run[0]["r_diff_pearson"]
        ]
        _dots(corr, x, values, COLOR[arm], filled=True)
    corr.axhline(0.0, color="black", ls=":", lw=0.8)
    corr.set_xticks(range(len(ARMS)), [NAME[a] for a in ARMS], fontsize=8)
    corr.set_ylabel(r"Pearson $r(\Delta P_{\mathfrak{g}}, \Delta L)$")
    corr.set_title("(c) no purity-loss coupling here (MLP runs)", fontsize=10, loc="left")
    corr.set_xlim(-0.5, len(ARMS) - 0.5)

    # (d): accuracy, raw -> mlp per arm.
    ticks, labels = [], []
    for i, arm in enumerate(ARMS):
        x = 1.6 * i
        for pre, offset, filled in (("none", 0.0, False), ("mlp", 0.6, True)):
            sel = _runs([r for r in rows if r["ansatz"] == arm and r["preconditioner"] == pre])
            _dots(acc, x + offset, [float(run[0]["acc_known"]) for run in sel.values()],
                  COLOR[arm], filled)  # fmt: skip
            ticks.append(x + offset)
            labels.append("MLP" if pre == "mlp" else "raw")
        dim_g = next(r["dim_g"] for r in rows if r["ansatz"] == arm)
        acc.text(
            x + 0.3,
            -0.13,
            f"{NAME[arm]} ({dim_g})",
            transform=acc.get_xaxis_transform(),
            ha="center",
            fontsize=8,
            color=COLOR[arm],
            clip_on=False,
        )
    acc.set_xticks(ticks, labels, fontsize=8)
    acc.set_ylabel("test accuracy (known topologies)")
    acc.set_title(
        r"(d) accuracy rises with $\dim\mathfrak{g}$; MLP adds ~0.01", fontsize=10, loc="left"
    )
    acc.set_xlim(-0.5, 1.6 * (len(ARMS) - 1) + 1.1)

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
