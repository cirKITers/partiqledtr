"""Figures for ROADMAP phase 4b.

Three claims, one figure each:

1. ``purity_vs_task`` -- the input-distribution sensitivity the unflattening theory
   measures and the task performance it is supposed to bear on are **orthogonal**.
   The depth arm moves performance at fixed purity (a vertical spread), the encoding
   arm moves purity over 5.7x at fixed performance (a horizontal spread). A scalar
   summary of the angle distribution is deliberately *not* an axis here: g-purity is
   the theory's own scalar, and it is the one the closed forms are written in.
2. ``preconditioner_channel`` -- what the trained preconditioner can do to the encoded
   distribution depends on the encoding weights, which is spectral preconditioning
   seen from the model side rather than from the data.
3. ``angle_plane`` -- the angle distribution itself, in the two coordinates that
   separate its regimes (``DECISIONS.md`` D92). One point per qubit, never pooled.

    python dev/s2-expressivity/figures.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from partiqledtr.analysis import uniform_prior_mean
from partiqledtr.models.qfm import N_QUBITS

#: This study's folder, resolved from this module rather than the working directory,
#: so a driver behaves the same wherever it is started from.
STUDY = Path(__file__).resolve().parent
RESULTS = STUDY / "results"
FIGURES = STUDY / "figures"

#: Categorical slots 1-3 of the validated palette. Three is the cap for scatter
#: forms, where every pair is on screen at once; the fourth slot fails the
#: all-pairs floors. Marker shape repeats the encoding so identity is never colour
#: alone, and every point is directly labelled, which is also the relief the aqua
#: slot's contrast warning requires.
SERIES = {
    "depth": ("#2a78d6", "o", "depth (arm A)"),
    "encoding": ("#eb6834", "s", "encoding (arm B)"),
    "ansatz": ("#1baf7a", "^", "ansatz (arm C)"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

#: Directly labelled points: the extremes of each arm and every ansatz. The rest of
#: arm B is the cluster that sits on the preconditioned limit with no effect on the
#: score -- labelling it would crowd the figure to say "these are interchangeable",
#: which is exactly what an unlabelled cluster already says. The table carries the
#: per-cell detail.
LABELLED = {
    "L2",
    "L16",
    "ham-dia",
    "ter-dia",
    "legacy ham-cyc",
    "legacy ter-cyc",
    "Brickwork",
    "Ring",
    "AllPairs",
    "C19",
}

#: The classical reference, from `arm_baseline`. Quoted rather than plotted: at
#: 0.953 / 0.740 it is off every scale the quantum arms occupy, and squeezing it in
#: would cost the resolution the quantum comparison needs.
GNN = {"accuracy": 0.953, "perfect": 0.740}


def _style() -> None:
    """Recessive axes, thin marks, ink-coloured text."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.size": 8.5,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )


def configurations() -> list[dict[str, Any]]:
    """One row per configuration, averaged over seeds."""
    # Keyed on the configuration, not on the arm that ran it: the phase-4 baseline
    # is a cell of all three arms, and keeping it three times would both duplicate a
    # point and throw away the seeds the other arms spent on it.
    rows: dict[tuple, list[dict[str, Any]]] = {}
    for path in sorted(RESULTS.glob("arm_*.json")):
        for record in json.loads(path.read_text()):
            config = (record.get("final_metrics") or {}).get("config") or {}
            if "error" in record or config.get("model") != "qfm":
                continue
            axes = ("encoding", "angle_map", "ansatz", "n_layers", "enc_weights", "enc_reupload")
            # `frontend` is what the 280 phase-4b runs recorded; the axis was renamed
            # to `preconditioner` afterwards, and their result files were not rewritten.
            arm = config.get("preconditioner", config.get("frontend"))
            key = (arm, *(config.get(a) for a in axes))
            rows.setdefault(key, []).append(record)

    def get(record: dict[str, Any], name: str) -> Any:
        # The `*_repaired` keys were written once, by a repair script that has since
        # been retired: records produced before the purity subset was fixed (D105)
        # measured the observable on one topology, and the recomputed value sits
        # beside the original rather than over it. Anything recorded after the fix
        # carries the plain key alone, which is what the fallback reads.
        final = record["final_metrics"]
        return final.get(f"{name}_repaired", final.get(name))

    out = []
    for key, group in rows.items():
        preconditioner, encoding, _map, ansatz, n_layers, weights, reupload = key
        # One seed per (configuration, seed): the same cell in two arm files is the
        # same run of the same thing, so it must not count twice.
        group = list({r["cell"]["seed"]: r for r in group}.values())
        final = [r["final_metrics"] for r in group]
        mu = uniform_prior_mean(ansatz, N_QUBITS)
        known = [r["test_metrics"]["known"] for r in group]
        out.append(
            {
                "preconditioner": preconditioner,
                "ansatz": ansatz,
                "n_layers": n_layers,
                "weights": weights,
                "reupload": reupload,
                "clustered": encoding == "legacy",
                "n_params": final[0]["n_params"],
                "dim_g": (group[0].get("dla_report") or {}).get("dim_g"),
                "d_z": (group[0].get("dla_report") or {}).get("n_diag_words"),
                "purity_start": get(group[0], "g_purity_initial") / mu,
                "purity_end": st.mean(get(r, "val_g_purity") for r in group) / mu,
                "purity_end_sd": (
                    st.stdev(get(r, "val_g_purity") / mu for r in group) if len(group) > 1 else 0.0
                ),
                "mu_n": mu,
                "n_seeds": len(group),
                "angles_start": get(group[0], "angle_stats_initial"),
                "angles_end": get(group[0], "angle_stats_final"),
                "loss": st.mean(f["train_loss"] for f in final),
                # Per seed, so a reader can derive any interval rather than the one
                # we happened to summarise. `purity_start` carries no spread: it is
                # a property of data plus encoding, identical at every seed.
                "seeds": sorted(f["seed"] for f in final),
                # The engine's run id where the cell went through it, so a row of
                # the per-seed csv joins to `fluksio export runs` -- and through
                # that to `code_digest`, which is the only record of what code ran.
                "runs": {r["final_metrics"]["seed"]: r.get("run", "") for r in group},
                "per_seed": {
                    "purity_end": [get(r, "val_g_purity") / mu for r in group],
                    "train_loss": [f["train_loss"] for f in final],
                    **{
                        f"{metric}_{subset}": [r["test_metrics"][subset][metric] for r in group]
                        for subset in ("known", "unknown")
                        for metric in ("accuracy", "perfect", "valid_tree_strict")
                    },
                },
                **{
                    metric: (
                        st.mean(k[metric] for k in known),
                        st.stdev(k[metric] for k in known) if len(known) > 1 else 0.0,
                    )
                    for metric in ("accuracy", "perfect")
                },
            }
        )
    return out


def _series_of(row: dict[str, Any]) -> str:
    """Which axis this configuration varies from the phase-4 baseline."""
    if row["n_layers"] != 2:
        return "depth"
    return "ansatz" if row["ansatz"] != "XY_Brickwork" else "encoding"


def _tag(row: dict[str, Any]) -> str:
    """The shortest label that identifies a configuration in a figure."""
    if row["n_layers"] != 2:
        return f"L{row['n_layers']}"
    if row["ansatz"] != "XY_Brickwork":
        return row["ansatz"].replace("XY_", "").replace("Circuit_", "C")
    stem = {"hamming": "ham", "ternary": "ter", "binary": "bin", "ternary_pair": "pair"}
    name = f"{stem[row['weights']]}-{row['reupload'][:3]}"
    return f"legacy {name}" if row["clustered"] else name


def _place(points, axis):
    """Stack labels within a column and return them with their leader lines.

    Many configurations share an x exactly -- depth and ansatz do not touch the
    encoding, so they sit at the same purity, which is itself the point of the
    figure. Labels are stacked by y *within* a column and joined to their mark by a
    leader, rather than nudged in x, which would misplace the data.
    """
    (x0, x1), (y0, y1) = axis.get_xlim(), axis.get_ylim()
    width, step = 0.045 * (x1 - x0), 0.052 * (y1 - y0)
    columns: dict[int, list] = {}
    for x, y, tag in points:
        columns.setdefault(round((x - x0) / width), []).append((x, y, tag))

    placed = []
    for column in columns.values():
        column.sort(key=lambda p: p[1])
        # Centre the stack on the column's own points, so labels stay local.
        middle = sum(p[1] for p in column) / len(column)
        base = middle - step * (len(column) - 1) / 2
        for index, (x, y, tag) in enumerate(column):
            at = min(max(base + index * step, y0 + step / 2), y1 - step / 2)
            side = "right" if x > x0 + 0.70 * (x1 - x0) else "left"
            placed.append((x, y, at, tag, side))
    return placed


def purity_vs_task(rows: list[dict[str, Any]], path: Path) -> None:
    """Claim 1: the theory's variable and the task's score do not move together."""
    rows = [r for r in rows if r["preconditioner"] == "none"]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharex=True)
    for axis, metric, name in zip(
        axes, ("accuracy", "perfect"), ("per-element accuracy", "Perfect-LCAG"), strict=True
    ):
        axis.axvline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        for series, (colour, marker, legend) in SERIES.items():
            group = [r for r in rows if _series_of(r) == series]
            if not group:
                continue
            # Filled marker: floor-free (d_Z = 0), so the input distribution decides.
            # Open marker: floored, so the algebra pins the purity whatever the data
            # does -- the certificate, carried on the mark rather than in a caption.
            for floored in (False, True):
                part = [r for r in group if (r["d_z"] > 0) is floored]
                if not part:
                    continue
                axis.errorbar(
                    [r["purity_start"] for r in part],
                    [r[metric][0] for r in part],
                    yerr=[r[metric][1] for r in part],
                    fmt=marker,
                    ms=7,
                    mfc="#fcfcfb" if floored else colour,
                    mec=colour,
                    mew=1.6,
                    ecolor=colour,
                    elinewidth=1.2,
                    capsize=0,
                    ls="none",
                    label=None if floored else legend,
                    zorder=3,
                )
        span = axis.get_xlim()[1] - axis.get_xlim()[0]
        for x, y, at, tag, side in _place(
            [(r["purity_start"], r[metric][0], _tag(r)) for r in rows if _tag(r) in LABELLED],
            axis,
        ):
            shift = -0.028 * span if side == "right" else 0.028 * span
            axis.annotate(
                tag,
                xy=(x, y),
                xytext=(x + shift, at),
                ha=side,
                va="center",
                fontsize=6.4,
                color=MUTED,
                arrowprops={
                    "arrowstyle": "-",
                    "color": GRID,
                    "lw": 0.7,
                    "shrinkA": 1,
                    "shrinkB": 4,
                },
            )
        axis.set_title(name + " (known topologies)", loc="left", color=INK)
        axis.set_xlabel(r"encoded-state g-purity  $P_\mathfrak{g}\,/\,\mu_n$")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("score on the known subset")
    for axis in axes:
        # Outside the data area: the line is a reference, not a series, and at this
        # density any in-axis placement lands on a mark.
        axis.annotate(
            r"$P_\mathfrak{g}=\mu_n$",
            (1.0, axis.get_ylim()[1]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=7,
            color=MUTED,
        )
    handles = [
        plt.Line2D(
            [], [], ls="none", marker="o", ms=7, mfc=MUTED, mec=MUTED, label="floor-free ($d_Z=0$)"
        ),
        plt.Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=7,
            mfc="#fcfcfb",
            mec=MUTED,
            mew=1.6,
            label="floored ($d_Z>0$)",
        ),
    ]
    axes[0].legend(loc="upper left", fontsize=7.5, labelcolor=MUTED)
    axes[1].legend(handles=handles, loc="lower right", fontsize=7.5, labelcolor=MUTED)
    figure.suptitle(
        "Input-distribution sensitivity does not predict task performance",
        x=0.008,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.008,
        0.905,
        "Depth moves the score at fixed purity; the encoding moves purity 5.7x at fixed "
        f"score.   Classical GNN, off scale: {GNN['accuracy']:.3f} accuracy, "
        f"{GNN['perfect']:.3f} Perfect-LCAG.",
        ha="left",
        fontsize=7.8,
        color=MUTED,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.885))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def preconditioner_channel(rows: list[dict[str, Any]], path: Path) -> None:
    """Claim 2: what the preconditioner can do to the purity depends on the encoding.

    Spectral preconditioning and the DLA floor, seen from the model side. Drawn as
    a strip of the per-seed outcomes rather than an arrow to their mean, because
    the mean is not the finding: on a floor-free Hamming arm the trained preconditioner
    lands anywhere between an annihilated state and one above where it started, and
    an arrow would report the midpoint of that as though it were a displacement.
    """
    cells = _fig2_cells(rows)
    families = {
        "floor_free_hamming": ("#2a78d6", "floor-free, Hamming - preconditioner unconstrained"),
        "floor_free_dissociated": ("#eb6834", "floor-free, dissociated - encoding pins it"),
        "floored": ("#1baf7a", r"floored ($d_Z>0$) - the algebra pins it"),
    }

    figure, axis = plt.subplots(figsize=(7.8, 3.8))
    axis.axvline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    seen = set()
    for index, row in enumerate(cells):
        colour, legend = families[_family_of(row)]
        ends = row["per_seed"]["purity_end"]
        axis.plot(
            ends,
            [index] * len(ends),
            "o",
            ms=5,
            mfc=colour,
            mec="#fcfcfb",
            mew=0.7,
            alpha=0.75,
            ls="none",
            zorder=3,
            label=None if legend in seen else legend,
        )
        seen.add(legend)
        # Where it started, and where those seeds ended on average.
        axis.plot([row["purity_start"]], [index], "|", ms=15, mew=2.0, color=INK, zorder=4)
        axis.plot([st.mean(ends)], [index], "|", ms=15, mew=2.0, color=colour, zorder=4)
        axis.annotate(
            f"sd {st.stdev(ends):.2f}" if len(ends) > 1 else "",
            (max(ends), index),
            textcoords="offset points",
            xytext=(10, 0),
            va="center",
            fontsize=7,
            color=MUTED,
        )
    axis.set_yticks(range(len(cells)), [_tag(r) for r in cells], fontsize=7.5)
    axis.set_xlabel(
        r"encoded-state g-purity  $P_\mathfrak{g}\,/\,\mu_n$"
        "   (one dot per seed; black tick: before training)"
    )
    axis.set_xlim(-0.08, 2.35)
    axis.set_ylim(-0.7, len(cells) - 0.3)
    axis.grid(axis="y", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="center right", fontsize=7.5, labelcolor=MUTED)
    axis.set_title(
        "How far a trained preconditioner can move the encoded state, and how reliably",
        loc="left",
        color=INK,
        fontsize=10.5,
        pad=24,
    )
    spread = {
        name: st.mean(st.stdev(r["per_seed"]["purity_end"]) for r in cells if _family_of(r) == name)
        for name in families
    }
    axis.annotate(
        "spread of the outcome (mean sd over seeds):  "
        + "     ".join(
            f"{name.replace('floor_free_', '')} {spread[name]:.3f}"
            for name in ("floor_free_hamming", "floor_free_dissociated", "floored")
        ),
        (0.0, 1.0),
        xycoords="axes fraction",
        textcoords="offset points",
        xytext=(0, 7),
        fontsize=7.6,
        color=MUTED,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def angle_plane(rows: list[dict[str, Any]], path: Path) -> None:
    r"""Claim 3: the angle distribution needs two coordinates, not one.

    A g-purity rises both when angles spread toward uniform and when they pin near
    :math:`\pi/2`, and those are opposite in what they do to the input information
    (``DECISIONS.md`` D92). ``mean_sin2`` separates them -- 0.5 uniform, 1 pinned,
    0 clustered -- and total variation says how far from uniform the law is. One
    point per *qubit*: sites peaking at different angles average into something that
    looks flat, which is the artefact the latent-drift memo warns about.
    """
    families = {
        "hamming": ("#2a78d6", "pair-polar, Hamming"),
        "dissociated": ("#eb6834", "pair-polar, dissociated"),
        "clustered": ("#1baf7a", "legacy (clustered)"),
    }

    def family_of(row: dict[str, Any]) -> str:
        if row["clustered"]:
            return "clustered"
        return "hamming" if row["weights"] == "hamming" else "dissociated"

    figure, axis = plt.subplots(figsize=(6.6, 4.6))
    for name, (mark, note) in {
        "uniform": ((0.0, 0.5), "uniform law"),
        "pinned": ((1.0, 1.0), r"pinned at $\pi/2$"),
        "collapsed": ((1.0, 0.0), "collapsed at 0"),
    }.items():
        del name
        axis.plot(*mark, marker="x", ms=9, mew=1.6, color=MUTED, zorder=2)
        # The uniform corner sits inside the data, so its label drops below it.
        offset = (-8, 0) if mark[0] > 0.5 else (0, -12)
        axis.annotate(
            note,
            mark,
            textcoords="offset points",
            xytext=offset,
            ha="right" if mark[0] > 0.5 else "center",
            va="center" if mark[0] > 0.5 else "top",
            fontsize=7.4,
            color=MUTED,
        )

    seen = set()
    for row in (r for r in rows if r["preconditioner"] == "none"):
        colour, legend = families[family_of(row)]
        stats = row["angles_start"]
        for site, (tv, sin2) in enumerate(
            zip(stats["tv_uniform"], stats["mean_sin2"], strict=True)
        ):
            # Circles are the (px, py) azimuths, triangles the (pz, E) sites, which
            # E >= |pz| confines to about [pi/4, 3pi/4] whatever the data does (D79).
            azimuth = site % 2 == 0
            axis.plot(
                tv,
                sin2,
                marker="o" if azimuth else "^",
                ms=6.5,
                mfc=colour,
                mec="#fcfcfb",
                mew=0.9,
                alpha=0.85,
                ls="none",
                zorder=3,
                label=None if legend in seen else legend,
            )
            seen.add(legend)

    axis.set_xlabel("total variation from uniform   (per qubit)")
    axis.set_ylabel(r"$\langle \sin^2\theta \rangle$   (per qubit)")
    axis.set_xlim(-0.05, 1.05)
    axis.set_ylim(-0.08, 1.08)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    shapes = [
        plt.Line2D(
            [], [], ls="none", marker="o", ms=6.5, mfc=MUTED, mec="#fcfcfb", label=r"$\phi$ site"
        ),
        plt.Line2D(
            [], [], ls="none", marker="^", ms=6.5, mfc=MUTED, mec="#fcfcfb", label=r"$\alpha$ site"
        ),
    ]
    first = axis.legend(loc="upper left", fontsize=7.4, labelcolor=MUTED)
    axis.add_artist(first)
    axis.legend(handles=shapes, loc="lower left", fontsize=7.4, labelcolor=MUTED)
    axis.set_title(
        "One number cannot describe the angle distribution",
        loc="left",
        color=INK,
        fontsize=10.5,
        pad=30,
    )
    axis.annotate(
        "Equal distance from uniform, opposite consequences: the two corners on the right "
        "are\nthe collapsed and the pinned law, and a scalar summary puts them in the same place.",
        (0.0, 1.0),
        xycoords="axes fraction",
        textcoords="offset points",
        xytext=(0, 5),
        fontsize=7.4,
        color=MUTED,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _spread(values: list[float]) -> tuple[float, float, float]:
    """Mean, sample standard deviation and standard error of a seed group."""
    mean = st.mean(values)
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return mean, sd, sd / len(values) ** 0.5 if values else 0.0


def _family_of(row: dict[str, Any]) -> str:
    """Why the preconditioner can or cannot move this configuration's purity."""
    if row["d_z"] > 0:
        return "floored"
    return "floor_free_hamming" if row["weights"] == "hamming" else "floor_free_dissociated"


def _fig2_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The configurations figure 2 draws, in the order it draws them."""
    cells = [r for r in rows if r["preconditioner"] == "mlp" and not r["clustered"]]
    order = {"floored": 0, "floor_free_dissociated": 1, "floor_free_hamming": 2}
    cells.sort(key=lambda r: (order[_family_of(r)], abs(r["purity_end"] - r["purity_start"])))
    return cells


#: Metrics carried per configuration, each as mean / sd / sem over seeds.
_SPREAD_METRICS = (
    "purity_end",
    "shift",
    "train_loss",
    "accuracy_known",
    "perfect_known",
    "valid_tree_strict_known",
    "accuracy_unknown",
)


def preconditioner_channel_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the data behind :func:`preconditioner_channel`, one row per configuration.

    Every metric appears as ``<name>``, ``<name>_sd`` and ``<name>_sem`` over the
    seed group, so an error bar is a column rather than a recomputation -- ``sd``
    for the spread of runs, ``sem`` for the uncertainty on the mean, and
    ``n_seeds`` so either can be turned into the other or into a t-interval.

    ``purity_start`` has no spread by construction: it is a property of data plus
    encoding, and a zero-init preconditioner is the identity at epoch 0, so every seed
    of a configuration starts at the same value. ``shift`` is therefore
    ``purity_end - purity_start`` per seed, and its spread is the end's.

    The companion ``*_seeds.csv`` carries the individual runs.
    """
    cells = _fig2_cells(rows)
    context = [
        "label",
        "family",
        "ansatz",
        "enc_weights",
        "enc_reupload",
        "n_layers",
        "preconditioner",
        "dim_g",
        "d_z",
        "mu_n",
        "n_params",
        "n_seeds",
        "purity_start",
        "purity_end_min",
        "purity_end_max",
    ]
    columns = context + [
        f"{metric}{suffix}" for metric in _SPREAD_METRICS for suffix in ("", "_sd", "_sem")
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in cells:
            start = row["purity_start"]
            samples = dict(row["per_seed"])
            samples["shift"] = [value - start for value in samples["purity_end"]]
            record = {
                "label": _tag(row),
                "family": _family_of(row),
                "ansatz": row["ansatz"],
                "enc_weights": row["weights"],
                "enc_reupload": row["reupload"],
                "n_layers": row["n_layers"],
                "preconditioner": row["preconditioner"],
                "dim_g": row["dim_g"],
                "d_z": row["d_z"],
                "mu_n": round(row["mu_n"], 6),
                "n_params": row["n_params"],
                "n_seeds": row["n_seeds"],
                "purity_start": round(start, 6),
                # The spread is the finding on the floor-free Hamming cells, where
                # sd alone understates a range that runs from an annihilated state
                # to one above where it started.
                "purity_end_min": round(min(samples["purity_end"]), 6),
                "purity_end_max": round(max(samples["purity_end"]), 6),
            }
            for metric in _SPREAD_METRICS:
                mean, sd, sem = _spread(samples[metric])
                record[metric] = round(mean, 6)
                record[f"{metric}_sd"] = round(sd, 6)
                record[f"{metric}_sem"] = round(sem, 6)
            writer.writerow(record)


def preconditioner_channel_seeds_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write one row per (configuration, seed) behind figure 2.

    The summary file is derived from this one; anything it does not carry -- a
    median, a bootstrap interval, a paired test across seeds -- can be computed
    here instead of re-run.
    """
    columns = [
        "label",
        "family",
        "seed",
        "run",
        "purity_start",
        "purity_end",
        "shift",
        "train_loss",
        "accuracy_known",
        "perfect_known",
        "valid_tree_strict_known",
        "accuracy_unknown",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in _fig2_cells(rows):
            start, samples = row["purity_start"], row["per_seed"]
            for index, seed in enumerate(row["seeds"]):
                writer.writerow(
                    {
                        "label": _tag(row),
                        "family": _family_of(row),
                        "seed": seed,
                        "run": row["runs"].get(seed, ""),
                        "purity_start": round(start, 6),
                        "purity_end": round(samples["purity_end"][index], 6),
                        "shift": round(samples["purity_end"][index] - start, 6),
                        **{
                            name: round(samples[name][index], 6)
                            for name in (
                                "train_loss",
                                "accuracy_known",
                                "perfect_known",
                                "valid_tree_strict_known",
                                "accuracy_unknown",
                            )
                        },
                    }
                )


def main() -> None:
    """Write every figure."""
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = configurations()
    bare = sum(r["preconditioner"] == "none" for r in rows)
    print(f"{len(rows)} configurations ({bare} without a preconditioner)")
    purity_vs_task(rows, FIGURES / "fig1_purity_vs_task.png")
    preconditioner_channel(rows, FIGURES / "fig2_preconditioner_channel.png")
    angle_plane(rows, FIGURES / "fig3_angle_plane.png")
    preconditioner_channel_csv(rows, FIGURES / "fig2_preconditioner_channel.csv")
    preconditioner_channel_seeds_csv(rows, FIGURES / "fig2_preconditioner_channel_seeds.csv")
    print("wrote fig1, fig2, fig3 and both fig2 csv files")


if __name__ == "__main__":
    main()
