"""Per-cell trajectories of the smoke block: loss, g-purity and angle shape.

Three panels per cell, one line per seed. The middle panel is the study's
question -- whether the purity trajectory and the loss move together now that
the readout is in the algebra -- and the right panel separates the two ways a
purity can rise (spread toward uniform vs pin at pi/2, D92).

    python dev/s3-readout-channel/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from partiqledtr.analysis import uniform_prior_mean

STUDY = Path(__file__).resolve().parent
FIGURES = STUDY / "figures"

PANELS = (
    ("val_loss", "validation loss"),
    ("g_purity", "g-purity / mu_n"),
    ("mean_sin2", "mean sin^2(theta)"),
)


def main() -> None:
    """Render the smoke block's trajectories, one figure for all cells."""
    records = [
        r for r in json.loads((STUDY / "results" / "smoke.json").read_text()) if "error" not in r
    ]

    def cell_key(record):
        cell = record["cell"]
        return " ".join(f"{k}={v}" for k, v in sorted(cell.items()) if k != "seed")

    cells = sorted({cell_key(r) for r in records})
    FIGURES.mkdir(exist_ok=True)

    fig, axes = plt.subplots(len(cells), len(PANELS), figsize=(11, 3 * len(cells)), squeeze=False)
    for row, cell in enumerate(cells):
        group = [r for r in records if cell_key(r) == cell]
        mu = uniform_prior_mean(group[0]["final_metrics"]["config"]["ansatz"], 4)
        for col, (key, title) in enumerate(PANELS):
            ax = axes[row][col]
            for record in group:
                epochs = [t["epoch"] for t in record["trace"]]
                values = [t[key] / (mu if key == "g_purity" else 1.0) for t in record["trace"]]
                ax.plot(epochs, values, lw=0.9, alpha=0.8)
            if key == "g_purity":
                ax.axhline(1.0, ls=":", lw=0.8, color="gray")  # the uniform-prior mean
            ax.set_title(f"{cell}: {title}", fontsize=9)
            ax.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(FIGURES / "smoke_trajectories.png", dpi=150)
    print(f"wrote {FIGURES / 'smoke_trajectories.png'}")


if __name__ == "__main__":
    main()
