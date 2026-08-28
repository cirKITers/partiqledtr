"""Render the encoding x weight g-purity table of a finished ``generate`` run.

ROADMAP phase 4b arm B's central question, answered without training anything:
spectral preconditioning is a property of data plus encoding, so if an exponential
spectrum lifts a collapsed encoding off the floor it has to show here first
(``DECISIONS.md`` D97).

    python dev/s1-encoding-purity/encoding_table.py <generate-run-id>
"""

from __future__ import annotations

import sys

from fluksio.sdk.client import Client


def main() -> None:
    """Print the report of one run as a markdown table, purity and below-threshold."""
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    report = (Client().run(sys.argv[1]).get("result") or {}).get("encoding_report")
    if not report:
        raise SystemExit(f"run {sys.argv[1]} has no encoding_report")

    arms = [key for key in report if isinstance(report[key], dict)]
    cells = list(report[arms[0]])
    ansatz, threshold = report.get("ansatz"), report[arms[0]][cells[0]]["threshold"]
    uniform = report[arms[0]][cells[0]]["uniform_mean"]
    print(f"ansatz {ansatz}, mu_4 = {uniform:.4f}, threshold mu_4/2 = {threshold:.4f}\n")
    print("| cell | " + " | ".join(arms) + " |")
    print("| --- |" + " --- |" * len(arms))
    for cell in cells:
        row = [
            f"{report[arm][cell]['mean_purity']:.3f} ({report[arm][cell]['below_threshold']:.0%})"
            for arm in arms
        ]
        print(f"| {cell} | " + " | ".join(row) + " |")
    print("\nmean g-purity (fraction of edges below mu_4/2)")


if __name__ == "__main__":
    main()
