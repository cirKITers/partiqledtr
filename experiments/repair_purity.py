"""Recompute the encoded-state observables of finished cells on a representative subset.

The g-purity and angle statistics every run recorded were measured on the first 64
validation events, which
the split's topology ordering makes a single topology at a single multiplicity
(``DECISIONS.md`` D105). The *initial* purity is a property of data plus encoding
alone -- no trained parameter enters it, and a zero-init front end is the identity
at epoch 0 -- so it can be recomputed exactly, offline, for every cell that has
already run.

What cannot be recovered without a re-fit is the *final* purity of a cell whose
front end trains, since that depends on parameters no checkpoint here kept. Those
cells are listed rather than guessed.

    python experiments/repair_purity.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx

from partiqledtr.analysis import uniform_prior_mean
from partiqledtr.models.qfm import QFMConstellation
from partiqledtr.train import _PURITY_SEED, _split_arrays

#: Repo root, resolved from this module rather than the working directory, so a
#: driver behaves the same wherever it is started from.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"


def observables_of(
    config: dict[str, Any], split: dict[str, np.ndarray], take: np.ndarray
) -> tuple[float, dict[str, list[float]]]:
    """The encoded-state g-purity and angle statistics of one configuration."""
    module = QFMConstellation(
        4,
        int(config["n_classes"]),
        ansatz=config["ansatz"],
        n_layers=int(config["n_layers"]),
        angle_map=config["angle_map"],
        enc_weights=config.get("enc_weights", "hamming"),
        enc_reupload=config.get("enc_reupload", "diagonal"),
        rngs=nnx.Rngs(0),
    )
    features, mask, _ = _split_arrays(split, config["encoding"])
    x, m = jnp.asarray(features[take]), jnp.asarray(mask[take])
    return float(module.g_purity(x, m)), module.angle_stats(x, m)


def main() -> None:
    """Patch every arm file with a corrected initial purity, and say what is missing."""
    with np.load(DATA / "val.npz") as data:
        split = {key: data[key] for key in data.files}
    n_val = len(split["n_fsps"])
    take = np.random.default_rng(_PURITY_SEED).choice(n_val, size=min(64, n_val), replace=False)
    topologies = len(set(np.asarray(split["topology_id"])[take].tolist()))
    print(f"subset: {len(take)} of {n_val} validation events, spanning {topologies} topologies\n")

    cache: dict[tuple, tuple[float, dict[str, list[float]]]] = {}
    stale: list[str] = []
    for path in sorted(OUT.glob("arm_*.json")):
        records = json.loads(path.read_text())
        patched = 0
        for record in records:
            config = (record.get("final_metrics") or {}).get("config")
            if not config or config.get("model") != "qfm":
                continue
            axes = ("encoding", "angle_map", "ansatz", "n_layers", "enc_weights", "enc_reupload")
            key = tuple(config.get(name) for name in axes)
            if key not in cache:
                cache[key] = observables_of(config, split, take)
            purity, angles = cache[key]
            record["final_metrics"]["g_purity_initial_repaired"] = purity
            record["final_metrics"]["angle_stats_initial_repaired"] = angles
            record["final_metrics"]["purity_mu"] = uniform_prior_mean(config["ansatz"], 4)
            if config.get("frontend") == "none":
                # No front end trains, so the encoded angles never move.
                record["final_metrics"]["val_g_purity_repaired"] = purity
                record["final_metrics"]["angle_stats_final_repaired"] = angles
            else:
                stale.append(f"{path.stem}: {config.get('frontend')} {key}")
            patched += 1
        path.write_text(json.dumps(records, indent=1))
        print(f"{path.name}: patched {patched} records")

    print(f"\n{len(cache)} distinct encoder configurations:")
    for key, (value, _) in sorted(cache.items(), key=lambda kv: kv[1][0]):
        mu = uniform_prior_mean(key[2], 4)
        print(f"  {value:7.4f} ({value / mu:.2f} mu)  {key}")
    if stale:
        print(f"\nfinal purity needs a re-fit for {len(stale)} records (front end trains)")


if __name__ == "__main__":
    main()
