#!/usr/bin/env python3
"""Bounded swap/affine comparison on synthetic or audited saved activations.

This measures geometry only. It never loads the specialist or recipient models.
Synthetic generator adapted from the user's Apache-2.0 swap prototype.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lemmaferry.rosetta import PairedSplit  # noqa: E402
from lemmaferry.translators import fit_translator, load_translator  # noqa: E402
from lemmaferry.translator_cli import add_translator_arguments, swap_options  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def synthetic_pairs(seed=2026, sizes=(512, 96, 128)) -> dict[str, PairedSplit]:
    """Favorable nonlinear views of four shared factors, dimensions 12 and 10.

    Held-out rows are new samples from the same planted distribution. The family
    labels here identify sample partitions, not independent theorem structures.
    """
    rng = np.random.default_rng(seed)
    qa = np.linalg.qr(rng.normal(size=(12, 7)))[0].T
    qb = np.linalg.qr(rng.normal(size=(10, 7)))[0].T
    offset_a, offset_b = rng.normal(size=12), rng.normal(size=10)
    pairs = {}
    for split, count in zip(("train", "val", "test"), sizes):
        z = rng.uniform(-1.5, 1.5, size=(count, 4))
        a = np.column_stack([z, z[:, 0] * z[:, 1], z[:, 2] ** 2, np.sin(z[:, 3])])
        b = np.column_stack([z, np.sin(1.3 * z[:, 0]), z[:, 1] * z[:, 2], np.tanh(z[:, 3])])
        pairs[split] = PairedSplit(
            (a @ qa + offset_a).astype(np.float32), (b @ qb + offset_b).astype(np.float32),
            [f"toy_{split}_{i:05d}" for i in range(count)],
            [f"synthetic_sample_partition_{split}"] * count, split,
        )
    return pairs


def audited_pairs(source: Path, target: Path, correspondence: Path | None = None):
    name = "fit_rosetta" if correspondence is not None else "fit_activations"
    spec = importlib.util.spec_from_file_location("_comparison_loader", Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if correspondence is None:
        sm, sx, tm, ty = module.paired_collections(source, target)
        provenance = {"pairing_policy": "audited legacy identical-input collections"}
    else:
        sm, sx, tm, ty, manifest = module.paired_domains(source, target, correspondence)
        file = correspondence / "manifest.json" if correspondence.is_dir() else correspondence
        provenance = {"pairing_policy": "audited distinct formal/language views",
                      "correspondence_manifest_sha256": digest(file),
                      "library_snapshots_sha256": manifest["library_snapshots_sha256"]}
    provenance.update(math=sm, language=tm,
                      math_manifest_sha256=digest(source / "manifest.json"),
                      language_manifest_sha256=digest(target / "manifest.json"))
    pairs = {split: PairedSplit(sx[split]["vectors"], ty[split]["vectors"],
                               sx[split]["ids"].tolist(), sx[split]["families"].tolist(), split,
                               source_ids=sx[split]["ids"].tolist(), target_ids=ty[split]["ids"].tolist())
             for split in ("train", "val", "test")}
    return pairs, provenance


def run_comparison(pairs, output: Path, *, seeds, swap_config, provenance, synthetic=False):
    import torch
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("comparison requires distinct nonnegative integer seeds")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "lemmaferry.swap.comparison.v1", "seeds": list(seeds),
        "swap_config": swap_config, "affine_objective": "raw_sum",
        "affine_lambdas": [0.0, 0.001, 0.1, 1.0],
        "selection": "Fixed budget; validation-only bidirectional checkpoint selection. Test never changes settings.",
        "acceptance": ("Paired test nRMSE below shuffled and mean in both directions; reload exact."
                       if synthetic else "Audits, finite metrics, matched neural budgets, exact reload; no improvement required."),
        "splits": {s: p.fingerprint() for s, p in pairs.items()},
        "provenance": provenance,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(
            list((ROOT / "src/lemmaferry").rglob("*.py")) + [Path(__file__).resolve(),
                ROOT / "experiments/fit_activations.py", ROOT / "experiments/fit_rosetta.py"])},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__,
                        "torch_threads": torch.get_num_threads(), "device": "cpu"},
        "scope": "Saved-vector geometric alignment; no model collection, recipient training, inference assistance or capability evidence.",
        "compute_comparison": "Paired and shuffled neural arms are matched. Affine is descriptive and not compute matched.",
    }
    write(output / "protocol.json", protocol)  # before any fitting
    if synthetic:
        np.savez_compressed(output / "synthetic-pairs.npz", **{
            f"{side}_{split}": value for split, pair in pairs.items()
            for side, value in (("math", pair.x), ("language", pair.y),
                                ("ids", np.asarray(pair.ids)), ("families", np.asarray(pair.families)))
        })
    runs = []
    for seed in seeds:
        for backend in ("swap", "affine"):
            started = time.perf_counter()
            pair = fit_translator(pairs["train"], pairs["val"], translator=backend,
                                  lambdas=protocol["affine_lambdas"], seed=seed,
                                  swap_config=swap_config if backend == "swap" else None)
            pair.protocol["extraction"] = provenance
            seconds = time.perf_counter() - started
            bundle = output / f"seed-{seed}-{backend}"
            pair.save(bundle)
            restored = load_translator(bundle)
            for method, values in (("to_language", pairs["test"].x), ("to_math", pairs["test"].y)):
                for control in ("paired", "shuffled", "mean"):
                    np.testing.assert_array_equal(getattr(pair, method)(values, control=control),
                                                  getattr(restored, method)(values, control=control))
            report = restored.evaluate(pairs["test"])
            report.update(seed=seed, backend=backend, fitting_wall_seconds=seconds,
                          reload_predictions_exact=True)
            write(bundle / "results.json", report)
            compact = {key: report[key] for key in (
                "seed", "backend", "fitting_wall_seconds", "reload_predictions_exact", "test_metrics", "round_trip"
            )}
            compact["bundle_manifest_sha256"] = digest(bundle / "manifest.json")
            compact["training_protocol"] = {key: pair.protocol[key] for key in (
                "config", "selection", "controls", "training_budget", "training"
            ) if key in pair.protocol}
            for key in ("self_reconstruction", "empty_routing"):
                if key in report:
                    compact[key] = report[key]
            runs.append(compact)
            print(json.dumps({"seed": seed, "backend": backend, "wall_seconds": seconds,
                              "test_metrics": report["test_metrics"]}), flush=True)
    summary = {"status": "completed", "protocol_sha256": digest(output / "protocol.json"),
               "scope": protocol["scope"], "compute_comparison": protocol["compute_comparison"],
               "counts": {s: len(p.ids) for s, p in pairs.items()}, "runs": runs}
    if synthetic:
        summary["synthetic_acceptance_passed"] = all(
            metrics["paired"]["normalized_rmse"] < min(metrics[c]["normalized_rmse"]
                 for c in ("shuffled_training_pairs", "training_mean"))
            for run in runs if run["backend"] == "swap" for metrics in run["test_metrics"].values())
    write(output / "summary.json", summary)
    write(output / "artifact-sha256.json", {str(p.relative_to(output)): digest(p)
          for p in sorted(output.rglob("*")) if p.is_file()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--correspondence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--torch-threads", type=int, default=1)
    add_translator_arguments(parser)
    parser.set_defaults(translator="swap")
    args = parser.parse_args()
    if args.translator != "swap" or args.torch_threads < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("comparison requires --translator swap, positive thread count and distinct seeds")
    if args.synthetic and any((args.source, args.target, args.correspondence)):
        parser.error("synthetic mode does not take collection inputs")
    if not args.synthetic and (args.source is None or args.target is None):
        parser.error("supply --synthetic or both --source and --target")
    import torch
    torch.set_num_threads(args.torch_threads)
    if args.synthetic:
        pairs = synthetic_pairs()
        provenance = {"dataset": "planted nonlinear four-factor views", "data_seed": 2026,
                      "limitation": "New samples from the same planted distribution, not held-out theorem structures.",
                      "models": None, "lean": "not used"}
    else:
        pairs, provenance = audited_pairs(args.source, args.target, args.correspondence)
    run_comparison(pairs, args.output, seeds=args.seeds, swap_config=swap_options(args),
                   provenance=provenance, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
