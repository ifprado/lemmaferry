#!/usr/bin/env python3
"""Fit T and S to explicitly paired formal/language activation collections.

Unlike fit_activations.py's same-input baseline, these inputs may use different
texts, tokenizers, token counts and dimensions. Their exact input bytes, row
identities and library snapshots must match a cross-view correspondence manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lemmaferry.correspondence import load_correspondence  # noqa: E402
from lemmaferry.rosetta import OBJECTIVES, PairedSplit  # noqa: E402
from lemmaferry.translators import fit_translator  # noqa: E402
from lemmaferry.translator_cli import add_translator_arguments, swap_options  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("_lemmaferry_collections", Path(__file__).with_name("fit_activations.py"))
_COLLECTOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COLLECTOR)


def paired_domains(source: Path, target: Path, correspondence: Path) -> tuple[dict, dict, dict, dict, dict]:
    manifest = load_correspondence(correspondence)
    sm, sx = _COLLECTOR.load_collection(source)
    tm, ty = _COLLECTOR.load_collection(target)
    for split in ("train", "val", "test"):
        info = manifest["splits"][split]
        for label, model_manifest, rows in (("math", sm, sx), ("language", tm, ty)):
            if model_manifest["inputs"][split]["sha256"] != info[label]["sha256"]:
                raise ValueError(f"{split}: {label} collector input differs from bound correspondence view")
            ids = [row["id"] for row in info["pairs"]]
            families = [row["family"] for row in info["pairs"]]
            if not np.array_equal(rows[split]["ids"], ids):
                raise ValueError(f"{split}: {label} IDs differ or rows were reordered")
            if not np.array_equal(rows[split]["families"], families):
                raise ValueError(f"{split}: {label} families differ from correspondence")
    return sm, sx, tm, ty, manifest


def fit_collections(
    source: Path, target: Path, correspondence: Path, output: Path,
    lambdas=(0.0, 0.001, 0.1, 1.0), *, objective="raw_sum", seed=1729,
    translator="affine", swap_config=None,
) -> dict:
    source, target, correspondence, output = map(Path, (source, target, correspondence, output))
    if output.exists():
        raise ValueError("output already exists; preserve frozen results")
    sm, sx, tm, ty, manifest = paired_domains(source, target, correspondence)
    pairs = {split: PairedSplit(sx[split]["vectors"], ty[split]["vectors"],
                               sx[split]["ids"].tolist(), sx[split]["families"].tolist(), split,
                               source_ids=sx[split]["ids"].tolist(), target_ids=ty[split]["ids"].tolist())
             for split in ("train", "val", "test")}
    pair = fit_translator(pairs["train"], pairs["val"], translator=translator,
                          lambdas=lambdas, objective=objective, seed=seed, swap_config=swap_config)
    manifest_path = correspondence / "manifest.json" if correspondence.is_dir() else correspondence
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    pair.protocol["extraction"] = {
        "math": sm, "language": tm, "correspondence_manifest_sha256": sha(manifest_path),
        "library_snapshots_sha256": manifest["library_snapshots_sha256"],
        "math_manifest_sha256": sha(source / "manifest.json"),
        "language_manifest_sha256": sha(target / "manifest.json"),
        "pairing_policy": "Exact bound formal/natural input-file hashes and ordered IDs/families; tokenizer and token hashes may differ between domains.",
    }
    report = pair.evaluate(pairs["test"])
    saved = pair.save(output)
    report["bundle"] = {"manifest": "manifest.json", "manifest_content_sha256": saved["manifest_content_sha256"],
                        "archive_sha256": saved["archive_sha256"]}
    path = output / "results.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "results.json.sha256").write_text(f"{sha(path)}  results.json\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--math", type=Path, required=True, help="Formal-view collector output")
    parser.add_argument("--target", "--language", type=Path, required=True, help="Language-view collector output")
    parser.add_argument("--correspondence", type=Path, required=True, help="Bound cross-view manifest or its directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.001, 0.1, 1.0])
    parser.add_argument("--objective", choices=tuple(OBJECTIVES), default="raw_sum")
    parser.add_argument("--seed", type=int, default=1729)
    add_translator_arguments(parser)
    args = parser.parse_args()
    report = fit_collections(args.source, args.target, args.correspondence, args.output,
                             args.lambdas, objective=args.objective, seed=args.seed,
                             translator=args.translator,
                             swap_config=swap_options(args) if args.translator == "swap" else None)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "selection": report["protocol"]["selection"], "scope": report["protocol"]["scope"]}, indent=2))


if __name__ == "__main__":
    main()
