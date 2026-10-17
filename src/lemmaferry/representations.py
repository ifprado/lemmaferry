"""Per-iteration formal/language extraction and supervised Rosetta calibration.

Calibration uses accepted library examples only. Its held-out structural-family
test is a geometry diagnostic; it is separate from the frozen capability task P.
The source may be a pinned specialist or an explicitly labelled untrained fixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .correspondence import build_correspondence, write_correspondence
from .rosetta import PairedSplit
from .translators import fit_translator


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _formula(value: Any, prefix: str) -> str:
    if type(value) is int and value >= 0:
        return f"{prefix}{value}"
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("unrecognized constructive formula in correspondence")
    op, children = next(iter(value.items()))
    if op not in ("imp", "and") or not isinstance(children, list) or len(children) != 2:
        raise ValueError("unrecognized constructive formula operation")
    left, right = (_formula(child, prefix) for child in children)
    return f"({left} {'implies' if op == 'imp' else 'and'} {right})"


def _proof_description(proof: Any, record: Mapping[str, Any], known: Mapping[str, Mapping[str, Any]], depth=0) -> str:
    if depth > 900 or not isinstance(proof, dict) or len(proof) != 1:
        raise ValueError("invalid or excessively nested constructive proof for rendering")
    op, value = next(iter(proof.items()))
    prefix = record.get("proposal", {}).get("label_prefix", "T")
    recur = lambda node: _proof_description(node, record, known, depth + 1)
    if op == "hyp":
        premises = record["payload"]["premises"]
        if type(value) is not int or not 0 <= value < len(premises):
            raise ValueError("unknown proof premise in correspondence")
        return f"premise {value} ({_formula(premises[value], prefix)})"
    if op in ("apply", "and_intro"):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("proof operation needs two arguments")
        left, right = map(recur, value)
        return f"apply [{left}] to [{right}]" if op == "apply" else f"form the conjunction of [{left}] and [{right}]"
    if op in ("and_left", "and_right"):
        return f"take the {'left' if op == 'and_left' else 'right'} conjunct of [{recur(value)}]"
    if op == "lemma":
        if not isinstance(value, dict) or set(value) != {"id", "atoms", "arguments"} or value["id"] not in known:
            raise ValueError("unavailable or malformed earlier lemma in correspondence")
        atoms, arguments = value["atoms"], value["arguments"]
        dependency = known[value["id"]]
        if not isinstance(atoms, list) or len(atoms) != dependency["payload"]["atom_count"]:
            raise ValueError("dependency atom substitution differs")
        if not isinstance(arguments, list) or len(arguments) != len(dependency["payload"]["premises"]):
            raise ValueError("dependency premise arguments differ")
        substitution = "; ".join(f"atom {i} denotes {_formula(atom, prefix)}" for i, atom in enumerate(atoms))
        arguments_text = "; ".join(recur(arg) for arg in arguments)
        return f"use earlier theorem {value['id']} with [{substitution}] and premise arguments [{arguments_text}]"
    raise ValueError(f"unrecognized constructive proof operation: {op}")


def _language_view(record: Mapping[str, Any], known: Mapping[str, Mapping[str, Any]]) -> str:
    completed: set[str] = set()
    active: set[str] = set()
    parts = []

    def visit(current: Mapping[str, Any]) -> None:
        identifier = current["id"]
        if identifier in completed:
            return
        if identifier in active:
            raise ValueError("cyclic theorem dependencies in correspondence")
        active.add(identifier)
        for dependency in current.get("dependencies", []):
            if dependency not in known:
                raise ValueError("correspondence needs the entire verified dependency closure")
            visit(known[dependency])
        proof = current["payload"]["proof"]
        description = _proof_description(proof, current, known)
        parts.append(f"Theorem {identifier}: {current['natural_language']}\nConstructive derivation: {description}.")
        active.remove(identifier)
        completed.add(identifier)

    visit(record)
    return "\n\n".join(parts)


def _model_metadata(model) -> dict[str, Any]:
    config = dict(model.config)
    backend = model.backend
    if backend == "huggingface_residual_adapter":
        for key in ("revision", "tokenizer_revision"):
            if not isinstance(config.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", config[key]):
                raise ValueError("pretrained representation models require immutable model/tokenizer revisions")
        if not isinstance(config.get("model_id"), str) or not config["model_id"]:
            raise ValueError("pretrained representation model identifier missing")
        role = "pinned pretrained model; specialist capability is not assumed from its identifier"
    else:
        role = "untrained or deterministic pipeline fixture; not a mathematical specialist or general-purpose transfer result"
    config_hash = _sha(_canonical(config))
    return {
        "backend": backend, "config": config, "config_sha256": config_hash,
        "hidden_size": model.hidden_size, "boundary": model.representation_boundary,
        "position": "last prompt token; no answer tokens", "role": role,
    }


def _state_hash(model) -> str | None:
    if callable(getattr(model, "parameter_sha256", None)):
        return model.parameter_sha256()
    if hasattr(model, "module"):
        from .learning import parameter_sha256
        return parameter_sha256(model)
    return None


def fit_library_rosetta(
    records: Iterable[Mapping[str, Any]], snapshot_hash: str, math_model, language_model,
    output: str | Path, lambdas=(0.0, 0.001, 0.1, 1.0), seed: int = 1729, *,
    extractor: Callable | None = None, translator: str = "affine",
    objective: str = "raw_sum", swap_config=None,
) -> dict[str, Any]:
    """Extract before learning, fit T/S, and supply per-record auxiliary targets.

    ``records`` must come from the caller's validated accepted-library lookup;
    a JSON status flag is not a substitute for that lookup. External evaluation
    rows are rejected and never used for fitting or selecting regularization.
    The returned pair is a Python object; report and target mappings are JSON-safe.
    """
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; preserve the per-iteration evidence")
    rows = list(records)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("nonempty accepted records with unique IDs required")
    if not isinstance(snapshot_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot_hash):
        raise ValueError("immutable library snapshot SHA256 required")
    known = {row["id"]: row for row in rows}
    for row in rows:
        learning_example = row.get("learning_example", {})
        for item in (row, learning_example):
            provenance = item.get("provenance", {})
            if (item.get("split", "train") != "train" or item.get("evaluation_only", False)
                    or provenance.get("evaluation_only", False)
                    or provenance.get("use") == "independent_evaluation_only"):
                raise ValueError("evaluation split records cannot enter Rosetta calibration")
        for field in ("id", "family", "formal_text", "natural_language"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"accepted record lacks {field}")
    families = sorted({row["family"] for row in rows})
    if len(families) < 3:
        raise ValueError("Rosetta calibration requires at least three structural families")
    family_splits = {family: ("train", "val", "test")[i % 3] for i, family in enumerate(families)}
    views = {split: [] for split in ("train", "val", "test")}
    all_views = []
    for row in rows:
        correspondence = build_correspondence(
            row["id"], row["family"], row["formal_text"], _language_view(row, known),
            {"library_snapshot_sha256": snapshot_hash, "record_sha256": _sha(_canonical(row)),
             "record_provenance": dict(row.get("provenance", {})),
             "proof_renderer": "constructive_proof_DSL_v1; dependency closure included; separate semantic audit required"},
        )
        views[family_splits[row["family"]]].append(correspondence)
        all_views.append(correspondence)
    if len(views["train"]) < 2:
        raise ValueError("Rosetta calibration requires at least two training records")
    metadata = {"math": _model_metadata(math_model), "language": _model_metadata(language_model)}
    before = {"math": _state_hash(math_model), "language": _state_hash(language_model)}
    if extractor is None:
        from .learning import represent
        extractor = represent
    x = np.asarray(extractor(math_model, [row["formal_text"] for row in all_views]), dtype=np.float64)
    y = np.asarray(extractor(language_model, [row["language_text"] for row in all_views]), dtype=np.float64)
    for name, array, model in (("math", x, math_model), ("language", y, language_model)):
        if array.shape != (len(rows), model.hidden_size) or not np.isfinite(array).all():
            raise ValueError(f"{name} extractor returned a nonfinite or incorrectly shaped matrix")
    after = {"math": _state_hash(math_model), "language": _state_hash(language_model)}
    if before != after:
        raise ValueError("representation extraction changed model parameters")
    indices = {split: [i for i, row in enumerate(rows) if family_splits[row["family"]] == split]
               for split in ("train", "val", "test")}
    pairs = {split: PairedSplit(x[idx], y[idx], [rows[i]["id"] for i in idx],
                               [rows[i]["family"] for i in idx], split)
             for split, idx in indices.items()}
    pair = fit_translator(pairs["train"], pairs["val"], translator=translator,
                          lambdas=lambdas, objective=objective, seed=seed, swap_config=swap_config)
    pair.protocol["extraction"] = {
        "library_snapshot_sha256": snapshot_hash, "models": metadata,
        "trainable_parameter_sha256_before": before, "trainable_parameter_sha256_after": after,
        "parameters_unchanged": True if all(before.values()) else "hash unavailable for a fixture",
        "family_split": family_splits, "timing": "before the current recipient training update",
        "gradient_boundary": "detached NumPy activation snapshots; no donor or recipient model tensors enter translator training",
        "calibration_scope": "Library-only structural-family geometric holdout; these records may subsequently be used for training. No frozen capability evaluation task P is used.",
    }
    output.mkdir(parents=True, exist_ok=False)
    correspondence_manifest = write_correspondence(views, output / "views")
    pair.protocol["extraction"]["correspondence_manifest_content_sha256"] = correspondence_manifest["manifest_content_sha256"]
    np.savez_compressed(output / "activations.npz", math=x, language=y,
                        ids=np.asarray([row["id"] for row in rows]),
                        families=np.asarray([row["family"] for row in rows]))
    saved = pair.save(output / "maps")
    report = pair.evaluate(pairs["test"])
    report["artifacts"] = {
        "activations": "activations.npz", "activations_sha256": _sha((output / "activations.npz").read_bytes()),
        "map_manifest": "maps/manifest.json", "map_manifest_content_sha256": saved["manifest_content_sha256"],
        "correspondence_manifest": "views/manifest.json",
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    translated = pair.to_language(x)
    shuffled = pair.to_language(x, control="shuffled")
    mean = pair.to_language(x, control="mean")
    return {
        "pair": pair, "report": report,
        "translated_targets": {row["id"]: value.tolist() for row, value in zip(rows, translated)},
        "language_targets": {row["id"]: value.tolist() for row, value in zip(rows, y)},
        "shuffled_targets": {row["id"]: value.tolist() for row, value in zip(rows, shuffled)},
        "mean_targets": {row["id"]: value.tolist() for row, value in zip(rows, mean)},
    }
