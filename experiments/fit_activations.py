#!/usr/bin/env python3
"""Fit an affine map to completed activation collections, without loading LLMs.

Validation selects ridge strength; the selected training-only map is evaluated
on test once. Geometric alignment is not a capability-transfer experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lemmaferry.alignment import AffineRidge, regression_metrics  # noqa: E402

SPLITS = ("train", "val", "test")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def load_collection(directory: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Validate collector metadata, archive integrity, and split boundaries."""
    path = directory / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read collector manifest: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ValueError(f"{path}: collector status must be completed")
    if not isinstance(manifest.get("model"), str) or not manifest["model"]:
        raise ValueError(f"{path}: model identifier missing")
    if not isinstance(manifest.get("revision"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["revision"]
    ):
        raise ValueError(f"{path}: immutable model revision missing")
    if type(manifest.get("layer_index")) is not int or manifest["layer_index"] < 0:
        raise ValueError(f"{path}: nonnegative layer_index required")
    if manifest.get("position") != "last_nonpadding_token":
        raise ValueError(f"{path}: expected last_nonpadding_token selection")
    if manifest.get("input_format") != "literal_text_no_chat_template_no_special_tokens":
        raise ValueError(f"{path}: unsupported input format")
    if manifest.get("pooling") != "none":
        raise ValueError(f"{path}: expected unpooled token activations")
    _hash(manifest.get("tokenizer_sha256"), f"{path}: tokenizer_sha256")
    if not isinstance(manifest.get("inputs"), dict) or not isinstance(manifest.get("outputs"), dict):
        raise ValueError(f"{path}: input and output metadata required")
    seen_ids: set[str] = set()
    family_split: dict[str, str] = {}
    dimension = None
    splits = {}
    for split in SPLITS:
        info = manifest["inputs"].get(split)
        output_info = manifest["outputs"].get(split)
        if not isinstance(info, dict) or not isinstance(output_info, dict):
            raise ValueError(f"{path}: missing {split} metadata")
        _hash(info.get("sha256"), f"{path}: {split} input hash")
        file = directory / f"{split}.npz"
        if output_info.get("path") != file.name:
            raise ValueError(f"{path}: unexpected {split} output path")
        expected_hash = _hash(output_info.get("sha256"), f"{path}: {split} output hash")
        if not file.is_file() or digest(file) != expected_hash:
            raise ValueError(f"{file}: missing archive or output checksum mismatch")
        try:
            with np.load(file, allow_pickle=False) as archive:
                rows = {key: archive[key] for key in (
                    "vectors", "ids", "families", "token_sha256", "lengths"
                )}
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError(f"{file}: invalid collector archive") from exc
        vectors = rows["vectors"]
        if vectors.ndim != 2 or min(vectors.shape) == 0 or vectors.dtype.kind not in "fi":
            raise ValueError(f"{file}: vectors must be a nonempty numeric matrix")
        if not np.isfinite(vectors).all():
            raise ValueError(f"{file}: vectors contain nonfinite values")
        count, width = vectors.shape
        if info.get("count") != count or output_info.get("shape") != [count, width]:
            raise ValueError(f"{file}: metadata count or shape mismatch")
        if dimension is not None and width != dimension:
            raise ValueError(f"{file}: dimensions changed across splits")
        dimension = width
        for key in ("ids", "families", "token_sha256"):
            values = rows[key]
            if values.shape != (count,) or values.dtype.kind not in "US":
                raise ValueError(f"{file}: {key} must be one string per vector")
            if not all(str(value).strip() for value in values):
                raise ValueError(f"{file}: empty {key}")
        if rows["lengths"].shape != (count,) or rows["lengths"].dtype.kind not in "iu":
            raise ValueError(f"{file}: lengths must be one integer per vector")
        if np.any(rows["lengths"] < 1):
            raise ValueError(f"{file}: token lengths must be positive")
        for token_hash in rows["token_sha256"]:
            _hash(str(token_hash), f"{file}: token hash")
        for identifier, family in zip(rows["ids"].tolist(), rows["families"].tolist()):
            if identifier in seen_ids:
                raise ValueError(f"{file}: repeated ID {identifier!r}")
            seen_ids.add(identifier)
            if family in family_split and family_split[family] != split:
                raise ValueError(f"{file}: family {family!r} crosses splits")
            family_split[family] = split
        splits[split] = rows
    if len(splits["train"]["vectors"]) < 2:
        raise ValueError("At least two training examples required")
    return manifest, splits


def paired_collections(source: Path, target: Path) -> tuple[dict, dict, dict, dict]:
    sm, sx = load_collection(source)
    tm, ty = load_collection(target)
    if sm["tokenizer_sha256"] != tm["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprints differ; same-input baseline requires equality")
    for split in SPLITS:
        if sm["inputs"][split]["sha256"] != tm["inputs"][split]["sha256"]:
            raise ValueError(f"{split}: original text input hashes differ")
        for key in ("ids", "families", "token_sha256", "lengths"):
            if not np.array_equal(sx[split][key], ty[split][key]):
                raise ValueError(f"{split}: paired {key} differ or rows are reordered")
    return sm, sx, tm, ty


def _evaluate(prediction: np.ndarray, actual: np.ndarray, mean: np.ndarray, scaler) -> dict:
    metrics = regression_metrics(prediction, actual, scaler)
    mse = float(np.mean((prediction - actual) ** 2))
    baseline_mse = float(np.mean((actual - mean) ** 2))

    def unit_rows(values):
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.where(norm > 0, norm, 1)

    predicted_units, actual_units = unit_rows(prediction), unit_rows(actual)
    scores = predicted_units @ actual_units.T
    metrics.update({
        "mse": mse,
        "mse_divided_by_training_mean_baseline_mse": mse / baseline_mse if baseline_mse else None,
        "mean_matched_cosine_similarity": float(np.mean(np.sum(predicted_units * actual_units, axis=1))),
        "cosine_retrieval_top1_accuracy": float(np.mean(np.argmax(scores, axis=1) == np.arange(len(actual)))),
        "retrieval_candidate_count": len(actual),
        "retrieval_chance_accuracy": 1 / len(actual),
    })
    return metrics


def fit_collections(
    source: Path, target: Path, output: Path,
    alphas: list[float] | None = None, seed: int = 1729,
) -> dict:
    """Select on validation; do not refit on validation or tune against test."""
    if output.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    grid = sorted(set([0.0001, 0.001, 0.01, 0.1, 1.0] if alphas is None else alphas))
    if not grid or any(not np.isfinite(a) or a < 0 for a in grid):
        raise ValueError("alphas must be a nonempty list of finite nonnegative values")
    sm, sx, tm, ty = paired_collections(source, target)
    train_x, train_y = sx["train"]["vectors"], ty["train"]["vectors"]
    validation = []
    best_model, best_score, selected_alpha = None, float("inf"), None
    for alpha in grid:
        model = AffineRidge(alpha=alpha).fit(train_x, train_y)
        score = regression_metrics(model.predict(sx["val"]["vectors"]),
                                   ty["val"]["vectors"], model.target_scaler_)
        validation.append({"alpha": alpha, **score})
        if score["normalized_rmse"] < best_score:
            best_model, best_score, selected_alpha = model, score["normalized_rmse"], alpha
    assert best_model is not None
    permutation = np.random.default_rng(seed).permutation(len(train_y))
    shuffled = AffineRidge(alpha=selected_alpha).fit(train_x, train_y[permutation])
    test_x, test_y = sx["test"]["vectors"], ty["test"]["vectors"]
    training_mean = train_y.mean(axis=0)
    predictions = {
        "paired_affine": best_model.predict(test_x),
        "shuffled_training_pairs": shuffled.predict(test_x),
        "training_mean": np.broadcast_to(training_mean, test_y.shape),
    }
    if test_x.shape[1] == test_y.shape[1]:
        predictions["raw_identity"] = test_x
    test_metrics = {name: _evaluate(pred, test_y, training_mean, best_model.target_scaler_)
                    for name, pred in predictions.items()}
    families = ty["test"]["families"]
    by_family = {}
    for family in sorted(set(families.tolist())):
        mask = families == family
        by_family[family] = {
            name: _evaluate(pred[mask], test_y[mask], training_mean, best_model.target_scaler_)
            for name, pred in predictions.items()
        }
    report = {
        "status": "completed",
        "experiment": "same_input_frozen_activation_affine_alignment",
        "scope": "Geometric alignment only. No activation intervention, capability transfer, or LLM training performed.",
        "seed": seed,
        "selected_alpha": selected_alpha,
        "selection": "Lowest validation normalized RMSE; ties favor smaller alpha. No train-plus-validation refit.",
        "validation_scores": validation,
        "normalization": "Per-coordinate target standard deviation fitted on training rows only; constant columns use 1.",
        "test_metrics": test_metrics,
        "test_metrics_by_family": by_family,
        "retrieval": "Raw cosine retrieval within each reported test candidate pool; ties select first row.",
        "counts": {split: len(sx[split]["vectors"]) for split in SPLITS},
        "source_dim": train_x.shape[1], "target_dim": train_y.shape[1],
        "source": {"directory": str(source), "manifest_sha256": digest(source / "manifest.json"), "manifest": sm},
        "target": {"directory": str(target), "manifest_sha256": digest(target / "manifest.json"), "manifest": tm},
        "map": {"path": "translator.npz", "formula": "target_row = source_row @ coef.T + intercept"},
    }
    output.mkdir(parents=True, exist_ok=False)
    map_path = output / "translator.npz"
    np.savez_compressed(
        map_path, coef=best_model.coef_, intercept=best_model.intercept_,
        source_mean=best_model.source_scaler_.mean, source_scale=best_model.source_scaler_.scale,
        target_mean=best_model.target_scaler_.mean, target_scale=best_model.target_scaler_.scale,
        alpha=np.asarray(selected_alpha),
    )
    report["map"]["sha256"] = digest(map_path)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory for map and results")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0001, 0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    try:
        report = fit_collections(args.source, args.target, args.output, args.alphas, args.seed)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Selected alpha={report['selected_alpha']}; wrote {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
