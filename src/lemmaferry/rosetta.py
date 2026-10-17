"""Supervised, bidirectional affine correspondence with held-out evaluation.

Round trips are geometric diagnostics. Neither an inverse nor a low regression
error establishes semantic equivalence or useful capability transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .alignment import AffineRidge, Standardizer, _matrix, regression_metrics

Array = NDArray[np.float64]
OBJECTIVES = {
    "raw_sum": "sum_i ||W x_i + b - y_i||^2 + lambda ||W||_F^2; b unpenalized",
    "standardized_mean": (
        "mean_i ||B x_standardized_i - y_standardized_i||^2 + lambda ||B||_F^2; "
        "train-only centering/scaling, then converted to raw coordinates"
    ),
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha(value: ArrayLike) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    return _sha(_json_bytes(list(array.shape)) + array.tobytes())


def _labels(values: Iterable[str], count: int, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != count or any(not isinstance(v, str) or not v.strip() for v in result):
        raise ValueError(f"{name} must contain one nonempty string per row")
    return result


@dataclass(frozen=True)
class PairedSplit:
    """Paired domains with explicit identity and structural split membership.

    When independently collected matrices are joined, pass source_ids and
    target_ids to reject accidental row reordering. Callers remain responsible
    for the semantic correspondence of the two views.
    """

    x: ArrayLike
    y: ArrayLike
    ids: Iterable[str]
    families: Iterable[str]
    split: str
    source_ids: Iterable[str] | None = None
    target_ids: Iterable[str] | None = None

    def __post_init__(self) -> None:
        x, y = _matrix(self.x, "x").copy(), _matrix(self.y, "y").copy()
        if x.shape[0] != y.shape[0]:
            raise ValueError("one mathematical and language vector required per pair")
        ids = _labels(self.ids, len(x), "ids")
        families = _labels(self.families, len(x), "families")
        if len(set(ids)) != len(ids):
            raise ValueError("repeated paired IDs")
        if self.split not in ("train", "val", "validation", "test"):
            raise ValueError("split must be train, val/validation, or test")
        for name in ("source_ids", "target_ids"):
            values = getattr(self, name)
            if values is not None and _labels(values, len(x), name) != ids:
                raise ValueError(f"{name} differ from pair IDs or rows were reordered")
            object.__setattr__(self, name, ids)
        x.flags.writeable = False
        y.flags.writeable = False
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "families", families)
        object.__setattr__(self, "split", "val" if self.split == "validation" else self.split)

    def fingerprint(self) -> dict[str, Any]:
        return {
            "split": self.split, "count": len(self.ids),
            "math_dimension": self.x.shape[1], "language_dimension": self.y.shape[1],
            "ids": list(self.ids), "families": list(self.families),
            "math_vectors_sha256": _array_sha(self.x),
            "language_vectors_sha256": _array_sha(self.y),
        }


@dataclass(frozen=True)
class AffineMap:
    """Raw-coordinate map; rows transform as x @ coef.T + intercept."""

    coef: ArrayLike
    intercept: ArrayLike

    def __post_init__(self) -> None:
        coef = _matrix(self.coef, "coef").copy()
        intercept = np.asarray(self.intercept, dtype=np.float64).copy()
        if intercept.shape != (coef.shape[0],) or not np.isfinite(intercept).all():
            raise ValueError("intercept must have one finite value per output coordinate")
        coef.flags.writeable = False
        intercept.flags.writeable = False
        object.__setattr__(self, "coef", coef)
        object.__setattr__(self, "intercept", intercept)

    def predict(self, values: ArrayLike) -> Array:
        values = _matrix(values, "values")
        if values.shape[1] != self.coef.shape[1]:
            raise ValueError("input dimension differs from affine map")
        return values @ self.coef.T + self.intercept


def fit_affine(
    source: ArrayLike, target: ArrayLike, regularization: float = 0.0,
    objective: str = "raw_sum",
) -> AffineMap:
    """Fit either the stated raw sum-of-squares objective or the old baseline.

    Centering eliminates the unpenalized intercept. The dual solve is used when
    feature count exceeds sample count; no hidden division by n changes lambda.
    """
    x, y = _matrix(source, "source"), _matrix(target, "target")
    if len(x) != len(y):
        raise ValueError("source and target need paired rows")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective}")
    if objective == "standardized_mean":
        old = AffineRidge(alpha=regularization).fit(x, y)
        return AffineMap(old.coef_, old.intercept_)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    if regularization == 0:
        beta = np.linalg.lstsq(xc, yc, rcond=None)[0]
    elif x.shape[1] > len(x):
        gram = xc @ xc.T + regularization * np.eye(len(x))
        beta = xc.T @ np.linalg.solve(gram, yc)
    else:
        gram = xc.T @ xc + regularization * np.eye(x.shape[1])
        beta = np.linalg.solve(gram, xc.T @ yc)
    return AffineMap(beta.T, y.mean(axis=0) - x.mean(axis=0) @ beta)


def _score(pred: Array, target: Array, scaler: Standardizer) -> dict[str, Any]:
    result = regression_metrics(pred, target, scaler)
    mse = float(np.mean((pred - target) ** 2))
    baseline = float(np.mean((target - scaler.mean) ** 2))
    pnorm = np.linalg.norm(pred, axis=1, keepdims=True)
    ynorm = np.linalg.norm(target, axis=1, keepdims=True)
    pu, yu = pred / np.where(pnorm > 0, pnorm, 1), target / np.where(ynorm > 0, ynorm, 1)
    result.update(
        mse=mse, mse_relative_to_train_mean=mse / baseline if baseline else None,
        mean_matched_cosine=float(np.mean(np.sum(pu * yu, axis=1))),
        cosine_retrieval_top1=float(np.mean(np.argmax(pu @ yu.T, axis=1) == np.arange(len(pred)))),
        retrieval_candidates=len(pred), retrieval_chance=1 / len(pred),
    )
    return result


def _disjoint(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    if set(a["ids"]) & set(b["ids"]):
        raise ValueError("paired IDs cross splits")
    if set(a["families"]) & set(b["families"]):
        raise ValueError("structural families cross splits")


class RosettaPair:
    """Two independently validation-selected maps with frozen train statistics."""

    @classmethod
    def fit(
        cls, train: PairedSplit, validation: PairedSplit,
        lambdas: Iterable[float] = (0.0, 0.001, 0.1, 1.0), *,
        objective: str = "raw_sum", seed: int = 1729,
        math_reference: ArrayLike | None = None,
        language_reference: ArrayLike | None = None,
    ) -> "RosettaPair":
        if train.split != "train" or validation.split != "val":
            raise ValueError("fit requires explicit train and validation splits")
        if len(train.ids) < 2:
            raise ValueError("at least two training pairs required for shuffled control")
        if train.x.shape[1] != validation.x.shape[1] or train.y.shape[1] != validation.y.shape[1]:
            raise ValueError("domain dimensions changed across splits")
        _disjoint(train.fingerprint(), validation.fingerprint())
        grid = sorted(set(float(v) for v in lambdas))
        if not grid or any(not np.isfinite(v) or v < 0 for v in grid):
            raise ValueError("lambdas must be nonempty, finite and nonnegative")
        if objective not in OBJECTIVES:
            raise ValueError(f"unknown objective: {objective}")
        self = cls()
        self.math_scaler = Standardizer.fit(train.x)
        self.language_scaler = Standardizer.fit(train.y)
        self.maps: dict[str, AffineMap] = {}
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(train.ids))
        for _ in range(100):
            if np.all(permutation != np.arange(len(permutation))):
                break
            permutation = rng.permutation(len(train.ids))
        else:
            permutation = np.roll(np.arange(len(train.ids)), 1)
        selection = {}
        tasks = (
            ("T", train.x, train.y, validation.x, validation.y, self.language_scaler),
            ("S", train.y, train.x, validation.y, validation.x, self.math_scaler),
            ("T_shuffled", train.x, train.y[permutation], validation.x, validation.y, self.language_scaler),
            ("S_shuffled", train.y, train.x[permutation], validation.y, validation.x, self.math_scaler),
        )
        for name, x, y, vx, vy, scaler in tasks:
            scores = []
            candidates = []
            for strength in grid:
                candidate = fit_affine(x, y, strength, objective)
                score = float(np.mean((candidate.predict(vx) - vy) ** 2))
                scores.append({"lambda": strength, "validation_mse": score})
                candidates.append(candidate)
            best = min(range(len(grid)), key=lambda i: (scores[i]["validation_mse"], grid[i]))
            self.maps[name] = candidates[best]
            selection[name] = {"selected_lambda": grid[best], "validation_scores": scores}

        def reference(value: ArrayLike | None, mean: Array, name: str) -> Array:
            result = mean.copy() if value is None else np.asarray(value, dtype=float).copy()
            if result.shape != mean.shape or not np.isfinite(result).all():
                raise ValueError(f"{name} must be a fixed finite vector of the domain dimension")
            result.flags.writeable = False
            return result

        self.math_reference = reference(math_reference, self.math_scaler.mean, "math_reference")
        self.language_reference = reference(language_reference, self.language_scaler.mean, "language_reference")
        self.protocol = {
            "schema": "lemmaferry.rosetta.v1", "objective_mode": objective,
            "objective": OBJECTIVES[objective], "seed": seed, "lambda_grid": grid,
            "selection_metric": "raw coordinate mean squared error on validation only",
            "selection": selection, "train": train.fingerprint(), "validation": validation.fingerprint(),
            "shuffled_train_permutation": permutation.tolist(),
            "refit_after_selection": False,
            "reference_policy": "Frozen training mean unless an explicit fixed vector was supplied; a missing slot supplies no inferred semantic pairing.",
            "scope": "Supervised geometric correspondence; round-trip consistency does not establish semantics or reasoning transfer.",
        }
        return self

    def to_language(self, x: ArrayLike) -> Array:
        return self.maps["T"].predict(x)

    def to_math(self, y: ArrayLike) -> Array:
        return self.maps["S"].predict(y)

    def F(self, x: ArrayLike | None, y: ArrayLike | None) -> tuple[Array, Array]:
        """Return (S(y), T(x)); missing slots use frozen reference vectors."""
        xx = None if x is None else _matrix(x, "x")
        yy = None if y is None else _matrix(y, "y")
        count = len(xx) if xx is not None else len(yy) if yy is not None else 1
        if xx is None:
            xx = np.broadcast_to(self.math_reference, (count, len(self.math_reference)))
        if yy is None:
            yy = np.broadcast_to(self.language_reference, (count, len(self.language_reference)))
        if len(xx) != len(yy):
            raise ValueError("F needs equal row counts in both slots")
        return self.to_math(yy), self.to_language(xx)

    def evaluate(self, test: PairedSplit) -> dict[str, Any]:
        if test.split != "test":
            raise ValueError("evaluate requires an explicit test split")
        fingerprint = test.fingerprint()
        _disjoint(self.protocol["train"], fingerprint)
        _disjoint(self.protocol["validation"], fingerprint)
        forward, reverse = self.to_language(test.x), self.to_math(test.y)
        test_metrics = {
            "math_to_language": {
                "paired": _score(forward, test.y, self.language_scaler),
                "shuffled_training_pairs": _score(self.maps["T_shuffled"].predict(test.x), test.y, self.language_scaler),
                "training_mean": _score(np.broadcast_to(self.language_scaler.mean, test.y.shape), test.y, self.language_scaler),
            },
            "language_to_math": {
                "paired": _score(reverse, test.x, self.math_scaler),
                "shuffled_training_pairs": _score(self.maps["S_shuffled"].predict(test.y), test.x, self.math_scaler),
                "training_mean": _score(np.broadcast_to(self.math_scaler.mean, test.x.shape), test.x, self.math_scaler),
            },
        }
        first = self.F(test.x, test.y)
        second = self.F(*first)
        return {
            "status": "completed", "protocol": self.protocol, "test": fingerprint,
            "test_metrics": test_metrics,
            "round_trip": {
                "S_of_T_math": _score(second[0], test.x, self.math_scaler),
                "T_of_S_language": _score(second[1], test.y, self.language_scaler),
                "interpretation": "F(F(x,y)) is a consistency diagnostic; an invertible wrong correspondence can also pass it.",
            },
        }

    def save(self, directory: str | Path) -> dict[str, Any]:
        """Write maps and provenance to a new directory; reject overwrites."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {f"{name}_{field}": getattr(model, field)
                  for name, model in self.maps.items() for field in ("coef", "intercept")}
        arrays.update(
            math_mean=self.math_scaler.mean, math_scale=self.math_scaler.scale,
            language_mean=self.language_scaler.mean, language_scale=self.language_scaler.scale,
            math_reference=self.math_reference, language_reference=self.language_reference,
        )
        archive = directory / "maps.npz"
        np.savez_compressed(archive, **arrays)
        manifest = {
            "schema": "lemmaferry.rosetta.bundle.v1", "protocol": self.protocol,
            "archive": "maps.npz", "archive_sha256": _sha(archive.read_bytes()),
            "arrays_sha256": {key: _array_sha(value) for key, value in arrays.items()},
        }
        manifest["manifest_content_sha256"] = _sha(_json_bytes(manifest))
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        return manifest

    @classmethod
    def load(cls, directory: str | Path) -> "RosettaPair":
        """Validate all archive and parameter hashes before loading numeric data."""
        directory = Path(directory)
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            checksum = manifest.pop("manifest_content_sha256")
            if checksum != _sha(_json_bytes(manifest)):
                raise ValueError("manifest checksum mismatch")
            if manifest.get("schema") != "lemmaferry.rosetta.bundle.v1" or manifest.get("archive") != "maps.npz":
                raise ValueError("unsupported Rosetta bundle")
            archive = directory / "maps.npz"
            if manifest["archive_sha256"] != _sha(archive.read_bytes()):
                raise ValueError("map archive checksum mismatch")
            with np.load(archive, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            if set(arrays) != set(manifest["arrays_sha256"]):
                raise ValueError("map array fields differ from manifest")
            for key, value in arrays.items():
                if manifest["arrays_sha256"][key] != _array_sha(value) or not np.isfinite(value).all():
                    raise ValueError(f"map parameter checksum mismatch: {key}")
            self = cls()
            self.protocol = manifest["protocol"]
            if self.protocol.get("schema") != "lemmaferry.rosetta.v1":
                raise ValueError("unsupported Rosetta protocol")
            self.maps = {name: AffineMap(arrays[f"{name}_coef"], arrays[f"{name}_intercept"])
                         for name in ("T", "S", "T_shuffled", "S_shuffled")}
            for domain in ("math", "language"):
                mean, scale = arrays[f"{domain}_mean"], arrays[f"{domain}_scale"]
                if mean.ndim != 1 or scale.shape != mean.shape or np.any(scale <= 0):
                    raise ValueError("invalid saved training statistics")
                setattr(self, f"{domain}_scaler", Standardizer(mean, scale))
                ref = arrays[f"{domain}_reference"]
                if ref.shape != mean.shape:
                    raise ValueError("invalid reference dimension")
                ref.flags.writeable = False
                setattr(self, f"{domain}_reference", ref)
            math_dim, language_dim = len(self.math_scaler.mean), len(self.language_scaler.mean)
            for name, dims in (("T", (language_dim, math_dim)), ("S", (math_dim, language_dim)),
                               ("T_shuffled", (language_dim, math_dim)), ("S_shuffled", (math_dim, language_dim))):
                if self.maps[name].coef.shape != dims:
                    raise ValueError("map dimensions disagree with training statistics")
            return self
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Rosetta bundle: {directory}") from exc
