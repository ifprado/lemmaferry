"""Small, inspectable baselines for paired vector-space alignment.

These routines align numeric arrays. They do not establish semantic equivalence
or demonstrate transfer of an LLM's reasoning ability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _matrix(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return array


@dataclass(frozen=True)
class Standardizer:
    """Column statistics fitted on training rows only."""

    mean: FloatArray
    scale: FloatArray

    @classmethod
    def fit(cls, training: ArrayLike) -> "Standardizer":
        data = _matrix(training, "training")
        scale = data.std(axis=0)
        # A constant training column has no observed variation. Keep it finite
        # without looking at held-out data to choose a different scale.
        scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
        return cls(mean=data.mean(axis=0), scale=scale)

    def transform(self, values: ArrayLike) -> FloatArray:
        data = _matrix(values, "values")
        if data.shape[1] != self.mean.size:
            raise ValueError("column count differs from fitted training data")
        return (data - self.mean) / self.scale

    def inverse_transform(self, values: ArrayLike) -> FloatArray:
        data = _matrix(values, "values")
        if data.shape[1] != self.mean.size:
            raise ValueError("column count differs from fitted training data")
        return data * self.scale + self.mean


class AffineRidge:
    """Fit T(x) = W x + b to paired training rows.

    The fitting objective in standardized coordinates is
    mean(||X B - Y||^2) + alpha * ||B||^2. Both standardizers use
    training rows exclusively. Prediction accepts raw source coordinates and
    returns raw target coordinates. Dimensions may differ between domains.
    """

    def __init__(self, alpha: float = 1e-6) -> None:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.alpha = float(alpha)
        self.source_scaler_: Standardizer | None = None
        self.target_scaler_: Standardizer | None = None
        self.coef_: FloatArray | None = None
        self.intercept_: FloatArray | None = None

    def fit(self, source: ArrayLike, target: ArrayLike) -> "AffineRidge":
        x, y = _matrix(source, "source"), _matrix(target, "target")
        if x.shape[0] != y.shape[0]:
            raise ValueError("source and target need one paired row per example")
        self.source_scaler_ = Standardizer.fit(x)
        self.target_scaler_ = Standardizer.fit(y)
        xs = self.source_scaler_.transform(x)
        ys = self.target_scaler_.transform(y)
        if self.alpha == 0:
            beta = np.linalg.lstsq(xs, ys, rcond=None)[0]
        else:
            gram = xs.T @ xs / x.shape[0]
            gram += self.alpha * np.eye(x.shape[1])
            beta = np.linalg.solve(gram, xs.T @ ys / x.shape[0])
        self.coef_ = (
            beta.T * self.target_scaler_.scale[:, None]
            / self.source_scaler_.scale[None, :]
        )
        self.intercept_ = (
            self.target_scaler_.mean
            - self.coef_ @ self.source_scaler_.mean
        )
        return self

    def predict(self, source: ArrayLike) -> FloatArray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit the translator before predicting")
        x = _matrix(source, "source")
        if x.shape[1] != self.coef_.shape[1]:
            raise ValueError("source dimension differs from fitted translator")
        return x @ self.coef_.T + self.intercept_


class OrthogonalProcrustes:
    """Equal-dimension rigid alignment, with centering and reflections allowed.

    This baseline preserves distances. It cannot model arbitrary stretching,
    nonlinear correspondences, or translations between different dimensions.
    """

    def fit(self, source: ArrayLike, target: ArrayLike) -> "OrthogonalProcrustes":
        x, y = _matrix(source, "source"), _matrix(target, "target")
        if x.shape != y.shape:
            raise ValueError("orthogonal alignment requires equal array shapes")
        self.source_mean_ = x.mean(axis=0)
        self.target_mean_ = y.mean(axis=0)
        u, _, vt = np.linalg.svd(
            (x - self.source_mean_).T @ (y - self.target_mean_)
        )
        self.rotation_ = u @ vt
        return self

    def predict(self, source: ArrayLike) -> FloatArray:
        if not hasattr(self, "rotation_"):
            raise RuntimeError("fit the translator before predicting")
        x = _matrix(source, "source")
        if x.shape[1] != self.rotation_.shape[0]:
            raise ValueError("source dimension differs from fitted translator")
        return (x - self.source_mean_) @ self.rotation_ + self.target_mean_


def regression_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    training_target_scaler: Standardizer,
) -> dict[str, float]:
    """Report raw RMSE and per-coordinate train-standardized RMSE.

    normalized_rmse = sqrt(mean(((prediction - target) / train_std)**2)).
    It does not recompute scales on test data or divide by a test-set range.
    Constant training coordinates use scale 1, as Standardizer documents.
    """
    pred, actual = _matrix(prediction, "prediction"), _matrix(target, "target")
    if pred.shape != actual.shape:
        raise ValueError("prediction and target shapes must match")
    if actual.shape[1] != training_target_scaler.mean.size:
        raise ValueError("target dimension differs from training scaler")
    error = pred - actual
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": float(
            np.sqrt(np.mean((error / training_target_scaler.scale) ** 2))
        ),
    }


def synthetic_alignment_experiment(
    seed: int = 7,
    train_size: int = 256,
    test_size: int = 256,
    source_dim: int = 12,
    target_dim: int = 8,
    latent_dim: int = 4,
    noise: float = 0.01,
) -> tuple[dict, dict[str, FloatArray]]:
    """Artificial shared-factor benchmark, with held-out evaluation.

    Shared factors and the task are deliberately planted by construction.
    Successful alignment here is a code check, not evidence about LLMs.
    """
    if min(train_size, test_size, source_dim, target_dim, latent_dim) < 1:
        raise ValueError("sizes and dimensions must be positive")
    if latent_dim > min(source_dim, target_dim):
        raise ValueError("each representation must expose all planted factors")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(latent_dim, source_dim))
    b = rng.normal(size=(latent_dim, target_dim))
    source_offset = rng.normal(size=source_dim) * 2
    target_offset = rng.normal(size=target_dim) * 2
    latent = rng.normal(size=(train_size + test_size, latent_dim))
    source = latent @ a + source_offset + noise * rng.normal(
        size=(latent.shape[0], source_dim)
    )
    target = latent @ b + target_offset + noise * rng.normal(
        size=(latent.shape[0], target_dim)
    )
    x_train, x_test = source[:train_size], source[train_size:]
    y_train, y_test = target[:train_size], target[train_size:]
    translator = AffineRidge().fit(x_train, y_train)
    wrong = AffineRidge().fit(x_train, y_train[rng.permutation(train_size)])
    predictions = {
        "paired_affine": translator.predict(x_test),
        "wrong_pairs": wrong.predict(x_test),
        "training_mean": np.broadcast_to(y_train.mean(axis=0), y_test.shape),
    }
    # A target-space decoder is trained only on receiver training states. The
    # task labels come from a planted latent linear decision boundary.
    task_direction = np.linspace(1.0, -0.4, latent_dim)
    task_score = latent @ task_direction
    target_decoder = AffineRidge().fit(y_train, task_score[:train_size, None])
    expected_label = task_score[train_size:] > 0

    def accuracy(states: FloatArray) -> float:
        predicted_label = target_decoder.predict(states)[:, 0] > 0
        return float(np.mean(predicted_label == expected_label))

    assert translator.target_scaler_ is not None
    metrics = {
        name: {
            **regression_metrics(values, y_test, translator.target_scaler_),
            "receiver_task_accuracy": accuracy(values),
        }
        for name, values in predictions.items()
    }
    report = {
        "experiment": "artificial_shared_latent_alignment",
        "scope": "Synthetic vectors with planted shared factors; not LLM skill transfer.",
        "config": {
            "seed": seed,
            "train_size": train_size,
            "test_size": test_size,
            "source_dim": source_dim,
            "target_dim": target_dim,
            "latent_dim": latent_dim,
            "noise_std": noise,
            "ridge_alpha": translator.alpha,
        },
        "normalization": "Per-coordinate target standard deviation from training rows only.",
        "receiver_task": "Sign of a planted linear combination of latent factors.",
        "receiver_task_accuracy_on_true_target": accuracy(y_test),
        "metrics": metrics,
    }
    return report, {"target_test": y_test, **predictions}


def symmetric_cloud_ambiguity(points: int = 32) -> dict:
    """Counterexample: identical unpaired geometry can hide opposite semantics.

    The identity and a half-turn map the same ring onto itself. Both have an
    exact inverse and equal unpaired geometric error. Their planted semantic
    labels disagree. This is an analytic control, not a learned-model result.
    """
    if points < 4 or points % 2:
        raise ValueError("points must be even and at least four")
    angle = (np.arange(points) + 0.25) * 2 * np.pi / points
    cloud = np.column_stack((np.cos(angle), np.sin(angle)))
    labels = cloud[:, 0] > 0

    def evaluate(matrix: FloatArray) -> dict[str, float]:
        translated = cloud @ matrix
        distances = np.linalg.norm(translated[:, None] - cloud[None, :], axis=2)
        return {
            "unpaired_nearest_neighbor_rmse": float(
                np.sqrt(np.mean(np.min(distances, axis=1) ** 2))
            ),
            "round_trip_rmse": float(np.sqrt(np.mean((translated @ matrix - cloud) ** 2))),
            "paired_rmse": float(np.sqrt(np.mean((translated - cloud) ** 2))),
            "semantic_label_accuracy": float(np.mean((translated[:, 0] > 0) == labels)),
        }

    return {
        "experiment": "unpaired_symmetric_cloud_ambiguity",
        "scope": "Exact counterexample with two hand-specified maps; no training performed.",
        "points": points,
        "identity": evaluate(np.eye(2)),
        "half_turn": evaluate(-np.eye(2)),
        "interpretation": "Matching clouds and perfect cycles do not identify semantic correspondence.",
    }
