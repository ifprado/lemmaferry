"""Typed two-port cross-autoencoder and an optional reversible variant.

Adapted from the user's Apache-2.0 standalone LemmaFerry swap prototype. A is the mathematical representation; B is
its general-model counterpart. Outputs retain their domain types, while their
content travels across ports. No Lean checker or language model is included.
SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .alignment import Standardizer, _matrix
from .rosetta import PairedSplit, _array_sha, _disjoint, _json_bytes, _score, _sha


@dataclass(frozen=True)
class Slot:
    """A batch of vectors and one explicit presence flag per row.

    A numeric zero is a valid vector. It is NOT automatically an empty slot.
    Use Slot.empty or a false presence flag to request the reference marker.
    """
    values: Tensor
    present: Tensor

    @classmethod
    def full(cls, values: Tensor) -> Slot:
        if values.ndim != 2:
            raise ValueError("Slot.full expects a [batch, dimension] matrix")
        return cls(values, torch.ones(values.shape[0], dtype=torch.bool, device=values.device))

    @classmethod
    def empty(cls, batch: int, dimension: int, *, like: Tensor) -> Slot:
        if batch < 1 or dimension < 1:
            raise ValueError("batch and dimension must be positive")
        return cls(like.new_zeros((batch, dimension)),
                   torch.zeros(batch, dtype=torch.bool, device=like.device))


def _validate_matrix(x: Tensor, dimension: int, name: str) -> None:
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != dimension:
        raise ValueError(f"{name} must have shape [nonempty batch, {dimension}]")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must contain finite floating-point numbers")


def _validate_slot(slot: Slot, dimension: int) -> None:
    _validate_matrix(slot.values, dimension, "slot.values")
    if slot.present.dtype != torch.bool or slot.present.shape != (len(slot.values),):
        raise ValueError("slot.present must be a boolean vector with one entry per row")
    if slot.present.device != slot.values.device:
        raise ValueError("slot values and presence flags must use the same device")


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                         nn.Linear(hidden_dim, output_dim))


class DomainStatistics(nn.Module):
    """Normalization fitted exclusively on the training set, stored in checkpoints."""
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, training: Tensor) -> None:
        _validate_matrix(training, self.mean.numel(), "training")
        if bool(self.fitted):
            raise RuntimeError("Normalization is already fitted; construct a fresh model to refit")
        if training.shape[0] < 2:
            raise ValueError("At least two training rows are required")
        values = training.to(device=self.mean.device, dtype=self.mean.dtype)
        scale = values.std(dim=0, unbiased=False)
        self.mean.copy_(values.mean(dim=0))
        self.scale.copy_(torch.where(scale > 1e-6, scale, torch.ones_like(scale)))
        self.fitted.fill_(True)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.scale

    def restore(self, x: Tensor) -> Tensor:
        return x * self.scale + self.mean

    def error(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(self.normalize(prediction), self.normalize(target))


class TwoPortBridge(nn.Module):
    """Mask-aware swap wrapper shared by both concrete architectures."""
    dim_a: int
    dim_b: int

    def a_to_b(self, a: Tensor) -> Tensor:
        raise NotImplementedError

    def b_to_a(self, b: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, a: Slot, b: Slot) -> tuple[Slot, Slot]:
        _validate_slot(a, self.dim_a)
        _validate_slot(b, self.dim_b)
        if len(a.values) != len(b.values):
            raise ValueError("The two ports must have equal batch sizes")
        if a.values.device != b.values.device or a.values.dtype != b.values.dtype:
            raise ValueError("Both ports must use the same device and floating dtype")
        # No path connects A's input to A's output, or B's input to B's output.
        # In particular, an output cannot copy its paired target input.
        safe_a = torch.where(a.present[:, None], a.values, torch.zeros_like(a.values))
        safe_b = torch.where(b.present[:, None], b.values, torch.zeros_like(b.values))
        translated_a = self.b_to_a(safe_b)
        translated_b = self.a_to_b(safe_a)
        out_a = torch.where(b.present[:, None], translated_a, torch.zeros_like(translated_a))
        out_b = torch.where(a.present[:, None], translated_b, torch.zeros_like(translated_b))
        return Slot(out_a, b.present.clone()), Slot(out_b, a.present.clone())


class SwapAutoencoder(TwoPortBridge):
    """F(a,b) = (D_A(E_B(b)), D_B(E_A(a))), with explicit empty markers.

    Four trainable MLPs. The latent exchange is fixed wiring. Self-reconstruction,
    paired translation, cycle reconstruction, and latent agreement train the MLPs.
    A and B may have different dimensions. Cycles are approximate, not guaranteed.
    """
    def __init__(self, dim_a: int, dim_b: int, latent_dim: int = 8,
                 hidden_dim: int = 64) -> None:
        super().__init__()
        if min(dim_a, dim_b, latent_dim, hidden_dim) < 1:
            raise ValueError("All dimensions must be positive")
        self.dim_a, self.dim_b = dim_a, dim_b
        self.latent_dim, self.hidden_dim = latent_dim, hidden_dim
        self.stats_a = DomainStatistics(dim_a)
        self.stats_b = DomainStatistics(dim_b)
        self.encoder_a = mlp(dim_a, hidden_dim, latent_dim)
        self.encoder_b = mlp(dim_b, hidden_dim, latent_dim)
        self.decoder_a = mlp(latent_dim, hidden_dim, dim_a)
        self.decoder_b = mlp(latent_dim, hidden_dim, dim_b)

    def configuration(self) -> dict[str, int]:
        return dict(dim_a=self.dim_a, dim_b=self.dim_b,
                    latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)

    def fit_normalization(self, train_a: Tensor, train_b: Tensor) -> None:
        if train_a.shape[0] != train_b.shape[0]:
            raise ValueError("Normalization inputs must be paired training rows")
        if bool(self.stats_a.fitted) or bool(self.stats_b.fitted):
            raise RuntimeError("Normalization cannot be refitted")
        _validate_matrix(train_a, self.dim_a, "train_a")
        _validate_matrix(train_b, self.dim_b, "train_b")
        self.stats_a.fit(train_a)
        self.stats_b.fit(train_b)

    def encode_a(self, a: Tensor) -> Tensor:
        return self.encoder_a(self.stats_a.normalize(a))

    def encode_b(self, b: Tensor) -> Tensor:
        return self.encoder_b(self.stats_b.normalize(b))

    def decode_a(self, z: Tensor) -> Tensor:
        return self.stats_a.restore(self.decoder_a(z))

    def decode_b(self, z: Tensor) -> Tensor:
        return self.stats_b.restore(self.decoder_b(z))

    def a_to_b(self, a: Tensor) -> Tensor:
        return self.decode_b(self.encode_a(a))

    def b_to_a(self, b: Tensor) -> Tensor:
        return self.decode_a(self.encode_b(b))

    def losses(self, a: Tensor, b: Tensor, permutation: Tensor | None = None,
               *, self_weight: float = 0.25, cycle_weight: float = 0.1,
               latent_weight: float = 0.01) -> dict[str, Tensor]:
        """Correspondence pairs are (a[i], b[i]). Slots use independent indices.

        With j = permutation[i], train F(a[i],b[j]) -> (a[j],b[i]).
        All reconstruction errors use train-only, per-domain normalized units.
        """
        _validate_matrix(a, self.dim_a, "a")
        _validate_matrix(b, self.dim_b, "b")
        if len(a) != len(b):
            raise ValueError("Paired batch sizes differ")
        if min(self_weight, cycle_weight, latent_weight) < 0:
            raise ValueError("Loss weights must be nonnegative")
        if permutation is None:
            permutation = torch.randperm(len(a), device=a.device)
        if (permutation.dtype != torch.long or permutation.shape != (len(a),)
                or permutation.device != a.device
                or not torch.equal(permutation.sort().values, torch.arange(len(a), device=a.device))):
            raise ValueError("permutation must contain every batch index exactly once")
        za, zb = self.encode_a(a), self.encode_b(b)
        predicted_a, predicted_b = self.decode_a(zb), self.decode_b(za)
        # This is precisely the independently indexed two-port swap objective.
        pair = self.stats_a.error(predicted_a[permutation], a[permutation])
        pair = pair + self.stats_b.error(predicted_b, b)
        own = self.stats_a.error(self.decode_a(za), a) + self.stats_b.error(self.decode_b(zb), b)
        cycle = self.stats_a.error(self.b_to_a(predicted_b), a)
        cycle = cycle + self.stats_b.error(self.a_to_b(predicted_a), b)
        latent = F.mse_loss(za, zb)
        total = pair + self_weight * own + cycle_weight * cycle + latent_weight * latent
        return dict(total=total, pair=pair, self_reconstruction=own,
                    cycle=cycle, latent=latent)


class AdditiveCoupling(nn.Module):
    """Invertible coupling: unchanged coordinates determine an additive shift.

    This is an established coupling-layer construction, not a novelty claim.
    """
    def __init__(self, dimension: int, hidden_dim: int, parity: int) -> None:
        super().__init__()
        self.register_buffer("mask", ((torch.arange(dimension) + parity) % 2).float())
        self.shift = mlp(dimension, hidden_dim, dimension)

    def forward(self, x: Tensor, inverse: bool = False) -> Tensor:
        delta = self.shift(x * self.mask) * (1 - self.mask)
        return x - delta if inverse else x + delta


class InvertibleCoordinates(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AdditiveCoupling(dimension, hidden_dim, i % 2)
                                     for i in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, z: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            z = layer(z, inverse=True)
        return z


class ReversibleSwap(TwoPortBridge):
    """Equal-dimensional alternative: F = H^{-1} J H, hence F^2 = identity.

    Inversion holds algebraically for every parameter setting, up to numerical
    arithmetic in execution. Semantic correspondence still needs paired training.
    The supplied demo trains SwapAutoencoder, NOT this optional alternative.
    """
    def __init__(self, dimension: int, hidden_dim: int = 32, depth: int = 4) -> None:
        super().__init__()
        if dimension < 2 or hidden_dim < 1 or depth < 1:
            raise ValueError("dimension >= 2, hidden_dim >= 1, and depth >= 1 required")
        self.dim_a = self.dim_b = dimension
        self.h_a = InvertibleCoordinates(dimension, hidden_dim, depth)
        self.h_b = InvertibleCoordinates(dimension, hidden_dim, depth)

    def a_to_b(self, a: Tensor) -> Tensor:
        return self.h_b.inverse(self.h_a(a))

    def b_to_a(self, b: Tensor) -> Tensor:
        return self.h_a.inverse(self.h_b(b))


@dataclass(frozen=True)
class SwapConfig:
    """Registered full training budget; validation selects a retained checkpoint."""
    latent_dim: int = 8
    hidden_dim: int = 64
    steps: int = 128
    batch_size: int = 32
    learning_rate: float = 0.002
    validation_every: int = 16
    self_weight: float = 0.25
    cycle_weight: float = 0.10
    latent_weight: float = 0.01
    weight_decay: float = 0.0001
    gradient_clip: float = 5.0
    device: str = "cpu"

    def __post_init__(self):
        for name in ("latent_dim", "hidden_dim", "steps", "batch_size", "validation_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "self_weight", "cycle_weight", "latent_weight", "weight_decay", "gradient_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0 or self.gradient_clip == 0:
            raise ValueError("learning rate and gradient clip must be positive")
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError("swap device must be cpu, mps or cuda")


def shuffled_permutation(count: int, seed: int) -> np.ndarray:
    """One seeded derangement defines the complete joint shuffled training pairs."""
    if type(count) is not int or count < 2:
        raise ValueError("at least two training pairs required")
    rng = np.random.default_rng(seed + 10000)
    for _ in range(100):
        result = rng.permutation(count)
        if np.all(result != np.arange(count)):
            return result
    return np.roll(np.arange(count), 1)


def training_schedule(count: int, config: SwapConfig, seed: int):
    """Deterministic row batches and independent cross-port permutations."""
    rng = np.random.default_rng(seed + 20000)
    for _ in range(config.steps):
        # When the registered batch exceeds the corpus, cycle the shuffled rows.
        # Every step still has the same fixed tensor shapes and pair exposures.
        rows = np.resize(rng.permutation(count), config.batch_size).astype(np.int64)
        ports = rng.permutation(config.batch_size).astype(np.int64)
        yield rows, ports


def _schedule_hash(count: int, config: SwapConfig, seed: int) -> str:
    digest = hashlib.sha256()
    for rows, ports in training_schedule(count, config, seed):
        digest.update(rows.astype("<i8").tobytes())
        digest.update(ports.astype("<i8").tobytes())
    return digest.hexdigest()


def _state_arrays(model: SwapAutoencoder) -> dict[str, np.ndarray]:
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def _state_hash(arrays: Mapping[str, np.ndarray]) -> str:
    return _sha(_json_bytes({name: _array_sha(value) for name, value in sorted(arrays.items())}))


def _model(dim_a: int, dim_b: int, config: SwapConfig, seed: int) -> SwapAutoencoder:
    if min(dim_a, dim_b) < 1 or max(dim_a, dim_b, config.latent_dim, config.hidden_dim) > 100000:
        raise ValueError("invalid or excessive swap architecture dimension")
    parameters = 2 * config.hidden_dim * (dim_a + dim_b + 2 * config.latent_dim + 4) + dim_a + dim_b + 2 * config.latent_dim
    if parameters > 50_000_000:
        raise ValueError("swap architecture exceeds the 50 million parameter bundle limit")
    # Creating or loading a translator must not reset the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SwapAutoencoder(dim_a, dim_b, config.latent_dim, config.hidden_dim)


def _finite_tensor(values, device="cpu"):
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("swap inputs overflow float32")
    return torch.tensor(array, dtype=torch.float32, device=device)


class SwapPair:
    """Matched paired/shuffled cross-autoencoders behind the translator API."""
    backend = "swap_autoencoder"

    @classmethod
    def fit(cls, train: PairedSplit, validation: PairedSplit, *,
            config: SwapConfig | Mapping[str, Any] | None = None, seed: int = 1729):
        config = config if isinstance(config, SwapConfig) else SwapConfig(**dict(config or {}))
        if type(seed) is not int or not 0 <= seed < 2**63 - 20000:
            raise ValueError("seed must be a nonnegative integer below 2**63 - 20000")
        if train.split != "train" or validation.split != "val":
            raise ValueError("swap fitting requires train and validation splits")
        if len(train.ids) < 2:
            raise ValueError("at least two training pairs required")
        if (train.x.shape[1], train.y.shape[1]) != (validation.x.shape[1], validation.y.shape[1]):
            raise ValueError("domain dimensions changed across splits")
        _disjoint(train.fingerprint(), validation.fingerprint())
        if config.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self = cls()
        self.models = {}
        self.config = config
        permutation = shuffled_permutation(len(train.ids), seed)
        train_x, train_y = _finite_tensor(train.x), _finite_tensor(train.y)
        vx, vy = _finite_tensor(validation.x, config.device), _finite_tensor(validation.y, config.device)
        training = {}
        validation_steps = [step for step in range(1, config.steps + 1)
                            if step % config.validation_every == 0 or step == config.steps]
        for arm in ("paired", "shuffled"):
            model = _model(train.x.shape[1], train.y.shape[1], config, seed)
            # Fit both arms on the exact same unpermuted training marginals.
            model.fit_normalization(train_x, train_y)
            initial = _state_hash(_state_arrays(model))
            model.to(config.device)
            a = train_x.to(config.device)
            b = train_y[(permutation if arm == "shuffled" else np.arange(len(train.ids)))].to(config.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay, betas=(0.9, 0.999), eps=1e-8)
            trace, best_score, best_step, best_state = [], float("inf"), None, None
            for step, (rows, ports) in enumerate(training_schedule(len(train.ids), config, seed), start=1):
                model.train()
                selected = torch.tensor(rows, device=config.device)
                cross_ports = torch.tensor(ports, device=config.device)
                losses = model.losses(a[selected], b[selected], cross_ports,
                                      self_weight=config.self_weight, cycle_weight=config.cycle_weight,
                                      latent_weight=config.latent_weight)
                if not bool(torch.isfinite(losses["total"])):
                    raise ValueError("nonfinite swap training loss")
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                if step in validation_steps:
                    model.eval()
                    with torch.inference_mode():
                        score = float((model.stats_b.error(model.a_to_b(vx), vy)
                                       + model.stats_a.error(model.b_to_a(vy), vx)).cpu())
                    if not np.isfinite(score):
                        raise ValueError("nonfinite swap validation loss")
                    trace.append({"step": step, "validation_bidirectional_normalized_mse": score,
                                  "training_total": float(losses["total"].detach().cpu())})
                    if score < best_score:
                        best_score, best_step, best_state = score, step, _state_arrays(model)
            if best_state is None:
                raise RuntimeError("no validation checkpoint selected")
            model.to("cpu")
            model.load_state_dict({name: torch.from_numpy(value.copy()) for name, value in best_state.items()}, strict=True)
            model.eval().requires_grad_(False)
            self.models[arm] = model
            training[arm] = {
                "initial_state_sha256": initial, "batch_order_sha256": _schedule_hash(len(train.ids), config, seed),
                "updates": config.steps, "pair_exposures": config.steps * config.batch_size,
                "validation_steps": validation_steps, "selected_step": best_step, "trace": trace,
                "selected_state_sha256": _state_hash(best_state),
            }
        for field in ("initial_state_sha256", "batch_order_sha256", "updates", "pair_exposures", "validation_steps"):
            if training["paired"][field] != training["shuffled"][field]:
                raise RuntimeError(f"neural control budgets differ: {field}")
        model = self.models["paired"]
        self.math_scaler = Standardizer(model.stats_a.mean.numpy().astype(float), model.stats_a.scale.numpy().astype(float))
        self.language_scaler = Standardizer(model.stats_b.mean.numpy().astype(float), model.stats_b.scale.numpy().astype(float))
        self.protocol = {
            "schema": "lemmaferry.swap.v1", "backend": self.backend, "translator": "swap",
            "objective_mode": "standardized_multiloss", "seed": seed, "config": asdict(config),
            "train": train.fingerprint(), "validation": validation.fingerprint(),
            "shuffled_train_permutation": permutation.tolist(),
            "shuffled_pairing_policy": "joint pairs (x_i,y_perm[i]) in BOTH directions and all loss terms; one seeded derangement",
            "permutation_algorithm": "numpy.default_rng(seed+10000); first derangement within 100 draws, else cyclic shift",
            "schedule_algorithm": "numpy.default_rng(seed+20000); each step resize(permutation(n),batch), then permutation(batch)",
            "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8},
            "normalization": "CPU float32 mean and population std on training only; std <= 1e-6 replaced with 1",
            "selection_metric": "validation sum of bidirectional per-coordinate train-standardized MSE; earliest strict minimum",
            "refit_after_selection": False, "training": training,
            "source_sha256": _sha(Path(__file__).read_bytes()),
            "environment": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
            "empty_policy": "Explicit Slot presence flag; false payload is exactly zero, present zero remains data",
            "scope": "Paired numeric representation learning; cycles, latent agreement and empty routing do not establish semantic or capability transfer.",
        }
        self.validate_training_data(train, validation)
        return self

    def to_language(self, values, control="paired") -> np.ndarray:
        data = _matrix(values, "values")
        if data.shape[1] != len(self.math_scaler.mean):
            raise ValueError("mathematical dimension differs from translator")
        if control == "mean":
            return np.broadcast_to(self.language_scaler.mean, (len(data), len(self.language_scaler.mean))).copy()
        if control not in self.models:
            raise ValueError("control must be paired, shuffled or mean")
        with torch.inference_mode():
            return self.models[control].a_to_b(_finite_tensor(data)).detach().numpy().astype(float)

    def to_math(self, values, control="paired") -> np.ndarray:
        data = _matrix(values, "values")
        if data.shape[1] != len(self.language_scaler.mean):
            raise ValueError("language dimension differs from translator")
        if control == "mean":
            return np.broadcast_to(self.math_scaler.mean, (len(data), len(self.math_scaler.mean))).copy()
        if control not in self.models:
            raise ValueError("control must be paired, shuffled or mean")
        with torch.inference_mode():
            return self.models[control].b_to_a(_finite_tensor(data)).detach().numpy().astype(float)

    def F(self, x: Slot, y: Slot, control="paired") -> tuple[Slot, Slot]:
        if control not in self.models:
            raise ValueError("two-port F supports paired or shuffled learned models")
        with torch.inference_mode():
            return self.models[control](x, y)

    def evaluate(self, test: PairedSplit) -> dict[str, Any]:
        if test.split != "test":
            raise ValueError("evaluate requires an explicit test split")
        _disjoint(self.protocol["train"], test.fingerprint())
        _disjoint(self.protocol["validation"], test.fingerprint())
        metrics = {"math_to_language": {}, "language_to_math": {}}
        for control, label in (("paired", "paired"), ("shuffled", "shuffled_training_pairs"), ("mean", "training_mean")):
            metrics["math_to_language"][label] = _score(self.to_language(test.x, control), test.y, self.language_scaler)
            metrics["language_to_math"][label] = _score(self.to_math(test.y, control), test.x, self.math_scaler)
        model = self.models["paired"]
        a, b = _finite_tensor(test.x), _finite_tensor(test.y)
        with torch.inference_mode():
            own_a = model.decode_a(model.encode_a(a)).numpy().astype(float)
            own_b = model.decode_b(model.encode_b(b)).numpy().astype(float)
            forward_empty = model(Slot.full(a), Slot.empty(len(a), model.dim_b, like=a))
            reverse_empty = model(Slot.empty(len(b), model.dim_a, like=b), Slot.full(b))
            two = model(*forward_empty)
        return {
            "status": "completed", "protocol": json.loads(json.dumps(self.protocol)), "test": test.fingerprint(),
            "test_metrics": metrics,
            "round_trip": {
                "S_of_T_math": _score(self.to_math(self.to_language(test.x)), test.x, self.math_scaler),
                "T_of_S_language": _score(self.to_language(self.to_math(test.y)), test.y, self.language_scaler),
                "interpretation": "Approximate cross-autoencoder consistency; this is not semantic or task validation.",
            },
            "self_reconstruction": {"math": _score(own_a, test.x, self.math_scaler),
                                    "language": _score(own_b, test.y, self.language_scaler)},
            "empty_routing": {
                "empty_payloads_exact_zero": bool(not forward_empty[0].values.any() and not reverse_empty[1].values.any()),
                "empty_flags_crossed": bool(not forward_empty[0].present.any() and forward_empty[1].present.all()
                                             and reverse_empty[0].present.all() and not reverse_empty[1].present.any()),
                "two_swaps_presence_restored": bool(two[0].present.all() and not two[1].present.any()),
                "interpretation": "Presence routing is architectural, not learned semantic evidence.",
            },
        }

    def validate_training_data(self, train: PairedSplit, validation: PairedSplit) -> None:
        """Bind a loaded bundle to externally retained calibration data."""
        if train.fingerprint() != self.protocol["train"] or validation.fingerprint() != self.protocol["validation"]:
            raise ValueError("saved swap training/validation fingerprint differs")
        stats = DomainStatistics(train.x.shape[1])
        stats.fit(_finite_tensor(train.x))
        other = DomainStatistics(train.y.shape[1])
        other.fit(_finite_tensor(train.y))
        for actual, expected in ((self.math_scaler.mean, stats.mean.numpy()), (self.math_scaler.scale, stats.scale.numpy()),
                                 (self.language_scaler.mean, other.mean.numpy()), (self.language_scaler.scale, other.scale.numpy())):
            if not np.array_equal(actual, expected):
                raise ValueError("saved swap normalization differs from train-only statistics")
        if self.protocol["shuffled_train_permutation"] != shuffled_permutation(len(train.ids), self.protocol["seed"]).tolist():
            raise ValueError("saved shuffled pairing differs from the declared seeded derangement")

    def save(self, directory: str | Path) -> dict[str, Any]:
        self._validate_protocol(self.protocol, self.config)
        if any(_state_hash(_state_arrays(model)) != self.protocol["training"][arm]["selected_state_sha256"]
               for arm, model in self.models.items()):
            raise ValueError("swap selected weights changed after fitting")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {f"{arm}::{name}": value for arm, model in self.models.items()
                  for name, value in _state_arrays(model).items()}
        arrays.update(math_mean=self.math_scaler.mean, math_scale=self.math_scaler.scale,
                      language_mean=self.language_scaler.mean, language_scale=self.language_scaler.scale)
        archive = directory / "maps.npz"
        np.savez_compressed(archive, **arrays)
        manifest = {
            "schema": "lemmaferry.swap.bundle.v1", "archive": "maps.npz", "archive_sha256": _sha(archive.read_bytes()),
            "protocol": self.protocol,
            "arrays_sha256": {name: _array_sha(value) for name, value in arrays.items()},
            "array_specs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        }
        manifest["manifest_content_sha256"] = _sha(_json_bytes(manifest))
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        return manifest

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            content = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
            if manifest.get("manifest_content_sha256") != _sha(_json_bytes(content)):
                raise ValueError("swap manifest checksum mismatch")
            expected_fields = {"schema", "archive", "archive_sha256", "protocol", "arrays_sha256", "array_specs", "manifest_content_sha256"}
            if set(manifest) != expected_fields or manifest["schema"] != "lemmaferry.swap.bundle.v1" or manifest["archive"] != "maps.npz":
                raise ValueError("unsupported swap bundle schema")
            protocol = manifest["protocol"]
            config = SwapConfig(**protocol["config"])
            cls._validate_protocol(protocol, config)
            archive = directory / "maps.npz"
            if _sha(archive.read_bytes()) != manifest["archive_sha256"]:
                raise ValueError("swap archive checksum mismatch")
            with np.load(archive, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            if set(arrays) != set(manifest["arrays_sha256"]) or set(arrays) != set(manifest["array_specs"]):
                raise ValueError("swap numeric fields differ from manifest")
            for name, value in arrays.items():
                if value.dtype.kind not in "fb" or not np.isfinite(value).all():
                    raise ValueError("swap numeric values must be finite float/bool arrays")
                if _array_sha(value) != manifest["arrays_sha256"][name]:
                    raise ValueError("swap parameter checksum mismatch")
                if manifest["array_specs"][name] != {"shape": list(value.shape), "dtype": str(value.dtype)}:
                    raise ValueError("swap array shape/dtype metadata differs")
            self = cls()
            self.config, self.protocol, self.models = config, protocol, {}
            dims = protocol["train"]["math_dimension"], protocol["train"]["language_dimension"]
            required = {"math_mean", "math_scale", "language_mean", "language_scale"}
            for arm in ("paired", "shuffled"):
                model = _model(*dims, config, protocol["seed"])
                state = {}
                for name, template in model.state_dict().items():
                    key = f"{arm}::{name}"
                    required.add(key)
                    value = arrays[key]
                    if value.shape != tuple(template.shape) or value.dtype != template.numpy().dtype:
                        raise ValueError("swap weight dimensions/dtype differ from architecture")
                    state[name] = value
                if _state_hash(state) != protocol["training"][arm]["selected_state_sha256"]:
                    raise ValueError("swap selected state hash differs")
                initial_state = _state_arrays(model)
                for name, value in state.items():
                    if name.startswith(("stats_a.", "stats_b.")):
                        initial_state[name] = value.copy()
                if _state_hash(initial_state) != protocol["training"][arm]["initial_state_sha256"]:
                    raise ValueError("swap initialization differs from seed and architecture")
                model.load_state_dict({name: torch.from_numpy(value.copy()) for name, value in state.items()}, strict=True)
                model.eval().requires_grad_(False)
                if not bool(model.stats_a.fitted) or not bool(model.stats_b.fitted):
                    raise ValueError("swap normalization must be fitted")
                self.models[arm] = model
            if set(arrays) != required:
                raise ValueError("unexpected swap parameter fields")
            for domain, dimension, stats_name in (("math", dims[0], "stats_a"), ("language", dims[1], "stats_b")):
                mean, scale = arrays[f"{domain}_mean"], arrays[f"{domain}_scale"]
                if mean.shape != (dimension,) or scale.shape != mean.shape or np.any(scale <= 0):
                    raise ValueError("invalid swap normalization dimensions or scales")
                for model in self.models.values():
                    stats = getattr(model, stats_name)
                    if not np.array_equal(mean, stats.mean.numpy()) or not np.array_equal(scale, stats.scale.numpy()):
                        raise ValueError("swap model statistics differ across arms or metadata")
                setattr(self, f"{domain}_scaler", Standardizer(mean.copy(), scale.copy()))
            return self
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid swap bundle: {directory}") from exc

    @staticmethod
    def _validate_protocol(protocol, config):
        if (protocol.get("schema") != "lemmaferry.swap.v1" or protocol.get("backend") != "swap_autoencoder"
                or protocol.get("translator") != "swap" or protocol.get("objective_mode") != "standardized_multiloss"):
            raise ValueError("unsupported swap protocol")
        seed = protocol["seed"]
        if type(seed) is not int or not 0 <= seed < 2**63 - 20000:
            raise ValueError("invalid swap seed")
        if (protocol.get("optimizer") != {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8}
                or protocol.get("refit_after_selection") is not False):
            raise ValueError("unsupported swap optimizer or checkpoint refit protocol")
        for label, split in (("train", "train"), ("validation", "val")):
            info = protocol[label]
            count = info["count"]
            if type(count) is not int or count < (2 if split == "train" else 1) or info["split"] != split:
                raise ValueError("invalid swap split count or role")
            if len(info["ids"]) != count or len(set(info["ids"])) != count or len(info["families"]) != count:
                raise ValueError("invalid swap paired IDs or families")
            if any(not isinstance(v, str) or not v.strip() for key in ("ids", "families") for v in info[key]):
                raise ValueError("invalid swap identity labels")
            for key in ("math_dimension", "language_dimension"):
                if type(info[key]) is not int or info[key] < 1:
                    raise ValueError("invalid swap dimensions")
            for key in ("math_vectors_sha256", "language_vectors_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", info[key]):
                    raise ValueError("invalid swap activation fingerprint")
        _disjoint(protocol["train"], protocol["validation"])
        if any(protocol["train"][key] != protocol["validation"][key] for key in ("math_dimension", "language_dimension")):
            raise ValueError("swap dimensions change across splits")
        count = protocol["train"]["count"]
        if protocol["shuffled_train_permutation"] != shuffled_permutation(count, seed).tolist():
            raise ValueError("swap permutation is not the declared seeded derangement")
        validation_steps = [step for step in range(1, config.steps + 1)
                            if step % config.validation_every == 0 or step == config.steps]
        for arm in ("paired", "shuffled"):
            training = protocol["training"][arm]
            if (training["updates"] != config.steps or training["pair_exposures"] != config.steps * config.batch_size
                    or training["validation_steps"] != validation_steps
                    or training["batch_order_sha256"] != _schedule_hash(count, config, seed)):
                raise ValueError("swap training budget or schedule differs from config")
            trace = training["trace"]
            if [row["step"] for row in trace] != validation_steps:
                raise ValueError("swap validation trace differs from schedule")
            if any(not np.isfinite(row["validation_bidirectional_normalized_mse"]) or row["validation_bidirectional_normalized_mse"] < 0
                   or not np.isfinite(row["training_total"]) or row["training_total"] < 0 for row in trace):
                raise ValueError("nonfinite swap selection trace")
            best = min(trace, key=lambda row: (row["validation_bidirectional_normalized_mse"], row["step"]))
            if training["selected_step"] != best["step"]:
                raise ValueError("swap checkpoint was not selected by validation minimum")
            for key in ("initial_state_sha256", "selected_state_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", training[key]):
                    raise ValueError("invalid swap state hash")
        if protocol["training"]["paired"]["initial_state_sha256"] != protocol["training"]["shuffled"]["initial_state_sha256"]:
            raise ValueError("swap controls do not share initialization")
