"""Bounded, auditable verified-generation / translation / learning cycles.

This implements the experimental protocol, not the conjecture that it will
produce sustained capability gains. External evaluation never controls a cycle.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass(frozen=True)
class LoopConfig:
    iterations: int = 2
    proposals_per_iteration: int = 6
    seed: int = 1729
    backend: str = "tiny"
    experience_mode: str = "language"
    representations: bool = True
    steps: int = 2
    batch_size: int = 1
    max_length: int = 2048
    learning_rate: float = 0.001
    auxiliary_weight: float = 0.1
    mastery_threshold: float = 0.7
    recipient_model: str | None = None
    recipient_revision: str | None = None
    math_model: str | None = None
    math_revision: str | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.iterations, self.steps, self.batch_size, self.max_length) < 1:
            raise ValueError("iteration and training budgets must be positive")
        if self.proposals_per_iteration < 6:
            raise ValueError("at least six proposals are needed for structural calibration splits")
        if self.backend not in {"tiny", "pretrained"}:
            raise ValueError("backend must be tiny or pretrained")
        if self.experience_mode not in {"language", "translated"}:
            raise ValueError("experience_mode must be language or translated")
        if self.experience_mode == "translated" and not self.representations:
            raise ValueError("translated experience requires paired representations")
        if self.learning_rate <= 0 or self.auxiliary_weight <= 0 or self.mastery_threshold < 0:
            raise ValueError("invalid learning rate, auxiliary weight, or mastery threshold")
        if self.backend == "pretrained":
            import re
            required = [(self.recipient_model, self.recipient_revision)]
            if self.representations:
                required.append((self.math_model, self.math_revision))
            if any(not name or not revision or not re.fullmatch(r"[0-9a-f]{40}", revision)
                   for name, revision in required):
                raise ValueError("pretrained models require explicit model names and immutable revisions")


def _recipient(config: LoopConfig, *, mathematical: bool = False, shared_base_from=None):
    from .learning import HFRecipient, TinyRecipient
    seed = config.seed + (1 if mathematical else 0)
    if config.backend == "tiny":
        return TinyRecipient(seed=seed, max_length=config.max_length, device=config.device)
    return HFRecipient(model_id=config.math_model if mathematical else config.recipient_model,
                       revision=config.math_revision if mathematical else config.recipient_revision,
                       seed=seed, max_length=config.max_length, device=config.device,
                       local_files_only=True, shared_base_from=shared_base_from)


def _paired_comparison(treatment: dict, control: dict, *, seed: int) -> dict:
    """Fixed-example paired bootstrap; small fixture intervals are not broad claims."""
    import numpy as np
    left, right = treatment["records"], control["records"]
    if [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("evaluation examples differ across treatment and control")
    differences = np.asarray([a["answer_nll"] - b["answer_nll"] for a, b in zip(left, right)])
    rng = np.random.default_rng(seed)
    replicates = differences[rng.integers(0, len(left), size=(2000, len(left)))].mean(axis=1)
    return {
        "risk_difference_treatment_minus_control": float(differences.mean()),
        "risk_difference_bootstrap_95": np.quantile(replicates, [0.025, 0.975]).tolist(),
        "accuracy_difference_treatment_minus_control": treatment["accuracy"] - control["accuracy"],
        "interpretation": "Negative risk difference favors treatment; this interval conditions on this fixed small task set.",
    }


def run_loop(output: Path | str, evaluation: Path | str, *,
             config: LoopConfig | None = None, lean: Path | str | None = None) -> dict:
    """Execute finite registered cycles. Reusing a directory cannot reset history.

    A tiny run is an engineering fixture. A pretrained run uses only explicitly
    pinned, already-local checkpoints and updates a general-model adapter. All
    new theorems are admitted by the same real Lean gate in either mode.
    """
    from .generation import propose
    from .learning import (TrainingBudget, build_experience, evaluate, freeze_evaluation,
                           parameter_sha256, save_recipient, score_candidates,
                           train_update, useful_control)
    from .library import Foundation, VerifiedLibrary

    config = config or LoopConfig()
    output, evaluation = Path(output), Path(evaluation)
    if output.exists():
        raise ValueError("choose a fresh output directory; existing libraries and checkpoints are preserved")
    source_rows = [json.loads(line) for line in evaluation.read_text().splitlines() if line.strip()]
    output.mkdir(parents=True)
    frozen = freeze_evaluation(output / "evaluation.json", source_rows)
    foundation = Foundation()
    library = VerifiedLibrary(output / "library", foundation)
    protocol = {
        "schema": "lemmaferry.constructive-loop.v1", "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "foundation": asdict(foundation),
        "initial_library": library.snapshot(), "evaluation_source_sha256": _hash(evaluation),
        "fixed_evaluation_sha256": frozen.sha256,
        "risk": "mean negative log likelihood of the complete declared answer, including EOS",
        "training_control": "correct ordinary-language factual supervision; matched steps, padded tokens, answer tokens and initialization",
        "translated_control": "same formal donor information supplied as text, with an equally computed recipient-state auxiliary loss",
        "correspondence_controls": "identical verified prompts and answer supervision, with shuffled-pair or training-mean auxiliary targets",
        "feedback_policy": "only post-update loss on accepted training examples; external evaluation is never a curriculum input",
        "representation_policy": "separate formal phi,p and natural-language derivation r; family-held-out calibration; validation-only lambda selection",
        "iteration_policy": "fixed budget, no evaluation-based early stopping, layer selection, or iterative success claim",
        "source_sha256": {p.name: _hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "scope": ("tiny randomly initialized neural fixtures; no specialist or general-purpose capability claim"
                  if config.backend == "tiny" else "local pinned pretrained specialist and recipient; recipient residual-adapter updates"),
    }
    _write(output / "protocol.json", protocol)
    summary: dict[str, Any] = {
        "schema": protocol["schema"], "status": "started", "protocol_sha256": _hash(output / "protocol.json"),
        "fixed_evaluation_sha256": frozen.sha256, "scope": protocol["scope"], "iterations": [],
        "claim": "Mechanism execution only. Increasingly broad or sustained reasoning gains remain an experimental conjecture.",
    }
    try:
        verified = _recipient(config)
        shared = verified if config.backend == "pretrained" else None
        arms = {"verified": verified, "ordinary_control": _recipient(config, shared_base_from=shared)}
        if config.experience_mode == "translated":
            for name in ("text_control", "shuffled_control", "mean_control"):
                arms[name] = _recipient(config, shared_base_from=shared)
        initial_hashes = {name: parameter_sha256(model) for name, model in arms.items()}
        if len(set(initial_hashes.values())) != 1:
            raise ValueError("treatment and controls must start with identical trainable weights")
        summary["initial_parameter_sha256"] = initial_hashes
        summary["initial_checkpoints"] = {
            name: save_recipient(model, output / "initial" / f"{name}.pt")
            for name, model in arms.items()
        }
        # Initial measurement is recorded, never inspected by proposal/selection logic.
        summary["initial_evaluation"] = evaluate(arms["verified"], frozen)
        source_model = _recipient(config, mathematical=True) if config.representations else None
        if source_model is not None:
            summary["source_checkpoint"] = save_recipient(source_model, output / "source-initial.pt")
        budget = TrainingBudget(steps=config.steps, batch_size=config.batch_size,
                                max_length=config.max_length, learning_rate=config.learning_rate,
                                auxiliary_weight=config.auxiliary_weight if config.experience_mode == "translated" else 0.0)
        feedback = None
        for iteration in range(config.iterations):
            directory = output / f"iteration-{iteration:03d}"
            directory.mkdir()
            before = library.snapshot()
            candidates = propose(library, iteration=iteration, count=config.proposals_per_iteration,
                                 seed=config.seed, feedback=feedback)
            _write(directory / "proposals.json", candidates)
            admissions = [library.accept(candidate, lean=lean) for candidate in candidates]
            _write(directory / "admissions.json", admissions)
            if any(row["status"] != "accepted" for row in admissions):
                reasons = [str(row.get("reason")) for row in admissions if row["status"] != "accepted"]
                raise ValueError("Lean admission or structural novelty failed: " + ", ".join(reasons))
            accepted = [row["record"] for row in admissions]
            after = library.snapshot()
            if after["record_count"] != before["record_count"] + len(accepted):
                raise ValueError("library did not grow by exactly the admitted records")
            experience = build_experience(accepted, library_hash=after["snapshot_hash"])
            controls = useful_control(experience)
            by_arm = {"verified": experience, "ordinary_control": controls}
            calibration = None
            if source_model is not None:
                from .representations import fit_library_rosetta
                fitted = fit_library_rosetta(library.records(), after["snapshot_hash"], source_model,
                                             arms["verified"], directory / "rosetta", seed=config.seed)
                calibration = fitted["report"]
                if config.experience_mode == "translated":
                    for row in experience:
                        row["translated_target"] = fitted["translated_targets"][row["id"]]
                    for name, targets in (("shuffled_control", "shuffled_targets"), ("mean_control", "mean_targets")):
                        rows = []
                        for row in experience:
                            item = dict(row)
                            item["id"] = name + ":" + row["id"]
                            item["translated_target"] = fitted[targets][row["id"]]
                            item["provenance"] = {"use": name, "matched_to": row["id"]}
                            rows.append(item)
                        by_arm[name] = rows
                    # Exact same donor information is available through this text arm.
                    pair_manifest = json.loads((directory / "rosetta" / "views" / "manifest.json").read_text())
                    donor_text = {row["id"]: row["formal_text"] for part in pair_manifest["splits"].values()
                                  for row in part["pairs"]}
                    text_rows = []
                    for row in experience:
                        text = dict(row)
                        text["id"] = "text-control:" + row["id"]
                        text["prompt"] = "Verified mathematical information:\n" + donor_text[row["id"]] + "\n" + row["prompt"]
                        text.pop("translated_target", None)
                        text["provenance"] = {"use": "same_information_text_control", "matched_to": row["id"]}
                        text_rows.append(text)
                    by_arm["text_control"] = text_rows
                    # Both controls retain their own pre-update states as
                    # auxiliary targets, with the same loss operations.
                    from .learning import represent
                    for control_name in ("ordinary_control", "text_control"):
                        rows = by_arm[control_name]
                        own_states = represent(arms[control_name], [r["prompt"] for r in rows])
                        for row, state in zip(rows, own_states):
                            row["translated_target"] = state.tolist()
            _write(directory / "experience.json", by_arm)
            training = {name: train_update(arms[name], rows, budget=budget,
                                          fixed_evaluation=frozen, seed=config.seed + iteration)
                        for name, rows in by_arm.items()}
            for key in ("updates_this_iteration", "cumulative_updates", "processed_padded_tokens",
                        "supervised_answer_tokens", "examples_seen"):
                if len({r[key] for r in training.values()}) != 1:
                    raise ValueError("training budget differs across arms: " + key)
            scores = score_candidates(arms["verified"], experience, fixed_evaluation=frozen)
            family_by_id = {r["id"]: r["family"] for r in experience}
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in scores:
                grouped[family_by_id[row["id"]]].append(row["score"])
            feedback = {
                "source": "training_curriculum_only", "iteration": iteration,
                "family_losses": {family: sum(values) / len(values) for family, values in sorted(grouped.items())},
                "mastery_threshold": config.mastery_threshold,
                "recipient_parameter_sha256": parameter_sha256(arms["verified"]),
                "external_evaluation_used": False,
            }
            checkpoints = {name: save_recipient(model, directory / f"{name}.pt") for name, model in arms.items()}
            # Report P after the update; nothing in this result enters feedback.
            measurements = {name: evaluate(model, frozen) for name, model in arms.items()}
            comparisons = {name: _paired_comparison(measurements["verified"], measured, seed=config.seed)
                           for name, measured in measurements.items() if name != "verified"}
            record = {"iteration": iteration, "library_before": before, "library_after": after,
                      "admitted": len(accepted), "dependency_reusing_records": sum(bool(r["dependencies"]) for r in accepted),
                      "training": training, "evaluation": measurements, "comparisons": comparisons,
                      "calibration": calibration, "feedback_for_next_generation": feedback,
                      "checkpoints": checkpoints}
            _write(directory / "result.json", record)
            summary["iterations"].append(record)
            _write(output / "summary.json", summary)
        frozen.verify()
        summary.update(status="completed", library=library.snapshot(),
                       completed_iterations=len(summary["iterations"]), guided_iterations=max(0, config.iterations - 1),
                       final_comparisons=summary["iterations"][-1]["comparisons"],
                       evaluation_used_for_generation=False,
                       sustained_broad_improvement_established=False)
        _write(output / "summary.json", summary)
        files = {str(p.relative_to(output)): _hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
        _write(output / "artifact-sha256.json", files)
        return summary
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write(output / "summary.json", summary)
        raise
