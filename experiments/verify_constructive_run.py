#!/usr/bin/env python3
"""Independently check run hashes, replay Lean, and reload every recipient arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _target_tolerances(pair) -> tuple[float, float]:
    """Keep affine float64 checks; allow 64 float32 eps for neural replay.

    The neural bundle stores float32 parameters and normalization buffers. CPU
    versus accelerator GEMM/GELU evaluation can differ by a few ULPs; 64 eps
    covers that rounding without accepting meaningful target changes. This is
    also the tolerance for re-extracting float32 recipient states. It never
    relaxes the historical affine tolerance.
    """
    if getattr(pair, "backend", None) == "swap_autoencoder":
        import numpy as np
        epsilon = float(np.finfo(np.float32).eps) * 64
        return epsilon, epsilon
    return 1e-8, 1e-8


def _registered_swap_config(config: dict) -> dict:
    from dataclasses import asdict
    from lemmaferry.translators import SwapConfig
    names = ("latent_dim", "hidden_dim", "steps", "batch_size", "learning_rate", "validation_every")
    # Bridge optimization is registered on CPU independently of the recipient
    # extraction device; recipient MPS/CUDA does not change this bridge budget.
    return asdict(SwapConfig(**{name: config["swap_" + name] for name in names}, device="cpu"))


def _verify_swap_calibration(pair, manifest, arrays, iteration, protocol, by_id):
    """Bind a neural bundle to registered settings and the accepted data.

    Unlike envelope hashes, these checks also reject self-consistent edits to a
    bundle's manifest, row labels, pairing permutation or training statistics.
    Actual model extraction is checked separately, so rewriting array hashes
    cannot make fabricated activations into model evidence.
    """
    import numpy as np
    from lemmaferry.representations import _canonical, _language_view
    from lemmaferry.rosetta import PairedSplit

    if protocol["config"].get("translator") != "swap":
        raise ValueError("neural translator differs from the registered backend")
    if pair.protocol["config"] != _registered_swap_config(protocol["config"]):
        raise ValueError("neural translator configuration differs from registration")
    if pair.protocol["seed"] != protocol["config"]["seed"]:
        raise ValueError("neural translator seed differs from registration")
    if pair.protocol["source_sha256"] != protocol["source_sha256"]["swap.py"]:
        raise ValueError("neural implementation provenance differs from registration")
    identifiers = [str(value) for value in arrays["ids"]]
    if identifiers != iteration["library_after"]["record_ids"]:
        raise ValueError("neural activation row order differs from the library snapshot")
    known = {identifier: by_id[identifier] for identifier in identifiers}
    families = [known[identifier]["family"] for identifier in identifiers]
    if list(arrays["families"]) != families:
        raise ValueError("neural activation families differ from accepted records")
    family_split = {family: ("train", "val", "test")[i % 3]
                    for i, family in enumerate(sorted(set(families)))}
    extraction = pair.protocol["extraction"]
    if extraction["family_split"] != family_split:
        raise ValueError("neural family split differs from the accepted curriculum")
    expected_ids = set(identifiers)
    actual_ids = set()
    splits = {}
    for split in ("train", "val", "test"):
        rows = manifest["splits"][split]["pairs"]
        indices = [i for i, family in enumerate(families) if family_split[family] == split]
        if [row["id"] for row in rows] != [identifiers[i] for i in indices]:
            raise ValueError("neural correspondence rows differ from the accepted family split")
        for row in rows:
            record = known[row["id"]]
            provenance = row["provenance"]
            if (row["family"] != record["family"] or row["formal_text"] != record["formal_text"]
                    or row["language_text"] != _language_view(record, known)
                    or provenance["record_sha256"] != hashlib.sha256(_canonical(record)).hexdigest()
                    or provenance["record_provenance"] != record.get("provenance", {})
                    or provenance["library_snapshot_sha256"] != iteration["library_after"]["snapshot_hash"]):
                raise ValueError("neural correspondence provenance differs from the accepted theorem")
            actual_ids.add(row["id"])
        splits[split] = PairedSplit(arrays["math"][indices], arrays["language"][indices],
                                    [identifiers[i] for i in indices],
                                    [families[i] for i in indices], split)
    if actual_ids != expected_ids:
        raise ValueError("neural correspondences do not cover the library snapshot")
    for name, split in (("train", "train"), ("validation", "val")):
        if pair.protocol[name] != splits[split].fingerprint():
            raise ValueError("neural calibration activations differ from the fitted split fingerprint")
    rng = np.random.default_rng(pair.protocol["seed"] + 10000)
    for _ in range(100):
        permutation = rng.permutation(len(splits["train"].ids))
        if np.all(permutation != np.arange(len(permutation))):
            break
    else:
        permutation = np.roll(np.arange(len(splits["train"].ids)), 1)
    if pair.protocol["shuffled_train_permutation"] != permutation.tolist():
        raise ValueError("neural pairing permutation differs from the registered seed")
    atol, rtol = _target_tolerances(pair)
    for name, values in (("math", splits["train"].x), ("language", splits["train"].y)):
        values = np.asarray(values, dtype=np.float32)
        mean = values.mean(axis=0, dtype=np.float32)
        scale = values.std(axis=0, dtype=np.float32)
        scale = np.where(scale > 1e-6, scale, np.ones_like(scale))
        scaler = getattr(pair, name + "_scaler")
        if not (np.allclose(scaler.mean, mean, atol=atol, rtol=rtol)
                and np.allclose(scaler.scale, scale, atol=atol, rtol=rtol)):
            raise ValueError("neural normalization is not fitted exclusively on training activations")
    pair.validate_training_data(splits["train"], splits["val"])
    # A selected-step label and a self-consistent trace are insufficient: the
    # retained weights must actually attain that recorded validation score.
    validation = splits["val"]
    for control in ("paired", "shuffled"):
        forward = (pair.to_language(validation.x, control=control) - validation.y) / pair.language_scaler.scale
        reverse = (pair.to_math(validation.y, control=control) - validation.x) / pair.math_scaler.scale
        score = float(np.mean(forward ** 2) + np.mean(reverse ** 2))
        training = pair.protocol["training"][control]
        selected = next(row for row in training["trace"] if row["step"] == training["selected_step"])
        if not np.isclose(score, selected["validation_bidirectional_normalized_mse"], atol=atol, rtol=rtol):
            raise ValueError("neural selected checkpoint does not reproduce its validation score")
    return splits["test"]


def _check_neural_report(measured, reported, atol, rtol) -> bool:
    """Compare the held-out numeric report without relaxing counts or labels."""
    import numpy as np
    if isinstance(measured, dict):
        return (isinstance(reported, dict) and set(measured) == set(reported)
                and all(_check_neural_report(value, reported[key], atol, rtol) for key, value in measured.items()))
    if isinstance(measured, list):
        return (isinstance(reported, list) and len(measured) == len(reported)
                and all(_check_neural_report(a, b, atol, rtol) for a, b in zip(measured, reported)))
    if type(measured) is float:
        return isinstance(reported, (int, float)) and np.isfinite(reported) and bool(np.isclose(measured, reported, atol=atol, rtol=rtol))
    return type(measured) is type(reported) and measured == reported


def _replay_swap_extraction(directory, pair, manifest, arrays, iteration, protocol, result, device) -> None:
    import numpy as np
    import torch
    from lemmaferry.learning import load_recipient, parameter_sha256, represent
    from lemmaferry.representations import _model_metadata

    source = Path(__file__).resolve().parents[1] / "src" / "lemmaferry"
    current_hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(source.rglob("*.py"))}
    if protocol["source_sha256"] != current_hashes:
        raise ValueError("neural replay source provenance differs from the registered recursive source hashes")
    extraction = pair.protocol["extraction"]
    if extraction.get("parameters_unchanged") is not True or extraction.get("timing") != "before the current recipient training update":
        raise ValueError("neural extraction is not bound to unchanged pre-update models")
    index = iteration["iteration"]
    paths = {"math": directory / "source-initial.pt",
             "language": directory / (f"iteration-{index-1:03d}/verified.pt" if index else "initial/verified.pt")}
    source_receipt = result["source_checkpoint"]
    if hashlib.sha256(paths["math"].read_bytes()).hexdigest() != source_receipt["sha256"]:
        raise ValueError("mathematical source checkpoint differs from its receipt")
    rows = {row["id"]: row for part in manifest["splits"].values() for row in part["pairs"]}
    atol, rtol = _target_tolerances(pair)
    for domain, field in (("math", "formal_text"), ("language", "language_text")):
        checkpoint = torch.load(paths[domain], map_location="cpu", weights_only=True)
        model = load_recipient(paths[domain], device=device)
        digest = parameter_sha256(model)
        if (extraction["trainable_parameter_sha256_before"][domain] != digest
                or extraction["trainable_parameter_sha256_after"][domain] != digest):
            raise ValueError("neural model parameter provenance differs from its checkpoint")
        metadata = _model_metadata(model)
        # Device is an execution choice; the saved metadata must still describe
        # the original checkpoint, even when replay explicitly requests CPU.
        metadata["config"] = checkpoint["config"]
        metadata["config_sha256"] = hashlib.sha256(json.dumps(checkpoint["config"], sort_keys=True,
                                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if metadata != extraction["models"][domain]:
            raise ValueError("neural extraction model configuration differs from its checkpoint")
        values = represent(model, [rows[str(identifier)][field] for identifier in arrays["ids"]])
        if values.shape != arrays[domain].shape or not np.allclose(values, arrays[domain], atol=atol, rtol=rtol):
            raise ValueError("neural saved activations do not reproduce from the pre-update model")
        if parameter_sha256(model) != digest:
            raise ValueError("neural replay changed extraction model parameters")
        del model


def verify_run(directory: Path, *, device: str = "cpu") -> dict:
    import numpy as np
    import torch
    from lemmaferry.learning import (evaluate, load_evaluation, load_recipient,
                                    parameter_sha256, represent, useful_control)
    from lemmaferry.library import VerifiedLibrary
    from lemmaferry.translators import load_translator
    from lemmaferry.correspondence import load_correspondence

    directory = directory.resolve()
    hashes = json.loads((directory / "artifact-sha256.json").read_text())
    actual_files = {str(p.relative_to(directory)) for p in directory.rglob("*")
                    if p.is_file() and p != directory / "artifact-sha256.json"}
    if set(hashes) != actual_files:
        raise ValueError("artifact inventory has missing or unexpected files")
    for name, expected in hashes.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory):
            raise ValueError("artifact path leaves the run directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("artifact checksum mismatch: " + name)
    result = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if result["status"] != "completed" or hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest() != result["protocol_sha256"]:
        raise ValueError("run incomplete or registered protocol changed")
    evaluation = load_evaluation(directory / "evaluation.json")
    if evaluation.sha256 != result["fixed_evaluation_sha256"] or evaluation.sha256 != protocol["fixed_evaluation_sha256"]:
        raise ValueError("independent evaluation differs from registration")
    if result["evaluation_used_for_generation"] or result["sustained_broad_improvement_established"]:
        raise ValueError("unsupported evaluation feedback or capability claim")
    library = VerifiedLibrary(directory / "library")  # fresh real Lean replay
    records = library.records()
    by_id = {r["id"]: r for r in records}
    if library.snapshot() != result["library"]:
        raise ValueError("cumulative library differs from final result")
    evaluation_ids = {r["id"] for r in evaluation.examples}
    evaluation_prompts = {r["prompt"] for r in evaluation.examples}
    previous = {}
    for arm, receipt in result["initial_checkpoints"].items():
        checkpoint = directory / "initial" / f"{arm}.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("initial checkpoint differs from its receipt")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        digest = hashlib.sha256()
        for name, value in state["trainable_state_dict"].items():
            digest.update(name.encode())
            digest.update(value.contiguous().numpy().tobytes())
        if (digest.hexdigest() != state["parameter_sha256"] or state["updates"] != 0
                or state["optimizer_state_dict"] is not None):
            raise ValueError("initial parameters or optimizer differ from registration")
        previous[arm] = state["parameter_sha256"]
    if len(set(previous.values())) != 1 or previous != result["initial_parameter_sha256"]:
        raise ValueError("training arms did not start with identical registered weights")
    reloads = []
    translator_replays = []
    prior_snapshot = protocol["initial_library"]
    for iteration in result["iterations"]:
        index = iteration["iteration"]
        folder = directory / f"iteration-{index:03d}"
        if iteration["library_before"] != prior_snapshot:
            raise ValueError("library snapshot history has a gap")
        admitted_ids = iteration["library_after"]["record_ids"][iteration["library_before"]["record_count"]:]
        if len(admitted_ids) != iteration["admitted"] or any(i not in by_id for i in admitted_ids):
            raise ValueError("iteration admission count differs from accepted library")
        prior_snapshot = iteration["library_after"]
        experience = json.loads((folder / "experience.json").read_text())
        if {r["id"] for r in experience["verified"]} != set(admitted_ids):
            raise ValueError("training experience is not bound to this iteration's verified admissions")
        feedback = iteration["feedback_for_next_generation"]
        if feedback["source"] != "training_curriculum_only" or feedback["external_evaluation_used"]:
            raise ValueError("generation feedback was not restricted to training curriculum")
        if index:
            proposals = json.loads((folder / "proposals.json").read_text())
            preceding = result["iterations"][index - 1]["feedback_for_next_generation"]
            if any(p["provenance"]["curriculum_feedback"] != preceding for p in proposals):
                raise ValueError("later generation did not use the recorded updated recipient")
        for key in ("processed_padded_tokens", "supervised_answer_tokens", "cumulative_updates", "examples_seen"):
            if len({v[key] for v in iteration["training"].values()}) != 1:
                raise ValueError("training budgets differ across arms")
        ordinary = {row["id"]: row for row in useful_control(experience["verified"])}
        if any(row["prompt"] != ordinary[row["id"]]["prompt"] or row["answer"] != ordinary[row["id"]]["answer"]
               for row in experience["ordinary_control"]):
            raise ValueError("ordinary control is not the registered true-supervision control")
        for arm, training in iteration["training"].items():
            if any(r["id"] in evaluation_ids or r["prompt"] in evaluation_prompts for r in experience[arm]):
                raise ValueError("independent evaluation leaked into training")
            if arm in previous and training["parameter_sha256_before"] != previous[arm]:
                raise ValueError("recipient parameters were reset between iterations")
            if protocol["config"].get("translator") == "swap" and protocol["config"]["experience_mode"] == "translated":
                targets = np.asarray([r["translated_target"] for r in experience[arm]], dtype=np.float32)
                target_hash = hashlib.sha256(targets.tobytes(order="C")).hexdigest()
                if (training.get("auxiliary_targets_sha256") != target_hash
                        or training.get("auxiliary_targets_detached") is not True):
                    raise ValueError("neural recipient target snapshot differs from the detached training receipt")
            previous[arm] = training["parameter_sha256_after"]
            checkpoint = folder / f"{arm}.pt"
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != iteration["checkpoints"][arm]["sha256"]:
                raise ValueError("checkpoint differs from execution record")
            model = load_recipient(checkpoint, device=device)
            if parameter_sha256(model) != previous[arm] or model.updates != training["cumulative_updates"]:
                raise ValueError("checkpoint did not preserve parameter or optimizer history")
            steps = {int(state["step"]) for state in model.optimizer.state.values() if "step" in state}
            if steps != {model.updates}:
                raise ValueError("AdamW moments were reset between iterations")
            measured = evaluate(model, evaluation)
            reported = iteration["evaluation"][arm]
            if ([r["prediction"] for r in measured["records"]] != [r["prediction"] for r in reported["records"]]
                    or abs(measured["risk_mean_answer_nll"] - reported["risk_mean_answer_nll"]) > 1e-4
                    or measured["accuracy"] != reported["accuracy"]
                    or any(abs(a["answer_nll"] - b["answer_nll"]) > 1e-3
                           for a, b in zip(measured["records"], reported["records"]))):
                raise ValueError("standalone checkpoint evaluation differs from report")
            if arm in {"ordinary_control", "text_control"} and protocol["config"]["experience_mode"] == "translated":
                predecessor = (directory / f"iteration-{index-1:03d}" / f"{arm}.pt" if index
                               else directory / "initial" / f"{arm}.pt")
                prior = load_recipient(predecessor, device=device)
                if parameter_sha256(prior) != training["parameter_sha256_before"]:
                    raise ValueError("control auxiliary target uses a different initial model")
                expected_targets = represent(prior, [r["prompt"] for r in experience[arm]])
                if not np.allclose(expected_targets, [r["translated_target"] for r in experience[arm]], atol=1e-4, rtol=1e-4):
                    raise ValueError("control auxiliary targets differ from their own pre-update context states")
                del prior
            reloads.append({"iteration": index, "arm": arm, "updates": model.updates,
                            "risk_mean_answer_nll": measured["risk_mean_answer_nll"], "accuracy": measured["accuracy"]})
            del model
        if iteration["calibration"] is not None:
            pair = load_translator(folder / "rosetta/maps")
            manifest = load_correspondence(folder / "rosetta/views/manifest.json")
            if manifest["library_snapshots_sha256"] != [iteration["library_after"]["snapshot_hash"]]:
                raise ValueError("Rosetta correspondences refer to a different library")
            if pair.protocol["extraction"]["library_snapshot_sha256"] != iteration["library_after"]["snapshot_hash"]:
                raise ValueError("Rosetta maps refer to a different library")
            formal = {r["id"]: r["formal_text"] for part in manifest["splits"].values() for r in part["pairs"]}
            with np.load(folder / "rosetta/activations.npz", allow_pickle=False) as arrays:
                indices = {str(identifier): i for i, identifier in enumerate(arrays["ids"])}
                if set(indices) != set(iteration["library_after"]["record_ids"]):
                    raise ValueError("Rosetta activations differ from the accepted library snapshot")
                neural = getattr(pair, "backend", None) == "swap_autoencoder"
                if protocol["config"].get("translator", "affine") == "swap" and not neural:
                    raise ValueError("saved translator differs from the registered neural backend")
                if neural:
                    saved_report = json.loads((folder / "rosetta/results.json").read_text())
                    if saved_report != iteration["calibration"] or pair.protocol != saved_report["protocol"]:
                        raise ValueError("neural calibration report differs from the saved bundle")
                    if saved_report["artifacts"]["activations_sha256"] != hashlib.sha256((folder / "rosetta/activations.npz").read_bytes()).hexdigest():
                        raise ValueError("neural activation archive differs from the calibration receipt")
                    test = _verify_swap_calibration(pair, manifest, arrays, iteration, protocol, by_id)
                    _replay_swap_extraction(directory, pair, manifest, arrays, iteration, protocol, result, device)
                    measured_report = pair.evaluate(test)
                    reported_geometry = {key: value for key, value in saved_report.items() if key != "artifacts"}
                    if not _check_neural_report(measured_report, reported_geometry, *_target_tolerances(pair)):
                        raise ValueError("neural held-out calibration report does not reproduce")
                atol, rtol = _target_tolerances(pair)
                translator_replays.append({"iteration": index, "backend": getattr(pair, "backend", "affine"),
                                           "atol": atol, "rtol": rtol, "activation_reextraction": neural})
                if protocol["config"]["experience_mode"] == "translated":
                    targets = pair.to_language(arrays["math"])
                    if any(not np.allclose(row["translated_target"], targets[indices[row["id"]]], atol=atol, rtol=rtol)
                           for row in experience["verified"]):
                        raise ValueError("training targets were not produced by the selected translator")
                    for arm, expected_targets in (("shuffled_control", pair.to_language(arrays["math"], control="shuffled")),
                                                  ("mean_control", pair.to_language(arrays["math"], control="mean"))):
                        for row in experience[arm]:
                            identifier = row["provenance"]["matched_to"]
                            source = next(r for r in experience["verified"] if r["id"] == identifier)
                            if (row["prompt"] != source["prompt"] or row["answer"] != source["answer"]
                                    or not np.allclose(row["translated_target"], expected_targets[indices[identifier]], atol=atol, rtol=rtol)):
                                raise ValueError("correspondence control differs from its declared mapping")
                    text_rows = {r["provenance"]["matched_to"]: r for r in experience["text_control"]}
                    for row in experience["verified"]:
                        expected = "Verified mathematical information:\n" + formal[row["id"]] + "\n" + row["prompt"]
                        if text_rows[row["id"]]["prompt"] != expected or text_rows[row["id"]]["answer"] != row["answer"]:
                            raise ValueError("text baseline does not receive exactly the mathematical donor information")
    return {"status": "passed", "hashed_files_checked": len(hashes), "real_lean_replay": True,
            "verified_library_records": len(records), "iterations": len(result["iterations"]),
            "fixed_evaluation_sha256": evaluation.sha256, "standalone_checkpoint_reloads": reloads,
            "evaluation_feedback_detected": False, "training_budgets_matched": True,
            "translator_replays": translator_replays}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.torch_threads)
    result = verify_run(args.directory, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
