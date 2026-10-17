#!/usr/bin/env python3
"""Independently check run hashes, replay Lean, and reload every recipient arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def verify_run(directory: Path, *, device: str = "cpu") -> dict:
    import numpy as np
    import torch
    from lemmaferry.learning import (evaluate, load_evaluation, load_recipient,
                                    parameter_sha256, represent, useful_control)
    from lemmaferry.library import VerifiedLibrary
    from lemmaferry.rosetta import RosettaPair
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
            pair = RosettaPair.load(folder / "rosetta/maps")
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
                if protocol["config"]["experience_mode"] == "translated":
                    targets = pair.to_language(arrays["math"])
                    if any(not np.allclose(row["translated_target"], targets[indices[row["id"]]], atol=1e-8, rtol=1e-8)
                           for row in experience["verified"]):
                        raise ValueError("training targets were not produced by the selected translator")
                    for arm, expected_targets in (("shuffled_control", pair.maps["T_shuffled"].predict(arrays["math"])),
                                                  ("mean_control", np.broadcast_to(pair.language_scaler.mean, targets.shape))):
                        for row in experience[arm]:
                            identifier = row["provenance"]["matched_to"]
                            source = next(r for r in experience["verified"] if r["id"] == identifier)
                            if (row["prompt"] != source["prompt"] or row["answer"] != source["answer"]
                                    or not np.allclose(row["translated_target"], expected_targets[indices[identifier]], atol=1e-8, rtol=1e-8)):
                                raise ValueError("correspondence control differs from its declared mapping")
                    text_rows = {r["provenance"]["matched_to"]: r for r in experience["text_control"]}
                    for row in experience["verified"]:
                        expected = "Verified mathematical information:\n" + formal[row["id"]] + "\n" + row["prompt"]
                        if text_rows[row["id"]]["prompt"] != expected or text_rows[row["id"]]["answer"] != row["answer"]:
                            raise ValueError("text baseline does not receive exactly the mathematical donor information")
    return {"status": "passed", "hashed_files_checked": len(hashes), "real_lean_replay": True,
            "verified_library_records": len(records), "iterations": len(result["iterations"]),
            "fixed_evaluation_sha256": evaluation.sha256, "standalone_checkpoint_reloads": reloads,
            "evaluation_feedback_detected": False, "training_budgets_matched": True}


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
