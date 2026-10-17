#!/usr/bin/env python3
"""Review stored Qwen numerical evidence without instantiating pretrained models.

This does not re-extract activations or independently measure model answers.
The full run verifier performs those separate model-dependent checks.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lemmaferry.correspondence import load_correspondence
from lemmaferry.library import VerifiedLibrary
from lemmaferry.rosetta import PairedSplit, RosettaPair

SPEC = importlib.util.spec_from_file_location("_independent_tiny_review", Path(__file__).with_name("independent-review.py"))
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(checkpoint):
    result = hashlib.sha256()
    for name, value in checkpoint["trainable_state_dict"].items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    torch.set_num_threads(1)
    run = ROOT / "runs/constructive-loop-qwen-v2"
    summary = json.loads((run / "summary.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    assert summary["status"] == "completed" and protocol["config"]["backend"] == "pretrained"
    hashes = json.loads((run / "artifact-sha256.json").read_text())
    assert set(hashes) == {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()} - {"artifact-sha256.json"}
    for name, expected in hashes.items():
        assert digest(run / name) == expected, name
    assert digest(run / "protocol.json") == summary["protocol_sha256"]
    source_hashes_match = {name: digest(ROOT / "src/lemmaferry" / name) == value for name, value in protocol["source_sha256"].items()}
    assert all(source_hashes_match.values())
    library = VerifiedLibrary(run / "library")
    records = library.records()
    assert library.snapshot() == summary["library"]
    assert not summary["evaluation_used_for_generation"]
    assert not summary["sustained_broad_improvement_established"]
    assert len(set(summary["initial_parameter_sha256"].values())) == 1
    findings = []
    previous = summary["initial_parameter_sha256"].copy()
    for iteration in summary["iterations"]:
        index = iteration["iteration"]
        folder = run / f"iteration-{index:03d}"
        pair = RosettaPair.load(folder / "rosetta/maps")
        manifest = load_correspondence(folder / "rosetta/views")
        with np.load(folder / "rosetta/activations.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        ids, families = arrays["ids"].tolist(), arrays["families"].tolist()
        positions = {identifier: i for i, identifier in enumerate(ids)}
        family_splits = pair.protocol["extraction"]["family_split"]
        pairs = {}
        for split in ("train", "val", "test"):
            indices = [i for i, family in enumerate(families) if family_splits[family] == split]
            pairs[split] = PairedSplit(arrays["math"][indices], arrays["language"][indices],
                                      [ids[i] for i in indices], [families[i] for i in indices], split)
        refit = RosettaPair.fit(pairs["train"], pairs["val"], pair.protocol["lambda_grid"],
                                objective=pair.protocol["objective_mode"], seed=pair.protocol["seed"])
        for name, fitted in refit.maps.items():
            np.testing.assert_allclose(fitted.coef, pair.maps[name].coef, atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(fitted.intercept, pair.maps[name].intercept, atol=1e-8, rtol=1e-8)
        geometry = pair.evaluate(pairs["test"])
        REVIEW.compare(geometry["test_metrics"], iteration["calibration"]["test_metrics"])
        REVIEW.compare(geometry["round_trip"], iteration["calibration"]["round_trip"])
        experience = json.loads((folder / "experience.json").read_text())
        originals = {row["id"]: row for row in experience["verified"]}
        donor_text = {row["id"]: row["formal_text"] for part in manifest["splits"].values() for row in part["pairs"]}
        predictions = pair.to_language(arrays["math"])
        for row in experience["verified"]:
            np.testing.assert_allclose(row["translated_target"], predictions[positions[row["id"]]], atol=1e-9, rtol=1e-9)
        for arm, targets in (("shuffled_control", pair.maps["T_shuffled"].predict(arrays["math"])),
                             ("mean_control", np.broadcast_to(pair.language_scaler.mean, predictions.shape))):
            for row in experience[arm]:
                identifier = row["provenance"]["matched_to"]
                original = originals[identifier]
                assert all(row[key] == original[key] for key in ("prompt", "answer", "choices"))
                np.testing.assert_allclose(row["translated_target"], targets[positions[identifier]], atol=1e-9, rtol=1e-9)
        for row in experience["text_control"]:
            identifier = row["provenance"]["matched_to"]
            assert row["prompt"] == "Verified mathematical information:\n" + donor_text[identifier] + "\n" + originals[identifier]["prompt"]
            assert row["answer"] == originals[identifier]["answer"]
        for arm, training in iteration["training"].items():
            assert training["parameter_sha256_before"] == previous[arm]
            checkpoint_path = folder / f"{arm}.pt"
            assert digest(checkpoint_path) == iteration["checkpoints"][arm]["sha256"]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            assert checkpoint["backend"] == "huggingface_residual_adapter"
            assert tensor_hash(checkpoint) == checkpoint["parameter_sha256"] == training["parameter_sha256_after"]
            assert checkpoint["updates"] == training["cumulative_updates"]
            optimizer_steps = {int(value["step"]) for value in checkpoint["optimizer_state_dict"]["state"].values()}
            assert optimizer_steps == {checkpoint["updates"]}
            assert checkpoint["config"]["model_id"] == protocol["config"]["recipient_model"]
            assert checkpoint["config"]["revision"] == protocol["config"]["recipient_revision"]
            previous[arm] = checkpoint["parameter_sha256"]
        budgets = ("updates_this_iteration", "cumulative_updates", "processed_padded_tokens", "supervised_answer_tokens", "examples_seen")
        assert all(len({value[field] for value in iteration["training"].values()}) == 1 for field in budgets)
        feedback = iteration["feedback_for_next_generation"]
        assert feedback["source"] == "training_curriculum_only" and not feedback["external_evaluation_used"]
        assert feedback["recipient_parameter_sha256"] == previous["verified"]
        if index:
            proposals = json.loads((folder / "proposals.json").read_text())
            assert all(p["provenance"]["curriculum_feedback"] == summary["iterations"][index-1]["feedback_for_next_generation"] for p in proposals)
        findings.append({
            "iteration": index, "reported_risks": {arm: data["risk_mean_answer_nll"] for arm, data in iteration["evaluation"].items()},
            "reported_accuracies": {arm: data["accuracy"] for arm, data in iteration["evaluation"].items()},
            "T_S_and_shuffled_maps_refit_from_raw_arrays": True,
            "translated_shuffled_and_mean_targets_recomputed": True,
            "identical_prompt_and_label_correspondence_controls": True, "exact_donor_text_control": True,
            "adapter_tensor_hashes_and_optimizer_steps_verified": True,
            "round_trip_mse": {name: geometry["round_trip"][name]["mse"] for name in ("S_of_T_math", "T_of_S_language")},
        })
    report = {
        "status": "passed_limited_numeric_review", "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "independent Rosetta implementation agent; not human review", "run": str(run.relative_to(ROOT)),
        "command": [sys.executable, str(Path(__file__).relative_to(ROOT))],
        "hashed_files_checked": len(hashes), "complete_hash_keyset_checked": True,
        "verified_library_records_real_Lean_replayed": len(records), "source_hashes_match_execution": source_hashes_match,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__), ROOT / "src/lemmaferry/rosetta.py", ROOT / "src/lemmaferry/loop.py", ROOT / "src/lemmaferry/learning.py"]},
        "run_artifact_manifest_sha256": digest(run / "artifact-sha256.json"), "iterations": findings,
        "not_independently_recomputed": ["pretrained model activation extraction", "ordinary/text self-state auxiliary targets", "capability answer NLL and predictions", "training-example feedback loss values"],
        "scope": "Stored numeric and provenance consistency only. No pretrained models were instantiated. The separate full verifier is required for model-dependent checks.",
        "history": "The earlier three-arm Qwen run was exploratory; the five-arm follow-up adds shuffled-pair and mean-target functional controls without tuning layers, regularization grid, tasks, or success thresholds.",
        "claim": "No semantic correspondence, useful transfer, or sustained broad reasoning improvement is established by this review.",
    }
    output = ROOT / "artifacts/constructive-loop/qwen-independent-numeric-review.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(output.relative_to(ROOT)), "iterations": findings}, indent=2))


if __name__ == "__main__":
    main()
