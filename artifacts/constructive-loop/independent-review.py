#!/usr/bin/env python3
"""Recompute the small fixture's targets, calibration and feedback independently."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lemmaferry.correspondence import load_correspondence
from lemmaferry.learning import TinyRecipient, load_evaluation, load_recipient, parameter_sha256, represent, score_candidates
from lemmaferry.rosetta import PairedSplit, RosettaPair


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            compare(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            compare(left, right)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/constructive-loop-v3")
    parser.add_argument("--output", default="artifacts/constructive-loop/independent-review.json")
    parser.add_argument("--allow-historical-notes", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = ROOT / args.run
    summary = json.loads((run / "summary.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    assert protocol["config"]["backend"] == "tiny", "This review script intentionally loads only tiny fixtures"
    command = [sys.executable, "experiments/verify_constructive_run.py", str(run.relative_to(ROOT)), "--torch-threads", "1"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    verification = json.loads(completed.stdout)
    assert summary["status"] == "completed"
    hashes = json.loads((run / "artifact-sha256.json").read_text())
    expected_files = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()} - {"artifact-sha256.json"}
    extra_artifacts = sorted(expected_files - set(hashes))
    missing_artifacts = sorted(set(hashes) - expected_files)
    assert not missing_artifacts
    if not args.allow_historical_notes:
        assert not extra_artifacts
    recorded_source_match = {}
    for name, recorded in protocol["source_sha256"].items():
        recorded_source_match[name] = digest(ROOT / "src/lemmaferry" / name) == recorded
    if not args.allow_historical_notes:
        assert all(recorded_source_match.values())
    frozen = load_evaluation(run / "evaluation.json")
    findings = []
    for iteration in summary["iterations"]:
        index = iteration["iteration"]
        folder = run / f"iteration-{index:03d}"
        pair = RosettaPair.load(folder / "rosetta/maps")
        manifest = load_correspondence(folder / "rosetta/views")
        with np.load(folder / "rosetta/activations.npz", allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        ids = arrays["ids"].tolist()
        idx = {identifier: i for i, identifier in enumerate(ids)}
        families = arrays["families"].tolist()
        family_splits = pair.protocol["extraction"]["family_split"]
        split_data = {}
        for split in ("train", "val", "test"):
            indices = [i for i, family in enumerate(families) if family_splits[family] == split]
            split_data[split] = PairedSplit(arrays["math"][indices], arrays["language"][indices],
                                           [ids[i] for i in indices], [families[i] for i in indices], split)
        actual_metrics = pair.evaluate(split_data["test"])
        compare(actual_metrics["test_metrics"], iteration["calibration"]["test_metrics"])
        compare(actual_metrics["round_trip"], iteration["calibration"]["round_trip"])
        refit = RosettaPair.fit(split_data["train"], split_data["val"],
                                pair.protocol["lambda_grid"], objective=pair.protocol["objective_mode"],
                                seed=pair.protocol["seed"])
        for name in pair.maps:
            np.testing.assert_allclose(refit.maps[name].coef, pair.maps[name].coef, atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(refit.maps[name].intercept, pair.maps[name].intercept, atol=1e-9, rtol=1e-9)
        extraction = pair.protocol["extraction"]
        source = TinyRecipient(**extraction["models"]["math"]["config"])
        source_prompts = {row["id"]: row["formal_text"] for part in manifest["splits"].values() for row in part["pairs"]}
        recipient_prompts = {row["id"]: row["language_text"] for part in manifest["splits"].values() for row in part["pairs"]}
        source_recomputed = represent(source, [source_prompts[identifier] for identifier in ids])
        np.testing.assert_allclose(source_recomputed, arrays["math"], atol=1e-6, rtol=1e-6)
        assert parameter_sha256(source) == extraction["trainable_parameter_sha256_before"]["math"]
        del source
        experience = json.loads((folder / "experience.json").read_text())
        predictions = pair.to_language(arrays["math"])
        for row in experience["verified"]:
            np.testing.assert_allclose(row["translated_target"], predictions[idx[row["id"]]], atol=1e-10, rtol=1e-10)
        original_rows = {row["id"]: row for row in experience["verified"]}
        correspondence_controls = {}
        for arm, targets in (("shuffled_control", pair.maps["T_shuffled"].predict(arrays["math"])),
                             ("mean_control", np.broadcast_to(pair.language_scaler.mean, predictions.shape))):
            if arm not in experience:
                continue
            for row in experience[arm]:
                original_id = row["provenance"]["matched_to"]
                original = original_rows[original_id]
                assert row["prompt"] == original["prompt"]
                assert row["answer"] == original["answer"]
                assert row["choices"] == original["choices"]
                np.testing.assert_allclose(row["translated_target"], targets[idx[original_id]], atol=1e-10, rtol=1e-10)
            correspondence_controls[arm] = {"target_formula_verified": True, "identical_prompts_answers_and_choices": True}
        for row in experience["text_control"]:
            original_id = row["provenance"]["matched_to"]
            original = next(r for r in experience["verified"] if r["id"] == original_id)
            assert row["prompt"] == "Verified mathematical information:\n" + source_prompts[original_id] + "\n" + original["prompt"]
        for arm in iteration["training"]:
            model = load_recipient(run / (f"iteration-{index-1:03d}" if index else "initial") / f"{arm}.pt")
            before = iteration["training"][arm]["parameter_sha256_before"]
            assert parameter_sha256(model) == before
            if arm == "verified":
                recipient_recomputed = represent(model, [recipient_prompts[identifier] for identifier in ids])
                np.testing.assert_allclose(recipient_recomputed, arrays["language"], atol=1e-6, rtol=1e-6)
            elif arm in ("ordinary_control", "text_control"):
                states = represent(model, [r["prompt"] for r in experience[arm]])
                np.testing.assert_allclose(states, [r["translated_target"] for r in experience[arm]], atol=1e-6, rtol=1e-6)
            assert parameter_sha256(model) == before
            del model
        post_update = load_recipient(folder / "verified.pt")
        scores = score_candidates(post_update, experience["verified"], fixed_evaluation=frozen)
        by_id = {row["id"]: row for row in experience["verified"]}
        groups = defaultdict(list)
        for row in scores:
            groups[by_id[row["id"]]["family"]].append(row["score"])
        recalculated_feedback = {family: sum(values) / len(values) for family, values in groups.items()}
        compare(recalculated_feedback, iteration["feedback_for_next_generation"]["family_losses"])
        budget_fields = ("updates_this_iteration", "cumulative_updates", "processed_padded_tokens", "supervised_answer_tokens", "examples_seen")
        assert all(len({arm[key] for arm in iteration["training"].values()}) == 1 for key in budget_fields)
        findings.append({
            "iteration": index,
            "risks": {arm: value["risk_mean_answer_nll"] for arm, value in iteration["evaluation"].items()},
            "accuracies": {arm: value["accuracy"] for arm, value in iteration["evaluation"].items()},
            "matched_training_budgets": {key: iteration["training"]["verified"][key] for key in budget_fields},
            "round_trip_mse": {key: actual_metrics["round_trip"][key]["mse"] for key in ("S_of_T_math", "T_of_S_language")},
            "maps_refit_from_raw_activations": True, "source_and_recipient_extraction_recomputed": True,
            "translated_targets_match_T_of_math": True, "text_control_has_exact_donor_information": True,
            "control_auxiliary_targets_match_own_preupdate_states": True,
            "correspondence_controls": correspondence_controls,
            "feedback_recomputed_from_training_only": True,
        })
        del post_update
    sources = [ROOT / "src/lemmaferry" / name for name in protocol["source_sha256"]]
    sources.extend([ROOT / "experiments/verify_constructive_run.py", Path(__file__)])
    report = {
        "status": "passed", "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": "independent Rosetta implementation agent; not human review",
        "run": str(run.relative_to(ROOT)), "command": command,
        "execution": {"returncode": completed.returncode, "parsed_stdout": verification, "stderr": completed.stderr},
        "run_artifact_manifest_sha256": digest(run / "artifact-sha256.json"),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "registered_source_matches_current": recorded_source_match,
        "complete_artifact_hash_keyset_checked": not extra_artifacts,
        "historical_notes": {"unhashed_replay_artifacts": extra_artifacts,
                             "source_changed_after_run": [name for name, matched in recorded_source_match.items() if not matched]},
        "study_history": {
            "earlier_three_arm_run": "runs/constructive-loop-v2",
            "earlier_status": "exploratory engineering execution; preceded addition of correspondence-specific functional controls",
            "protocol_change": "Added shuffled-pair and training-mean auxiliary targets with identical verified prompts and labels; no layer, lambda grid, task set or success threshold was selected from those results.",
            "earlier_review": "artifacts/constructive-loop/independent-review-three-arm-exploratory.json",
        },
        "iterations": findings,
        "static_review": {
            "external_P_values_used_for_generation": False,
            "evaluation_flow": "P is measured before/after updates; proposal feedback is recomputed only on accepted training experience.",
            "budget_scope": "Equal optimizer steps, padded training tokens, supervised answer tokens and initial weights. This is not equal total end-to-end runtime: calibration and extraction add overhead.",
            "target_scope": "T maps formal-certificate states to natural-derivation states; its targets supervise a recipient question-prompt state via an auxiliary loss.",
        },
        "limitations": [
            f"Randomly initialized tiny models and {len(frozen.examples)} explicitly authored evaluation fixtures; no specialist or general-purpose capability evidence.",
            "All arms have identical answer accuracy after both iterations; text control has lower answer NLL than translated treatment.",
            "Family-held-out calibration examples can subsequently enter D_t, and proof dependency constituents need not be held out.",
            "Round-trip consistency and geometric alignment do not establish semantic preservation or sustained broad reasoning improvements.",
        ],
    }
    destination = ROOT / args.output
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "report": str(destination.relative_to(ROOT)), "iterations": findings}, indent=2))


if __name__ == "__main__":
    main()
