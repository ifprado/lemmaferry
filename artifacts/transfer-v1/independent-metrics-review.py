#!/usr/bin/env python3
"""Independently recompute reported metrics and budgets; does not tune or train."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch
from transformers.models.qwen2 import modeling_qwen2

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from lemmaferry.logic import Edge, check_example, validate_path

paths = set()


def digest(path):
    path = Path(path)
    paths.add(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    digest(path)
    return json.loads(Path(path).read_text())


def rows(path):
    digest(path)
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-10), (actual, expected)


inputs = {split: rows(f"data/transfer-v1/anchors/{split}.jsonl") for split in ("train", "val", "test")}
assert {key: len(value) for key, value in inputs.items()} == {"train": 256, "val": 64, "test": 128}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    assert not ({r["lineage"] for r in inputs[a]} & {r["lineage"] for r in inputs[b]})
formal_by_id = {}
audit_count = 0
for split in ("train", "validation", "test"):
    for row in rows(f"data/transfer-v1/checked/{split}.checked.jsonl"):
        check_example(row)
        status = row["verification"]
        assert status["lean"] == "verified" and status["lean_axioms"] == "none"
        log = (Path("lean") / status["lean_log"]).resolve()
        digest(log)
        text = log.read_text()
        for name in (row["lean_name"], row["lean_name"] + "_long"):
            assert re.search(rf"(?m)^'{re.escape(name)}' does not depend on any axioms\s*$", text)
            audit_count += 1
        formal_by_id[row["id"]] = row
assert audit_count == 896
for split, records in inputs.items():
    for row in records:
        formal = formal_by_id[row["source_record_id"]]
        assert row["text"] == row["text_baseline"]
        assert formal["lean_declaration"].strip() in row["text"]
        edges = [Edge(**edge) for edge in formal["edges"]]
        valid = {label: validate_path(edges, formal["source_atom"], formal["target_atom"], path)
                 for label, path in row["choices"].items()}
        assert valid[row["answer"]] and sum(valid.values()) == 1

affine = read("artifacts/transfer-v1/affine-results.json")
source = read("artifacts/transfer-v1/source-evaluation/results.json")
recipient = read("artifacts/transfer-v1/recipient-evaluation/results.json")
student = read("artifacts/student-v1/results.json")
for report in (affine, source, recipient, student):
    assert report["status"] == "completed"
arrays = {}
for role in ("source", "target"):
    folder = Path(f"runs/transfer-v1/{role}")
    manifest = read(folder / "manifest.json")
    assert manifest["collector_sha256"] == digest("experiments/collect_activations.py")
    assert manifest["model_implementation"]["sha256"] == digest(inspect.getfile(modeling_qwen2))
    assert digest(folder / "manifest.json") == affine[role]["manifest_sha256"]
    arrays[role] = {}
    for split in inputs:
        assert digest(f"data/transfer-v1/anchors/{split}.jsonl") == manifest["inputs"][split]["sha256"]
        file = folder / f"{split}.npz"
        assert digest(file) == manifest["outputs"][split]["sha256"]
        with np.load(file, allow_pickle=False) as archive:
            arrays[role][split] = {key: archive[key] for key in archive.files}
        assert arrays[role][split]["ids"].tolist() == [r["id"] for r in inputs[split]]
for split in inputs:
    for key in ("ids", "families", "token_sha256", "lengths"):
        assert np.array_equal(arrays["source"][split][key], arrays["target"][split][key])
for report in (source, recipient):
    assert report["input_sha256"] == digest("data/transfer-v1/anchors/test.jsonl")
    assert report["source_manifest_sha256"] == digest("runs/transfer-v1/source/manifest.json")
    assert report["target_manifest_sha256"] == digest("runs/transfer-v1/target/manifest.json")
    assert report["translator_report_sha256"] == digest("runs/transfer-v1/affine/results.json")
    assert report["code_sha256"]["runner"] == digest("experiments/evaluate_intervention.py")
    assert report["code_sha256"]["qwen2_implementation"] == digest(inspect.getfile(modeling_qwen2))
    for package, version in report["packages"].items():
        assert importlib.metadata.version(package) == version
assert digest("artifacts/transfer-v1/affine-results.json") == digest("runs/transfer-v1/affine/results.json")
assert digest("runs/transfer-v1/affine/translator.npz") == affine["map"]["sha256"]

def supervised_metrics(records, expected):
    assert [r["id"] for r in records] == [r["id"] for r in expected]
    for record, original in zip(records, expected):
        assert record["answer"] == original["answer"]
        assert record["lineage"] == original["lineage"] and record["family"] == original["family"]
        assert record["correct"] == (record["prediction"] == original["answer"])
        assert record["prediction"] in ("A", "B")
    return {"count": len(records), "correct": sum(r["correct"] for r in records),
            "accuracy": float(np.mean([r["correct"] for r in records])),
            "answer_token_nll": float(np.mean([r["answer_token_nll"] for r in records])),
            "prediction_label_counts": {label: sum(r["prediction"] == label for r in records) for label in ("A", "B")}}

functional = {}
for name, report in (("source", source), ("recipient", recipient)):
    assert len(report["records"]) == 128
    modes = list(report["metrics"])
    for mode in modes:
        flattened = [{**{key: row[key] for key in ("id", "answer", "lineage", "family")},
                      **row["scores"][mode]} for row in report["records"]]
        computed = supervised_metrics(flattened, inputs["test"])
        stored = report["metrics"][mode]
        close(computed["accuracy"], stored["two_choice_accuracy"])
        close(computed["answer_token_nll"], stored["answer_token_nll"])
        close(np.mean([r["two_choice_nll"] for r in flattened]), stored["two_choice_nll"])
        close(np.mean([r["greedy_token_correct"] for r in flattened]), stored["greedy_token_accuracy"])
        assert all(r["choice_token_ids"] == [362, 425] for r in flattened)
        for family, scores in report["by_family"].items():
            selected = [r for r in flattened if r["family"] == family]
            close(np.mean([r["correct"] for r in selected]), scores[mode]["two_choice_accuracy"])
            close(np.mean([r["answer_token_nll"] for r in selected]), scores[mode]["answer_token_nll"])
        functional[mode] = computed
    permutation = report["donor_derangement"]
    assert sorted(permutation) == list(range(128)) and all(i != value for i, value in enumerate(permutation))
    rng = np.random.default_rng(1729)
    expected = rng.permutation(128)
    while np.any(expected == np.arange(128)):
        expected = rng.permutation(128)
    assert permutation == expected.tolist()
assert source["donor_derangement"] == recipient["donor_derangement"]
for row in recipient["records"]:
    assert row["scores"]["self_patch"]["prediction"] == row["scores"]["ordinary_recipient"]["prediction"]
close(max(row["self_patch_max_absolute_logit_error"] for row in recipient["records"]),
      recipient["max_self_patch_absolute_logit_error"])

paired = {}
groups = {}
for row in recipient["records"]:
    groups.setdefault(row["lineage"], []).append(row)
names = sorted(groups)
for baseline, stored in recipient["paired_comparisons"].items():
    rng = np.random.default_rng(1729)
    samples = []
    for _ in range(2000):
        selected = rng.integers(len(names), size=len(names))
        values = [int(row["scores"]["translated_donor"]["correct"]) - int(row["scores"][baseline]["correct"])
                  for i in selected for row in groups[names[int(i)]]]
        samples.append(float(np.mean(values)))
    delta = functional["translated_donor"]["accuracy"] - functional[baseline]["accuracy"]
    interval = np.quantile(samples, [.025, .975]).tolist()
    close(delta, stored["accuracy_difference"])
    assert np.allclose(interval, stored["lineage_cluster_bootstrap_95_percent_interval"], atol=1e-12, rtol=1e-12)
    paired[baseline] = {"accuracy_difference": delta, "cluster_bootstrap_interval": interval,
                        "lineages": len(names), "grammars": 2}

# Independent dual ridge solve checks the stored primal fit and validation choice.
with np.load("runs/transfer-v1/affine/translator.npz", allow_pickle=False) as archive:
    saved = {key: archive[key] for key in archive.files}
x = arrays["source"]["train"]["vectors"].astype(np.float64)
y = arrays["target"]["train"]["vectors"].astype(np.float64)
xmean, ymean = x.mean(0), y.mean(0)
xscale, yscale = x.std(0), y.std(0)
xscale = np.where(xscale > np.finfo(float).eps, xscale, 1)
yscale = np.where(yscale > np.finfo(float).eps, yscale, 1)
for key, expected in (("source_mean", xmean), ("target_mean", ymean), ("source_scale", xscale), ("target_scale", yscale)):
    assert np.allclose(saved[key], expected, atol=1e-12, rtol=1e-12)
xs, ys = (x - xmean) / xscale, (y - ymean) / yscale
kernel = xs @ xs.T / len(x)
validation = []
for score in affine["validation_scores"]:
    alpha = score["alpha"]
    beta = xs.T @ np.linalg.solve(kernel + alpha * np.eye(len(x)), ys) / len(x)
    coef = beta.T * yscale[:, None] / xscale[None, :]
    intercept = ymean - coef @ xmean
    prediction = arrays["source"]["val"]["vectors"] @ coef.T + intercept
    error = prediction - arrays["target"]["val"]["vectors"]
    normalized = float(np.sqrt(np.mean((error / yscale) ** 2)))
    close(normalized, score["normalized_rmse"])
    validation.append({"alpha": alpha, "normalized_rmse": normalized})
    if alpha == affine["selected_alpha"]:
        assert np.allclose(coef, saved["coef"], atol=1e-9, rtol=1e-8)
        assert np.allclose(intercept, saved["intercept"], atol=1e-9, rtol=1e-8)
assert min(validation, key=lambda row: (row["normalized_rmse"], row["alpha"]))["alpha"] == affine["selected_alpha"]
test_y = arrays["target"]["test"]["vectors"]
prediction = arrays["source"]["test"]["vectors"] @ saved["coef"].T + saved["intercept"]
mean = arrays["target"]["train"]["vectors"].mean(0)
mse = float(np.mean((prediction - test_y) ** 2))
ratio = mse / float(np.mean((test_y - mean) ** 2))
pu = prediction / np.linalg.norm(prediction, axis=1, keepdims=True)
tu = test_y / np.linalg.norm(test_y, axis=1, keepdims=True)
retrieval = float(np.mean((pu @ tu.T).argmax(1) == np.arange(128)))
close(mse, affine["test_metrics"]["paired_affine"]["mse"])
close(ratio, affine["test_metrics"]["paired_affine"]["mse_divided_by_training_mean_baseline_mse"])
close(retrieval, affine["test_metrics"]["paired_affine"]["cosine_retrieval_top1_accuracy"])

# Reconstruct every student's vocabulary, token count, batch padding, and checkpoint identity.
assert student["code_sha256"] == digest("experiments/train_fixed_student.py")
for package, version in student["packages"].items():
    assert importlib.metadata.version(package) == version
tokenize = lambda text: re.findall(r"[A-Za-z]+|[0-9]|[^\w\s]", text)
words = sorted({token for row in inputs["train"] for token in tokenize(row["ordinary_text"])})
vocabulary = {token: i for i, token in enumerate(["<PAD>", "<UNK>"] + words)}
lengths = [len(tokenize(row["ordinary_text"])) for row in inputs["train"]]
student_runs = []
assert len(student["runs"]) == 6
for run in student["runs"]:
    for split in ("val", "test"):
        computed = supervised_metrics(run[{"val": "validation", "test": "test"}[split]]["records"], inputs[split])
        close(computed["accuracy"], run[{"val": "validation", "test": "test"}[split]]["two_choice_accuracy"])
        close(computed["answer_token_nll"], run[{"val": "validation", "test": "test"}[split]]["answer_token_nll"])
        for family, expected_accuracy in run[{"val": "validation", "test": "test"}[split]]["by_family"].items():
            selected = [row for row in run[{"val": "validation", "test": "test"}[split]]["records"] if row["family"] == family]
            close(np.mean([row["correct"] for row in selected]), expected_accuracy)
    if "untrained_test" in run:
        untrained = supervised_metrics(run["untrained_test"]["records"], inputs["test"])
        close(untrained["accuracy"], run["untrained_test"]["two_choice_accuracy"])
        close(untrained["answer_token_nll"], run["untrained_test"]["answer_token_nll"])
    order_rng = np.random.default_rng(run["seed"])
    padded = 0
    for _ in range(3):
        order = order_rng.permutation(256)
        padded += sum(8 * max(lengths[int(i)] for i in order[start:start + 8]) for start in range(0, 256, 8))
    assert run["updates"] == 96 and run["supervised_answer_tokens"] == 768
    assert run["input_prompt_tokens"] == 3 * sum(lengths) and run["padded_input_tokens"] == padded
    checkpoint = Path(run["checkpoint"]["path"])
    assert digest(checkpoint) == run["checkpoint"]["sha256"]
    saved_student = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert saved_student["vocabulary"] == vocabulary
    assert saved_student["architecture"] == student["architecture"]
    assert saved_student["seed"] == run["seed"] and saved_student["arm"] == run["arm"]
    assert saved_student["input_sha256"] == student["input_sha256"]
    assert sum(value.numel() for value in saved_student["state_dict"].values()) == run["parameters"] == 539437
    reload = read(f"artifacts/student-v1/reloaded/seed{run['seed']}-{run['arm']}.json")
    assert reload["status"] == "completed" and reload["checkpoint_sha256"] == run["checkpoint"]["sha256"]
    assert [row["prediction"] for row in reload["records"]] == [row["prediction"] for row in run["test"]["records"]]
    assert [row["id"] for row in reload["records"]] == [row["id"] for row in run["test"]["records"]]
    student_runs.append({"seed": run["seed"], "arm": run["arm"], "test_accuracy": run["test"]["two_choice_accuracy"],
                         "updates": 96, "answer_tokens": 768, "prompt_tokens": run["input_prompt_tokens"],
                         "padded_tokens": padded, "parameters": 539437, "checkpoint_hash_and_reload_match": True})
for seed in (1729, 1730, 1731):
    pair = [row for row in student_runs if row["seed"] == seed]
    assert len(pair) == 2 and {row["arm"] for row in pair} == {"verified_supervision", "permuted_labels"}
    for key in ("updates", "answer_tokens", "prompt_tokens", "padded_tokens", "parameters"):
        assert pair[0][key] == pair[1][key]
for arm in ("verified_supervision", "permuted_labels"):
    values = [row["test_accuracy"] for row in student_runs if row["arm"] == arm]
    close(np.mean(values), student["summary"][arm]["test_accuracy_mean"])
    close(np.std(values), student["summary"][arm]["test_accuracy_std_across_seeds"])
for split in inputs:
    assert student["input_sha256"][split] == digest(f"data/transfer-v1/anchors/{split}.jsonl")

control = read("runs/transfer-v1/independent-control.json")
assert control["status"] == "passed"
assert control["ordinary_scores"]["prediction"] == recipient["records"][0]["scores"]["ordinary_recipient"]["prediction"]
close(control["ordinary_scores"]["answer_token_nll"], recipient["records"][0]["scores"]["ordinary_recipient"]["answer_token_nll"])
for path in ("artifacts/real-model-plan.json", "artifacts/transfer-v1/independent-control.py", __file__,
             "experiments/fit_activations.py", "src/lemmaferry/alignment.py", "experiments/evaluate_student_checkpoint.py", "CONTRIBUTING.md"):
    digest(path)
result = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
          "reviewer": {"kind": "independent delegated AI agent", "human_signoff": False,
                       "human_example_review": "pending; AI review does not substitute for Ian's review"},
          "formal_inputs": {"examples": 448, "retained_zero_axiom_audits_checked": audit_count,
                            "exact_donor_text_baseline_and_unique_valid_choice_checked": True,
                            "lineage_disjoint_splits": True,
                            "answer_label_counts": {split: {label: sum(row["answer"] == label for row in records) for label in ("A", "B")} for split, records in inputs.items()}},
          "training_majority_label_baseline": {"label": "B", "training_count": 135,
                                               "test_correct": 65, "test_accuracy": 65 / 128},
          "functional_metrics_recomputed_from_128_rows": functional,
          "paired_intervals_independently_recomputed": paired,
          "geometry": {"selected_alpha": affine["selected_alpha"], "independent_dual_validation_scores": validation,
                       "train_only_scaling_verified": True, "primal_map_matches_independent_dual_solution": True,
                       "test_mse": mse, "test_mse_ratio_to_training_mean": ratio, "retrieval_top1": retrieval,
                       "candidate_pool": 128},
          "student_runs": student_runs, "student_summary": student["summary"],
          "independent_principal_control": control,
          "manually_inspected_example": {"id": "transfer_test_000000", "answer": "A",
                 "route": "Rune0 -> Rune8 -> Rune4 -> Rune1 -> Rune12",
                 "rules": ["e2", "e9", "e5", "e4"],
                 "invalid_B": "After e2,e9, Rune4 is known; e4 requires Rune1, so the swapped final rules fail.",
                 "Lean_term": "h_e4 (h_e5 (h_e9 (h_e2 (h_start))))"},
          "interpretation": [
              "Geometric alignment succeeds on this fixed test pool; it does not establish useful reasoning transfer.",
              "Translated accuracy66/128 is below ordinary69/128 and exact proof text82/128. The ordinary comparison's conditional bootstrap interval includes zero.",
              "Translated answer-token NLL improves slightly over ordinary but remains worse than exact proof text; claims should distinguish NLL and accuracy.",
              "Only two test grammars were sampled; lineage bootstrap intervals are conditional on those grammars.",
              "The single-token oracle text-state patch matches ordinary accuracy, suggesting the tested one-position interface is a limiting factor; this is an inference, not proof of a general impossibility.",
              "The correct-label student mean47.65625% does not beat the permuted-label mean49.21875%. This small weak-control experiment provides no basis for iterative generation or broader capability claims."],
          "review_limits": ["NLL means were recomputed from stored per-row NLLs; full vocabulary logits for every row were not retained.",
                            "The real-model principal control was independently rerun on one fixed example; all128 intervention rows were not rerun.",
                            "Matching student initialization and optimizer policy were verified in the hashed source; all six saved checkpoint hashes and reported standalone reload predictions were checked."],
          "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths, key=str)}}
Path("artifacts/transfer-v1/independent-review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": result["status"], "geometry": result["geometry"],
                  "functional_correct_counts": {name: value["correct"] for name, value in functional.items()},
                  "student_means": student["summary"], "files_hashed": len(paths)}, indent=2))
