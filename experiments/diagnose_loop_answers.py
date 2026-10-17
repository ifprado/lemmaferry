#!/usr/bin/env python3
"""Posthoc content/EOS diagnosis of a completed loop; never train or retune.

Registered full-answer risk includes end-of-sequence likelihood. This standalone
analysis reports that risk's content and EOS components separately, plus choice-
normalized A/B risks and predictions with and without EOS. These diagnostics
must not replace the registered endpoint or feed a generation/training decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_risk(costs: list[float], gold: int) -> float:
    """Negative log probability of the gold label within declared choices."""
    minimum = min(costs)
    return costs[gold] - minimum + math.log(sum(math.exp(minimum - cost) for cost in costs))


def decompose_example(model, example: dict) -> dict:
    """One frozen forward per choice; content consists of every target but EOS."""
    import torch
    choices = list(example.get("choices", ["A", "B"]))
    sequences = [model.encode(example["prompt"], choice) for choice in choices]
    maximum = max(len(ids) for ids, _ in sequences)
    if maximum > model.config["max_length"]:
        raise ValueError("diagnostic example exceeds the registered recipient context")
    ids = torch.full((len(choices), maximum), model.pad_id, device=model.device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, (tokens, _) in enumerate(sequences):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=model.device)
        mask[i, :len(tokens)] = 1
    model.module.eval()
    with torch.inference_mode():
        logits = model.module(ids, mask)
        detail = {}
        for i, (_, labels) in enumerate(sequences):
            positions = [j for j, label in enumerate(labels) if label != -100]
            targets = [labels[j] for j in positions]
            if len(positions) < 2 or targets[-1] != model.eos_id or positions[0] < 1:
                raise ValueError("expected at least one content target followed by explicit EOS")
            selected = logits[i, torch.tensor([j - 1 for j in positions], device=model.device)].float()
            nll = torch.nn.functional.cross_entropy(selected,
                       torch.tensor(targets, device=model.device), reduction="none").cpu().tolist()
            detail[choices[i]] = {
                "content_nll": sum(nll[:-1]), "eos_nll": nll[-1],
                "full_answer_nll": sum(nll), "content_token_count": len(nll) - 1,
                "content_token_ids": targets[:-1], "eos_token_id": targets[-1],
                "per_content_token_nll": nll[:-1]}
    gold = choices.index(example["answer"])
    record = {"id": example["id"], "family": example.get("family", "unspecified"),
              "answer": example["answer"], "choices": detail}
    for name, key in (("full", "full_answer_nll"), ("content", "content_nll"), ("eos_only", "eos_nll")):
        costs = [detail[choice][key] for choice in choices]
        prediction = choices[min(range(len(costs)), key=costs.__getitem__)]
        record[name + "_prediction"] = prediction
        record[name + "_correct"] = prediction == example["answer"]
        record[name + "_choice_normalized_nll"] = normalized_risk(costs, gold)
    record["gold_content_nll"] = detail[example["answer"]]["content_nll"]
    record["gold_eos_nll"] = detail[example["answer"]]["eos_nll"]
    record["gold_full_answer_nll"] = detail[example["answer"]]["full_answer_nll"]
    return record


def diagnose_model(model, evaluation) -> dict:
    evaluation.verify()
    if model.frozen_evaluation_sha256 not in (None, evaluation.sha256):
        raise ValueError("checkpoint was registered against a different evaluation distribution")
    records = [decompose_example(model, row) for row in evaluation.examples]
    summary = {"count": len(records)}
    for key in ("gold_content_nll", "gold_eos_nll", "gold_full_answer_nll",
                "full_choice_normalized_nll", "content_choice_normalized_nll", "eos_only_choice_normalized_nll"):
        summary["mean_" + key] = sum(row[key] for row in records) / len(records)
    for key in ("full", "content", "eos_only"):
        summary[key + "_choice_accuracy"] = sum(row[key + "_correct"] for row in records) / len(records)
    summary["content_full_prediction_disagreements"] = sum(
        row["content_prediction"] != row["full_prediction"] for row in records)
    return {"summary": summary, "records": records}


def _checkpoint(run: Path, stage: str, arm: str, metadata: dict) -> Path:
    path = run / stage / f"{arm}.pt"
    if not path.exists():
        path = Path(metadata["path"])
    if not path.exists() or sha256(path) != metadata["sha256"]:
        raise ValueError(f"missing or changed checkpoint for {stage}/{arm}")
    return path


def _load_shared(path: Path, *, device: str, shared=None):
    """Load only adapters for later HF arms, sharing the same frozen base."""
    import torch
    from lemmaferry.learning import HFRecipient, load_recipient, parameter_sha256
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["backend"] != HFRecipient.backend:
        return load_recipient(path, device=device)
    config = dict(value["config"])
    config["device"] = device
    # Keep diagnosis offline even if an older checkpoint allowed downloads.
    config["local_files_only"] = True
    model = HFRecipient(**config, shared_base_from=shared)
    missing, unexpected = model.module.load_state_dict(value["trainable_state_dict"], strict=False)
    if unexpected or any(not key.startswith("base.") for key in missing):
        raise ValueError("adapter checkpoint contains unexpected parameters")
    model.updates = value["updates"]
    model.frozen_evaluation_sha256 = value["fixed_evaluation_sha256"]
    if parameter_sha256(model) != value["parameter_sha256"]:
        raise ValueError("restored adapter checksum mismatch")
    return model


def run(args) -> dict:
    import torch
    from lemmaferry.learning import HFRecipient, load_evaluation, parameter_sha256
    if args.output.exists() and not args.overwrite:
        raise ValueError("choose a new diagnostic report or explicitly pass --overwrite")
    torch.set_num_threads(args.torch_threads)
    started = time.perf_counter()
    run_path = args.run.resolve()
    source = run_path / "summary.json"
    registered = json.loads(source.read_text())
    if registered.get("status") != "completed":
        raise ValueError("only completed runs can receive this posthoc diagnosis")
    index = len(registered["iterations"]) - 1 if args.iteration == "last" else int(args.iteration)
    iteration = registered["iterations"][index]
    evaluation = load_evaluation(run_path / "evaluation.json")
    if evaluation.sha256 != registered["fixed_evaluation_sha256"]:
        raise ValueError("registered evaluation hash does not match the saved fixed P")
    tasks = []
    if not args.no_initial:
        initial = registered["initial_checkpoints"]["verified"]
        tasks.append(("initial", "initial", "verified", initial, registered["initial_evaluation"]))
    for arm, metadata in sorted(iteration["checkpoints"].items()):
        tasks.append((arm, f"iteration-{index:03d}", arm, metadata, iteration["evaluation"][arm]))
    report = {"schema": "lemmaferry.answer-decomposition.v1", "status": "started",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Posthoc diagnostic only; no retraining, endpoint replacement, generation feedback, or hyperparameter selection.",
              "run": str(run_path), "iteration": index, "device": args.device,
              "torch_version": str(torch.__version__), "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
              "source_summary_sha256": sha256(source), "fixed_evaluation_sha256": evaluation.sha256,
              "script_sha256": sha256(Path(__file__)),
              "learning_implementation_sha256": sha256(ROOT / "src/lemmaferry/learning.py"),
              "definitions": {"content": "all teacher-forced target tokens preceding the final explicit EOS; includes leading separator whitespace where tokenized separately",
                              "choice_normalized_nll": "-log softmax(-candidate sequence NLL)[gold], using only the declared A/B choices",
                              "full_answer": "content NLL + EOS NLL; the registered endpoint remains unchanged",
                              "eos_only": "diagnostic preference from end-of-sequence likelihood after each answer; not semantic answer accuracy"},
              "models": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def write():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write()
    shared = None
    try:
        for name, stage, arm, metadata, expected in tasks:
            path = _checkpoint(run_path, stage, arm, metadata)
            model = _load_shared(path, device=args.device, shared=shared)
            if model.backend == HFRecipient.backend and shared is None:
                shared = model
            measured = diagnose_model(model, evaluation)
            expected_by_id = {row["id"]: row for row in expected["records"]}
            errors = [abs(row["gold_full_answer_nll"] - expected_by_id[row["id"]]["answer_nll"])
                      for row in measured["records"]]
            measured["registered_endpoint_max_abs_error"] = max(errors)
            measured["registered_full_predictions_reproduced"] = all(
                row["full_prediction"] == expected_by_id[row["id"]]["prediction"] for row in measured["records"])
            if max(errors) > 1e-3 or not measured["registered_full_predictions_reproduced"]:
                raise ValueError(f"diagnostic did not reproduce registered full-answer endpoint for {name}")
            measured.update(checkpoint={"path": str(path), "sha256": metadata["sha256"]},
                            recipient_parameter_sha256=parameter_sha256(model),
                            cumulative_updates=model.updates, model_config=model.config,
                            representation_boundary=model.representation_boundary)
            report["models"][name] = measured
            print(f"{name}: full accuracy {measured['summary']['full_choice_accuracy']:.4f}; "
                  f"content accuracy {measured['summary']['content_choice_accuracy']:.4f}; "
                  f"content/EOS risk {measured['summary']['mean_gold_content_nll']:.5f}/"
                  f"{measured['summary']['mean_gold_eos_nll']:.5f}", flush=True)
            write()
            if model is not shared:
                del model
                gc.collect()
        if "verified" in report["models"]:
            treatment = report["models"]["verified"]["summary"]
            report["treatment_minus_control"] = {
                name: {key: treatment[key] - row["summary"][key] for key in treatment if key != "count"}
                for name, row in report["models"].items() if name not in {"verified", "initial"}}
        evaluation.verify()
        report.update(status="completed", total_seconds=time.perf_counter() - started,
                      registered_endpoint_unchanged=True, evaluation_used_for_feedback=False)
        write()
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write()
        raise
    finally:
        if shared is not None:
            del shared
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iteration", default="last", help="iteration index, or last")
    parser.add_argument("--no-initial", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("torch threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
