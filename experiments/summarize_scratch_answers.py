"""Describe completed scratch-study answers without loading models or tuning.

This standard-library-only postprocessing command separates normalized A/B
choice loss from the QA loss that also scores the space and EOS tokens.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def diagnose(root: Path, output: Path | None = None) -> dict:
    root = root.resolve()
    output = output.resolve() if output is not None else root / "answer-diagnostics.json"
    inputs: dict[str, str] = {}

    def read(relative: str) -> dict:
        raw = (root / relative).read_bytes()
        inputs[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    plan = read("plan.json")
    seeds = plan["config"]["seeds"]
    sizes = plan["config"]["sizes"]
    arms = plan["arms"]
    summary = read("summary.json")
    expected = len(seeds) * len(sizes) * len(arms)
    if summary["status"] != "completed" or summary["completed_conditions"] != expected:
        raise ValueError(
            "A completed report is required: "
            f'{summary["completed_conditions"]}/{expected}, {summary["status"]}'
        )
    if "swap_paired" not in arms or "text_only" not in arms:
        raise ValueError("The registered study must contain paired and text-only arms")

    reference_ids: dict[str, list[str]] = {}
    reference_input: dict[str, str] = {}

    def metrics(evaluation: dict) -> dict:
        result = {}
        for split in ("test", "harder"):
            value = evaluation[split]
            rows = value["predictions"]
            n = len(rows)
            if not n:
                raise ValueError(f"The {split} evaluation has no questions")
            ids = [row["id"] for row in rows]
            if split in reference_ids and ids != reference_ids[split]:
                raise ValueError("Evaluation question identity or order differs")
            reference_ids[split] = ids
            if split in reference_input and value["input_sha256"] != reference_input[split]:
                raise ValueError("Evaluation question bytes differ")
            reference_input[split] = value["input_sha256"]
            counts = Counter(row["prediction"] for row in rows)
            answers = Counter(row["answer"] for row in rows)
            correct = sum(row["prediction"] == row["answer"] for row in rows)
            if n != value["overall"]["total"] or correct != value["overall"]["correct"]:
                raise ValueError("Reported correct/total counts disagree with predictions")
            losses = []
            for row in rows:
                lp = row["answer_letter_log_probabilities"]
                if len(lp) != 2 or not all(math.isfinite(v) for v in lp):
                    raise ValueError("Invalid A/B log probabilities")
                if abs(math.log(sum(math.exp(v) for v in lp))) > 1e-5:
                    raise ValueError("A/B log probabilities are not normalized")
                expected_label = "AB"[max(range(2), key=lp.__getitem__)]
                if row["prediction"] != expected_label:
                    raise ValueError("Prediction is not the declared A/B argmax")
                if row["answer"] not in ("A", "B"):
                    raise ValueError("Expected answer is not A or B")
                losses.append(-lp["AB".index(row["answer"])])
            result[split] = {
                "correct": correct,
                "total": n,
                "accuracy": correct / n,
                "generation_correct": value["overall"]["generation_correct"],
                "generation_accuracy": value["overall"]["generation_accuracy"],
                "generation_valid_letter_fraction": sum(
                    row["generation"].strip() in ("A", "B") for row in rows
                ) / n,
                "full_qa_mean_token_nll": value["overall"]["answer_loss"],
                "conditional_choice_nll": statistics.mean(losses),
                "answer_counts": {"A": answers["A"], "B": answers["B"]},
                "prediction_counts": {"A": counts["A"], "B": counts["B"]},
                "dominant_prediction_fraction": max(counts.values()) / n,
                "constant_label": len(counts) == 1,
            }
        return result

    by_condition = {}
    for seed in seeds:
        for size in sizes:
            for arm in arms:
                relative = f"study/{seed}-{size}/{arm}/evaluation.json"
                by_condition[seed, size, arm] = metrics(read(relative))

    aggregates, differences, scaling = [], [], []
    for split in ("test", "harder"):
        for size in sizes:
            for arm in arms:
                per_seed = [
                    {"seed": seed, **by_condition[seed, size, arm][split]}
                    for seed in seeds
                ]
                aggregate = {
                    "split": split, "size": size, "arm": arm, "per_seed": per_seed,
                    "constant_label_models": sum(value["constant_label"] for value in per_seed),
                    "models": len(per_seed),
                    "correct_evaluations_sum": sum(value["correct"] for value in per_seed),
                    "total_evaluations_sum": sum(value["total"] for value in per_seed),
                }
                for metric in (
                    "accuracy", "conditional_choice_nll", "full_qa_mean_token_nll",
                    "dominant_prediction_fraction",
                ):
                    aggregate[metric] = summarize([value[metric] for value in per_seed])
                aggregates.append(aggregate)
            for arm in arms:
                if arm == "swap_paired":
                    continue
                differences.append({
                    "split": split, "size": size,
                    "comparison": "swap_paired minus " + arm, "seed_order": seeds,
                    "accuracy_difference_pp": summarize([
                        100 * (by_condition[seed, size, "swap_paired"][split]["accuracy"]
                               - by_condition[seed, size, arm][split]["accuracy"])
                        for seed in seeds
                    ]),
                    "conditional_choice_nll_difference": summarize([
                        by_condition[seed, size, "swap_paired"][split]["conditional_choice_nll"]
                        - by_condition[seed, size, arm][split]["conditional_choice_nll"]
                        for seed in seeds
                    ]),
                })
        for arm in arms:
            scaling.append({
                "split": split, "arm": arm,
                "size_difference": [max(sizes), min(sizes)], "seed_order": seeds,
                "accuracy_difference_pp": summarize([
                    100 * (by_condition[seed, max(sizes), arm][split]["accuracy"]
                           - by_condition[seed, min(sizes), arm][split]["accuracy"])
                    for seed in seeds
                ]),
            })

    baselines = []
    for seed in seeds:
        relative = f"starting/{seed}-evaluation.json"
        if (root / relative).exists():
            baselines.append({"seed": seed, "metrics": metrics(read(relative))})
    command = [
        ".venv-scratch/bin/python", "experiments/summarize_scratch_answers.py",
        "--run", str(root),
    ]
    if output != root / "answer-diagnostics.json":
        command.extend(["--output", str(output)])
    report = {
        "schema": "lemmaferry.scratch.answer-diagnostics.v1",
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Posthoc descriptive analysis only. No training, model loading, "
                 "hyperparameter selection, or generation feedback.",
        "run": str(root), "registered_plan_sha256": inputs["plan.json"],
        "input_summary_sha256": inputs["summary.json"],
        "input_summary_status": summary["status"], "conditions": expected,
        "seeds": seeds, "sizes": sizes,
        "formulas": {
            "conditional_choice_nll": 'mean(-row["answer_letter_log_probabilities"]["AB".index(row["answer"])])',
            "dominant_prediction_fraction": 'max(Counter(row["prediction"]).values()) / number_of_questions',
            "constant_label": "number_of_distinct_predicted_labels == 1",
            "accuracy_difference_pp": "100 * (accuracy_paired - accuracy_control), "
                                      "matched by model seed and pool size",
            "dispersion": "statistics.stdev over registered model-seed values; "
                          "descriptive sample SD, not confidence interval",
            "format_reference": "log(2)/3: ideal space/EOS prediction plus a "
                                "uniform A/B letter, averaged over three QA tokens",
        },
        "references": {
            "binary_uniform_choice_nll": math.log(2),
            "uniform_letter_with_perfect_format_mean_token_nll": math.log(2) / 3,
        },
        "aggregates": aggregates, "paired_control_differences": differences,
        "largest_minus_smallest": scaling, "starting_baselines": baselines,
        "limitations": [
            "Model seeds share one generated dataset and the same questions in each "
            "evaluation split; summed evaluations are not independent theorem samples.",
            "QA NLL averages the predictable space and EOS together with the answer "
            "letter; low QA NLL alone is not reasoning accuracy.",
            "The proof encoder reconstructs bytes and is lossy; noncollapse and "
            "improved geometry do not certify retention or use of mathematically "
            "sufficient information.",
            "Canonical proof input omits query-specific option order and surface "
            "atom-name mapping present in the solution-state target; whole-state "
            "regression can penalize unavailable nuisance information.",
            "This is a fixed recipient-update budget study with nested pools, not "
            "a test of jointly unlimited data and compute or a broad refutation "
            "of mathematical transfer.",
        ],
        "input_evaluation_sha256": {
            name: digest for name, digest in inputs.items() if name.endswith("evaluation.json")
        },
        "all_input_sha256": inputs,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reproduction_command": shlex.join(command),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Completed scratch-study run")
    parser.add_argument("--output", type=Path, help="Default: RUN/answer-diagnostics.json")
    args = parser.parse_args()
    report = diagnose(args.run, args.output)
    for row in report["paired_control_differences"]:
        if row["comparison"] == "swap_paired minus text_only":
            delta = row["accuracy_difference_pp"]
            print(f'{row["split"]} size={row["size"]}: paired minus text '
                  f'{delta["mean"]:+.6f} pp; sample SD={delta["sample_sd"]}')
    output = args.output.resolve() if args.output is not None else args.run.resolve() / "answer-diagnostics.json"
    print(f"Saved {output}; SHA256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
