#!/usr/bin/env python3
"""Run the artificial alignment baseline and an ambiguity counterexample.

Example from the repository root:
    python experiments/toy_alignment.py --seed 7 --output artifacts/alignment.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Direct checkout execution is convenient for a first experiment. Installed
# packages work too; this path always selects this checkout's implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lemmaferry.alignment import (  # noqa: E402
    symmetric_cloud_ambiguity,
    synthetic_alignment_experiment,
)


def plot_report(report: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; JSON results are still complete.", file=sys.stderr)
        return False
    if path.suffix.lower() not in {".svg", ".png"}:
        raise ValueError("plot path must end in .svg or .png")
    matplotlib.rcParams["svg.hashsalt"] = "lemmaferry-toy-alignment"
    metrics = report["paired_alignment"]["metrics"]
    names = ["paired_affine", "wrong_pairs", "training_mean"]
    labels = ["Paired affine", "Wrong pairs", "Training mean"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    colors = ["#6649bd", "#b77849", "#888b94"]
    axes[0].bar(labels, [metrics[key]["normalized_rmse"] for key in names], color=colors)
    axes[0].set_ylabel("Target RMSE / training-coordinate scale")
    axes[0].set_title("Held-out reconstruction error")
    axes[1].bar(labels, [metrics[key]["receiver_task_accuracy"] for key in names], color=colors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Receiver task accuracy")
    axes[1].set_title("Planted binary task")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=9)
    figure.suptitle("Artificial shared-factor test. No LLMs or Lean proofs used.", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, metadata={"Date": None} if path.suffix == ".svg" else {})
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--output", type=Path, help="Also write complete JSON to this path.")
    parser.add_argument("--plot", type=Path, help="Optional .svg or .png chart; requires matplotlib.")
    arguments = parser.parse_args()
    paired, _ = synthetic_alignment_experiment(
        seed=arguments.seed,
        train_size=arguments.train_size,
        test_size=arguments.test_size,
        noise=arguments.noise,
    )
    report = {
        "paired_alignment": paired,
        "ambiguity_control": symmetric_cloud_ambiguity(),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    if arguments.plot:
        plot_report(report, arguments.plot)
    print(serialized, end="")


if __name__ == "__main__":
    main()
