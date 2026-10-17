#!/usr/bin/env python3
"""Plot measured geometry and answer accuracy from the frozen experiment."""
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[1]
    fit = json.loads((root / "artifacts/transfer-v1/affine-results.json").read_text())
    task = json.loads((root / "artifacts/transfer-v1/recipient-evaluation/results.json").read_text())
    if fit["status"] != "completed" or task["status"] != "completed":
        raise ValueError("Only completed experiments may be plotted")
    matplotlib.rcParams["svg.hashsalt"] = "lemmaferry-real-model-results"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    names = ["paired_affine", "shuffled_training_pairs", "training_mean"]
    bars = axes[0].bar(["Paired affine", "Shuffled pairs", "Training mean"],
                       [fit["test_metrics"][n]["mse_divided_by_training_mean_baseline_mse"] for n in names],
                       color=["#6546b5", "#bd8254", "#8a8d94"])
    axes[0].bar_label(bars, fmt="%.3f", padding=4)
    axes[0].set_ylabel("Test vector MSE / training-mean baseline MSE")
    axes[0].set_ylim(0, 1.5)
    axes[0].set_title("Geometry improves")
    modes = ["ordinary_recipient", "translated_donor", "proof_text_recipient"]
    bars = axes[1].bar(["Ordinary", "Translated state", "Same proof as text"],
                       [100 * task["metrics"][m]["two_choice_accuracy"] for m in modes],
                       color=["#8a8d94", "#6546b5", "#287862"])
    axes[1].bar_label(bars, fmt="%.1f%%", padding=4)
    axes[1].axhline(50, color="#777", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Held-out answer accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("No overall accuracy gain")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=9)
    fig.suptitle("Qwen2.5 Math → general, layer 14; 256 train / 64 validation / 128 test", fontsize=11)
    fig.supxlabel("Proof-assisted single-token choice; two held-out graph grammars. No broad transfer claim.", fontsize=9)
    output = root / "artifacts/transfer-v1/comparison"
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
