#!/usr/bin/env python3
"""Run registered constructive-library, Rosetta, and recipient-learning cycles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lemmaferry.loop import LoopConfig, run_loop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluation", type=Path, default=Path("benchmarks/independent-v1.jsonl"))
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--proposals-per-iteration", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--backend", choices=("tiny", "pretrained"), default="tiny")
    parser.add_argument("--experience-mode", choices=("language", "translated"), default="language")
    parser.add_argument("--no-representations", action="store_true")
    parser.add_argument("--translator", choices=("affine", "swap"), default="affine")
    parser.add_argument("--translator-objective", choices=("raw_sum", "standardized_mean"), default="raw_sum",
                        help="affine fitting objective; swap uses its registered standardized multiloss")
    parser.add_argument("--swap-latent-dim", type=int, default=8)
    parser.add_argument("--swap-hidden-dim", type=int, default=64)
    parser.add_argument("--swap-steps", type=int, default=128)
    parser.add_argument("--swap-batch-size", type=int, default=32)
    parser.add_argument("--swap-learning-rate", type=float, default=0.002)
    parser.add_argument("--swap-validation-every", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--mastery-threshold", type=float, default=0.7)
    parser.add_argument("--recipient-model")
    parser.add_argument("--recipient-revision")
    parser.add_argument("--math-model")
    parser.add_argument("--math-revision")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--lean", type=Path)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    import torch
    torch.set_num_threads(args.torch_threads)
    options = vars(args).copy()
    output, evaluation, lean = options.pop("output"), options.pop("evaluation"), options.pop("lean")
    options.pop("torch_threads")
    options["representations"] = not options.pop("no_representations")
    summary = run_loop(output, evaluation, config=LoopConfig(**options), lean=lean)
    print(f"Completed {summary['completed_iterations']} iterations; "
          f"{summary['library']['record_count']} Lean-admitted lemmas. "
          f"Evidence: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
