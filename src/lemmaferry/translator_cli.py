"""Shared opt-in translator arguments; importing this module does not load Torch."""

from __future__ import annotations


def add_translator_arguments(parser) -> None:
    parser.add_argument("--translator", choices=("affine", "swap"), default="affine")
    parser.add_argument("--swap-latent-dim", type=int, default=8)
    parser.add_argument("--swap-hidden-dim", type=int, default=64)
    parser.add_argument("--swap-steps", type=int, default=128)
    parser.add_argument("--swap-batch-size", type=int, default=32)
    parser.add_argument("--swap-learning-rate", type=float, default=0.002)
    parser.add_argument("--swap-validation-every", type=int, default=16)


def swap_options(args) -> dict:
    return {key: getattr(args, "swap_" + key) for key in (
        "latent_dim", "hidden_dim", "steps", "batch_size", "learning_rate", "validation_every"
    )}
