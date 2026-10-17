"""Common translator API. Torch is imported only when a neural backend is used."""
from __future__ import annotations

import json
from pathlib import Path

from .rosetta import OBJECTIVES, RosettaPair


def fit_translator(train, validation, *, translator="affine", lambdas=(0., .001, .1, 1.),
                   objective="raw_sum", seed=1729, swap_config=None):
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown affine objective: {objective}")
    if translator == "affine":
        if swap_config is not None:
            raise ValueError("swap_config requires translator='swap'")
        return RosettaPair.fit(train, validation, lambdas, objective=objective, seed=seed)
    if translator == "swap":
        from .swap import SwapPair
        return SwapPair.fit(train, validation, config=swap_config, seed=seed)
    raise ValueError("translator must be affine or swap")


def load_translator(path):
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read translator manifest: {path}") from exc
    schema = manifest.get("schema") if isinstance(manifest, dict) else None
    if schema == "lemmaferry.rosetta.bundle.v1":
        return RosettaPair.load(path)
    if schema == "lemmaferry.swap.bundle.v1":
        from .swap import SwapPair
        return SwapPair.load(path)
    raise ValueError("unsupported translator bundle schema")


def __getattr__(name):
    if name in ("SwapConfig", "SwapPair"):
        from . import swap
        return getattr(swap, name)
    raise AttributeError(name)
