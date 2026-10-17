"""Bounded proposals from parameterized constructive families and accepted lemmas."""

from __future__ import annotations

import math
import random

from .library import FAMILIES, LibraryError, VerifiedLibrary, structural_hash, views


def _apply(function, argument):
    return {"apply": [function, argument]}


def _call(record, atoms, arguments):
    return {"lemma": {"id": record["id"], "atoms": atoms, "arguments": arguments}}


def make_proposal(family: str, size: int, library: VerifiedLibrary | None = None,
                  *, index: int = 0, iteration: int = 0, seed: int = 0,
                  provenance: dict | None = None, label_prefix: str = "T") -> dict:
    """Construct a certificate, using the largest smaller accepted family lemma.

    Size has no finite list of supported values. The verifier's process timeout
    and explicit proof-node budget bound one attempted computation.
    """
    records = library.records() if library is not None else []
    candidates = [r for r in records if r["family"] == family and r["payload"]["size"] < size]
    prior = max(candidates, key=lambda r: r["payload"]["size"]) if candidates else None
    dependencies = [prior["id"]] if prior else []
    if family == "implication_chain":
        start = prior["payload"]["size"] if prior else 0
        proof = _call(prior, list(range(start + 1)), [{"hyp": i} for i in range(start + 1)]) if prior else {"hyp": 0}
        for i in range(start + 1, size + 1):
            proof = _apply({"hyp": i}, proof)
    elif family == "and_introduction":
        if prior:
            offset = size - prior["payload"]["size"]
            proof = _call(prior, list(range(offset, size)), [{"hyp": i} for i in range(offset, size)])
            for i in reversed(range(offset)):
                proof = {"and_intro": [{"hyp": i}, proof]}
        else:
            proof = {"hyp": size - 1}
            for i in reversed(range(size - 1)):
                proof = {"and_intro": [{"hyp": i}, proof]}
    elif family == "and_elimination":
        if prior:
            offset = size - prior["payload"]["size"]
            premise = {"hyp": 0}
            for _ in range(offset):
                premise = {"and_right": premise}
            proof = _call(prior, list(range(offset, size)), [premise])
            index = offset + prior["payload"]["index"]
        else:
            proof = {"hyp": 0}
            for _ in range(index):
                proof = {"and_right": proof}
            if index < size - 1:
                proof = {"and_left": proof}
    else:
        raise LibraryError("unsupported constructive family")
    proposal = {"family": family, "size": size, "index": index, "proof": proof,
                "dependencies": dependencies, "label_prefix": label_prefix,
                "provenance": {"generator": "library_conditioned_constructive_v1", "iteration": iteration,
                               "seed": seed, "parent_library_snapshot": library.snapshot()["snapshot_hash"] if library else None,
                               "reused_lemma_ids": dependencies, **(provenance or {})}}
    proposal.update(views(proposal))
    proposal["structural_hash"] = structural_hash(proposal)
    return proposal


def propose(library: VerifiedLibrary, iteration: int = 0, count: int = 3, seed: int = 1729,
            feedback: dict | None = None) -> list[dict]:
    """Use training-curriculum loss to allocate a finite proposal budget.

    Weak families get more slots and the smallest structurally novel increment.
    Mastered families advance by two constructors. No evaluation corpus is read.
    """
    if type(count) is not int or count < 0 or type(iteration) is not int or iteration < 0:
        raise LibraryError("count and iteration must be nonnegative integers")
    if count > 10000:
        raise LibraryError("one call is limited to 10000 proposals; use multiple bounded calls")
    feedback = dict(feedback or {})
    if feedback and feedback.get("source", "training_curriculum_only") != "training_curriculum_only":
        raise LibraryError("generation feedback must come from the training curriculum only")
    losses = feedback.get("family_losses", {})
    threshold = float(feedback.get("mastery_threshold", .7))
    if not isinstance(losses, dict) or not math.isfinite(threshold):
        raise LibraryError("invalid curriculum feedback")
    if any(family not in FAMILIES or not math.isfinite(float(loss)) for family, loss in losses.items()):
        raise LibraryError("invalid family losses")
    records = library.records()
    known_shapes = {r["structural_hash"] for r in records}
    maxima = {family: max([r["payload"]["size"] for r in records if r["family"] == family] or [1]) for family in FAMILIES}
    order = list(FAMILIES)
    random.Random(f"constructive:{seed}:{iteration}").shuffle(order)
    if losses:
        order.sort(key=lambda family: -float(losses.get(family, -1)))
    focus = order[0] if losses and float(losses.get(order[0], -1)) > threshold else None
    result = []
    for position in range(count):
        family = focus if focus and position < (count + 1) // 2 else order[position % len(order)]
        mastered = family in losses and float(losses[family]) <= threshold
        increment = 2 if mastered else 1
        size = max(2, maxima[family] + increment)
        rationale = ("weak training family: allocate extra slot and minimum novel size" if family == focus
                     else "mastered training family: advance two constructors" if mastered
                     else "cover a constructive family with the next novel size")
        while True:
            candidate = make_proposal(family, size, library, iteration=iteration, seed=seed,
                                      provenance={"selection_rationale": rationale,
                                                  "curriculum_feedback": feedback,
                                                  "size_increment": increment})
            if candidate["structural_hash"] not in known_shapes:
                break
            size += increment
        known_shapes.add(candidate["structural_hash"])
        maxima[family] = size
        result.append(candidate)
    return result
