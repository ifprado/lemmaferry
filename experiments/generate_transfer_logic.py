#!/usr/bin/env python3
"""Generate a frozen, topology-held-out extension for the real-model experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lemmaferry.logic import (Edge, check_example, lean_declaration, path_atoms,
                             render_problem, shortest_path)


def make_record(index: int, split: str, seed: int = 1729) -> dict:
    if split not in ("train", "validation", "test"):
        raise ValueError("unknown split")
    rng = random.Random(f"transfer-v1:{seed}:{split}:{index}")
    raw = []
    next_atom = 1

    def route(start, length, target=None):
        nonlocal next_atom
        vertices = [start]
        for _ in range(length - 1):
            vertices.append(next_atom)
            next_atom += 1
        if target is None:
            target = next_atom
            next_atom += 1
        vertices.append(target)
        indices = []
        for a, b in zip(vertices, vertices[1:]):
            indices.append(len(raw))
            raw.append((a, b))
        return target, indices

    if split == "train":
        family = "transfer_parallel_routes"
        a = rng.randint(3, 6)
        b = a + rng.randint(1, 4)
        target, _ = route(0, a)
        _, long_indices = route(0, b, target)
        lengths = [a, b]
    elif split == "validation":
        family = "transfer_triple_routes"
        a = rng.randint(4, 6)
        b = a + rng.randint(1, 2)
        c = b + rng.randint(1, 2)
        target, _ = route(0, a)
        route(0, b, target)
        _, long_indices = route(0, c, target)
        lengths = [a, b, c]
    else:
        family = "transfer_chained_diamonds" if index % 2 == 0 else "transfer_cyclic_diamonds"
        a, c = rng.randint(2, 3), rng.randint(2, 3)
        b, d = a + rng.randint(1, 2), c + rng.randint(2, 3)
        middle, _ = route(0, a)
        _, first = route(0, b, middle)
        target, _ = route(middle, c)
        _, second = route(middle, d, target)
        long_indices = first + second
        lengths = [a, b, c, d]
        if index % 2:
            raw.append((middle, 0))
    distractors = rng.randint(1, 3)
    for _ in range(distractors):
        raw.append((next_atom, next_atom + 1))
        next_atom += 2
    # In this restricted construction, the ordered route lengths, cycle flag,
    # and number of isolated directed edges identify the rooted topology up to
    # atom renaming and edge ordering. They are not a generic graph hash.
    signature = {"grammar": family, "ordered_route_lengths": lengths,
                 "isolated_directed_edges": distractors}
    lineage = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    permutation = list(range(next_atom))
    rng.shuffle(permutation)
    order = list(range(len(raw)))
    rng.shuffle(order)
    edge_id = {old: f"e{i}" for i, old in enumerate(order)}
    edges = [Edge(f"e{i}", permutation[raw[old][0]], permutation[raw[old][1]])
             for i, old in enumerate(order)]
    start, target = permutation[0], permutation[target]
    short = shortest_path(edges, start, target)
    long = [edge_id[i] for i in long_indices]
    labels = [f"Rune{i}" for i in range(next_atom)]
    name = f"lf_transfer_{split}_{index:06d}"
    statement, proof, declaration = lean_declaration(name, next_atom, edges, start, target, short)
    _, long_proof, long_declaration = lean_declaration(name + "_long", next_atom, edges, start, target, long)
    record = {"schema_version": 1, "id": f"transfer_{split}_{index:06d}",
              "family": family, "lineage": lineage, "structure_signature": signature,
              "split": split, "seed": seed, "rendering_template": "stations",
              "atoms": [{"id": i, "lean": f"p{i}", "label": label} for i, label in enumerate(labels)],
              "source_atom": start, "target_atom": target,
              "edges": [e.as_dict() for e in edges], "short_path": short, "long_path": long,
              "lean_name": name, "lean_statement": statement, "lean_proof": proof,
              "lean_declaration": declaration, "long_lean_proof": long_proof,
              "long_lean_declaration": long_declaration,
              "natural_language": render_problem(labels, edges, start, target, "stations"),
              "answer": {"reachable": True, "rule_chain": short,
                         "atom_chain": [labels[i] for i in path_atoms(edges, start, short)]},
              "provenance": {"generator": "transfer_implication_graph_v1",
                             "short_cost": len(short), "long_cost": len(long),
                             "novel_mathematics": False,
                             "split_policy": "Entire topology grammars held out; shared language and lexicon.",
                             "lineage_policy": "Rooted route-length/cycle/distractor signatures group renamings within each split."},
              "verification": {"python_certificate": "verified", "render_roundtrip": "verified",
                               "lean": "not_run", "lean_axioms": "not_run"}}
    check_example(record)
    return record


def generate(output: Path, counts: tuple[int, int, int] = (256, 64, 128), seed: int = 1729):
    if output.exists():
        raise ValueError("Choose a new output directory")
    corpus = {split: [make_record(i, split, seed) for i in range(count)]
              for split, count in zip(("train", "validation", "test"), counts)}
    seen = {}
    for split, records in corpus.items():
        for record in records:
            if record["lineage"] in seen and seen[record["lineage"]] != split:
                raise ValueError("Structural lineage crossed splits")
            seen[record["lineage"]] = split
    output.mkdir(parents=True)
    metadata = {}
    for split, records in corpus.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        metadata[split] = {"count": len(records), "lineages": len({r["lineage"] for r in records}),
                           "families": sorted({r["family"] for r in records}),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps({"seed": seed, "generator": "transfer_implication_graph_v1", "splits": metadata,
                                                     "lean_status": "not_run", "note": "Compiler evidence belongs in separately checked records."}, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs=3, default=(256, 64, 128))
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, tuple(args.counts), args.seed), indent=2))
