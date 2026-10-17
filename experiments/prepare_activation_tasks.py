#!/usr/bin/env python3
"""Build auditable two-choice views of Lean-checked implication records.

Both choices have the same length. The wrong choice is an invalid rule chain;
the donor and text-control prompt receive the exact same verified proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lemmaferry.logic import Edge, check_example, validate_path


def choice_view(record: dict, seed: int = 1729) -> dict:
    check_example(record)
    status = record["verification"]
    if status.get("lean") != "verified" or status.get("lean_axioms") != "none":
        raise ValueError("Activation tasks require separately Lean-verified records")
    rng = random.Random(f"choice:{seed}:{record['id']}")
    correct = record["short_path"]
    edges = [Edge(**edge) for edge in record["edges"]]
    wrong = correct.copy()
    # Preserve length and most of the route. Swap the final two applications.
    wrong[-2], wrong[-1] = wrong[-1], wrong[-2]
    if validate_path(edges, record["source_atom"], record["target_atom"], wrong):
        raise ValueError("The decoy must not be a valid source-to-target chain")
    answer = "A" if rng.randrange(2) == 0 else "B"
    choices = {answer: correct, "B" if answer == "A" else "A": wrong}
    facts = "\n".join(record["natural_language"].splitlines()[:-1])
    target = record["atoms"][record["target_atom"]]["label"]
    question = (f"Which listed rule chain establishes {target} from the given fact? "
                "Apply the rules in the listed order. Reply with A or B only.\n"
                + "\n".join(f"{label}: {', '.join(choices[label])}" for label in ("A", "B")))
    ordinary = f"{facts}\n\n{question}\nAnswer:"
    mapping = "; ".join(f"{a['lean']} denotes {a['label']}" for a in record["atoms"])
    information = ("Verified mathematical certificate (implications are one-way):\n"
                   + mapping + "\n" + record["lean_declaration"])
    assisted = f"{facts}\n\n{information}\n{question}\nAnswer:"
    return {"id": record["id"], "family": record["family"], "text": assisted,
            "ordinary_text": ordinary, "mathematical_information": information,
            "text_baseline": assisted, "answer": answer, "choices": choices,
            "lineage": record.get("lineage", record["family"]),
            "source_record_id": record["id"], "formal_verification": status,
            "scope": "Proof-assisted evidence compression; the donor and text baseline see identical text."}


def prepare(inputs: Path, output: Path, count: int | tuple[int, int, int], seed: int = 1729) -> dict:
    if output.exists():
        raise ValueError("Choose a new output directory to preserve frozen inputs")
    counts = (count, count, count) if isinstance(count, int) else count
    if len(counts) != 3 or min(counts) < 2:
        raise ValueError("At least two examples per split are needed")
    views = {}
    metadata = {}
    for (source_split, split), split_count in zip((("train", "train"), ("validation", "val"), ("test", "test")), counts):
        path = inputs / f"{source_split}.checked.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) < split_count:
            raise ValueError(f"{path}: fewer than {split_count} checked records")
        views[split] = [choice_view(row, seed) for row in rows[:split_count]]
        metadata[split] = {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "count": split_count}
    output.mkdir(parents=True)
    for split, rows in views.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        metadata[split].update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               answers={label: sum(row["answer"] == label for row in rows) for label in ("A", "B")})
    report = {"status": "prepared", "seed": seed, "inputs": metadata,
              "scope": "Constructive graph proof assistance. Pilot data have topology/template confounds; expanded data share one renderer and hold out two test grammars. Neither supports broad model claims.",
              "information_policy": "Collector source and target see identical proof-assisted prompts. Functional text baseline receives that exact donor prompt; ordinary recipient receives the question and rules without the certificate.",
              "answer_policy": "Two equal-length candidate chains, one valid and one invalid. No trivial reachable/not-reachable label.",
              "split_policy": "Entire graph families stay in their original split; no claim that renamed members within one family are independent structures."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("results/lf-002/checked"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--counts", type=int, nargs=3, help="Train, validation, test counts; overrides --count")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    counts = tuple(args.counts) if args.counts else args.count
    report = prepare(args.inputs, args.output, counts, args.seed)
    print(f"Prepared {sum(info['count'] for info in report['inputs'].values())} paired proof-assisted prompts in {args.output}")
