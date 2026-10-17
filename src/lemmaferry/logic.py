"""Small constructive implication problems and their checked graph certificates.

This is a controlled synthetic domain, not a generator of novel mathematics.
Python certificate checks are independent of Lean; neither substitutes for Lean.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class Edge:
    id: str
    source: int
    target: int

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "source": self.source, "target": self.target}


def shortest_path(edges: Iterable[Edge], start: int, target: int) -> list[str] | None:
    """Breadth-first search using edge IDs; returns None if no route exists."""
    edges = list(edges)
    if len({edge.id for edge in edges}) != len(edges):
        raise ValueError("edge IDs must be unique")
    adjacency: dict[int, list[Edge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge)
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        atom, path = queue.popleft()
        if atom == target:
            return path
        for edge in adjacency.get(atom, []):
            if edge.target not in visited:
                visited.add(edge.target)
                queue.append((edge.target, path + [edge.id]))
    return None


def path_atoms(edges: Iterable[Edge], start: int, path: Iterable[str]) -> list[int]:
    edges = list(edges)
    lookup = {edge.id: edge for edge in edges}
    if len(lookup) != len(edges):
        raise ValueError("edge IDs must be unique")
    atoms = [start]
    for edge_id in path:
        if edge_id not in lookup:
            raise ValueError(f"unknown edge: {edge_id}")
        edge = lookup[edge_id]
        if edge.source != atoms[-1]:
            raise ValueError(f"disconnected edge in path: {edge_id}")
        atoms.append(edge.target)
    return atoms


def validate_path(edges: Iterable[Edge], start: int, target: int, path: Iterable[str]) -> bool:
    try:
        return path_atoms(edges, start, path)[-1] == target
    except (ValueError, TypeError):
        return False


def proof_term(path: Iterable[str]) -> str:
    """Construct a proof by repeatedly applying an implication to a known fact."""
    term = "h_start"
    for edge_id in path:
        if not re.fullmatch(r"e[0-9]+", edge_id):
            raise ValueError("unsafe edge ID")
        term = f"h_{edge_id} ({term})"
    return term


def lean_declaration(
    name: str, atom_count: int, edges: Iterable[Edge], start: int, target: int,
    path: Iterable[str],
) -> tuple[str, str, str]:
    edges, path = list(edges), list(path)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("unsafe Lean theorem name")
    if not (0 <= start < atom_count and 0 <= target < atom_count):
        raise ValueError("atom outside declared range")
    if any(not (0 <= e.source < atom_count and 0 <= e.target < atom_count) for e in edges):
        raise ValueError("edge atom outside declared range")
    if any(not re.fullmatch(r"e[0-9]+", edge.id) for edge in edges):
        raise ValueError("unsafe edge ID")
    if not validate_path(edges, start, target, path):
        raise ValueError("cannot construct a proof from an invalid path")
    arguments = [f"    ({' '.join(f'p{i}' for i in range(atom_count))} : Prop)",
                 f"    (h_start : p{start})"]
    arguments += [f"    (h_{edge.id} : p{edge.source} → p{edge.target})" for edge in edges]
    statement = f"theorem {name}\n" + "\n".join(arguments) + f"\n    : p{target}"
    term = proof_term(path)
    return statement, term, statement + f" :=\n  {term}\n"


def render_problem(labels: list[str], edges: list[Edge], start: int, target: int,
                   template: str) -> str:
    """Render a reversible controlled language. No LLM is used in rendering."""
    if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label) for label in labels):
        raise ValueError("labels must be single safe words")
    if template == "stations":
        lines = [f"Station {labels[start]} is active."]
        lines += [f"Rule {e.id}: If station {labels[e.source]} is active, then station {labels[e.target]} is active." for e in edges]
        lines += [f"Give a shortest rule chain establishing that station {labels[target]} is active."]
    elif template == "tasks":
        lines = [f"Task {labels[start]} is enabled."]
        lines += [f"Dependency {e.id}: Enabling {labels[e.source]} enables {labels[e.target]}." for e in edges]
        lines += [f"Give a shortest dependency chain enabling {labels[target]}."]
    elif template == "beacons":
        lines = [f"Beacon {labels[start]} is lit."]
        lines += [f"Link {e.id}: A lit {labels[e.source]} lights {labels[e.target]}." for e in edges]
        lines += [f"Give a shortest link chain lighting {labels[target]}."]
    else:
        raise ValueError(f"unknown rendering template: {template}")
    return "\n".join(lines)


def parse_problem(text: str, template: str) -> tuple[str, str, list[tuple[str, str, str]]]:
    """Strict parser for the controlled language, used for semantic round trips."""
    patterns = {
        "stations": (r"Station (\w+) is active\.",
                     r"Rule (e\d+): If station (\w+) is active, then station (\w+) is active\.",
                     r"Give a shortest rule chain establishing that station (\w+) is active\."),
        "tasks": (r"Task (\w+) is enabled\.",
                  r"Dependency (e\d+): Enabling (\w+) enables (\w+)\.",
                  r"Give a shortest dependency chain enabling (\w+)\."),
        "beacons": (r"Beacon (\w+) is lit\.",
                    r"Link (e\d+): A lit (\w+) lights (\w+)\.",
                    r"Give a shortest link chain lighting (\w+)\."),
    }
    if template not in patterns:
        raise ValueError("unknown rendering template")
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("missing facts or question")
    fact_pattern, edge_pattern, question_pattern = patterns[template]
    fact = re.fullmatch(fact_pattern, lines[0])
    question = re.fullmatch(question_pattern, lines[-1])
    if not fact or not question:
        raise ValueError("invalid fact or question")
    parsed_edges = []
    for line in lines[1:-1]:
        match = re.fullmatch(edge_pattern, line)
        if not match:
            raise ValueError("invalid rule")
        parsed_edges.append(match.groups())
    return fact.group(1), question.group(1), parsed_edges


def _graph(family: str, rng: random.Random) -> tuple[int, list[Edge], int, int, list[str]]:
    """Construct distinct topology families. All contain a longer valid route."""
    raw: list[tuple[int, int]] = []
    next_atom = 1
    start = 0

    def route(a: int, length: int, b: int | None = None) -> tuple[int, list[int]]:
        nonlocal next_atom
        vertices = [a]
        for _ in range(length - 1):
            vertices.append(next_atom)
            next_atom += 1
        if b is None:
            b = next_atom
            next_atom += 1
        vertices.append(b)
        indices = []
        for left, right in zip(vertices, vertices[1:]):
            indices.append(len(raw))
            raw.append((left, right))
        return b, indices

    if family == "parallel_routes":
        target, _ = route(start, 2)
        _, long_indices = route(start, 4, target)
    elif family == "triple_routes":
        target, _ = route(start, 3)
        _, long_indices = route(start, 5, target)
        route(start, 4, target)
    elif family == "chained_diamonds":
        middle, _ = route(start, 2)
        _, first_long = route(start, 3, middle)
        target, _ = route(middle, 2)
        _, second_long = route(middle, 4, target)
        long_indices = first_long + second_long
    else:
        raise ValueError(f"unknown graph family: {family}")

    # Unreachable distractor component cannot shorten the source-to-target route.
    for _ in range(rng.randrange(1, 4)):
        raw.append((next_atom, next_atom + 1))
        next_atom += 2

    # Permute node numbering and edge order so IDs do not reveal route order.
    permutation = list(range(next_atom))
    rng.shuffle(permutation)
    order = list(range(len(raw)))
    rng.shuffle(order)
    old_to_id = {old: f"e{i}" for i, old in enumerate(order)}
    edges = [Edge(f"e{i}", permutation[raw[old][0]], permutation[raw[old][1]])
             for i, old in enumerate(order)]
    return next_atom, edges, permutation[start], permutation[target], [old_to_id[i] for i in long_indices]


SPLIT_SETTINGS = {
    "train": ("parallel_routes", "stations", "Vela"),
    "validation": ("triple_routes", "tasks", "Nori"),
    "test": ("chained_diamonds", "beacons", "Zuri"),
}


def make_example(index: int, split: str = "train", seed: int = 0) -> dict[str, Any]:
    if split not in SPLIT_SETTINGS:
        raise ValueError(f"unknown split: {split}")
    if index < 0:
        raise ValueError("index must be nonnegative")
    family, template, prefix = SPLIT_SETTINGS[split]
    # Separate reproducible random streams across splits.
    rng = random.Random(f"lemmaferry:{seed}:{split}:{index}")
    count, edges, start, target, long_path = _graph(family, rng)
    short_path = shortest_path(edges, start, target)
    assert short_path is not None and len(short_path) < len(long_path)
    labels = [f"{prefix}_{index}_{i}" for i in range(count)]
    name = f"lf_{split}_{index:06d}"
    statement, term, declaration = lean_declaration(name, count, edges, start, target, short_path)
    _, long_term, long_declaration = lean_declaration(name + "_long", count, edges, start, target, long_path)
    problem = render_problem(labels, edges, start, target, template)
    parsed = parse_problem(problem, template)
    expected = (labels[start], labels[target], [(e.id, labels[e.source], labels[e.target]) for e in edges])
    if parsed != expected:
        raise AssertionError("rendering changed graph semantics")
    route_atoms = path_atoms(edges, start, short_path)
    example = {
        "schema_version": 1,
        "id": f"{split}_{index:06d}",
        "family": family,
        "split": split,
        "seed": seed,
        "rendering_template": template,
        "atoms": [{"id": i, "lean": f"p{i}", "label": label} for i, label in enumerate(labels)],
        "source_atom": start,
        "target_atom": target,
        "edges": [edge.as_dict() for edge in edges],
        "short_path": short_path,
        "long_path": long_path,
        "lean_name": name,
        "lean_statement": statement,
        "lean_proof": term,
        "lean_declaration": declaration,
        "long_lean_proof": long_term,
        "long_lean_declaration": long_declaration,
        "natural_language": problem,
        "answer": {"reachable": True, "rule_chain": short_path,
                   "atom_chain": [labels[i] for i in route_atoms]},
        "provenance": {
            "generator": "constructive_implication_graph_v1",
            "certificate": "ordered edge applications from a single assumed proposition",
            "proof_cost": "number of implication applications, not discovery time or global proof complexity",
            "short_cost": len(short_path),
            "long_cost": len(long_path),
            "rendering": "deterministic controlled language with exact parser round trip",
            "assumptions": [f"p{start}"] + [f"p{e.source} -> p{e.target}" for e in edges],
            "novel_mathematics": False,
            "binary_classification_warning": "All generated goals are reachable. Evaluate chain correctness, not yes/no accuracy.",
        },
        "verification": {
            "python_certificate": "verified",
            "render_roundtrip": "verified",
            "lean": "not_run",
            "lean_axioms": "not_run",
        },
    }
    check_example(example)
    return example


def check_example(example: dict[str, Any]) -> None:
    """Raise on an invalid certificate or a mismatch between graph and text."""
    edges = [Edge(**edge) for edge in example["edges"]]
    start, target = example["source_atom"], example["target_atom"]
    shortest = shortest_path(edges, start, target)
    if shortest is None or not validate_path(edges, start, target, example["short_path"]):
        raise ValueError("invalid short certificate")
    if len(shortest) != len(example["short_path"]):
        raise ValueError("claimed short route is not shortest")
    if not validate_path(edges, start, target, example["long_path"]):
        raise ValueError("invalid long certificate")
    if len(example["long_path"]) <= len(shortest):
        raise ValueError("long route must be strictly longer")
    labels = [atom["label"] for atom in example["atoms"]]
    if [atom["id"] for atom in example["atoms"]] != list(range(len(labels))):
        raise ValueError("atom IDs must be consecutive and ordered")
    if len(set(labels)) != len(labels):
        raise ValueError("atom labels must be unique")
    expected = (labels[start], labels[target], [(e.id, labels[e.source], labels[e.target]) for e in edges])
    if parse_problem(example["natural_language"], example["rendering_template"]) != expected:
        raise ValueError("rendered problem does not match certificate graph")
    correct_answer = {"reachable": True, "rule_chain": example["short_path"],
                      "atom_chain": [labels[i] for i in path_atoms(edges, start, example["short_path"])]}
    if example["answer"] != correct_answer:
        raise ValueError("answer does not match shortest certificate")
    name = example["lean_name"]
    statement, proof, declaration = lean_declaration(name, len(labels), edges, start, target, example["short_path"])
    if (example["lean_statement"], example["lean_proof"], example["lean_declaration"]) != (statement, proof, declaration):
        raise ValueError("Lean source does not match graph certificate")
    _, long_proof, long_declaration = lean_declaration(name + "_long", len(labels), edges, start, target, example["long_path"])
    if (example["long_lean_proof"], example["long_lean_declaration"]) != (long_proof, long_declaration):
        raise ValueError("long Lean source does not match graph certificate")


def generate_examples(count: int, split: str = "train", seed: int = 0) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    return [make_example(i, split, seed) for i in range(count)]
