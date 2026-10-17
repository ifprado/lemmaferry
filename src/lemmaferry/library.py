"""Content-addressed constructive lemmas, admitted only by the pinned real Lean.

The trusted input language is deliberately restricted. It cannot express raw
Lean commands, custom axioms, classical tactics, imports, or proof placeholders.
Stored status flags are never accepted as a substitute for compiler execution.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
import uuid

PIN = "leanprover/lean4:v4.35.0"
LEAN_COMMIT = "eb6cafb7290e8bd5f2937522d8d804f58f02bafc"
FAMILIES = ("implication_chain", "and_introduction", "and_elimination")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LibraryError(ValueError):
    """An invalid proposal, foundation, or persisted library."""


@dataclass(frozen=True)
class Foundation:
    name: str = "lemmaferry_constructive_foundation_v1"
    lean_toolchain: str = PIN
    lean_commit: str = LEAN_COMMIT
    logic: str = "constructive_prop_v1"
    schema_version: int = 1

    @property
    def hash(self) -> str:
        return content_hash(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)


def _nat(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise LibraryError(f"{label} must be an integer >= {minimum}")
    return value


def conjunction(atoms: list[Any]):
    result = atoms[-1]
    for atom in reversed(atoms[:-1]):
        result = {"and": [atom, result]}
    return result


def statement(family: str, size: int, index: int = 0) -> dict:
    _nat(size, "size", 2)
    if family == "implication_chain":
        return {"atom_count": size + 1,
                "premises": [0] + [{"imp": [i, i + 1]} for i in range(size)],
                "conclusion": size}
    if family == "and_introduction":
        return {"atom_count": size, "premises": list(range(size)),
                "conclusion": conjunction(list(range(size)))}
    if family == "and_elimination":
        _nat(index, "index")
        if index >= size:
            raise LibraryError("projection index outside conjunction")
        return {"atom_count": size, "premises": [conjunction(list(range(size)))],
                "conclusion": index}
    raise LibraryError("unsupported constructive family")


def formula_text(formula, atom_count: int, prefix="p") -> str:
    if type(formula) is int:
        if not 0 <= formula < atom_count:
            raise LibraryError("formula atom outside declared scope")
        return f"{prefix}{formula}"
    if not isinstance(formula, dict) or len(formula) != 1:
        raise LibraryError("invalid formula DSL")
    operator, arguments = next(iter(formula.items()))
    if operator not in ("and", "imp") or not isinstance(arguments, list) or len(arguments) != 2:
        raise LibraryError("only constructive And and implication formulas are supported")
    symbol = "∧" if operator == "and" else "→"
    return f"({formula_text(arguments[0], atom_count, prefix)} {symbol} {formula_text(arguments[1], atom_count, prefix)})"


def normalize_proposal(proposal: dict) -> dict:
    if not isinstance(proposal, dict):
        raise LibraryError("proposal must be a dictionary")
    allowed = {"family", "size", "index", "proof", "dependencies", "label_prefix", "provenance", "learning_example", "natural_language", "structural_hash"}
    if set(proposal) - allowed:
        raise LibraryError("unknown proposal fields; caller verification or raw Lean is forbidden")
    family, size = proposal["family"], _nat(proposal["size"], "size", 2)
    index = _nat(proposal.get("index", 0), "index") if family == "and_elimination" else 0
    statement(family, size, index)
    prefix = proposal.get("label_prefix", "T")
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z]+", prefix):
        raise LibraryError("label prefix must contain letters only")
    dependencies = proposal.get("dependencies", [])
    if not isinstance(dependencies, list) or len(set(dependencies)) != len(dependencies):
        raise LibraryError("dependencies must be a unique list")
    if any(not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{64}", identifier) for identifier in dependencies):
        raise LibraryError("dependencies must be accepted content hashes")
    provenance = proposal.get("provenance", {})
    if not isinstance(provenance, dict):
        raise LibraryError("provenance must be an object")
    result = {"family": family, "size": size, "index": index, "proof": copy.deepcopy(proposal["proof"]),
              "dependencies": sorted(dependencies), "label_prefix": prefix, "provenance": copy.deepcopy(provenance)}
    canonical(result)
    return result


def structural_hash(proposal: dict) -> str:
    return content_hash({"family": proposal["family"], "size": proposal["size"],
                         "index": proposal.get("index", 0) if proposal["family"] == "and_elimination" else 0})


def views(proposal: dict) -> dict:
    """Controlled theorem rendering and a valid/invalid constructive choice."""
    family, size, index = proposal["family"], proposal["size"], proposal.get("index", 0)
    shape = statement(family, size, index)
    prefix = proposal.get("label_prefix", "T")
    atoms = [f"{prefix}{i}" for i in range(shape["atom_count"])]
    premises = [formula_text(f, shape["atom_count"], prefix) for f in shape["premises"]]
    goal = formula_text(shape["conclusion"], shape["atom_count"], prefix)
    natural = "Assume " + "; ".join(premises) + ". Establish " + goal + "."
    if family == "implication_chain":
        rules = [f"r{i + 1}: {atoms[i]} implies {atoms[i + 1]}" for i in range(size)]
        context = f"Initially {atoms[0]} holds. " + "; ".join(rules) + f". Which ordered rule list establishes {atoms[-1]}?"
        good = ", ".join(f"r{i + 1}" for i in range(size))
        bad = ", ".join(["r2", "r1"] + [f"r{i + 1}" for i in range(2, size)])
    elif family == "and_introduction":
        context = "Known individual facts: " + ", ".join(atoms) + ". Which assembly supplies every conjunct exactly as required by " + goal + "?"
        good = "assemble " + ", ".join(atoms)
        bad = "assemble " + ", ".join(atoms[:-1] + [atoms[0]])
    else:
        context = "Known conjunction: " + premises[0] + f". Positions are numbered from zero. Which projection establishes {atoms[index]}?"
        good, bad = f"project position {index}", f"project position {(index + 1) % size}"
    certificate = (f"Starting from {atoms[0]}, apply implication assumptions in order from r1 through r{size}."
                   if family == "implication_chain" else
                   "Introduce the right-associated conjunction using the individual facts in their listed order."
                   if family == "and_introduction" else
                   f"Project the right component {index} times" + (", then the left component." if index < size - 1 else "."))
    natural += " Certificate: " + certificate
    if proposal.get("dependencies"):
        natural += " Formal proof reuse: " + ", ".join(proposal["dependencies"]) + "."
    answer = "A" if int(structural_hash(proposal)[-1], 16) % 2 == 0 else "B"
    choices = {"A": good if answer == "A" else bad, "B": good if answer == "B" else bad}
    prompt = context + f"\nA: {choices['A']}\nB: {choices['B']}\nAnswer:"
    return {"natural_language": natural,
            "learning_example": {"id": "construction_" + structural_hash(proposal), "prompt": prompt,
                                 "answer": answer, "choices": choices, "family": family, "split": "train",
                                 "provenance": {"structural_hash": structural_hash(proposal), "size": size,
                                                "constructive_operation": family, "source": "restricted_constructive_DSL"}}}


def _proof_text(proof, shape: dict, known: dict[str, dict], used: set[str], budget: list[int]) -> str:
    budget[0] -= 1
    if budget[0] < 0:
        raise LibraryError("proof exceeds the explicit per-proposal node budget")
    if not isinstance(proof, dict) or len(proof) != 1:
        raise LibraryError("invalid proof DSL; raw Lean is forbidden")
    key, value = next(iter(proof.items()))
    recur = lambda child: _proof_text(child, shape, known, used, budget)
    if key == "hyp":
        _nat(value, "hypothesis index")
        if value >= len(shape["premises"]):
            raise LibraryError("unknown hypothesis")
        return f"h{value}"
    if key in ("apply", "and_intro"):
        if not isinstance(value, list) or len(value) != 2:
            raise LibraryError("binary proof constructor requires two arguments")
        left, right = map(recur, value)
        return f"({left} ({right}))" if key == "apply" else f"(And.intro ({left}) ({right}))"
    if key in ("and_left", "and_right"):
        return f"(And.{'left' if key == 'and_left' else 'right'} ({recur(value)}))"
    if key == "lemma":
        if not isinstance(value, dict) or set(value) != {"id", "atoms", "arguments"}:
            raise LibraryError("lemma call needs exact ID, atoms and arguments")
        identifier = value["id"]
        if identifier not in known:
            raise LibraryError("dependency is not an earlier accepted lemma")
        dependency = known[identifier]
        atoms, arguments = value["atoms"], value["arguments"]
        if not isinstance(atoms, list) or len(atoms) != dependency["payload"]["atom_count"]:
            raise LibraryError("dependency atom arity mismatch")
        if not isinstance(arguments, list) or len(arguments) != len(dependency["payload"]["premises"]):
            raise LibraryError("dependency premise arity mismatch")
        used.add(identifier)
        substitutions = [formula_text(f, shape["atom_count"]) for f in atoms]
        return "(" + " ".join([dependency["lean_name"]] + [f"({a})" for a in substitutions] + [f"({recur(a)})" for a in arguments]) + ")"
    raise LibraryError("unsupported proof operator; axioms, classical and placeholders are forbidden")


def _declarations(proposal: dict, identifier: str, known: dict) -> dict:
    shape = statement(proposal["family"], proposal["size"], proposal["index"])
    used = set()
    term = _proof_text(proposal["proof"], shape, known, used, [100000])
    if used != set(proposal["dependencies"]):
        raise LibraryError("declared dependencies must exactly match proof lemma calls")
    atoms = " ".join(f"p{i}" for i in range(shape["atom_count"]))
    definition_name, name = "lf_def_" + identifier, "lf_lib_" + identifier
    definition = f"def {definition_name} ({atoms} : Prop) : Prop := {formula_text(shape['conclusion'], shape['atom_count'])}\n"
    binders = "\n".join(f"    (h{i} : {formula_text(f, shape['atom_count'])})" for i, f in enumerate(shape["premises"]))
    declaration = f"theorem {name} ({atoms} : Prop)\n{binders}\n    : {definition_name} {atoms} :=\n  {term}\n"
    return {"payload": {"family": proposal["family"], "size": proposal["size"], "index": proposal["index"], **shape, "proof": proposal["proof"]},
            "definition_name": definition_name, "definition": definition, "lean_name": name,
            "lean_declaration": declaration, "lean_proof": term}


def closed_formal_text(record: dict, known: dict[str, dict]) -> str:
    """Self-contained minimal prior-lemma closure, including fresh definitions."""
    ordered, seen = [], set()

    def visit(item):
        if item["id"] in seen:
            return
        for identifier in item["dependencies"]:
            visit(known[identifier])
        ordered.append(item)
        seen.add(item["id"])

    visit(record)
    return "import Init\nset_option autoImplicit false\n\n" + "\n".join(item["definition"] + item["lean_declaration"] for item in ordered)


class VerifiedLibrary:
    def __init__(self, root: Path | str, foundation: Foundation | None = None):
        self.root = Path(root).resolve()
        self.foundation = foundation or Foundation()
        if self.foundation.lean_toolchain != PIN or self.foundation.lean_commit != LEAN_COMMIT or self.foundation.logic != "constructive_prop_v1":
            raise LibraryError("unsupported foundation: pinned constructive Lean is required")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "records").mkdir(exist_ok=True)
        (self.root / "audits").mkdir(exist_ok=True)
        foundation_path = self.root / "foundation.json"
        expected = {"foundation": self.foundation.to_dict(), "foundation_hash": self.foundation.hash}
        if foundation_path.exists():
            if json.loads(foundation_path.read_text()) != expected:
                raise LibraryError("foundation changed; create a separate library")
        else:
            self._write_new(foundation_path, expected)
        pin_path = self.root / "lean-toolchain"
        if pin_path.exists() and pin_path.read_text().strip() != PIN:
            raise LibraryError("persisted Lean toolchain changed")
        if not pin_path.exists():
            pin_path.write_text(PIN + "\n")
        if not (self.root / "index.json").exists():
            self._write_new(self.root / "index.json", {"foundation_hash": self.foundation.hash, "entries": []})
        records = self.records()
        if records:
            with tempfile.TemporaryDirectory(prefix=".lean-replay-", dir=self.root) as temporary:
                result = self._compile(records, None, audit_directory=Path(temporary))
            if result["status"] != "verified":
                raise LibraryError("persisted library failed fresh Lean replay: " + result["reason"])

    @staticmethod
    def _write_new(path: Path, value: dict):
        with path.open("x", encoding="utf-8") as handle:
            handle.write(canonical(value) + "\n")

    def _foundation_check(self):
        expected = {"foundation": self.foundation.to_dict(), "foundation_hash": self.foundation.hash}
        if json.loads((self.root / "foundation.json").read_text()) != expected or (self.root / "lean-toolchain").read_text().strip() != PIN:
            raise LibraryError("immutable foundation or toolchain was modified")

    def records(self) -> list[dict]:
        self._foundation_check()
        index = json.loads((self.root / "index.json").read_text())
        if index["foundation_hash"] != self.foundation.hash:
            raise LibraryError("index foundation mismatch")
        if {entry["id"] + ".json" for entry in index["entries"]} != {path.name for path in (self.root / "records").glob("*.json")}:
            raise LibraryError("record set differs from index; truncation or unindexed records detected")
        result, known, structures = [], {}, set()
        for ordinal, entry in enumerate(index["entries"]):
            path = self.root / "records" / (entry["id"] + ".json")
            if file_hash(path) != entry["record_sha256"]:
                raise LibraryError("immutable record hash mismatch")
            record = json.loads(path.read_text())
            proposal = normalize_proposal(record["proposal"])
            identifier = content_hash({"foundation_hash": self.foundation.hash, "proposal": proposal})
            if identifier != record["id"] or identifier != entry["id"] or record["ordinal"] != ordinal:
                raise LibraryError("record identity/order mismatch")
            if record["content_hash"] != identifier or record["family"] != proposal["family"]:
                raise LibraryError("record content/family metadata differs from its canonical proposal")
            if record["foundation_hash"] != self.foundation.hash or record["previous_record"] != (result[-1]["id"] if result else None):
                raise LibraryError("record foundation/dependency chain mismatch")
            rebuilt = _declarations(proposal, identifier, known)
            if any(record[key] != value for key, value in rebuilt.items()):
                raise LibraryError("record declarations differ from restricted DSL")
            if record["structural_hash"] in structures or record["structural_hash"] != structural_hash(proposal):
                raise LibraryError("duplicate or changed structural theorem")
            if record["dependencies"] != proposal["dependencies"]:
                raise LibraryError("record dependency mismatch")
            if record["formal_text"] != closed_formal_text(record, known):
                raise LibraryError("formal dependency closure was changed")
            if any(record[key] != value for key, value in views(proposal).items()):
                raise LibraryError("controlled language/choice view was changed")
            verification = record["verification"]
            source, log = self.root / verification["source_path"], self.root / verification["log_path"]
            if not all(path.resolve().is_relative_to((self.root / "audits").resolve()) for path in (source, log)):
                raise LibraryError("verification artifacts must belong to this library's audit directory")
            if verification.get("lean") != "verified" or verification.get("axioms") != [] or file_hash(source) != verification["source_sha256"] or file_hash(log) != verification["log_sha256"]:
                raise LibraryError("missing or forged verification evidence")
            expected_source = self._source(result + [record])
            if source.read_text() != expected_source:
                raise LibraryError("compiler source does not match the cumulative library")
            if not self._audits(log.read_text(), result + [record]):
                raise LibraryError("retained axiom audit is incomplete")
            result.append(record)
            known[identifier] = record
            structures.add(record["structural_hash"])
        return result

    def snapshot(self) -> dict:
        records = self.records()
        payload = {"foundation_hash": self.foundation.hash, "record_ids": [r["id"] for r in records]}
        return {**payload, "record_count": len(records), "head": records[-1]["id"] if records else None,
                "snapshot_hash": content_hash(payload)}

    @staticmethod
    def _source(records: list[dict]) -> str:
        text = "import Init\nset_option autoImplicit false\nset_option linter.unusedVariables false\n\n"
        for record in records:
            text += record["definition"] + record["lean_declaration"]
            for name in (record["definition_name"], record["lean_name"]):
                text += f"#print axioms {name}\n"
            text += "\n"
        return text

    @staticmethod
    def _audits(output: str, records: list[dict]) -> bool:
        return all(re.search(rf"(?m)^'{re.escape(record[key])}' does not depend on any axioms\s*$", output)
                   for record in records for key in ("definition_name", "lean_name"))

    def _compile(self, records: list[dict], lean, audit_directory: Path | None = None) -> dict:
        candidate = str(lean) if lean is not None else str(Path.home() / ".elan/bin/lean")
        executable = shutil.which(candidate)
        if executable is None:
            return {"status": "not_run", "reason": "lean_executable_not_found"}
        # The compiler is a trust anchor. An arbitrary script printing successful
        # audit sentences is not an alternative Lean installation.
        trusted = Path.home() / ".elan/bin/lean"
        elan = Path.home() / ".elan/bin/elan"
        allowed = {trusted.resolve()}
        environment = dict(os.environ)
        cleared = ("ELAN_TOOLCHAIN", "LEAN_PATH", "LEAN_SRC_PATH", "LEAN_SYSROOT")
        for variable in cleared:
            environment.pop(variable, None)
        actual_compiler = None
        if elan.exists():
            try:
                selected = subprocess.run([str(elan), "which", "lean"], cwd=self.root, env=environment,
                                          capture_output=True, text=True, timeout=15, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return {"status": "error", "reason": f"compiler_resolution_failed: {exc}"}
            if selected.returncode == 0:
                actual_compiler = Path(selected.stdout.strip()).resolve()
                allowed.add(actual_compiler)
        if Path(executable).resolve() not in allowed:
            return {"status": "error", "reason": "untrusted_lean_executable"}
        if actual_compiler is None or not actual_compiler.is_file():
            return {"status": "not_run", "reason": "pinned_lean_toolchain_not_installed"}
        token = uuid.uuid4().hex
        directory = audit_directory or self.root / "audits"
        source_path, log_path = directory / f"{token}.lean", directory / f"{token}.log"
        source_path.write_text(self._source(records), encoding="utf-8")
        try:
            version = subprocess.run([executable, "--version"], cwd=self.root, env=environment,
                                     capture_output=True, text=True, timeout=15, check=False)
            actual_version = (version.stdout + version.stderr).strip()
            if version.returncode != 0 or "version 4.35.0," not in actual_version or LEAN_COMMIT not in actual_version:
                return {"status": "error", "reason": "foundation_compiler_version_mismatch"}
            process = subprocess.run([executable, str(source_path)], cwd=self.root, env=environment,
                                     capture_output=True, text=True, timeout=60, check=False)
            output = process.stdout + process.stderr
            log_path.write_text(output, encoding="utf-8")
            if process.returncode != 0 or not self._audits(output, records):
                return {"status": "error", "reason": "lean_proof_or_zero_axiom_audit_failed",
                        "source_path": str(source_path.relative_to(self.root)), "log_path": str(log_path.relative_to(self.root))}
            return {"status": "verified", "reason": "real_lean_zero_axiom_audits", "lean": "verified", "axioms": [],
                    "actual_version": actual_version, "executable": executable,
                    "launcher_sha256": file_hash(Path(executable)),
                    "compiler_path": str(actual_compiler), "compiler_sha256": file_hash(actual_compiler),
                    "cleared_environment_overrides": list(cleared),
                    "source_path": str(source_path.relative_to(self.root)), "source_sha256": file_hash(source_path),
                    "log_path": str(log_path.relative_to(self.root)), "log_sha256": file_hash(log_path),
                    "audited_declarations": len(records) * 2}
        except (OSError, subprocess.TimeoutExpired) as exc:
            log_path.write_text(str(exc), encoding="utf-8")
            return {"status": "error", "reason": f"compiler_execution_failed: {exc}"}

    def accept(self, proposal: dict, lean=None) -> dict:
        try:
            lock = self.root / ".writer-lock"
            lock.mkdir()
        except FileExistsError:
            return {"status": "rejected", "reason": "library_writer_already_active", "record": None}
        try:
            records = self.records()
            normalized = normalize_proposal(proposal)
            shape_hash = structural_hash(normalized)
            if any(record["structural_hash"] == shape_hash for record in records):
                return {"status": "rejected", "reason": "structural_duplicate", "record": None}
            known = {record["id"]: record for record in records}
            identifier = content_hash({"foundation_hash": self.foundation.hash, "proposal": normalized})
            declarations = _declarations(normalized, identifier, known)
            record = {"id": identifier, "content_hash": identifier, "foundation_hash": self.foundation.hash,
                      "ordinal": len(records), "previous_record": records[-1]["id"] if records else None,
                      "family": normalized["family"], "structural_hash": shape_hash,
                      "dependencies": normalized["dependencies"], "proposal": normalized,
                      "provenance": {**normalized["provenance"], "generator_scope": "parameterized constructive DSL; no novel mathematics claim"},
                      **declarations, **views(normalized)}
            record["formal_text"] = closed_formal_text(record, known)
            checked = self._compile(records + [record], lean)
            if checked["status"] != "verified":
                return {**checked, "status": "not_run" if checked["status"] == "not_run" else "rejected", "record": None}
            record["verification"] = checked
            record["accepted_at_utc"] = datetime.now(timezone.utc).isoformat()
            path = self.root / "records" / f"{identifier}.json"
            self._write_new(path, record)
            index_path = self.root / "index.json"
            index = json.loads(index_path.read_text())
            index["entries"].append({"id": identifier, "record_sha256": file_hash(path)})
            temporary = self.root / (".index-" + uuid.uuid4().hex)
            temporary.write_text(canonical(index) + "\n")
            os.replace(temporary, index_path)
            return {"status": "accepted", "reason": "real_lean_zero_axiom_audits", "record": copy.deepcopy(record),
                    "snapshot": self.snapshot()}
        except (LibraryError, KeyError, TypeError, ValueError, OSError, RecursionError) as exc:
            return {"status": "rejected", "reason": str(exc), "record": None}
        finally:
            lock.rmdir()
