from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from lemmaferry.generation import make_proposal, propose
from lemmaferry.library import Foundation, LibraryError, VerifiedLibrary

LEAN = Path.home() / ".elan/bin/lean"


class LibraryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.library = VerifiedLibrary(self.root)

    def test_foundation_is_frozen_and_existing_library_cannot_switch(self):
        with self.assertRaises(FrozenInstanceError):
            self.library.foundation.logic = "classical"
        with self.assertRaisesRegex(LibraryError, "foundation changed"):
            VerifiedLibrary(self.root, replace(Foundation(), name="different_foundation"))
        with self.assertRaisesRegex(LibraryError, "unsupported foundation"):
            VerifiedLibrary(self.root / "other", replace(Foundation(), lean_toolchain="leanprover/lean4:v4.36.0"))

    def test_missing_executable_is_not_run_and_does_not_change_snapshot(self):
        before = self.library.snapshot()
        result = self.library.accept(make_proposal("implication_chain", 2), lean=self.root / "absent")
        self.assertEqual(result["status"], "not_run")
        self.assertEqual(self.library.snapshot(), before)

    def test_caller_verified_flags_raw_lean_and_classical_are_rejected(self):
        original = make_proposal("implication_chain", 2)
        for mutation in (dict(verification={"lean": "verified"}), dict(lean_declaration="axiom escape : False")):
            self.assertEqual(self.library.accept({**original, **mutation})["status"], "rejected")
        for proof in ({"raw": "by sorry"}, {"Classical.choice": {"hyp": 0}}, {"axiom": "False"}):
            self.assertEqual(self.library.accept({**original, "proof": proof})["status"], "rejected")
        self.assertEqual(self.library.snapshot()["record_count"], 0)

    def test_forged_compiler_success_script_is_not_a_trust_anchor(self):
        fake = self.root / "fake-lean"
        fake.write_text("#!/bin/sh\nprintf 'Lean (version 4.35.0, commit eb6cafb7290e8bd5f2937522d8d804f58f02bafc)\\n'\n")
        fake.chmod(0o755)
        result = self.library.accept(make_proposal("implication_chain", 2), lean=fake)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "untrusted_lean_executable")

    def test_persisted_pin_modification_is_detected(self):
        (self.root / "lean-toolchain").write_text("leanprover/lean4:nightly\n")
        with self.assertRaisesRegex(LibraryError, "modified"):
            self.library.records()


@unittest.skipUnless(LEAN.exists(), "official pinned elan Lean is required for real verification")
class RealLeanLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.library = VerifiedLibrary(self.root)

    def accept(self, proposal):
        result = self.library.accept(proposal, lean=LEAN)
        self.assertEqual(result["status"], "accepted", result)
        return result["record"]

    def test_two_cycles_accumulate_actual_dependencies_and_fresh_definitions(self):
        for candidate in propose(self.library, iteration=0, count=3, seed=7):
            self.accept(candidate)
        first = self.library.snapshot()
        old_bytes = {p.name: p.read_bytes() for p in (self.root / "records").glob("*.json")}
        for candidate in propose(self.library, iteration=1, count=3, seed=7):
            record = self.accept(candidate)
            self.assertTrue(record["dependencies"])
            self.assertIn(record["dependencies"][0], first["record_ids"])
            self.assertEqual(record["verification"]["axioms"], [])
            self.assertIn("def " + record["definition_name"], record["formal_text"])
        second = self.library.snapshot()
        self.assertEqual(second["record_count"], 6)
        self.assertNotEqual(first["snapshot_hash"], second["snapshot_hash"])
        self.assertEqual(second["record_ids"][:3], first["record_ids"])
        for name, payload in old_bytes.items():
            self.assertEqual((self.root / "records" / name).read_bytes(), payload)
        self.assertEqual(VerifiedLibrary(self.root).snapshot(), second)

    def test_reopening_replays_real_lean_without_changing_any_persisted_file(self):
        self.accept(make_proposal("implication_chain", 2))
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        reopened = VerifiedLibrary(self.root)
        self.assertEqual(reopened.snapshot(), self.library.snapshot())
        after = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_wrong_typed_proof_fails_real_lean_and_retains_failure(self):
        candidate = make_proposal("implication_chain", 2)
        candidate["proof"] = {"hyp": 0}
        result = self.library.accept(candidate, lean=LEAN)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "lean_proof_or_zero_axiom_audit_failed")
        self.assertIn("error:", (self.root / result["log_path"]).read_text())
        self.assertEqual(self.library.snapshot()["record_count"], 0)

    def test_renaming_and_different_proof_spelling_do_not_create_novelty(self):
        original = make_proposal("and_introduction", 2)
        self.accept(original)
        duplicate = copy.deepcopy(original)
        duplicate["label_prefix"] = "Renamed"
        duplicate["provenance"] = {"iteration": 999, "seed": 999}
        duplicate["proof"] = {"and_intro": [{"and_left": {"and_intro": [{"hyp": 0}, {"hyp": 0}]}}, {"hyp": 1}]}
        self.assertEqual(self.library.accept(duplicate)["reason"], "structural_duplicate")
        self.assertEqual(self.library.snapshot()["record_count"], 1)

    def test_dependency_must_exist_be_used_and_have_correct_instantiation(self):
        previous = self.accept(make_proposal("implication_chain", 2))
        candidate = make_proposal("implication_chain", 3, self.library)
        candidate["dependencies"] = []
        self.assertIn("exactly match", self.library.accept(candidate)["reason"])
        candidate = make_proposal("and_introduction", 2)
        candidate["dependencies"] = [previous["id"]]
        self.assertIn("exactly match", self.library.accept(candidate)["reason"])
        candidate = make_proposal("implication_chain", 3, self.library)
        candidate["proof"]["apply"][1]["lemma"]["atoms"] = [0, 2, 1]
        result = self.library.accept(candidate)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "lean_proof_or_zero_axiom_audit_failed")
        candidate = make_proposal("implication_chain", 3, self.library)
        candidate["dependencies"] = ["0" * 64]
        candidate["proof"]["apply"][1]["lemma"]["id"] = "0" * 64
        self.assertIn("not an earlier", self.library.accept(candidate)["reason"])

    def test_forged_persisted_evidence_and_record_mutation_fail_closed(self):
        record = self.accept(make_proposal("and_elimination", 2))
        log = self.root / record["verification"]["log_path"]
        log.write_text("compiler exited zero but no audit was retained\n")
        with self.assertRaisesRegex(LibraryError, "forged verification"):
            self.library.records()
        with self.assertRaises(LibraryError):
            VerifiedLibrary(self.root)

    def test_index_truncation_and_record_mutation_are_detected(self):
        record = self.accept(make_proposal("implication_chain", 2))
        index = self.root / "index.json"
        original = index.read_text()
        value = json.loads(original)
        value["entries"] = []
        index.write_text(json.dumps(value))
        with self.assertRaisesRegex(LibraryError, "truncation"):
            self.library.snapshot()
        index.write_text(original)
        path = self.root / "records" / (record["id"] + ".json")
        path.write_text(path.read_text().replace('"lean":"verified"', '"lean":"not_run"'))
        with self.assertRaisesRegex(LibraryError, "record hash"):
            self.library.records()

    def test_actual_compiler_hash_is_separate_from_launcher_and_environment_is_cleared(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"LEAN_PATH": "/wrong/path", "LEAN_SRC_PATH": "/wrong/source", "LEAN_SYSROOT": "/wrong/sysroot", "ELAN_TOOLCHAIN": "not-installed"}):
            record = self.accept(make_proposal("and_introduction", 2))
        verification = record["verification"]
        self.assertIn("toolchains", verification["compiler_path"])
        self.assertNotEqual(verification["launcher_sha256"], verification["compiler_sha256"])
        self.assertEqual(set(verification["cleared_environment_overrides"]), {"ELAN_TOOLCHAIN", "LEAN_PATH", "LEAN_SRC_PATH", "LEAN_SYSROOT"})

    def test_forged_metadata_is_rejected_even_if_file_hash_is_rewritten(self):
        from lemmaferry.library import file_hash
        record = self.accept(make_proposal("implication_chain", 2))
        path = self.root / "records" / (record["id"] + ".json")
        value = json.loads(path.read_text())
        value["family"] = "and_introduction"
        path.write_text(json.dumps(value))
        index_path = self.root / "index.json"
        index = json.loads(index_path.read_text())
        index["entries"][0]["record_sha256"] = file_hash(path)
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(LibraryError, "canonical proposal"):
            self.library.records()

    def test_closed_formal_view_compiles_without_unrelated_library_items(self):
        self.accept(make_proposal("implication_chain", 2))
        unrelated = self.accept(make_proposal("and_introduction", 2))
        record = self.accept(make_proposal("implication_chain", 7, self.library))
        self.assertNotIn(unrelated["lean_name"], record["formal_text"])
        source = self.root / "closed.lean"
        source.write_text(record["formal_text"] + f"\n#print axioms {record['lean_name']}\n")
        result = subprocess.run([str(LEAN), str(source)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"'{record['lean_name']}' does not depend on any axioms", result.stdout)

    def test_sizes_are_parameterized_beyond_small_seed_shapes(self):
        for family, size in (("implication_chain", 17), ("and_introduction", 19), ("and_elimination", 23)):
            self.accept(make_proposal(family, size))
        self.assertEqual(self.library.snapshot()["record_count"], 3)


if __name__ == "__main__":
    unittest.main()
