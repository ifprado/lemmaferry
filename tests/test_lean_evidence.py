from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lemmaferry.logic import make_example

spec = importlib.util.spec_from_file_location("verify_lean_evidence", ROOT / "experiments/verify_lean_evidence.py")
evidence = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(evidence)


class LeanEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.inputs, self.results = self.base / "inputs", self.base / "results"
        self.inputs.mkdir()
        self.results.mkdir()
        self.toolchain = self.base / "lean-toolchain"
        self.toolchain.write_text("leanprover/lean4:v4.35.0\n")
        version = "Lean (version 4.35.0, mock evidence fixture)"
        (self.results / "lean-version.txt").write_text(version + "\n")
        # These fixtures exercise evidence validation only. No compiler ran here.
        for split in ("train", "validation", "test"):
            record = make_example(0, split, 7)
            (self.inputs / f"{split}.jsonl").write_text(json.dumps(record) + "\n")
            log = self.results / f"{split}.log"
            log.write_text("\n".join(f"'{name}' does not depend on any axioms" for name in
                                     (record["lean_name"], record["lean_name"] + "_long")))
            record["verification"].update(lean="verified", lean_axioms="none",
                                           short_proof="verified", long_proof="verified",
                                           reason="verified", lean_version=version, lean_log=str(log))
            (self.results / f"{split}.checked.jsonl").write_text(json.dumps(record) + "\n")

    def verify(self, count=1):
        return evidence.verify(self.inputs, self.results, self.base, self.toolchain, count)

    def test_complete_fixture_has_hashes_for_records_and_audits(self):
        result = self.verify()
        self.assertEqual(result["total_checked_proofs"], 6)
        self.assertIn(str(self.inputs / "train.jsonl"), result["sha256"])
        self.assertIn(str((self.results / "train.log").resolve()), result["sha256"])

    def test_missing_long_audit_is_rejected_despite_verified_status(self):
        log = self.results / "train.log"
        log.write_text(log.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "lacks the zero-axiom audit"):
            self.verify()

    def test_checked_payload_must_equal_input(self):
        path = self.results / "train.checked.jsonl"
        record = json.loads(path.read_text())
        record["seed"] = 999
        path.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "differs from generated input"):
            self.verify()

    def test_insufficient_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 20"):
            self.verify(count=20)

    def test_wrong_compiler_version_is_rejected(self):
        (self.results / "lean-version.txt").write_text("Lean (version 4.34.0)\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
