from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lemmaferry.logic import (Edge, SPLIT_SETTINGS, check_example, generate_examples,
                             make_example, parse_problem, shortest_path, validate_path)


class GraphCertificateTests(unittest.TestCase):
    def test_bfs_handles_cycles_and_unreachable_targets(self):
        edges = [Edge("e0", 0, 1), Edge("e1", 1, 0), Edge("e2", 1, 2), Edge("e3", 0, 2)]
        self.assertEqual(shortest_path(edges, 0, 2), ["e3"])
        self.assertIsNone(shortest_path(edges, 0, 99))
        self.assertEqual(shortest_path(edges, 0, 0), [])

    def test_certificate_rejects_disconnected_or_unknown_steps(self):
        edges = [Edge("e0", 0, 1), Edge("e1", 2, 3)]
        self.assertFalse(validate_path(edges, 0, 3, ["e0", "e1"]))
        self.assertFalse(validate_path(edges, 0, 1, ["unknown"]))
        self.assertTrue(validate_path(edges, 0, 1, ["e0"]))

    def test_generated_corpus_certificates_across_all_splits(self):
        for split in SPLIT_SETTINGS:
            for example in generate_examples(20, split, 19):
                check_example(example)
                self.assertGreater(len(example["long_path"]), len(example["short_path"]))
                self.assertEqual(example["verification"]["lean"], "not_run")

    def test_structure_template_and_lexicon_holdouts(self):
        examples = [make_example(0, split, 7) for split in SPLIT_SETTINGS]
        self.assertEqual(len({e["family"] for e in examples}), 3)
        self.assertEqual(len({e["rendering_template"] for e in examples}), 3)
        self.assertEqual([len(e["short_path"]) for e in examples], [2, 3, 4])
        lexicons = [{a["label"] for a in e["atoms"]} for e in examples]
        self.assertFalse(lexicons[0] & lexicons[1] | lexicons[0] & lexicons[2] | lexicons[1] & lexicons[2])

    def test_reproducibility(self):
        self.assertEqual(make_example(4, "test", 11), make_example(4, "test", 11))
        self.assertNotEqual(make_example(4, "test", 11)["edges"], make_example(4, "test", 12)["edges"])

    def test_mismatched_text_and_forged_lean_are_rejected(self):
        original = make_example(0)
        text = copy.deepcopy(original)
        text["natural_language"] = text["natural_language"].replace("is active.", "is asleep.", 1)
        with self.assertRaises(ValueError):
            check_example(text)
        proof = copy.deepcopy(original)
        proof["lean_declaration"] += "\naxiom secret : False\n"
        with self.assertRaises(ValueError):
            check_example(proof)

    def test_nonshortest_answer_is_rejected(self):
        example = make_example(0)
        example["short_path"] = example["long_path"]
        with self.assertRaises(ValueError):
            check_example(example)

    def test_parser_rejects_unrecognized_sentences(self):
        with self.assertRaises(ValueError):
            parse_problem("Make something up.", "stations")


class LeanStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("check_lean", ROOT / "experiments" / "check_lean.py")
        cls.checker = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.checker)

    def test_absent_lean_never_becomes_verified(self):
        example = make_example(0)
        example["verification"]["lean"] = "verified"  # Stale success must be cleared.
        example["verification"].update(short_proof="verified", long_proof="verified",
                                       lean_version="old compiler", lean_log="old.log",
                                       lean_executable="/old/lean")
        counts = self.checker.check_records([example], lean=None)
        self.assertEqual(counts, {"verified": 0, "not_run": 1, "error": 0})
        self.assertEqual(example["verification"]["lean"], "not_run")
        self.assertEqual(example["verification"]["short_proof"], "not_run")
        self.assertEqual(example["verification"]["long_proof"], "not_run")
        for field in ("lean_version", "lean_log", "lean_executable"):
            self.assertNotIn(field, example["verification"])

    def test_exit_zero_without_axiom_output_is_not_verification(self):
        example = make_example(0)
        fake_exit_zero = subprocess.CompletedProcess(["fake-lean"], 0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as temporary:
            # This tests the subprocess boundary only. No Lean compiler ran.
            with patch.object(self.checker.subprocess, "run", return_value=fake_exit_zero):
                passed, _output, reason = self.checker.run_batch(
                    [example], "fake-lean", 1, Path(temporary), "empty_output")
        self.assertFalse(passed)
        self.assertEqual(reason, "axiom_audit_failed_or_unrecognized")

    def test_axiom_audit_is_fail_closed(self):
        self.assertTrue(self.checker.audit_no_axioms("'lf_train_000000' does not depend on any axioms\n", "lf_train_000000"))
        self.assertFalse(self.checker.audit_no_axioms("'lf_train_000000' depends on axioms: [sorryAx]", "lf_train_000000"))
        self.assertFalse(self.checker.audit_no_axioms("", "lf_train_000000"))
        self.assertFalse(self.checker.audit_no_axioms("'other' does not depend on any axioms\n", "lf_train_000000"))

    def test_forged_source_fails_before_lean_invocation(self):
        example = make_example(0)
        example["lean_declaration"] = "theorem danger : False := by sorry"
        counts = self.checker.check_records([example], lean=None)
        self.assertEqual(counts["error"], 1)
        self.assertEqual(example["verification"]["lean"], "error")


if __name__ == "__main__":
    unittest.main()
