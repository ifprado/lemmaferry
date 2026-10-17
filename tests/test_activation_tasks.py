from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lemmaferry.logic import Edge, make_example, validate_path

spec = importlib.util.spec_from_file_location("prepare_activation_tasks", ROOT / "experiments/prepare_activation_tasks.py")
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)


class ChoiceTaskTests(unittest.TestCase):
    def test_candidate_labels_and_text_control_preserve_information(self):
        for split in ("train", "validation", "test"):
            for i in range(10):
                r = make_example(i, split, 7)
                # This fixture tests view construction; it is not compiler evidence.
                r["verification"].update(lean="verified", lean_axioms="none")
                view = tasks.choice_view(r)
                edges = [Edge(**e) for e in r["edges"]]
                for label, chain in view["choices"].items():
                    self.assertEqual(validate_path(edges, r["source_atom"], r["target_atom"], chain), label == view["answer"])
                self.assertEqual(len(view["choices"]["A"]), len(view["choices"]["B"]))
                self.assertEqual(view["text"], view["text_baseline"])
                self.assertNotIn("Verified mathematical certificate", view["ordinary_text"])
                self.assertTrue(view["text"].endswith("Answer:"))

    def test_unchecked_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Lean-verified"):
            tasks.choice_view(make_example(0))


if __name__ == "__main__":
    unittest.main()
