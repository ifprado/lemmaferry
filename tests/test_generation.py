from pathlib import Path
import tempfile
import unittest

from lemmaferry.generation import make_proposal, propose
from lemmaferry.library import FAMILIES, LibraryError, VerifiedLibrary


class ConstructiveGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.library = VerifiedLibrary(Path(self.temporary.name))

    def test_seed_proposals_cover_structures_with_unique_nontrivial_choices(self):
        candidates = propose(self.library, count=3, iteration=0, seed=7)
        self.assertEqual({p["family"] for p in candidates}, set(FAMILIES))
        self.assertEqual(len({p["structural_hash"] for p in candidates}), 3)
        self.assertEqual(candidates, propose(self.library, count=3, iteration=0, seed=7))
        for proposal in candidates:
            example = proposal["learning_example"]
            self.assertIn(example["answer"], ("A", "B"))
            self.assertNotEqual(example["choices"]["A"], example["choices"]["B"])
            self.assertEqual(list(example["choices"]), ["A", "B"])

    def test_training_loss_changes_family_budget_and_mastery_changes_size(self):
        weak = {"family_losses": {"and_introduction": 3.0, "implication_chain": 0.2},
                "mastery_threshold": .7, "source": "training_curriculum_only"}
        candidates = propose(self.library, count=6, iteration=2, seed=7, feedback=weak)
        self.assertGreater(sum(p["family"] == "and_introduction" for p in candidates), 2)
        self.assertTrue(all(p["provenance"]["curriculum_feedback"] == weak for p in candidates))
        mastered = [p for p in candidates if p["family"] == "implication_chain"]
        self.assertTrue(mastered)
        self.assertEqual(mastered[0]["provenance"]["size_increment"], 2)
        self.assertEqual(len({p["structural_hash"] for p in candidates}), 6)

    def test_external_test_feedback_and_unbounded_attempt_counts_are_rejected(self):
        with self.assertRaises(LibraryError):
            propose(self.library, feedback={"source": "external_test", "family_losses": {"implication_chain": 0.1}})
        with self.assertRaises(LibraryError):
            propose(self.library, count=10001)

    def test_structural_identity_ignores_cosmetic_labels(self):
        a = make_proposal("implication_chain", 31, label_prefix="Aster")
        b = make_proposal("implication_chain", 31, label_prefix="Birch")
        self.assertEqual(a["structural_hash"], b["structural_hash"])
        self.assertNotEqual(a["natural_language"], b["natural_language"])


if __name__ == "__main__":
    unittest.main()
