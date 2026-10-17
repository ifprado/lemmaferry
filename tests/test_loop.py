"""Integration tests for the real checker / optimizer / feedback boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from lemmaferry.loop import LoopConfig, run_loop


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_LEAN = (Path.home() / ".elan/bin/lean").exists()
ROOT = Path(__file__).resolve().parents[1]


class LoopConfigurationTests(unittest.TestCase):
    def test_requires_translation_and_immutable_pretrained_revisions(self):
        with self.assertRaises(ValueError):
            LoopConfig(experience_mode="translated", representations=False)
        with self.assertRaises(ValueError):
            LoopConfig(backend="pretrained", recipient_model="anything", recipient_revision="main")


@unittest.skipUnless(HAS_TORCH and HAS_LEAN, "optional Torch runtime and pinned Lean required")
class ConstructiveLoopTests(unittest.TestCase):
    def test_two_complete_translated_cycles_reuse_library_and_persistent_weights(self):
        import torch
        from lemmaferry.learning import load_evaluation, load_recipient, parameter_sha256
        from lemmaferry.library import VerifiedLibrary
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            result = run_loop(output, ROOT / "benchmarks/independent-v1.jsonl",
                              config=LoopConfig(iterations=2, steps=1, max_length=2048,
                                                experience_mode="translated"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["library"]["record_count"], 12)
            self.assertEqual(result["guided_iterations"], 1)
            self.assertFalse(result["evaluation_used_for_generation"])
            self.assertFalse(result["sustained_broad_improvement_established"])
            self.assertGreater(result["iterations"][1]["dependency_reusing_records"], 0)
            frozen = load_evaluation(output / "evaluation.json")
            for name in ("verified", "ordinary_control", "text_control", "shuffled_control", "mean_control"):
                first = result["iterations"][0]["training"][name]
                second = result["iterations"][1]["training"][name]
                self.assertEqual(first["parameter_sha256_after"], second["parameter_sha256_before"])
                self.assertEqual(second["cumulative_updates"], 2)
                reloaded = load_recipient(output / "iteration-001" / f"{name}.pt")
                self.assertEqual(parameter_sha256(reloaded), second["parameter_sha256_after"])
                self.assertEqual(reloaded.frozen_evaluation_sha256, frozen.sha256)
            for iteration in result["iterations"]:
                self.assertFalse(iteration["feedback_for_next_generation"]["external_evaluation_used"])
                self.assertEqual(len({r["processed_padded_tokens"] for r in iteration["training"].values()}), 1)
                self.assertEqual(len({r["supervised_answer_tokens"] for r in iteration["training"].values()}), 1)
                self.assertTrue(iteration["calibration"])
            proposals = json.loads((output / "iteration-001/proposals.json").read_text())
            self.assertTrue(all(p["provenance"]["curriculum_feedback"]["source"] == "training_curriculum_only"
                                for p in proposals))
            reopened = VerifiedLibrary(output / "library")
            self.assertEqual(reopened.snapshot(), result["library"])
            specification = importlib.util.spec_from_file_location("verify_constructive_run", ROOT / "experiments/verify_constructive_run.py")
            verifier = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(verifier)
            checked = verifier.verify_run(output)
            self.assertEqual(len(checked["standalone_checkpoint_reloads"]), 10)
            inventory = output / "artifact-sha256.json"
            hashes = json.loads(inventory.read_text())
            hashes.pop(next(iter(hashes)))
            inventory.write_text(json.dumps(hashes))
            with self.assertRaisesRegex(ValueError, "inventory"):
                verifier.verify_run(output)
            with self.assertRaisesRegex(ValueError, "fresh output"):
                run_loop(output, ROOT / "benchmarks/independent-v1.jsonl")


if __name__ == "__main__":
    unittest.main()
