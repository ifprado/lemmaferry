"""Dependency-free interface checks for the library-to-Rosetta iteration bridge."""

import copy
import hashlib
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lemmaferry.correspondence import load_correspondence
from lemmaferry.representations import fit_library_rosetta
from lemmaferry.translators import load_translator


class FixtureEncoder:
    backend = "deterministic_test_fixture"
    representation_boundary = "fixture vector, not a language model activation"
    hidden_size = 3

    def __init__(self, role):
        self.config = {"role": role, "max_length": 10000}
        self.state = "a" * 64

    def parameter_sha256(self):
        return self.state


def fixture_extract(model, prompts):
    return np.asarray([[len(p), sum(p.encode()) % 37, hashlib.sha256(p.encode()).digest()[0]]
                       for p in prompts], dtype=float)


def records():
    result = []
    shapes = (
        ("and_elimination", [{"and": [0, 1]}], {"and_left": {"hyp": 0}}),
        ("and_introduction", [0, 1], {"and_intro": [{"hyp": 0}, {"hyp": 1}]}),
        ("implication_chain", [0, {"imp": [0, 1]}, {"imp": [1, 2]}],
         {"apply": [{"hyp": 2}, {"apply": [{"hyp": 1}, {"hyp": 0}]}]}),
    )
    for family, premises, proof in shapes:
        for index in range(2):
            identifier = f"{family}_{index}"
            result.append({
                "id": identifier, "family": family, "formal_text": f"theorem {identifier} := formal_fixture_{index}",
                "natural_language": f"Constructive {family} statement {index}.",
                "payload": {"atom_count": 3 if family == "implication_chain" else 2,
                            "premises": premises, "proof": proof},
                "proposal": {"label_prefix": "T"}, "dependencies": [],
                "provenance": {"scope": "test fixture, not a verified library"},
            })
    return result


class RepresentationBridgeTests(unittest.TestCase):
    def fit(self, root, data=None, extractor=fixture_extract):
        return fit_library_rosetta(data or records(), "f" * 64,
                                   FixtureEncoder("math"), FixtureEncoder("language"), root,
                                   lambdas=(0, 1), extractor=extractor)

    def test_bridge_exports_distinct_views_maps_targets_and_model_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "translation"
            result = self.fit(path)
            self.assertEqual(set(result), {"pair", "report", "translated_targets", "language_targets", "shuffled_targets", "mean_targets"})
            self.assertEqual(len(result["translated_targets"]), 6)
            self.assertTrue(all(len(value) == 3 for value in result["translated_targets"].values()))
            self.assertEqual(set(result["shuffled_targets"]), set(result["translated_targets"]))
            self.assertEqual(set(result["mean_targets"]), set(result["translated_targets"]))
            manifest = load_correspondence(path / "views")
            self.assertIn("take the left conjunct", manifest["splits"]["train"]["pairs"][0]["language_text"])
            self.assertIn("form the conjunction", manifest["splits"]["val"]["pairs"][0]["language_text"])
            self.assertIn("apply", manifest["splits"]["test"]["pairs"][0]["language_text"])
            self.assertNotEqual(manifest["splits"]["train"]["math"]["sha256"],
                                manifest["splits"]["train"]["language"]["sha256"])
            report = json.loads((path / "results.json").read_text())
            extraction = report["protocol"]["extraction"]
            self.assertTrue(extraction["parameters_unchanged"])
            self.assertIn("pipeline fixture", extraction["models"]["math"]["role"])
            self.assertEqual(extraction["library_snapshot_sha256"], "f" * 64)
            reloaded = load_translator(path / "maps")
            with np.load(path / "activations.npz", allow_pickle=False) as archive:
                expected = reloaded.to_language(archive["math"])
                for identifier, value in zip(archive["ids"].tolist(), expected):
                    np.testing.assert_array_equal(result["translated_targets"][identifier], value)
                shuffled = reloaded.to_language(archive["math"], control="shuffled")
                for identifier, value in zip(archive["ids"].tolist(), shuffled):
                    np.testing.assert_array_equal(result["shuffled_targets"][identifier], value)
                    mean = reloaded.to_language(archive["math"], control="mean")
                    np.testing.assert_array_equal(result["mean_targets"][identifier], mean[0])
            # View tampering is detected by the same public correspondence loader.
            with (path / "views" / "math" / "train.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_correspondence(path / "views")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch runtime required")
    def test_swap_bridge_saved_controls_match_frozen_target_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swap"
            math, language = FixtureEncoder("math"), FixtureEncoder("language")
            result = fit_library_rosetta(records(), "f" * 64, math, language, path,
                        translator="swap", extractor=fixture_extract, seed=21,
                        swap_config={"latent_dim": 2, "hidden_dim": 4, "steps": 8,
                                     "batch_size": 2, "learning_rate": .01, "validation_every": 4})
            self.assertEqual(math.state, "a" * 64)
            self.assertEqual(language.state, "a" * 64)
            reloaded = load_translator(path / "maps")
            with np.load(path / "activations.npz", allow_pickle=False) as arrays:
                ids, source = arrays["ids"].tolist(), arrays["math"]
                for control, key in (("paired", "translated_targets"), ("shuffled", "shuffled_targets"),
                                     ("mean", "mean_targets")):
                    expected = reloaded.to_language(source, control=control)
                    for identifier, vector in zip(ids, expected):
                        np.testing.assert_allclose(result[key][identifier], vector, rtol=0, atol=0)
            # The next iteration must extract the recipient's current weights;
            # previous fitted targets remain plain detached numeric snapshots.
            language.state = "c" * 64
            newer = fit_library_rosetta(records(), "e" * 64, math, language, Path(tmp) / "next",
                        translator="swap", extractor=fixture_extract, seed=21,
                        swap_config={"latent_dim": 2, "hidden_dim": 4, "steps": 8,
                                     "batch_size": 2, "learning_rate": .01, "validation_every": 4})
            old_hash = result["report"]["protocol"]["extraction"]["trainable_parameter_sha256_before"]["language"]
            new_hash = newer["report"]["protocol"]["extraction"]["trainable_parameter_sha256_before"]["language"]
            self.assertEqual(old_hash, "a" * 64)
            self.assertEqual(new_hash, "c" * 64)
            self.assertEqual(result["report"]["protocol"]["extraction"]["library_snapshot_sha256"], "f" * 64)

    def test_exact_lemma_call_and_dependency_closure_are_rendered(self):
        data = records()
        dependency, dependent = data[-2], data[-1]
        dependent["dependencies"] = [dependency["id"]]
        dependent["payload"]["proof"] = {"lemma": {"id": dependency["id"], "atoms": [0, 1, 2],
                                                    "arguments": [{"hyp": i} for i in range(3)]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "translation"
            self.fit(path, data)
            manifest = load_correspondence(path / "views")
            text = manifest["splits"]["test"]["pairs"][1]["language_text"]
            self.assertIn(f"Theorem {dependency['id']}", text)
            self.assertIn(f"use earlier theorem {dependency['id']}", text)
            self.assertIn("atom 2 denotes T2", text)
            bad = copy.deepcopy(data)
            bad[-1]["payload"]["proof"] = {"unsupported": {}}
            with self.assertRaisesRegex(ValueError, "unrecognized constructive proof"):
                self.fit(Path(tmp) / "bad", bad)

    def test_rejects_evaluation_rows_bad_shapes_changed_weights_and_context_overflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = records()
            data[0]["learning_example"] = {"split": "test"}
            with self.assertRaisesRegex(ValueError, "evaluation split"):
                self.fit(path / "evaluation", data)
            with self.assertRaisesRegex(ValueError, "incorrectly shaped"):
                self.fit(path / "shape", extractor=lambda m, p: np.ones((len(p), 2)))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.fit(path / "nonfinite", extractor=lambda m, p: np.full((len(p), 3), np.nan))
            def mutate(model, prompts):
                model.state = "b" * 64
                return fixture_extract(model, prompts)
            with self.assertRaisesRegex(ValueError, "changed model parameters"):
                self.fit(path / "mutating", extractor=mutate)
            def overlong(model, prompts):
                raise ValueError("representation prompt exceeds recipient context")
            with self.assertRaisesRegex(ValueError, "exceeds recipient context"):
                self.fit(path / "context", extractor=overlong)


if __name__ == "__main__":
    unittest.main()
