"""Independent neural-run checks, including edits with refreshed envelope hashes.

The correspondence rows here are explicit renderer fixtures, not Lean claims.
The actual two-cycle Lean integration is exercised by test_loop.py.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None
specification = importlib.util.spec_from_file_location("verify_constructive_run", ROOT / "experiments/verify_constructive_run.py")
verifier = importlib.util.module_from_spec(specification)
specification.loader.exec_module(verifier)


def _fixture_records():
    rows = []
    for family, premises, proof in (
        ("and_elimination", [{"and": [0, 1]}], {"and_left": {"hyp": 0}}),
        ("and_introduction", [0, 1], {"and_intro": [{"hyp": 0}, {"hyp": 1}]}),
        ("implication_chain", [0, {"imp": [0, 1]}], {"apply": [{"hyp": 1}, {"hyp": 0}]}),
    ):
        for index in range(4):
            identifier = f"{family}_{index}"
            rows.append({"id": identifier, "family": family,
                         "formal_text": f"theorem {identifier} := renderer_fixture_{index}",
                         "natural_language": f"Constructive {family} statement {index}.",
                         "payload": {"atom_count": 2, "premises": premises, "proof": proof},
                         "proposal": {"label_prefix": "T"}, "dependencies": [],
                         "provenance": {"scope": "renderer fixture; not a Lean-verified record"}})
    return rows


@unittest.skipUnless(HAS_TORCH, "optional Torch runtime required")
class SwapVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from lemmaferry.learning import TinyRecipient, save_recipient
        from lemmaferry.representations import fit_library_rosetta
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "fixture"
        cls.rows = _fixture_records()
        cls.snapshot = "f" * 64
        cls.config = {"translator": "swap", "seed": 7, "device": "cpu",
                      "swap_latent_dim": 3, "swap_hidden_dim": 8, "swap_steps": 2,
                      "swap_batch_size": 4, "swap_learning_rate": 0.002,
                      "swap_validation_every": 1}
        source = TinyRecipient(seed=8, hidden=8, heads=2, max_length=2048)
        language = TinyRecipient(seed=7, hidden=8, heads=2, max_length=2048)
        cls.source_receipt = save_recipient(source, cls.root / "source-initial.pt")
        save_recipient(language, cls.root / "initial/verified.pt")
        config = verifier._registered_swap_config(cls.config)
        fit_library_rosetta(cls.rows, cls.snapshot, source, language,
                           cls.root / "iteration-000/rosetta", seed=7,
                           translator="swap", swap_config=config)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        from lemmaferry.correspondence import load_correspondence
        from lemmaferry.translators import load_translator
        self.directory = Path(self.temporary.name) / self._testMethodName
        shutil.copytree(self.root, self.directory)
        self.rosetta = self.directory / "iteration-000/rosetta"
        self.pair = load_translator(self.rosetta / "maps")
        self.manifest = load_correspondence(self.rosetta / "views")
        with np.load(self.rosetta / "activations.npz", allow_pickle=False) as archive:
            self.arrays = {name: archive[name].copy() for name in archive.files}
        self.iteration = {"iteration": 0, "library_after": {
            "record_ids": [row["id"] for row in self.rows], "snapshot_hash": self.snapshot}}
        source = ROOT / "src/lemmaferry"
        self.protocol = {"config": copy.deepcopy(self.config), "source_sha256": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*.py"))}}
        self.by_id = {row["id"]: row for row in self.rows}
        self.result = {"source_checkpoint": copy.deepcopy(self.source_receipt)}

    def check(self):
        verifier._verify_swap_calibration(self.pair, self.manifest, self.arrays,
                                         self.iteration, self.protocol, self.by_id)

    def replay(self):
        verifier._replay_swap_extraction(self.directory, self.pair, self.manifest,
                                         self.arrays, self.iteration, self.protocol,
                                         self.result, "cpu")

    def rewrite_bundle_protocol(self, edit):
        """Simulate a self-consistent edited manifest, not just a stale checksum."""
        from lemmaferry.translators import load_translator
        path = self.rosetta / "maps/manifest.json"
        manifest = json.loads(path.read_text())
        edit(manifest["protocol"])
        manifest.pop("manifest_content_sha256")
        manifest["manifest_content_sha256"] = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        path.write_text(json.dumps(manifest))
        self.pair = load_translator(self.rosetta / "maps")

    def test_safe_reload_all_controls_and_preupdate_extraction(self):
        self.check()
        self.replay()
        from lemmaferry.translators import load_translator
        second = load_translator(self.rosetta / "maps")
        for control in ("paired", "shuffled", "mean"):
            np.testing.assert_array_equal(self.pair.to_language(self.arrays["math"], control=control),
                                          second.to_language(self.arrays["math"], control=control))
            np.testing.assert_array_equal(self.pair.to_math(self.arrays["language"], control=control),
                                          second.to_math(self.arrays["language"], control=control))
        self.assertEqual(verifier._target_tolerances(object()), (1e-8, 1e-8))
        self.assertEqual(verifier._target_tolerances(self.pair), (64 * np.finfo(np.float32).eps,) * 2)

    def test_rejects_modified_activations_even_when_archive_hash_is_refreshed(self):
        self.arrays["math"][0, 0] += 0.25
        np.savez_compressed(self.rosetta / "activations.npz", **self.arrays)
        report = json.loads((self.rosetta / "results.json").read_text())
        report["artifacts"]["activations_sha256"] = hashlib.sha256((self.rosetta / "activations.npz").read_bytes()).hexdigest()
        (self.rosetta / "results.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.check()
        # Actual extraction detects fabrication independently of fit fingerprints.
        with self.assertRaisesRegex(ValueError, "activations do not reproduce"):
            self.replay()

    def test_rejects_reordered_ids_and_wrong_structural_families(self):
        self.arrays["ids"][[0, 1]] = self.arrays["ids"][[1, 0]]
        with self.assertRaisesRegex(ValueError, "row order"):
            self.check()
        self.arrays["ids"][[0, 1]] = self.arrays["ids"][[1, 0]]
        self.arrays["families"][0] = "wrong_family"
        with self.assertRaisesRegex(ValueError, "families"):
            self.check()

    def test_rejects_altered_language_view_with_recomputed_correspondence_hashes(self):
        from lemmaferry.correspondence import build_correspondence
        rows = self.manifest["splits"]["train"]["pairs"]
        row = rows[0]
        rows[0] = build_correspondence(row["id"], row["family"], row["formal_text"],
                                       row["language_text"] + " A forged conclusion.", row["provenance"])
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.check()

    def test_rejects_configuration_and_seed_edits_with_fresh_manifest_checksum(self):
        # Weight decay does not change tensor dimensions, so the numeric loader
        # can accept this self-consistent bundle; the frozen run must reject it.
        self.rewrite_bundle_protocol(lambda p: p["config"].update(weight_decay=0.01))
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.check()
        self.pair.protocol["config"] = verifier._registered_swap_config(self.protocol["config"])
        self.pair.protocol["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed"):
            self.check()

    def test_rejects_another_valid_derangement_even_with_fresh_checksum(self):
        old = self.pair.protocol["shuffled_train_permutation"]
        candidate = next(np.roll(np.arange(len(old)), shift).tolist()
                         for shift in range(1, len(old)) if np.roll(np.arange(len(old)), shift).tolist() != old)
        with self.assertRaisesRegex(ValueError, "permutation|pairing|derangement"):
            self.rewrite_bundle_protocol(lambda p: p.update(shuffled_train_permutation=candidate))
            self.check()

    def test_rejects_model_and_recursive_code_provenance_edits(self):
        metadata = self.pair.protocol["extraction"]["models"]["math"]
        metadata["config"]["seed"] += 1
        metadata["config_sha256"] = hashlib.sha256(json.dumps(metadata["config"], sort_keys=True,
                                                             separators=(",", ":")).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.replay()
        self.protocol["source_sha256"].pop("swap.py")
        with self.assertRaisesRegex(ValueError, "recursive source"):
            self.replay()

    def test_retained_weights_must_reproduce_the_selected_validation_score(self):
        training = self.pair.protocol["training"]["paired"]
        selected = next(row for row in training["trace"] if row["step"] == training["selected_step"])
        selected["validation_bidirectional_normalized_mse"] += max(1.0, abs(selected["validation_bidirectional_normalized_mse"]) * 0.25)
        with self.assertRaisesRegex(ValueError, "validation score"):
            self.check()

    def test_bridge_stays_cpu_when_recipient_registration_uses_an_accelerator(self):
        self.protocol["config"]["device"] = "mps"
        self.check()


if __name__ == "__main__":
    unittest.main()
