"""Exercise neural fitting through both audited collection protocols."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch runtime not installed")
class SwapFittingTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.config = dict(steps=2, validation_every=1, batch_size=4, hidden_dim=8, latent_dim=3)

    def test_neural_fitting_clis_complete_and_report_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = load("_cli_distinct_fixture", "tests/test_rosetta.py")
            source, target, views = fixture.fake_cross_domain(Path(tmp))
            command = [sys.executable, str(ROOT / "experiments/fit_rosetta.py"),
                       "--source", str(source), "--target", str(target),
                       "--correspondence", str(views), "--output", str(Path(tmp) / "cli-fit"),
                       "--translator", "swap", "--seed", "7", "--swap-steps", "2",
                       "--swap-hidden-dim", "8", "--swap-latent-dim", "3",
                       "--swap-validation-every", "1", "--swap-batch-size", "4"]
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(set(result["selection"]), {"paired", "shuffled"})
            self.assertTrue(all(1 <= row["selected_step"] <= 2 for row in result["selection"].values()))

    def test_same_input_neural_fit_reloads_and_keeps_token_audit(self):
        fixture = load("_same_input_fixture", "tests/test_fit_activations.py")
        from lemmaferry.translators import load_translator
        with tempfile.TemporaryDirectory() as tmp:
            source, target = fixture.fake_collections(Path(tmp))
            output = Path(tmp) / "fit"
            report = fixture.fit.fit_collections(source, target, output, translator="swap",
                                                 seed=7, swap_config=self.config)
            self.assertEqual(report["status"], "completed")
            self.assertIn("identical-input", report["protocol"]["extraction"]["pairing_policy"])
            restored = load_translator(output)
            with np.load(source / "test.npz", allow_pickle=False) as data:
                for control in ("paired", "shuffled", "mean"):
                    self.assertEqual(restored.to_language(data["vectors"], control=control).shape, (30, 3))
            def alter(rows):
                rows["token_sha256"][0] = "0" * 64
            fixture.change_archive(target, "val", alter)
            with self.assertRaisesRegex(ValueError, "token_sha256"):
                fixture.fit.fit_collections(source, target, Path(tmp) / "invalid", translator="swap",
                                             swap_config=self.config)
            self.assertFalse((Path(tmp) / "invalid").exists())

    def test_distinct_view_neural_fit_allows_audited_different_tokenizers(self):
        fixture = load("_distinct_view_fixture", "tests/test_rosetta.py")
        from lemmaferry.translators import load_translator
        with tempfile.TemporaryDirectory() as tmp:
            source, target, views = fixture.fake_cross_domain(Path(tmp))
            output = Path(tmp) / "fit"
            report = fixture.fit_cli.fit_collections(source, target, views, output,
                                                     translator="swap", seed=7, swap_config=self.config)
            extraction = report["protocol"]["extraction"]
            self.assertNotEqual(extraction["math"]["tokenizer_sha256"],
                                extraction["language"]["tokenizer_sha256"])
            self.assertEqual(extraction["library_snapshots_sha256"], ["f" * 64])
            restored = load_translator(output)
            self.assertEqual(restored.protocol, report["protocol"])
            path = target / "manifest.json"
            metadata = json.loads(path.read_text())
            metadata["inputs"]["val"]["sha256"] = "0" * 64
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "collector input differs"):
                fixture.fit_cli.fit_collections(source, target, views, Path(tmp) / "invalid",
                                                 translator="swap", swap_config=self.config)
            self.assertFalse((Path(tmp) / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
