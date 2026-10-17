"""Collector-to-affine integration checks using fake numeric activations."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "fit_activations.py"
SPEC = importlib.util.spec_from_file_location("fit_activations", MODULE_PATH)
fit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit)


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fake_collections(root):
    rng = np.random.default_rng(57)
    w = rng.normal(size=(3, 5))
    offset = np.array([3., -2., 7.])
    source, target = root / "source", root / "target"
    for directory in (source, target):
        directory.mkdir()
    manifests = []
    for directory, is_target in ((source, False), (target, True)):
        manifests.append({
            "status": "completed", "model": "fake/target" if is_target else "fake/source",
            "revision": "a" * 40, "layer_index": 3,
            "position": "last_nonpadding_token", "pooling": "none",
            "input_format": "literal_text_no_chat_template_no_special_tokens",
            "tokenizer_sha256": hash_text("same-tokenizer"), "inputs": {}, "outputs": {},
        })
    for split, count in (("train", 80), ("val", 25), ("test", 30)):
        x = rng.normal(size=(count, 5))
        y = x @ w.T + offset
        shared = {
            "ids": np.asarray([f"{split}-{i}" for i in range(count)]),
            "families": np.asarray([f"{split}-family-{i % 2}" for i in range(count)]),
            "token_sha256": np.asarray([hash_text(f"tokens:{split}:{i}") for i in range(count)]),
            "lengths": np.full(count, 10, dtype=np.int64),
        }
        for directory, manifest, vectors in zip((source, target), manifests, (x, y)):
            path = directory / f"{split}.npz"
            np.savez_compressed(path, vectors=vectors, **shared)
            manifest["inputs"][split] = {"path": f"{split}.jsonl", "sha256": hash_text(split), "count": count}
            manifest["outputs"][split] = {"path": path.name, "shape": list(vectors.shape), "sha256": fit.digest(path)}
    for directory, manifest in zip((source, target), manifests):
        (directory / "manifest.json").write_text(json.dumps(manifest))
    return source, target


def change_archive(directory, split, update):
    path = directory / f"{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    update(values)
    np.savez_compressed(path, **values)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][split]["sha256"] = fit.digest(path)
    manifest_path.write_text(json.dumps(manifest))


class FitActivationTests(unittest.TestCase):
    def test_end_to_end_selects_on_validation_and_saves_raw_affine_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = fake_collections(root)
            output = root / "fit"
            report = fit.fit_collections(source, target, output, [0.0, 0.01, 1.0])
            self.assertEqual(report["selected_alpha"], 0.0)
            self.assertEqual(len(report["validation_scores"]), 3)
            metrics = report["test_metrics"]
            self.assertLess(metrics["paired_affine"]["normalized_rmse"], 1e-12)
            self.assertGreater(metrics["shuffled_training_pairs"]["normalized_rmse"], .5)
            self.assertGreater(metrics["training_mean"]["normalized_rmse"], .5)
            self.assertEqual(metrics["paired_affine"]["cosine_retrieval_top1_accuracy"], 1.0)
            with np.load(output / "translator.npz") as saved, np.load(source / "test.npz") as sx, np.load(target / "test.npz") as ty:
                pred = sx["vectors"] @ saved["coef"].T + saved["intercept"]
                np.testing.assert_allclose(pred, ty["vectors"], atol=1e-12)
                with np.load(source / "train.npz") as train:
                    np.testing.assert_allclose(saved["source_mean"], train["vectors"].mean(axis=0))
            self.assertEqual(json.loads((output / "results.json").read_text())["status"], "completed")
            with self.assertRaisesRegex(ValueError, "already exists"):
                fit.fit_collections(source, target, output)

    def test_rejects_partial_or_malformed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = fake_collections(Path(tmp))
            path = source / "manifest.json"
            original = json.loads(path.read_text())
            for update in ({"status": "started"}, {"layer_index": None}, {"revision": "main"}):
                path.write_text(json.dumps({**original, **update}))
                with self.assertRaises(ValueError):
                    fit.paired_collections(source, target)

    def test_rejects_token_mismatch_even_when_ids_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = fake_collections(Path(tmp))
            def update(values):
                values["token_sha256"][0] = hash_text("different tokens")
            change_archive(target, "val", update)
            with self.assertRaisesRegex(ValueError, "token_sha256"):
                fit.paired_collections(source, target)

    def test_rejects_row_reordering_input_mismatch_and_archive_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = fake_collections(Path(tmp))
            def reorder(values):
                for key in values:
                    values[key] = values[key][::-1]
            change_archive(target, "test", reorder)
            with self.assertRaisesRegex(ValueError, "ids differ"):
                fit.paired_collections(source, target)
        with tempfile.TemporaryDirectory() as tmp:
            source, target = fake_collections(Path(tmp))
            path = target / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["inputs"]["train"]["sha256"] = hash_text("other input")
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "input hashes"):
                fit.paired_collections(source, target)
            (source / "test.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                fit.load_collection(source)

    def test_rejects_family_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = fake_collections(Path(tmp))
            def update(values):
                values["families"] = values["families"].astype("U40")
                values["families"][0] = "train-family-0"
            change_archive(source, "test", update)
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                fit.load_collection(source)


if __name__ == "__main__":
    unittest.main()
