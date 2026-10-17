"""Known correspondence, objective, held-out hygiene and artifact integration."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lemmaferry.alignment import AffineRidge
from lemmaferry.correspondence import build_correspondence, load_correspondence, write_correspondence
from lemmaferry.rosetta import PairedSplit, RosettaPair, fit_affine

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "fit_rosetta.py"
_SPEC = importlib.util.spec_from_file_location("fit_rosetta", _PATH)
fit_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fit_cli)


def synthetic_pairs():
    rng = np.random.default_rng(24)
    transform = rng.normal(size=(3, 5))
    offset = rng.normal(size=5)
    result = {}
    for split, count in (("train", 60), ("val", 20), ("test", 30)):
        x = rng.normal(size=(count, 3))
        result[split] = PairedSplit(x, x @ transform + offset,
                                    [f"{split}-{i}" for i in range(count)],
                                    [f"{split}-structure-{i % 3}" for i in range(count)], split)
    return result


def fake_cross_domain(root):
    pairs = synthetic_pairs()
    views = {}
    for split, data in pairs.items():
        views[split] = [build_correspondence(identifier, family,
                                            f"theorem formal_{identifier} : True := True.intro",
                                            f"The statement numbered {identifier} holds, with a proof.",
                                            {"library_snapshot_sha256": "f" * 64})
                        for identifier, family in zip(data.ids, data.families)]
    directory = root / "views"
    manifest = write_correspondence(views, directory)
    for domain in ("math", "language"):
        path = root / domain
        path.mkdir()
        metadata = {
            "status": "completed", "model": f"fake/{domain}", "revision": "a" * 40,
            "layer_index": 2, "position": "last_nonpadding_token", "pooling": "none",
            "input_format": "literal_text_no_chat_template_no_special_tokens",
            "tokenizer_sha256": hashlib.sha256(domain.encode()).hexdigest(),
            "inputs": {}, "outputs": {},
        }
        for split, data in pairs.items():
            vectors = data.x if domain == "math" else data.y
            archive = path / f"{split}.npz"
            np.savez_compressed(archive, vectors=vectors, ids=np.asarray(data.ids),
                                families=np.asarray(data.families),
                                token_sha256=np.asarray([hashlib.sha256(f"{domain}-{i}".encode()).hexdigest() for i in data.ids]),
                                lengths=np.full(len(data.ids), 11 if domain == "math" else 19))
            metadata["inputs"][split] = {**manifest["splits"][split][domain], "count": len(data.ids)}
            metadata["outputs"][split] = {"path": archive.name, "shape": list(vectors.shape),
                                            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        (path / "manifest.json").write_text(json.dumps(metadata))
    return root / "math", root / "language", directory


class RosettaTests(unittest.TestCase):
    def test_raw_objective_matches_normal_equations_and_intercept_is_unpenalized(self):
        rng = np.random.default_rng(5)
        for count, dimension in ((40, 3), (4, 20)):
            x = rng.normal(size=(count, dimension)) + 50
            y = rng.normal(size=(count, 2)) + [1000, -800]
            strength = 2.5
            model = fit_affine(x, y, strength)
            error = model.predict(x) - y
            np.testing.assert_allclose(error.sum(axis=0), 0, atol=1e-10)
            np.testing.assert_allclose(x.T @ error + strength * model.coef.T, 0, atol=1e-9)
            # This would fail if lambda were silently multiplied by sample count.
            xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
            expected = np.linalg.solve(xc.T @ xc + strength * np.eye(dimension), xc.T @ yc)
            np.testing.assert_allclose(model.coef.T, expected, atol=1e-11)

    def test_standardized_mode_preserves_original_baseline(self):
        rng = np.random.default_rng(8)
        x, y = rng.normal(size=(25, 4)) * [1, 10, 100, 1000], rng.normal(size=(25, 2))
        legacy = AffineRidge(alpha=.3).fit(x, y)
        wrapped = fit_affine(x, y, .3, "standardized_mean")
        np.testing.assert_allclose(wrapped.predict(x), legacy.predict(x), atol=1e-13)
        self.assertGreater(np.max(np.abs(fit_affine(x, y, .3).coef - wrapped.coef)), .001)

    def test_bidirectional_different_dimensions_and_F_squared(self):
        data = synthetic_pairs()
        pair = RosettaPair.fit(data["train"], data["val"], [0, .1, 10])
        report = pair.evaluate(data["test"])
        for direction in ("math_to_language", "language_to_math"):
            metrics = report["test_metrics"][direction]
            self.assertLess(metrics["paired"]["mse"], 1e-25)
            self.assertGreater(metrics["shuffled_training_pairs"]["mse"], .1)
            self.assertGreater(metrics["training_mean"]["mse"], .1)
        for name in ("T", "S"):
            self.assertEqual(pair.protocol["selection"][name]["selected_lambda"], 0)
        xx, yy = pair.F(*pair.F(data["test"].x, data["test"].y))
        np.testing.assert_allclose(xx, data["test"].x, atol=1e-12)
        np.testing.assert_allclose(yy, data["test"].y, atol=1e-12)

    def test_reference_slot_is_fixed_and_does_not_encode_other_slot(self):
        data = synthetic_pairs()
        reference = np.array([3., 2., 1.])
        pair = RosettaPair.fit(data["train"], data["val"], math_reference=reference)
        translated_y, reference_output = pair.F(None, data["test"].y)
        np.testing.assert_allclose(reference_output, np.repeat(pair.to_language(reference[None]), 30, axis=0))
        _, other_output = pair.F(None, data["test"].y + 999)
        np.testing.assert_array_equal(reference_output, other_output)
        self.assertEqual(translated_y.shape, (30, 3))
        self.assertEqual(pair.F(None, None)[0].shape, (1, 3))

    def test_identity_order_and_split_leaks_fail(self):
        data = synthetic_pairs()
        with self.assertRaisesRegex(ValueError, "reordered"):
            PairedSplit(data["test"].x, data["test"].y, data["test"].ids,
                        data["test"].families, "test", target_ids=reversed(data["test"].ids))
        pair = RosettaPair.fit(data["train"], data["val"])
        for ids, families in ((data["train"].ids[:30], data["test"].families),
                              (data["test"].ids, [data["val"].families[0]] * 30)):
            leaking = PairedSplit(data["test"].x, data["test"].y, ids, families, "test")
            with self.assertRaisesRegex(ValueError, "cross splits"):
                pair.evaluate(leaking)
        with self.assertRaisesRegex(ValueError, "test split"):
            pair.evaluate(data["val"])

    def test_test_evaluation_cannot_update_training_or_selected_maps(self):
        data = synthetic_pairs()
        pair = RosettaPair.fit(data["train"], data["val"])
        before = pair.maps["T"].coef.copy()
        protocol = json.dumps(pair.protocol, sort_keys=True)
        extreme = PairedSplit(data["test"].x * 1e4, data["test"].y * -1e4,
                               data["test"].ids, data["test"].families, "test")
        pair.evaluate(extreme)
        np.testing.assert_array_equal(pair.maps["T"].coef, before)
        self.assertEqual(json.dumps(pair.protocol, sort_keys=True), protocol)
        np.testing.assert_allclose(pair.math_scaler.mean, data["train"].x.mean(axis=0))

    def test_save_reload_and_reject_archive_or_protocol_tampering(self):
        data = synthetic_pairs()
        pair = RosettaPair.fit(data["train"], data["val"])
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "maps"
            pair.save(bundle)
            restored = RosettaPair.load(bundle)
            np.testing.assert_array_equal(restored.to_language(data["test"].x), pair.to_language(data["test"].x))
            self.assertEqual(restored.evaluate(data["test"]), pair.evaluate(data["test"]))
            with self.assertRaises(FileExistsError):
                pair.save(bundle)
            manifest_path = bundle / "manifest.json"
            original = manifest_path.read_text()
            manifest = json.loads(original)
            manifest["protocol"]["seed"] = 999
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest checksum"):
                RosettaPair.load(bundle)
            manifest_path.write_text(original)
            with (bundle / "maps.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "archive checksum"):
                RosettaPair.load(bundle)

    def test_distinct_view_collection_fit_preserves_tokenizer_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, correspondence = fake_cross_domain(root)
            report = fit_cli.fit_collections(source, target, correspondence, root / "fit", [0, .1])
            self.assertLess(report["test_metrics"]["math_to_language"]["paired"]["mse"], 1e-25)
            self.assertEqual(report["protocol"]["extraction"]["library_snapshots_sha256"], ["f" * 64])
            extraction = report["protocol"]["extraction"]
            self.assertNotEqual(extraction["math"]["tokenizer_sha256"], extraction["language"]["tokenizer_sha256"])
            RosettaPair.load(root / "fit")
            with self.assertRaisesRegex(ValueError, "already exists"):
                fit_cli.fit_collections(source, target, correspondence, root / "fit")

    def test_cross_domain_fit_rejects_altered_inputs_and_reordered_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, correspondence = fake_cross_domain(root)
            view = correspondence / "language" / "val.jsonl"
            original = view.read_text()
            view.write_text(original + "\n")
            with self.assertRaisesRegex(ValueError, "input checksum"):
                fit_cli.paired_domains(source, target, correspondence)
            view.write_text(original)
            path = target / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["inputs"]["val"]["sha256"] = "0" * 64
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "collector input differs"):
                fit_cli.paired_domains(source, target, correspondence)

    def test_correspondence_requires_library_snapshot_and_disjoint_families(self):
        with self.assertRaisesRegex(ValueError, "library_snapshot"):
            build_correspondence("a", "f", "formal", "language", {})
        with self.assertRaisesRegex(ValueError, "distinct"):
            build_correspondence("a", "f", "same", " same ", {"library_snapshot_sha256": "f" * 64})
        with tempfile.TemporaryDirectory() as tmp:
            source, target, views = fake_cross_domain(Path(tmp))
            manifest = load_correspondence(views)
            self.assertEqual(manifest["splits"]["train"]["count"], 60)
            rows = {split: info["pairs"] for split, info in manifest["splits"].items()}
            old = rows["test"][0]
            rows["test"][0] = build_correspondence(old["id"], rows["train"][0]["family"], old["formal_text"], old["language_text"], old["provenance"])
            with self.assertRaisesRegex(ValueError, "families cross splits"):
                write_correspondence(rows, Path(tmp) / "leaking")


if __name__ == "__main__":
    unittest.main()
