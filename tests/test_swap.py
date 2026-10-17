"""Structural guarantees, matched controls and safe neural translator bundles."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lemmaferry.rosetta import PairedSplit, _array_sha, _json_bytes, _sha

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from lemmaferry.swap import (DomainStatistics, ReversibleSwap, Slot, SwapAutoencoder,
                                SwapConfig, SwapPair, _state_hash, shuffled_permutation)
    from lemmaferry.translators import fit_translator, load_translator


def pairs():
    rng = np.random.default_rng(42)
    result = {}
    for split, count in (("train", 12), ("val", 5), ("test", 7)):
        x = rng.normal(size=(count, 3))
        y = np.column_stack((x, x[:, 0] * x[:, 1])) + [3., -1., 7., 4.]
        result[split] = PairedSplit(x, y, [f"{split}-{i}" for i in range(count)],
                                    [f"{split}-structure"] * count, split)
    return result


def rewrite_bundle(directory, mutate_manifest=None, mutate_arrays=None):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if mutate_arrays:
        archive = directory / "maps.npz"
        with np.load(archive, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        mutate_arrays(arrays, manifest)
        np.savez_compressed(archive, **arrays)
        manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        manifest["arrays_sha256"] = {name: _array_sha(value) for name, value in arrays.items()}
        manifest["array_specs"] = {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()}
    if mutate_manifest:
        mutate_manifest(manifest)
    manifest.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = _sha(_json_bytes(manifest))
    path.write_text(json.dumps(manifest))


@unittest.skipUnless(HAS_TORCH, "optional torch runtime not installed")
class SwapArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = SwapAutoencoder(3, 4, latent_dim=2, hidden_dim=8)
        self.a, self.b = torch.randn(6, 3), torch.randn(6, 4)

    def test_crossed_port_values_and_input_gradients(self):
        a, b = self.a.requires_grad_(), self.b.requires_grad_()
        out_a, out_b = self.model(Slot.full(a), Slot.full(b))
        self.assertEqual(out_a.values.shape, (6, 3))
        self.assertEqual(out_b.values.shape, (6, 4))
        ga, gb = torch.autograd.grad(out_b.values.sum(), (a, b), allow_unused=True)
        self.assertIsNotNone(ga)
        self.assertIsNone(gb)
        other_a, other_b = self.model(Slot.full(a.detach()), Slot.full(b.detach() + 7))
        torch.testing.assert_close(other_b.values, out_b.values, rtol=0, atol=0)
        self.assertFalse(torch.equal(other_a.values, out_a.values))

    def test_empty_flags_and_numeric_zero_have_distinct_meanings(self):
        a = Slot.full(torch.zeros_like(self.a))
        b = Slot.empty(6, 4, like=self.b)
        left, right = self.model(a, b)
        self.assertFalse(left.present.any())
        self.assertTrue(torch.equal(left.values, torch.zeros_like(self.a)))
        self.assertTrue(right.present.all())
        back_a, back_b = self.model(left, right)
        self.assertTrue(back_a.present.all())
        self.assertFalse(back_b.present.any() or back_b.values.any())
        both = self.model(Slot.empty(6, 3, like=self.a), b)
        self.assertFalse(any(slot.present.any() or slot.values.any() for slot in both))

    def test_mixed_presence_and_nonpresent_payload_are_exactly_masked(self):
        mask = torch.tensor([True, False, True, False, True, False])
        a, b = self.model(Slot(self.a, mask), Slot(self.b, ~mask))
        self.assertTrue(torch.equal(a.present, ~mask))
        self.assertTrue(torch.equal(b.present, mask))
        self.assertFalse(a.values[mask].any() or b.values[~mask].any())

    def test_all_four_networks_receive_loss_gradients(self):
        losses = self.model.losses(self.a, self.b, torch.arange(5, -1, -1))
        torch.testing.assert_close(losses["total"], losses["pair"] + .25 * losses["self_reconstruction"]
                                   + .10 * losses["cycle"] + .01 * losses["latent"])
        losses["total"].backward()
        for part in (self.model.encoder_a, self.model.encoder_b, self.model.decoder_a, self.model.decoder_b):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters()))

    def test_cross_slot_index_permutation_does_not_change_mean_objective(self):
        identity = self.model.losses(self.a, self.b, torch.arange(6))
        crossed = self.model.losses(self.a, self.b, torch.arange(5, -1, -1))
        for name in identity:
            torch.testing.assert_close(identity[name], crossed[name])
        # Pairing is different from port order: a true wrong pair changes both
        # directional targets and the latent agreement, not just one direction.
        wrong = self.model.losses(self.a, self.b.flip(0), torch.arange(6))
        self.assertNotEqual(float(identity["latent"].detach()), float(wrong["latent"].detach()))

    def test_train_only_statistics_constant_columns_and_refit_rejection(self):
        a = self.a.clone()
        a[:, 1] = 5
        self.model.fit_normalization(a, self.b)
        expected = self.model.stats_a.mean.clone()
        self.assertEqual(float(self.model.stats_a.scale[1]), 1)
        self.model.a_to_b(a + 100000)
        torch.testing.assert_close(self.model.stats_a.mean, expected, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            self.model.fit_normalization(a, self.b)

    def test_invalid_slots_and_permutations_fail(self):
        for a in (Slot(self.a, torch.ones(6)), Slot.full(self.a[:, :2]),
                  Slot(self.a, torch.tensor([True]))):
            with self.assertRaises(ValueError):
                self.model(a, Slot.full(self.b))
        bad = self.a.clone()
        bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.model(Slot.full(bad), Slot.full(self.b))
        with self.assertRaises(ValueError):
            self.model.losses(self.a, self.b, torch.zeros(6, dtype=torch.long))

    def test_reversible_swaps_and_tagged_empty_cycle_float64(self):
        model = ReversibleSwap(4, hidden_dim=8, depth=4).double()
        a, b = torch.randn(5, 4, dtype=torch.float64), torch.randn(5, 4, dtype=torch.float64)
        first = model(Slot.full(a), Slot.full(b))
        second = model(*first)
        torch.testing.assert_close(second[0].values, a, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(second[1].values, b, rtol=1e-10, atol=1e-10)
        roundtrip = model(*model(Slot.full(a), Slot.empty(5, 4, like=a)))
        torch.testing.assert_close(roundtrip[0].values, a, rtol=1e-10, atol=1e-10)
        self.assertFalse(roundtrip[1].present.any() or roundtrip[1].values.any())


@unittest.skipUnless(HAS_TORCH, "optional torch runtime not installed")
class SwapTranslatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data = pairs()
        self.config = SwapConfig(steps=6, batch_size=5, validation_every=2,
                                 hidden_dim=8, latent_dim=3)
        self.pair = fit_translator(self.data["train"], self.data["val"], translator="swap",
                                   seed=7, swap_config=self.config)

    def test_controls_full_budget_shared_initialization_and_frozen_numpy_inference(self):
        protocol = self.pair.protocol
        left, right = protocol["training"]["paired"], protocol["training"]["shuffled"]
        for key in ("initial_state_sha256", "batch_order_sha256", "updates", "pair_exposures", "validation_steps"):
            self.assertEqual(left[key], right[key])
        self.assertEqual(left["updates"], 6)
        self.assertEqual(left["pair_exposures"], 30)
        self.assertEqual(left["validation_steps"], [2, 4, 6])
        permutation = protocol["shuffled_train_permutation"]
        self.assertTrue(all(i != j for i, j in enumerate(permutation)))
        self.assertEqual(sorted(permutation), list(range(12)))
        self.assertEqual(permutation, shuffled_permutation(12, 7).tolist())
        for arm in ("paired", "shuffled"):
            best = min(protocol["training"][arm]["trace"], key=lambda row: row["validation_bidirectional_normalized_mse"])
            self.assertEqual(protocol["training"][arm]["selected_step"], best["step"])
            self.assertFalse(any(p.requires_grad for p in self.pair.models[arm].parameters()))
        for control in ("paired", "shuffled", "mean"):
            self.assertEqual(self.pair.to_language(self.data["test"].x, control).shape, (7, 4))
            self.assertEqual(self.pair.to_math(self.data["test"].y, control).shape, (7, 3))
        self.assertIsInstance(self.pair.to_language(self.data["test"].x), np.ndarray)

    def test_joint_shuffled_pairing_drives_forward_reverse_and_latent_losses(self):
        # A one-step, full-corpus experiment is reproduced with the *single joint*
        # wrong pairing. Reversing a separately shuffled source would not match.
        from lemmaferry.swap import _model, _state_arrays, training_schedule
        config = SwapConfig(steps=1, batch_size=12, hidden_dim=8, latent_dim=3, validation_every=1)
        fitted = SwapPair.fit(self.data["train"], self.data["val"], config=config, seed=7)
        model = _model(3, 4, config, 7)
        x, y = torch.tensor(self.data["train"].x, dtype=torch.float32), torch.tensor(self.data["train"].y, dtype=torch.float32)
        model.fit_normalization(x, y)
        y = y[shuffled_permutation(12, 7)]
        rows, ports = next(training_schedule(12, config, 7))
        loss = model.losses(x[rows], y[rows], torch.tensor(ports))["total"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        for name, value in _state_arrays(model).items():
            np.testing.assert_array_equal(value, _state_arrays(fitted.models["shuffled"])[name])

    def test_test_diagnostics_do_not_select_or_refit_and_leaks_fail(self):
        before = json.dumps(self.pair.protocol, sort_keys=True)
        report = self.pair.evaluate(self.data["test"])
        self.assertEqual(before, json.dumps(self.pair.protocol, sort_keys=True))
        self.assertTrue(report["empty_routing"]["empty_payloads_exact_zero"])
        self.assertIn("self_reconstruction", report)
        report["protocol"]["seed"] = 999
        self.assertEqual(self.pair.protocol["seed"], 7)
        leaking = PairedSplit(self.data["test"].x, self.data["test"].y, self.data["test"].ids,
                              ["train-structure"] * 7, "test")
        with self.assertRaisesRegex(ValueError, "cross splits"):
            self.pair.evaluate(leaking)
        with self.assertRaisesRegex(ValueError, "test split"):
            self.pair.evaluate(self.data["val"])

    def test_save_load_frozen_predictions_and_training_data_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle"
            saved = self.pair.save(path)
            self.assertEqual(saved["schema"], "lemmaferry.swap.bundle.v1")
            loaded = load_translator(path)
            loaded.validate_training_data(self.data["train"], self.data["val"])
            for control in ("paired", "shuffled", "mean"):
                np.testing.assert_array_equal(loaded.to_language(self.data["test"].x, control),
                                               self.pair.to_language(self.data["test"].x, control))
            changed = PairedSplit(self.data["train"].x + 1, self.data["train"].y,
                                  self.data["train"].ids, self.data["train"].families, "train")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                loaded.validate_training_data(changed, self.data["val"])
            with self.assertRaises(FileExistsError):
                self.pair.save(path)

    def test_rejects_plain_archive_or_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, corrupt in (("archive", lambda p: (p / "maps.npz").write_bytes(b"bad")),
                                  ("manifest", lambda p: (p / "manifest.json").write_text("{}"))):
                path = Path(tmp) / name
                self.pair.save(path)
                corrupt(path)
                with self.assertRaises(ValueError):
                    load_translator(path)

    def test_resigned_bad_protocol_permutation_schedule_ids_and_selection_fail(self):
        changes = (
            lambda m: m["protocol"].update(shuffled_train_permutation=list(range(12))),
            lambda m: m["protocol"]["training"]["paired"].update(updates=2),
            lambda m: m["protocol"]["train"]["ids"].__setitem__(0, m["protocol"]["train"]["ids"][1]),
            lambda m: m["protocol"]["validation"]["families"].__setitem__(0, "train-structure"),
            lambda m: m["protocol"]["training"]["paired"].update(selected_step=1),
            lambda m: m["protocol"]["training"]["paired"].update(initial_state_sha256="0" * 64),
            lambda m: m["protocol"].update(optimizer={"name": "SGD"}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for i, change in enumerate(changes):
                path = Path(tmp) / str(i)
                self.pair.save(path)
                rewrite_bundle(path, mutate_manifest=change)
                with self.subTest(change=i), self.assertRaises(ValueError):
                    load_translator(path)

    def test_resigned_bad_shapes_and_unfitted_statistics_fail(self):
        def wrong_shape(arrays, manifest):
            arrays["paired::encoder_a.0.weight"] = arrays["paired::encoder_a.0.weight"][:, :-1]
            state = {key.split("::", 1)[1]: value for key, value in arrays.items() if key.startswith("paired::")}
            manifest["protocol"]["training"]["paired"]["selected_state_sha256"] = _state_hash(state)
        def unfitted(arrays, manifest):
            arrays["paired::stats_a.fitted"] = np.array(False)
            state = {key.split("::", 1)[1]: value for key, value in arrays.items() if key.startswith("paired::")}
            manifest["protocol"]["training"]["paired"]["selected_state_sha256"] = _state_hash(state)
        with tempfile.TemporaryDirectory() as tmp:
            for i, change in enumerate((wrong_shape, unfitted)):
                path = Path(tmp) / str(i)
                self.pair.save(path)
                rewrite_bundle(path, mutate_arrays=change)
                with self.assertRaises(ValueError):
                    load_translator(path)


if __name__ == "__main__":
    unittest.main()
