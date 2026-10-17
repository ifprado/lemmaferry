"""Tests of held-out isolation, persistent updates, and real adapter gradients."""
from __future__ import annotations

from dataclasses import replace
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lemmaferry.learning import (FrozenEvaluation, TrainingBudget, TinyRecipient, HFRecipient,
    build_experience, evaluate, freeze_evaluation, load_evaluation, load_recipient, represent,
    save_recipient, score_candidates, train_update, useful_control)


def choice(identifier="train:0", answer="A"):
    return {"id": identifier, "prompt": "Given a route from the starting point, choose its shorter valid derivation.\nA: first\nB: second\nAnswer:",
            "answer": answer, "family": "test_fixture", "split": "train"}


def fixed(path):
    return freeze_evaluation(path, [{"id": "external:0", "prompt": "What is two plus three?\nA: five\nB: six\nAnswer:",
                                    "answer": "A", "family": "independent_arithmetic"}])


class LearningDataTests(unittest.TestCase):
    def test_budget_rejects_nonfinite_or_fractional_configuration(self):
        for kwargs in ({"steps": 1.5}, {"gradient_clip": float("nan")},
                       {"auxiliary_weight": float("inf")}, {"learning_rate": -0.1}):
            with self.assertRaises(ValueError):
                TrainingBudget(**kwargs)

    def test_frozen_evaluation_rejects_replacement_or_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation.json"
            registered = fixed(path)
            self.assertEqual(registered.sha256, fixed(path).sha256)
            with self.assertRaises(ValueError):
                freeze_evaluation(path, [choice("different")])
            data = json.loads(path.read_text())
            data["examples"][0]["answer"] = "B"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                registered.verify()
            with self.assertRaises(ValueError):
                load_evaluation(path)

    def test_experience_binds_snapshot_and_factual_control_preserves_budget(self):
        record = {"id": "lemma:0", "payload": {"proof": "not the learning record"},
                  "learning_example": choice()}
        rows = build_experience([record], library_hash="a" * 64)
        self.assertEqual(rows[0]["id"], "lemma:0")
        self.assertEqual(rows[0]["provenance"]["accepted_library_id"], "lemma:0")
        self.assertEqual(rows[0]["provenance"]["accepted_library_hash"], "a" * 64)
        controls = useful_control(rows)
        self.assertEqual(controls[0]["answer"], rows[0]["answer"])
        self.assertEqual(len(controls[0]["prompt"].encode()), len(rows[0]["prompt"].encode()))
        self.assertIn("recorded color is red", controls[0]["prompt"])
        with self.assertRaises(ValueError):
            build_experience([record], library_hash="unverified")


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch runtime required")
class PersistentLearningTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.evaluation = fixed(Path(self.tmp.name) / "evaluation.json")
        self.rows = [choice(), choice("train:1", "B")]
        self.budget = TrainingBudget(steps=2, batch_size=2, max_length=128)

    def recipient(self):
        return TinyRecipient(seed=11, hidden=16, heads=2, max_length=128)

    def test_weights_optimizer_and_predictions_survive_reload_and_continue(self):
        model = self.recipient()
        initial = evaluate(model, self.evaluation)
        first = train_update(model, self.rows, budget=self.budget, fixed_evaluation=self.evaluation)
        self.assertNotEqual(first["parameter_sha256_before"], first["parameter_sha256_after"])
        path = Path(self.tmp.name) / "recipient.pt"
        saved = save_recipient(model, path)
        loaded = load_recipient(path)
        self.assertEqual(evaluate(loaded, self.evaluation), evaluate(model, self.evaluation))
        self.assertEqual(saved["cumulative_updates"], 2)
        a = train_update(model, self.rows, budget=self.budget, fixed_evaluation=self.evaluation, seed=18)
        b = train_update(loaded, self.rows, budget=self.budget, fixed_evaluation=self.evaluation, seed=18)
        self.assertEqual(a, b)
        self.assertEqual(a["cumulative_updates"], 4)
        self.assertNotEqual(initial["recipient_parameter_sha256"], a["parameter_sha256_after"])
        self.assertTrue(all(int(v["step"]) == 4 for v in loaded.optimizer.state.values()))

    def test_useful_control_matches_shapes_and_evaluation_never_enters_feedback(self):
        treatment, control = self.recipient(), self.recipient()
        a = train_update(treatment, self.rows, budget=self.budget, fixed_evaluation=self.evaluation)
        b = train_update(control, useful_control(self.rows), budget=self.budget, fixed_evaluation=self.evaluation)
        for key in ("processed_padded_tokens", "nonpadding_tokens", "supervised_answer_tokens", "examples_seen"):
            self.assertEqual(a[key], b[key], key)
        ranking = score_candidates(treatment, self.rows, fixed_evaluation=self.evaluation)
        self.assertEqual({r["id"] for r in ranking}, {r["id"] for r in self.rows})
        with self.assertRaises(ValueError):
            score_candidates(treatment, self.evaluation.examples, fixed_evaluation=self.evaluation)
        with self.assertRaises(ValueError):
            train_update(treatment, self.evaluation.examples, budget=self.budget, fixed_evaluation=self.evaluation)

    def test_translated_targets_receive_gradients_at_declared_boundary(self):
        model = self.recipient()
        vectors = represent(model, [r["prompt"] for r in self.rows])
        self.assertEqual(vectors.shape, (2, 16))
        rows = [{**r, "translated_target": [0.0] * 16} for r in self.rows]
        report = train_update(model, rows, budget=replace(self.budget, auxiliary_weight=0.1),
                              fixed_evaluation=self.evaluation)
        self.assertGreater(report["mean_auxiliary_hidden_mse"], 0)
        self.assertEqual(represent(model, []).shape, (0, 16))
        broken = [{**r, "translated_target": [float("nan")] * 16} for r in self.rows]
        before = model.updates
        with self.assertRaises(ValueError):
            train_update(model, broken, budget=replace(self.budget, auxiliary_weight=0.1),
                         fixed_evaluation=self.evaluation)
        self.assertEqual(model.updates, before)

    def test_auxiliary_targets_are_frozen_before_any_recipient_forward(self):
        baseline, mutated = self.recipient(), self.recipient()
        original = [{**row, "translated_target": [0.0] * 16} for row in self.rows]
        caller_rows = copy.deepcopy(original)
        def change_caller_inputs(module, args, output):
            for row in caller_rows:
                row["translated_target"][:] = [1000.0] * 16
        handle = mutated.module.register_forward_hook(change_caller_inputs)
        try:
            left = train_update(baseline, original, budget=replace(self.budget, auxiliary_weight=.1),
                                fixed_evaluation=self.evaluation)
            right = train_update(mutated, caller_rows, budget=replace(self.budget, auxiliary_weight=.1),
                                 fixed_evaluation=self.evaluation)
        finally:
            handle.remove()
        self.assertEqual(caller_rows[0]["translated_target"][0], 1000)
        self.assertEqual(left, right)
        self.assertTrue(right["auxiliary_targets_detached"])
        self.assertIsNotNone(right["auxiliary_targets_sha256"])

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "optional Transformers runtime required")
    def test_pretrained_backend_updates_only_adapter_and_reloads_it(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        class ByteTokenizer:
            eos_token_id, pad_token_id = 2, 0
            def encode(self, text, add_special_tokens=False):
                return [1] + [b + 3 for b in text.encode()]
        def make_base(*args, **kwargs):
            torch.manual_seed(7)
            return Qwen2ForCausalLM(Qwen2Config(vocab_size=259, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, max_position_embeddings=128))
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=ByteTokenizer()), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=make_base):
            model = HFRecipient(model_id="local/test", revision="a" * 40, rank=4, max_length=128)
            base_before = {k: v.clone() for k, v in model.module.base.state_dict().items()}
            before = model.module.up.weight.clone()
            report = train_update(model, self.rows, budget=self.budget, fixed_evaluation=self.evaluation)
            self.assertFalse(torch.equal(before, model.module.up.weight))
            self.assertTrue(all(torch.equal(v, model.module.base.state_dict()[k]) for k, v in base_before.items()))
            self.assertEqual(report["backend"], HFRecipient.backend)
            # A second arm reuses only the frozen model/tokenizer; its own
            # zero adapter reproduces the baseline, without stacking arm hooks.
            other = HFRecipient(model_id="local/test", revision="a" * 40, rank=4, max_length=128,
                                shared_base_from=model)
            self.assertIs(other.module.base, model.module.base)
            self.assertIs(other.tokenizer, model.tokenizer)
            self.assertIsNot(other.module.up.weight, model.module.up.weight)
            self.assertFalse(other.module.base.get_output_embeddings()._forward_pre_hooks)
            left = represent(model, [self.rows[0]["prompt"]])
            right = represent(other, [self.rows[0]["prompt"]])
            self.assertFalse((left == right).all())
            self.assertTrue((left == represent(model, [self.rows[0]["prompt"]])).all())
            auxiliary = [{**row, "translated_target": [0.0] * 16} for row in self.rows]
            second = train_update(other, auxiliary, budget=replace(self.budget, auxiliary_weight=0.1),
                                  fixed_evaluation=self.evaluation)
            self.assertGreater(second["mean_auxiliary_hidden_mse"], 0)
            self.assertTrue((left == represent(model, [self.rows[0]["prompt"]])).all())
            self.assertFalse(other.module.base.get_output_embeddings()._forward_pre_hooks)
            with self.assertRaises(ValueError):
                HFRecipient(model_id="local/different", revision="a" * 40, max_length=128,
                            shared_base_from=model)
            checkpoint = Path(self.tmp.name) / "adapter.pt"
            save_recipient(model, checkpoint)
            loaded = load_recipient(checkpoint)
            self.assertEqual(evaluate(model, self.evaluation), evaluate(loaded, self.evaluation))
            self.assertEqual(represent(loaded, [self.rows[0]["prompt"]]).shape, (1, 16))


if __name__ == "__main__":
    unittest.main()
