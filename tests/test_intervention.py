from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "optional real-model runtime required")
class InterventionTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        spec = importlib.util.spec_from_file_location("evaluate_intervention", ROOT / "experiments/evaluate_intervention.py")
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)
        torch.manual_seed(7)
        self.torch = torch
        self.model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32,
                intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=64)).eval()
        self.tokens = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones((1, 4), dtype=torch.long)}

    def test_self_patch_full_and_cached_reproduce_logits_at_exact_boundary(self):
        with self.torch.inference_mode():
            logits, state, cache = self.runner.forward_state(self.model, self.tokens, 1, use_cache=True)
            expected = self.model.model(**self.tokens, output_hidden_states=True, use_cache=False).hidden_states[1][:, -1]
            self.assertTrue(self.torch.equal(state, expected))
            for saved in (None, cache):
                actual = self.runner.replay_patch(self.model, self.tokens, 1, state, saved)
                self.assertTrue(self.torch.allclose(logits, actual, atol=1e-6, rtol=1e-6))

    def test_patch_changes_prediction_and_does_not_mutate_shared_cache(self):
        with self.torch.inference_mode():
            logits, state, cache = self.runner.forward_state(self.model, self.tokens, 1, use_cache=True)
            before = tuple((k.clone(), v.clone()) for k, v in cache.to_legacy_cache())
            changed = self.runner.replay_patch(self.model, self.tokens, 1, -state, cache)
            self.assertFalse(self.torch.allclose(logits, changed))
            for (a, b), (c, d) in zip(before, cache.to_legacy_cache()):
                self.assertTrue(self.torch.equal(a, c) and self.torch.equal(b, d))


if __name__ == "__main__":
    unittest.main()
