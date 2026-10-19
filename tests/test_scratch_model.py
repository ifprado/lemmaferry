"""Causal, bottleneck and exact continuation checks for the local-only models."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from lemmaferry.scratch import model as net
    from lemmaferry.scratch.io import save, load


@unittest.skipUnless(HAS_TORCH, 'optional Torch runtime required')
class ScratchModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.config = net.ModelConfig(layers=2, width=16, heads=2, context=64)

    def test_byte_vocabulary_and_exact_prefix_contract(self):
        text = 'A\x00é\n'
        ids = net.encode(text)
        self.assertEqual(ids, [net.BOS] + [b + 3 for b in text.encode()])
        self.assertEqual(net.decode(ids + [net.EOS, net.PAD]), text)
        self.assertEqual(net.encode('A')[1], ord('A') + 3)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            net.ModelConfig(width=17, heads=2)
        with self.assertRaisesRegex(ValueError, 'prefix'):
            net.batch(['different A'], 64, 'cpu', ['Question:'])
        with self.assertRaisesRegex(ValueError, 'refusing truncation'):
            net.batch(['x' * 64], 64, 'cpu')
        with self.assertRaises(ValueError):
            net.LM(self.config)(torch.zeros((1, 65), dtype=torch.long))

    def test_future_tokens_and_answer_letter_cannot_leak_into_prediction(self):
        model = net.LM(self.config).eval()
        prompt = 'Which follows?\nAnswer:'
        x, labels, positions = net.batch([prompt + ' A', prompt + ' B'], 64, 'cpu', [prompt, prompt])
        logits, _ = model(x)
        # The answer letter is predicted AFTER the fixed space and BEFORE that
        # answer byte enters the model's causal context.
        index = int(positions[0]) + 1
        torch.testing.assert_close(logits[0, :index + 1], logits[1, :index + 1], rtol=0, atol=0)
        self.assertEqual(int(x[0, index]), ord(' ') + 3)
        self.assertEqual(int(labels[0, index + 1]), ord('A') + 3)
        self.assertEqual(int(labels[1, index + 1]), ord('B') + 3)
        self.assertTrue(labels[0, :int(positions[0]) + 1].eq(-100).all())
        retained = []
        handle = model.tokens.register_forward_hook(lambda m, args, output: retained.append(output))
        output, _ = model(x[:1])
        retained[0].retain_grad()
        output[0, index, 9].backward()
        handle.remove()
        self.assertEqual(float(retained[0].grad[0, index + 1:].abs().sum()), 0.)
        self.assertGreater(float(retained[0].grad[0, :index + 1].abs().sum()), 0.)

    def test_shifted_loss_padding_and_all_parameter_gradients(self):
        model = net.LM(self.config)
        x, labels, _ = net.batch(['alpha', 'beta'], 64, 'cpu')
        logits, _ = model(x)
        actual = net.loss(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, net.VOCAB),
                                                      labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        self.assertTrue(all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertTrue(model.tokens.weight.grad[net.PAD].eq(0).all())
        altered = logits.detach().clone()
        # Positions whose next label is padding must have no effect on loss.
        ignored = labels[:, 1:].eq(-100)
        altered[:, :-1][ignored] = 1000 * torch.randn_like(altered[:, :-1][ignored])
        torch.testing.assert_close(net.loss(altered, labels), actual.detach(), rtol=0, atol=0)

    def test_representation_uses_final_content_byte_and_dynamic_padding(self):
        model = net.LM(self.config).eval()
        texts = ['short', 'a slightly longer sentence']
        expected = []
        with torch.no_grad():
            for text in texts:
                ids = torch.tensor([net.encode(text)])
                _, hidden = model(ids)
                expected.append(hidden[0, -1])
            vectors = net.represent(model, texts, 'cpu')
        torch.testing.assert_close(vectors, torch.stack(expected), rtol=1e-6, atol=1e-6)
        self.assertFalse(vectors.requires_grad)
        fixed, _, _ = net.batch(texts, 64, 'cpu')
        dynamic, _, _ = net.batch(texts, 64, 'cpu', pad_to_context=False)
        self.assertEqual(fixed.shape, (2, 64))
        self.assertLess(dynamic.shape[1], fixed.shape[1])
        # Extraction needs no EOS slot and permits the exact full context.
        self.assertEqual(net.represent(model, ['x' * 63], 'cpu').shape, (1, 16))

    def test_checkpoint_restores_rng_weights_and_adamw_moments_bitwise(self):
        model = net.LM(self.config)
        opt = net.optimizer(model, lr=.003)
        def update(recipient, optimizer):
            x = torch.randint(3, net.VOCAB, (2, 12))
            optimizer.zero_grad(set_to_none=True)
            value = net.loss(recipient(x)[0], x)
            value.backward()
            optimizer.step()
            return value.detach().clone()
        update(model, opt)
        random.seed(33)
        np.random.seed(34)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'recipient.pt'
            net.checkpoint(path, model, opt, 1, marker='checked')
            expected_python, expected_numpy = random.random(), np.random.rand()
            expected_loss = update(model, opt)
            expected_model = net.state_digest(model.state_dict())
            expected_optimizer = net.state_digest(opt.state_dict())
            restored, loaded_opt, state = net.restore(path, 'cpu')
            self.assertEqual(state['metadata']['marker'], 'checked')
            self.assertEqual(state['updates'], 1)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.rand(), expected_numpy)
            torch.testing.assert_close(update(restored, loaded_opt), expected_loss, rtol=0, atol=0)
            self.assertEqual(net.state_digest(restored.state_dict()), expected_model)
            self.assertEqual(net.state_digest(loaded_opt.state_dict()), expected_optimizer)
            self.assertTrue(all(float(v['step']) == 2 for v in loaded_opt.state.values()))
            # Recomputing the outer artifact hash cannot hide invalid metadata.
            altered = load(path)
            altered['tokenizer']['offset'] = 4
            save(path, altered)
            with self.assertRaisesRegex(ValueError, 'tokenizer'):
                net.restore(path, 'cpu')

    def test_checkpoint_rejects_tensor_dtype_and_optimizer_shape_changes(self):
        model = net.LM(self.config)
        opt = net.optimizer(model)
        x, y, _ = net.batch(['tiny batch'], 64, 'cpu')
        net.loss(model(x)[0], y).backward()
        opt.step()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'recipient.pt'
            net.checkpoint(path, model, opt, 1)
            original = load(path)
            changed = copy.deepcopy(original)
            key = next(iter(changed['model']))
            changed['model'][key] = changed['model'][key].double()
            changed['model_sha256'] = net.state_digest(changed['model'])
            save(path, changed)
            with self.assertRaisesRegex(ValueError, 'dtype'):
                net.restore(path, 'cpu')
            changed = copy.deepcopy(original)
            first = next(iter(changed['optimizer']['state'].values()))
            first['exp_avg'] = torch.zeros(1)
            changed['optimizer_sha256'] = net.state_digest(changed['optimizer'])
            save(path, changed)
            with self.assertRaisesRegex(ValueError, 'moment'):
                net.restore(path, 'cpu')

    def test_proof_decoder_only_sees_vector_and_padding_cannot_change_it(self):
        ae = net.ProofAE(width=12, latent=5, context=64)
        ids, _, _ = net.batch(['proof one', 'proof two'], 64, 'cpu')
        shorter, _, _ = net.batch(['proof one', 'proof two'], 64, 'cpu', pad_to_context=False)
        z = ae.encode(ids)
        torch.testing.assert_close(z, ae.encode(shorter), rtol=1e-6, atol=1e-6)
        # Decoder output is completely specified by z and positions; prior input
        # calls cannot supply a hidden teacher-forcing cache or a copied prefix.
        fixed = ae.decode(z.detach(), 20)
        ae.encode(ids.flip(0))
        torch.testing.assert_close(ae.decode(z.detach(), 20), fixed, rtol=0, atol=0)
        self.assertFalse(torch.equal(ae.decode(torch.zeros_like(z), 20), fixed))
        logits = ae(ids)
        value = torch.nn.functional.cross_entropy(logits.reshape(-1, net.VOCAB), ids.reshape(-1), ignore_index=net.PAD)
        value.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in ae.parameters()))
        with self.assertRaises(ValueError):
            ae.encode(torch.zeros((1, 4), dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
