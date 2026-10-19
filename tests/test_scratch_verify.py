"""Independent evidence checks; numerical fixtures do not claim Lean admission."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from lemmaferry.scratch import experiment, logic, model as net, verify
    from lemmaferry.scratch.io import digest, load, save, state_digest


def dataset_fixture():
    rows, ids = [], set()
    for split in ('train', 'validation', 'test', 'harder'):
        for seed in range(10001, 30001):
            row = logic.candidate(seed, split == 'harder')
            if row is None or row['id'] in ids: continue
            bucket = int(row['id'][:8], 16) % 10
            if split == 'train' and bucket >= 8: continue
            if split == 'validation' and bucket != 8: continue
            if split == 'test' and bucket != 9: continue
            row = logic.render(row, 'A')
            if not logic._fits(row, 512): continue
            row['split'] = split
            try: logic.validate(row)
            except ValueError: continue
            rows.append(row); ids.add(row['id']); break
        else: raise AssertionError('could not construct the numerical certificate fixture')
    return {'schema': 2, 'rows': rows, 'context': 512, 'common': copy.deepcopy(logic.COMMON),
            'common_training_examples': 0, 'fingerprint': digest(rows)}


@unittest.skipUnless(HAS_TORCH, 'optional Torch runtime required')
class ScratchVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.dataset = dataset_fixture()

    def setUp(self):
        torch.manual_seed(71)
        self.config = {'model': {'layers': 1, 'width': 16, 'heads': 2, 'context': 64},
                       'learning_rate': .0005, 'batch': 4, 'recipient_steps': 16, 'stages': 2}

    def checkpoint(self, path):
        model = net.LM(net.ModelConfig(**self.config['model']))
        optimizer = net.optimizer(model, self.config['learning_rate'])
        x, y, _ = net.batch(['ordinary language'], 64, 'cpu')
        optimizer.zero_grad(set_to_none=True)
        net.loss(model(x)[0], y).backward(); optimizer.step()
        net.checkpoint(path, model, optimizer, 1, fixture=True)
        return model

    def test_typed_certificates_countermodels_and_structural_splits(self):
        report = verify.validate_dataset(self.dataset)
        self.assertEqual(report['split_counts'], {'train': 1, 'validation': 1, 'test': 1, 'harder': 1})
        self.assertTrue(report['all_negative_countermodels_checked'])
        self.assertNotIn('real_lean_admission', report)
        for mutation in ('fingerprint', 'countermodel', 'duplicate', 'dependency', 'rendering'):
            altered = copy.deepcopy(self.dataset)
            if mutation == 'fingerprint': altered['fingerprint'] = '0' * 64
            elif mutation == 'countermodel': altered['rows'][0]['countermodel'] = [True] * 4
            elif mutation == 'duplicate': altered['rows'].append(copy.deepcopy(altered['rows'][0]))
            elif mutation == 'dependency': altered['rows'][0]['dependencies'] = ['held_out_solution']
            else: altered['rows'][0]['solution'] += 'unbound'
            if mutation != 'fingerprint': altered['fingerprint'] = digest(altered['rows'])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.validate_dataset(altered)

    def test_choice_shape_and_display_length_controls_are_independent(self):
        row = {'goal': ['imp', 0, ['and', 1, 2]],
               'distractor': ['imp', 3, ['and', 1, 0]],
               'question': 'Given facts.\nA: (if rain then (snow and wind))\nB: (if mist then (snow and rain))\nAnswer:'}
        verify.validate_surface_controls(row)
        altered = copy.deepcopy(row)
        altered['distractor'][2][0] = 'imp'
        with self.assertRaisesRegex(ValueError, 'skeleton'):
            verify.validate_surface_controls(altered)
        altered = copy.deepcopy(row)
        altered['question'] = altered['question'].replace('mist', 'longer_word')
        with self.assertRaisesRegex(ValueError, 'byte lengths'):
            verify.validate_surface_controls(altered)

    def test_stale_checksum_and_resigned_inner_model_hash_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'recipient.pt'
            self.checkpoint(path)
            state = verify._recipient_state(path, self.config, 1)
            self.assertEqual(state['updates'], 1)
            state['model']['head.weight'][0, 0] += 1
            save(path, state)  # Re-sign only the external checksum pointer.
            with self.assertRaisesRegex(ValueError, 'inner checksum'):
                verify._recipient_state(path, self.config, 1)
            self.checkpoint(path)
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                verify.checked_checkpoint(path)

    def test_config_and_optimizer_history_cannot_be_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'recipient.pt'
            self.checkpoint(path)
            wrong = copy.deepcopy(self.config); wrong['model']['width'] = 32
            with self.assertRaisesRegex(ValueError, 'architecture'):
                verify._recipient_state(path, wrong, 1)
            state = load(path)
            for value in state['optimizer']['state'].values(): value['step'].zero_()
            state['optimizer_sha256'] = state_digest(state['optimizer'])
            save(path, state)
            with self.assertRaisesRegex(ValueError, 'moment counters'):
                verify._recipient_state(path, self.config, 1)

    def test_equal_budget_and_order_are_independent_of_pool_size(self):
        _, large = verify.expected_schedule(16, self.config, 7, 0)
        _, small = verify.expected_schedule(4, self.config, 7, 0)
        self.assertEqual(large['unique_problems_consumed'], 16)
        self.assertEqual(small['unique_problems_consumed'], 4)
        for key in ('updates', 'problem_exposures', 'supervised_tokens', 'padded_tokens'):
            self.assertEqual(large[key], small[key])
        self.assertEqual(large['supervised_tokens'], 8 * 2 * 67)
        self.assertEqual(verify.expected_schedule(16, self.config, 7, 0)[0],
                         verify.expected_schedule(16, self.config, 7, 0)[0])
        self.assertNotEqual(verify.expected_schedule(16, self.config, 7, 0)[0],
                            verify.expected_schedule(16, self.config, 7, 1)[0])

    def test_independent_question_only_likelihood_nll_and_generation(self):
        questions = [{'id': f'question-{i}', 'question': 'Choose a letter.\nAnswer:', 'answer': answer,
                      'family': 'fixture', 'depth': 1} for i, answer in enumerate(('A', 'B'))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'recipient.pt'
            self.checkpoint(path)
            receipt = experiment.evaluate_recipient(path, questions, 'cpu', batch_size=2)
            checked = verify.verify_evaluation(path, receipt, questions, self.config, samples=2)
            self.assertEqual(checked['independently_replayed_rows'], 2)
            self.assertTrue(checked['greedy_generation_replayed'])
            bad = copy.deepcopy(receipt)
            bad['predictions'][0]['canonical_proof'] = 'hidden proof'
            with self.assertRaisesRegex(ValueError, 'question-only'):
                verify.verify_evaluation(path, bad, questions, self.config, samples=0)
            bad = copy.deepcopy(receipt)
            for row in bad['predictions']:
                row['generation'] = 'altered'; row['generation_correct'] = False
            bad['overall'] = verify._evaluation_metrics(bad['predictions'])
            bad['by_family']['fixture'] = verify._evaluation_metrics(bad['predictions'])
            bad['by_depth']['1'] = verify._evaluation_metrics(bad['predictions'])
            with self.assertRaisesRegex(ValueError, 'greedy generation'):
                verify.verify_evaluation(path, bad, questions, self.config, samples=1)


if __name__ == '__main__':
    unittest.main()
