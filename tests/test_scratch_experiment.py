import tempfile
import unittest
from pathlib import Path
import numpy as np
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional torch dependency')
class ScratchExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_matched_schedule_consumes_largest_pool_and_masks_exact_budget(self):
        from lemmaferry.scratch.experiment import batch_schedule, training_batch
        c = {'batch': 16, 'recipient_steps': 1024, 'stages': 2, 'model': {'context': 512}}
        rows = [{'id': str(i), 'question': 'Given snow. Which follows? A) snow B) rain Answer:',
                 'answer': 'AB'[i % 2], 'canonical_proof': 'premises:' + ' P -> Q' * (10 + i % 3)} for i in range(4096)]
        schedule = batch_schedule(rows, c, 7, 0)
        self.assertEqual(len({i for ids in schedule for i in ids}), 4096)
        self.assertEqual(schedule, batch_schedule(rows, c, 7, 0))
        self.assertNotEqual(schedule, batch_schedule(rows, c, 7, 1))
        x, y, positions = training_batch(rows, schedule[0], c, 'cpu')
        self.assertEqual(tuple(x.shape), (16, 512))
        self.assertEqual(y.ne(-100).sum(1).tolist(), [3, 64] * 8)
        self.assertTrue(torch.equal(y[y != -100], x[y != -100]))

    def test_auxiliary_targets_are_detached_and_recipient_parameters_train(self):
        from lemmaferry.scratch import model as net
        from lemmaferry.scratch.experiment import step
        torch.manual_seed(7)
        model = net.LM(net.ModelConfig(layers=1, width=16, heads=2, context=32))
        opt = net.optimizer(model)
        x, labels, pos = net.batch(['question A', 'question B'], 32, 'cpu', ['question', 'question'])
        targets = torch.randn(2, 16, requires_grad=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        text, aux = step(model, opt, x, labels, pos, targets, .1, torch.ones(16))
        self.assertIsNone(targets.grad)
        self.assertTrue(np.isfinite([text, aux]).all())
        self.assertTrue(all(not torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_independent_answer_predictions_do_not_depend_on_scoring_labels(self):
        from lemmaferry.scratch import model as net
        from lemmaferry.scratch.experiment import evaluate_recipient
        torch.manual_seed(19)
        m = net.LM(net.ModelConfig(layers=1, width=16, heads=2, context=64))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'recipient.pt'
            net.checkpoint(p, m, net.optimizer(m), 0)
            q = {'id': 'question', 'question': 'Given rain. A) rain B) snow Answer:', 'answer': 'A', 'family': 'hyp', 'depth': 1}
            a = evaluate_recipient(p, [q], 'cpu')['predictions'][0]
            b = evaluate_recipient(p, [{**q, 'answer': 'B'}], 'cpu')['predictions'][0]
            self.assertEqual(a['prediction'], b['prediction'])
            self.assertEqual(a['generation'], b['generation'])
            self.assertEqual(a['answer_letter_log_probabilities'], b['answer_letter_log_probabilities'])

    def test_continuation_schedule_prioritizes_every_new_problem(self):
        from lemmaferry.scratch.experiment import batch_schedule
        rows = [{'id': str(i)} for i in range(100)]
        c = {'batch': 4, 'recipient_steps': 4, 'stages': 1, 'priority_ids': ['97', '98', '99']}
        indices = sum(batch_schedule(rows, c, 31, 0), [])
        self.assertEqual({rows[i]['id'] for i in indices[:3]}, set(c['priority_ids']))


if __name__ == '__main__':
    unittest.main()
