"""Constructive soundness, binder hygiene, structural splits and real Lean cache."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lemmaferry.scratch import logic as l


def sample(seed=10000, harder=False):
    for value in range(seed, seed + 10000):
        row = l.candidate(value, harder)
        if row is not None:
            row = l.render(row, 'A')
            if l._fits(row, 512): return row
    raise AssertionError('bounded test fixture unavailable')


class ScratchLogicTests(unittest.TestCase):
    def test_binder_insertion_and_beta_reduction_do_not_capture_variables(self):
        context = [0, 1]
        proof = ['intro', 2, ['pair', ['hyp', 0], ['intro', 3, ['pair', ['hyp', 2], ['hyp', 3]]]]]
        self.assertEqual(l.infer(proof, context), l.infer(l.shift(proof, 2), context + [1]))
        self.assertIn('fun h3', l.term(l.shift(proof, 2), 3))
        function = ['intro', ['imp', 1, 1], ['intro', 2, ['pair', ['hyp', 1], ['hyp', 2]]]]
        argument = ['intro', 1, ['hyp', 1]]
        application = ['apply', function, argument]
        reduced = l.normalize(application, 1)
        self.assertEqual(l.infer(application, [0]), l.infer(reduced, [0]))
        self.assertEqual(reduced, ['intro', 2, ['pair', ['intro', 1, ['hyp', 2]], ['hyp', 1]]])

    def test_all_atom_bijections_and_premise_orders_share_one_structure(self):
        premises = [0, ['imp', 0, ['and', 1, 2]], ['imp', 2, 3]]
        goal = ['and', 1, 3]
        expected = l.structure(premises, goal)
        for permutation in itertools.permutations(range(4)):
            self.assertEqual(l.structure([l.substitute(p, permutation) for p in reversed(premises)],
                                         l.substitute(goal, permutation)), expected)
        self.assertEqual(l.structure(premises + [0], goal), expected)
        self.assertNotEqual(l.structure(premises, ['and', 3, 1]), expected)

    def test_raw_lean_unknown_lemmas_and_forged_renderings_are_rejected(self):
        for proof in [['sorry'], ['axiom', 0], ['lemma', 'test_solution', 0, 1], ['hyp', True], ['hyp', -1], [['apply']]]:
            with self.assertRaises(ValueError): l.infer(proof, [0])
        row = sample()
        for field, value in [('lean_declaration', 'axiom bad : False\n'), ('lean_proof', 'by sorry'),
                             ('dependencies', ['heldout']), ('countermodel', [0, 1, 0, 1]),
                             ('family', 'forged'), ('depth', 99), ('answer', 'B')]:
            corrupted = copy.deepcopy(row); corrupted[field] = value
            with self.assertRaises(ValueError): l.validate(corrupted)
        corrupted = dict(row, verified=True)
        with self.assertRaises(ValueError): l.validate(corrupted)

    def test_generated_positive_and_negative_choices_are_semantically_distinct(self):
        families, structures = set(), set()
        for seed in range(10000, 10500):
            row = l.candidate(seed)
            if row is None: continue
            row = l.render(row, 'AB'[seed % 2]); l.validate(row)
            families.add(row['family']); structures.add(row['id'])
            self.assertEqual(l.infer(row['proof'], row['premises']), row['goal'])
            for bits in itertools.product((False, True), repeat=4):
                if all(l.truth(p, bits) for p in row['premises']): self.assertTrue(l.truth(row['goal'], bits))
            self.assertFalse(l.truth(row['distractor'], row['countermodel']))
        self.assertGreaterEqual(len(families), 3)
        self.assertGreater(len(structures), 30)
        ordinary, harder = sample(), sample(harder=True)
        self.assertLessEqual(ordinary['depth'], 3)
        self.assertGreaterEqual(harder['depth'], 4)
        self.assertGreaterEqual(harder['proof_depth'], 5)

    def test_every_model_view_includes_real_headers_in_its_length_gate(self):
        row = sample()
        self.assertTrue(l._fits(row, 512))
        self.assertGreaterEqual(len(row['canonical_proof'].encode()), 63)
        limit = max(len(s.encode()) + 2 for s in [row['question'] + ' A', row['question'] + ' B',
                    row['solution'], row['canonical_proof'], 'Proof:\n' + row['canonical_proof']])
        self.assertTrue(l._fits(row, limit))
        self.assertFalse(l._fits(row, limit - 1))

    def test_choices_match_every_connective_depth_count_and_displayed_byte_length(self):
        def shape(f): return 'atom' if type(f) is int else (f[0], shape(f[1]), shape(f[2]))
        def counts(f):
            if type(f) is int: return (1, 0, 0)
            a, b = counts(f[1]), counts(f[2])
            return tuple(a[i] + b[i] + int(i == (1 if f[0] == 'and' else 2)) for i in range(3))
        inspected = 0
        for harder in (False, True):
            for seed in range(10000, 11000):
                row = l.candidate(seed, harder)
                if row is None: continue
                row = l.render(row, 'AB'[seed % 2]); l.validate(row)
                self.assertEqual(shape(row['goal']), shape(row['distractor']))
                self.assertEqual(counts(row['goal']), counts(row['distractor']))
                self.assertEqual(l.formula_depth(row['goal']), l.formula_depth(row['distractor']))
                a = row['question'].split('\nA: ', 1)[1].split('\nB: ', 1)[0]
                b = row['question'].split('\nB: ', 1)[1].split('\nAnswer:', 1)[0]
                self.assertEqual(len(a.encode()), len(b.encode()))
                inspected += 1
        self.assertGreater(inspected, 300)

    def test_missing_lean_cannot_create_a_verified_dataset_or_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(l._AdmissionLibrary, '_compile', return_value={'status': 'not_run', 'reason': 'lean_missing'}):
                with self.assertRaisesRegex(ValueError, 'not_run'):
                    l.generate(temporary, {'train': 2})
            self.assertFalse((Path(temporary) / 'dataset.json').exists())
            self.assertEqual(list((Path(temporary) / 'lean-cache/audits').glob('*.json')), [])

    @unittest.skipUnless((Path.home() / '.elan/bin/lean').exists(), 'pinned real Lean required')
    def test_real_lean_batches_audit_common_lemmas_and_detect_cache_corruption(self):
        rows = [sample(), sample(20000, harder=True)]
        with tempfile.TemporaryDirectory() as temporary:
            result = l.admit(temporary, rows)
            self.assertEqual(result['records'], 2)
            receipt = next((Path(temporary) / 'lean-cache/audits').glob('*.json'))
            value = json.loads(receipt.read_text())
            log = Path(temporary) / 'lean-cache' / value['verification']['log_path']
            self.assertIn("'lf_identity' does not depend on any axioms", log.read_text())
            self.assertIn("'lf_project' does not depend on any axioms", log.read_text())
            self.assertEqual(value['verification']['audited_declarations'], 4)
            self.assertEqual(l.admit(temporary, rows), result)
            log.write_text(log.read_text().replace("'lf_identity'", "'forged'"))
            value['verification']['log_sha256'] = l.digest(log.read_bytes())
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'unverified'): l.admit(temporary, rows)
            l.admit(temporary, rows, replay=True)
            source = Path(temporary) / 'lean-cache' / json.loads(receipt.read_text())['verification']['source_path']
            source.write_text(source.read_text() + '\naxiom forged : False\n')
            with self.assertRaisesRegex(ValueError, 'corrupt'): l.admit(temporary, rows)

    @unittest.skipUnless((Path.home() / '.elan/bin/lean').exists(), 'pinned real Lean required')
    def test_frozen_splits_reserved_pilot_ids_and_separate_extensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pilot = l.generate(base / 'pilot', {'train': 4, 'validation': 2}, seed=9000000)
            reserved = [row['id'] for row in pilot['rows']]
            main = l.generate(base / 'main', {'train': 12, 'validation': 4, 'test': 4, 'harder': 4}, reserved_ids=reserved)
            original = (base / 'main/dataset.json').read_bytes()
            self.assertFalse(set(reserved) & {row['id'] for row in main['rows']})
            for split in ('train', 'validation', 'test', 'harder'):
                rows = [r for r in main['rows'] if r['split'] == split]
                self.assertEqual(sum(r['answer'] == 'A' for r in rows), len(rows) // 2)
            extension = l.generate(base / 'main', extra=4)
            self.assertEqual((base / 'main/dataset.json').read_bytes(), original)
            self.assertEqual(extension['parent_fingerprint'], main['fingerprint'])
            self.assertEqual(len(extension['rows']), 4)
            self.assertFalse({r['id'] for r in main['rows']} & {r['id'] for r in extension['rows']})
            second = l.generate(base / 'main', extra=2)
            self.assertFalse({r['id'] for r in extension['rows']} & {r['id'] for r in second['rows']})
            self.assertTrue((base / 'main/extensions/extension-001.json').exists())


if __name__ == '__main__': unittest.main()
